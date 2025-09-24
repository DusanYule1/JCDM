from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput
from .attention import BasicTransformerBlock
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from .motion_module import get_motion_module
import os, json

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class PriorTransformerOutput(BaseOutput):
    predicted_image_embedding: torch.FloatTensor

class MyPriorTransformer(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 32,
            attention_head_dim: int = 64,
            num_layers: int = 10,
            embedding_dim: int = 1280,
            num_embeddings=1,
            additional_embeddings=3,
            dropout: float = 0.0,
            time_embed_act_fn: str = "silu",
            norm_in_type: Optional[str] = None,
            embedding_proj_norm_type: Optional[str] = None,
            encoder_hid_proj_type: Optional[str] = "linear",
            added_emb_type: Optional[str] = "prd",
            time_embed_dim: Optional[int] = None,
            embedding_proj_dim: Optional[int] = None,
            clip_embed_dim: Optional[int] = None,

            unet_use_cross_frame_attention=None,
            unet_use_temporal_attention=None,

            use_motion_module=None,
            motion_module_type=None,
            motion_module_kwargs=None,
    ):
        super().__init__()
        self.pose_mlp = MLP(in_dim=36, hidden_dim=512, out_dim=1280)

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        time_embed_dim = time_embed_dim or inner_dim
        embedding_proj_dim = embedding_proj_dim or embedding_dim
        clip_embed_dim = clip_embed_dim or embedding_dim

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, out_dim=inner_dim,
                                                act_fn=time_embed_act_fn)

        self.proj_in = nn.Linear(embedding_dim, inner_dim)

        if embedding_proj_norm_type is None:
            self.embedding_proj_norm = None
        elif embedding_proj_norm_type == "layer":
            self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim)
        else:
            raise ValueError(f"unsupported embedding_proj_norm_type: {embedding_proj_norm_type}")

        self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim)
        self.embedding_proj1 = nn.Linear(embedding_proj_dim, inner_dim)
        self.embedding_proj2 = nn.Linear(embedding_proj_dim, inner_dim)

        self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings + additional_embeddings, inner_dim))

        if added_emb_type == "prd":
            self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        elif added_emb_type is None:
            self.prd_embedding = None
        else:
            raise ValueError(
                f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )

        from itertools import chain
        self.transformer_blocks = nn.ModuleList(
            list(chain(*[
                (
                    BasicTransformerBlock(
                        inner_dim,
                        num_attention_heads,
                        attention_head_dim,
                        dropout=dropout,
                        activation_fn="gelu",
                        attention_bias=True,
                        unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                        unet_use_temporal_attention=unet_use_temporal_attention,
                    ),
                    get_motion_module(
                        in_channels=inner_dim,
                        prior_state =True,
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    ) if use_motion_module else None
                )
                for d in range(num_layers)
            ]))
        )

        if norm_in_type == "layer":
            self.norm_in = nn.LayerNorm(inner_dim)
        elif norm_in_type is None:
            self.norm_in = None
        else:
            raise ValueError(f"Unsupported norm_in_type: {norm_in_type}.")

        self.norm_out = nn.LayerNorm(inner_dim)

        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim)

        causal_attention_mask = torch.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer("causal_attention_mask", causal_attention_mask, persistent=False)

        self.clip_mean = torch.tensor(-0.016)
        self.clip_std = torch.tensor(0.415)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module,
                                        processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def forward(
            self,
            hidden_states,
            timestep: Union[torch.Tensor, float, int],
            input_pose_embeds: torch.FloatTensor,
            input_imgs_proj_embeds: Optional[torch.FloatTensor] = None,
            input_masked_label_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            return_dict: bool = True,
    ):
        batch_size = hidden_states.shape[0]

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=hidden_states.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(hidden_states.device)

        timesteps = timesteps * torch.ones(batch_size, dtype=timesteps.dtype, device=timesteps.device)

        timesteps_projected = self.time_proj(timesteps)
        timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        dtype = next(self.pose_mlp.parameters()).dtype
        device = next(self.pose_mlp.parameters()).device

        input_pose_embeds = input_pose_embeds.to(dtype=dtype, device=device)
        input_pose_embeds = self.pose_mlp(input_pose_embeds).unsqueeze(1)

        pose_embeds = self.embedding_proj(input_pose_embeds)
        imgs_proj_embeds = self.embedding_proj1(input_imgs_proj_embeds)
        masked_label_embeds = self.embedding_proj2(input_masked_label_embeds)

        hidden_states = self.proj_in(hidden_states)

        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        additional_embeds = []

        if len(pose_embeds.shape) == 2:
            pose_embeds = pose_embeds[:, None, :]

        if len(imgs_proj_embeds.shape) == 2:
            imgs_proj_embeds = imgs_proj_embeds[:, None, :]

        if len(masked_label_embeds.shape) == 2:
            masked_label_embeds = masked_label_embeds[:, None, :]

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[:, None, :]

        additional_embeds = additional_embeds + [
            pose_embeds,
            imgs_proj_embeds,
            masked_label_embeds,
            time_embeddings[:, None, :],
            hidden_states,
        ]

        if self.prd_embedding is not None:
            prd_embedding = self.prd_embedding.to(hidden_states.dtype).expand(batch_size, -1, -1)
            additional_embeds.append(prd_embedding)

        hidden_states = torch.cat(
            additional_embeds,
            dim=1,
        )

        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = F.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        final_attention_mask = self.causal_attention_mask.to(hidden_states.dtype)
        final_attention_mask = final_attention_mask.repeat_interleave(self.config.num_attention_heads, dim=0)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)

        attention_mask = None
        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)
        for block in self.transformer_blocks:
            if isinstance(block, BasicTransformerBlock):
                 hidden_states = block(hidden_states, attention_mask=attention_mask)
            elif block is not None:
                 hidden_states = block(hidden_states, encoder_hidden_states=None)

        hidden_states = self.norm_out(hidden_states)

        hidden_states = hidden_states[:, -1]

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal prior's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["num_embeddings"] = 1
        config["additional_embeddings"] = 5

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **unet_additional_kwargs)

        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")

        match_dict = {k: v for k, v in state_dict.items() if not k.startswith("positional_embedding")}

        m, u = model.load_state_dict(match_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")

        return model
