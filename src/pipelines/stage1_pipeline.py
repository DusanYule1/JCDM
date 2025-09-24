from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from src.models.stage1_prior_module import MyPriorTransformer
from diffusers.schedulers import UnCLIPScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_prior import KandinskyPriorPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Seq_Inpaint_Prior_Pipeline(DiffusionPipeline):
    def __init__(
        self,
        prior: MyPriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        scheduler: UnCLIPScheduler,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            scheduler=scheduler,
            image_encoder=image_encoder,
        )

    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_prior.KandinskyPriorPipeline.get_zero_embed
    def get_zero_embed(self, batch_size=1, device=None):
        device = device or self.device
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        pose_coordinate: Union[str, List[str]],
        imgs_proj_embeds1: Union[str, List[str]],
        mask_label: Union[str, List[str]],
        video_length: Optional[int] = 4,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None, 
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        device = self._execution_device

        batch_size = 1


        self._guidance_scale = guidance_scale

        # prior
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size*video_length, embedding_dim),
            imgs_proj_embeds1.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        pose_coordinate = torch.cat([pose_coordinate] * 2) if self.do_classifier_free_guidance else pose_coordinate
        imgs_proj_embeds1 = torch.cat([imgs_proj_embeds1] * 2) if self.do_classifier_free_guidance else imgs_proj_embeds1
        mask_label = torch.cat([mask_label] * 2) if self.do_classifier_free_guidance else mask_label

        self._num_timesteps = len(timesteps)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # print(latent_model_input.shape, t.shape, prompt_embeds.shape, text_encoder_hidden_states.shape, imgs_proj_embeds1.shape, mask_label.shape)
            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                input_pose_embeds=pose_coordinate,
                input_imgs_proj_embeds=imgs_proj_embeds1,
                input_masked_label_embeds = mask_label,
                attention_mask=None,
            ).predicted_image_embedding
            # print(f"predicted_image_embedding shape: {predicted_image_embedding.shape}")

            if self.do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + self.guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                text_encoder_hidden_states = callback_outputs.pop(
                    "text_encoder_hidden_states", text_encoder_hidden_states
                )
                text_mask = callback_outputs.pop("text_mask", text_mask)

        latents = self.prior.post_process_latents(latents)

        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is None:
            zero_embeds = self.get_zero_embed(latents.shape[0], device=latents.device)
        else:
            image_embeddings, zero_embeds = image_embeddings.chunk(2)

        self.maybe_free_model_hooks()

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()
            zero_embeds = zero_embeds.cpu().numpy()

        if not return_dict:
            return (image_embeddings, zero_embeds)

        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)