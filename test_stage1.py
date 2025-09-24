import os
import json
import torch
from torch import nn
from PIL import Image
import numpy as np
from src.models.stage1_prior_module import MyPriorTransformer
from src.pipelines.stage1_pipeline import Seq_Inpaint_Prior_Pipeline
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPTokenizer,
    CLIPImageProcessor,
)
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.multiprocessing as mp
import time
from omegaconf import OmegaConf
import random
from diffusers import UnCLIPScheduler

def generate_array():
    # Randomly select start position for 1s (from 1 to 3)
    ones_start = random.randint(1, 3)
    return [0] * (4 - ones_start) + [1] * ones_start

def split_list(n, m):
    quotient = n // m
    remainder = n % m
    result = []
    start = 0
    for i in range(m):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        result.append(list(range(start, end)))
        start = end
    return result


class Stage1InferenceRunner(nn.Module):
    def __init__(
            self, 
            unet_additional_kwargs, 
            # visible_situation,  # visible_situation [0, 0, 1, 1]: unmask unmask mask mask /example
            pretrained_model_name_or_path = "./Ckpts/kandinsky-2-2-prior", 
            module_ckpt = ""
            ):
        super().__init__()
        # self.visible_situation = visible_situation
        self.device = torch.device(f"cuda:0")

        self.seed_number = 42
        self.img_width = 512
        self.img_height = 512
        self.guidance_scale = 2.0
        self.num_inference_steps = 20
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.mymodule_path = module_ckpt
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed_number)

        self.prior = MyPriorTransformer.from_pretrained_2d(
            self.pretrained_model_name_or_path,
            subfolder="prior",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )

        scheduler = UnCLIPScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")

        if self.mymodule_path != "":
            prior_dict = torch.load(self.mymodule_path, map_location="cpu")["module"]
            self.prior.load_state_dict(prior_dict)
        else:
            print("Warning: mymodule_path is empty, skipping prior weight loading.")

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="image_encoder"
        ).to(self.device)

        self.clip_image_processor = CLIPImageProcessor()

        self.pipe = Seq_Inpaint_Prior_Pipeline(
            prior=self.prior,
            image_encoder=self.image_encoder,
            scheduler=scheduler
        ).to(self.device)

        print('Stage1InferenceRunner initialized.')

        # Prepare black and white reference images
        self.black_img_clip = self.clip_image_processor(
            images=Image.new("RGB", (self.img_width, self.img_height), (0, 0, 0)),
            return_tensors="pt"
        ).pixel_values.squeeze(0).to(self.device)

        self.white_img_clip = self.clip_image_processor(
            images=Image.new("RGB", (self.img_width, self.img_height), (255, 255, 255)),
            return_tensors="pt"
        ).pixel_values.squeeze(0).to(self.device)


    def get_clip_mean(self):
        return self.prior.clip_mean.clone()
    
    def get_clip_std(self):
        return self.prior.clip_std.clone()


    def forward(self, imgs_proj_embeds, mask_label_embeds, input_pose_coordinate):
        """
        Args:
            imgs_proj_embeds: shape [b, 1, 1280]
            mask_label_embeds: shape [b, 1, 1280]
            input_pose_coordinate: shape [b, 4, 36]
        Returns:
            output: shape [b, 4, 1280]
        """
        output = self.pipe(
            pose_coordinate=input_pose_coordinate,
            imgs_proj_embeds1=imgs_proj_embeds,
            mask_label=mask_label_embeds,
            video_length=4,
            height=self.img_height,
            width=self.img_width,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            num_inference_steps=self.num_inference_steps,
        )
        print("runner output shape: ", output[0].shape)  # [b, 4, 1280]
        return output[0].cpu()

def inference(args, rank, indexs, unet_additional_kwargs, visible_situation):
    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    save_dir = "./stage1/{}/{}_guidancescale{}_seed{}_numsteps{}/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number, args.num_inference_steps)

    save_dir_metric = "./stage1/{}/metric_{}_guidancescale{}_seed{}_reg/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    clip_image_processor = CLIPImageProcessor()

    model_ckpt = args.mymodule_path

    prior = MyPriorTransformer.from_pretrained_2d(
        args.pretrained_model_name_or_path,
        subfolder="prior",
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs))

    scheduler = UnCLIPScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    prior_dict = torch.load(model_ckpt, map_location="cpu")["module"]
    prior.load_state_dict(prior_dict)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder").to(device)

    pipe = Seq_Inpaint_Prior_Pipeline(
        prior=prior,
        image_encoder=image_encoder,
        scheduler=scheduler
    ).to(device)
    print('====================== dataset and model loaded ===================')

    with open(args.data_json_path, 'r') as f:
        test_data_list = json.load(f)
    with open(args.pose_json_path, 'r') as f:
        pose_data_dict = json.load(f)

    number = 0
    sum_simm = 0
    start_time = time.time()

    for number, index in enumerate(indexs):
        number += 1
        data_item = test_data_list[index]
        image_paths = [os.path.join(args.image_folder_path, data_item[f'img{i+1}']) for i in range(4)]

        images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]

        reference_images = [
            clip_image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            for img in images
        ]
        reference_image = torch.stack(reference_images, dim=0).to(
            memory_format=torch.contiguous_format).float().to(device)

        with torch.no_grad():
            target_embed = image_encoder(reference_image).image_embeds  # shape: [4, 1280]

        pose_list = []
        for i in range(4):
            img_name = data_item[f'img{i+1}']
            pose_entry = next((item for item in pose_data_dict if item["img"] == img_name), None)
            if pose_entry is None:
                raise ValueError(f"Pose data not found for image: {img_name}")
            if "coordinate" not in pose_entry:
                raise ValueError(f"'coordinate' key not found in pose data for image: {img_name}")
            coords = pose_entry["coordinate"]
            coords_flat = [x for pair in coords for x in pair]
            pose_list.append(coords_flat)
        input_pose_coordinate = torch.tensor(pose_list, dtype=torch.float32).to(device)
        print("input_pose_coordinate shape: ", input_pose_coordinate.shape)

        reference_image0 = (clip_image_processor(images=images[0], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image1 = (clip_image_processor(images=images[1], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image2 = (clip_image_processor(images=images[2], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image3 = (clip_image_processor(images=images[3], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image = torch.stack(
            [reference_image0, reference_image1, reference_image2, reference_image3], dim=0).to(
            memory_format=torch.contiguous_format).float()
        with torch.no_grad():
            target_embed = (image_encoder(reference_image.to(device)).image_embeds)

        black_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (0, 0, 0)),
                                               return_tensors="pt").pixel_values).squeeze(dim=0)
        white_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (255, 255, 255)),
                                               return_tensors="pt").pixel_values).squeeze(dim=0)
        if args.mode == 'visualization':
            source_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            mask_label_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            with torch.no_grad():
                imgs_proj_embeds = image_encoder(source_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)
                mask_label_embeds = image_encoder(mask_label_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)

        elif args.mode == 'continue':
            source_clip = []
            mask_label_clip = []

            for i in range(4):
                if visible_situation[i] == 0:
                    source_clip.append(reference_images[i])
                    mask_label_clip.append(white_img_clip)
                else:
                    source_clip.append(black_img_clip)
                    mask_label_clip.append(black_img_clip)

            source_clip = torch.stack(source_clip, dim=0)
            mask_label_clip = torch.stack(mask_label_clip, dim=0)

            with torch.no_grad():
                imgs_proj_embeds = image_encoder(source_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)
                mask_label_embeds = image_encoder(mask_label_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)
        else:
            raise ValueError("Invalid mode")

        if args.autoreg:
            print("autoreg mode")
            for i in range(4):
                print("index: {}, i: {}".format(index, i))
                if i == 0:
                    image_bemds = torch.empty(0, 1, 1280).to(device)
                elif i == 1:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)
                    image_bemds = image1
                    source_clip = torch.stack((black_img_clip, black_img_clip, black_img_clip), dim=0)
                    mask_label_clip = torch.stack([white_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)

                elif i == 2:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)
                    image2 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)
                    image_bemds = torch.cat([image1, image2], dim=0)
                    source_clip = torch.stack((black_img_clip, black_img_clip), dim=0)
                    mask_label_clip = torch.stack([white_img_clip, white_img_clip, black_img_clip, black_img_clip], dim=0)

                elif i == 3:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)
                    image2 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)
                    image3 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(2)))).unsqueeze(0).unsqueeze(0).to(device)
                    image_bemds = torch.cat([image1, image2, image3], dim=0)
                    source_clip = black_img_clip.unsqueeze(0)
                    mask_label_clip = torch.stack([white_img_clip, white_img_clip, white_img_clip, black_img_clip], dim=0)

                with torch.no_grad():
                    imgs_proj_embeds = image_encoder(source_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)
                    imgs_proj_embeds = torch.cat([image_bemds, imgs_proj_embeds], dim=0)
                    mask_label_embeds = image_encoder(mask_label_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)

                output = pipe(
                    pose_coordinate=input_pose_coordinate,
                    imgs_proj_embeds1=imgs_proj_embeds,
                    mask_label=mask_label_embeds,
                    video_length=4,
                    height=args.img_height,
                    width=args.img_width,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    num_inference_steps=args.num_inference_steps,
                )

                feature = output[0][i].cpu().detach().numpy()
                cosine_similarities = F.cosine_similarity(output[0][i], target_embed[i:i+1, :].squeeze(1))
                print("{}-{}:  {}".format(index, i, cosine_similarities))
                np.save('{}-{}_{}.npy'.format(save_dir_metric, index, i), feature)
                sum_simm += cosine_similarities.item()
        else:
            print("non-autoreg mode")
            output = pipe(
                pose_coordinate=input_pose_coordinate,
                imgs_proj_embeds1=imgs_proj_embeds,
                mask_label=mask_label_embeds,
                video_length=4,
                height=args.img_height,
                width=args.img_width,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            )

            cosine_scores_visible = []
            for j in range(4):
                feature = output[0][j].cpu().detach().numpy()
                cosine_similarities = F.cosine_similarity(output[0][j], target_embed[j:j+1, :].squeeze(1).squeeze(), dim=0)
                print("{}-{}:  {}".format(index, j, cosine_similarities))
                np.save('{}{}_{}.npy'.format(save_dir_metric, index, str(j)), feature)
                sum_simm += cosine_similarities.item()

                if visible_situation[j] == 1:
                    cosine_scores_visible.append(cosine_similarities.item())

            if cosine_scores_visible:
                avg_visible_similarity = sum(cosine_scores_visible) / len(cosine_scores_visible)
                print("Average similarity (visible == 1):", avg_visible_similarity)
            else:
                print("No visible == 1 frames to compute average similarity.")

        feature = output[0].cpu().detach().numpy()
        print("features shape is: ", feature.shape)
        np.save('{}{}.npy'.format(save_dir, index), feature)

    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 prior model inference script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="continue", help="[visualization, continue]")
    parser.add_argument("--data_json_path", type=str, default="")
    parser.add_argument("--pose_json_path", type=str, default="")
    parser.add_argument("--image_folder_path", type=str, default="")
    parser.add_argument('--autoreg', action='store_true', help='Enable autoregressive testing')
    parser.add_argument("--guidance_scale", type=int, default=2.0)
    parser.add_argument("--seed_number", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--exp_name", type=str, default="stage1/FlintstonesSV")
    parser.add_argument("--weights_number", type=int, default=100000)
    parser.add_argument("--mymodule_path", type=str, default="")

    args = parser.parse_args()
    print(args)

    num_devices = 1
    print("using {} process(es) for inference".format(num_devices))

    config = OmegaConf.load('./configs/testing.yaml')
    mp.set_start_method("spawn")

    with open(args.data_json_path, 'r') as f:
        test_data_list = json.load(f)
    random_index = random.randint(0, len(test_data_list) - 1)
    data_list = [[random_index]]

    print(f"Randomly selected index: {random_index}")
    print("Selected data:", test_data_list[random_index])

    print('=====')
    print(config)
    processes = []

    visible_situation = [0, 0, 1, 1]
    print("Generated array:", visible_situation)
    print("Count of 1s:", visible_situation.count(1))

    for rank in range(num_devices):
        p = mp.Process(target=inference, args=(args, rank, data_list[rank], config['unet_additional_kwargs'], visible_situation))
        processes.append(p)
        p.start()

    for rank, p in enumerate(processes):
        p.join()
