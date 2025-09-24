import argparse
from datetime import datetime
from pathlib import Path
import torch
from diffusers import AutoencoderKL, DDIMScheduler
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.pose_guider import PoseGuider
from test_stage1 import Stage1InferenceRunner
from pipeline.pipeline_pose2img import Pose2ImagePipeline

from dwpose import DWposeDetector
from einops import rearrange, repeat
import numpy as np
import random
from transformers import Dinov2Model
import os
import json
import cv2

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
def stitch_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im

def generate_array():
    ones_start = random.randint(1, 3)
    return [0] * (4 - ones_start) + [1] * ones_start

def big2small_image(big_img): # b, h,w, c
    big_img = rearrange(big_img, "b h w c -> b c h w")

    bs, _, height, width = big_img.shape
    image1 = big_img[:, :, :height//2 , :width//2]
    image2 = big_img[:, :, :height//2, width//2:]
    image3 = big_img[:, :, height//2:, :width//2]
    image4 = big_img[:, :, height//2:, width//2:]

    batch_image = torch.stack([image1, image2, image3, image4], dim=0) # f, b, c, h, w
    batch_image = rearrange(batch_image, "f b c h w -> b c f h w")
    return batch_image



def concat_big_img(img_list,  width, height,):
    scale_transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
            ]
        )
    img1, img2, img3, img4 = scale_transform(img_list[0]), scale_transform(img_list[1]), scale_transform(img_list[2]), scale_transform(img_list[3])

    width, height = img1.size

    if  len(img1.getbands()) == 1:
        final_image = Image.new('L', (width * 2, height * 2))
    else:
        final_image = Image.new('RGB', (width * 2, height * 2))

    final_image.paste(img1, (0, 0))
    final_image.paste(img2, (width, 0))
    final_image.paste(img3, (0, height))
    final_image.paste(img4, (width, height))
    return final_image


def tensor2list(bs):
    reshaped_tensor = bs.reshape(-1, bs.size(-1))

    splitted_tensors = torch.split(reshaped_tensor, bs.size(1) * bs.size(2), dim=0)

    x_list = [t.view(bs.size(1), bs.size(2), bs.size(-1)) for t in splitted_tensors]
    return x_list


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def inference_pose(img_path, detector):
    print(img_path)
    pil_image = Image.open(img_path).convert("RGB")
    dwpose_image, score, _, _ = detector(pil_image, output_type='np')
    pil_dwpose_image = Image.fromarray(dwpose_image)

    return pil_dwpose_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/prompts/test_end2end.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--base_root", type=str,default='./data/output')
    parser.add_argument("--save_dir_name", type=str,default='./all_logs/test_results')
    parser.add_argument("--denoising_unet_path", type=str, default='.../stage1_log/denoising_unet-25001.pth')
    parser.add_argument("--pose_guider_path", type=str, default='.../stage1_log/pose_guider-25001.pth')
    parser.add_argument("--linear_path", type=str, default='.../stage1_log/linear_emb-25001.pth')
    parser.add_argument("--motion_module_path", type=str, default='.../stage2_log/motion_module-10001.pth')
 
    parser.add_argument("--json1_path", type=str, default=".../inference.json", help="test data json file")
    parser.add_argument("--json2_path", type=str, default=".../keypoints_results.json", help="Path to json2 file.")
    parser.add_argument("--image_folder_path", type=str, default=".../imgs",help="Folder path for ref images.")
    parser.add_argument("--pose_img_folder_path", type=str, default=".../dwposes", help="Folder path for pose images.")
    parser.add_argument("--img_width",type=int,default=768,help="img_width",)
    parser.add_argument("--img_height",type=int,default=768,help="img_height",)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    if args.denoising_unet_path is not None:
        config.denoising_unet_path = args.denoising_unet_path
    if args.motion_module_path is not None:
        config.motion_module_path = args.motion_module_path
    if args.linear_path is not None:
        config.linear_path = args.linear_path
    if args.pose_guider_path is not None:
        config.pose_guider_path = args.pose_guider_path

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)


    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    detector = DWposeDetector().to("cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda", dtype=weight_dtype)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    linear_emb = nn.Linear(1280, 768).to(
        dtype=weight_dtype, device="cuda"
    )

    linear_emb = linear_emb.to("cuda")

    image_enc = Dinov2Model.from_pretrained(
        config.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")


    image_encoder_1280 = CLIPVisionModelWithProjection.from_pretrained(
        "./Ckpts/kandinsky-2-2-prior",
        subfolder="image_encoder"
        ).to('cuda', dtype=weight_dtype)


    stage1_config = OmegaConf.load('./configs/testing.yaml')

    stage1_runner = Stage1InferenceRunner(
        stage1_config['unet_additional_kwargs'],
        module_ckpt = "./output_stage1/100500/mp_rank_00_model_states.pt",
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )

    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    linear_emb.load_state_dict(
        torch.load(config.linear_path, map_location="cpu"),
    )

    linear_emb = linear_emb.to("cuda")

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    base_root = args.base_root
    save_dir_name = args.save_dir_name
    save_dir = Path(f"{base_root}/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    clip_image_processor = CLIPImageProcessor()

    # stage1 datas begin
    inference_num = 1
    # for _____ in range(inference_num):
    #     with open(args.json1_path, 'r') as f:
    #         test_data_list = json.load(f)
    #     with open(args.json2_path, 'r') as f:
    #         pose_data_dict = json.load(f)

    #     index = random.choice(range(len(test_data_list)))
    #     data_item = test_data_list[index]

    with open(args.json1_path, 'r') as f:
        test_data_list = json.load(f)

    with open(args.json2_path, 'r') as f:
        pose_data_dict = json.load(f)

    for index in range(inference_num):
        _____ = index
        data_item = test_data_list[index]

        image_paths = []
        for i in range(4):
            img_name = data_item[f'img{i+1}']
            candidate_path = os.path.join(args.image_folder_path, img_name)
            if os.path.exists(candidate_path):
                image_paths.append(candidate_path)
            else:
                fallback_path = os.path.join("../another_path/imgs", img_name)
                image_paths.append(fallback_path)
        images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
        images_pil = [Image.fromarray(img) for img in images]

        reference_images = [
            clip_image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            for img in images
        ]
        reference_image = torch.stack(reference_images, dim=0).to(memory_format=torch.contiguous_format).float().to("cuda") # 4 3 224 224
        print(f"Reference images shape: {reference_image.shape}")

        # visible_situation = generate_array() # 0 unmask / 1 mask
        visible_situation = [0, 0, 1, 1]

        print(f"Visible situation: {visible_situation}")

        black_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (0, 0, 0)),
                                                    return_tensors="pt").pixel_values).squeeze(dim=0)
        white_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (255, 255, 255)),
                                                    return_tensors="pt").pixel_values).squeeze(dim=0)

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

            imgs_proj_embeds = image_encoder_1280(source_clip.to('cuda'),output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]
            mask_label_embeds = image_encoder_1280(mask_label_clip.to('cuda'),output_hidden_states=True).image_embeds.unsqueeze(1)

        pose_list = []
        for i in range(4):
            img_name = data_item[f'img{i+1}']
            pose_entry = next((item for item in pose_data_dict if item["img"] == img_name), None)
            
            coords = None
            if pose_entry and "coordinate" in pose_entry and len(pose_entry["coordinate"]) > 0:
                print(f"Found pre-computed pose for '{img_name}' in JSON file.")
                coords = pose_entry["coordinate"]
            else:
                print(f"Pose for '{img_name}' not found or is empty in JSON. Inferring on-the-fly...")
                pil_image_for_pose = images_pil[i]
                bgr_image_for_pose = cv2.cvtColor(np.array(pil_image_for_pose), cv2.COLOR_RGB2BGR)
                
                detector_output = detector(bgr_image_for_pose)
                if detector_output is not None and len(detector_output) > 1 and isinstance(detector_output[1], np.ndarray):
                    keypoints_np = detector_output[1]
                    coords = keypoints_np.tolist()
                    print(f"Successfully inferred pose for '{img_name}'. Detected {len(coords)} keypoints.")
                else:
                    print(f"Warning: Pose detection failed for '{img_name}'. Using a default zero-pose.")
                    coords = [[0, 0] for _ in range(18)]


            coords_flat = [value for pair in coords for value in pair]
            
            if len(coords_flat) < 36:
                padding = [0] * (36 - len(coords_flat))
                print("less than 36, padding")
                coords_flat.extend(padding)
            
            coords_flat = coords_flat[:36]
            pose_list.append(coords_flat)

        input_pose_coordinate = torch.tensor(pose_list, dtype=torch.float32).to('cuda')  # 4 36

        predict_emb = stage1_runner(imgs_proj_embeds, mask_label_embeds, input_pose_coordinate)
        predict_emb = predict_emb.to('cuda')

        if predict_emb.ndim == 2:
            predict_emb = predict_emb.unsqueeze(0)

        embeds_768 = linear_emb(predict_emb)
        # stage1 datas end

        images_pil = [Image.fromarray(img) for img in images]
        pose_paths = []
        for i in range(4):
            img_name = data_item[f'img{i+1}']
            candidate_pose_path = os.path.join(args.pose_img_folder_path, img_name)
            if os.path.exists(candidate_pose_path):
                pose_paths.append(candidate_pose_path)
            else:
                fallback_pose_path = os.path.join(".../another_path/dwpose", img_name)
                pose_paths.append(fallback_pose_path)


        pose_pil_list = [Image.open(p).convert("RGB") for p in pose_paths]
        
        pose_pil_big_image = concat_big_img(pose_pil_list, args.W, args.H)

        black_img_pil = Image.new("RGB", (args.W, args.H), (0, 0, 0))
        image_mask_pil_image_list = []
        for i in range(4):
            if visible_situation[i] == 0:
                image_mask_pil_image_list.append(images_pil[i])
            else:
                image_mask_pil_image_list.append(black_img_pil)
        
        image_mask_big_image = concat_big_img(image_mask_pil_image_list, args.W, args.H)

        white1 = Image.new("L", (args.W // 8, args.H // 8), 255)
        black0 = Image.new("L", (args.W // 8, args.H // 8), 0)
        flag_label_list = []
        for i in range(4):
            if visible_situation[i] == 0: 
                flag_label_list.append(white1)
            else:
                flag_label_list.append(black0)

        flag_label_mask_big_image = concat_big_img(flag_label_list, args.W // 8, args.H // 8)
        pixel_values_flag_label = transforms.ToTensor()(flag_label_mask_big_image).to(device="cuda", dtype=weight_dtype)

        image = pipe(
            pose_image=pose_pil_big_image,
            image_mask=image_mask_big_image,
            flag_label=pixel_values_flag_label,
            encoder_hidden_states=embeds_768,
            class_labels=predict_emb,
            width=width * 2,
            height=height * 2,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        ).images  # b,h,w,c


        create_comparison_image = True
        result_big_image = image[0]

        output_filename = data_item['img1'].split('.')[0]
        save_path_base = f"{save_dir}/{output_filename}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}"

        if create_comparison_image:
            original_big_image = concat_big_img(images_pil, args.W, args.H)
            comparison_image = stitch_images_horizontally([
                original_big_image,
                image_mask_big_image,
                pose_pil_big_image,
                result_big_image
            ])
            comparison_image.save(f"{save_path_base}.png")
            print(f"save: {save_dir}/{_____}_{output_filename}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_comparison.png")
        else:
            result_big_image.save(f"{save_path_base}.png")
            print(f"save: {save_dir}/{output_filename}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.png")



if __name__ == "__main__":
    main()