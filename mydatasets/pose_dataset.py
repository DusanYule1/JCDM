import json
import os
import random
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageChops
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class ResizeAspect(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):
        w, h = img.size
        if w / h > self.output_size[1] / self.output_size[0]:
            oh = self.output_size[0]
            ow = int(self.output_size[0] * w / h)
        else:
            ow = self.output_size[1]
            oh = int(self.output_size[1] * h / w)
        return img.resize((ow, oh), Image.BICUBIC)


def augmentation(images, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    if isinstance(images, List):
        transformed_images = [transform(img) for img in images]
        ret_tensor = torch.stack(transformed_images, dim=0)
    else:
        ret_tensor = transform(images)
    return ret_tensor


def concat_big_img(img_list, width, height, state):
    scale_transform = transforms.Compose(
        [
            ResizeAspect((height, width)),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]
    )

    img1 = augmentation(img_list[0], scale_transform, state[0])
    img2 = augmentation(img_list[1], scale_transform, state[1])
    img3 = augmentation(img_list[2], scale_transform, state[2])
    img4 = augmentation(img_list[3], scale_transform, state[3])

    width, height = img1.size

    if len(img1.getbands()) == 1:
        final_image = Image.new('L', (width * 2, height * 2))
    else:
        final_image = Image.new('RGB', (width * 2, height * 2))

    final_image.paste(img1, (0, 0))
    final_image.paste(img2, (width, 0))
    final_image.paste(img3, (0, height))
    final_image.paste(img4, (width, height))

    return final_image


class HumanPoseDataset(Dataset):
    def __init__(self, width, height, image_root, pose_root, json_file):
        super().__init__()

        with open(json_file, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        print('Loaded dataset size:', len(self.data))

        self.image_root = image_root
        self.pose_root = pose_root
        self.width = width
        self.height = height

        self.clip_image_processor = CLIPImageProcessor.from_pretrained("/opt/data/private/Ckpts/clip-vit-large-patch14")

        self.pixel_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.ref_vae_transform = transforms.Compose(
            [
                ResizeAspect((height, width)),
                transforms.RandomCrop((height, width)),
                transforms.ToTensor(),
            ]
        )


        self.alt_image_root = "/opt/data/private/Datasets/final_dance/imgs"
        self.alt_pose_root = "/opt/data/private/Datasets/final_dance/dwpose"

    def __getitem__(self, index):
        image_meta = self.data[index]
        img_filenames = [
            image_meta["img1"],
            image_meta["img2"],
            image_meta["img3"],
            image_meta["img4"],
        ]

        person_pil_image_list = []
        pose_pil_image_list = []

        # for img_filename in img_filenames:
        #     img_path = os.path.join(self.image_root, img_filename)
        #     pose_path = os.path.join(self.pose_root, img_filename)
        for img_filename in img_filenames:
            # 原始路径
            img_path = os.path.join(self.image_root, img_filename)
            pose_path = os.path.join(self.pose_root, img_filename)

            # 如果找不到就换备用路径
            if not os.path.exists(img_path):
                img_path = os.path.join(self.alt_image_root, img_filename)
            if not os.path.exists(pose_path):
                pose_path = os.path.join(self.alt_pose_root, img_filename)


            person_pil_image_list.append(Image.open(img_path).convert("RGB"))
            pose_pil_image_list.append(Image.open(pose_path).convert("RGB"))

        # sync random state for all transforms
        state = [torch.get_rng_state() for _ in range(4)]

        person_pil_big_image = concat_big_img(person_pil_image_list, self.width, self.height, state)
        pixel_values_person = self.pixel_transform(person_pil_big_image)

        pose_pil_big_image = concat_big_img(pose_pil_image_list, self.width, self.height, state)
        pixel_values_pose = self.cond_transform(pose_pil_big_image)

        mask_choices = [
            [1, 0, 0, 0],   # 尝试一张可见的概率更大
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ]
        random_list = random.choice(mask_choices)

        # image + mask
        image_mask_pil_image_list = [
            ImageChops.multiply(person_pil_image, Image.new('RGB', person_pil_image.size, (mask * 255,) * 3))
            for person_pil_image, mask in zip(person_pil_image_list, random_list)
        ]
        image_mask_big_image = concat_big_img(image_mask_pil_image_list, self.width, self.height, state)
        pixel_values_image_mask = self.pixel_transform(image_mask_big_image)

        # flag label
        white1 = Image.new("L", (self.width // 8, self.height // 8), 255)
        flag_label = [
            ImageChops.multiply(white1, Image.new('L', white1.size, (random_list[i] * 255)))
            for i in range(4)
        ]
        flag_label_mask_big_image = concat_big_img(flag_label, self.width // 8, self.height // 8, state)
        pixel_values_flag_label = self.cond_transform(flag_label_mask_big_image)

        # clip and vae ref image
        index_list = [i for i, v in enumerate(random_list) if v == 1]
        random_index = random.choice(index_list)

        clip_ref_img = self.clip_image_processor(images=person_pil_image_list[random_index], return_tensors="pt").pixel_values[0]
        vae_ref_img = augmentation(person_pil_image_list[random_index], self.ref_vae_transform, state[random_index])

        # 四张全部过clip，变成 (4, 3, 224, 224)
        reference_images_clip_list = self.clip_image_processor(
            images=person_pil_image_list, return_tensors="pt"
        ).pixel_values  # shape: (4, 3, 224, 224)


        sample = dict( # width 512 height 768 -> (b, c, h, w)
            pixel_values_person=pixel_values_person, # (b, 3, 1024, 1024)
            pixel_values_pose=pixel_values_pose, # (b, 3, 1024, 1024)
            pixel_values_image_mask= pixel_values_image_mask, # (b, 3, 1024, 1024)
            pixel_values_flag_label = pixel_values_flag_label, # (b, 1, 128, 128)
            clip_ref_img=clip_ref_img, # (b, 3, 224, 224)
            vae_ref_img =vae_ref_img, # (b, 3, 512, 512)
            reference_images_clip=reference_images_clip_list,  # (b, 4, 3, 224, 224)
        )

        return sample

    def __len__(self):
        return len(self.data)
