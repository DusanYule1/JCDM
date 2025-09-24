import random
import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from einops import rearrange
import torch.nn.functional as F
from torch import nn

class MyPoseDataset(Dataset):
    def __init__(self,
                 data_json_path,
                 pose_json_path,
                 image_folder_path,
                 num_frames=4,
                 size=512,
                 max_masked_frames=3,
                 min_masked_frames=1):
        super(MyPoseDataset, self).__init__()

        self.image_folder_path = image_folder_path
        self.num_frames = num_frames
        self.size = size

        self.max_masked_frames = max_masked_frames
        self.min_masked_frames = min_masked_frames

        with open(data_json_path, 'r') as f:
            self.image_sequences = json.load(f)

        self.pose_data_map = {}
        with open(pose_json_path, 'r') as f:
            pose_list = json.load(f)
            for item in pose_list:
                self.pose_data_map[item["img"]] = item["coordinate"]

        self.clip_image_processor = CLIPImageProcessor.from_pretrained("/opt/data/private/Ckpts/clip-vit-large-patch14")

        self.augment = transforms.Compose([
            transforms.Resize([self.size, self.size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.mask_augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.black_img_pil = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        self.black_img_clip = (self.clip_image_processor(images=Image.new("RGB", (self.size, self.size), (0, 0, 0)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)
        self.white_img_clip = (self.clip_image_processor(images=Image.new("RGB", (self.size, self.size), (255, 255, 255)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, index):
        sequence_info = self.image_sequences[index]

        all_frames_pil = []
        all_frames_clip = []
        all_pose_coordinates = []

        for i in range(1, self.num_frames + 1):
            img_filename = sequence_info[f"img{i}"]
            img_path = os.path.join(self.image_folder_path, img_filename)
            try:
                pil_image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"Dataset Warning: Image file not found {img_path}, using black image instead.")
                pil_image = self.black_img_pil.copy()
            
            all_frames_pil.append(pil_image)

            clip_processed_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values.squeeze(dim=0)
            all_frames_clip.append(clip_processed_image)

            if img_filename in self.pose_data_map:
                pose_coords = self.pose_data_map[img_filename]
                pose_coords_flat = [coord for pair in pose_coords for coord in pair]
                all_pose_coordinates.append(torch.tensor(pose_coords_flat, dtype=torch.float32))
            else:
                print(f"Warning: Pose data not found for {img_filename}, using zeros.")
                num_keypoints = len(self.pose_data_map.get(next(iter(self.pose_data_map)), [[0,0]]*18))
                all_pose_coordinates.append(torch.zeros(num_keypoints * 2, dtype=torch.float32))

        image0 = self.augment(all_frames_pil[0])
        image1 = self.augment(all_frames_pil[1])
        image2 = self.augment(all_frames_pil[2])
        image3 = self.augment(all_frames_pil[3])
        black_img = self.mask_augment(Image.new("RGB", (self.size,self.size), (0, 0, 0)))

        black0 = torch.zeros(1, int(self.size / 8), int(self.size / 8))
        white1 = torch.ones(1, int(self.size / 8), int(self.size / 8))

        num_to_mask = random.randint(self.min_masked_frames, self.max_masked_frames)

        if num_to_mask == 3:
            source   =  torch.stack([image0, black_img, black_img, black_img], dim=0)
            source_clip = torch.stack([all_frames_clip[0], self.black_img_clip, self.black_img_clip, self.black_img_clip], dim=0)
            mask_label = torch.stack([white1, black0, black0, black0], dim=0)
            mask_label_clip = torch.stack([self.white_img_clip, self.black_img_clip, self.black_img_clip, self.black_img_clip], dim=0)
        elif num_to_mask == 2:
            source   =  torch.stack([image0, image1, black_img, black_img], dim=0)
            source_clip = torch.stack([all_frames_clip[0], all_frames_clip[1], self.black_img_clip, self.black_img_clip], dim=0)
            mask_label = torch.stack([white1, white1, black0, black0], dim=0)
            mask_label_clip = torch.stack([self.white_img_clip, self.white_img_clip, self.black_img_clip, self.black_img_clip], dim=0)
        elif num_to_mask == 1:
            source   =  torch.stack([image0, image1, image2, black_img], dim=0)
            source_clip = torch.stack([all_frames_clip[0], all_frames_clip[1], all_frames_clip[2], self.black_img_clip], dim=0)
            mask_label = torch.stack([white1, white1, white1, black0], dim=0)
            mask_label_clip = torch.stack([self.white_img_clip, self.white_img_clip, self.white_img_clip, self.black_img_clip], dim=0)
        else:
            source   =  torch.stack([black_img, black_img, black_img, black_img], dim=0)
            source_clip = torch.stack([self.black_img_clip, self.black_img_clip, self.black_img_clip, self.black_img_clip], dim=0)
            mask_label = torch.stack([black0, black0, black0, black0], dim=0)
            mask_label_clip = torch.stack([self.black_img_clip, self.black_img_clip, self.black_img_clip, self.black_img_clip], dim=0)

        pose_coordinates_tensor = torch.stack(all_pose_coordinates)

        reference_image = torch.stack([image0, image1, image2, image3], dim=0)
        reference_image_clip = torch.stack(all_frames_clip)

        return {
            "source": source,
            "source_clip": source_clip,
            "reference_image": reference_image,
            "reference_image_clip": reference_image_clip,
            "mask_label": mask_label,
            "mask_label_clip": mask_label_clip,
            "pose_coordinates": pose_coordinates_tensor
        }

def MyPoseCollateFn(data):
    reference_image_clip = torch.stack([example["reference_image_clip"] for example in data]).to(memory_format=torch.contiguous_format).float()
    source_clip_image = torch.stack([example["source_clip"] for example in data]).to(memory_format=torch.contiguous_format).float()
    mask_label_image = torch.stack([example["mask_label_clip"] for example in data]).to(memory_format=torch.contiguous_format).float()

    source_image = torch.stack([example["source"] for example in data])
    source_image = source_image.to(memory_format=torch.contiguous_format).float()

    target_image = torch.stack([example["reference_image"] for example in data])
    target_image = target_image.to(memory_format=torch.contiguous_format).float()

    masked_label = torch.stack([example["mask_label"] for example in data])
    masked_label = masked_label.to(memory_format=torch.contiguous_format).float()

    pose_coordinates = torch.stack([example["pose_coordinates"] for example in data])
    pose_coordinate = pose_coordinates.to(memory_format=torch.contiguous_format).float()

    return {
        "source_clip_image": source_clip_image,
        "reference_image": reference_image_clip,
        "source_image": source_image,
        "target_image": target_image,
        "masked_label":masked_label,
        "masked_label_clip":mask_label_image,
        "pose_coordinate":pose_coordinate
    }

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
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

if __name__ == "__main__":
    YOUR_DATA_JSON_PATH = "/opt/data/private/output_100.json"
    YOUR_POSE_JSON_PATH = "/opt/data/private/keypoints_results.json"
    YOUR_IMAGE_FOLDER_PATH = "/opt/data/private/Datasets/final_dance/imgs"

    if not os.path.isdir(YOUR_IMAGE_FOLDER_PATH):
        print(f"Error: image_folder_path not found: {YOUR_IMAGE_FOLDER_PATH}")
        exit()

    print("\nInitializing MyPoseDataset...")
    try:
        dataset = MyPoseDataset(
            data_json_path=YOUR_DATA_JSON_PATH,
            pose_json_path=YOUR_POSE_JSON_PATH,
            image_folder_path=YOUR_IMAGE_FOLDER_PATH
        )

        if len(dataset) > 0:
            print(f"Dataset initialized successfully, contains {len(dataset)} samples.")

            print("\nGetting first sample (dataset[0])...")
            sample = dataset[0]

            print("\nTensor shapes in sample:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
                else:
                    print(f"  - {key}: (non-tensor) {type(value)}")

            print("\nTesting MyPoseCollateFn...")
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=MyPoseCollateFn)

            try:
                batch_sample = next(iter(dataloader))
                print("\nShapes in batch sample (batch_size=2):")
                for key, value in batch_sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  - {key}: {value.shape}")
                    else:
                        print(f"  - {key}: (non-tensor) {type(value)}")

                print("\nTesting image_encoder...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                weight_dtype = torch.float32

                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "/opt/data/private/Ckpts/kandinsky-2-2-prior",
                    subfolder="image_encoder"
                ).to(device, dtype=weight_dtype)

                target_image = batch_sample["reference_image"]
                target_image = rearrange(target_image, "b f c h w -> (b f) c h w")
                image_embeds = image_encoder(target_image.to(device, dtype=weight_dtype)).image_embeds.unsqueeze(1)

                print("\nImage Encoder test result:")
                print(f"  - image_embeds: {image_embeds.shape}")

            except StopIteration:
                print("Error: DataLoader is empty, cannot get batch sample.")
            except Exception as e_collate:
                print(f"Error occurred in CollateFn or Image Encoder test: {e_collate}")
                import traceback
                traceback.print_exc()

        else:
            print("Error: Dataset is empty. Check your JSON files and image folder.")

    except Exception as e:
        print(f"\nError running main: {e}")
        import traceback
        traceback.print_exc()

    pose_encoder = MLP(in_dim=36, hidden_dim=512, out_dim=1280)
    pose_coordinate = batch_sample["pose_coordinate"]
    print(f"\nPose Coordinate test result:")
    print(f"  - pose_coordinate: {pose_coordinate.shape}")
    pose_coordinate = rearrange(pose_coordinate, "b f x -> (b f) x")
    pose_embeds = pose_encoder(pose_coordinate).unsqueeze(1)
    print(f"\nPose Encoder test result:")
    print(f"  - pose_embeds: {pose_embeds.shape}")
