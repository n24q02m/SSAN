import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Config:
    """Configuration class to handle different environments"""

    def __init__(self):
        self.is_kaggle = "KAGGLE_KERNEL_RUN" in os.environ
        if self.is_kaggle:
            self.data_dir = "/kaggle/input"
            self.working_dir = "/kaggle/working/SSAN_Enhance"
            self.dataset_paths = {
                "CelebA_Spoof": f"{self.data_dir}/celeba-spoof-face-anti-spoofing-dataset",
                "CATI_FAS": f"{self.data_dir}/cati-fas-face-anti-spoofing-dataset",
                "LCC_FASD": f"{self.data_dir}/lcc-fasd-face-anti-spoofing-dataset",
                "NUAAA": f"{self.data_dir}/nuaaa-face-anti-spoofing-dataset",
                "Zalo_AIC": f"{self.data_dir}/zalo-aic-face-anti-spoofing-dataset",
            }
        else:
            self.data_dir = "./data"
            self.working_dir = "."
            self.dataset_paths = {
                "CelebA_Spoof": f"{self.data_dir}/CelebA_Spoof_dataset",
                "CATI_FAS": f"{self.data_dir}/CATI_FAS_dataset",
                "LCC_FASD": f"{self.data_dir}/LCC_FASD_dataset",
                "NUAAA": f"{self.data_dir}/NUAAA_dataset",
                "Zalo_AIC": f"{self.data_dir}/Zalo_AIC_dataset",
            }

        # Common configurations
        self.img_size = 256
        self.output_dir = f"{self.working_dir}/output"
        self.seed = 42
        self.num_workers = 4 # Bạn có thể điều chỉnh số lượng workers
        os.makedirs(self.output_dir, exist_ok=True)

def get_transforms(mode="train"):
    """Get albumentations transforms for train/val/test"""
    if mode == "train":
        return A.Compose(
            [
                A.Rotate(limit=10, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                    ],
                    p=0.3,
                ),
                A.Resize(Config().img_size, Config().img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
    else:
        return A.Compose(
            [
                A.Resize(Config().img_size, Config().img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

class FASDataset(Dataset):
    def __init__(self, data_dirs, config, transform=None, mode='train'):
        self.data_dirs = data_dirs
        self.config = config
        self.transform = transform
        self.mode = mode
        self.data = []

        if mode == 'train' or mode == 'val':
            for data_dir in data_dirs:
                dataset_name = os.path.basename(os.path.dirname(data_dir)) if os.path.basename(data_dir) in ["live", "spoof"] else os.path.basename(data_dir)
                if dataset_name in ["CelebA_Spoof", "CATI_FAS"]:
                    live_dir = os.path.join(data_dir, "live", "live")
                    spoof_dir = os.path.join(data_dir, "spoof", "spoof")
                else:
                    live_dir = os.path.join(data_dir, "live")
                    spoof_dir = os.path.join(data_dir, "spoof")

                # Gộp chung live và spoof images
                for img_path in glob.glob(os.path.join(live_dir, "*.jpg")) + glob.glob(os.path.join(live_dir, "*.png")):
                    if dataset_name in ["CelebA_Spoof", "CATI_FAS"]:
                        bbox_path = img_path.replace("live/live", "live/").replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    else:
                        bbox_path = img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    if os.path.exists(bbox_path):
                        self.data.append((img_path, 1, dataset_name, bbox_path))  # 1 for live

                for img_path in glob.glob(os.path.join(spoof_dir, "*.jpg")) + glob.glob(os.path.join(spoof_dir, "*.png")):
                    if dataset_name in ["CelebA_Spoof", "CATI_FAS"]:
                        bbox_path = img_path.replace("spoof/spoof", "spoof/").replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    else:
                        bbox_path = img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    if os.path.exists(bbox_path):
                        self.data.append((img_path, 0, dataset_name, bbox_path))  # 0 for spoof

            if mode == 'train':
                train_data, _ = train_test_split(self.data, test_size=1-config.train_ratio, random_state=config.seed)
                self.data = train_data
            elif mode == 'val':
                _, val_data = train_test_split(self.data, test_size=1-config.train_ratio, random_state=config.seed)
                self.data = val_data

        elif mode == 'test':
            data_dir = data_dirs[0] # CATI_FAS
            dataset_name = os.path.basename(data_dir)
            live_dir = os.path.join(data_dir, "live", "live")
            spoof_dir = os.path.join(data_dir, "spoof", "spoof")
            for img_path in glob.glob(os.path.join(live_dir, "*.jpg")) + glob.glob(os.path.join(live_dir, "*.png")):
                bbox_path = img_path.replace("live/live", "live/").replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                if os.path.exists(bbox_path):
                    self.data.append((img_path, 1, dataset_name, bbox_path))  # 1 for live
            for img_path in glob.glob(os.path.join(spoof_dir, "*.jpg")) + glob.glob(os.path.join(spoof_dir, "*.png")):
                bbox_path = img_path.replace("spoof/spoof", "spoof/").replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                if os.path.exists(bbox_path):
                    self.data.append((img_path, 0, dataset_name, bbox_path))  # 0 for spoof

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, dataset_name, bbox_path = self.data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lấy bounding box
        with open(bbox_path, "r") as f:
            bbox_info = f.readline().strip().split()
            x, y, w, h = map(int, bbox_info)

        # Cắt ảnh theo bounding box
        image = image[y:y+h, x:x+w]

        # Resize
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))

        # Depth Map (Không áp dụng Dense Face Alignment, gán depth map bằng 0)
        depth_map = np.zeros((32, 32))

        # Data Augmentation (chỉ áp dụng cho train)
        if self.mode == 'train' and self.transform:
            transformed = self.transform(image=image, bboxes=[[0, 0, self.config.img_size - 1, self.config.img_size - 1]], labels=[label]) # Thêm bbox giả
            image = transformed["image"]

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(self.config.dataset_names.index(dataset_name), dtype=torch.long)

        return image, depth_map, label, domain

# Phân chia dữ liệu theo protocol
def split_data(dataset_paths, config):
    if config.protocol == "protocol_1":
        train_dirs = [dataset_paths["CelebA_Spoof"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
    elif config.protocol == "protocol_2":
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        val_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
    elif config.protocol == "protocol_3":
        train_dirs = [dataset_paths["LCC_FASD"], dataset_paths["NUAAA"], dataset_paths["Zalo_AIC"]]
        val_dirs = [dataset_paths["LCC_FASD"], dataset_paths["NUAAA"], dataset_paths["Zalo_AIC"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
    elif config.protocol == "protocol_4":
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        val_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        test_dirs = [dataset_paths["Zalo_AIC"]]
    else:
        raise ValueError(f"Invalid protocol: {config.protocol}")

    # Điều chỉnh tỷ lệ train/val cho protocol_1 và protocol_4
    if config.protocol == "protocol_1":
        config.train_ratio = 0.9
    elif config.protocol == "protocol_4":
        config.train_ratio = 0.8
    else:
        config.train_ratio = 0.8

    return train_dirs, val_dirs, test_dirs

# Thiết lập DataLoader
def get_dataloaders(train_dirs, val_dirs, test_dirs, config):
    train_dataset = FASDataset(train_dirs, config, transform=get_transforms("train"), mode='train')
    val_dataset = FASDataset(val_dirs, config, transform=get_transforms("val"), mode='val')
    test_dataset = FASDataset(test_dirs, config, transform=get_transforms("test"), mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


# In mẫu ảnh
def print_sample_images(dataset_paths, output_dir, num_samples=10, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, dataset_path in dataset_paths.items():
        if dataset_name in ["CelebA_Spoof", "CATI_FAS"]:
            live_dir = os.path.join(dataset_path, "live", "live")
            spoof_dir = os.path.join(dataset_path, "spoof", "spoof")
        else:
            live_dir = os.path.join(dataset_path, "live")
            spoof_dir = os.path.join(dataset_path, "spoof")

        live_images = glob.glob(os.path.join(live_dir, "*.jpg")) + glob.glob(
            os.path.join(live_dir, "*.png")
        )
        spoof_images = glob.glob(os.path.join(spoof_dir, "*.jpg")) + glob.glob(
            os.path.join(spoof_dir, "*.png")
        )

        if not live_images or not spoof_images:
            print(f"No images found for {dataset_name}")
            continue

        sample_dir = os.path.join(output_dir, f"{dataset_name}_sample")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"Printing {num_samples} sample images for {dataset_name}...")

        for i in range(num_samples):
            try:
                live_img_path = random.choice(live_images)
                spoof_img_path = random.choice(spoof_images)

                # Đọc ảnh gốc
                live_img = cv2.imread(live_img_path)
                spoof_img = cv2.imread(spoof_img_path)

                if live_img is None or spoof_img is None:
                    print(f"Could not read images: {live_img_path} or {spoof_img_path}")
                    continue

                # Convert to RGB for display
                live_img = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
                spoof_img = cv2.cvtColor(spoof_img, cv2.COLOR_BGR2RGB)

                # Get bounding box paths
                if dataset_name in ["CelebA_Spoof", "CATI_FAS"]:
                    live_bbox_path = (
                        live_img_path.replace("live/live", "live")
                        .replace(".jpg", "_BB.txt")
                        .replace(".png", "_BB.txt")
                    )
                    spoof_bbox_path = (
                        spoof_img_path.replace("spoof/spoof", "spoof")
                        .replace(".jpg", "_BB.txt")
                        .replace(".png", "_BB.txt")
                    )
                else:
                    live_bbox_path = live_img_path.replace(".jpg", "_BB.txt").replace(
                        ".png", "_BB.txt"
                    )
                    spoof_bbox_path = spoof_img_path.replace(".jpg", "_BB.txt").replace(
                        ".png", "_BB.txt"
                    )

                # Process cropped images
                live_img_cropped = live_img.copy()
                spoof_img_cropped = spoof_img.copy()

                if os.path.exists(live_bbox_path):
                    with open(live_bbox_path, "r") as f:
                        bbox_info = f.readline().strip().split()
                        x, y, w, h = map(int, bbox_info[:4])
                        # Draw rectangle on original image
                        cv2.rectangle(live_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Crop image
                        if (
                            y >= 0
                            and y + h <= live_img_cropped.shape[0]
                            and x >= 0
                            and x + w <= live_img_cropped.shape[1]
                        ):
                            live_img_cropped = live_img_cropped[y : y + h, x : x + w]

                if os.path.exists(spoof_bbox_path):
                    with open(spoof_bbox_path, "r") as f:
                        bbox_info = f.readline().strip().split()
                        x, y, w, h = map(int, bbox_info[:4])
                        # Draw rectangle on original image
                        cv2.rectangle(spoof_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Crop image
                        if (
                            y >= 0
                            and y + h <= spoof_img_cropped.shape[0]
                            and x >= 0
                            and x + w <= spoof_img_cropped.shape[1]
                        ):
                            spoof_img_cropped = spoof_img_cropped[y : y + h, x : x + w]

                # Save cropped images
                if live_img_cropped is not None and live_img_cropped.size > 0:
                    cv2.imwrite(
                        os.path.join(
                            sample_dir, f"{dataset_name}_live_{i}_cropped.jpg"
                        ),
                        cv2.cvtColor(live_img_cropped, cv2.COLOR_RGB2BGR),
                    )
                if spoof_img_cropped is not None and spoof_img_cropped.size > 0:
                    cv2.imwrite(
                        os.path.join(
                            sample_dir, f"{dataset_name}_spoof_{i}_cropped.jpg"
                        ),
                        cv2.cvtColor(spoof_img_cropped, cv2.COLOR_RGB2BGR),
                    )

                # Save original images with bbox
                plt.figure(figsize=(5, 5))
                plt.imshow(live_img)
                plt.axis("off")
                plt.savefig(os.path.join(sample_dir, f"{dataset_name}_live_{i}.png"))
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(spoof_img)
                plt.axis("off")
                plt.savefig(os.path.join(sample_dir, f"{dataset_name}_spoof_{i}.png"))
                plt.close()

            except Exception as e:
                print(f"Error processing sample {i} for {dataset_name}: {str(e)}")
                continue


# Hàm chính để chuẩn bị dữ liệu
def prepare_data(config):
    dataset_paths = config.dataset_paths
    config.dataset_names = list(dataset_paths.keys())
    print_sample_images(dataset_paths, config.output_dir, seed=config.seed)
    train_dirs, val_dirs, test_dirs = split_data(dataset_paths, config)
    dataloaders = get_dataloaders(train_dirs, val_dirs, test_dirs, config)
    return dataloaders

def main():
    """Main function for testing the implementation"""
    config = Config()
    config.batch_size = 8
    config.protocol = "protocol_1" # Bạn có thể thay đổi protocol

    # Test data preparation
    dataloaders = prepare_data(config)

    # Test dataloaders
    for mode in ["train", "val", "test"]:
        print(f"\nTesting {mode} dataloader")
        loader = dataloaders[mode]
        print(f"Number of batches: {len(loader)}")
        images, depth_maps, labels, domains = next(iter(loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Depth map batch shape: {depth_maps.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Domain batch shape: {domains.shape}")

if __name__ == "__main__":
    main()
