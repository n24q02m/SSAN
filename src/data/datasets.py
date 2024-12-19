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


class Config:
    """Configuration class to handle different environments"""

    def __init__(self):
        self.is_kaggle = "KAGGLE_KERNEL_RUN" in os.environ
        if self.is_kaggle:
            self.data_dir = "/kaggle/input"
            self.working_dir = "/kaggle/working/SSAN_Enhance"
            self.dataset_paths = {
                "celeba": f"{self.data_dir}/celeba-spoof-face-anti-spoofing-dataset",
                "cati": f"{self.data_dir}/cati-fas-face-anti-spoofing-dataset",
                "lcc": f"{self.data_dir}/lcc-fasd-face-anti-spoofing-dataset",
                "nuaaa": f"{self.data_dir}/nuaaa-face-anti-spoofing-dataset",
                "zalo": f"{self.data_dir}/zalo-aic-face-anti-spoofing-dataset",
            }
        else:
            self.data_dir = "./data"
            self.working_dir = "."
            self.dataset_paths = {
                "celeba": f"{self.data_dir}/CelebA_Spoof_dataset",
                "cati": f"{self.data_dir}/CATI_FAS_dataset",
                "lcc": f"{self.data_dir}/LCC_FASD_dataset",
                "nuaaa": f"{self.data_dir}/NUAAA_dataset",
                "zalo": f"{self.data_dir}/Zalo_AIC_dataset",
            }

        # Common configurations
        self.img_size = 256
        self.output_dir = f"{self.working_dir}/output"
        os.makedirs(self.output_dir, exist_ok=True)


def get_transforms(mode="train"):
    """Get albumentations transforms for train/val/test"""
    if mode == "train":
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
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
            ]
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
    def __init__(self, image_paths, bbox_paths, labels, transform=None):
        self.image_paths = image_paths
        self.bbox_paths = bbox_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bbox_path = self.bbox_paths[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class DatasetManager:
    """Manager class for handling datasets"""

    def __init__(self):
        self.config = Config()

    def load_dataset_paths(self, dataset_name):
        """Load image paths and labels for a dataset"""
        dataset_path = self.config.dataset_paths[dataset_name]

        # Handle different directory structures
        if dataset_name in ["celeba", "cati"]:
            live_path = os.path.join(dataset_path, "live", "live")
            spoof_path = os.path.join(dataset_path, "spoof", "spoof")
        else:
            live_path = os.path.join(dataset_path, "live")
            spoof_path = os.path.join(dataset_path, "spoof")

        # Get both jpg and png images
        live_images = []
        spoof_images = []
        for ext in ["*.jpg", "*.png"]:
            live_images.extend(
                glob.glob(os.path.join(live_path, "**", ext), recursive=True)
            )
            spoof_images.extend(
                glob.glob(os.path.join(spoof_path, "**", ext), recursive=True)
            )

        # Get corresponding bbox files
        live_bbox = [p.replace(os.path.splitext(p)[1], "_BB.txt") for p in live_images]
        spoof_bbox = [
            p.replace(os.path.splitext(p)[1], "_BB.txt") for p in spoof_images
        ]

        # Filter out files without bbox
        valid_live = [
            (img, bbox)
            for img, bbox in zip(live_images, live_bbox)
            if os.path.exists(bbox)
        ]
        valid_spoof = [
            (img, bbox)
            for img, bbox in zip(spoof_images, spoof_bbox)
            if os.path.exists(bbox)
        ]

        # Unzip into separate lists
        image_paths = [x[0] for x in valid_live + valid_spoof]
        bbox_paths = [x[1] for x in valid_live + valid_spoof]
        labels = [1] * len(valid_live) + [0] * len(valid_spoof)

        return image_paths, bbox_paths, labels

    def read_bbox(self, bbox_path):
        """Read bounding box coordinates from file"""
        with open(bbox_path, "r") as f:
            lines = f.readlines()
            y1, x1, w, h = map(float, lines[:4])
        return x1, y1, w, h

    def sample_and_save_images(self, dataset_name, num_samples=10, seed=42):
        """Sample and save random images with bbox overlay from dataset"""
        np.random.seed(seed)
        random.seed(seed)

        image_paths, bbox_paths, _ = self.load_dataset_paths(dataset_name)
        indices = random.sample(
            range(len(image_paths)), min(num_samples, len(image_paths))
        )

        output_dir = os.path.join(self.config.output_dir, f"{dataset_name}_sample")
        os.makedirs(output_dir, exist_ok=True)

        for i, idx in enumerate(indices):
            # Read image
            img_path = image_paths[idx]
            bbox_path = bbox_paths[idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw bbox
            x1, y1, w, h = self.read_bbox(bbox_path)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=(0, 255, 0),
                thickness=2,
            )

            # Save visualization
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"sample_{i}.png"))
            plt.close()

    def get_protocol_splits(self, protocol_num):
        """Get dataset splits according to protocol number"""
        if protocol_num == 1:
            # Protocol 1: Large-Scale Training
            celeba_paths, celeba_labels = self.load_dataset_paths("celeba")
            total = len(celeba_paths)
            train_idx = int(0.9 * total)

            # Shuffle consistently
            indices = np.arange(total)
            np.random.seed(42)
            np.random.shuffle(indices)

            train_paths = [celeba_paths[i] for i in indices[:train_idx]]
            train_labels = [celeba_labels[i] for i in indices[:train_idx]]
            val_paths = [celeba_paths[i] for i in indices[train_idx:]]
            val_labels = [celeba_labels[i] for i in indices[train_idx:]]

            test_paths, test_labels = self.load_dataset_paths("cati")

        elif protocol_num == 2:
            # Protocol 2: Multi-Domain Training
            celeba_paths, celeba_labels = self.load_dataset_paths("celeba")
            lcc_paths, lcc_labels = self.load_dataset_paths("lcc")
            nuaaa_paths, nuaaa_labels = self.load_dataset_paths("nuaaa")

            # Use 50% of CelebA-Spoof
            celeba_indices = np.random.choice(
                len(celeba_paths), size=len(celeba_paths) // 2, replace=False
            )
            celeba_paths = [celeba_paths[i] for i in celeba_indices]
            celeba_labels = [celeba_labels[i] for i in celeba_indices]

            train_paths = celeba_paths + lcc_paths + nuaaa_paths
            train_labels = celeba_labels + lcc_labels + nuaaa_labels

            # Use 10% for validation
            total = len(train_paths)
            train_idx = int(0.9 * total)
            indices = np.arange(total)
            np.random.shuffle(indices)

            val_paths = [train_paths[i] for i in indices[train_idx:]]
            val_labels = [train_labels[i] for i in indices[train_idx:]]
            train_paths = [train_paths[i] for i in indices[:train_idx]]
            train_labels = [train_labels[i] for i in indices[:train_idx]]

            test_paths, test_labels = self.load_dataset_paths("cati")

        elif protocol_num == 3:
            # Protocol 3: Balanced Small-Scale Training
            lcc_paths, lcc_labels = self.load_dataset_paths("lcc")
            nuaaa_paths, nuaaa_labels = self.load_dataset_paths("nuaaa")
            zalo_paths, zalo_labels = self.load_dataset_paths("zalo")

            train_paths = lcc_paths + nuaaa_paths + zalo_paths
            train_labels = lcc_labels + nuaaa_labels + zalo_labels

            # Use 10% for validation
            total = len(train_paths)
            train_idx = int(0.9 * total)
            indices = np.arange(total)
            np.random.shuffle(indices)

            val_paths = [train_paths[i] for i in indices[train_idx:]]
            val_labels = [train_labels[i] for i in indices[train_idx:]]
            train_paths = [train_paths[i] for i in indices[:train_idx]]
            train_labels = [train_labels[i] for i in indices[:train_idx]]

            test_paths, test_labels = self.load_dataset_paths("cati")

        elif protocol_num == 4:
            # Protocol 4: Cross-Domain Testing
            celeba_paths, celeba_labels = self.load_dataset_paths("celeba")
            lcc_paths, lcc_labels = self.load_dataset_paths("lcc")
            nuaaa_paths, nuaaa_labels = self.load_dataset_paths("nuaaa")

            train_paths = celeba_paths + lcc_paths + nuaaa_paths
            train_labels = celeba_labels + lcc_labels + nuaaa_labels

            # Use 10% for validation
            total = len(train_paths)
            train_idx = int(0.9 * total)
            indices = np.arange(total)
            np.random.shuffle(indices)

            val_paths = [train_paths[i] for i in indices[train_idx:]]
            val_labels = [train_labels[i] for i in indices[train_idx:]]
            train_paths = [train_paths[i] for i in indices[:train_idx]]
            train_labels = [train_labels[i] for i in indices[:train_idx]]

            test_paths, test_labels = self.load_dataset_paths("zalo")

        return (
            (train_paths, train_labels),
            (val_paths, val_labels),
            (test_paths, test_labels),
        )


def get_dataloaders(protocol_num, batch_size=32, num_workers=4):
    manager = DatasetManager()
    (
        (train_paths, train_bbox, train_labels),
        (val_paths, val_bbox, val_labels),
        (test_paths, test_bbox, test_labels),
    ) = manager.get_protocol_splits(protocol_num)

    # Only apply augmentation to training data
    train_dataset = FASDataset(
        train_paths, train_bbox, train_labels, transform=get_transforms("train")
    )
    val_dataset = FASDataset(
        val_paths, val_bbox, val_labels, transform=get_transforms("val")
    )
    test_dataset = FASDataset(
        test_paths, test_bbox, test_labels, transform=get_transforms("test")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def main():
    """Main function for testing the implementation"""
    manager = DatasetManager()

    # Sample and save images from each dataset
    for dataset_name in ["celeba", "cati", "lcc", "nuaaa", "zalo"]:
        print(f"Sampling images from {dataset_name}...")
        manager.sample_and_save_images(dataset_name)

    # Test dataloader creation for each protocol
    for protocol in range(1, 5):
        print(f"\nTesting Protocol {protocol}")
        train_loader, val_loader, test_loader = get_dataloaders(protocol)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
