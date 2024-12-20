import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import torch
import numpy as np 
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import multiprocessing as mp
from functools import partial
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

from src.config import Config

def get_transforms(mode="train", config=None):
    """Get albumentations transforms for train/val/test"""
    if mode == "train":
        return A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Resize(config.img_size, config.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(config.img_size, config.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])

def split_data(dataset_paths, config):
    """Split data according to protocol"""
    if config.protocol == "protocol_1":
        # Single Large-Scale Dataset (CelebA-Spoof)
        train_dirs = [dataset_paths["CelebA_Spoof"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]] 
        test_dirs = [dataset_paths["CelebA_Spoof"]]
        config.train_ratio = 0.6  # 60% train, 20% val, 20% test

    elif config.protocol == "protocol_2":
        # Multi-Scale Training (CelebA-Spoof + CATI-FAS)
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        val_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        test_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        # CelebA-Spoof: 30%(train)/10%(val)/10%(test)
        # CATI-FAS: 60%(train)/20%(val)/20%(test)
        config.train_ratio = {
            "CelebA_Spoof": 0.6,  # 30% of total = 60% of 50%
            "CATI_FAS": 0.6       # 60% of total
        }

    elif config.protocol == "protocol_3":
        # Cross-Dataset Evaluation
        train_dirs = [dataset_paths["CATI_FAS"], dataset_paths["Zalo_AIC"]]
        val_dirs = [dataset_paths["CATI_FAS"], dataset_paths["Zalo_AIC"]]
        test_dirs = [dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        # CATI-FAS: 80%(train)/20%(val)
        # Zalo-AIC: 60%(train)/40%(val)
        # LCC-FASD, NUAAA: 30%(test)
        config.train_ratio = {
            "CATI_FAS": 0.8,
            "Zalo_AIC": 0.6
        }

    elif config.protocol == "protocol_4":
        # Domain Generalization
        train_dirs = [dataset_paths["CATI_FAS"], dataset_paths["LCC_FASD"],
                     dataset_paths["NUAAA"], dataset_paths["Zalo_AIC"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["CelebA_Spoof"]]
        # All medium datasets for training
        # CelebA-Spoof: 2.5%(val)/2.5%(test)
        config.train_ratio = 1.0  # Use all data for training
        config.val_test_ratio = 0.025  # 2.5% for val/test each

    else:
        raise ValueError(f"Invalid protocol: {config.protocol}")

    return train_dirs, val_dirs, test_dirs

def create_protocol_data(protocol, dataset_paths, config):
    print(f"\nCreating {protocol} splits...")
    config.protocol = protocol
    train_dirs, val_dirs, test_dirs = split_data(dataset_paths, config)
    
    protocol_dir = config.protocol_dir / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    for mode, dirs in [("train", train_dirs), ("val", val_dirs), ("test", test_dirs)]:
        data = []
        for data_dir in dirs:
            dataset_name = Path(data_dir).name
            if config.is_kaggle:
                dataset_name = dataset_name.replace("-face-anti-spoofing-dataset", "")
            else:
                dataset_name = dataset_name.replace("_dataset", "")
            
            for folder, label in [("live", 1), ("spoof", 0)]:
                folder_dir = Path(data_dir) / folder
                data.extend([
                    {
                        "dataset": dataset_name,
                        "filename": img_path.stem,
                        "label": label,
                        "folder": folder
                    }
                    for img_path in folder_dir.glob("*.[jp][pn][g]")
                    if (img_path.parent / f"{img_path.stem}_BB.txt").exists()
                ])
        
        all_data[mode] = pd.DataFrame(data)
    
    # Train/val split
    if "train" in all_data:
        if isinstance(config.train_ratio, dict):
            # Handle per-dataset ratios
            train_dfs = []
            for dataset_name, ratio in config.train_ratio.items():
                dataset_df = all_data["train"][all_data["train"]["dataset"] == dataset_name]
                train_dfs.append(dataset_df.sample(frac=ratio, random_state=config.seed))
            train_df = pd.concat(train_dfs)
        else:
            # Use global ratio
            train_df = all_data["train"].sample(frac=config.train_ratio, random_state=config.seed)
            
        train_df.to_csv(protocol_dir / "train.csv", index=False)
        print(f"Created train.csv with {len(train_df)} samples")

        if "val" in all_data:
            val_df = all_data["val"][~all_data["val"]["filename"].isin(train_df["filename"])]
            val_df.to_csv(protocol_dir / "val.csv", index=False)
            print(f"Created val.csv with {len(val_df)} samples")
    
    if "test" in all_data:
        all_data["test"].to_csv(protocol_dir / "test.csv", index=False)
        print(f"Created test.csv with {len(all_data['test'])} samples")

def create_protocol_splits(dataset_paths, config):
    """Create CSV files for all protocols in parallel"""
    protocols = ["protocol_1", "protocol_2", "protocol_3", "protocol_4"]
    
    # Sử dụng multiprocessing để xử lý song song các protocol
    with mp.Pool(min(len(protocols), mp.cpu_count())) as pool:
        pool.map(
            partial(create_protocol_data, dataset_paths=dataset_paths, config=config),
            protocols
        )

class FASDataset(Dataset):
    def __init__(self, protocol_csv, config, transform=None):
        self.config = config
        self.transform = transform
        self.data = pd.read_csv(protocol_csv)
        
        # Cache image paths và bbox
        self.cached_paths = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            if config.is_kaggle:
                dataset_dir = config.data_dir / f"{row['dataset']}-face-anti-spoofing-dataset"
            else:
                dataset_dir = config.data_dir / f"{row['dataset']}_dataset"
                
            img_dir = dataset_dir / row["folder"]
            img_path = next(img_dir.glob(f"{row['filename']}.*"))
            bbox_path = img_path.parent / f"{img_path.stem}_BB.txt"
            
            with open(bbox_path) as f:
                x, y, w, h = map(int, f.read().strip().split()[:4])
                
            self.cached_paths.append({
                'img_path': img_path,
                'bbox': (x, y, w, h)
            })
            
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cached = self.cached_paths[idx]
        
        # Load and process image
        image = cv2.imread(str(cached['img_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        orig_h, orig_w = image.shape[:2]
        
        # Get cached bbox
        x, y, w, h = cached['bbox']
        
        # Process image efficiently
        scale = self.config.face_det_size / max(orig_h, orig_w)
        if scale < 1:
            # Resize image và bbox một lần
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            image = cv2.resize(image, (new_w, new_h))
            x, y, w, h = [int(v * scale) for v in (x, y, w, h)]

        # Crop & resize 
        if (y >= 0 and y + h <= image.shape[0] and
            x >= 0 and x + w <= image.shape[1]):
            image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))
        
        # Create depth map một lần
        depth_map = torch.zeros((1, self.config.depth_map_size, self.config.depth_map_size))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensors
        label = torch.tensor(row["label"], dtype=torch.long)
        domain = torch.tensor(self.config.dataset_names.index(row["dataset"]), dtype=torch.long)

        return image, depth_map, label, domain

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_dataloaders(config):
    """Create data loaders with prefetching"""
    protocol_dir = config.protocol_dir / config.protocol
    
    train_dataset = FASDataset(
        protocol_dir / "train.csv",
        config,
        transform=get_transforms("train", config)
    )
    val_dataset = FASDataset(
        protocol_dir / "val.csv", 
        config,
        transform=get_transforms("val", config)
    )
    test_dataset = FASDataset(
        protocol_dir / "test.csv",
        config, 
        transform=get_transforms("test", config)
    )

    return {
        "train": DataLoaderX(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True  # Sử dụng pinned memory cho GPU
        ),
        "val": DataLoaderX(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        "test": DataLoaderX(
            test_dataset,
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    }

def main():
    """Generate protocol splits"""
    config = Config()
    create_protocol_splits(config.dataset_paths, config)

if __name__ == "__main__":
    main()