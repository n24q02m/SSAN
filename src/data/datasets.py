import os
import sys
import cv2
import torch
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Invalid SOS parameters for sequential JPEG")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mode="train", config=None):
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
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
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
                A.Resize(config.img_size, config.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def split_data(dataset_paths, config):
    """Split data according to protocol"""
    if config.protocol == "protocol_1":
        # Single Large-Scale Dataset (CelebA-Spoof)
        train_dirs = [dataset_paths["CelebA_Spoof"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["CelebA_Spoof"]]
        config.ratios = {"CelebA_Spoof": {"train": 0.6, "val": 0.2, "test": 0.2}}

    elif config.protocol == "protocol_2":
        # Multi-Scale Training (CelebA-Spoof + CATI-FAS)
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        val_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        test_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["CATI_FAS"]]
        config.ratios = {
            "CelebA_Spoof": {
                "train": 0.3,  # 30% train
                "val": 0.1,  # 10% val
                "test": 0.1,  # 10% test
            },
            "CATI_FAS": {
                "train": 0.6,  # 60% train
                "val": 0.2,  # 20% val
                "test": 0.2,  # 20% test
            },
        }

    elif config.protocol == "protocol_3":
        # Cross-Dataset Evaluation
        train_dirs = [dataset_paths["CATI_FAS"], dataset_paths["Zalo_AIC"]]
        val_dirs = [dataset_paths["CATI_FAS"], dataset_paths["Zalo_AIC"]]
        test_dirs = [dataset_paths["LCC_FASD"], dataset_paths["NUAAA"]]
        config.ratios = {
            "CATI_FAS": {
                "train": 0.8,  # 80% train
                "val": 0.2,  # 20% val
            },
            "Zalo_AIC": {
                "train": 0.6,  # 60% train
                "val": 0.4,  # 40% val
            },
            "LCC_FASD": {"test": 0.3},  # 30% test
            "NUAAA": {"test": 0.3},  # 30% test
        }

    elif config.protocol == "protocol_4":
        # Domain Generalization
        train_dirs = [
            dataset_paths["CATI_FAS"],
            dataset_paths["LCC_FASD"],
            dataset_paths["NUAAA"],
            dataset_paths["Zalo_AIC"],
        ]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["CelebA_Spoof"]]
        config.ratios = {
            "CATI_FAS": {"train": 1.0},  # 100% train
            "LCC_FASD": {"train": 1.0},  # 100% train
            "NUAAA": {"train": 1.0},  # 100% train
            "Zalo_AIC": {"train": 1.0},  # 100% train
            "CelebA_Spoof": {"val": 0.025, "test": 0.025},  # 2.5% val  # 2.5% test
        }

    else:
        raise ValueError(f"Invalid protocol: {config.protocol}")

    return train_dirs, val_dirs, test_dirs


def create_protocol_data(protocol, dataset_paths, config):
    print(f"\nCreating {protocol} splits...")
    config.protocol = protocol
    train_dirs, val_dirs, test_dirs = split_data(dataset_paths, config)

    protocol_dir = config.protocol_dir / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train.csv", "val.csv", "test.csv"]:
        file_path = protocol_dir / split
        if file_path.exists():
            file_path.unlink()

    # Initialize dictionaries for each split
    train_data = []
    val_data = []
    test_data = []

    # Process each split separately to avoid DataFrame concurrency issues
    for data_dir in train_dirs:
        dataset_name = Path(data_dir).name
        if config.is_kaggle:
            dataset_name = dataset_name.replace("-face-anti-spoofing-dataset", "")
        else:
            dataset_name = dataset_name.replace("_dataset", "")

        for folder, label in [("live", 1), ("spoof", 0)]:
            folder_dir = Path(data_dir) / folder
            if not folder_dir.exists():
                continue

            for img_path in folder_dir.glob("*.[jp][pn][g]"):
                if (img_path.parent / f"{img_path.stem}_BB.txt").exists():
                    train_data.append(
                        {
                            "dataset": dataset_name,
                            "filename": img_path.stem,
                            "label": label,
                            "folder": folder,
                        }
                    )

    # Same for validation dirs
    for data_dir in val_dirs:
        dataset_name = Path(data_dir).name
        if config.is_kaggle:
            dataset_name = dataset_name.replace("-face-anti-spoofing-dataset", "")
        else:
            dataset_name = dataset_name.replace("_dataset", "")

        for folder, label in [("live", 1), ("spoof", 0)]:
            folder_dir = Path(data_dir) / folder
            if not folder_dir.exists():
                continue

            for img_path in folder_dir.glob("*.[jp][pn][g]"):
                if (img_path.parent / f"{img_path.stem}_BB.txt").exists():
                    val_data.append(
                        {
                            "dataset": dataset_name,
                            "filename": img_path.stem,
                            "label": label,
                            "folder": folder,
                        }
                    )

    # And test dirs
    for data_dir in test_dirs:
        dataset_name = Path(data_dir).name
        if config.is_kaggle:
            dataset_name = dataset_name.replace("-face-anti-spoofing-dataset", "")
        else:
            dataset_name = dataset_name.replace("_dataset", "")

        for folder, label in [("live", 1), ("spoof", 0)]:
            folder_dir = Path(data_dir) / folder
            if not folder_dir.exists():
                continue

            for img_path in folder_dir.glob("*.[jp][pn][g]"):
                if (img_path.parent / f"{img_path.stem}_BB.txt").exists():
                    test_data.append(
                        {
                            "dataset": dataset_name,
                            "filename": img_path.stem,
                            "label": label,
                            "folder": folder,
                        }
                    )

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Apply dataset ratios if defined
    if config.ratios:
        for dataset_name, ratios in config.ratios.items():
            # Kiểm tra ratios[split] thay vì split in ratios
            if "train" in ratios.keys() and not train_df.empty:
                dataset_indices = train_df[train_df["dataset"] == dataset_name].index
                if len(dataset_indices) > 0:
                    size = int(len(dataset_indices) * ratios["train"])
                    keep_indices = np.random.choice(
                        dataset_indices, size=size, replace=False
                    )
                    train_df = pd.concat(
                        [
                            train_df[train_df["dataset"] != dataset_name],
                            train_df.loc[keep_indices],
                        ]
                    )

            if "val" in ratios.keys() and not val_df.empty:
                dataset_indices = val_df[val_df["dataset"] == dataset_name].index
                if len(dataset_indices) > 0:
                    size = int(len(dataset_indices) * ratios["val"])
                    keep_indices = np.random.choice(
                        dataset_indices, size=size, replace=False
                    )
                    val_df = pd.concat(
                        [
                            val_df[val_df["dataset"] != dataset_name],
                            val_df.loc[keep_indices],
                        ]
                    )

            if "test" in ratios.keys() and not test_df.empty:
                dataset_indices = test_df[test_df["dataset"] == dataset_name].index
                if len(dataset_indices) > 0:
                    size = int(len(dataset_indices) * ratios["test"])
                    keep_indices = np.random.choice(
                        dataset_indices, size=size, replace=False
                    )
                    test_df = pd.concat(
                        [
                            test_df[test_df["dataset"] != dataset_name],
                            test_df.loc[keep_indices],
                        ]
                    )

    # Save splits
    if not train_df.empty:
        train_df.to_csv(protocol_dir / "train.csv", index=False)
        print(f"Created train.csv with {len(train_df)} samples")

    if not val_df.empty:
        val_df.to_csv(protocol_dir / "val.csv", index=False)
        print(f"Created val.csv with {len(val_df)} samples")

    if not test_df.empty:
        test_df.to_csv(protocol_dir / "test.csv", index=False)
        print(f"Created test.csv with {len(test_df)} samples")


def create_protocol_splits(dataset_paths, config):
    """Create CSV files for all protocols in parallel using ProcessPoolExecutor"""
    protocols = ["protocol_1", "protocol_2", "protocol_3", "protocol_4"]

    # Use ProcessPoolExecutor for parallel protocol creation
    with ProcessPoolExecutor(
        max_workers=min(len(protocols), mp.cpu_count())
    ) as executor:
        list(
            executor.map(
                partial(
                    create_protocol_data, dataset_paths=dataset_paths, config=config
                ),
                protocols,
            )
        )


def format_filename(filename):
    """Convert filename to 6 digits format"""
    try:
        # Convert to integer and back to string with leading zeros
        return f"{int(filename):06d}"
    except ValueError:
        return filename


def parallel_process_image(args):
    """Process single image in parallel"""
    idx, row, config, dataset_map = args
    try:
        dataset_dir = config.data_dir / dataset_map[row["dataset"]]
        img_dir = dataset_dir / row["folder"]

        filename = format_filename(row["filename"])

        # Try both extensions
        img_patterns = [f"{filename}.jpg", f"{filename}.png"]
        img_path = None
        for pattern in img_patterns:
            potential_path = img_dir / pattern
            if potential_path.exists():
                img_path = potential_path
                break

        if img_path is None:
            return None

        bbox_path = img_path.parent / f"{filename}_BB.txt"
        if not bbox_path.exists():
            return None

        with open(bbox_path) as f:
            x, y, w, h = map(int, f.read().strip().split()[:4])

        return {"img_path": img_path, "bbox": (x, y, w, h)}

    except Exception as e:
        return None


class FASDataset(Dataset):
    def __init__(self, protocol_csv, config, transform=None):
        self.config = config
        self.transform = transform

        # Dataset name mapping
        self.dataset_map = {
            "CelebA_Spoof": (
                "CelebA_Spoof_dataset"
                if not config.is_kaggle
                else "celeba-spoof-face-anti-spoofing-dataset"
            ),
            "CATI_FAS": (
                "CATI_FAS_dataset"
                if not config.is_kaggle
                else "cati-fas-face-anti-spoofing-dataset"
            ),
            "LCC_FASD": (
                "LCC_FASD_dataset"
                if not config.is_kaggle
                else "lcc-fasd-face-anti-spoofing-dataset"
            ),
            "NUAAA": (
                "NUAAA_dataset"
                if not config.is_kaggle
                else "nuaaa-face-anti-spoofing-dataset"
            ),
            "Zalo_AIC": (
                "Zalo_AIC_dataset"
                if not config.is_kaggle
                else "zalo-aic-face-anti-spoofing-dataset"
            ),
        }

        # Read and sample data based on fraction
        df = pd.read_csv(protocol_csv)
        if config.fraction < 1.0:
            # Ensure balanced sampling between classes
            sampled_dfs = []
            for label in df["label"].unique():
                label_df = df[df["label"] == label]
                n_samples = int(len(label_df) * config.fraction)
                sampled_df = label_df.sample(n=n_samples, random_state=config.seed)
                sampled_dfs.append(sampled_df)
            self.data = pd.concat(sampled_dfs).reset_index(drop=True)
        else:
            self.data = df

        # Cache image paths and bboxes
        self.cached_paths = []
        self.data["filename"] = self.data["filename"].apply(format_filename)

        # Process images in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for idx in range(len(self.data)):
                row = self.data.iloc[idx]
                futures.append(
                    executor.submit(
                        parallel_process_image, (idx, row, config, self.dataset_map)
                    )
                )

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing images") as pbar:
                self.cached_paths = []
                for future in futures:
                    result = future.result()
                    if result is not None:
                        self.cached_paths.append(result)
                    pbar.update(1)

        if not self.cached_paths:
            raise RuntimeError("No valid images found in dataset")

        print(f"Successfully loaded {len(self.cached_paths)} valid images")

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.cached_paths)

    def __getitem__(self, idx):
        # Convert idx to integer
        idx = int(idx)  # Add this line

        # Make sure idx is within bounds
        if idx >= len(self.cached_paths):
            raise IndexError("Index out of bounds")

        row = self.data.iloc[idx]
        cached = self.cached_paths[idx]

        # Load and process image
        image = cv2.imread(str(cached["img_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Get cached bbox
        x, y, w, h = cached["bbox"]

        # Process image efficiently
        scale = self.config.face_det_size / max(orig_h, orig_w)
        if scale < 1:
            # Resize image và bbox một lần
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            image = cv2.resize(image, (new_w, new_h))
            x, y, w, h = [int(v * scale) for v in (x, y, w, h)]

        # Crop & resize
        if y >= 0 and y + h <= image.shape[0] and x >= 0 and x + w <= image.shape[1]:
            image = image[y : y + h, x : x + w]
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))

        # Create depth map một lần
        depth_map = torch.zeros(
            (1, self.config.depth_map_size, self.config.depth_map_size)
        )

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensors
        label = torch.tensor(row["label"], dtype=torch.long)
        domain = torch.tensor(
            self.config.dataset_names.index(row["dataset"]), dtype=torch.long
        )

        return image, depth_map, label, domain


class DataLoaderX(DataLoader):
    """DataLoader with background generator for faster data loading"""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=mp.cpu_count() * 2)


def get_dataloaders(config):
    """Create data loaders with prefetching"""
    print("Creating dataloaders...")
    protocol_dir = config.protocol_dir / config.protocol

    # Get unique dataset names from protocol CSVs
    dataset_names = []
    print("\nLoading protocol files...")
    for split in tqdm(["train", "val", "test"], desc="Reading CSV files"):
        csv_path = protocol_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dataset_names.extend(df["dataset"].unique())
    config.dataset_names = sorted(list(set(dataset_names)))
    print(f"Found datasets: {config.dataset_names}")

    # If using GPU, workers may not be needed
    if config.device == "cuda":
        config.num_workers = 0  # Disable workers when using GPU
        pin_memory = True
    else:
        pin_memory = False

    print("\nLoading datasets:")
    with tqdm(total=3, desc="Creating datasets") as pbar:
        print("Loading training data...")
        train_dataset = FASDataset(
            protocol_dir / "train.csv",
            config,
            transform=get_transforms("train", config),
        )
        pbar.update(1)

        print("Loading validation data...")
        val_dataset = FASDataset(
            protocol_dir / "val.csv", config, transform=get_transforms("val", config)
        )
        pbar.update(1)

        print("Loading test data...")
        test_dataset = FASDataset(
            protocol_dir / "test.csv", config, transform=get_transforms("test", config)
        )
        pbar.update(1)

    print("\nDataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    print("\nCreating DataLoaders...")
    loaders = {
        "train": DataLoaderX(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(mp.cpu_count(), 8),  # Limit max workers
            pin_memory=pin_memory,
            prefetch_factor=2,
            persistent_workers=True,
        ),
        "val": DataLoaderX(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoaderX(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
    }
    print("DataLoaders created successfully")
    return loaders


def main():
    """Generate protocol splits"""
    config = Config()
    create_protocol_splits(config.dataset_paths, config)


if __name__ == "__main__":
    main()
