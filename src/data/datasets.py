import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

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

class FASDataset(Dataset):
    def __init__(self, data_dirs, config, transform=None, mode='train'):
        self.data_dirs = data_dirs
        self.config = config
        self.transform = transform
        self.mode = mode
        self.data = []

        if mode == 'train' or mode == 'val':
            for data_dir in data_dirs:
                dataset_name = Path(data_dir).name.replace("_dataset", "")
                live_dir = os.path.join(data_dir, "live")
                spoof_dir = os.path.join(data_dir, "spoof")

                # Gather images and their bbox files
                for img_path in glob.glob(os.path.join(live_dir, "*.jpg")) + \
                              glob.glob(os.path.join(live_dir, "*.png")):
                    bbox_path = img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    if os.path.exists(bbox_path):
                        self.data.append((img_path, 1, dataset_name, bbox_path))

                for img_path in glob.glob(os.path.join(spoof_dir, "*.jpg")) + \
                              glob.glob(os.path.join(spoof_dir, "*.png")):
                    bbox_path = img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    if os.path.exists(bbox_path):
                        self.data.append((img_path, 0, dataset_name, bbox_path))

            if mode == 'train':
                train_data, _ = train_test_split(self.data, test_size=1-config.train_ratio, 
                                               random_state=config.seed)
                self.data = train_data
            else:
                _, val_data = train_test_split(self.data, test_size=1-config.train_ratio,
                                             random_state=config.seed)
                self.data = val_data
        else:
            # Test mode
            data_dir = data_dirs[0]
            dataset_name = Path(data_dir).name.replace("_dataset", "")
            
            for folder in ['live', 'spoof']:
                folder_dir = os.path.join(data_dir, folder)
                label = 1 if folder == 'live' else 0
                
                for img_path in glob.glob(os.path.join(folder_dir, "*.jpg")) + \
                              glob.glob(os.path.join(folder_dir, "*.png")):
                    bbox_path = img_path.replace(".jpg", "_BB.txt").replace(".png", "_BB.txt")
                    if os.path.exists(bbox_path):
                        self.data.append((img_path, label, dataset_name, bbox_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, dataset_name, bbox_path = self.data[idx]
        
        # Read image and get original size
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Read bbox
        with open(bbox_path, "r") as f:
            bbox_info = f.readline().strip().split()
            x, y, w, h = map(int, bbox_info[:4])

        # Resize image first like in face detection
        scale = self.config.face_det_size / max(orig_h, orig_w)
        if scale < 1:
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            # Scale bbox
            x = int(x * scale)
            y = int(y * scale)
            w = int(w * scale) 
            h = int(h * scale)
        
        # Crop using scaled bbox
        if (y >= 0 and y + h <= image.shape[0] and 
            x >= 0 and x + w <= image.shape[1]):
            image = image[y:y+h, x:x+w]

        # Final resize to model input size
        image = cv2.resize(image, (self.config.img_size, self.config.img_size))

        # Create depth map
        depth_map = np.zeros((self.config.depth_map_size, self.config.depth_map_size))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(self.config.dataset_names.index(dataset_name), dtype=torch.long)

        return image, depth_map, label, domain

def split_data(dataset_paths, config):
    """Split data according to protocol"""
    if config.protocol == "protocol_1":
        train_dirs = [dataset_paths["CelebA_Spoof"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
        config.train_ratio = 0.9

    elif config.protocol == "protocol_2":
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["NUAAA"]]
        val_dirs = [dataset_paths["LCC_FASD"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
        config.train_ratio = 0.8

    elif config.protocol == "protocol_3":
        train_dirs = [dataset_paths["NUAAA"], dataset_paths["Zalo_AIC"]]
        val_dirs = [dataset_paths["Zalo_AIC"]]
        test_dirs = [dataset_paths["CATI_FAS"]]
        config.train_ratio = 0.9

    elif config.protocol == "protocol_4":
        train_dirs = [dataset_paths["CelebA_Spoof"], dataset_paths["NUAAA"], 
                     dataset_paths["LCC_FASD"]]
        val_dirs = [dataset_paths["CelebA_Spoof"]]
        test_dirs = [dataset_paths["Zalo_AIC"]]
        config.train_ratio = 0.8

    else:
        raise ValueError(f"Invalid protocol: {config.protocol}")

    return train_dirs, val_dirs, test_dirs

def get_dataloaders(train_dirs, val_dirs, test_dirs, config):
    """Create data loaders"""
    train_dataset = FASDataset(train_dirs, config, transform=get_transforms("train", config), 
                              mode='train')
    val_dataset = FASDataset(val_dirs, config, transform=get_transforms("val", config), 
                            mode='val')
    test_dataset = FASDataset(test_dirs, config, transform=get_transforms("test", config), 
                             mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                          num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=config.num_workers)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

def prepare_data(config):
    """Main function to prepare data"""
    config.dataset_names = [p.name.replace("_dataset","") 
                          for p in Path(config.data_dir).glob("*_dataset")]
                          
    train_dirs, val_dirs, test_dirs = split_data(config.dataset_paths, config)
    dataloaders = get_dataloaders(train_dirs, val_dirs, test_dirs, config)
    return dataloaders

def main():
    """Test the implementation"""
    config = Config()
    config.protocol = "protocol_1"
    
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