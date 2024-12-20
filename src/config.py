import os
from pathlib import Path

class Config:
    """Configuration class to handle different environments"""
    def __init__(self):
        # Determine environment
        self.is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        
        if self.is_kaggle:
            self.root_dir = Path('/kaggle/working/SSAN_Enhance')
            self.data_dir = Path('/kaggle/input')
            self.dataset_paths = {
                "CelebA_Spoof": self.data_dir / "celeba-spoof-face-anti-spoofing-dataset",
                "CATI_FAS": self.data_dir / "cati-fas-face-anti-spoofing-dataset",
                "LCC_FASD": self.data_dir / "lcc-fasd-face-anti-spoofing-dataset",
                "NUAAA": self.data_dir / "nuaaa-face-anti-spoofing-dataset",
                "Zalo_AIC": self.data_dir / "zalo-aic-face-anti-spoofing-dataset"
            }
        else:
            self.root_dir = Path('.')
            self.data_dir = self.root_dir / "data"
            self.dataset_paths = {
                "CelebA_Spoof": self.data_dir / "CelebA_Spoof_dataset",
                "CATI_FAS": self.data_dir / "CATI_FAS_dataset", 
                "LCC_FASD": self.data_dir / "LCC_FASD_dataset",
                "NUAAA": self.data_dir / "NUAAA_dataset", 
                "Zalo_AIC": self.data_dir / "Zalo_AIC_dataset"
            }
        
        self.output_dir = self.root_dir / "output"
        self.protocol_dir = self.data_dir / "protocols"

        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configs
        self.img_size = 256 # Size for model input  
        self.face_det_size = 640 # Size for face detection
        self.depth_map_size = 32
        self.batch_size = 8
        self.num_workers = 4
        self.seed = 42

        # Training configs
        self.dataset_names = None
        self.protocol = None 
        self.train_ratio = None
        self.val_test_ratio = None