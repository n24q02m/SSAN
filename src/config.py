import os
from pathlib import Path
import torch


class Config:
    """Configuration class to handle different environments"""

    def __init__(self):
        # Determine environment
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

        if self.is_kaggle:
            self.root_dir = Path("/kaggle/working/SSAN")
            self.data_dir = Path("/kaggle/input")
            self.dataset_paths = {
                "CelebA_Spoof": self.data_dir
                / "celeba-spoof-face-anti-spoofing-dataset",
                "CATI_FAS": self.data_dir / "cati-fas-face-anti-spoofing-dataset",
                "LCC_FASD": self.data_dir / "lcc-fasd-face-anti-spoofing-dataset",
                "NUAAA": self.data_dir / "nuaaa-face-anti-spoofing-dataset",
                "Zalo_AIC": self.data_dir / "zalo-aic-face-anti-spoofing-dataset",
            }
            self.protocol_dir = self.root_dir / "data" / "protocols"
        else:
            self.root_dir = Path(".")
            self.data_dir = self.root_dir / "data"
            self.dataset_paths = {
                "CelebA_Spoof": self.data_dir / "CelebA_Spoof_dataset",
                "CATI_FAS": self.data_dir / "CATI_FAS_dataset",
                "LCC_FASD": self.data_dir / "LCC_FASD_dataset",
                "NUAAA": self.data_dir / "NUAAA_dataset",
                "Zalo_AIC": self.data_dir / "Zalo_AIC_dataset",
            }
            self.protocol_dir = self.data_dir / "protocols"

        self.output_dir = self.root_dir / "output"

        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_dir.mkdir(parents=True, exist_ok=True)

        # Model configs
        self.img_size = 256  # Size for model input
        self.face_det_size = 640  # Size for face detection
        self.depth_map_size = 32
        self.batch_size = 8
        self.num_workers = 4
        self.seed = 42

        # Training configs
        self.dataset_names = None
        self.protocol = None
        self.ratios = None

        # Training hyperparameters
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 5e-5
        self.momentum = 0.9
        self.optimizer = "adam"  # 'adam' or 'sgd'
        self.scheduler = "step"  # 'step' or 'cosine'
        self.step_size = 30  # for StepLR
        self.gamma = 0.1  # for StepLR
        self.warmup_epochs = 5  # for CosineAnnealingWarmRestarts
        self.min_lr = 1e-6  # for CosineAnnealingWarmRestarts

        # Model configs
        self.num_domains = 5
        self.ada_blocks = 2
        self.dropout = 0.1

        # Loss weights
        self.lambda_adv = 0.1  # Weight for domain adversarial loss
        self.lambda_contrast = 0.1  # Weight for contrastive loss

        # Early stopping
        self.patience = 10
        self.early_stopping = True

        # Debug config
        self.fraction = 1.0

        # Device config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
