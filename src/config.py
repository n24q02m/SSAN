import os

class Config:
    """Configuration class to handle different environments"""
    def __init__(self):
        self.data_dir = "./data"
        self.output_dir = "./output"
        self.dataset_paths = {
            "CelebA_Spoof": f"{self.data_dir}/CelebA_Spoof_dataset",
            "CATI_FAS": f"{self.data_dir}/CATI_FAS_dataset",
            "LCC_FASD": f"{self.data_dir}/LCC_FASD_dataset", 
            "NUAAA": f"{self.data_dir}/NUAAA_dataset",
            "Zalo_AIC": f"{self.data_dir}/Zalo_AIC_dataset"
        }
        
        # Training configs 
        self.img_size = 256 # Size for model input
        self.face_det_size = 640 # Size for face detection
        self.depth_map_size = 32
        self.batch_size = 8
        self.num_workers = 4
        self.seed = 42

        # Make output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # To be set later
        self.dataset_names = None
        self.protocol = None
        self.train_ratio = None