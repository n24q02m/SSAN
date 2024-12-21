import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import matplotlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.runner.trainer import Trainer
from src.model.ssan import SSAN
from src.model.losses import ClassificationLoss, ContrastiveLoss, DomainAdversarialLoss

matplotlib.use('Agg')

class TestTrainer:
    @pytest.fixture
    def setup(self):
        """Setup common test fixtures"""
        self.batch_size = 4
        self.img_size = 256
        self.num_domains = 5

        # Mock data
        self.sample_batch = {
            'images': torch.randn(self.batch_size, 3, self.img_size, self.img_size),
            'depth_maps': torch.randn(self.batch_size, 1, 32, 32),
            'labels': torch.randint(0, 2, (self.batch_size,)),
            'domains': torch.randint(0, self.num_domains, (self.batch_size,))
        }
        
        # Mock config
        class Config:
            def __init__(self):
                self.num_epochs = 2
                self.patience = 3
                self.batch_size = 4
                self.output_dir = "tests/output"
                self.run_name = "test_run"
                self.lambda_adv = 0.1
                self.lambda_contrast = 0.1
                
        self.config = Config()
        
        # Model and optimizer
        self.model = SSAN(num_domains=self.num_domains)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        
        # Loss functions
        self.criterion = {
            'cls': ClassificationLoss(),
            'domain': DomainAdversarialLoss(),
            'contrast': ContrastiveLoss()
        }

        # Create mock dataloaders
        self.train_loader = self._create_mock_dataloader() 
        self.val_loader = self._create_mock_dataloader()
        self.test_loader = self._create_mock_dataloader()

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader, 
            test_loader=self.test_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            config=self.config,
            device='cpu'
        )

    def _create_mock_dataloader(self):
        """Create a mock dataloader with balanced labels"""
        # Create balanced dataset with both positive and negative samples 
        dataset = []
        for i in range(8):  # 8 samples total
            dataset.append((
                torch.randn(3, self.img_size, self.img_size),  # Image
                torch.randn(1, 32, 32),  # Depth map
                torch.tensor(i % 2).long(),  # Alternate between 0 and 1 labels
                torch.randint(0, self.num_domains, ()).long()  # Random domain - Changed to create scalar
            ))
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def test_init(self, setup):
        """Test trainer initialization"""
        assert isinstance(self.trainer.model, SSAN)
        assert self.trainer.lambda_val == 0.0
        assert self.trainer.current_epoch == 0
        assert self.trainer.device == 'cpu'

    def test_train_step(self, setup):
        """Test single training step"""
        batch = next(iter(self.train_loader))
        metrics = self.trainer._train_step(batch)
        
        # Check metrics
        assert 'total' in metrics
        assert 'cls_loss' in metrics
        assert 'domain_loss' in metrics
        assert 'contrast_loss' in metrics
        assert 'accuracy' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_evaluate(self, setup):
        """Test evaluation with CSV logging"""
        metrics = self.trainer.evaluate(self.val_loader, mode='val')
        
        # Kiểm tra metrics cơ bản
        required_metrics = ['auc', 'accuracy', 'tpr@fpr=0.01', 'fpr@tpr=0.99']
        assert all(k in metrics for k in required_metrics)
        assert all(isinstance(v, float) for v in metrics.values())
        
        # Kiểm tra log file
        log_file = self.trainer.log_dir / 'training.log'
        assert log_file.exists()
        
        # Kiểm tra CSV file
        val_csv = self.trainer.csv_dir / 'val_metrics.csv'
        if val_csv.exists():  # CSV might not exist if evaluate is called directly
            df = pd.read_csv(val_csv)
            assert len(df) > 0
            for metric in required_metrics:
                assert metric in df.columns

    def test_train_epoch(self, setup):
        """Test training for one epoch with CSV logging"""
        # Train một epoch
        metrics = self.trainer.train_epoch(epoch=0)
        
        # Kiểm tra metrics cơ bản
        required_metrics = ['loss', 'cls_loss', 'domain_loss', 'contrast_loss']
        assert all(k in metrics for k in required_metrics)
        assert all(isinstance(v, float) for v in metrics.values())
        
        # Kiểm tra file CSV được tạo
        train_csv = self.trainer.csv_dir / 'train_metrics.csv'
        assert train_csv.exists()
        df = pd.read_csv(train_csv)
        assert len(df) > 0
        for metric in required_metrics:
            assert metric in df.columns

    def test_checkpoint_saving(self, setup):
        """Test checkpoint saving with metrics"""
        # Train một epoch và lưu
        train_metrics = self.trainer.train_epoch(0)
        val_metrics = self.trainer.evaluate(self.val_loader)
        self.trainer._save_checkpoints(0, val_metrics)

        # Kiểm tra checkpoint file
        latest_ckpt = self.trainer.ckpt_dir / 'latest.pth'
        assert latest_ckpt.exists()

        # Load và verify checkpoint
        ckpt = torch.load(latest_ckpt, weights_only=False)
        assert 'epoch' in ckpt
        assert 'model_state_dict' in ckpt
        assert 'optimizer_state_dict' in ckpt
        assert 'scheduler_state_dict' in ckpt
        assert 'metrics' in ckpt
        
        # Verify metrics trong checkpoint
        saved_metrics = ckpt['metrics']
        required_metrics = ['auc', 'accuracy', 'tpr@fpr=0.01', 'fpr@tpr=0.99']
        assert all(k in saved_metrics for k in required_metrics)

    def test_save_metrics_to_csv(self, setup):
        """Test saving metrics to CSV file"""
        # Tạo metrics mẫu
        test_metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'auc': 0.85,
            'tpr@fpr=0.01': 0.7
        }
        
        # Lưu metrics
        self.trainer._save_metrics_to_csv(test_metrics, 'train', epoch=0)
        
        # Kiểm tra file CSV
        csv_path = self.trainer.csv_dir / 'train_metrics.csv'
        assert csv_path.exists()
        
        # Đọc và verify nội dung
        df = pd.read_csv(csv_path)
        for key, value in test_metrics.items():
            assert key in df.columns
            assert np.isclose(df[key].iloc[0], value)

    def test_plot_training_curves(self, setup):
        """Test plotting of training curves"""
        # Tạo dữ liệu mẫu cho train và val
        train_metrics = {'loss': 0.5, 'accuracy': 0.8, 'auc': 0.85}
        val_metrics = {'loss': 0.4, 'accuracy': 0.85, 'auc': 0.88}
        
        # Lưu metrics
        self.trainer._save_metrics_to_csv(train_metrics, 'train', epoch=0)
        self.trainer._save_metrics_to_csv(val_metrics, 'val', epoch=0)
        
        # Vẽ biểu đồ
        self.trainer._plot_training_curves()
        
        # Kiểm tra files được tạo
        expected_plots = ['loss_curve.png', 'accuracy_curve.png', 'auc_curve.png']
        for plot in expected_plots:
            assert (self.trainer.plot_dir / plot).exists()

    def test_directory_setup(self, setup):
        """Test directory creation and structure"""
        required_dirs = [
            self.trainer.ckpt_dir,
            self.trainer.log_dir,
            self.trainer.csv_dir,
            self.trainer.plot_dir
        ]
        
        for dir_path in required_dirs:
            assert dir_path.exists()
            assert dir_path.is_dir()

    def test_full_training_loop(self, setup):
        """Test complete training loop with all components"""
        # Chạy một epoch train đầy đủ
        self.trainer.train()
        
        # Verify output structure
        assert (self.trainer.csv_dir / 'train_metrics.csv').exists()
        assert (self.trainer.csv_dir / 'val_metrics.csv').exists()
        assert (self.trainer.plot_dir / 'loss_curve.png').exists()
        assert (self.trainer.plot_dir / 'accuracy_curve.png').exists()
        assert (self.trainer.plot_dir / 'auc_curve.png').exists()
        assert (self.trainer.ckpt_dir / 'latest.pth').exists()
        assert (self.trainer.log_dir / 'training.log').exists()

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after each test"""
        yield
        import shutil
        shutil.rmtree(self.config.output_dir, ignore_errors=True)

if __name__ == '__main__':
    pytest.main([__file__])