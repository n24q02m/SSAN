import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from src.runner.trainer import Trainer
from src.model.ssan import SSAN
from src.model.losses import ClassificationLoss, ContrastiveLoss, DomainAdversarialLoss

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
        """Create a mock dataloader"""
        # Create single batch of data (not nested)
        dataset = [(
            torch.randn(3, self.img_size, self.img_size),  # Change shape to [C,H,W]
            torch.randn(1, 32, 32),
            torch.tensor(0).long(),  # Single label
            torch.tensor(0).long()   # Single domain
        )] * 8  # 8 samples 
        
        # Create proper BatchSampler
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
        """Test evaluation"""
        metrics = self.trainer.evaluate(self.val_loader, mode='val')
        
        # Check evaluation metrics
        required_metrics = ['auc', 'accuracy', 'tpr@fpr=0.01', 'fpr@tpr=0.99']
        assert all(k in metrics for k in required_metrics)
        assert all(isinstance(v, float) for v in metrics.values())

    def test_train_epoch(self, setup):
        """Test training for one epoch"""
        metrics = self.trainer.train_epoch(epoch=0)
        
        # Check training metrics
        required_metrics = ['loss', 'cls_loss', 'domain_loss', 'contrast_loss']
        assert all(k in metrics for k in required_metrics)
        assert all(isinstance(v, float) for v in metrics.values())

    def test_checkpoint_saving(self, setup):
        """Test checkpoint saving and loading"""
        # Train for 1 epoch and save
        self.trainer.train_epoch(0)
        val_metrics = self.trainer.evaluate(self.val_loader)
        self.trainer._save_checkpoints(0, val_metrics)

        # Check if checkpoint files exist
        ckpt_path = Path(self.config.output_dir) / 'checkpoints' / self.config.run_name
        assert (ckpt_path / 'latest.pth').exists()

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after each test"""
        yield
        import shutil
        shutil.rmtree(self.config.output_dir, ignore_errors=True)

if __name__ == '__main__':
    pytest.main([__file__])