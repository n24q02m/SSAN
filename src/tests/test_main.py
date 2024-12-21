import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main import parse_args, set_seed, get_optimizer, get_scheduler, main
from src.model.ssan import SSAN
from src.config import Config

class TestMain:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup common test fixtures"""
        # Mock arguments
        class Args:
            mode = 'train'
            protocol = 'protocol_1'
            checkpoint = None
            epochs = 2
            lr = 0.001
            batch_size = 4
            optimizer = 'adam'
            scheduler = 'step'
            device = 'cpu'
            num_workers = 0
            
        self.args = Args()
        
        # Mock config
        self.config = Config()
        self.config.num_domains = 5
        self.config.batch_size = 4
        self.config.output_dir = Path("output/test_main")
        
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        yield

    def test_parse_args(self):
        """Test argument parsing"""
        with patch('argparse.ArgumentParser.parse_args',
                  return_value=self.args):
            args = parse_args()
            
        assert args.mode == 'train'
        assert args.protocol == 'protocol_1'
        assert args.device == 'cpu'
        assert args.optimizer == 'adam'
        assert args.scheduler == 'step'

    def test_set_seed(self, setup):
        """Test seed setting"""
        set_seed(42)
        
        x1 = torch.randn(3, 32, 32)
        set_seed(42)
        x2 = torch.randn(3, 32, 32)
        
        assert torch.all(x1 == x2)

    def test_get_optimizer(self, setup):
        """Test optimizer creation"""
        model = SSAN(num_domains=self.config.num_domains)
        
        # Test Adam
        self.config.optimizer = 'adam'
        self.config.learning_rate = 0.001
        self.config.weight_decay = 0.0001
        optimizer = get_optimizer(model, self.config)
        assert isinstance(optimizer, torch.optim.Adam)
        
        # Test SGD
        self.config.optimizer = 'sgd'
        self.config.momentum = 0.9
        optimizer = get_optimizer(model, self.config)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_get_scheduler(self, setup):
        """Test scheduler creation"""
        model = SSAN(num_domains=self.config.num_domains)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test StepLR
        self.config.scheduler = 'step'
        self.config.step_size = 10
        self.config.gamma = 0.1
        scheduler = get_scheduler(optimizer, self.config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test CosineAnnealingWarmRestarts
        self.config.scheduler = 'cosine'
        self.config.warmup_epochs = 5
        self.config.min_lr = 1e-6
        scheduler = get_scheduler(optimizer, self.config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_main_train_mode(self):
        """Test main function in training mode""" 
        with patch('src.main.parse_args', return_value=self.args), \
             patch('src.main.get_dataloaders') as mock_get_dataloaders, \
             patch('src.main.Trainer') as mock_trainer, \
             patch('src.main.find_optimal_batch_size', return_value=4), \
             patch('src.main.find_optimal_workers', return_value=2):
            
            # Mock dataloaders with non-empty iterator
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter([
                (torch.randn(4, 3, 256, 256), # Images
                 torch.randn(4, 1, 32, 32),   # Depth maps  
                 torch.randint(0, 2, (4,)),   # Labels
                 torch.randint(0, 5, (4,)))   # Domain labels
            ])
            
            mock_get_dataloaders.return_value = {
                'train': mock_train_loader,
                'val': MagicMock(),
                'test': MagicMock()
            }
            
            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {'accuracy': 0.9, 'auc': 0.95}
            mock_trainer.return_value = mock_trainer_instance
            
            main()
            
            # Verify trainer was initialized and called
            assert mock_trainer.called
            assert mock_trainer_instance.train.called
            
    def test_main_test_mode(self):
        """Test main function in test mode"""
        self.args.mode = 'test'
        self.args.checkpoint = 'model.pth'
        
        with patch('src.main.parse_args', return_value=self.args), \
             patch('src.main.get_dataloaders') as mock_get_dataloaders, \
             patch('src.main.Predictor') as mock_predictor, \
             patch('src.main.find_optimal_batch_size', return_value=4), \
             patch('src.main.find_optimal_workers', return_value=2):

            # Mock dataloaders with non-empty iterator  
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter([
                (torch.randn(4, 3, 256, 256),
                 torch.randn(4, 1, 32, 32),
                 torch.randint(0, 2, (4,)),
                 torch.randint(0, 5, (4,)))
            ])

            mock_get_dataloaders.return_value = {
                'train': mock_train_loader,
                'val': MagicMock(), 
                'test': MagicMock()
            }
            
            # Mock predictor
            mock_predictor_instance = MagicMock()
            mock_predictor_instance.predict.return_value = {
                'predictions': [0, 1, 0],
                'probabilities': [0.1, 0.9, 0.2]
            }
            mock_predictor.from_checkpoint.return_value = mock_predictor_instance
            
            main()
            
            # Verify predictor was initialized and called
            assert mock_predictor.from_checkpoint.called
            assert mock_predictor_instance.predict.called

    def test_main_error_handling(self, setup):
        """Test error handling in main"""
        self.args.mode = 'test'
        self.args.checkpoint = None
        
        with patch('src.main.parse_args', return_value=self.args), \
            patch('src.main.get_dataloaders') as mock_get_dataloaders, \
            patch('src.main.find_optimal_batch_size', return_value=4), \
            patch('src.main.find_optimal_workers', return_value=2):
            
            # Mock dataloaders with non-empty iterator
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter([
                (torch.randn(4, 3, 256, 256),  # Images
                torch.randn(4, 1, 32, 32),    # Depth maps
                torch.randint(0, 2, (4,)),    # Labels 
                torch.randint(0, 5, (4,)))    # Domain labels
            ])
            
            mock_get_dataloaders.return_value = {
                'train': mock_train_loader,
                'val': MagicMock(),
                'test': MagicMock()
            }
            
            with pytest.raises(ValueError) as excinfo:
                main()
            assert "Checkpoint path required for test mode" in str(excinfo.value)

    @pytest.fixture(autouse=True) 
    def cleanup(self, setup):
        """Cleanup after each test"""
        yield
        import shutil
        shutil.rmtree(self.config.output_dir, ignore_errors=True)

if __name__ == '__main__':
    pytest.main([__file__])