import os
import sys
import shutil
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.ssan import SSAN
from src.runner.optimizers import (
    find_optimal_batch_size,
    find_optimal_workers,
    HyperparameterOptimizer,
)


class TestOptimizers:
    @pytest.fixture
    def setup(self):
        """Setup common test fixtures"""
        self.batch_size = 4
        self.img_size = 256
        self.num_domains = 5

        # Create small mock dataset with balanced labels
        self.dataset = []
        for i in range(16):  # 16 samples total
            self.dataset.append(
                (
                    torch.randn(3, self.img_size, self.img_size),  # Image
                    torch.randn(1, 32, 32),  # Depth map
                    torch.tensor(i % 2).long(),  # Alternate between 0/1 labels
                    torch.randint(0, self.num_domains, ()).long(),  # Domain label
                )
            )

        # Create dataloaders with balanced data
        self.train_loader = DataLoader(
            self.dataset[:12],  # 12 training samples
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.dataset[12:],  # 4 validation samples
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Create model
        self.model = SSAN(num_domains=self.num_domains)

        # Create sample batch
        self.sample_batch = self.dataset[0]

        # Mock config with correct output path
        class Config:
            def __init__(self):
                self.device = "cpu"
                self.num_domains = 5
                self.num_epochs = 2
                self.patience = 3
                self.output_dir = Path("output/test_optimizers")
                self.lambda_adv = 0.1
                self.lambda_contrast = 0.1

        self.config = Config()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def test_find_optimal_batch_size(self, setup):
        """Test optimal batch size finder"""
        optimal_batch = find_optimal_batch_size(
            model=self.model,
            sample_batch=self.sample_batch,
            min_batch=2,
            max_batch=8,
            max_memory_use=0.8,
        )

        # Check output
        assert isinstance(optimal_batch, int)
        assert 2 <= optimal_batch <= 8

        # Test edge cases
        with pytest.raises(RuntimeError):
            find_optimal_batch_size(
                self.model,
                self.sample_batch,
                max_memory_use=1.5,  # Invalid memory ratio
            )

    def test_find_optimal_workers(self, setup):
        """Test optimal worker count finder"""
        self.train_loader = DataLoader(
            self.dataset[:12],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            prefetch_factor=2,
        )

        optimal_workers = find_optimal_workers(self.train_loader, max_workers=4)

        assert isinstance(optimal_workers, int)
        assert 0 <= optimal_workers <= 4

    def test_hyperparameter_optimizer_init(self, setup):
        """Test hyperparameter optimizer initialization"""
        optimizer = HyperparameterOptimizer(
            model_class=SSAN,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            study_name="test_study",
            n_trials=2,
            timeout=60,
            output_dir=self.config.output_dir,
        )

        assert optimizer.model_class == SSAN
        assert optimizer.train_loader == self.train_loader
        assert optimizer.val_loader == self.val_loader
        assert optimizer.config == self.config
        assert optimizer.n_trials == 2
        assert optimizer.timeout == 60
        assert str(optimizer.output_dir) == str(self.config.output_dir)

    def test_hyperparameter_optimization(self, setup):
        """Test full hyperparameter optimization"""
        optimizer = HyperparameterOptimizer(
            model_class=SSAN,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            study_name="test_optimization",
            n_trials=2,  # Small number for testing
            timeout=60,
            output_dir=self.config.output_dir,
        )

        # Run optimization
        best_params = optimizer.optimize()

        # Check results
        assert isinstance(best_params, dict)
        required_params = [
            "learning_rate",
            "weight_decay",
            "lambda_adv",
            "lambda_contrast",
            "optimizer",
            "scheduler",
            "dropout",
        ]
        assert all(p in best_params for p in required_params)

        # Check if results file was created
        results_file = optimizer.output_dir / "test_optimization_results.csv"
        assert results_file.exists()

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after each test"""
        yield
        shutil.rmtree(self.config.output_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])
