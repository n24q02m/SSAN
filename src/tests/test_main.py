import os
import sys
import pytest
import torch
from torch.optim import Adam, SGD, lr_scheduler
import shutil
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
            mode = "train"
            protocol = "protocol_1"
            checkpoint = None
            epochs = 2
            lr = 0.001
            batch_size = 4
            optimizer = "adam"
            scheduler = "step"
            device = "cpu"
            num_workers = 0
            debug_fraction = 1.0
            auto_hp = False
            hp_trials = 2
            hp_timeout = 60

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
        with patch("argparse.ArgumentParser.parse_args", return_value=self.args):
            args = parse_args()

        assert args.mode == "train"
        assert args.protocol == "protocol_1"
        assert args.device == "cpu"
        assert args.optimizer == "adam"
        assert args.scheduler == "step"

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
        self.config.optimizer = "adam"
        self.config.learning_rate = 0.001
        self.config.weight_decay = 0.0001
        optimizer = get_optimizer(model, self.config)
        assert isinstance(optimizer, Adam)

        # Test SGD
        self.config.optimizer = "sgd"
        self.config.momentum = 0.9
        optimizer = get_optimizer(model, self.config)
        assert isinstance(optimizer, SGD)

    def test_get_scheduler(self, setup):
        """Test scheduler creation"""
        model = SSAN(num_domains=self.config.num_domains)
        optimizer = Adam(model.parameters(), lr=0.001)

        # Test StepLR
        self.config.scheduler = "step"
        self.config.step_size = 10
        self.config.gamma = 0.1
        scheduler = get_scheduler(optimizer, self.config)
        assert isinstance(scheduler, lr_scheduler.StepLR)

        # Test CosineAnnealingWarmRestarts
        self.config.scheduler = "cosine"
        self.config.warmup_epochs = 5
        self.config.min_lr = 1e-6
        scheduler = get_scheduler(optimizer, self.config)
        assert isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts)

    def test_auto_hyperparameter_optimization(self, setup):
        """Test automatic hyperparameter optimization"""
        self.args.auto_hp = True
        self.args.hp_trials = 2
        self.args.hp_timeout = 60

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch(
            "src.main.HyperparameterOptimizer"
        ) as mock_hp_optimizer, patch(
            "src.main.Trainer"
        ) as mock_trainer:

            # Mock dataloaders with sample batch
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),
                    )  # Domain labels
                ]
            )

            mock_val_loader = MagicMock()
            mock_test_loader = MagicMock()

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": mock_val_loader,
                "test": mock_test_loader,
            }

            # Mock hyperparameter optimization results
            mock_hp_optimizer_instance = MagicMock()
            mock_hp_optimizer_instance.optimize.return_value = {
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "lambda_adv": 0.1,
                "lambda_contrast": 0.1,
                "optimizer": "adam",
                "scheduler": "step",
                "dropout": 0.2,
            }
            mock_hp_optimizer.return_value = mock_hp_optimizer_instance

            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"accuracy": 0.9, "auc": 0.95}
            mock_trainer.return_value = mock_trainer_instance

            main()

            # Verify hyperparameter optimization was run
            mock_hp_optimizer.assert_called_once()
            mock_hp_optimizer_instance.optimize.assert_called_once()

            # Verify trainer was initialized with optimized parameters
            mock_trainer.assert_called_once()
            _, kwargs = mock_trainer.call_args
            assert kwargs["config"].learning_rate == 0.001
            assert kwargs["config"].dropout == 0.2

            # Verify training was executed
            assert mock_trainer_instance.train.called

    def test_dataset_loading(self, setup):
        """Test dataset loading and processing in main"""
        self.args.mode = "train"
        self.args.protocol = "protocol_1"
        self.args.batch_size = 4  # Set fixed batch size
        self.args.num_workers = 4  # Set fixed workers

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Trainer") as mock_trainer:

            # Mock dataset loading
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),  # Domain labels
                    )
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": MagicMock(),
                "test": MagicMock(),
            }

            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"accuracy": 0.9}
            mock_trainer.return_value = mock_trainer_instance

            main()

            # Verify dataloaders were created with correct config
            assert mock_get_dataloaders.call_count == 1
            config = mock_get_dataloaders.call_args[0][0]

            # Check protocol specific settings
            assert config.protocol == "protocol_1"
            assert hasattr(config, "img_size")
            assert hasattr(config, "depth_map_size")

    def test_debug_dataset_loading(self, setup):
        """Test dataset loading with debug fraction"""
        self.args.mode = "train"
        self.args.debug_fraction = 0.1
        self.args.batch_size = 4  # Set fixed batch size
        self.args.num_workers = 4  # Set fixed workers

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Trainer") as mock_trainer:

            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),
                        torch.randn(4, 1, 32, 32),
                        torch.randint(0, 2, (4,)),
                        torch.randint(0, 5, (4,)),
                    )
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": MagicMock(),
                "test": MagicMock(),
            }

            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"accuracy": 0.9}
            mock_trainer.return_value = mock_trainer_instance

            main()

            assert mock_get_dataloaders.call_count == 1
            config = mock_get_dataloaders.call_args[0][0]
            assert config.debug_fraction == 0.1

    def test_protocol_specific_dataset_loading(self, setup):
        """Test dataset loading for different protocols"""
        protocols = ["protocol_1", "protocol_2", "protocol_3", "protocol_4"]

        for protocol in protocols:
            self.args.mode = "train"
            self.args.protocol = protocol
            self.args.batch_size = 4  # Set fixed batch size
            self.args.num_workers = 4  # Set fixed workers

            with patch("src.main.parse_args", return_value=self.args), patch(
                "src.main.get_dataloaders"
            ) as mock_get_dataloaders, patch("src.main.Trainer") as mock_trainer:

                mock_train_loader = MagicMock()
                mock_train_loader.__iter__.return_value = iter(
                    [
                        (
                            torch.randn(4, 3, 256, 256),
                            torch.randn(4, 1, 32, 32),
                            torch.randint(0, 2, (4,)),
                            torch.randint(0, 5, (4,)),
                        )
                    ]
                )

                mock_get_dataloaders.return_value = {
                    "train": mock_train_loader,
                    "val": MagicMock(),
                    "test": MagicMock(),
                }

                # Mock trainer
                mock_trainer_instance = MagicMock()
                mock_trainer_instance.train.return_value = {"accuracy": 0.9}
                mock_trainer.return_value = mock_trainer_instance

                main()

                assert mock_get_dataloaders.call_count == 1
                config = mock_get_dataloaders.call_args[0][0]
                assert config.protocol == protocol

                mock_get_dataloaders.reset_mock()

    def test_debug_fraction(self, setup):
        """Test debug fraction functionality"""
        self.args.debug_fraction = 0.1  # Set to use 10% data
        self.args.batch_size = 4
        self.args.num_workers = 4

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Trainer") as mock_trainer, patch(
            "src.main.find_optimal_batch_size", return_value=4
        ), patch(
            "src.main.find_optimal_workers", return_value=4
        ):

            # Mock dataloader với sample batch
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),
                    )  # Domain labels
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": MagicMock(),
                "test": MagicMock(),
            }

            mock_trainer_instance = MagicMock()
            mock_trainer.return_value = mock_trainer_instance

            main()

            # Verify debug_fraction was set in config
            mock_get_dataloaders.assert_called_once()
            config = mock_get_dataloaders.call_args[0][0]
            assert config.debug_fraction == 0.1

    def test_main_train_mode(self):
        """Test main function in training mode"""
        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Trainer") as mock_trainer, patch(
            "src.main.find_optimal_batch_size", return_value=4
        ), patch(
            "src.main.find_optimal_workers", return_value=2
        ), patch(
            "src.main.HyperparameterOptimizer", return_value=MagicMock()
        ):

            # Mock dataloaders with non-empty iterator
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),
                    )  # Domain labels
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": MagicMock(),
                "test": MagicMock(),
            }

            # Mock trainer
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"accuracy": 0.9, "auc": 0.95}
            mock_trainer.return_value = mock_trainer_instance

            main()

            # Verify trainer was initialized and called
            assert mock_trainer.called
            assert mock_trainer_instance.train.called

    def test_main_test_mode(self):
        """Test main function in test mode"""
        # Case 1: Test with checkpoint file
        self.args.mode = "test"
        self.args.checkpoint = "model.pth"
        self.args.batch_size = 4
        self.args.num_workers = 4

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.parent = mock_path
        mock_path.stem = "test_image"

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Predictor") as mock_predictor, patch(
            "pathlib.Path.glob"
        ) as mock_glob, patch(
            "pathlib.Path.__truediv__", return_value=mock_path
        ):

            # Mock glob để trả về đường dẫn ảnh giả
            mock_glob.return_value = iter([mock_path])

            # Mock test_loader với data giả
            mock_test_loader = MagicMock()
            mock_test_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),
                    )  # Domain labels
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": MagicMock(),
                "val": MagicMock(),
                "test": mock_test_loader,
            }

            mock_predictor_instance = MagicMock()
            mock_predictor.from_checkpoint.return_value = mock_predictor_instance

            main()

            # Verify predictor initialization và predictions
            mock_predictor.from_checkpoint.assert_called_once()
            assert mock_predictor_instance.predict.called

        # Case 2: Test with checkpoint directory
        self.args.checkpoint = "checkpoints/"
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.parent = mock_path
        mock_path.stem = "test_image"

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch("src.main.Predictor") as mock_predictor, patch(
            "pathlib.Path.glob"
        ) as mock_glob, patch(
            "pathlib.Path.__truediv__", return_value=mock_path
        ):

            # Mock glob
            mock_glob.return_value = iter([mock_path])

            mock_get_dataloaders.return_value = {
                "train": MagicMock(),
                "val": MagicMock(),
                "test": MagicMock(),
            }

            mock_predictor_instance = MagicMock()
            mock_predictor.from_checkpoint.return_value = mock_predictor_instance

            main()

            mock_predictor.from_checkpoint.assert_called_once()

        # Case 3: Test without checkpoint path
        self.args.checkpoint = None

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders:

            # Mock dataloaders
            mock_get_dataloaders.return_value = {
                "train": MagicMock(),
                "val": MagicMock(),
                "test": MagicMock(),
            }

            with pytest.raises(ValueError) as excinfo:
                main()
            assert "Checkpoint path required for test mode" in str(excinfo.value)

    def test_main_error_handling(self, setup):
        """Test error handling in main"""
        self.args.mode = "test"
        self.args.checkpoint = None

        with patch("src.main.parse_args", return_value=self.args), patch(
            "src.main.get_dataloaders"
        ) as mock_get_dataloaders, patch(
            "src.main.find_optimal_batch_size", return_value=4
        ), patch(
            "src.main.find_optimal_workers", return_value=2
        ):

            # Mock dataloaders with non-empty iterator
            mock_train_loader = MagicMock()
            mock_train_loader.__iter__.return_value = iter(
                [
                    (
                        torch.randn(4, 3, 256, 256),  # Images
                        torch.randn(4, 1, 32, 32),  # Depth maps
                        torch.randint(0, 2, (4,)),  # Labels
                        torch.randint(0, 5, (4,)),
                    )  # Domain labels
                ]
            )

            mock_get_dataloaders.return_value = {
                "train": mock_train_loader,
                "val": MagicMock(),
                "test": MagicMock(),
            }

            with pytest.raises(ValueError) as excinfo:
                main()
            assert "Checkpoint path required for test mode" in str(excinfo.value)

    @pytest.fixture(autouse=True)
    def cleanup(self, setup):
        """Cleanup after each test"""
        yield
        shutil.rmtree(self.config.output_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__])
