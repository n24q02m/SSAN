import os
import sys
import argparse
import torch
from torch.optim import lr_scheduler, Adam, SGD
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc
from torch.cuda import empty_cache

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import Config
from src.data.datasets import (
    create_protocol_data,
    get_dataloaders,
    create_protocol_splits,
)
from src.model.ssan import SSAN
from src.model.losses import ClassificationLoss, ContrastiveLoss, DomainAdversarialLoss
from src.runner.trainer import Trainer
from src.runner.predictor import Predictor
from src.runner.optimizers import (
    HyperparameterOptimizer,
    find_optimal_batch_size,
    find_optimal_workers,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSAN Training and Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Hiển thị giá trị mặc định
    )

    # Basic configs
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Running mode: train model or test with checkpoint",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["protocol_1", "protocol_2", "protocol_3", "protocol_4"],
        help="""Training protocol:
                protocol_1: Single dataset (CelebA-Spoof)
                protocol_2: Multi-dataset (CelebA-Spoof + CATI-FAS)
                protocol_3: Cross-dataset evaluation
                protocol_4: Domain generalization""",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (.pth) or folder containing checkpoints for testing",
    )

    # Training configs
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--auto_hp",
        action="store_true",
        help="Automatically optimize hyperparameters using Optuna",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use for all splits (0.01-1.0)",
    )
    parser.add_argument("--lr", type=float, help="Learning rate (default: from config)")
    parser.add_argument("--batch_size", type=int, help="Batch size (default: auto)")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        help="Optimizer type (default: from config)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["step", "cosine"],
        help="Learning rate scheduler (default: from config)",
    )

    # Device configs
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of data loading workers (default: auto)"
    )

    args = parser.parse_args()

    return args


def print_config(args, config):
    """Print configuration settings"""
    print("\n=== Configuration ===")
    print(f"Mode: {args.mode}")
    print(f"Protocol: {args.protocol}")
    print(f"Device: {args.device}")

    if args.mode == "train":
        print("\nTraining settings:")
        print(f"- Epochs: {config.num_epochs}")
        print(f"- Learning rate: {config.learning_rate}")
        print(f"- Batch size: {config.batch_size}")
        print(f"- Optimizer: {config.optimizer}")
        print(f"- Scheduler: {config.scheduler}")
        print(f"- Workers: {config.num_workers}")

    if args.fraction < 1.0:
        print(f"\nUsing {args.fraction*100:.1f}% of data")
    print("===================\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_optimizer(model, config):
    if config.optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )


def get_scheduler(optimizer, config):
    if config.scheduler == "step":
        return lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    else:
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.warmup_epochs, eta_min=config.min_lr
        )


def configure_process_pool():
    """Configure process pool based on system"""
    cpu_count = mp.cpu_count()
    if torch.cuda.is_available():
        # Fewer processes when using GPU to avoid memory issues
        return min(4, cpu_count)
    return cpu_count


class ParameterCache:
    """Cache class for optimal parameters"""

    def __init__(self):
        self._cache = {}
        self._found = False

    def get(self, key):
        return self._cache.get(key) if self._found else None

    def set(self, key, value):
        self._cache[key] = value
        self._found = True

    def has_optimal_params(self):
        return self._found


# Create global cache instance
parameter_cache = ParameterCache()


def get_optimal_parameters(model, sample_batch, train_loader, device):
    """Calculate optimal parameters"""
    batch_size = find_optimal_batch_size(model, sample_batch)
    workers = find_optimal_workers(train_loader)
    return batch_size, workers


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        empty_cache()
        gc.collect()


def main():
    args = parse_args()

    # Configure process pool
    num_processes = configure_process_pool()

    # Clear GPU memory before starting
    clear_gpu_memory()

    # Validate fraction
    if args.fraction < 0.01 or args.fraction > 1.0:
        raise ValueError("Fraction must be between 0.01 (1%) and 1.0 (100%)")

    print("\nInitializing...")
    config = Config()
    config.protocol = args.protocol

    # Override config with args
    if args.fraction:
        config.fraction = max(0.01, args.fraction)
        print(f"Using {config.fraction*100:.1f}% of data")
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.optimizer:
        config.optimizer = args.optimizer
    if args.scheduler:
        config.scheduler = args.scheduler
    if args.num_workers:
        config.num_workers = args.num_workers

    # Print configuration
    print_config(args, config)

    # Set random seed
    set_seed(config.seed)

    # Create dataloaders once
    print("\nLoading datasets...")
    dataloaders = get_dataloaders(config)

    # Initialize model once
    model = SSAN(
        num_domains=config.num_domains,
        ada_blocks=config.ada_blocks,
        dropout=config.dropout,
    )

    # Move model to device
    if args.device == "cuda":
        model = model.to(args.device)

    print("\nFinding optimal parameters...")
    # Only find optimal parameters if not cached
    if not parameter_cache.has_optimal_params():
        if not args.batch_size or not args.num_workers:
            sample_batch = next(iter(dataloaders["train"]))

            if not args.batch_size:
                print("Finding optimal batch size...")
                config.batch_size = find_optimal_batch_size(model, sample_batch)
                print(f"Optimal batch size: {config.batch_size}")

            if not args.num_workers:
                print("Finding optimal number of workers...")
                config.num_workers = find_optimal_workers(dataloaders["train"])
                print(f"Optimal workers: {config.num_workers}")

            # Cache the parameters
            parameter_cache.set(
                f"{args.device}_{config.img_size}_{config.protocol}",
                (config.batch_size, config.num_workers),
            )

            # Recreate dataloaders with optimal values
            print("\nRecreating dataloaders with optimal values...")
            dataloaders = get_dataloaders(config)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if args.mode == "train":
            print("\nPreparing for training...")

            if args.auto_hp:
                print("\nStarting hyperparameter optimization...")

                hp_optimizer = HyperparameterOptimizer(
                    model_class=SSAN,
                    train_loader=dataloaders["train"],
                    val_loader=dataloaders["val"],
                    config=config,
                    study_name=f"ssan_{config.protocol}",
                    output_dir=config.output_dir / "hp_optimization",
                )

                best_params = hp_optimizer.optimize()

                if best_params:  # Only update if we got valid parameters
                    print("\nBest hyperparameters found:")
                    for param, value in best_params.items():
                        print(f"  {param}: {value}")
                        setattr(config, param, value)

                    # Recreate everything with optimized parameters
                    if "batch_size" in best_params:
                        print("\nRecreating dataloaders with optimal batch size...")
                        dataloaders = get_dataloaders(config)
                else:
                    print("\nNo optimal parameters found, using default values")

            # Initialize model AFTER hyperparameter optimization
            print("\nInitializing model and training components...")
            model = SSAN(
                num_domains=config.num_domains,
                ada_blocks=config.ada_blocks,
                dropout=config.dropout,
            ).to(args.device)
            print("Model initialized")

            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)
            criterion = {
                "cls": ClassificationLoss(),
                "domain": DomainAdversarialLoss(),
                "contrast": ContrastiveLoss(),
            }

            # Enable cuDNN benchmarking
            if args.device == "cuda":
                torch.backends.cudnn.benchmark = True

            # Use gradient scaling for mixed precision training
            scaler = torch.amp.GradScaler()

            # Initialize trainer
            trainer = Trainer(
                model=model,
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                test_loader=dataloaders["test"],
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                config=config,
                device=args.device,
            )

            # Train with final parameters
            best_metrics = trainer.train()
            print("Training completed. Best metrics:", best_metrics)

            # Clear memory after training
            clear_gpu_memory()

        else:  # Test mode
            if not args.checkpoint:
                raise ValueError("Checkpoint path required for test mode")

            # Initialize model and predictor
            model = SSAN(
                num_domains=config.num_domains,
                ada_blocks=config.ada_blocks,
                dropout=0.0,  # No dropout for inference
            )

            # Disable gradients for inference
            with torch.no_grad():
                predictor = Predictor.from_checkpoint(
                    checkpoint_path=args.checkpoint,
                    model=model,
                    test_loader=dataloaders["test"],
                    device=args.device,
                    output_dir=config.output_dir,
                )

                # Run prediction
                results = predictor.predict()
                print("Testing completed")

            clear_gpu_memory()

    return best_metrics


if __name__ == "__main__":
    main()
