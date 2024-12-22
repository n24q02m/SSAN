import argparse
import torch
import random
import numpy as np

from .config import Config
from .data.datasets import get_dataloaders
from .model.ssan import SSAN
from .model.losses import ClassificationLoss, ContrastiveLoss, DomainAdversarialLoss
from .runner.trainer import Trainer
from .runner.predictor import Predictor
from .runner.optimizers import find_optimal_batch_size, find_optimal_workers

def parse_args():
    parser = argparse.ArgumentParser(
        description='SSAN Training and Testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Hiển thị giá trị mặc định
    )
    
    # Basic configs
    basic_group = parser.add_argument_group('Basic configurations')
    basic_group.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'test'],
        help='Running mode: train model or test with checkpoint'
    )
    basic_group.add_argument(
        '--protocol', type=str, required=True,
        choices=['protocol_1', 'protocol_2', 'protocol_3', 'protocol_4'],
        help='''Training protocol:
                protocol_1: Single dataset (CelebA-Spoof)
                protocol_2: Multi-dataset (CelebA-Spoof + CATI-FAS)
                protocol_3: Cross-dataset evaluation
                protocol_4: Domain generalization'''
    )
    basic_group.add_argument(
        '--checkpoint', type=str,
        help='Path to checkpoint file (.pth) or folder containing checkpoints for testing'
    )

    # Training configs
    train_group = parser.add_argument_group('Training configurations') 
    train_group.add_argument(
        '--epochs', type=int,
        help='Number of training epochs (default: from config)'
    )
    train_group.add_argument(
        '--lr', type=float,
        help='Learning rate (default: from config)'
    )
    train_group.add_argument(
        '--batch_size', type=int,
        help='Batch size (default: auto)'
    )
    train_group.add_argument(
        '--optimizer', type=str,
        choices=['adam', 'sgd'],
        help='Optimizer type (default: from config)'
    )
    train_group.add_argument(
        '--scheduler', type=str,
        choices=['step', 'cosine'],
        help='Learning rate scheduler (default: from config)'
    )

    # Device configs
    device_group = parser.add_argument_group('Device configurations')
    device_group.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    device_group.add_argument(
        '--num_workers', type=int,
        help='Number of data loading workers (default: auto)'
    )

    # Debug configs
    debug_group = parser.add_argument_group('Debug configurations')
    debug_group.add_argument(
        '--debug_fraction', type=float, default=1.0,
        help='Fraction of data to use (0-1)'
    )

    # Short config aliases
    basic_group.add_argument(
        '-m', '--mode_short', dest='mode',
        type=str, choices=['train', 'test'],
        help='Short for --mode'
    )
    basic_group.add_argument(
        '-p', '--protocol_short', dest='protocol',
        type=str, choices=['p1', 'p2', 'p3', 'p4'],
        help='Short for --protocol (p1=protocol_1, etc)'
    )
    train_group.add_argument(
        '-e', '--epochs_short', dest='epochs',
        type=int, help='Short for --epochs'
    )
    train_group.add_argument(
        '-b', '--batch_short', dest='batch_size',
        type=int, help='Short for --batch_size'
    )

    args = parser.parse_args()

    # Convert short protocol names to full names
    if args.protocol in ['p1', 'p2', 'p3', 'p4']:
        args.protocol = f"protocol_{args.protocol[1]}"

    return args

def print_config(args, config):
    """Print configuration settings"""
    print("\n=== Configuration ===")
    print(f"Mode: {args.mode}")
    print(f"Protocol: {args.protocol}")
    print(f"Device: {args.device}")
    
    if args.mode == 'train':
        print("\nTraining settings:")
        print(f"- Epochs: {config.num_epochs}")
        print(f"- Learning rate: {config.learning_rate}")
        print(f"- Batch size: {config.batch_size}")
        print(f"- Optimizer: {config.optimizer}")
        print(f"- Scheduler: {config.scheduler}")
        print(f"- Workers: {config.num_workers}")
    
    if args.debug_fraction < 1.0:
        print(f"\nDebug mode: Using {args.debug_fraction*100:.1f}% of data")
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
    if config.optimizer == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        return torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

def get_scheduler(optimizer, config):
    if config.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.warmup_epochs,
            eta_min=config.min_lr
        )

def main():
    args = parse_args()
    
    # Initialize config
    config = Config()
    config.protocol = args.protocol
    
    # Override config with args
    if args.debug_fraction:
        config.debug_fraction = args.debug_fraction
        print(f"Using {config.debug_fraction*100:.1f}% of data for debugging")
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
    
    # Create dataloaders
    dataloaders = get_dataloaders(config)
    
    # Find optimal batch size and workers if not specified
    if not args.batch_size or not args.num_workers:
        model = SSAN(
            num_domains=config.num_domains,
            ada_blocks=config.ada_blocks,
            dropout=config.dropout
        ).to(args.device)
        
        sample_batch = next(iter(dataloaders['train']))
        if not args.batch_size:
            config.batch_size = find_optimal_batch_size(model, sample_batch)
        if not args.num_workers:
            config.num_workers = find_optimal_workers(dataloaders['train'])
            
        # Recreate dataloaders with optimal values
        dataloaders = get_dataloaders(config)
    
    if args.mode == 'train':
        # Initialize model
        model = SSAN(
            num_domains=config.num_domains,
            ada_blocks=config.ada_blocks, 
            dropout=config.dropout
        ).to(args.device)
        
        # Initialize training components
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        criterion = {
            'cls': ClassificationLoss(),
            'domain': DomainAdversarialLoss(),
            'contrast': ContrastiveLoss()
        }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test'],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            device=args.device
        )
        
        # Train
        best_metrics = trainer.train()
        print("Training completed. Best metrics:", best_metrics)
        
    else:  # Test mode
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for test mode")
            
        # Initialize model and predictor
        model = SSAN(
            num_domains=config.num_domains,
            ada_blocks=config.ada_blocks,
            dropout=0.0  # No dropout for inference
        )
        
        predictor = Predictor.from_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            test_loader=dataloaders['test'],
            device=args.device,
            output_dir=config.output_dir
        )
        
        # Run prediction
        results = predictor.predict()
        print("Testing completed")

if __name__ == '__main__':
    main()