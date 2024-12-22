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
    parser = argparse.ArgumentParser()
    
    # Basic configs
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--protocol', type=str, required=True, choices=['protocol_1', 'protocol_2', 'protocol_3', 'protocol_4'])
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for testing')
    
    # Training configs 
    parser.add_argument('--epochs', type=int, help='Override default epochs')
    parser.add_argument('--lr', type=float, help='Override default learning rate')
    parser.add_argument('--batch_size', type=int, help='Override default batch size')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine'])
    
    # Device configs
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, help='Override default workers')
    
    # Debug configs
    parser.add_argument('--debug_fraction', type=float, default=1.0, help='Fraction of data to use (0-1)')
    
    return parser.parse_args()

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