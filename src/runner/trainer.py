import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional

class Trainer:
    """SSAN Trainer with mixed precision training and comprehensive metrics tracking"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: Dict[str, nn.Module],
        config: Any,
        device: str,
        callbacks: Optional[Dict[str, callable]] = None
    ):
        """Initialize trainer with all components
        
        Args:
            model: SSAN model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader  
            optimizer: Optimizer (e.g. Adam)
            scheduler: Learning rate scheduler
            criterion: Dictionary of loss functions
            config: Training configuration 
            device: Device to run on
            callbacks: Optional dictionary of callback functions
        """
        # Core components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        self.callbacks = callbacks or {}

        # Training state
        self.lambda_val = 0.0  # Lambda for GRL
        self.current_epoch = 0
        self.global_step = 0
        self.scaler = GradScaler()  # For mixed precision training
        
        # Metrics tracking
        self.best_metrics = {
            'epoch': 0,
            'auc': 0,
            'accuracy': 0, 
            'loss': float('inf'),
            'tpr@fpr=0.1': 0,
            'hter': 1.0,
            'val_loss': float('inf')
        }

        # Early stopping
        self.patience = config.patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.should_stop = False

        # Setup paths and logging
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self) -> None:
        """Create necessary directories"""
        self.output_dir = Path(self.config.output_dir)
        self.ckpt_dir = self.output_dir / 'checkpoints' / self.config.run_name
        self.log_dir = self.output_dir / 'logs' / self.config.run_name
        
        for d in [self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = self.log_dir / 'training.log'
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            self.model.train()
            metrics = self._init_metrics()
            
            # Update lambda for GRL
            self.lambda_val = min(1.0, epoch / self.config.num_epochs)
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                batch_metrics = self._train_step(batch)
                self._update_metrics(metrics, batch_metrics)
                self._update_progress(pbar, metrics)
                self.global_step += 1

                # Call batch end callback
                if 'on_batch_end' in self.callbacks:
                    self.callbacks['on_batch_end'](self, batch_metrics)

            # Calculate epoch averages
            metrics = self._finalize_metrics(metrics)
            self._log_metrics('train', epoch, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error in training epoch {epoch}: {str(e)}")
            raise

    def _train_step(self, batch) -> Dict[str, float]:
        """Execute one training step
        
        Args:
            batch: Current batch data
            
        Returns:
            Dictionary containing batch metrics
        """
        images, depth_maps, labels, domains = [x.to(self.device) for x in batch]

        # Mixed precision forward pass
        with autocast():
            pred, domain_pred, feat_orig, feat_style, contrast_labels = \
                self.model.shuffle_style_assembly(images, labels, domains, self.lambda_val)

            losses = self._compute_losses(pred, domain_pred, labels, domains,
                                       feat_orig, feat_style, contrast_labels)

        # Optimize
        self.optimizer.zero_grad()
        self.scaler.scale(losses['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {**losses, 'accuracy': self._compute_accuracy(pred, labels)}

    def evaluate(self, loader: torch.utils.data.DataLoader, mode: str = 'val') -> Dict[str, float]:
        """Evaluate model on given data loader
        
        Args:
            loader: Data loader to evaluate on
            mode: Evaluation mode ('val' or 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_labels, all_scores = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{mode.upper()} Eval'):
                batch_labels, batch_scores = self._eval_step(batch)
                all_labels.extend(batch_labels)
                all_scores.extend(batch_scores)

        return self.calculate_metrics(np.array(all_labels), np.array(all_scores))

    def train(self) -> Dict[str, float]:
        """Main training loop
        
        Returns:
            Best metrics achieved during training
        """
        try:
            self.logger.info("Starting training...")
            
            # Training loop
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Call epoch start callback
                if 'on_epoch_start' in self.callbacks:
                    self.callbacks['on_epoch_start'](self, epoch)
                
                # Train and evaluate
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader, mode='val')
                
                # Update learning rate
                self.scheduler.step()
                
                # Check early stopping
                if self._check_early_stopping(val_metrics['loss']):
                    break
                    
                # Save checkpoints
                self._save_checkpoints(epoch, val_metrics)
                
                # Call epoch end callback
                if 'on_epoch_end' in self.callbacks:
                    self.callbacks['on_epoch_end'](self, train_metrics, val_metrics)
                
            # Final evaluation
            self.logger.info("Training completed. Running final evaluation...")
            test_metrics = self.evaluate(self.test_loader, mode='test') 
            self._log_final_results(test_metrics)

            return self.best_metrics

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise