import torch
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve

class Trainer:
    """SSAN Trainer with mixed precision training and comprehensive metrics tracking"""
    
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        criterion: Dict[str, Module],
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

    def _init_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary"""
        return {
            'loss': 0.0,
            'cls_loss': 0.0,  
            'domain_loss': 0.0,
            'contrast_loss': 0.0,
            'accuracy': 0.0,
            'count': 0
        }

    def _compute_losses(self, pred, domain_pred, labels, domains, feat_orig, feat_style, contrast_labels) -> Dict[str, Any]:
        """Compute all losses
        
        Returns:
            Dictionary containing total loss tensor and individual loss values
        """
        cls_loss = self.criterion['cls'](pred, labels)
        domain_loss = self.criterion['domain'](domain_pred, domains) 
        contrast_loss = self.criterion['contrast'](feat_orig, feat_style, contrast_labels)
        
        total_loss = cls_loss + \
                    self.config.lambda_adv * domain_loss + \
                    self.config.lambda_contrast * contrast_loss

        return {
            'total': total_loss,  # Keep tensor for backprop
            'cls_loss': cls_loss.item(),
            'domain_loss': domain_loss.item(), 
            'contrast_loss': contrast_loss.item()
        }

    def _compute_accuracy(self, pred, labels) -> float:
        """Compute classification accuracy"""
        # Fix: Use sigmoid and threshold for binary classification
        pred_cls = (torch.sigmoid(pred) > 0.5).float() 
        return (pred_cls == labels).float().mean().item()

    def _update_metrics(self, metrics: Dict[str, float], batch_metrics: Dict[str, float]) -> None:
        """Update running metrics with batch metrics"""
        for k in metrics:
            if k != 'count':
                metrics[k] += batch_metrics.get(k, 0.0)
        metrics['count'] += 1

    def _finalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate final average metrics"""
        count = metrics.pop('count')
        return {k: v/count for k, v in metrics.items()}

    def _update_progress(self, pbar, metrics: Dict[str, float]) -> None:
        """Update progress bar with current metrics"""
        avg_metrics = {k: v/metrics['count'] for k, v in metrics.items() if k != 'count'}
        pbar.set_postfix(avg_metrics)

    def _eval_step(self, batch) -> tuple:
        """Execute one evaluation step"""
        images, depth_maps, labels, domains = batch
        
        images = images.to(self.device)  # [B,C,H,W]  
        labels = labels.to(self.device)  # [B]

        with torch.no_grad():
            pred, _ = self.model(images)  # Forward without domain adversarial
            # Fix: Flatten prediction to match label shape 
            pred = pred.view(pred.size(0), -1).mean(dim=1)  # [B]
            scores = torch.sigmoid(pred)  # Use sigmoid for binary classification
            return labels.cpu(), scores.cpu()

    def calculate_metrics(self, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""        
        # Calculate AUC
        auc = roc_auc_score(labels, scores)
        
        # Calculate TPR at specific FPR thresholds
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find TPR at FPR=0.01
        idx_01 = np.argmin(np.abs(fpr - 0.01))
        tpr_at_fpr01 = tpr[idx_01]
        
        # Find FPR at TPR=0.99
        idx_99 = np.argmin(np.abs(tpr - 0.99))
        fpr_at_tpr99 = fpr[idx_99]
        
        # Calculate accuracy at best threshold
        best_thresh_idx = np.argmax(tpr - fpr)
        accuracy = np.mean((scores >= thresholds[best_thresh_idx]) == labels)
        
        # Calculate loss using binary cross entropy
        loss = -np.mean(labels * np.log(scores + 1e-7) + (1 - labels) * np.log(1 - scores + 1e-7))
        
        return {
            'loss': loss,
            'auc': auc,
            'accuracy': accuracy,
            'tpr@fpr=0.01': tpr_at_fpr01,
            'fpr@tpr=0.99': fpr_at_tpr99
        }

    def _save_checkpoints(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoints"""
        # Save latest checkpoint
        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, latest_path)

        # Save best checkpoint if current model is best
        if metrics['accuracy'] > self.best_metrics['accuracy']:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics
            }, best_path)

    def _log_metrics(self, mode: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training/validation metrics
        
        Args:
            mode: 'train' or 'val'
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        # Log to file/console
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch+1} {mode}: {metric_str}')
        
        # Update best metrics if validation
        if mode == 'val':
            current_metric = metrics.get('accuracy', 0)
            if current_metric > self.best_metrics['accuracy']:
                self.best_metrics.update({
                    'epoch': epoch,
                    'accuracy': current_metric,
                    'auc': metrics.get('auc', 0),
                    'loss': metrics.get('loss', float('inf')),
                    'tpr@fpr=0.1': metrics.get('tpr@fpr=0.01', 0),
                    'hter': metrics.get('hter', 1.0)
                })

    def _log_final_results(self, metrics: Dict[str, float]) -> None:
        """Log final evaluation results
        
        Args:
            metrics: Dictionary of final metrics
        """
        # Log final metrics
        metric_str = ' '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Final test results: {metric_str}')
        
        # Log best metrics
        best_str = ' '.join([f'{k}={v:.4f}' for k, v in self.best_metrics.items()])
        self.logger.info(f'Best validation metrics: {best_str}')

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        if self.early_stopping_counter >= self.patience:
            self.should_stop = True
            self.logger.info(f'Early stopping triggered after {self.patience} epochs without improvement')
            
        return self.should_stop

    def _save_metrics_to_csv(self, metrics: Dict[str, float], mode: str, epoch: int) -> None:
        """Save metrics to CSV file
        
        Args:
            metrics: Metrics dictionary to save
            mode: 'train' or 'val' or 'test'
            epoch: Current epoch number
        """
        # Create DataFrame from metrics
        df = pd.DataFrame([metrics])
        
        # Save to CSV
        csv_path = self.csv_dir / f"{mode}_metrics.csv"
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)

    def _plot_training_curves(self) -> None:
        """Plot and save training curves"""
        try:
            # Read metrics from CSV
            train_df = pd.read_csv(self.csv_dir / "train_metrics.csv")
            val_df = pd.read_csv(self.csv_dir / "val_metrics.csv")
            
            # Plot metrics
            metrics_to_plot = [
                ('loss', 'Loss'),
                ('accuracy', 'Accuracy'), 
                ('auc', 'AUC-ROC')
            ]
            
            for metric, title in metrics_to_plot:
                if metric in train_df.columns and metric in val_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_df[metric], label='Train')
                    plt.plot(val_df[metric], label='Validation')
                    plt.title(f'{title} vs Epoch')
                    plt.xlabel('Epoch')
                    plt.ylabel(title)
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(self.plot_dir / f'{metric}_curve.png')
                    plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to plot training curves: {str(e)}")

    def setup_directories(self) -> None:
        """Create necessary directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config.output_dir) / f"train_{timestamp}"
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs' 
        self.csv_dir = self.output_dir / 'csv'
        self.plot_dir = self.output_dir / 'plots'
        
        for d in [self.ckpt_dir, self.log_dir, self.csv_dir, self.plot_dir]:
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
            all_labels, all_scores = [], []  # Thêm để tính AUC
            
            # Update lambda for GRL
            self.lambda_val = min(1.0, epoch / self.config.num_epochs)
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                batch_metrics = self._train_step(batch)
                self._update_metrics(metrics, batch_metrics)
                self._update_progress(pbar, metrics)
                self.global_step += 1
                
                # Collect predictions for AUC calculation
                images, _, labels, _ = batch
                with torch.no_grad():
                    pred, _ = self.model(images.to(self.device))
                    pred = torch.sigmoid(pred.view(pred.size(0), -1).mean(dim=1))
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(pred.cpu().numpy())

                if 'on_batch_end' in self.callbacks:
                    self.callbacks['on_batch_end'](self, batch_metrics)

            # Calculate epoch averages
            metrics = self._finalize_metrics(metrics)
            
            # Add AUC metric
            metrics.update(self.calculate_metrics(np.array(all_labels), np.array(all_scores)))
            
            self._log_metrics('train', epoch, metrics)
            self._save_metrics_to_csv(metrics, 'train', epoch)
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error in training epoch {epoch}: {str(e)}")
            raise

    def _train_step(self, batch) -> Dict[str, float]:
        """Execute one training step"""
        # Unpack and reshape batch correctly
        images, depth_maps, labels, domains = batch
        
        images = images.to(self.device)  # [B,C,H,W]
        depth_maps = depth_maps.to(self.device)  # [B,1,H,W] 
        labels = labels.to(self.device)  # [B]
        domains = domains.to(self.device)  # [B]

        # Mixed precision forward pass
        with autocast(device_type='cuda' if self.device=='cuda' else 'cpu'):
            pred, domain_pred, feat_orig, feat_style, contrast_labels = \
                self.model.shuffle_style_assembly(images, labels, domains, self.lambda_val)
            
            # Fix: Average spatial dimensions for prediction
            pred = F.adaptive_avg_pool2d(pred, 1).squeeze(-1).squeeze(-1)  # [B]
            
            losses = self._compute_losses(pred, domain_pred, labels, domains,
                                        feat_orig, feat_style, contrast_labels)

        # Optimize using raw tensor loss
        self.optimizer.zero_grad()
        self.scaler.scale(losses['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Return loss values for metrics
        return {
            'total': losses['total'].item(),
            'cls_loss': losses['cls_loss'],
            'domain_loss': losses['domain_loss'],
            'contrast_loss': losses['contrast_loss'],
            'accuracy': self._compute_accuracy(pred, labels)
        }
    
    def evaluate(self, loader: DataLoader, mode: str = 'val') -> Dict[str, float]:
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
        """Main training loop"""
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

                # Save metrics to CSV
                self._save_metrics_to_csv(train_metrics, 'train', epoch)
                self._save_metrics_to_csv(val_metrics, 'val', epoch)
                
                # Update learning rate
                self.scheduler.step()
                
                # Check early stopping
                if self._check_early_stopping(val_metrics['loss']):
                    break
                    
                # Save checkpoints
                self._save_checkpoints(epoch, val_metrics)
                
                # Plot training curves
                self._plot_training_curves()
                
                # Call epoch end callback
                if 'on_epoch_end' in self.callbacks:
                    self.callbacks['on_epoch_end'](self, train_metrics, val_metrics)
                
            # Final evaluation
            self.logger.info("Training completed. Running final evaluation...") 
            test_metrics = self.evaluate(self.test_loader, mode='test')
            self._save_metrics_to_csv(test_metrics, 'test', self.current_epoch)
            self._log_final_results(test_metrics)

            return self.best_metrics

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
