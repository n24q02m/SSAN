import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        config,
        device
    ):
        """SSAN Trainer
        
        Args:
            model: SSAN model
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            optimizer: Optimizer (e.g. Adam)
            scheduler: Learning rate scheduler
            criterion: Loss function
            config: Training configuration
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # Khởi tạo best metrics
        self.best_metrics = {
            'epoch': 0,
            'auc': 0,
            'tpr@fpr=0.1': 0, 
            'hter': 1.0
        }

        # Tạo thư mục lưu checkpoints
        self.ckpt_dir = Path(config.checkpoint_dir) / config.run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        cls_losses = 0 
        domain_losses = 0
        contrast_losses = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, depth_maps, labels, domains) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device) 
            domains = domains.to(self.device)
            
            # Forward pass với style shuffling
            pred, domain_pred, feat_orig, feat_style, contrast_labels = \
                self.model.shuffle_style_assembly(images, labels, domains, self.config.lambda_grl)

            # Tính các thành phần loss
            cls_loss = self.criterion['cls'](pred, labels)
            domain_loss = self.criterion['domain'](domain_pred, domains)
            contrast_loss = self.criterion['contrast'](feat_orig, feat_style, contrast_labels)
            
            # Tổng loss
            loss = cls_loss + self.config.lambda_adv * domain_loss + \
                   self.config.lambda_contrast * contrast_loss

            # Backward và optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            cls_losses += cls_loss.item()
            domain_losses += domain_loss.item()
            contrast_losses += contrast_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'cls_loss': cls_losses / (batch_idx + 1),
                'domain_loss': domain_losses / (batch_idx + 1),
                'contrast_loss': contrast_losses / (batch_idx + 1)
            })

        # Return average losses
        return {
            'loss': total_loss / len(self.train_loader),
            'cls_loss': cls_losses / len(self.train_loader),
            'domain_loss': domain_losses / len(self.train_loader),
            'contrast_loss': contrast_losses / len(self.train_loader)
        }

    @torch.no_grad()
    def evaluate(self, loader, mode='val'):
        """Evaluate model"""
        self.model.eval()
        
        all_labels = []
        all_scores = []
        
        for images, depth_maps, labels, domains in tqdm(loader, desc=f'{mode.upper()} Eval'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass (không cần style shuffling khi evaluate)
            pred, _ = self.model(images)
            scores = torch.softmax(pred, dim=1)[:, 1]  # Probability của class live
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

        # Convert to numpy arrays  
        labels = np.array(all_labels)
        scores = np.array(all_scores)

        # Tính các metrics
        metrics = self.calculate_metrics(labels, scores)

        return metrics

    def calculate_metrics(self, labels, scores):
        """Calculate FAS metrics: AUC, TPR@FPR, HTER"""
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Tính TPR tại FPR=10%
        idx = np.argmin(np.abs(fpr - 0.1))
        tpr_at_fpr = tpr[idx]

        # Tính HTER (Half Total Error Rate)
        threshold = 0.5
        predictions = (scores >= threshold).astype(int)
        far = np.mean(predictions[labels == 0])  # False Accept Rate 
        frr = 1 - np.mean(predictions[labels == 1])  # False Reject Rate
        hter = (far + frr) / 2

        return {
            'auc': roc_auc,
            'tpr@fpr=0.1': tpr_at_fpr,
            'hter': hter
        }

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(ckpt, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(ckpt, best_path)

    def train(self):
        """Main training loop"""
        for epoch in range(self.config.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.config.num_epochs}')
            
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, mode='val')
            
            # Update scheduler
            self.scheduler.step()

            # Update best metrics and save checkpoint
            if val_metrics['auc'] > self.best_metrics['auc']:
                self.best_metrics = {
                    'epoch': epoch,
                    **val_metrics
                }
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save latest checkpoint
            self.save_checkpoint(epoch, val_metrics)

            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Val TPR@FPR=0.1: {val_metrics['tpr@fpr=0.1']:.4f}")
            print(f"Val HTER: {val_metrics['hter']:.4f}")
            print(f"Best Val AUC: {self.best_metrics['auc']:.4f} (Epoch {self.best_metrics['epoch']})")

        # Final evaluation on test set
        print('\nEvaluating on test set...')
        test_metrics = self.evaluate(self.test_loader, mode='test')
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test TPR@FPR=0.1: {test_metrics['tpr@fpr=0.1']:.4f}") 
        print(f"Test HTER: {test_metrics['hter']:.4f}")