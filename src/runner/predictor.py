from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, Optional, Tuple

class Predictor:
    """SSAN model predictor for inference"""

    def __init__(
        self,
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader,
        device: str,
        output_dir: Optional[Path] = None
    ):
        """Initialize predictor
        
        Args:
            model: Trained SSAN model
            test_loader: Test data loader
            device: Device to run inference on
            output_dir: Optional directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.setup_directories()
        self.setup_logging()

    def _plot_roc_curve(self, labels: np.ndarray, scores: np.ndarray) -> None:
        """Plot and save ROC curve"""
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate') 
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save
            plt.savefig(self.plot_dir / 'roc_curve.png')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to plot ROC curve: {str(e)}")

    def setup_directories(self) -> None:
        """Create necessary directories"""
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / f"test_{timestamp}"
            self.csv_dir = self.output_dir / 'csv'
            self.plot_dir = self.output_dir / 'plots'
            self.log_dir = self.output_dir / 'logs'
            
            for d in [self.csv_dir, self.plot_dir, self.log_dir]:
                d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        if self.output_dir:
            # Đảm bảo thư mục log tồn tại
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / 'inference.log'
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO
            )
        self.logger = logging.getLogger(__name__)

    def predict_batch(self, batch: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a single batch
        
        Args:
            batch: Batch from dataloader containing (images, depth_maps, labels, domains)
            
        Returns:
            Tuple of predictions and probabilities
        """
        images, _, labels, _ = batch
        images = images.to(self.device)
        
        with torch.no_grad():
            # Forward pass without domain adversarial
            pred, _ = self.model(images)
            
            # Average spatial dimensions và squeeze hết các chiều phụ
            pred = F.adaptive_avg_pool2d(pred, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B]
            
            # Get probabilities
            probs = torch.sigmoid(pred)
            
            # Get binary predictions
            preds = (probs > 0.5).float()
            
        return preds, probs

    def predict(self) -> Dict[str, np.ndarray]:
        """Run inference on full test set
        
        Returns:
            Dictionary containing predictions, probabilities, and labels
        """
        self.logger.info("Starting inference...")
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Inference'):
                preds, probs = self.predict_batch(batch)
                _, _, labels, _ = batch
                
                # Convert to numpy và flatten
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten()) 
                all_labels.extend(labels.numpy().flatten())
        
        results = {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'labels': np.array(all_labels)
        }
        
        # Calculate and log metrics
        metrics = self.calculate_metrics(results)
        self.log_results(metrics)
        
        if self.output_dir:
            self.save_results(results, metrics)
            
        return results

    def calculate_metrics(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate inference metrics
        
        Args:
            results: Dictionary containing predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
        
        preds = results['predictions']
        probs = results['probabilities'] 
        labels = results['labels']
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'auc': roc_auc_score(labels, probs)
        }
        
        # Calculate TPR at specific FPR thresholds
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        # Find TPR at FPR=0.01 
        idx_01 = np.argmin(np.abs(fpr - 0.01))
        metrics['tpr@fpr=0.01'] = tpr[idx_01]
        
        # Calculate HTER
        idx_eer = np.argmin(np.abs(fpr - (1-tpr)))
        eer = (fpr[idx_eer] + (1-tpr[idx_eer])) / 2
        metrics['hter'] = eer
        
        return metrics

    def log_results(self, metrics: Dict[str, float]) -> None:
        """Log inference results
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        self.logger.info("Inference Results:")
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value:.4f}")

    def save_results(self, results: Dict[str, np.ndarray], metrics: Dict[str, float]) -> None:
        """Save prediction results, metrics and plots
        
        Args:
            results: Dictionary containing predictions and labels
            metrics: Dictionary of calculated metrics  
        """
        # Save predictions to CSV
        df = pd.DataFrame({
            'prediction': results['predictions'],
            'probability': results['probabilities'],
            'label': results['labels']
        })
        df.to_csv(self.csv_dir / 'predictions.csv', index=False)
        
        # Save metrics to CSV  
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.csv_dir / 'metrics.csv', index=False)
        
        # Plot ROC curve
        self._plot_roc_curve(results['labels'], results['probabilities'])
        
        self.logger.info(f"Results saved to {self.output_dir}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, test_loader, device, output_dir=None):
        """Load model from checkpoint path
        
        Args:
            checkpoint_path: Path to checkpoint file or folder containing checkpoints.
                        If folder, will try to load best.pth first, then latest.pth
        """
        if Path(checkpoint_path).is_dir():
            best_ckpt = Path(checkpoint_path) / 'best.pth'
            latest_ckpt = Path(checkpoint_path) / 'latest.pth'
            
            if best_ckpt.exists():
                checkpoint_path = best_ckpt
            elif latest_ckpt.exists(): 
                checkpoint_path = latest_ckpt
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return cls(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir
        )