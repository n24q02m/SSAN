import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from tqdm import tqdm
import numpy as np
import logging
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

matplotlib.use("Agg")


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
        callbacks: Optional[Dict[str, callable]] = None,
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
        self.lambda_val = 0.0
        self.current_epoch = 0
        self.global_step = 0
        self.scaler = GradScaler()

        # Metrics tracking
        self.best_metrics = {
            "epoch": 0,
            "auc": 0,
            "accuracy": 0,
            "loss": float("inf"),
            "tpr@fpr=0.1": 0,
            "hter": 1.0,
            "val_loss": float("inf"),
        }

        # Early stopping
        self.patience = config.patience
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")
        self.should_stop = False

        # Setup output directory first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config.output_dir
        if not hasattr(config, "run_dir"):
            config.run_dir = self.output_dir / f"train_{timestamp}"
        self.run_dir = config.run_dir

        # Now setup directories and logging
        self.setup_directories()
        self.setup_logging()

    def _init_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary"""
        return {
            "loss": 0.0,
            "cls_loss": 0.0,
            "domain_loss": 0.0,
            "contrast_loss": 0.0,
            "accuracy": 0.0,
            "count": 0,
        }

    def _compute_losses(
        self, pred, domain_pred, labels, domains, feat_orig, feat_style, contrast_labels
    ) -> Dict[str, Any]:
        """Compute all losses

        Returns:
            Dictionary containing total loss tensor and individual loss values
        """
        cls_loss = self.criterion["cls"](pred, labels)
        domain_loss = self.criterion["domain"](domain_pred, domains)
        contrast_loss = self.criterion["contrast"](
            feat_orig, feat_style, contrast_labels
        )

        total_loss = (
            cls_loss
            + self.config.lambda_adv * domain_loss
            + self.config.lambda_contrast * contrast_loss
        )

        return {
            "total": total_loss,  # Keep tensor for backprop
            "cls_loss": cls_loss.item(),
            "domain_loss": domain_loss.item(),
            "contrast_loss": contrast_loss.item(),
        }

    def _compute_accuracy(self, pred, labels) -> float:
        """Compute classification accuracy"""
        pred_cls = (torch.sigmoid(pred) > 0.5).float()
        return (pred_cls == labels).float().mean().item()

    def _update_metrics(
        self, metrics: Dict[str, float], batch_metrics: Dict[str, float]
    ) -> None:
        """Update running metrics with batch metrics"""
        for k in metrics:
            if k != "count":
                metrics[k] += batch_metrics.get(k, 0.0)
        metrics["count"] += 1

    def _finalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate final average metrics"""
        count = metrics.pop("count")
        return {k: v / count for k, v in metrics.items()}

    def _update_progress(self, pbar, metrics: Dict[str, float]) -> None:
        """Update progress bar with current metrics"""
        avg_metrics = {
            k: v / metrics["count"] for k, v in metrics.items() if k != "count"
        }
        pbar.set_postfix(avg_metrics)

    def _eval_step(self, batch) -> tuple:
        """Execute one evaluation step"""
        images, depth_maps, labels, domains = batch

        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            # Forward pass without domain adversarial
            pred, _ = self.model(images)

            # Fix: Reshape prediction tensor correctly
            pred = pred.view(pred.size(0), -1).mean(
                dim=1
            )  # Average over spatial dimensions
            scores = torch.sigmoid(pred)

            return labels.cpu(), scores.cpu()

    def calculate_metrics(
        self, labels: np.ndarray, scores: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                self.logger.warning(f"Batch contains only class {unique_labels[0]}")
                return {
                    "loss": -np.mean(
                        labels * np.log(scores + 1e-7)
                        + (1 - labels) * np.log(1 - scores + 1e-7)
                    ),
                    "auc": 0.5,  # AUC = 0.5 for random prediction
                    "accuracy": accuracy_score(labels, scores > 0.5),
                    "tpr@fpr=0.01": 0.0,
                    "fpr@tpr=0.99": 1.0,
                    "hter": 0.5,  # HTER = 0.5 for random prediction
                }

            # Tính metrics bình thường nếu có đủ classes
            fpr, tpr, thresholds = roc_curve(labels, scores)

            # Tính TPR@FPR=0.01 với nội suy tuyến tính cẩn thận hơn
            idx_fpr = (
                np.searchsorted(fpr, 0.01, side="right") - 1
            )  # Lấy điểm gần nhất bên trái
            if idx_fpr >= 0 and idx_fpr + 1 < len(fpr):
                # Nội suy tuyến tính giữa 2 điểm
                slope = (tpr[idx_fpr + 1] - tpr[idx_fpr]) / (
                    fpr[idx_fpr + 1] - fpr[idx_fpr]
                )
                tpr_at_fpr01 = tpr[idx_fpr] + slope * (0.01 - fpr[idx_fpr])
            else:
                # Nếu không thể nội suy, lấy giá trị gần nhất
                tpr_at_fpr01 = tpr[max(0, idx_fpr)]

            # Tương tự cho FPR@TPR=0.99
            idx_tpr = np.searchsorted(tpr, 0.99, side="left")
            if idx_tpr > 0 and idx_tpr < len(tpr):
                slope = (fpr[idx_tpr] - fpr[idx_tpr - 1]) / (
                    tpr[idx_tpr] - tpr[idx_tpr - 1]
                )
                fpr_at_tpr99 = fpr[idx_tpr - 1] + slope * (0.99 - tpr[idx_tpr - 1])
            else:
                fpr_at_tpr99 = fpr[min(len(fpr) - 1, idx_tpr)]

            return {
                "loss": -np.mean(
                    labels * np.log(scores + 1e-7)
                    + (1 - labels) * np.log(1 - scores + 1e-7)
                ),
                "auc": roc_auc_score(labels, scores),
                "accuracy": accuracy_score(labels, scores > 0.5),
                "tpr@fpr=0.01": tpr_at_fpr01,
                "fpr@tpr=0.99": fpr_at_tpr99,
                "hter": (1 - tpr_at_fpr01 + fpr_at_tpr99) / 2,
            }

        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {str(e)}")
            return {
                k: 0.0
                for k in [
                    "loss",
                    "auc",
                    "accuracy",
                    "tpr@fpr=0.01",
                    "fpr@tpr=0.99",
                    "hter",
                ]
            }

    def _save_checkpoints(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoints"""
        try:
            # Save latest checkpoint
            latest_path = self.ckpt_dir / "latest.pth"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
            }
            torch.save(checkpoint, latest_path)

            # Save best checkpoint based on multiple metrics
            is_best = False
            if not hasattr(self, "best_val_metrics"):
                self.best_val_metrics = metrics
                is_best = True
            else:
                # Compare using AUC as primary metric
                if metrics["auc"] > self.best_val_metrics["auc"]:
                    is_best = True
                elif metrics["auc"] == self.best_val_metrics["auc"]:
                    # Use accuracy as secondary metric
                    if metrics["accuracy"] > self.best_val_metrics["accuracy"]:
                        is_best = True

            if is_best:
                self.best_val_metrics = metrics.copy()
                best_path = self.ckpt_dir / "best.pth"
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved new best checkpoint at epoch {epoch+1}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoints: {str(e)}")

    def _log_metrics(self, mode: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Log training/validation metrics

        Args:
            mode: 'train' or 'val'
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        metric_str = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch+1} {mode}: {metric_str}")

        if mode == "val":
            # Update best metrics based on multiple criteria
            should_update = metrics["accuracy"] > self.best_metrics["accuracy"] or (
                metrics["accuracy"] == self.best_metrics["accuracy"]
                and metrics["auc"] > self.best_metrics["auc"]
            )

            if should_update:
                self.best_metrics = {
                    "epoch": epoch,
                    "accuracy": metrics["accuracy"],
                    "auc": metrics["auc"],
                    "loss": metrics["loss"],
                    "tpr@fpr=0.01": metrics.get("tpr@fpr=0.01", 0),
                    "hter": metrics.get("hter", 0),  # Fix: Initialize to 0
                    "val_loss": metrics["loss"],
                }

    def _log_final_results(self, metrics: Dict[str, float]) -> None:
        """Log final evaluation results"""
        # Log to file
        with open(self.log_dir / "final_results.txt", "w") as f:
            f.write("Final Test Results:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")

            f.write("\nBest Validation Metrics:\n")
            for k, v in self.best_metrics.items():
                f.write(f"{k}: {v:.6f}\n")

        # Log to console
        metric_str = " ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Final test results: {metric_str}")

        best_str = " ".join([f"{k}={v:.6f}" for k, v in self.best_metrics.items()])
        self.logger.info(f"Best validation metrics: {best_str}")

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

        # Add maximum counter to force stop
        max_patience = self.patience * 2
        if self.early_stopping_counter >= max_patience:
            self.should_stop = True
            self.logger.info(
                f"Forced early stopping after {max_patience} epochs without improvement"
            )
            return True

        if self.early_stopping_counter >= self.patience:
            self.should_stop = True
            self.logger.info(
                f"Early stopping triggered after {self.patience} epochs without improvement"
            )

        return self.should_stop

    def _save_metrics_to_csv(
        self, metrics: Dict[str, float], mode: str, epoch: int
    ) -> None:
        """Save metrics to CSV file only once per epoch"""
        csv_path = self.csv_dir / f"{mode}_metrics.csv"

        # Tạo DataFrame mới với epoch index
        metrics_df = pd.DataFrame([{**metrics, "epoch": epoch}])

        if not csv_path.exists():
            metrics_df.to_csv(csv_path, index=False)
        else:
            # Đọc file cũ và kiểm tra xem epoch đã tồn tại chưa
            existing_df = pd.read_csv(csv_path)
            if epoch not in existing_df["epoch"].values:
                metrics_df.to_csv(csv_path, mode="a", header=False, index=False)

    def _plot_training_curves(self) -> None:
        """Plot training curves with proper synchronization"""
        try:
            train_df = pd.read_csv(self.csv_dir / "train_metrics.csv")
            val_df = pd.read_csv(self.csv_dir / "val_metrics.csv")

            # Add epoch column if missing
            if "epoch" not in train_df.columns:
                train_df["epoch"] = range(len(train_df))
            if "epoch" not in val_df.columns:
                val_df["epoch"] = range(len(val_df))

            # Plot metrics
            for metric, title in [
                ("loss", "Loss"),
                ("accuracy", "Accuracy"),
                ("auc", "AUC-ROC"),
                ("hter", "HTER"),
            ]:
                plt.figure(figsize=(10, 6))
                plt.plot(train_df["epoch"], train_df[metric], "b-", label="Train")
                plt.plot(val_df["epoch"], val_df[metric], "r-", label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel(title)
                plt.title(f"{title} vs. Epoch")
                plt.legend()
                plt.grid(True)
                plt.savefig(self.plot_dir / f"{metric}_curve.png")
                plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to plot curves: {str(e)}")

    def setup_directories(self) -> None:
        """Create necessary directories"""
        # Use run_dir instead of creating new timestamped directory
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.csv_dir = self.run_dir / "csv"
        self.plot_dir = self.run_dir / "plots"

        for d in [self.ckpt_dir, self.log_dir, self.csv_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        # Make sure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "training.log"

        # Configure logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            force=True,  # Force reconfiguration
        )

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Create logger and add handlers
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)

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
            all_labels, all_scores = [], []

            # Update lambda for GRL
            self.lambda_val = min(1.0, epoch / self.config.num_epochs)

            pbar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )

            # Fix: Chỉ lưu metrics một lần cho mỗi epoch
            for batch_idx, batch in enumerate(pbar):
                batch_metrics = self._train_step(batch)
                self._update_metrics(metrics, batch_metrics)
                self._update_progress(pbar, metrics)

                # Collect predictions
                images, _, labels, _ = batch
                with torch.no_grad():
                    pred, _ = self.model(images.to(self.device))
                    pred = torch.sigmoid(pred.view(pred.size(0), -1).mean(dim=1))
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(pred.cpu().numpy())

            # Finalize metrics once per epoch
            metrics = self._finalize_metrics(metrics)
            metrics.update(
                self.calculate_metrics(np.array(all_labels), np.array(all_scores))
            )

            # Save metrics once
            self._log_metrics("train", epoch, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error in training epoch {epoch}: {str(e)}")
            raise

    def _train_step(self, batch) -> Dict[str, float]:
        """Execute one training step"""
        images, depth_maps, labels, domains = batch
        
        images = images.to(self.device)
        depth_maps = depth_maps.to(self.device) 
        labels = labels.to(self.device)
        domains = domains.to(self.device)

        with autocast(device_type="cuda" if self.device == "cuda" else "cpu"):
            # Forward pass
            pred, domain_pred, feat_orig, feat_style, contrast_labels = self.model.shuffle_style_assembly(
                images, labels, domains, self.lambda_val
            )
            pred = pred.view(pred.size(0), -1).mean(dim=1)
            losses = self._compute_losses(
                pred,
                domain_pred, 
                labels,
                domains,
                feat_orig,
                feat_style,
                contrast_labels,
            )

        # Optimize với gradient clipping
        self.optimizer.zero_grad()
        self.scaler.scale(losses["total"]).backward()
        
        # Thêm gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": losses["total"].item(),
            "cls_loss": losses["cls_loss"],
            "domain_loss": losses["domain_loss"],
            "contrast_loss": losses["contrast_loss"],
            "accuracy": self._compute_accuracy(pred, labels),
        }

    def evaluate(self, loader: DataLoader, mode: str = "val") -> Dict[str, float]:
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
            for batch in tqdm(loader, desc=f"{mode.upper()} Eval"):
                batch_labels, batch_scores = self._eval_step(batch)
                all_labels.extend(batch_labels)
                all_scores.extend(batch_scores)

        return self.calculate_metrics(np.array(all_labels), np.array(all_scores))

    def train(self) -> Dict[str, float]:
        """Main training loop"""
        try:
            self.logger.info("Starting training...")

            for epoch in range(self.config.num_epochs):
                # Train
                train_metrics = self.train_epoch(epoch)
                self._save_metrics_to_csv(train_metrics, "train", epoch)

                # Validation
                val_metrics = self.evaluate(self.val_loader, mode="val")
                self._save_metrics_to_csv(val_metrics, "val", epoch)

                # Log metrics
                self._log_metrics("train", epoch, train_metrics)
                self._log_metrics("val", epoch, val_metrics)

                # Update scheduler
                self.scheduler.step()

                # Save checkpoints và plot
                self._save_checkpoints(epoch, val_metrics)
                self._plot_training_curves()

                # Early stopping check
                if self._check_early_stopping(val_metrics["loss"]):
                    break

            # Final evaluation
            test_metrics = self.evaluate(self.test_loader, mode="test")
            self._log_final_results(test_metrics)

            return self.best_metrics

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise
