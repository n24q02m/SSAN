import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc

class Results:
    """Class for saving training/prediction results and visualizations"""
    
    def __init__(self, mode: str, run_name: Optional[str] = None):
        """Initialize results manager
        
        Args:
            mode: Either 'train' or 'predict'
            run_name: Optional custom run name, defaults to timestamp
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = run_name or timestamp
        self.output_dir = Path('./output') / f'{mode}_{self.run_name}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plot_dir = self.output_dir / 'plots'
        self.csv_dir = self.output_dir / 'csv'
        self.plot_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)

    def save_training_history(self, history: Dict[str, List[float]]) -> None:
        """Save training history metrics to CSV and generate plots
        
        Args:
            history: Dictionary containing lists of metrics for each epoch
        """
        # Save metrics to CSV
        df = pd.DataFrame(history)
        df.to_csv(self.csv_dir / 'training_history.csv', index=False)
        
        # Plot training curves
        plt.figure(figsize=(12, 8))
        for metric in ['loss', 'accuracy', 'auc']:
            if f'train_{metric}' in history and f'val_{metric}' in history:
                plt.subplot(2, 2, len(plt.get_fignums()))
                plt.plot(history[f'train_{metric}'], label=f'Train {metric}')
                plt.plot(history[f'val_{metric}'], label=f'Val {metric}')
                plt.title(f'Training and Validation {metric.title()}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.title())
                plt.legend()
                
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'training_curves.png')
        plt.close()

    def save_evaluation_results(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """Save evaluation metrics to CSV
        
        Args:
            metrics: Dictionary of evaluation metrics
            prefix: Optional prefix for filename
        """
        df = pd.DataFrame([metrics])
        filename = f'{prefix}_metrics.csv' if prefix else 'metrics.csv'
        df.to_csv(self.csv_dir / filename, index=False)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = 'ROC Curve'
    ) -> None:
        """Plot and save ROC curve
        
        Args:
            y_true: Ground truth labels
            y_score: Predicted probabilities/scores
            title: Plot title
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(self.plot_dir / 'roc_curve.png')
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Confusion Matrix'
    ) -> None:
        """Plot and save confusion matrix
        
        Args:
            y_true: Ground truth labels  
            y_pred: Predicted labels
            title: Plot title
        """
        cm = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.savefig(self.plot_dir / 'confusion_matrix.png')
        plt.close()

    def plot_score_distributions(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        title: str = 'Score Distributions'
    ) -> None:
        """Plot and save score distributions for live/spoof
        
        Args:
            scores: Predicted probabilities/scores
            labels: Ground truth labels
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.hist(scores[labels == 1], bins=50, alpha=0.5, 
                label='Live', density=True)
        plt.hist(scores[labels == 0], bins=50, alpha=0.5,
                label='Spoof', density=True)
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.savefig(self.plot_dir / 'score_distributions.png')
        plt.close()

    def save_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Save predictions and additional visualizations
        
        Args:
            predictions: Binary predictions
            probabilities: Predicted probabilities
            labels: Ground truth labels
        """
        # Save predictions to CSV
        df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'label': labels
        })
        df.to_csv(self.csv_dir / 'predictions.csv', index=False)
        
        # Generate plots
        self.plot_roc_curve(labels, probabilities)
        self.plot_confusion_matrix(labels, predictions)
        self.plot_score_distributions(probabilities, labels)