import os
import sys
import pandas as pd
import pytest
import torch
import numpy as np
from pathlib import Path
import matplotlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.runner.predictor import Predictor 
from src.model.ssan import SSAN

matplotlib.use('Agg')

class TestPredictor:
    @pytest.fixture
    def setup(self):
        """Setup common test fixtures"""
        self.batch_size = 4
        self.img_size = 256
        self.num_domains = 5

        # Mock data
        self.sample_batch = (
            torch.randn(self.batch_size, 3, self.img_size, self.img_size),  # Images
            torch.randn(self.batch_size, 1, 32, 32),  # Depth maps
            torch.randint(0, 2, (self.batch_size,)),  # Labels 
            torch.randint(0, self.num_domains, (self.batch_size,))  # Domains
        )

        # Model
        self.model = SSAN(num_domains=self.num_domains)
        
        # Create mock test dataloader
        self.test_loader = self._create_mock_dataloader()

        # Setup output directory
        self.output_dir = Path("output/test_predictor")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor
        self.predictor = Predictor(
            model=self.model,
            test_loader=self.test_loader,
            device='cpu',
            output_dir=self.output_dir
        )

    def _create_mock_dataloader(self):
        """Create a mock test dataloader with balanced labels"""
        dataset = []
        for i in range(8):  # 8 samples total
            dataset.append((
                torch.randn(3, self.img_size, self.img_size),  # Image
                torch.randn(1, 32, 32),  # Depth map 
                torch.tensor(i % 2).long(),  # Alternate between 0 and 1 labels
                torch.randint(0, self.num_domains, ()).long()  # Random domain
            ))
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def test_init(self, setup):
        """Test predictor initialization"""
        assert isinstance(self.predictor.model, SSAN)
        assert self.predictor.device == 'cpu'
        assert str(self.predictor.output_dir).startswith(str(self.output_dir))  # Chỉ check prefix
        # Check directory structure
        for dir_path in [self.predictor.csv_dir, self.predictor.plot_dir, self.predictor.log_dir]:
            assert dir_path.exists()
            assert dir_path.is_dir()

    def test_predict_batch(self, setup):
        """Test prediction on single batch"""
        # Get a batch
        batch = next(iter(self.test_loader))
        
        # Run prediction
        preds, probs = self.predictor.predict_batch(batch)
        
        # Check shapes and types
        assert preds.shape == (min(self.batch_size, len(batch[0])),)  # [B]
        assert probs.shape == (min(self.batch_size, len(batch[0])),)  # [B]
        assert preds.dtype == torch.float32
        assert probs.dtype == torch.float32
        
        # Check value ranges
        assert torch.all((preds == 0) | (preds == 1))  # Binary predictions
        assert torch.all((probs >= 0) & (probs <= 1))  # Valid probabilities
        
        # Test on device
        if self.predictor.device != 'cpu':
            assert next(self.predictor.model.parameters()).device == torch.device(self.predictor.device)

    def test_predict(self, setup):
        """Test full prediction pipeline"""
        results = self.predictor.predict()
        
        # Check return dictionary
        assert all(k in results for k in ['predictions', 'probabilities', 'labels'])
        assert all(isinstance(v, np.ndarray) for v in results.values())
        
        # Check shapes
        n_samples = len(self.test_loader.dataset)
        assert results['predictions'].shape == (n_samples,)
        assert results['probabilities'].shape == (n_samples,)
        assert results['labels'].shape == (n_samples,)
        
        # Check values
        assert np.all(np.isin(results['predictions'], [0, 1]))  # Binary predictions
        assert np.all((results['probabilities'] >= 0) & (results['probabilities'] <= 1))
        
        # Check log file creation
        assert (self.predictor.log_dir / 'inference.log').exists()

    def test_calculate_metrics(self, setup):
        """Test metrics calculation"""
        # Create mock results with known values
        mock_results = {
            'predictions': np.array([0, 1, 0, 1]),
            'probabilities': np.array([0.2, 0.8, 0.3, 0.7]),
            'labels': np.array([0, 1, 0, 0])
        }
        
        metrics = self.predictor.calculate_metrics(mock_results)
        
        # Check required metrics
        required_metrics = ['accuracy', 'auc', 'tpr@fpr=0.01', 'hter']
        assert all(k in metrics for k in required_metrics)
        
        # Check metric types and ranges
        assert all(isinstance(v, float) for v in metrics.values())
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['tpr@fpr=0.01'] <= 1
        assert 0 <= metrics['hter'] <= 1

        # Test with perfect predictions
        perfect_results = {
            'predictions': np.array([0, 1, 0, 1]),
            'probabilities': np.array([0.0, 1.0, 0.0, 1.0]),
            'labels': np.array([0, 1, 0, 1])
        }
        perfect_metrics = self.predictor.calculate_metrics(perfect_results)
        assert perfect_metrics['accuracy'] == 1.0
        assert perfect_metrics['auc'] == 1.0

    def test_from_checkpoint(self, setup):
        """Test loading model from checkpoint"""
        # Create a mock checkpoint
        checkpoint_path = self.output_dir / "mock_checkpoint.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)
        
        # Create new predictor from checkpoint
        predictor = Predictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=SSAN(num_domains=self.num_domains),
            test_loader=self.test_loader,
            device='cpu',
            output_dir=str(self.output_dir)
        )
        
        # Test prediction works
        batch = next(iter(self.test_loader))
        preds, probs = predictor.predict_batch(batch)
        
        assert preds.shape == (min(self.batch_size, len(batch[0])),)
        assert probs.shape == (min(self.batch_size, len(batch[0])),)

    def test_save_results(self, setup):
        """Test saving prediction results and visualization"""
        # Create mock results and metrics
        results = {
            'predictions': np.array([0, 1, 0, 1]),
            'probabilities': np.array([0.2, 0.8, 0.3, 0.7]),
            'labels': np.array([0, 1, 0, 0])
        }
        metrics = {
            'accuracy': 0.75,
            'auc': 0.8,
            'tpr@fpr=0.01': 0.7,
            'hter': 0.2
        }
        
        # Save results
        self.predictor.save_results(results, metrics)
        
        # Check files exist
        assert (self.predictor.csv_dir / 'predictions.csv').exists()
        assert (self.predictor.csv_dir / 'metrics.csv').exists()
        assert (self.predictor.plot_dir / 'roc_curve.png').exists()
        
        # Verify CSV contents
        pred_df = pd.read_csv(self.predictor.csv_dir / 'predictions.csv')
        assert all(c in pred_df.columns for c in ['prediction', 'probability', 'label'])
        
        metrics_df = pd.read_csv(self.predictor.csv_dir / 'metrics.csv')
        assert all(k in metrics_df.columns for k in metrics.keys())
        for k, v in metrics.items():
            assert np.isclose(metrics_df[k].iloc[0], v)

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup after each test"""
        yield
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)

if __name__ == '__main__':
    pytest.main([__file__])