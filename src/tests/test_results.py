import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.runner.results import Results

class TestResults:
    @pytest.fixture
    def setup(self):
        self.output_dir = Path("tests/output/results")
        self.results = Results(mode='test', run_name='test_run')
        
    def test_init(self, setup):
        """Test initialization"""
        assert self.results.output_dir.name == 'test_test_run'
        assert self.results.plot_dir.exists()
        assert self.results.csv_dir.exists()

    def test_save_training_history(self, setup):
        """Test saving training history"""
        history = {
            'train_loss': [0.5, 0.3],
            'val_loss': [0.4, 0.2],
            'train_accuracy': [0.8, 0.9],
            'val_accuracy': [0.85, 0.95]
        }
        self.results.save_training_history(history)
        assert (self.results.csv_dir / 'training_history.csv').exists()
        assert (self.results.plot_dir / 'training_curves.png').exists()

    def test_save_predictions(self, setup):
        """Test saving predictions"""
        predictions = np.array([0, 1, 0, 1])
        probabilities = np.array([0.2, 0.8, 0.3, 0.7]) 
        labels = np.array([0, 1, 0, 0])
        
        self.results.save_predictions(predictions, probabilities, labels)
        
        assert (self.results.csv_dir / 'predictions.csv').exists()
        assert (self.results.plot_dir / 'roc_curve.png').exists()
        assert (self.results.plot_dir / 'confusion_matrix.png').exists()
        assert (self.results.plot_dir / 'score_distributions.png').exists()

    @pytest.fixture(autouse=True) 
    def cleanup(self):
        """Cleanup after each test"""
        yield
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
        
if __name__ == '__main__':
    pytest.main([__file__])