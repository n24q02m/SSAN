import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import torch
import torch.nn as nn

from src.model.components import (
    FeatureGenerator, ContentExtractor, StyleExtractor,
    AdaIN, Classifier, DomainDiscriminator
)
from src.model.losses import (
    ClassificationLoss, ContrastiveLoss,
    DomainAdversarialLoss, SSANLoss
)
from src.model.ssan import SSAN

class TestComponents:
    def test_feature_generator(self):
        batch_size = 4
        channels = 3
        height = 256
        width = 256
        
        model = FeatureGenerator(input_channels=channels)
        x = torch.randn(batch_size, channels, height, width)
        
        final_feat, scale_feats = model(x)
        
        # Check output shapes
        assert final_feat.shape == (batch_size, 256, height//16, width//16)
        assert len(scale_feats) == 3  # 3 scale features
        
    def test_content_extractor(self):
        batch_size = 4
        channels = 256 
        height = 16
        width = 16
        
        model = ContentExtractor(in_channels=channels)
        x = torch.randn(batch_size, channels, height, width)
        
        out = model(x)
        assert out.shape == (batch_size, channels, height, width)
        
    def test_style_extractor(self):
        batch_size = 4
        channels = 128
        h1, w1 = 128, 128  # First scale
        h2, w2 = 64, 64    # Second scale
        h3, w3 = 32, 32    # Third scale
        
        model = StyleExtractor()
        feat_scales = [
            torch.randn(batch_size, channels, h1, w1),
            torch.randn(batch_size, channels, h2, w2),
            torch.randn(batch_size, channels, h3, w3)
        ]
        
        gamma, beta = model(feat_scales)
        assert gamma.shape == (batch_size, 256)
        assert beta.shape == (batch_size, 256)

class TestLosses:
    def test_classification_loss(self):
        batch_size = 4
        num_classes = 2
        
        criterion = ClassificationLoss()
        pred = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))
        
        loss = criterion(pred, target)
        assert isinstance(loss.item(), float)
        
    def test_contrastive_loss(self):
        batch_size = 4
        feat_dim = 256
        
        criterion = ContrastiveLoss()
        anchor = torch.randn(batch_size, feat_dim)
        positive = torch.randn(batch_size, feat_dim)
        labels = torch.randint(0, 2, (batch_size,)) * 2 - 1  # {-1, 1}
        
        loss = criterion(anchor, positive, labels)
        assert isinstance(loss.item(), float)
        
    def test_domain_adversarial_loss(self):
        batch_size = 4
        num_domains = 5
        
        criterion = DomainAdversarialLoss()
        pred = torch.randn(batch_size, num_domains)
        target = torch.randint(0, num_domains, (batch_size,))
        
        loss = criterion(pred, target)
        assert isinstance(loss.item(), float)

class TestSSAN:
    def test_forward(self):
        batch_size = 4
        channels = 3
        height = 256
        width = 256
        num_domains = 5
        
        model = SSAN(num_domains=num_domains)
        x = torch.randn(batch_size, channels, height, width)
        domain_labels = torch.randint(0, num_domains, (batch_size,))
        
        # Test normal forward
        pred, domain_pred = model(x, domain_labels)
        assert pred.shape == (batch_size, 1, height//16, width//16)
        assert domain_pred.shape == (batch_size, num_domains)
        
        # Test forward with feature return
        pred, domain_pred, feats = model(x, domain_labels, return_feats=True)
        assert 'content_feat' in feats
        assert 'style_feat' in feats
        assert 'gamma' in feats
        assert 'beta' in feats
        
    def test_shuffle_style_assembly(self):
        batch_size = 4
        channels = 3
        height = 256
        width = 256
        num_domains = 5
        
        model = SSAN(num_domains=num_domains)
        x = torch.randn(batch_size, channels, height, width)
        domain_labels = torch.randint(0, num_domains, (batch_size,))
        
        pred, domain_pred, feat_orig, feat_style, contrast_labels = model.shuffle_style_assembly(x, domain_labels)
        
        assert pred.shape == (batch_size, 1, height//16, width//16)
        assert domain_pred.shape == (batch_size, num_domains)
        assert feat_orig.shape == feat_style.shape
        assert contrast_labels.shape == (batch_size,)

if __name__ == '__main__':
    pytest.main([__file__])