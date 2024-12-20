import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (FeatureGenerator, ContentExtractor, StyleExtractor, 
                        AdaIN, Classifier, DomainDiscriminator)

class SSAN(nn.Module):
    """Shuffled Style Assembly Network (SSAN) for Face Anti-Spoofing"""
    def __init__(self, num_domains=5, ada_blocks=2, max_iter=4000):
        """
        Args:
            num_domains: Number of source domains
            ada_blocks: Number of AdaIN residual blocks
            max_iter: Maximum iterations for GRL
        """
        super().__init__()
        
        # Base feature generator
        self.feature_gen = FeatureGenerator(input_channels=3)

        # Content extractor
        self.content_extractor = ContentExtractor(in_channels=256)

        # Style extractor 
        self.style_extractor = StyleExtractor()

        # AdaIN residual blocks
        self.ada_blocks = nn.ModuleList([
            AdaIN(eps=1e-5) for _ in range(ada_blocks)
        ])

        # Final conv after AdaIN blocks
        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        # Classification head
        self.classifier = Classifier(in_channels=512)

        # Domain discriminator
        self.domain_disc = DomainDiscriminator(num_domains, max_iter)

    def forward(self, x, domain_labels=None, return_feats=False):
        """
        Args:
            x: Input images (B, C, H, W)
            domain_labels: Domain labels for discriminator (B,)
            return_feats: Whether to return intermediate features
        """
        # Extract multi-scale features
        feat_final, feat_scales = self.feature_gen(x)

        # Extract content features
        content_feat = self.content_extractor(feat_final)

        # Extract style features
        gamma, beta = self.style_extractor(feat_scales)

        # Style assembly through AdaIN blocks
        style_feat = content_feat
        for ada_block in self.ada_blocks:
            style_feat = ada_block(style_feat, gamma, beta)

        # Final conv
        style_feat = self.conv_final(style_feat)

        # Classification
        live_spoof_pred = self.classifier(style_feat)

        # Domain discrimination
        if domain_labels is not None:
            domain_pred = self.domain_disc(style_feat)
        else:
            domain_pred = None

        if return_feats:
            return live_spoof_pred, domain_pred, {
                'content_feat': content_feat,
                'style_feat': style_feat,
                'gamma': gamma,
                'beta': beta
            }
        
        return live_spoof_pred, domain_pred

    def shuffle_style_assembly(self, x, domain_labels=None):
        """Forward pass with shuffled style features for contrastive learning"""
        # Get batch size
        batch_size = x.size(0)
        
        # Random permutation for style shuffling
        rand_idx = torch.randperm(batch_size)
        x_shuffled = x[rand_idx]

        # Get predictions and features for original images
        pred_orig, domain_pred, feats_orig = self.forward(x, domain_labels, return_feats=True)

        # Get predictions and features for shuffled images 
        _, _, feats_shuffled = self.forward(x_shuffled, domain_labels, return_feats=True)

        # Create contrastive labels (1 if same class, -1 if different)
        contrast_labels = (domain_labels == domain_labels[rand_idx]).long() 
        contrast_labels = torch.where(contrast_labels == 1, 1, -1)

        return pred_orig, domain_pred, feats_orig['style_feat'], feats_shuffled['style_feat'], contrast_labels