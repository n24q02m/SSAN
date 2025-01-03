import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Binary cross entropy loss for live/spoof classification"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (B, 1)
            target: Binary labels (B,) with 0=spoof, 1=live
        """
        pred = pred.view(-1)  # Flatten prediction
        target = target.float().view(-1)  # Flatten target and convert to float
        return self.criterion(pred, target)


class ContrastiveLoss(nn.Module):
    """Contrastive loss between stylized features"""

    def __init__(self):
        super().__init__()

    def forward(self, anchor_feat, positive_feat, labels):
        """
        Args:
            anchor_feat: Original features (B, C, H, W)
            positive_feat: Shuffled style features (B, C, H, W)
            labels: Binary labels (B,) with 1=same class, -1=different class
        """
        # Stop gradient for anchor features
        anchor_feat = anchor_feat.detach()

        # Global average pooling to get feature vectors
        anchor_feat = (
            F.adaptive_avg_pool2d(anchor_feat, 1).squeeze(-1).squeeze(-1)
        )  # [B,C]
        positive_feat = (
            F.adaptive_avg_pool2d(positive_feat, 1).squeeze(-1).squeeze(-1)
        )  # [B,C]

        # Normalize features
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        positive_feat = F.normalize(positive_feat, p=2, dim=1)

        # Cosine similarity
        similarity = F.cosine_similarity(anchor_feat, positive_feat, dim=1)

        # Contrastive loss
        loss = -(similarity * labels.float())

        return loss.mean()


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss with gradient reversal"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_pred, domain_target):
        """
        Args:
            domain_pred: Domain classifier predictions (B, num_domains)
            domain_target: Domain labels (B,)
        """
        return self.criterion(domain_pred, domain_target)


class SSANLoss(nn.Module):
    """Combined loss for SSAN training"""

    def __init__(self, lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.cls_criterion = ClassificationLoss()
        self.contrast_criterion = ContrastiveLoss()
        self.domain_criterion = DomainAdversarialLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self,
        cls_pred,
        feat_orig,
        feat_style,
        contrast_labels,
        domain_pred,
        live_spoof_labels,
        domain_labels,
    ):
        """
        Args:
            cls_pred: Live/spoof predictions
            feat_orig: Original features
            feat_style: Stylized features
            contrast_labels: Labels for contrastive loss
            domain_pred: Domain predictions
            domain_target: Domain labels
        """
        # Classification loss using live/spoof labels
        cls_loss = self.cls_criterion(cls_pred, live_spoof_labels)

        # Contrastive loss
        contrast_loss = self.contrast_criterion(feat_orig, feat_style, contrast_labels)

        # Domain adversarial loss
        domain_loss = self.domain_criterion(domain_pred, domain_labels)

        # Total loss
        total_loss = (
            cls_loss + self.lambda1 * contrast_loss + self.lambda2 * domain_loss
        )

        return total_loss, {
            "cls_loss": cls_loss,
            "contrast_loss": contrast_loss,
            "domain_loss": domain_loss,
        }
