import torch
import torch.nn as nn


class FeatureGenerator(nn.Module):
    """Base CNN network to extract multi-scale features"""

    def __init__(self, input_channels=3):
        super().__init__()
        # Initial conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 3 main blocks with pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 196, 3, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            nn.Conv2d(196, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 196, 3, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            nn.Conv2d(196, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 196, 3, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            nn.Conv2d(196, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Final conv layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Get features from multiple scales
        x = self.conv1(x)  # Base features

        x1 = self.block1(x)  # First scale
        x2 = self.block2(x1)  # Second scale
        x3 = self.block3(x2)  # Third scale

        x4 = self.layer4(x3)  # Final features

        return x4, [x1, x2, x3]  # Return final features and intermediate features


class ContentExtractor(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class StyleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True))

        self.gamma = nn.Linear(256, 256)
        self.beta = nn.Linear(256, 256)

    def forward(self, features):
        x = features[0]  # Take first scale features
        x = self.ada_conv1(x) + features[1]
        x = self.ada_conv2(x) + features[2]
        x = self.ada_conv3(x)

        # Global pooling and FC
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Get AdaIN parameters
        gamma = self.gamma(x)
        beta = self.beta(x)

        return gamma, beta


class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, gamma, beta):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.decoder(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, num_domains, max_iter=4000):
        super().__init__()
        self.grl = GradientReversalLayer(max_iter)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(nn.Linear(512, num_domains), nn.ReLU(inplace=True))

    def forward(self, x, lambda_val):
        x = self.grl(x) * lambda_val
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer"""

    def __init__(self, max_iter):
        super().__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        self.iter_num += 1
        return x

    def backward(self, grad_output):
        coeff = (
            2.0 * (self.high - self.low) * self.iter_num / self.max_iter
            - (self.high - self.low)
            + self.low
        )
        return -coeff * grad_output
