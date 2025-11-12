from baseline import build_resnet50
import torch
from torchvision import models

"""
baseline.py

ResNet-50 model builder for PyTorch.

Usage:
    model = build_resnet50(num_classes=10, pretrained=True, freeze_backbone=False, dropout=0.5, in_channels=3)
"""

import torch.nn as nn


class BaselineResNet50(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 pretrained=True,
                 freeze_backbone=False,
                 dropout=0.0,
                 in_channels=3,
                 use_batchnorm=False):
        super().__init__()
        self.in_channels = in_channels
        # Load torchvision ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # If input channels differ from 3, adapt first conv
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize new conv weights (copy or kaiming)
            if pretrained:
                # Repeat or average the weights from RGB channels
                with torch.no_grad():
                    if in_channels < 3:
                        # average RGB weights to fewer channels
                        self.backbone.conv1.weight[:] = old_conv.weight[:, :in_channels, :, :].mean(dim=1, keepdim=True)
                    else:
                        # repeat weights to fill extra channels
                        repeats = (in_channels + 2) // 3
                        w = old_conv.weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                        self.backbone.conv1.weight[:] = w
            else:
                # default init (kaiming)
                nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier (fc)
        in_features = self.backbone.fc.in_features
        head = []
        if dropout and dropout > 0.0:
            head.append(nn.Dropout(p=dropout))
        if use_batchnorm:
            head.append(nn.BatchNorm1d(in_features))
        head.append(nn.Linear(in_features, num_classes))
        self.backbone.fc = nn.Sequential(*head)

    def forward(self, x):
        return self.backbone(x)


def build_resnet50(num_classes=1000,
                   pretrained=True,
                   freeze_backbone=False,
                   dropout=0.0,
                   in_channels=3,
                   use_batchnorm=False,
                   device=None):
    """
    Convenience factory. Returns a BaselineResNet50 on the requested device.
    """
    model = BaselineResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        in_channels=in_channels,
        use_batchnorm=use_batchnorm
    )
    if device:
        model.to(device)
    return model


if __name__ == "__main__":
    # Quick smoke test
    model = build_resnet50(num_classes=10, pretrained=False, dropout=0.5)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # expect (2, 10)