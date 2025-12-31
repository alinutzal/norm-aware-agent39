"""Model constructors"""

import torch.nn as nn
from torchvision.models import vision_transformer


def deit_tiny(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """DeiT-Tiny model for CIFAR-10"""
    model = vision_transformer.ViT_B_16(
        image_size=32,
        patch_size=4,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        attention_dropout=0.0,
        num_classes=num_classes,
    )
    return model
