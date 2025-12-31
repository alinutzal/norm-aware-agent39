"""Model constructors"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def deit_tiny(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """DeiT-Tiny model for CIFAR-10"""
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    
    # Adapt for CIFAR-10: modify for 32x32 images with 4x4 patches
    model.image_size = 32
    model.patch_size = 4
    model.conv_proj = nn.Conv2d(
        3, model.hidden_dim, kernel_size=4, stride=4, padding=0
    )
    
    # Adjust positional embeddings for 8x8 patches (32/4 = 8)
    # Original has 14x14 patches (224/16 = 14), so 196 patches + 1 class token = 197
    # We need 8x8 = 64 patches + 1 class token = 65
    num_patches = (32 // 4) ** 2  # 64
    new_pos_embedding = nn.Parameter(
        torch.zeros(1, num_patches + 1, model.hidden_dim)
    )
    # Initialize with truncated normal
    nn.init.trunc_normal_(new_pos_embedding, std=0.02)
    model.encoder.pos_embedding = new_pos_embedding
    
    # Modify classification head for custom num_classes
    model.heads.head = nn.Linear(model.hidden_dim, num_classes)
    
    return model
