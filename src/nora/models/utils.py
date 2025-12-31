"""Model utility functions"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_groups(model: nn.Module):
    """Group layers for differential learning rates"""
    layer_groups = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_groups.append(module)
    return layer_groups
