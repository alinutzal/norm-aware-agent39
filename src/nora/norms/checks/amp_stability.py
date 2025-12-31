"""AMP stability norm checks"""

import torch
import logging

logger = logging.getLogger(__name__)


def check_loss_finite(loss: torch.Tensor) -> bool:
    """Check if loss is finite"""
    is_finite = torch.isfinite(loss).all().item()
    if not is_finite:
        logger.error("Loss contains NaN or Inf")
    return is_finite


def check_gradients_finite(model: torch.nn.Module) -> bool:
    """Check if all gradients are finite"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                logger.error(f"Gradient NaN/Inf in {name}")
                return False
    return True
