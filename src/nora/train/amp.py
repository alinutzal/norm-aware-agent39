"""AMP (Automatic Mixed Precision) utilities"""

import torch


class AMPGuard:
    """Guard against NaN/Inf in AMP training"""

    def __init__(self, scaler: torch.cuda.amp.GradScaler):
        self.scaler = scaler

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check if loss is finite"""
        if not torch.isfinite(loss):
            return False
        return True

    def check_gradients(self, model: torch.nn.Module) -> bool:
        """Check if all gradients are finite"""
        for param in model.parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    return False
        return True
