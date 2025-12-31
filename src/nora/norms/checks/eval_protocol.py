"""Eval protocol norm checks"""

import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def check_eval_mode(model: nn.Module) -> bool:
    """Check if model is in eval mode"""
    if model.training:
        logger.error("Model is in training mode during evaluation")
        return False
    return True


def check_train_mode(model: nn.Module) -> bool:
    """Check if model is in train mode"""
    if not model.training:
        logger.error("Model is in eval mode during training")
        return False
    return True
