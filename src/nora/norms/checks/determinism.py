"""Determinism norm checks"""

import torch
import logging

logger = logging.getLogger(__name__)


def check_deterministic_seed() -> bool:
    """Check if seed was set"""
    try:
        seed = torch.initial_seed()
        if seed >= 0:
            logger.info(f"Deterministic seed detected: {seed}")
            return True
    except Exception as e:
        logger.error(f"Failed to check seed: {e}")
    return False


def check_deterministic_algorithms() -> bool:
    """Check if deterministic algorithms are enabled"""
    if hasattr(torch, "_deterministic"):
        is_deterministic = torch._deterministic
        logger.info(f"Deterministic mode: {is_deterministic}")
        return is_deterministic
    return False
