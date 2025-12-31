"""Checkpoint completeness norm checks"""

from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


def check_checkpoint_complete(checkpoint_path: Path) -> bool:
    """Check if checkpoint has all required components"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
        }
        
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            logger.error(f"Checkpoint missing keys: {missing_keys}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to validate checkpoint: {e}")
        return False
