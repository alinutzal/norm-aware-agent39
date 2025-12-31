"""Augmentation leak norm checks"""

import logging

logger = logging.getLogger(__name__)


def check_no_augmentation_on_eval(transform) -> bool:
    """Check that eval transform doesn't include augmentation"""
    # This is a simple check - in practice would need to inspect transform composition
    transform_str = str(transform)
    
    augmentation_keywords = ["crop", "flip", "rotate", "affine", "colorjitter"]
    for keyword in augmentation_keywords:
        if keyword.lower() in transform_str.lower():
            logger.error(f"Augmentation detected in eval transform: {keyword}")
            return False
    
    return True
