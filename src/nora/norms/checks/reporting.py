"""Reporting norm checks"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_metric_consistency(metrics: Dict[str, Any]) -> bool:
    """Check that metrics have consistent structure"""
    required_metrics = {"train_loss", "train_acc", "val_loss", "val_acc"}
    
    missing = required_metrics - set(metrics.keys())
    if missing:
        logger.warning(f"Missing expected metrics: {missing}")
        return False
    
    return True


def check_log_format(log_entry: Dict[str, Any]) -> bool:
    """Check that log entry is valid JSON with required fields"""
    required_fields = {"timestamp", "level", "message"}
    
    missing = required_fields - set(log_entry.keys())
    if missing:
        logger.error(f"Log entry missing fields: {missing}")
        return False
    
    return True
