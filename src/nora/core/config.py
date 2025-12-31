"""Core configuration loading and merging"""

from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration from file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override config into base config"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration against schema"""
    # Schema validation would go here
    return True
