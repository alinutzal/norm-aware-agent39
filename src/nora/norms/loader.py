"""Norm loading from YAML files"""

import logging
from pathlib import Path
from typing import Dict, Any

import yaml

from .schema import Norm

logger = logging.getLogger(__name__)


def load_norms(norms_dir: str = "norms") -> Dict[str, Norm]:
    """Load all norms from norms directory"""
    norms_path = Path(norms_dir)
    norms = {}

    if not norms_path.exists():
        logger.warning(f"Norms directory not found: {norms_path}")
        return norms

    # Load norm definitions from YAML files
    for yaml_file in norms_path.glob("*.yaml"):
        if yaml_file.name == "registry.yaml":
            continue

        with open(yaml_file, "r") as f:
            norm_defs = yaml.safe_load(f) or {}

        for norm_name, norm_def in norm_defs.items():
            norm = Norm(
                name=norm_name,
                description=norm_def.get("description", ""),
                check=norm_def.get("check", ""),
                severity=norm_def.get("severity", "medium"),
                auto_fix=norm_def.get("auto_fix", False),
                suggested_fix=norm_def.get("suggested_fix", ""),
            )
            norms[norm_name] = norm
            logger.debug(f"Loaded norm: {norm_name}")

    return norms


def load_registry(registry_path: str = "norms/registry.yaml") -> Dict[str, Any]:
    """Load norm registry"""
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}
    return registry
