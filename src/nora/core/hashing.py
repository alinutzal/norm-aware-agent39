"""Hashing utilities for dataset, config, and environment"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def hash_dict(d: Dict[str, Any]) -> str:
    """Compute hash of a dictionary"""
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def hash_file(filepath: Path) -> str:
    """Compute hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:8]


def get_env_hash() -> str:
    """Get hash of Python environment (versions of key packages)"""
    import torch
    import torchvision

    env_info = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    return hash_dict(env_info)
