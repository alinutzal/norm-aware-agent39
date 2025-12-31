"""Tests for determinism features"""

import pytest
import torch
from nora.core.reproducibility import set_seed, get_seed


def test_set_seed_deterministic():
    """Test that seed setting works"""
    set_seed(42, deterministic=True)
    seed1 = get_seed()
    
    set_seed(42, deterministic=True)
    seed2 = get_seed()
    
    assert seed1 == seed2


def test_different_seeds():
    """Test that different seeds produce different values"""
    set_seed(42, deterministic=True)
    x1 = torch.randn(10)
    
    set_seed(43, deterministic=True)
    x2 = torch.randn(10)
    
    assert not torch.allclose(x1, x2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
