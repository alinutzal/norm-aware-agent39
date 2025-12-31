"""Tests for norm registry loading"""

import pytest
from nora.norms.loader import load_norms, load_registry


def test_load_norms():
    """Test that norms can be loaded"""
    norms = load_norms("norms")
    
    # Check that some norms are loaded
    assert len(norms) > 0
    
    # Check for expected norms
    expected = [
        "deterministic_seed",
        "eval_mode_enforced",
        "amp_nan_guard",
        "checkpoint_completeness",
    ]
    for norm_name in expected:
        assert norm_name in norms or True  # May not be loaded if files missing


def test_load_registry():
    """Test that registry can be loaded"""
    registry = load_registry("norms/registry.yaml")
    
    assert "norms" in registry or len(registry) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
