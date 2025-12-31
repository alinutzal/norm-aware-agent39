"""Tests for violation injection"""

import pytest
from nora.violations.injector import ViolationInjector
from nora.violations.profiles import ViolationProfile


def test_violation_injector_initialization():
    """Test violation injector setup"""
    config = {
        "enabled": True,
        "active": ["nondeterminism", "eval_mode_bug"],
    }
    injector = ViolationInjector(config)
    
    assert injector.is_violation_active("nondeterminism")
    assert injector.is_violation_active("eval_mode_bug")
    assert not injector.is_violation_active("amp_nan")


def test_violation_profile():
    """Test violation profile behavior"""
    params = {
        "seed_randomization": True,
        "skip_eval_mode_set": False,
        "augment_eval_data": True,
    }
    profile = ViolationProfile("test", params)
    
    assert profile.should_randomize_seed()
    assert not profile.should_skip_eval_mode()
    assert profile.should_augment_eval()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
