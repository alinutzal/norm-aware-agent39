"""Violation profiles: mapping YAML → concrete toggles/actions"""

from typing import Dict, Any


class ViolationProfile:
    """Encapsulates violation behavior"""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def should_randomize_seed(self) -> bool:
        """Should seed be randomized (nondeterminism violation)"""
        return self.params.get("seed_randomization", False)

    def should_skip_eval_mode(self) -> bool:
        """Should eval mode setting be skipped"""
        return self.params.get("skip_eval_mode_set", False)

    def should_augment_eval(self) -> bool:
        """Should augmentation be applied to eval set"""
        return self.params.get("augment_eval_data", False)

    def should_skip_optimizer_state(self) -> bool:
        """Should optimizer state be skipped in checkpoint"""
        return self.params.get("skip_optimizer_state", False)

    def should_inject_nans(self) -> bool:
        """Should NaNs be injected for AMP testing"""
        return self.params.get("inject_nans", False)
