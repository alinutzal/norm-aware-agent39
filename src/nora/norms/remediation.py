"""Remediation suggestions and auto-fix policies"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class RemediationPolicy:
    """Suggests and applies fixes for norm violations"""

    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix

    def suggest_fix(self, violation: str) -> str:
        """Suggest fix for violation"""
        suggestions = {
            "deterministic_seed": "Call set_seed() at training start",
            "deterministic_algorithms": "Call torch.use_deterministic_algorithms(True)",
            "eval_mode_bug": "Call model.eval() before evaluation loop",
            "aug_leak": "Remove augmentation from eval transforms",
            "checkpoint_incomplete": "Save optimizer and scheduler state",
            "amp_nan": "Check loss scaling and add NaN guards",
        }
        return suggestions.get(violation, "Unknown violation")

    def auto_fix_available(self, violation: str) -> bool:
        """Check if violation has auto-fix capability"""
        auto_fixable = {
            "deterministic_seed",
            "deterministic_algorithms",
            "eval_mode_bug",
            "train_mode_bug",
            "checkpoint_incomplete",
        }
        return violation in auto_fixable
