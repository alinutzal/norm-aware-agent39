"""Violation injection for testing"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ViolationInjector:
    """Applies violation profiles to runtime"""

    def __init__(self, violation_config: Dict[str, Any]):
        self.config = violation_config
        self.active_violations = violation_config.get("active", [])
        logger.info(f"Violations to inject: {self.active_violations}")

    def is_violation_active(self, violation_name: str) -> bool:
        """Check if violation is active"""
        return violation_name in self.active_violations

    def get_violation_params(self, violation_name: str) -> Dict[str, Any]:
        """Get parameters for violation"""
        return self.config.get(violation_name, {})
