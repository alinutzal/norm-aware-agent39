"""Norm-aware agent orchestration (NORA)"""

import logging
from typing import Any, Dict

from .base import Agent
from ..norms.loader import load_norms

logger = logging.getLogger(__name__)


class NormAwareAgent(Agent):
    """Agent that makes decisions based on norm violations"""

    def __init__(self, norms_config_path: str):
        self.state = None
        self.norms = load_norms(norms_config_path)
        logger.info(f"Loaded {len(self.norms)} norms")

    def observe(self, state: Dict[str, Any]) -> None:
        """Observe current training state"""
        self.state = state
        logger.debug(f"NORA agent observed: {state}")

    def decide(self) -> Dict[str, Any]:
        """Decide actions based on norm violations"""
        if self.state is None:
            return {"action": "continue"}

        violations = []
        remediation = []

        # Check norms against current state
        for norm_name, norm_def in self.norms.items():
            # Placeholder: actual norm checking logic
            logger.debug(f"Checking norm: {norm_name}")

        return {
            "action": "continue",
            "violations": violations,
            "remediation": remediation,
        }

    def act(self, action: Dict[str, Any]) -> None:
        """Execute action and remediation"""
        action_type = action.get("action")
        logger.info(f"NORA executing action: {action_type}")

        if action.get("remediation"):
            logger.info(f"Applying remediation: {action['remediation']}")
