"""Reactive agent without norm awareness"""

import logging
from typing import Any, Dict

from .base import Agent

logger = logging.getLogger(__name__)


class ReactiveAgent(Agent):
    """Agent that reacts to training state without norm awareness"""

    def __init__(self):
        self.state = None

    def observe(self, state: Dict[str, Any]) -> None:
        """Observe current training state"""
        self.state = state
        logger.debug(f"Agent observed: {state}")

    def decide(self) -> Dict[str, Any]:
        """Decide what action to take based on metrics"""
        if self.state is None:
            return {"action": "continue"}

        # Simple heuristics without norm awareness
        if self.state.get("loss") > 10.0:
            return {"action": "reduce_lr", "factor": 0.5}

        return {"action": "continue"}

    def act(self, action: Dict[str, Any]) -> None:
        """Execute action"""
        action_type = action.get("action")
        logger.info(f"Agent executing action: {action_type}")
