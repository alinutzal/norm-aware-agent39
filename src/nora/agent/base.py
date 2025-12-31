"""Agent base interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Agent(ABC):
    """Base agent interface: observe → decide → act"""

    @abstractmethod
    def observe(self, state: Dict[str, Any]) -> None:
        """Observe current training state"""
        pass

    @abstractmethod
    def decide(self) -> Dict[str, Any]:
        """Decide what action to take"""
        pass

    @abstractmethod
    def act(self, action: Dict[str, Any]) -> None:
        """Execute action"""
        pass
