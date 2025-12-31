"""Hook points for norm checking"""

from typing import Callable, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HookManager:
    """Manages hook registration and execution"""

    def __init__(self):
        self.hooks: Dict[str, List[Callable]] = {}

    def register(self, hook_name: str, callback: Callable) -> None:
        """Register callback for hook point"""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
        logger.debug(f"Registered hook: {hook_name}")

    def execute(self, hook_name: str, *args, **kwargs) -> None:
        """Execute all callbacks for hook point"""
        if hook_name not in self.hooks:
            return

        logger.debug(f"Executing hook: {hook_name}")
        for callback in self.hooks[hook_name]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")


# Standard hook points
HOOK_POINTS = [
    "on_start",
    "on_epoch_start",
    "on_train_start",
    "on_batch_start",
    "on_forward_pass",
    "on_backward_pass",
    "on_optimizer_step",
    "on_batch_end",
    "on_train_end",
    "on_eval_start",
    "on_eval_batch",
    "on_eval_end",
    "on_epoch_end",
    "on_checkpoint_save",
    "on_log",
    "on_end",
]
