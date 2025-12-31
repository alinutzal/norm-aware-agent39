"""Agent-based norm enforcement and remediation"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

import torch

from ..norms.reproducibility_monitor import ReproducibilityNormMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """Agent's decision on how to handle a violation"""
    action: str  # "halt", "warn", "suggest_fix", "auto_fix", "continue"
    severity: str  # "low", "medium", "high"
    message: str
    suggested_fix: Optional[Dict[str, Any]] = None
    timestamp: str = ""


class NormAgent:
    """
    Agent for detecting, deciding on, and remediating norm violations.
    Behavior varies by regime (balanced vs strict).
    """

    def __init__(
        self,
        regime: str = "balanced",
        mode: str = "norm_aware",
        events_file: str = None,
        cfg: Any = None,
    ):
        self.regime = regime  # balanced, strict, exploratory
        self.mode = mode  # pipeline, agentic, norm_aware
        self.cfg = cfg
        self.events_file = events_file
        self.decisions: list = []
        self.remediation_count = 0
        self.halt_count = 0

        # Regime-specific behavior
        self.strategy = self._get_strategy(regime)
        logger.info(f"Initialized NormAgent: regime={regime}, mode={mode}, strategy={self.strategy}")

    def _get_strategy(self, regime: str) -> Dict[str, Any]:
        """Define agent behavior for each regime"""
        strategies = {
            "strict": {
                "halt_on_violations": True,
                "auto_fix": False,
                "suggest_fixes": True,
                "log_level": "error",
                "require_validation": True,
                "violation_tolerance": 0,  # No tolerance
                "max_retries": 1,
            },
            "balanced": {
                "halt_on_violations": False,
                "auto_fix": True,
                "suggest_fixes": True,
                "log_level": "warning",
                "require_validation": False,
                "violation_tolerance": 1,  # Allow 1 low-severity violation
                "max_retries": 3,
            },
            "exploratory": {
                "halt_on_violations": False,
                "auto_fix": False,
                "suggest_fixes": True,
                "log_level": "info",
                "require_validation": False,
                "violation_tolerance": 99,  # Very tolerant
                "max_retries": 0,
            },
        }
        return strategies.get(regime, strategies["balanced"])

    def decide_on_violation(
        self,
        violation_name: str,
        violation_severity: str,
        evidence: Dict[str, Any],
    ) -> AgentDecision:
        """
        Decide how to respond to a detected violation.
        Returns: AgentDecision (action, message, suggested fix)
        """
        decision = self._determine_action(
            violation_name, violation_severity, evidence
        )
        self.decisions.append(decision)

        # Log decision
        self._record_decision(decision, violation_name, violation_severity, evidence)

        return decision

    def _determine_action(
        self,
        violation_name: str,
        violation_severity: str,
        evidence: Dict[str, Any],
    ) -> AgentDecision:
        """Determine action based on regime and violation severity"""

        if self.strategy["halt_on_violations"]:
            # Strict: halt on any violation
            action = "halt"
            message = f"STRICT: Halting training due to {violation_severity} violation: {violation_name}"
            self.halt_count += 1
        elif self.strategy["auto_fix"]:
            # Balanced: auto-fix with validation
            action = "auto_fix"
            message = f"BALANCED: Auto-fixing {violation_name}, will validate"
            suggested_fix = self._suggest_fix(violation_name, evidence)
            self.remediation_count += 1
        else:
            # Exploratory: warn only
            action = "warn"
            message = f"EXPLORATORY: Warning about {violation_name}, continuing"

        return AgentDecision(
            action=action,
            severity=violation_severity,
            message=message,
            suggested_fix=self._suggest_fix(violation_name, evidence) if action in ["auto_fix", "suggest_fix"] else None,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    def _suggest_fix(self, violation_name: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest a fix for a violation"""
        fixes = {
            "missing_seed": {
                "action": "set_seed",
                "parameters": {
                    "seed": 42,
                    "deterministic": True,
                    "cudnn_benchmark": False,
                },
                "description": "Set seed to 42 and enable deterministic mode",
            },
            "nondeterminism_enabled": {
                "action": "restore_determinism",
                "parameters": {
                    "deterministic": True,
                    "cudnn_benchmark": False,
                    "dataloader_seed_workers": True,
                },
                "description": "Disable cuDNN benchmark and restore deterministic algorithms",
            },
            "untracked_config_change": {
                "action": "persist_config",
                "parameters": {
                    "include_config": True,
                    "include_overrides": True,
                },
                "description": "Save config and overrides to checkpoint artifacts",
            },
            "amp_nan": {
                "action": "fix_amp",
                "parameters": {
                    "grad_scaler": True,
                    "warmup_epochs": 5,
                    "loss_nan_policy": "halt",
                },
                "description": "Enable GradScaler, restore warmup, and halt on NaN",
            },
            "eval_mode_bug": {
                "action": "fix_eval_mode",
                "parameters": {
                    "force_eval_mode": True,
                },
                "description": "Ensure model.eval() is called during evaluation",
            },
            "aug_leak": {
                "action": "fix_augmentation",
                "parameters": {
                    "eval_augment_random": False,
                },
                "description": "Disable stochastic augmentation on eval set",
            },
            "checkpoint_incomplete": {
                "action": "fix_checkpoint",
                "parameters": {
                    "include_optimizer": True,
                    "include_scheduler": True,
                    "include_scaler": True,
                },
                "description": "Save complete checkpoint state",
            },
        }
        return fixes.get(violation_name, {"description": "No standard fix available"})

    def _record_decision(
        self,
        decision: AgentDecision,
        violation_name: str,
        violation_severity: str,
        evidence: Dict[str, Any],
    ) -> None:
        """Record agent decision to events file"""
        if not self.events_file:
            return

        event = {
            "timestamp": decision.timestamp,
            "event_type": "AGENT_DECISION",
            "agent_action": decision.action,
            "violation_name": violation_name,
            "violation_severity": violation_severity,
            "regime": self.regime,
            "mode": self.mode,
            "message": decision.message,
            "suggested_fix": decision.suggested_fix,
            "evidence": evidence,
        }

        Path(self.events_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def remediate(self, cfg: Any, violation_name: str) -> Tuple[bool, str]:
        """
        Attempt remediation for a violation.
        Returns: (success, message)
        """
        if self.regime == "strict":
            return False, "Strict regime: no remediation attempted"

        # For balanced/exploratory, attempt fix
        suggested_fix = self._suggest_fix(violation_name, {})
        
        try:
            if violation_name == "missing_seed":
                cfg.seed = suggested_fix["parameters"]["seed"]
                cfg.repro.deterministic = True
                cfg.repro.cudnn_benchmark = False
                return True, "Restored seed and determinism"

            elif violation_name == "nondeterminism_enabled":
                cfg.repro.deterministic = True
                cfg.repro.cudnn_benchmark = False
                cfg.repro.dataloader_seed_workers = True
                return True, "Restored determinism"

            elif violation_name == "untracked_config_change":
                cfg.checkpoint.include_config = True
                cfg.checkpoint.include_overrides = True
                return True, "Enabled config persistence"

            elif violation_name == "amp_nan":
                cfg.amp.grad_scaler = True
                cfg.scheduler.warmup_epochs = 5
                cfg.amp.loss_nan_policy = "halt"
                return True, "Fixed AMP configuration"

            return False, f"No remediation available for {violation_name}"

        except Exception as e:
            return False, f"Remediation failed: {e}"

    def should_halt(self) -> bool:
        """Determine if training should halt"""
        if self.regime == "strict":
            return self.halt_count > 0
        elif self.regime == "balanced":
            return self.halt_count > 3  # Allow up to 3 halts before actually halting
        else:
            return False

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute agent performance metrics"""
        total_decisions = len(self.decisions)
        auto_fixes = len([d for d in self.decisions if d.action == "auto_fix"])
        halts = len([d for d in self.decisions if d.action == "halt"])
        warnings = len([d for d in self.decisions if d.action == "warn"])

        return {
            "total_decisions": total_decisions,
            "auto_fixes": auto_fixes,
            "halts": halts,
            "warnings": warnings,
            "remediation_success_count": self.remediation_count,
            "regime": self.regime,
            "mode": self.mode,
        }
