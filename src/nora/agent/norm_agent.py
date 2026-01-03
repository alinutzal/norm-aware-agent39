"""Agent-based norm enforcement and remediation"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

import torch
from omegaconf import OmegaConf

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
        norm_aware: bool = True,
    ):
        self.regime = regime  # balanced, strict, exploratory
        self.mode = mode  # pipeline, agentic, norm_aware
        self.norm_aware = norm_aware  # True for norm_aware, False for agentic
        self.cfg = cfg
        self.events_file = events_file
        self.decisions: list = []
        self.remediation_count = 0
        self.halt_count = 0
        
        # Performance tracking for agentic mode
        self.performance_history = []
        self.lr_adjustments = 0
        self.early_stop_recommended = False

        # Regime-specific behavior
        self.strategy = self._get_strategy(regime)
        mode_type = "norm-aware" if norm_aware else "performance-optimizing"
        logger.info(f"Initialized NormAgent: regime={regime}, mode={mode} ({mode_type}), strategy={self.strategy}")

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
            "excessive_batch_size": {
                "action": "reduce_batch_size",
                "parameters": {
                    "batch_size": 512,
                },
                "description": "Reduce batch size to 512 to save memory",
            },
            "inefficient_precision": {
                "action": "enable_amp",
                "parameters": {
                    "amp_enabled": True,
                    "dtype": "fp16",
                },
                "description": "Enable mixed precision training for efficiency",
            },
            "excessive_workers": {
                "action": "reduce_workers",
                "parameters": {
                    "num_workers": 8,
                },
                "description": "Reduce dataloader workers to 8",
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
                # Use bracket notation which works for both dict and OmegaConf
                cfg["seed"] = suggested_fix["parameters"]["seed"]
                return True, "Restored seed"

            elif violation_name == "nondeterminism_enabled":
                if "repro" in cfg:
                    cfg["repro"]["deterministic"] = True
                    cfg["repro"]["cudnn_benchmark"] = False
                    cfg["repro"]["dataloader_seed_workers"] = True
                return True, "Restored determinism"

            elif violation_name == "untracked_config_change":
                if "checkpoint" in cfg:
                    cfg["checkpoint"]["include_config"] = True
                    cfg["checkpoint"]["include_overrides"] = True
                return True, "Enabled config persistence"

            elif violation_name == "amp_nan":
                if "amp" in cfg:
                    cfg["amp"]["grad_scaler"] = True
                if "scheduler" in cfg:
                    cfg["scheduler"]["warmup_epochs"] = 5
                if "amp" in cfg:
                    cfg["amp"]["loss_nan_policy"] = "halt"
                return True, "Fixed AMP configuration"

            elif violation_name == "excessive_batch_size":
                if "train" in cfg:
                    cfg["train"]["batch_size"] = 512
                return True, "Reduced batch size to 512"

            elif violation_name == "inefficient_precision":
                if "amp" in cfg:
                    cfg["amp"]["enabled"] = True
                cfg["dtype"] = "fp16"
                return True, "Enabled mixed precision training"

            elif violation_name == "excessive_workers":
                if "dataset" in cfg:
                    cfg["dataset"]["num_workers"] = 8
                return True, "Reduced dataloader workers to 8"

            elif violation_name == "eval_mode_bug":
                if "train" in cfg:
                    cfg["train"]["force_eval_mode"] = True
                return True, "Enabled model eval mode during validation"

            elif violation_name == "aug_leak":
                if "dataset" in cfg:
                    cfg["dataset"]["eval_augment"] = {
                        "random_crop": False,
                        "crop_padding": 0,
                        "hflip": False,
                    }
                return True, "Disabled augmentation on eval data"

            elif violation_name == "reporting_single_run":
                if "reporting" in cfg:
                    cfg["reporting"]["require_multi_seed"] = True
                if "metrics" in cfg:
                    cfg["metrics"]["report_mean_std"] = True
                return True, "Enabled multi-seed reporting with uncertainty"

            elif violation_name == "checkpoint_incomplete":
                if "checkpoint" in cfg:
                    cfg["checkpoint"]["include_optimizer"] = True
                    cfg["checkpoint"]["include_scheduler"] = True
                    cfg["checkpoint"]["include_scaler"] = True
                return True, "Enabled complete checkpoint saving"

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

        metrics = {
            "total_decisions": total_decisions,
            "auto_fixes": auto_fixes,
            "halts": halts,
            "warnings": warnings,
            "remediation_success_count": self.remediation_count,
            "regime": self.regime,
            "mode": self.mode,
            "norm_aware": self.norm_aware,
        }
        
        # Add agentic-specific metrics
        if not self.norm_aware:
            metrics.update({
                "lr_adjustments": self.lr_adjustments,
                "early_stop_recommended": self.early_stop_recommended,
                "epochs_observed": len(self.performance_history),
            })
        
        return metrics

    def observe_training_state(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """
        Observe training state for agentic mode.
        Called after each epoch to track performance.
        """
        if self.norm_aware:
            # Norm-aware mode doesn't need runtime observation
            return
        
        state = {
            "epoch": epoch,
            "train_loss": train_metrics.get("train_loss", float("inf")),
            "train_acc": train_metrics.get("train_acc", 0.0),
            "val_loss": val_metrics.get("val_loss", float("inf")),
            "val_acc": val_metrics.get("val_acc", 0.0),
        }
        self.performance_history.append(state)
        logger.debug(f"Agentic agent observed epoch {epoch}: val_loss={state['val_loss']:.4f}, val_acc={state['val_acc']:.2f}%")

    def decide_training_action(self, current_lr: float) -> Dict[str, Any]:
        """
        Make performance-based training decisions (agentic mode only).
        Returns action to take (continue, adjust_lr, early_stop).
        """
        if self.norm_aware or len(self.performance_history) < 2:
            return {"action": "continue"}
        
        current = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        # Check for training issues
        decision = {"action": "continue", "reason": None}
        
        # 1. Check for loss explosion
        if current["train_loss"] > 100.0 or current["val_loss"] > 100.0:
            decision = {
                "action": "reduce_lr",
                "factor": 0.5,
                "reason": "Loss explosion detected",
            }
            self.lr_adjustments += 1
            logger.warning(f"Agentic: Loss explosion (train={current['train_loss']:.2f}, val={current['val_loss']:.2f})")
        
        # 2. Check for validation loss increase (potential overfitting)
        elif len(self.performance_history) >= 3:
            recent_val_losses = [h["val_loss"] for h in self.performance_history[-3:]]
            if all(recent_val_losses[i] < recent_val_losses[i+1] for i in range(len(recent_val_losses)-1)):
                if self.regime == "strict":
                    decision = {
                        "action": "early_stop",
                        "reason": "Validation loss increasing for 3 consecutive epochs",
                    }
                    self.early_stop_recommended = True
                    logger.warning("Agentic (strict): Early stop recommended - overfitting detected")
                else:
                    decision = {
                        "action": "reduce_lr",
                        "factor": 0.5,
                        "reason": "Validation loss increasing - attempting LR reduction",
                    }
                    self.lr_adjustments += 1
        
        # 3. Check for stagnation (loss not decreasing)
        elif len(self.performance_history) >= 5:
            recent_val_losses = [h["val_loss"] for h in self.performance_history[-5:]]
            loss_variance = max(recent_val_losses) - min(recent_val_losses)
            if loss_variance < 0.01:  # Very little change
                decision = {
                    "action": "reduce_lr",
                    "factor": 0.1,
                    "reason": "Loss stagnation detected",
                }
                self.lr_adjustments += 1
                logger.info("Agentic: Loss stagnation - reducing LR aggressively")
        
        # Record decision
        if decision["action"] != "continue":
            agent_decision = AgentDecision(
                action=decision["action"],
                severity="medium",
                message=f"Agentic decision: {decision['action']} - {decision.get('reason', 'performance optimization')}",
                suggested_fix=decision,
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            self.decisions.append(agent_decision)
            self._record_agentic_decision(agent_decision, current)
        
        return decision

    def _record_agentic_decision(self, decision: AgentDecision, state: Dict[str, Any]) -> None:
        """Record agentic decision to events file"""
        if not self.events_file:
            return

        event = {
            "timestamp": decision.timestamp,
            "event_type": "AGENTIC_DECISION",
            "agent_action": decision.action,
            "regime": self.regime,
            "mode": self.mode,
            "message": decision.message,
            "suggested_fix": decision.suggested_fix,
            "training_state": state,
        }

        Path(self.events_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
