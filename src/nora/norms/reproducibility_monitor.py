"""Reproducibility norm monitor and measurer"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np


@dataclass
class NormEvent:
    """Record of norm violation detection/remediation"""
    timestamp: str
    norm_name: str
    violation_name: str
    event_type: str  # "VIOLATION_DETECTED", "REMEDIATION_STARTED", "REMEDIATION_SUCCESS", "REMEDIATION_FAILED"
    severity: str  # "low", "medium", "high"
    time_to_detection_sec: float = 0.0
    message: str = ""
    evidence: Dict[str, Any] = None


class ReproducibilityNormMonitor:
    """
    Monitor and measure reproducibility violations.
    Tracks: detection rate, time-to-intervention, remediation success, severity scores.
    """

    def __init__(self, events_file: str, cfg: Any = None):
        self.events_file = events_file
        self.cfg = cfg
        self.events: List[NormEvent] = []
        self.start_time = time.time()
        self.severity_weights = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0,
        }

    def detect_missing_seed(self, cfg: Any) -> bool:
        """Check if seed is properly set and propagated"""
        evidence = {}

        # Check 1: Seed is None or explicitly disabled
        seed = cfg.get("seed") if isinstance(cfg, dict) else getattr(cfg, "seed", None)
        if seed is None or seed == -1:
            evidence["seed_missing"] = True
            self.record_event(
                norm_name="reproducibility",
                violation_name="missing_seed",
                event_type="VIOLATION_DETECTED",
                severity="medium",
                message="Seed is None or disabled - torch/numpy/dataloader randomness uncontrolled",
                evidence=evidence,
            )
            return True

        # Check 2: Skip seed init flag (training should NOT set this)
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else getattr(cfg, "train", {})
        skip_seed = train_cfg.get("skip_seed_init", False) if isinstance(train_cfg, dict) else getattr(train_cfg, "skip_seed_init", False)
        
        if skip_seed:
            evidence["skip_seed_init"] = True
            self.record_event(
                norm_name="reproducibility",
                violation_name="missing_seed",
                event_type="VIOLATION_DETECTED",
                severity="medium",
                message="Seed initialization skipped - reproducibility not guaranteed",
                evidence=evidence,
            )
            return True

        return False

    def detect_nondeterminism(self, cfg: Any) -> bool:
        """Check if non-determinism is enabled"""
        evidence = {}

        repro = cfg.get("repro", {}) if isinstance(cfg, dict) else getattr(cfg, "repro", {})
        deterministic = repro.get("deterministic", True) if isinstance(repro, dict) else getattr(repro, "deterministic", True)
        cudnn_benchmark = repro.get("cudnn_benchmark", False) if isinstance(repro, dict) else getattr(repro, "cudnn_benchmark", False)

        # Check 1: Deterministic algorithms disabled
        if not deterministic:
            evidence["deterministic_disabled"] = True

        # Check 2: cuDNN benchmark enabled (racing)
        if cudnn_benchmark:
            evidence["cudnn_benchmark_enabled"] = True

        if evidence:
            self.record_event(
                norm_name="reproducibility",
                violation_name="nondeterminism_enabled",
                event_type="VIOLATION_DETECTED",
                severity="high",
                message="Non-determinism enabled: cuDNN races or deterministic algos disabled",
                evidence=evidence,
            )
            return True

        return False

    def detect_untracked_config_change(self, cfg: Any, saved_config_path: str = None) -> bool:
        """Check if config changes are tracked and persisted"""
        evidence = {}

        # Check 1: Config not saved in checkpoint
        checkpoint = cfg.get("checkpoint", {}) if isinstance(cfg, dict) else getattr(cfg, "checkpoint", {})
        include_config = checkpoint.get("include_config", True) if isinstance(checkpoint, dict) else getattr(checkpoint, "include_config", True)
        include_overrides = checkpoint.get("include_overrides", True) if isinstance(checkpoint, dict) else getattr(checkpoint, "include_overrides", True)

        if not include_config or not include_overrides:
            evidence["config_not_persisted"] = True

        # Check 2: Config validation skipped
        reporting = cfg.get("reporting", {}) if isinstance(cfg, dict) else getattr(cfg, "reporting", {})
        skip_validation = reporting.get("skip_config_validation", False) if isinstance(reporting, dict) else getattr(reporting, "skip_config_validation", False)
        if skip_validation:
            evidence["config_validation_skipped"] = True

        if evidence:
            self.record_event(
                norm_name="reproducibility",
                violation_name="untracked_config_change",
                event_type="VIOLATION_DETECTED",
                severity="high",
                message="Config changes not tracked: overrides not persisted or validation skipped",
                evidence=evidence,
            )
            return True

        return False

    def detect_excessive_batch_size(self, cfg: Any) -> bool:
        """Check if batch size is unreasonably large"""
        evidence = {}
        
        train = cfg.get("train", {}) if isinstance(cfg, dict) else getattr(cfg, "train", {})
        batch_size = train.get("batch_size", 128) if isinstance(train, dict) else getattr(train, "batch_size", 128)
        
        # CIFAR-10 typical batch sizes: 128-512
        # Flag if > 1024
        if batch_size > 1024:
            evidence["batch_size"] = batch_size
            evidence["recommended_max"] = 512
            self.record_event(
                norm_name="resource_governance",
                violation_name="excessive_batch_size",
                event_type="VIOLATION_DETECTED",
                severity="medium",
                message=f"Excessive batch size ({batch_size}) wastes memory",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_inefficient_precision(self, cfg: Any) -> bool:
        """Check if precision is inefficient"""
        evidence = {}
        
        amp = cfg.get("amp", {}) if isinstance(cfg, dict) else getattr(cfg, "amp", {})
        amp_enabled = amp.get("enabled", False) if isinstance(amp, dict) else getattr(amp, "enabled", False)
        dtype = cfg.get("dtype", "fp16") if isinstance(cfg, dict) else getattr(cfg, "dtype", "fp16")
        
        # Should use AMP for GPU training
        if not amp_enabled and dtype == "fp32":
            evidence["amp_disabled"] = True
            evidence["dtype"] = dtype
            self.record_event(
                norm_name="resource_governance",
                violation_name="inefficient_precision",
                event_type="VIOLATION_DETECTED",
                severity="low",
                message="Not using mixed precision training - inefficient GPU usage",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_excessive_workers(self, cfg: Any) -> bool:
        """Check if dataloader workers are excessive"""
        evidence = {}
        
        dataset = cfg.get("dataset", {}) if isinstance(cfg, dict) else getattr(cfg, "dataset", {})
        num_workers = dataset.get("num_workers", 4) if isinstance(dataset, dict) else getattr(dataset, "num_workers", 4)
        
        # Typical: 4-8 workers per GPU
        # Flag if > 16
        if num_workers > 16:
            evidence["num_workers"] = num_workers
            evidence["recommended_max"] = 8
            self.record_event(
                norm_name="resource_governance",
                violation_name="excessive_workers",
                event_type="VIOLATION_DETECTED",
                severity="low",
                message=f"Excessive dataloader workers ({num_workers}) wastes CPU",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_eval_mode_bug(self, cfg: Any) -> bool:
        """Check if eval mode is properly enforced during evaluation"""
        evidence = {}
        
        train = cfg.get("train", {}) if isinstance(cfg, dict) else getattr(cfg, "train", {})
        force_eval_mode = train.get("force_eval_mode", True) if isinstance(train, dict) else getattr(train, "force_eval_mode", True)
        
        # Violation: force_eval_mode is False (skip eval mode during validation)
        if not force_eval_mode:
            evidence["force_eval_mode"] = force_eval_mode
            self.record_event(
                norm_name="experimental_validity",
                violation_name="eval_mode_bug",
                event_type="VIOLATION_DETECTED",
                severity="high",
                message="Model eval() mode disabled during validation - dropout/batch norm will behave incorrectly",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_aug_leak(self, cfg: Any) -> bool:
        """Check if augmentation leaks into evaluation data"""
        evidence = {}
        
        dataset = cfg.get("dataset", {}) if isinstance(cfg, dict) else getattr(cfg, "dataset", {})
        eval_augment = dataset.get("eval_augment", {}) if isinstance(dataset, dict) else getattr(dataset, "eval_augment", {})
        
        # Check for stochastic augmentations in eval data
        random_crop = eval_augment.get("random_crop", False) if isinstance(eval_augment, dict) else getattr(eval_augment, "random_crop", False)
        hflip = eval_augment.get("hflip", False) if isinstance(eval_augment, dict) else getattr(eval_augment, "hflip", False)
        
        if random_crop or hflip:
            evidence["random_crop"] = random_crop
            evidence["hflip"] = hflip
            self.record_event(
                norm_name="experimental_validity",
                violation_name="aug_leak",
                event_type="VIOLATION_DETECTED",
                severity="high",
                message="Stochastic augmentation applied to evaluation data - breaks reproducibility and fairness",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_amp_nan(self, cfg: Any) -> bool:
        """Check if AMP is properly configured to avoid NaN/Inf"""
        evidence = {}
        
        amp = cfg.get("amp", {}) if isinstance(cfg, dict) else getattr(cfg, "amp", {})
        amp_enabled = amp.get("enabled", False) if isinstance(amp, dict) else getattr(amp, "enabled", False)
        grad_scaler = amp.get("grad_scaler", True) if isinstance(amp, dict) else getattr(amp, "grad_scaler", True)
        
        scheduler = cfg.get("scheduler", {}) if isinstance(cfg, dict) else getattr(cfg, "scheduler", {})
        warmup_epochs = scheduler.get("warmup_epochs", 1) if isinstance(scheduler, dict) else getattr(scheduler, "warmup_epochs", 1)
        
        # Violation: AMP enabled but without grad scaler or warmup
        if amp_enabled and (not grad_scaler or warmup_epochs == 0):
            evidence["amp_enabled"] = amp_enabled
            evidence["grad_scaler"] = grad_scaler
            evidence["warmup_epochs"] = warmup_epochs
            self.record_event(
                norm_name="experimental_validity",
                violation_name="amp_nan",
                event_type="VIOLATION_DETECTED",
                severity="high",
                message="AMP enabled without grad scaler or warmup - NaN/Inf likely during training",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_reporting_single_run(self, cfg: Any) -> bool:
        """Check if multi-seed reporting is enabled"""
        evidence = {}
        
        reporting = cfg.get("reporting", {}) if isinstance(cfg, dict) else getattr(cfg, "reporting", {})
        require_multi_seed = reporting.get("require_multi_seed", True) if isinstance(reporting, dict) else getattr(reporting, "require_multi_seed", True)
        
        metrics = cfg.get("metrics", {}) if isinstance(cfg, dict) else getattr(cfg, "metrics", {})
        report_mean_std = metrics.get("report_mean_std", True) if isinstance(metrics, dict) else getattr(metrics, "report_mean_std", True)
        
        # Violation: reporting single run without multi-seed or uncertainty
        if not require_multi_seed or not report_mean_std:
            evidence["require_multi_seed"] = require_multi_seed
            evidence["report_mean_std"] = report_mean_std
            self.record_event(
                norm_name="reporting_standards",
                violation_name="reporting_single_run",
                event_type="VIOLATION_DETECTED",
                severity="medium",
                message="Single-run reporting without multi-seed uncertainty - breaks scientific rigor",
                evidence=evidence,
            )
            return True
        
        return False

    def detect_checkpoint_incomplete(self, cfg: Any) -> bool:
        """Check if checkpoints are complete"""
        evidence = {}
        
        checkpoint = cfg.get("checkpoint", {}) if isinstance(cfg, dict) else getattr(cfg, "checkpoint", {})
        include_optimizer = checkpoint.get("include_optimizer", True) if isinstance(checkpoint, dict) else getattr(checkpoint, "include_optimizer", True)
        include_scheduler = checkpoint.get("include_scheduler", True) if isinstance(checkpoint, dict) else getattr(checkpoint, "include_scheduler", True)
        include_scaler = checkpoint.get("include_scaler", True) if isinstance(checkpoint, dict) else getattr(checkpoint, "include_scaler", True)
        
        # Violation: checkpoint missing critical components
        if not include_optimizer or not include_scheduler or not include_scaler:
            evidence["include_optimizer"] = include_optimizer
            evidence["include_scheduler"] = include_scheduler
            evidence["include_scaler"] = include_scaler
            self.record_event(
                norm_name="reporting_standards",
                violation_name="checkpoint_incomplete",
                event_type="VIOLATION_DETECTED",
                severity="medium",
                message="Checkpoint missing optimizer/scheduler/scaler - cannot resume training properly",
                evidence=evidence,
            )
            return True
        
        return False

    def record_event(
        self,
        norm_name: str,
        violation_name: str,
        event_type: str,
        severity: str,
        message: str,
        evidence: Dict[str, Any] = None,
    ) -> None:
        """Record a norm event"""
        elapsed = time.time() - self.start_time
        event = NormEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            norm_name=norm_name,
            violation_name=violation_name,
            event_type=event_type,
            severity=severity,
            time_to_detection_sec=elapsed,
            message=message,
            evidence=evidence or {},
        )
        self.events.append(event)

        # Append to JSONL file
        Path(self.events_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute reproducibility metrics"""
        if not self.events:
            return {}

        detected = [e for e in self.events if e.event_type == "VIOLATION_DETECTED"]
        remediated = [e for e in self.events if e.event_type == "REMEDIATION_SUCCESS"]
        failed = [e for e in self.events if e.event_type == "REMEDIATION_FAILED"]

        detection_rate = len(detected) / len(self.events) * 100 if self.events else 0
        remediation_success_rate = (
            len(remediated) / (len(remediated) + len(failed)) * 100
            if (remediated + failed)
            else 0
        )

        # Time to detection (seconds)
        times_to_detection = [e.time_to_detection_sec for e in detected]
        avg_time_to_detection = (
            sum(times_to_detection) / len(times_to_detection) if times_to_detection else 0
        )

        # Severity-weighted score (0-100)
        severity_score = 0
        for event in detected:
            weight = self.severity_weights.get(event.severity, 1.0)
            severity_score += weight * 10  # Each violation contributes weight * 10

        # Normalize to 0-100
        max_score = 30 * 10  # 3 violation types × 10
        normalized_severity_score = min(100, (severity_score / max_score) * 100)

        return {
            "detected_violations": len(detected),
            "detection_rate_pct": detection_rate,
            "avg_time_to_detection_sec": avg_time_to_detection,
            "remediation_success_rate_pct": remediation_success_rate,
            "severity_weighted_score": normalized_severity_score,
            "severity_breakdown": {
                "low": len([e for e in detected if e.severity == "low"]),
                "medium": len([e for e in detected if e.severity == "medium"]),
                "high": len([e for e in detected if e.severity == "high"]),
            },
        }
