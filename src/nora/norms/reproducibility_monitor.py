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
        seed = getattr(cfg, "seed", None)
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
        skip_seed = getattr(getattr(cfg, "train", {}), "skip_seed_init", False)
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

        repro = getattr(cfg, "repro", {})
        deterministic = getattr(repro, "deterministic", True)
        cudnn_benchmark = getattr(repro, "cudnn_benchmark", False)

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
        checkpoint = getattr(cfg, "checkpoint", {})
        include_config = getattr(checkpoint, "include_config", True)
        include_overrides = getattr(checkpoint, "include_overrides", True)

        if not include_config or not include_overrides:
            evidence["config_not_persisted"] = True

        # Check 2: Config validation skipped
        reporting = getattr(cfg, "reporting", {})
        skip_validation = getattr(reporting, "skip_config_validation", False)
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
