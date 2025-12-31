# src/nora/violations/injector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import time
import json

import torch


# ----------------------------
# Events writing (JSONL)
# ----------------------------

def _utc_ts() -> str:
    # ISO-like UTC timestamp (simple, stable)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_event(events_path: str, event: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


# ----------------------------
# Runtime context contract
# ----------------------------

@dataclass
class RuntimeContext:
    """
    Minimal runtime context your training engine can pass around.
    Keep this tiny. Everything optional.
    """
    run_id: str
    out_dir: str
    events_path: str                    # e.g., f"{out_dir}/norms/events.jsonl"
    seed: int
    cfg: Any                            # Hydra DictConfig or plain object
    # handles set later:
    train_loader: Optional[Any] = None
    eval_loader: Optional[Any] = None
    model: Optional[torch.nn.Module] = None


# ----------------------------
# Violation application
# ----------------------------

def apply_violation(ctx: RuntimeContext) -> None:
    """
    Applies runtime side effects for the injected violation profile.
    The Hydra config already contains the toggles; this function ensures
    the toggles are *actually enacted* at runtime.

    Always writes a VIOLATION_INJECTED event to norms/events.jsonl.
    """
    name = getattr(getattr(ctx.cfg, "violation", None), "name", "none")
    if name is None:
        name = "none"

    # Always record what we injected (even "none")
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_INJECTED",
        "violation_name": name,
        "severity": "info",
        "hook": "on_run_start",
        "message": f"Injected violation profile: {name}",
        "evidence": _evidence_snapshot(ctx, name),
    })

    # Apply concrete runtime side effects
    _APPLIERS.get(name, _apply_none)(ctx)


def _evidence_snapshot(ctx: RuntimeContext, violation_name: str) -> Dict[str, Any]:
    """
    Lightweight evidence snapshot for debugging + Table 1 traceability.
    Keep it stable and short.
    """
    cfg = ctx.cfg
    evidence: Dict[str, Any] = {
        "mode": getattr(getattr(cfg, "mode", None), "name", None),
        "regime": getattr(getattr(cfg, "regime", None), "name", None),
        "seed": ctx.seed,
    }

    # Common knobs that your violations touch
    repro = getattr(cfg, "repro", None)
    if repro is not None:
        evidence["repro"] = {
            "deterministic": getattr(repro, "deterministic", None),
            "cudnn_benchmark": getattr(repro, "cudnn_benchmark", None),
            "dataloader_seed_workers": getattr(repro, "dataloader_seed_workers", None),
        }

    amp = getattr(cfg, "amp", None)
    if amp is not None:
        evidence["amp"] = {
            "enabled": getattr(amp, "enabled", None),
            "grad_scaler": getattr(amp, "grad_scaler", None),
            "loss_nan_policy": getattr(amp, "loss_nan_policy", None),
        }

    dataset = getattr(cfg, "dataset", None)
    if dataset is not None:
        evidence["dataset"] = {
            "name": getattr(dataset, "name", None),
            "eval_augment": getattr(dataset, "eval_augment", None),
        }

    ckpt = getattr(cfg, "checkpoint", None)
    if ckpt is not None:
        evidence["checkpoint"] = {
            "include_optimizer": getattr(ckpt, "include_optimizer", None),
            "include_scheduler": getattr(ckpt, "include_scheduler", None),
            "include_scaler": getattr(ckpt, "include_scaler", None),
        }

    train = getattr(cfg, "train", None)
    if train is not None:
        evidence["train"] = {
            "force_eval_mode": getattr(train, "force_eval_mode", None),
        }

    opt = getattr(cfg, "optimizer", None)
    if opt is not None:
        evidence["optimizer"] = {
            "name": getattr(opt, "name", None),
            "lr": getattr(opt, "lr", None),
        }

    sched = getattr(cfg, "scheduler", None)
    if sched is not None:
        evidence["scheduler"] = {
            "name": getattr(sched, "name", None),
            "warmup_epochs": getattr(sched, "warmup_epochs", None),
        }

    return evidence


# ----------------------------
# Concrete appliers
# ----------------------------

def _apply_none(ctx: RuntimeContext) -> None:
    return


def _apply_nondeterminism(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/nondeterminism.yaml:
      repro.deterministic: false
      repro.cudnn_benchmark: true
      repro.dataloader_seed_workers: false
    """
    torch.backends.cudnn.benchmark = True
    # Deterministic algorithms OFF (to allow nondeterminism)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # If your dataloader module uses cfg.repro.dataloader_seed_workers
    # to set worker_init_fn, setting it false is enough. We record a runtime note.
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "nondeterminism",
        "severity": "info",
        "hook": "on_run_start",
        "message": "Applied nondeterminism: cudnn.benchmark=True, deterministic_algorithms=False, worker seeding disabled by config.",
        "evidence": {
            "torch.backends.cudnn.benchmark": torch.backends.cudnn.benchmark,
        },
    })


def _apply_amp_nan(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/amp_nan.yaml:
      amp.grad_scaler: false
      scheduler.warmup_epochs: 0
      optimizer.lr: 8e-4
      amp.loss_nan_policy: ignore
    Runtime side effect: if AMP enabled, ensure GradScaler is NOT used.
    Your training loop should honor cfg.amp.grad_scaler. We also add a runtime note.
    """
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "amp_nan",
        "severity": "info",
        "hook": "on_run_start",
        "message": "Applied AMP-NaN profile: expects AMP enabled with GradScaler disabled and aggressive LR/warmup=0.",
        "evidence": {},
    })


def _apply_eval_mode_bug(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/eval_mode_bug.yaml:
      train.force_eval_mode: false
    Runtime side effect: we set an explicit flag that your eval loop can read.
    """
    # Make the intent explicit at runtime:
    setattr(ctx, "skip_eval_mode_switch", True)

    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "eval_mode_bug",
        "severity": "info",
        "hook": "on_eval_start",
        "message": "Applied eval-mode bug: evaluation should run without calling model.eval().",
        "evidence": {"ctx.skip_eval_mode_switch": True},
    })


def _apply_aug_leak(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/aug_leak.yaml:
      dataset.eval_augment: stochastic transforms enabled
    Runtime side effect: ensure eval loader uses the configured eval transforms.
    Your dataloader builder should apply cfg.dataset.eval_augment deterministically.
    Here we only record a runtime note (the actual transform is in data pipeline).
    """
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "aug_leak",
        "severity": "info",
        "hook": "on_dataloader_built",
        "message": "Applied augmentation leakage: evaluation transforms include randomness per config.",
        "evidence": {"eval_augment": getattr(getattr(ctx.cfg, "dataset", None), "eval_augment", None)},
    })


def _apply_checkpoint_incomplete(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/checkpoint_incomplete.yaml:
      checkpoint.include_optimizer/scheduler/scaler: false
    Runtime side effect: your checkpoint saver should honor these flags.
    We record a runtime note so it's provable in the manifest/events.
    """
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "checkpoint_incomplete",
        "severity": "info",
        "hook": "on_checkpoint_save",
        "message": "Applied checkpoint incompleteness: saving model-only state (no optimizer/scheduler/scaler).",
        "evidence": {
            "include_optimizer": getattr(getattr(ctx.cfg, "checkpoint", None), "include_optimizer", None),
            "include_scheduler": getattr(getattr(ctx.cfg, "checkpoint", None), "include_scheduler", None),
            "include_scaler": getattr(getattr(ctx.cfg, "checkpoint", None), "include_scaler", None),
        },
    })


def _apply_reporting_single_run(ctx: RuntimeContext) -> None:
    """
    Matches configs/violation/reporting_single_run.yaml:
      reporting.require_multi_seed: false
      metrics.report_mean_std: false
    Runtime side effect: your report generator should honor this.
    We log a note to make it explicit.
    """
    append_event(ctx.events_path, {
        "ts": _utc_ts(),
        "run_id": ctx.run_id,
        "event_type": "VIOLATION_RUNTIME_APPLIED",
        "violation_name": "reporting_single_run",
        "severity": "info",
        "hook": "on_run_end",
        "message": "Applied single-run reporting: disables multi-seed requirement and mean/std summary.",
        "evidence": {
            "require_multi_seed": getattr(getattr(ctx.cfg, "reporting", None), "require_multi_seed", None),
            "report_mean_std": getattr(getattr(ctx.cfg, "metrics", None), "report_mean_std", None),
        },
    })


_APPLIERS = {
    "none": _apply_none,
    "nondeterminism": _apply_nondeterminism,
    "amp_nan": _apply_amp_nan,
    "eval_mode_bug": _apply_eval_mode_bug,
    "aug_leak": _apply_aug_leak,
    "checkpoint_incomplete": _apply_checkpoint_incomplete,
    "reporting_single_run": _apply_reporting_single_run,
}
