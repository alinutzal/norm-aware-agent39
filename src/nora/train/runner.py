"""High-level training loop for NORA vision models."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..data.cifar10 import get_cifar10_loaders
from ..models.deit_tiny import deit_tiny
from ..norms.reproducibility_monitor import ReproducibilityNormMonitor
from ..agent.norm_agent import NormAgent
from .checkpoint import save_checkpoint
from .engine import train_epoch, evaluate

logger = logging.getLogger(__name__)

# Simple registry for supported architectures
MODEL_REGISTRY = {
    "deit_tiny": deit_tiny,
}


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if deterministic:
        # Set CUBLAS workspace config for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logger.warning(f"Could not enable deterministic algorithms: {e}")


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Construct model from config."""
    arch = model_cfg.get("name", "deit_tiny")
    num_classes = model_cfg.get("num_classes", 10)
    pretrained = model_cfg.get("pretrained", False)

    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model architecture: {arch}")

    model_fn = MODEL_REGISTRY[arch]
    return model_fn(num_classes=num_classes, pretrained=pretrained)


def get_loaders(config: Dict[str, Any]):
    """Create train/val dataloaders for the configured dataset."""
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    
    dataset = dataset_cfg.get("name", "cifar10").lower()
    batch_size = train_cfg.get("batch_size", 128)
    num_workers = dataset_cfg.get("num_workers", 4)
    augment = bool(dataset_cfg.get("train_augment", {}))
    pin_memory = dataset_cfg.get("pin_memory", True)
    
    # Use absolute path to avoid downloading to run directory when hydra.job.chdir=true
    from hydra.utils import get_original_cwd
    data_dir = dataset_cfg.get("raw_dir", "data/raw")
    # Convert to absolute path from original working directory
    data_dir = os.path.join(get_original_cwd(), data_dir)

    if dataset == "cifar10":
        return get_cifar10_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            pin_memory=pin_memory,
            data_dir=data_dir,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_optimizer(config: Dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """Instantiate optimizer based on config."""
    opt_cfg = config.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(config: Dict[str, Any], optimizer: optim.Optimizer):
    """Create LR scheduler (optional)."""
    sched_cfg = config.get("scheduler", {})
    train_cfg = config.get("train", {})
    
    sched_name = sched_cfg.get("name", "none").lower()
    epochs = train_cfg.get("epochs", 1)
    warmup_epochs = max(0, int(sched_cfg.get("warmup_epochs", 0)))

    if sched_name in {"none", None}:
        return None

    if sched_name == "cosine":
        main_t_max = max(1, epochs - warmup_epochs)
        main = CosineAnnealingLR(optimizer, T_max=main_t_max)

        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])

        return main

    raise ValueError(f"Unsupported scheduler: {sched_name}")


def _checkpoint_violation_flags(violation_cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Extract checkpoint-related violation toggles."""
    active = set(violation_cfg.get("active", [])) if violation_cfg.get("enabled") else set()
    opts = violation_cfg.get("checkpoint_incomplete", {}) if "checkpoint_incomplete" in active else {}
    return {
        "skip_optimizer_state": opts.get("skip_optimizer_state", False),
        "skip_scheduler_state": opts.get("skip_scheduler_state", False),
    }


def _maybe_save_checkpoint(
    run_dir: Path,
    filename: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_cfg: Dict[str, Any],
    violation_cfg: Dict[str, Any],
) -> None:
    """Save checkpoint, honoring incomplete checkpoint violation if present."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / filename
    violation_flags = _checkpoint_violation_flags(violation_cfg)

    if violation_flags["skip_optimizer_state"] and violation_flags["skip_scheduler_state"]:
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics},
            checkpoint_path,
        )
        return

    if violation_flags["skip_optimizer_state"]:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=None,
            scheduler=None if violation_flags["skip_scheduler_state"] else scheduler,
            epoch=epoch,
            metrics=metrics,
        )
        return

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        scheduler=None if violation_flags["skip_scheduler_state"] else scheduler,
        epoch=epoch,
        metrics=metrics,
    )


def run_training(config: Dict[str, Any], output_dir: str = "data/processed/cifar10") -> Dict[str, Any]:
    """Execute end-to-end training based on merged config."""
    train_cfg = config.get("train", {})
    dataset_cfg = config.get("dataset", {})
    checkpoint_cfg = config.get("checkpoint", {})
    repro_cfg = config.get("repro", {})
    violation_cfg = config.get("violations", {}) or {}
    mode = config.get("modes", "pipeline")
    regime = config.get("regime", "balanced")

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility monitoring & agent initialization
    out_dir = Path(config.get("run", {}).get("out_dir", "runs/default"))
    events_file = out_dir / "norms" / "reproducibility_events.jsonl"
    repro_monitor = ReproducibilityNormMonitor(str(events_file), cfg=config)

    # Initialize agent (only if not in pipeline mode)
    agent = None
    if mode != "pipeline":
        agent = NormAgent(
            regime=regime,
            mode=mode,
            events_file=str(out_dir / "norms" / "agent_decisions.jsonl"),
            cfg=config,
        )
        logger.info(f"Agent enabled: mode={mode}, regime={regime}")

    # Check for reproducibility violations early
    missing_seed = repro_monitor.detect_missing_seed(config)
    nondeterminism = repro_monitor.detect_nondeterminism(config)
    untracked_config = repro_monitor.detect_untracked_config_change(config)

    violations_detected = []
    if missing_seed:
        violations_detected.append(("missing_seed", "medium", {"config": {"seed": "not set"}}))
    if nondeterminism:
        violations_detected.append(
            ("nondeterminism_enabled", "high", {"deterministic": False, "cudnn_benchmark": True})
        )
    if untracked_config:
        violations_detected.append(
            ("untracked_config_change", "high", {"include_config": False, "include_overrides": False})
        )

    if violations_detected:
        logger.warning(f"Reproducibility violations detected: {len(violations_detected)}")
        
        # Agent responds to violations
        if agent:
            for violation_name, severity, evidence in violations_detected:
                decision = agent.decide_on_violation(violation_name, severity, evidence)
                logger.info(f"Agent decision: {decision.message}")
                
                # Apply remediation if in balanced mode
                if decision.action == "auto_fix" and regime == "balanced":
                    success, msg = agent.remediate(config, violation_name)
                    logger.info(f"Remediation: {msg}")
                    if success:
                        # Re-check after remediation
                        if violation_name == "missing_seed":
                            missing_seed = False
                        elif violation_name == "nondeterminism_enabled":
                            nondeterminism = False
                        elif violation_name == "untracked_config_change":
                            untracked_config = False
                
                # Halt if strict
                if decision.action == "halt" and regime == "strict":
                    raise RuntimeError(
                        f"STRICT REGIME: Training halted due to {violation_name}. "
                        f"Fix: {decision.suggested_fix}"
                    )

    # Original seed initialization (only if not missing seed violation)
    seed = config.get("seed", 42)
    deterministic = repro_cfg.get("deterministic", True)
    benchmark = repro_cfg.get("cudnn_benchmark", False)
    nondet_active = violation_cfg.get("enabled") and "nondeterminism" in set(
        violation_cfg.get("active", [])
    )
    if nondet_active:
        logger.warning("Nondeterminism violation active: skipping deterministic seeding.")
    elif not missing_seed:  # Only set seed if not explicitly missing
        set_seed(seed, deterministic=deterministic, benchmark=benchmark)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_loaders(config)

    # Model
    model = build_model(config.get("model", {})).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    # AMP setup
    amp_cfg = config.get("amp", {})
    dtype = config.get("dtype", "fp16")
    use_amp = amp_cfg.get("enabled", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    epochs = int(train_cfg.get("epochs", 1))
    save_every = int(checkpoint_cfg.get("save_every_epochs", 0) or 0)

    history = []
    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            scaler=scaler if use_amp else None,
            amp_dtype=amp_dtype,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0].get("lr", config.get("optimizer", {}).get("lr", 0.0))
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(metrics)

        logger.info(
            (
                f"Epoch {epoch}/{epochs} - "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% lr={current_lr:.5f}"
            )
        )

        # Checkpointing: periodic
        if save_every and epoch % save_every == 0:
            _maybe_save_checkpoint(
                run_dir,
                f"checkpoint_epoch{epoch}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                checkpoint_cfg,
                violation_cfg,
            )

        # Checkpointing: best model
        if checkpoint_cfg.get("save_best", True) and val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            _maybe_save_checkpoint(
                run_dir,
                "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                checkpoint_cfg,
                violation_cfg,
            )

    # Final checkpoint
    _maybe_save_checkpoint(
        run_dir,
        "last.pt",
        model,
        optimizer,
        scheduler,
        epochs,
        history[-1],
        checkpoint_cfg,
        violation_cfg,
    )

    # Persist history
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save reproducibility metrics
    repro_metrics = repro_monitor.compute_metrics()
    repro_metrics_path = run_dir / "norms" / "reproducibility_metrics.json"
    repro_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(repro_metrics_path, "w") as f:
        json.dump(repro_metrics, f, indent=2)

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "epochs": epochs,
        "metrics_path": str(metrics_path),
        "reproducibility_metrics": repro_metrics,
    }
    
    # Add agent metrics if agent was used
    if agent:
        agent_metrics = agent.compute_metrics()
        summary["agent_metrics"] = agent_metrics
        agent_metrics_path = out_dir / "norms" / "agent_metrics.json"
        agent_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agent_metrics_path, "w") as f:
            json.dump(agent_metrics, f, indent=2)
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Training complete. Best val acc: {best_acc:.2f}% at epoch {best_epoch}. "
        f"Artifacts saved to {run_dir}"
    )
    return summary
