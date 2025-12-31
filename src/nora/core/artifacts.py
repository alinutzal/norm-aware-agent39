"""Artifact management: checkpoint paths, metrics, manifests"""

from pathlib import Path
from typing import Optional, Dict, Any
import json


class ArtifactManager:
    """Manages checkpoint and metric file organization for a run"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, epoch: int) -> Path:
        """Get path for epoch checkpoint"""
        return self.checkpoint_dir / f"epoch_{epoch:04d}.pt"

    def get_best_checkpoint_path(self) -> Path:
        """Get path for best model checkpoint"""
        return self.checkpoint_dir / "best.pt"

    def get_metrics_path(self) -> Path:
        """Get path for metrics JSON file"""
        return self.run_dir / "metrics.json"

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to JSON file"""
        with open(self.get_metrics_path(), "w") as f:
            json.dump(metrics, f, indent=2)

    def load_metrics(self) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        metrics_path = self.get_metrics_path()
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {}
