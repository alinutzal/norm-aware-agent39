"""Parse runs and aggregate metrics"""

import json
from pathlib import Path
from typing import List, Dict, Any


def parse_run_directory(run_dir: Path) -> Dict[str, Any]:
    """Parse a single run directory"""
    metrics_file = run_dir / "metrics.json"
    logs_file = run_dir / "logs.jsonl"

    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

    logs = []
    if logs_file.exists():
        with open(logs_file, "r") as f:
            for line in f:
                logs.append(json.loads(line))

    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "logs": logs,
    }


def aggregate_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Parse all runs in directory"""
    runs = []
    for run_dir in runs_dir.glob("run_*"):
        if run_dir.is_dir():
            runs.append(parse_run_directory(run_dir))
    return runs
