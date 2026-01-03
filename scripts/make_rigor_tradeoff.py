#!/usr/bin/env python3
"""Aggregate rigor-throughput runs and plot trade-off figure."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_runs(base_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for summary_path in base_dir.glob("*/summary.json"):
        try:
            summary = json.load(open(summary_path))
        except Exception:
            continue
        run_dir = summary_path.parent
        regime = run_dir.name.split("_")[0]

        repro = summary.get("reproducibility_metrics", {})
        agent = summary.get("agent_metrics", {})

        wall = summary.get("wall_time_sec")
        runs.append(
            {
                "regime": regime,
                "run_dir": str(run_dir),
                "wall_time_sec": wall,
                "violations": repro.get("detected_violations", 0),
                "rem_success_pct": repro.get("remediation_success_rate_pct", 0),
                "agent_success_count": agent.get("remediation_success_count", 0),
            }
        )
    return runs


def save_csv(runs: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["regime", "run_dir", "wall_time_sec", "violations", "rem_success_pct", "agent_success_count"]
        )
        for r in runs:
            writer.writerow(
                [
                    r["regime"],
                    r["run_dir"],
                    r.get("wall_time_sec"),
                    r.get("violations", 0),
                    r.get("rem_success_pct", 0),
                    r.get("agent_success_count", 0),
                ]
            )


def plot_tradeoff(runs: List[Dict[str, Any]], out_png: Path, xkey: str = "wall_time_sec", ykey: str = "violations") -> None:
    regimes = ["strict", "balanced", "exploratory"]
    colors = {"strict": "tab:red", "balanced": "tab:green", "exploratory": "tab:blue"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for reg in regimes:
        sub = [r for r in runs if r.get("regime") == reg and r.get(xkey) is not None]
        if not sub:
            continue
        xs = np.array([r.get(xkey, 0) for r in sub], dtype=float)
        ys = np.array([r.get(ykey, 0) for r in sub], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        ax.scatter(xs, ys, label=reg, color=colors.get(reg, "gray"), s=50)
        ax.plot(xs, ys, color=colors.get(reg, "gray"), alpha=0.6)

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Detected violations (lower is better)")
    ax.set_title("Rigor–throughput trade-off")
    ax.legend(title="Regime")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make rigor-throughput trade-off figure")
    parser.add_argument("--base-dir", required=True, help="Base directory containing runs/regime_seed/")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--png", default=None, help="Output PNG path")
    args = parser.parse_args()

    base = Path(args.base_dir)
    runs = load_runs(base)
    if not runs:
        raise SystemExit(f"No runs found in {base}")

    out_csv = Path(args.csv) if args.csv else base / "rigor_tradeoff.csv"
    out_png = Path(args.png) if args.png else base / "rigor_throughput.png"

    save_csv(runs, out_csv)
    plot_tradeoff(runs, out_png)

    print(f"Wrote CSV to {out_csv}")
    print(f"Wrote PNG to {out_png}")


if __name__ == "__main__":
    main()
