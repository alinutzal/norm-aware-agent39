#!/usr/bin/env python3
"""Generate figures: loss curves, accuracy plots"""

import argparse
from pathlib import Path

from nora.analysis.parse_runs import aggregate_runs
from nora.analysis.figures import plot_loss_curves, plot_accuracy_curves


def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--run-dir", type=str, default="runs", help="Runs directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    runs = aggregate_runs(runs_dir)
    
    print(f"Found {len(runs)} runs")
    
    # Generate figures
    plot_loss_curves(runs, output_dir / "loss_curves.png")
    plot_accuracy_curves(runs, output_dir / "accuracy_curves.png")
    
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
