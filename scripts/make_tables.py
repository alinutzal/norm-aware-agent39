#!/usr/bin/env python3
"""Generate Table 1/2: Results aggregated by mode, regime, violation"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from nora.analysis.parse_runs import aggregate_runs
from nora.analysis.tables import generate_latex_table


def main():
    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument("--run-dir", type=str, default="runs", help="Runs directory")
    parser.add_argument("--output", type=str, default="table.tex", help="Output LaTeX file")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.run_dir)
    runs = aggregate_runs(runs_dir)
    
    print(f"Found {len(runs)} runs")
    
    # Generate table
    latex = generate_latex_table(runs)
    
    with open(args.output, "w") as f:
        f.write(latex)
    
    print(f"Table written to {args.output}")


if __name__ == "__main__":
    main()
