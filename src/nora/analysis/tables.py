"""Generate tables (LaTeX/CSV) from run results"""

from typing import List, Dict, Any


def generate_latex_table(
    runs: List[Dict[str, Any]],
    caption: str = "Results",
) -> str:
    """Generate LaTeX table from runs"""
    latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{l|rr|rr}}
\\hline
Run & Train Loss & Train Acc & Val Loss & Val Acc \\\\
\\hline
"""

    for run in runs:
        metrics = run["metrics"]
        latex += f"{run['run_dir']} & {metrics.get('train_loss', 0):.4f} & {metrics.get('train_acc', 0):.2f} & {metrics.get('val_loss', 0):.4f} & {metrics.get('val_acc', 0):.2f} \\\\\n"

    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex
