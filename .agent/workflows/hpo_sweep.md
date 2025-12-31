---
description: Run an HPO Sweep (Governance Compliant)
---

This workflow orchestrates a hyperparameter optimization sweep, ensuring resource usage aligns with governance norms and results are properly tagged.

1. **Define Search Space & Governance Policy**
   Load the search configuration and verify it against resource budget constraints.
   // turbo
   ```bash
   python scripts/check_governance.py --config config/hpo_sweep_v1.yaml --budget-limit-usd 50 --gpu-limit 4
   ```

2. **Launch HPO Sweep**
   Start the sweep agent (e.g., Ray Tune / Optuna).
   // turbo
   ```bash
   python scripts/run_sweep.py \
     --config config/hpo_sweep_v1.yaml \
     --experiment-name "experiment_alpha_01" \
     --num-samples 20 \
     --concurrency 4
   ```

3. **Tag Best Trial**
   Automatically tag the best model for artifact tracking, adding metadata required for deployment approval.
   // turbo
   ```bash
   python scripts/register_best_model.py \
     --experiment-name "experiment_alpha_01" \
     --metric "val_loss" \
     --mode "min" \
     --tag "candidate-release"
   ```

4. **Generate Audit Report**
   Create a report of all trials, highlighting resource consumption and parameter sensitivity.
   // turbo
   ```bash
   python scripts/generate_hpo_report.py --experiment-name "experiment_alpha_01" --output reports/hpo_audit.pdf
   ```
