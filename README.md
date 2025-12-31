# NORA: Norm-Aware Agent for Reliable ML Training

A comprehensive pipeline for training machine learning models with norm-aware agent orchestration, focusing on reproducibility, validity, and stability.

## Overview

NORA (Norm-Aware Agent) is a framework for detecting and mitigating training violations through a hierarchy of norms, an agent-based decision system, and comprehensive experiment logging. It integrates both vision models (CIFAR-10, ViT) and tabular ML models (CatBoost on Bank Marketing dataset).

## Quick Start

### Installation
```bash
# Install dependencies
pip install -e .

# Create required directories
mkdir -p data/raw data/processed logs
```

### NORA Vision Model Training
```bash
# Run a single experiment (uses default: norm_aware mode, balanced regime)
python -m nora train

# Override mode and regime
python -m nora train mode=pipeline regime=strict

# Run with a specific violation injected
python -m nora train violation=nondeterminism seed=42

# Short training run for testing
python -m nora train train.epochs=2 seed=42

# Available modes: pipeline, agentic, norm_aware
# Available regimes: strict, balanced, exploratory
# Available violations: none, nondeterminism, amp_nan, eval_mode_bug, aug_leak, checkpoint_incomplete, reporting_single_run
```

### ML Pipeline (Bank Marketing Classification)
```bash
# Full pipeline: validate → build features → train → evaluate
bash scripts/ml_pipeline.sh

# Or individual commands
python -m nora ml validate --input data/bank-full.csv --schema config/schema.yaml
python -m nora ml build-features --input data/bank-full.csv --output data/processed/bank_marketing --seed 42
python -m nora ml train --input data/processed/bank_marketing/train.parquet --config config/model_params.yaml --output models/model.pkl
python -m nora ml evaluate --model models/model.pkl --test-set data/processed/bank_marketing/test.parquet --output reports/metrics.json
```

## Directory Structure

```
norm-aware-agent39/
├── configs/                    # Hydra configuration files
│   ├── base.yaml              # Main training config
│   ├── config.yaml            # Hydra composition root
│   ├── modes/                 # Execution modes (pipeline, agentic, norm_aware)
│   ├── regimes/               # Strictness levels (strict, balanced, exploratory)
│   ├── violations/            # Violation profiles for testing
│   ├── hydra/launcher/        # SLURM launcher config
│   ├── sweep/                 # Sweep configurations
│   └── ml/                    # ML pipeline configs
├── src/nora/                  # Core package
│   ├── train/                 # Training engine and runner
│   ├── models/                # Model definitions (ViT)
│   ├── data/                  # Data loaders (CIFAR-10, bank marketing)
│   ├── violations/            # Violation injection system
│   ├── norms/                 # Norm definitions
│   ├── agent/                 # Agent orchestration
│   └── cli.py                 # CLI entry point
├── scripts/                   # Experiment automation
│   ├── submit_sweep.sh        # SLURM sweep submission
│   ├── submit_single.sh       # Single SLURM job
│   ├── submit_array.sh        # SLURM array jobs
│   └── ml_pipeline.sh         # ML pipeline orchestration
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw datasets (CIFAR-10, bank-full.csv)
│   └── processed/             # Processed features
└── outputs/                   # Hydra outputs (gitignored)
```

## Configuration

NORA uses Hydra for hierarchical configuration composition:

1. **Base config** (`configs/base.yaml`) - Core training hyperparameters, dataset, model, optimizer, scheduler, AMP, reproducibility settings
2. **Mode** (`configs/modes/*.yaml`) - Execution mode: pipeline (standard), agentic (reactive), norm_aware (proactive)
3. **Regime** (`configs/regimes/*.yaml`) - Strictness level: strict, balanced, exploratory
4. **Violation** (`configs/violations/*.yaml`) - Optional violation injection for testing agent responses

Configuration is composed via `configs/config.yaml` with Hydra's defaults system:
```yaml
defaults:
  - base
  - mode: norm_aware      # Override with mode=<name>
  - regime: balanced      # Override with regime=<name>
  - optional violation: none  # Override with violation=<name>
```

Override any parameter from command line:
```bash
python -m nora train mode=agentic train.epochs=50 optimizer.lr=1e-3
```

## Norms

Norms are rules that the agent enforces:

- **Reproducibility** - Deterministic training, fixed seeds
- **Validity** - Training/eval mode correctness, augmentation leaks
- **Stability** - AMP NaN/Inf detection, checkpoint integrity
- **Reporting** - Metric consistency and logging requirements

See `norms/` for detailed definitions.

## Running Experiments

### Local Execution

**Single run:**
```bash
python -m nora train mode=norm_aware regime=balanced seed=42
```

**Parameter sweep (sequential):**
```bash
python -m nora train --multirun mode=pipeline,norm_aware seed=42,123,456
```

### SLURM Cluster Execution

**Before running on SLURM**, update your account in `configs/hydra/launcher/slurm.yaml`:
```yaml
account: m4439  # Your allocation
```experimental results:

```bash
python scripts/make_tables.py --run-dir outputs/
python scripts/make_figures.py --run-dir outputs/
```

Check violation events and norm compliance:
```bash
# View events from a run
cat outputs/<run_id>/norms/events.jsonl | jq .

# Aggregate metrics across seeds
python -m nora analyze --sweep-dir sweeps/<date>/<time>
```

**Multi-run sweep with Hydra SLURM launcher:**
```bash
python -m nora train \
    --multirun \
    hydra/launcher=slurm \
    mode=pipeline,agentic,norm_aware \
    regime=strict,balanced,exploratory \
    seed=42,123,456
```

**Full experimental matrix (162 jobs):**
```bash
python -m nora train \
    --multirun \
    hydra/launcher=slurm \
    mode=pipeline,agentic,norm_aware \
    regime=strict,balanced,exploratory \
    violation=none,nondeterminism,amp_nan,eval_mode_bug,aug_leak,checkpoint_incomplete \
    seed=42,123,456
```

**Monitor jobs:**
```bash
squeue --me
```

## Analysis

Generate tables and figures from runs:

```bash
python scripts/make_tables.py --run-dir runs/
python scripts/make_figures.py --run-dir runs/
```

## Citation

If you use NORA in your research, please cite:

```bibtex
[See CITATION.cff for details]
```

## License

See LICENSE file.
