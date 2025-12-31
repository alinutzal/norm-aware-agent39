# NORA: Norm-Aware Agent for Reliable ML Training

A comprehensive pipeline for training machine learning models with norm-aware agent orchestration, focusing on reproducibility, validity, and stability.

## Overview

NORA (Norm-Aware Agent) is a framework for detecting and mitigating training violations through a hierarchy of norms, an agent-based decision system, and comprehensive experiment logging. It integrates both vision models (CIFAR-10, ViT) and tabular ML models (CatBoost on Bank Marketing dataset).

## Quick Start

### NORA Vision Model Training
```bash
# Install dependencies
pip install -e .

# Run a single experiment
python -m nora train --config configs/base.yaml --mode norm_aware --regime balanced

# Run with a specific violation injected
python -m nora train --config configs/base.yaml --violation nondeterminism

# Generate experiment matrix (mode × regime × violation × seeds)
bash scripts/run_matrix.sh
```

### ML Pipeline (Bank Marketing Classification)
```bash
# Full pipeline: validate → build features → train → evaluate
bash scripts/ml_pipeline.sh data/bank-marketing.csv ./ml_output

# Or individual commands
python -m nora ml validate --input data/bank-marketing.csv --schema configs/ml/schema.yaml
python -m nora ml build-features --input data/bank-marketing.csv --output data/features --seed 42
python -m nora ml train --input data/features/train.parquet --config configs/ml/model_params.yaml --output models/model.pkl
python -m nora ml evaluate --model models/model.pkl --test-set data/features/test.parquet --output reports/metrics.json
```

## Directory Structure

- **configs/** - Configuration files for base, regimes, modes, and violations
- **norms/** - Norm definitions and registry
- **scripts/** - Experiment automation and analysis scripts
- **src/nora/** - Core package implementation
- **tests/** - Unit and integration tests
- **docker/** - Container definitions
- **runs/** - Experiment outputs (gitignored)

## Configuration

NORA uses hierarchical YAML configuration:

1. **Base config** (`configs/base.yaml`) - Shared defaults
2. **Regime** (`configs/regimes/*.yaml`) - Strictness level
3. **Mode** (`configs/modes/*.yaml`) - Execution mode
4. **Violation** (`configs/violations/*.yaml`) - Injected bugs

## Norms

Norms are rules that the agent enforces:

- **Reproducibility** - Deterministic training, fixed seeds
- **Validity** - Training/eval mode correctness, augmentation leaks
- **Stability** - AMP NaN/Inf detection, checkpoint integrity
- **Reporting** - Metric consistency and logging requirements

See `norms/` for detailed definitions.

## Running Experiments

### Single Run
```bash
bash scripts/run_one.sh configs/base.yaml norm_aware balanced
```

### Multiple Seeds
```bash
bash scripts/run_seeds.sh configs/base.yaml norm_aware balanced 10
```

### Full Matrix
```bash
bash scripts/run_matrix.sh
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
