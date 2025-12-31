# Quick Reference: NORA + ML Pipeline

## Installation & Setup
```bash
cd /pscratch/sd/a/alazar/norm-aware-agent39/nora-pipeline
pip install -e .
```

## ML Pipeline (Bank Marketing Classification)

### Complete Pipeline (One Command)
```bash
bash scripts/ml_pipeline.sh data/bank-marketing.csv ./output
```

### Step-by-Step Commands
```bash
# 1. Validate data against schema
python -m nora ml validate --input data.csv --schema configs/ml/schema.yaml

# 2. Build features (encode, split, prepare)
python -m nora ml build-features --input data.csv --output data/features --seed 42

# 3. Train CatBoost model
python -m nora ml train --input data/features/train.parquet \
    --config configs/ml/model_params.yaml --output model.pkl

# 4. Evaluate on test set
python -m nora ml evaluate --model model.pkl \
    --test-set data/features/test.parquet --output metrics.json
```

## Vision Model Training (NORA)

### Single Run
```bash
bash scripts/run_one.sh configs/base.yaml norm_aware balanced 42
```

### Multiple Seeds
```bash
bash scripts/run_seeds.sh configs/base.yaml norm_aware balanced 5
```

### Full Matrix (mode × regime × violation × seeds)
```bash
bash scripts/run_matrix.sh
# Creates experiments for:
# - Modes: pipeline, agentic, norm_aware
# - Regimes: strict, balanced, exploratory
# - Violations: none, nondeterminism, amp_nan, eval_mode_bug, aug_leak, checkpoint_incomplete
# - Seeds: 3 runs per config
```

## Configuration Files

### Base Configuration
- **`configs/base.yaml`** - Default settings (data, model, training, AMP, logging)
- **`configs/ml.yaml`** - ML pipeline settings

### Regimes (Norm Enforcement)
- **`configs/regimes/strict.yaml`** - All norms critical
- **`configs/regimes/balanced.yaml`** - Moderate enforcement
- **`configs/regimes/exploratory.yaml`** - Permissive mode

### Modes (Execution Style)
- **`configs/modes/pipeline.yaml`** - Standard training, no agent
- **`configs/modes/agentic.yaml`** - Reactive agent
- **`configs/modes/norm_aware.yaml`** - NORA agent with norms

### Violations (Injected Bugs)
- **`configs/violations/none.yaml`** - Clean training
- **`configs/violations/nondeterminism.yaml`** - Disable determinism
- **`configs/violations/amp_nan.yaml`** - AMP numerical issues
- **`configs/violations/eval_mode_bug.yaml`** - Eval mode not enforced
- **`configs/violations/aug_leak.yaml`** - Augmentation on test set
- **`configs/violations/checkpoint_incomplete.yaml`** - Missing checkpoint state

### ML Configs
- **`configs/ml/model_params.yaml`** - CatBoost hyperparameters
- **`configs/ml/schema.yaml`** - Data validation rules

## Norms (Rules Being Checked)

### Reproducibility
- `deterministic_seed`: Fixed seed at training start
- `deterministic_algorithms`: CUDA determinism enabled
- `no_dataloader_workers`: num_workers = 0

### Validity
- `eval_mode_enforced`: model.eval() during evaluation
- `no_augmentation_on_eval`: No augmentation on test data
- `train_mode_enforced`: model.train() during training

### Stability
- `amp_nan_guard`: No NaN/Inf in loss/gradients
- `checkpoint_completeness`: Full state saved (model, optimizer, scheduler, epoch)

### Reporting
- `metric_consistency`: All metrics logged every epoch
- `log_format`: Structured JSONL logging

## Output Structure

```
runs/
├── single_YYYYMMDD_HHMMSS/    # Single run output
│   ├── checkpoints/
│   ├── logs.jsonl
│   └── metrics.json
├── seeds_YYYYMMDD_HHMMSS/     # Multi-seed runs
│   ├── seed_1/, seed_2/, ...
├── matrix_YYYYMMDD_HHMMSS/    # Full matrix
│   ├── pipeline_strict_none_seed1/
│   ├── agentic_balanced_nondeterminism_seed1/
│   └── ...
└── (git-ignored)

ml_pipeline_output/
├── features/
│   ├── train.parquet
│   └── test.parquet
├── model.pkl
└── metrics.json
```

## Key CLI Entry Points

```bash
# Vision model training
python -m nora train [--config] [--regime] [--mode] [--violation] [--seed]

# ML pipeline
python -m nora ml validate --input <file> --schema <file>
python -m nora ml build-features --input <file> --output <dir> --seed <int>
python -m nora ml train --input <file> --config <file> --output <file>
python -m nora ml evaluate --model <file> --test-set <file> --output <file>
```

## Data Format

### Input (CSV)
Bank Marketing dataset with target column 'y' (yes/no)

### Intermediate (Parquet)
- `train.parquet`: Features + target for training
- `test.parquet`: Features + target for evaluation

### Output (JSON)
```json
{
  "auc": 0.85,
  "accuracy": 0.75,
  "precision": 0.80,
  "recall": 0.70,
  "f1": 0.75,
  "dataset_size": 4119
}
```

## Documentation Files

- **`README.md`** - Main documentation
- **`INTEGRATION_SUMMARY.md`** - What was integrated
- **`ML_INTEGRATION.md`** - Detailed ML pipeline guide
- **`CITATION.cff`** - Citation information
- **`LICENSE`** - MIT License

## Helpful Commands

```bash
# Test installation
pip install -e .

# Show help
python -m nora --help
python -m nora ml --help
python -m nora train --help

# List configurations
ls -la configs/
ls -la configs/{regimes,modes,violations,ml}/

# View logs from a run
cat runs/*/logs.jsonl | jq '.'  # Pretty-print JSON logs

# Generate analysis
python scripts/make_tables.py --run-dir runs/
python scripts/make_figures.py --run-dir runs/
```

## Project Structure Summary

```
nora-pipeline/
├── Vision Models: CIFAR-10, DeiT, ViT
├── ML Pipeline: CatBoost, Bank Marketing
├── Norms System: 10+ validation rules
├── Agent: Reactive & Norm-Aware
├── CLI: Unified command interface
└── Experiments: Full matrix generation
```

**Total:** 84 files, 19 directories, 2 pipelines unified
