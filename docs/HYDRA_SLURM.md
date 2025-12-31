# Hydra SLURM Launcher Setup

## Installation

```bash
pip install hydra-core hydra-submitit-launcher
```

## Usage

### Single Run (Local)
```bash
python -m nora train mode=norm_aware regime=balanced seed=42
```

### Single Run (SLURM)
```bash
sbatch scripts/submit_single.sh
```

### Parameter Sweep (SLURM)
```bash
python -m nora train \
    --multirun \
    hydra/launcher=slurm \
    mode=pipeline,agentic,norm_aware \
    regime=strict,balanced \
    seed=42,123,456
```

### Predefined Sweep Configs
```bash
# Violation study
python -m nora train \
    --config-name=sweep/violation_study \
    --multirun

# Mode comparison
python -m nora train \
    --config-name=sweep/mode_comparison \
    --multirun
```

### Array Jobs (SLURM native)
```bash
sbatch scripts/submit_array.sh
```

## Configuration Override

Override any config parameter from command line:
```bash
python -m nora train \
    train.epochs=100 \
    train.batch_size=512 \
    optimizer.lr=1e-3 \
    dataset.num_workers=16
```

## Sweep Patterns

Range: `seed=range(0,10)`
Choice: `mode=pipeline,agentic,norm_aware`
Grid: Hydra automatically creates cartesian product

## Output Structure

```
runs/
  2025-12-31_12-00-00__deit_tiny__cifar10__norm_aware__balanced__none__seed42/
    best.pt
    last.pt
    metrics.json
    summary.json
    .hydra/
      config.yaml
      overrides.yaml
```

## SLURM Configuration

Edit `configs/hydra/launcher/slurm.yaml` for your HPC system:
- partition
- account
- time limits
- resource requests
