# Example: Quick Start with Hydra + SLURM

## Local Testing
```bash
# Single run
python -m nora train

# With overrides
python -m nora train mode=pipeline seed=123 train.epochs=10
```

## SLURM Single Job
```bash
sbatch scripts/submit_single.sh
```

## SLURM Sweep (Hydra Multirun)
```bash
# Small sweep
python -m nora train --multirun \
    hydra/launcher=slurm \
    mode=norm_aware,pipeline \
    seed=42,123

# Full violation study
python -m nora train --multirun \
    hydra/launcher=slurm \
    violation=none,nondeterminism,amp_nan \
    mode=norm_aware \
    regime=balanced \
    seed=42,123,456
```

## SLURM Array Job (10 seeds)
```bash
sbatch scripts/submit_array.sh
```

## Check Job Status
```bash
squeue -u $USER
```

## View Results
```bash
# Latest run
ls -lt runs/ | head

# Specific run
cat runs/2025-12-31_*__seed42/summary.json
```
