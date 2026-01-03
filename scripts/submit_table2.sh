#!/bin/bash
# Run Table 2 modes x seeds in a single SLURM job

#SBATCH --job-name=table2-comprehensive
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --account=m4439_g
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/table2-%j.out
#SBATCH --error=logs/table2-%j.err

set -euo pipefail
mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

# Use absolute path to Python from venv
PYTHON_BIN="$(pwd)/.venv/bin/python"

# Config
SEEDS="1 2 3 4 5 6 7 8 9 10"          # 10 seeds for robustness
EPOCHS=50                              # full training
BASE_OUT="runs/table2_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

# SLURM defaults for interactive/local runs
JOB_ID=${SLURM_JOB_ID:-local}
NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

# Modes to run
MODES=(pipeline agentic norm_aware) #
REGIME=balanced

# All 8 violations to test (each mode runs clean + each violation)
VIOLATIONS=(
  "none"                    # baseline/clean
  "missing_seed"            # reproducibility
  "aug_leak"                # validity
  "eval_mode_bug"           # validity
  "amp_nan"                 # resource governance
  "excessive_batch_size"    # resource governance
  "checkpoint_incomplete"   # reporting
  "untracked_config_change" # reporting
)

VIOLATIONS=(
  "none" 
)

echo "=== Table 2: Comprehensive Mode & Violation Testing ==="
echo "Job ID: $JOB_ID"
echo "Output base: $BASE_OUT"
echo "Seeds: $SEEDS (10 seeds)"
echo "Modes: ${MODES[*]} (3 modes)"
echo "Violations: ${#VIOLATIONS[@]} conditions (clean + 8 violations)"
echo "Epochs: $EPOCHS"
echo "Nodes: $NODES"
echo "Total GPUs: $TOTAL_GPUS"
echo "Total runs: $((3 * ${#VIOLATIONS[@]} * 10)) = $(( 3 * $(echo ${VIOLATIONS[@]} | wc -w) * 10 ))"
echo ""

GPUS_AVAILABLE=$TOTAL_GPUS

run_config() {
  local mode=$1
  local seed=$2
  local violation=$3
  local task_id=$4
  
  local out_dir="$BASE_OUT/${mode}_${seed}_${violation}"
  echo "[task $task_id] mode=$mode seed=$seed violation=$violation -> $out_dir"
  
  # Assign GPU based on task_id (round-robin across available GPUs)
  local gpu=$((task_id % GPUS_PER_NODE))

  CUDA_VISIBLE_DEVICES=$gpu \
  "$PYTHON_BIN" -m nora train \
    modes=$mode regime=$REGIME seed=$seed \
    violation=$violation \
    hydra.run.dir=$out_dir \
    train.epochs=$EPOCHS \
    wandb.enabled=true \
    > $out_dir.log 2>&1 &
}

task_id=0
active_jobs=0
for mode in "${MODES[@]}"; do
  for violation in "${VIOLATIONS[@]}"; do
    for seed in $SEEDS; do
      run_config "$mode" "$seed" "$violation" "$task_id"
      task_id=$((task_id+1))

      # Keep at most GPUS_PER_NODE concurrent tasks; start next as soon as one finishes
      active_jobs=$((active_jobs+1))
      if (( active_jobs >= GPUS_PER_NODE )); then
        wait -n
        active_jobs=$((active_jobs-1))
      fi
    done
  done
done

echo ""
echo "Waiting for remaining runs to finish..."
wait
echo ""
echo "=== Aggregating Table 2 Results ==="
"$PYTHON_BIN" - <<PY
import json, glob, numpy as np, os
base = "$BASE_OUT"
modes = ['pipeline', 'agentic', 'norm_aware']
violations = ['none', 'missing_seed', 'nondeterminism_enabled', 'aug_leak', 'eval_mode_bug', 
              'amp_nan', 'excessive_batch_size', 'checkpoint_incomplete', 'untracked_config_change']

print(f"Base dir: {base}\n")
print(f"{'Mode':15} {'Violation':25} {'Runs':>4} {'Pass%':>8} {'Violations':>12} {'Fixes':>8} {'Mean Acc':>10} {'Std Acc':>10}")
print('-'*120)

for mode in modes:
  for viol in violations:
    # Match violation label (none -> clean label in dir)
    if viol == "none":
      runs = glob.glob(f"{base}/{mode}_*_clean") + glob.glob(f"{base}/{mode}_*_none")
    else:
      runs = glob.glob(f"{base}/{mode}_*_{viol}")
    
    summaries = [f"{r}/summary.json" for r in runs]
    vals = []
    total_violations = 0
    total_fixes = 0
    
    for f in summaries:
      try:
        s = json.load(open(f))
        vals.append(s.get('best_val_acc', 0))
        repro = s.get('reproducibility_metrics', {})
        agent = s.get('agent_metrics', {})
        total_violations += repro.get('detected_violations', 0)
        total_fixes += agent.get('auto_fixes', 0)
      except Exception:
        pass
    
    total = len(runs)
    done = len([f for f in summaries if os.path.exists(f)])
    pass_rate = (done / total * 100) if total else 0
    mean_acc = float(np.mean(vals)) if vals else 0.0
    std_acc = float(np.std(vals)) if vals else 0.0
    
    print(f"{mode:15} {viol:25} {total:4d} {pass_rate:8.1f} {total_violations:>12} {total_fixes:>8} {mean_acc:>10.2f} {std_acc:>10.4f}")

print('-'*120)

import os
PY

echo ""
echo "Done. Full results in: $BASE_OUT"
