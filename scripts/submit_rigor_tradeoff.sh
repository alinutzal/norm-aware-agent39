#!/bin/bash
# Run rigor-throughput sweeps (strict/balanced/exploratory) and plot figure

#SBATCH --job-name=rigor-tradeoff
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --nodes=3
#SBATCH --ntasks=9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/rigor-tradeoff-%j.out
#SBATCH --error=logs/rigor-tradeoff-%j.err

set -euo pipefail
mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

SEEDS="1 2 3"
EPOCHS=10
BASE_OUT="runs/rigor_tradeoff_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

REGIMES=(strict balanced exploratory)
MODE=norm_aware

echo "=== Rigor trade-off: regimes x seeds ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Output base: $BASE_OUT"
echo "Seeds: $SEEDS"
echo "Regimes: ${REGIMES[*]}"
echo ""

task_id=0
for regime in "${REGIMES[@]}"; do
  for seed in $SEEDS; do
    out_dir="$BASE_OUT/${regime}_${seed}"
    echo "[task $task_id] regime=$regime seed=$seed -> $out_dir"
    srun --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
         --exclusive bash -lc "python -m nora train \
           modes=$MODE regime=$regime seed=$seed \
           hydra.run.dir=$out_dir \
           train.epochs=$EPOCHS" \
           > $out_dir.log 2>&1 &
    task_id=$((task_id+1))
  done
done

echo ""
echo "Waiting for all runs to finish..."
wait

echo ""
echo "=== Aggregating rigor trade-off ==="
python3 scripts/make_rigor_tradeoff.py --base-dir "$BASE_OUT" || echo "aggregation failed"

echo ""
echo "Done. Full results in: $BASE_OUT"
