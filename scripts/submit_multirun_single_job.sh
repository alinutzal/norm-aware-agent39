#!/bin/bash
# Submit ONE SLURM job with 4 GPUs that runs multirun sweep locally in parallel

#SBATCH --job-name=nora-sweep
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err

# Ensure logs directory exists
mkdir -p logs

# Load modules
module load cudatoolkit/13.0 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Export CUDA devices for parallel execution
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== Running multirun sweep on single 4-GPU node ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
echo ""

# Export base output directory
BASE_OUT="runs/multirun_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

echo "=== Running 4 configs in parallel on 4 GPUs ==="
echo "Output directory: $BASE_OUT"
echo ""

# Define configs to run with unique output subdirectories
CONFIGS=(
    "modes=pipeline regime=balanced seed=42 hydra.run.dir=$BASE_OUT/0_pipeline_balanced"
    "modes=pipeline regime=strict seed=42 hydra.run.dir=$BASE_OUT/1_pipeline_strict"
    "modes=norm_aware regime=balanced seed=42 hydra.run.dir=$BASE_OUT/2_norm_aware_balanced"
    "modes=norm_aware regime=strict seed=42 hydra.run.dir=$BASE_OUT/3_norm_aware_strict"
)

# Run each config on a different GPU in parallel
for i in {0..3}; do
    GPU_ID=$i
    CONFIG="${CONFIGS[$i]}"
    
    echo "Starting job $i on GPU $GPU_ID"
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m nora train \
        $CONFIG \
        train.epochs=50 \
        &
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all 4 training runs to complete..."
wait

echo ""
echo "All jobs completed!"
echo "Results in: $BASE_OUT"
echo ""
echo "View results:"
echo "  ls -la $BASE_OUT/"
echo "  find $BASE_OUT -name 'metrics.json' -exec cat {} \;"
