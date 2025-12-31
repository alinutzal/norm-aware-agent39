#!/bin/bash
# Run FULL experimental matrix across 2 nodes with 4 GPUs each
# Total: 3 modes × 3 regimes × 7 violations × 3 seeds = 189 configs
# But we'll run 8 configs at a time in parallel (4 per node, 4 GPUs per node)

#SBATCH --job-name=nora-full-matrix
#SBATCH --partition=regular
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/full-matrix-%j.out
#SBATCH --error=logs/full-matrix-%j.err

mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

BASE_OUT="runs/fullmatrix_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

echo "=== Running FULL experimental matrix across 2 nodes ==="
echo "Modes: pipeline, agentic, norm_aware (3)"
echo "Regimes: strict, balanced, exploratory (3)"
echo "Violations: none, nondeterminism, amp_nan, eval_mode_bug, aug_leak, checkpoint_incomplete, reporting_single_run (7)"
echo "Seeds: 42, 123, 456 (3)"
echo "Total configs: 189"
echo ""
echo "Output: $BASE_OUT"
echo "Nodes: $SLURM_NODELIST"
echo "Parallelism: 8 jobs at a time (4 per node)"
echo ""

# Get node list
mapfile -t NODES < <(scontrol show hostname $SLURM_NODELIST)
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

# Generate all configs
MODES=("pipeline" "agentic" "norm_aware")
REGIMES=("strict" "balanced" "exploratory")
VIOLATIONS=("none" "nondeterminism" "amp_nan" "eval_mode_bug" "aug_leak" "checkpoint_incomplete" "reporting_single_run")
SEEDS=("42" "123" "456")

# Create array of all config combinations
CONFIG_ARRAY=()
JOB_NUM=0
for MODE in "${MODES[@]}"; do
    for REGIME in "${REGIMES[@]}"; do
        for VIOLATION in "${VIOLATIONS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                CONFIG="modes=$MODE regime=$REGIME violation=$VIOLATION seed=$SEED hydra.run.dir=$BASE_OUT/${JOB_NUM}_${MODE}_${REGIME}_${VIOLATION}_seed${SEED}"
                CONFIG_ARRAY+=("$CONFIG")
                ((JOB_NUM++))
            done
        done
    done
done

echo "Generated ${#CONFIG_ARRAY[@]} configs"
echo ""

# Run in batches of 8 (4 per node, 4 GPUs per node)
BATCH_SIZE=8
TOTAL_CONFIGS=${#CONFIG_ARRAY[@]}

for ((START=0; START<TOTAL_CONFIGS; START+=BATCH_SIZE)); do
    END=$((START + BATCH_SIZE))
    if [ $END -gt $TOTAL_CONFIGS ]; then
        END=$TOTAL_CONFIGS
    fi
    
    echo "========================================"
    echo "Batch $((START / BATCH_SIZE + 1)): Configs $START-$((END-1))"
    echo "========================================"
    
    # Submit 4 jobs to each node
    JOB_IN_BATCH=0
    for ((IDX=START; IDX<END; IDX++)); do
        NODE_IDX=$((JOB_IN_BATCH / 4))  # 0 for first 4, 1 for next 4
        GPU_IDX=$((JOB_IN_BATCH % 4))   # 0-3 for GPU index
        
        if [ $NODE_IDX -eq 0 ]; then
            NODE=$NODE0
        else
            NODE=$NODE1
        fi
        
        CONFIG="${CONFIG_ARRAY[$IDX]}"
        
        echo "[$IDX/$TOTAL_CONFIGS] Running on $NODE GPU $GPU_IDX: $CONFIG"
        
        # Run in background using srun to ensure it runs on the correct node
        srun --exclusive -N1 -w $NODE --gpus-per-node=1 bash -c \
            "CUDA_VISIBLE_DEVICES=$GPU_IDX python -m nora train $CONFIG train.epochs=1" &
        
        ((JOB_IN_BATCH++))
    done
    
    echo "Waiting for batch to complete..."
    wait
    echo "Batch complete!"
    echo ""
done

echo "========================================"
echo "All jobs completed!"
echo "Results in: $BASE_OUT"
echo "========================================"

# Count completed runs
COMPLETED=$(find "$BASE_OUT" -name "metrics.json" | wc -l)
echo "Completed: $COMPLETED / 189"
echo ""
echo "To view results:"
echo "  find $BASE_OUT -name metrics.json"
echo "  find $BASE_OUT -name events.jsonl"
