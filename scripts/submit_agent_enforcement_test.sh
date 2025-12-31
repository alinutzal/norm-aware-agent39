#!/bin/bash
# Run 4 experiments: norm_aware + (balanced|strict) with violation injection

#SBATCH --job-name=nora-agent-test
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/agent-test-%j.out
#SBATCH --error=logs/agent-test-%j.err

mkdir -p logs

# Load modules and activate environment
module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== Testing Agent-Based Training with Norm Enforcement ==="
echo "Job ID: $SLURM_JOB_ID"
echo ""

BASE_OUT="runs/agent_test_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

# 4 test configurations
CONFIGS=(
    # Balanced: auto-remediate nondeterminism
    "modes=norm_aware regime=balanced violation=nondeterminism_enabled seed=42 hydra.run.dir=$BASE_OUT/0_balanced_nondeterminism"
    
    # Strict: halt on missing seed
    "modes=norm_aware regime=strict violation=missing_seed seed=42 hydra.run.dir=$BASE_OUT/1_strict_missing_seed"
    
    # Balanced: auto-remediate config tracking
    "modes=norm_aware regime=balanced violation=untracked_config_change seed=42 hydra.run.dir=$BASE_OUT/2_balanced_config_tracking"
    
    # Strict: clean (no violations)
    "modes=norm_aware regime=strict violation=none seed=42 hydra.run.dir=$BASE_OUT/3_strict_clean"
)

LABELS=(
    "Balanced + Nondeterminism (should auto-fix)"
    "Strict + Missing Seed (should halt)"
    "Balanced + Config Change (should auto-fix)"
    "Strict + Clean (should succeed)"
)

echo "Running 4 agent-based training tests on 4 GPUs..."
echo "Output: $BASE_OUT"
echo ""

# Run in parallel on different GPUs
for i in {0..3}; do
    GPU_ID=$i
    CONFIG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    
    echo "[$i] $LABEL"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m nora train \
        $CONFIG \
        train.epochs=3 \
        &
done

echo ""
echo "Waiting for all 4 tests to complete..."
wait

echo ""
echo "=== Test Results ==="
for i in {0..3}; do
    LABEL="${LABELS[$i]}"
    RESULT_DIR=$(ls -d $BASE_OUT/*$i* 2>/dev/null | head -1)
    
    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo "Test $i: $LABEL"
        echo "  Directory: $RESULT_DIR"
        
        if [ -f "$RESULT_DIR/summary.json" ]; then
            echo "  Summary:"
            python -c "import json; s=json.load(open('$RESULT_DIR/summary.json')); print(f'    Best acc: {s.get(\"best_val_acc\", \"N/A\"):.2f}%')" 2>/dev/null || echo "    (no summary)"
        fi
        
        if [ -f "$RESULT_DIR/norms/agent_metrics.json" ]; then
            echo "  Agent Metrics:"
            python -c "import json; m=json.load(open('$RESULT_DIR/norms/agent_metrics.json')); print(f'    Total decisions: {m.get(\"total_decisions\", 0)}'); print(f'    Auto-fixes: {m.get(\"auto_fixes\", 0)}'); print(f'    Halts: {m.get(\"halts\", 0)}')" 2>/dev/null || echo "    (no agent metrics)"
        fi
        
        if [ -f "$RESULT_DIR/norms/reproducibility_metrics.json" ]; then
            echo "  Repro Metrics:"
            python -c "import json; r=json.load(open('$RESULT_DIR/norms/reproducibility_metrics.json')); print(f'    Violations detected: {r.get(\"violations_detected\", 0)}'); print(f'    Detection rate: {r.get(\"detection_rate_pct\", 0):.1f}%')" 2>/dev/null || echo "    (no repro metrics)"
        fi
    fi
done

echo ""
echo "Full results in: $BASE_OUT"
echo ""
echo "View all results:"
echo "  find $BASE_OUT -name '*.json' -exec echo {} \; -exec cat {} \; | head -100"
