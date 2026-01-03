#!/bin/bash
# Compare all 3 modes: pipeline, agentic, norm_aware

#SBATCH --job-name=mode-comparison
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=3
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/mode-comparison-%j.out
#SBATCH --error=logs/mode-comparison-%j.err

mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2

echo "=== Comparing 3 Execution Modes ==="
echo "Job ID: $SLURM_JOB_ID"
echo ""

BASE_OUT="runs/mode_comparison_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

# 3 modes with balanced regime
CONFIGS=(
    "modes=pipeline regime=balanced seed=42 hydra.run.dir=$BASE_OUT/0_pipeline"
    "modes=agentic regime=balanced seed=42 hydra.run.dir=$BASE_OUT/1_agentic"
    "modes=norm_aware regime=balanced seed=42 hydra.run.dir=$BASE_OUT/2_norm_aware"
)

LABELS=(
    "Pipeline (baseline)"
    "Agentic (performance optimization)"
    "Norm-aware (reproducibility enforcement)"
)

echo "Running 3 modes on 3 GPUs..."
echo "Output: $BASE_OUT"
echo ""

for i in {0..2}; do
    GPU_ID=$i
    CONFIG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"
    
    echo "[$i] $LABEL"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m nora train \
        $CONFIG \
        train.epochs=30 \
        optimizer.lr=0.01 \
        &
done

echo ""
echo "Waiting for all 3 modes to complete..."
wait

echo ""
echo "=== Mode Comparison Results ==="
for i in {0..2}; do
    LABEL="${LABELS[$i]}"
    RESULT_DIR=$(ls -d $BASE_OUT/*$i* 2>/dev/null | head -1)
    
    if [ -d "$RESULT_DIR" ]; then
        echo ""
        echo "Mode $i: $LABEL"
        echo "  Directory: $RESULT_DIR"
        
        if [ -f "$RESULT_DIR/summary.json" ]; then
            python -c "
import json
s = json.load(open('$RESULT_DIR/summary.json'))
print(f'  Best val acc: {s.get(\"best_val_acc\", 0):.2f}%')
print(f'  Best epoch: {s.get(\"best_epoch\", 0)}')
print(f'  Total epochs: {s.get(\"epochs\", 0)}')
" 2>/dev/null || echo "  (summary not available)"
        fi
        
        if [ -f "$RESULT_DIR/norms/agent_metrics.json" ]; then
            echo "  Agent:"
            python -c "
import json
m = json.load(open('$RESULT_DIR/norms/agent_metrics.json'))
print(f'    Norm-aware: {m.get(\"norm_aware\", \"N/A\")}')
print(f'    Decisions: {m.get(\"total_decisions\", 0)}')
if not m.get('norm_aware', True):
    print(f'    LR adjustments: {m.get(\"lr_adjustments\", 0)}')
    print(f'    Early stop: {m.get(\"early_stop_recommended\", False)}')
else:
    print(f'    Auto-fixes: {m.get(\"auto_fixes\", 0)}')
" 2>/dev/null || echo "    (no agent metrics)"
        fi
    fi
done

echo ""
echo "Full results: $BASE_OUT"
echo ""
echo "Compare metrics:"
echo "  for d in $BASE_OUT/*/; do echo \"\$(basename \$d)\"; cat \$d/summary.json | python -m json.tool | grep -E 'best_val_acc|best_epoch'; done"
