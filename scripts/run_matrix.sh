#!/bin/bash
# Run full experimental matrix: mode × regime × violation × seeds

set -e

MODES=("pipeline" "agentic" "norm_aware")
REGIMES=("strict" "balanced" "exploratory")
VIOLATIONS=("none" "nondeterminism" "amp_nan" "eval_mode_bug" "aug_leak" "checkpoint_incomplete")
NUM_SEEDS=3

echo "Running full experimental matrix"
echo "Modes: ${MODES[@]}"
echo "Regimes: ${REGIMES[@]}"
echo "Violations: ${VIOLATIONS[@]}"
echo "Seeds per config: $NUM_SEEDS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MATRIX_DIR="runs/matrix_${TIMESTAMP}"
mkdir -p "$MATRIX_DIR"

TOTAL=$((${#MODES[@]} * ${#REGIMES[@]} * ${#VIOLATIONS[@]} * NUM_SEEDS))
COUNT=0

for MODE in "${MODES[@]}"; do
    for REGIME in "${REGIMES[@]}"; do
        for VIOLATION in "${VIOLATIONS[@]}"; do
            for SEED in $(seq 1 $NUM_SEEDS); do
                ((COUNT++))
                echo "[$COUNT/$TOTAL] Mode=$MODE Regime=$REGIME Violation=$VIOLATION Seed=$SEED"
                
                python -m nora.cli \
                    --config configs/base.yaml \
                    --mode "$MODE" \
                    --regime "$REGIME" \
                    --violation "$VIOLATION" \
                    --seed "$SEED" \
                    --output-dir "$MATRIX_DIR/${MODE}_${REGIME}_${VIOLATION}_seed${SEED}" &
                
                # Limit parallelism
                if (( COUNT % 4 == 0 )); then
                    wait
                fi
            done
        done
    done
done

wait
echo "Full matrix complete: $MATRIX_DIR"
