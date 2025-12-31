#!/bin/bash
# Run multiple seeds with same configuration

set -e

CONFIG_FILE=${1:-configs/base.yaml}
MODE=${2:-pipeline}
REGIME=${3:-balanced}
NUM_SEEDS=${4:-5}

echo "Running $NUM_SEEDS seeds"
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Regime: $REGIME"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="runs/seeds_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

for SEED in $(seq 1 $NUM_SEEDS); do
    echo "Seed $SEED/$NUM_SEEDS"
    python -m nora.cli \
        --config "$CONFIG_FILE" \
        --mode "$MODE" \
        --regime "$REGIME" \
        --seed "$SEED" \
        --output-dir "$RUN_DIR/seed_$SEED" &
done

wait
echo "All seeds complete"
