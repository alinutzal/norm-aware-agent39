#!/bin/bash
# Run a single experiment with one seed

set -e

CONFIG_FILE=${1:-configs/base.yaml}
MODE=${2:-pipeline}
REGIME=${3:-balanced}
SEED=${4:-42}

echo "Running single experiment"
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Regime: $REGIME"
echo "Seed: $SEED"

python -m nora.cli \
    --config "$CONFIG_FILE" \
    --mode "$MODE" \
    --regime "$REGIME" \
    --seed "$SEED" \
    --output-dir "runs/single_$(date +%Y%m%d_%H%M%S)"

echo "Run complete"
