#!/bin/bash
# Full ML pipeline execution: validate → build features → train → evaluate

set -e

# Input data file (Bank Marketing dataset)
# bank_marketing.py saves the canonical CSV as bank-full.csv
INPUT_DATA=${1:-./data/bank-full.csv}
OUTPUT_DIR=${2:-./ml_pipeline_output}
# Standard location for processed features
FEATURES_DIR=./data/processed/bank_marketing

# Ensure output directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FEATURES_DIR"

echo "=========================================="
echo "NORA ML Pipeline Execution"
echo "=========================================="
echo "Input: $INPUT_DATA"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Validate data
echo "[1/4] Validating data..."
python -m nora ml validate \
    --input "$INPUT_DATA" \
    --schema configs/ml/schema.yaml

# Step 2: Build features
echo "[2/4] Building features..."
python -m nora ml build-features \
    --input "$INPUT_DATA" \
    --output "$FEATURES_DIR" \
    --seed 42

# Step 3: Train model
echo "[3/4] Training model..."
python -m nora ml train \
    --input "$FEATURES_DIR/train.parquet" \
    --config configs/ml/model_params.yaml \
    --output "$OUTPUT_DIR/model.pkl"

# Step 4: Evaluate model
echo "[4/4] Evaluating model..."
python -m nora ml evaluate \
    --model "$OUTPUT_DIR/model.pkl" \
    --test-set "$FEATURES_DIR/test.parquet" \
    --output "$OUTPUT_DIR/metrics.json"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
