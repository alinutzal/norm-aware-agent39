#!/bin/bash
# Test agentic mode with performance-based decisions

set -e

echo "=== Testing Agentic Mode ==="
echo ""

source .venv/bin/activate
mkdir -p logs

echo "Test 1: Agentic + Balanced (should adjust LR if needed)"
echo "========================================================"
python -m nora train \
    modes=agentic \
    regime=balanced \
    seed=42 \
    train.epochs=20 \
    optimizer.lr=0.1 \
    2>&1 | tee logs/agentic_balanced.log

echo ""
echo "Test 2: Agentic + Strict (should early stop on overfitting)"
echo "============================================================"
python -m nora train \
    modes=agentic \
    regime=strict \
    seed=42 \
    train.epochs=50 \
    optimizer.lr=0.001 \
    2>&1 | tee logs/agentic_strict.log

echo ""
echo "Test 3: Compare all 3 modes"
echo "============================"
python -m nora train \
    --multirun \
    modes=pipeline,agentic,norm_aware \
    regime=balanced \
    seed=42 \
    train.epochs=10

echo ""
echo "=== Tests Complete ==="
echo ""
echo "View logs:"
echo "  cat logs/agentic_balanced.log | grep -i 'agentic'"
echo "  cat logs/agentic_strict.log | grep -i 'agentic'"
echo ""
echo "View agent decisions:"
echo "  find runs -name 'agent_decisions.jsonl' -exec cat {} \;"
