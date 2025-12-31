#!/bin/bash
# Test agent-based training with balanced and strict norm enforcement

set -e

echo "=== Testing Agent-Based Training ==="
echo ""

# Activate environment
source .venv/bin/activate

mkdir -p logs

echo "Test 1: Balanced norm enforcement (auto-remediate)"
echo "=================================================="
python -m nora train \
    modes=norm_aware \
    regime=balanced \
    violation=nondeterminism_enabled \
    seed=42 \
    train.epochs=3 \
    2>&1 | tee logs/agent_balanced.log

echo ""
echo "Test 1 Results:"
echo "- Should detect nondeterminism violation"
echo "- Should auto-fix and continue training"
echo ""

echo "Test 2: Strict norm enforcement (halt on violation)"
echo "===================================================="
python -m nora train \
    modes=norm_aware \
    regime=strict \
    violation=missing_seed \
    train.epochs=3 \
    2>&1 | tee logs/agent_strict.log || true

echo ""
echo "Test 2 Results:"
echo "- Should detect missing_seed violation"
echo "- Should halt immediately (error expected)"
echo ""

echo "Test 3: Agentic mode with balanced regime (agent decides and acts)"
echo "=================================================================="
python -m nora train \
    modes=agentic \
    regime=balanced \
    violation=untracked_config_change \
    seed=42 \
    train.epochs=3 \
    2>&1 | tee logs/agent_agentic.log

echo ""
echo "Test 3 Results:"
echo "- Agentic mode with balanced: agent remediates config tracking"
echo ""

echo "=== All tests complete ==="
echo ""
echo "View results:"
echo "  cat logs/agent_balanced.log"
echo "  cat logs/agent_strict.log"
echo "  cat logs/agent_agentic.log"
