#!/bin/bash
# Test reproducibility violations and measure agent response

#SBATCH --job-name=nora-repro-test
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/repro-test-%j.out

mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

BASE_OUT="runs/repro_test_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

echo "=== Testing Reproducibility Violations ==="
echo "Output: $BASE_OUT"
echo ""

# Test 1: Missing seed
echo "Test 1: Missing seed violation"
python -m nora train \
    violation=missing_seed \
    train.epochs=1 \
    hydra.run.dir="$BASE_OUT/test1_missing_seed" \
    2>&1 | tee -a "$BASE_OUT/test1.log"

sleep 2

# Test 2: Non-determinism enabled
echo ""
echo "Test 2: Non-determinism enabled violation"
python -m nora train \
    violation=nondeterminism_enabled \
    train.epochs=1 \
    hydra.run.dir="$BASE_OUT/test2_nondeterminism" \
    2>&1 | tee -a "$BASE_OUT/test2.log"

sleep 2

# Test 3: Untracked config change
echo ""
echo "Test 3: Untracked config change violation"
python -m nora train \
    violation=untracked_config_change \
    train.epochs=1 \
    train.batch_size=512 \
    hydra.run.dir="$BASE_OUT/test3_untracked_config" \
    2>&1 | tee -a "$BASE_OUT/test3.log"

echo ""
echo "=== Test Results ==="
echo "Metrics saved to:"
find "$BASE_OUT" -name "reproducibility_metrics.json" -exec echo "  {}" \;

echo ""
echo "Viewing results:"
echo "  cat $BASE_OUT/test*/norms/reproducibility_metrics.json | jq ."
