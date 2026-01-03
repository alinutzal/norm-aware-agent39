#!/bin/bash
# Run all violations for Table 1: Detection and Remediation Performance
# Uses 3 nodes with 4 GPUs each (12 GPUs total)

#SBATCH --job-name=table1-violations
#SBATCH --partition=debug
#SBATCH --account=m4439
#SBATCH --constraint=gpu
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/table1-%j.out
#SBATCH --error=logs/table1-%j.err

mkdir -p logs

module load cudatoolkit/13.0 2>/dev/null || true
source .venv/bin/activate

echo "=== Running All Violations for Table 1 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo ""

BASE_OUT="runs/table1_$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BASE_OUT"

# All violations by category
VIOLATIONS=(
    # Reproducibility (3)
    "missing_seed"
    "nondeterminism_enabled"
    "untracked_config_change"
    
    # Experimental Validity (3)
    "eval_mode_bug"
    "aug_leak"
    "amp_nan"
    
    # Reporting Standards (2)
    "reporting_single_run"
    "checkpoint_incomplete"
    
    # Resource Governance (3)
    "excessive_batch_size"
    "inefficient_precision"
    "excessive_workers"
    
    # Baseline
    "none"
)

TOTAL_VIOLATIONS=${#VIOLATIONS[@]}
GPUS_AVAILABLE=$((SLURM_NNODES * 4))

echo "Total violations to test: $TOTAL_VIOLATIONS"
echo "Running in batches of $GPUS_AVAILABLE"
echo "Output directory: $BASE_OUT"
echo ""

# Function to run a violation on a specific GPU
run_violation() {
    local VIOLATION=$1
    local TASK_ID=$2
    local RUN_ID=$3
    
    echo "[Task $TASK_ID] Running violation: $VIOLATION"
    
    # Use srun with specific task ID - SLURM will assign GPU automatically
    srun --ntasks=1 --nodes=1 --gpus-per-task=1 --exact \
        python -m nora train \
            modes=norm_aware \
            regime=balanced \
            violation=$VIOLATION \
            train.epochs=5 \
            hydra.run.dir=$BASE_OUT/${RUN_ID}_${VIOLATION} \
            > $BASE_OUT/${RUN_ID}_${VIOLATION}.log 2>&1 &
}

# Run all violations in parallel
echo ""
echo "=== Running all violations in parallel ==="

RUN_ID=0
for VIOLATION in "${VIOLATIONS[@]}"; do
    run_violation "$VIOLATION" "$RUN_ID" "$(printf "%02d" $RUN_ID)"
    RUN_ID=$((RUN_ID + 1))
    sleep 0.1  # Small delay to stagger starts
done

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
wait

echo ""
echo "=== All violations completed ==="
echo ""

echo ""
echo "=== All violations completed ==="
echo ""

# Aggregate results
echo "Aggregating results..."
python -c "
import json
from pathlib import Path
from collections import defaultdict

base_out = Path('$BASE_OUT')

# Category mapping
categories = {
    'missing_seed': 'Reproducibility',
    'nondeterminism_enabled': 'Reproducibility',
    'untracked_config_change': 'Reproducibility',
    'eval_mode_bug': 'Experimental Validity',
    'aug_leak': 'Experimental Validity',
    'amp_nan': 'Experimental Validity',
    'reporting_single_run': 'Reporting Standards',
    'checkpoint_incomplete': 'Reporting Standards',
    'excessive_batch_size': 'Resource Governance',
    'inefficient_precision': 'Resource Governance',
    'excessive_workers': 'Resource Governance',
}

results = defaultdict(lambda: {'detected': 0, 'remediated': 0, 'total': 0})

# Collect metrics from each run - look for summary.json files
for summary_file in base_out.glob('*/summary.json'):
    try:
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Extract violation name from path
        violation_dir = summary_file.parent.name
        violation_name = violation_dir.split('_', 1)[1] if '_' in violation_dir else None
        
        # Skip if no recognized violation name
        if violation_name not in categories:
            continue
        
        category = categories[violation_name]
        
        # Get detection count from reproducibility_metrics (cap at 1 per run)
        repro_metrics = summary.get('reproducibility_metrics', {})
        detected = 1 if repro_metrics.get('detected_violations', 0) > 0 else 0
        
        # Get remediation count from agent_metrics (cap at 1 per run)
        agent_metrics = summary.get('agent_metrics', {})
        remediated = 1 if agent_metrics.get('remediation_success_count', 0) > 0 else 0
        
        results[category]['detected'] += detected
        results[category]['remediated'] += remediated
        results[category]['total'] += 1
        
    except Exception as e:
        print(f'Error processing {summary_file}: {e}')

# Print table
print('\\n=== Table 1: Violation Detection and Remediation Performance ===\\n')
print(f'{\"Norm Category\":<25} {\"Violations Injected\":<22} {\"Detection Rate (%)\":<20} {\"Remediation Success (%)\"}')
print('-' * 90)

total_detected = 0
total_remediated = 0
total_injected = 0

for category in ['Reproducibility', 'Experimental Validity', 'Reporting Standards', 'Resource Governance']:
    if category in results:
        r = results[category]
        total = r['total']
        detected = r['detected']
        remediated = r['remediated']
        
        detection_rate = (detected / total * 100) if total > 0 else 0
        remediation_rate = (remediated / total * 100) if total > 0 else 0
        
        print(f'{category:<25} {total:<22} {detection_rate:<20.1f} {remediation_rate:.1f}')
        
        total_detected += detected
        total_remediated += remediated
        total_injected += total

print('-' * 90)
overall_detection = (total_detected / total_injected * 100) if total_injected > 0 else 0
overall_remediation = (total_remediated / total_injected * 100) if total_injected > 0 else 0
print(f'{\"Overall\":<25} {total_injected:<22} {overall_detection:<20.1f} {overall_remediation:.1f}')
print()

# Save to JSON
output = {
    'by_category': dict(results),
    'overall': {
        'total_injected': total_injected,
        'total_detected': total_detected,
        'total_remediated': total_remediated,
        'detection_rate_pct': overall_detection,
        'remediation_rate_pct': overall_remediation,
    }
}

with open('$BASE_OUT/table1_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Results saved to: $BASE_OUT/table1_results.json')
"

echo ""
echo "Full results in: $BASE_OUT"
echo ""
echo "View individual logs:"
echo "  ls -1 $BASE_OUT/*.log"
echo ""
echo "View table results:"
echo "  cat $BASE_OUT/table1_results.json"
