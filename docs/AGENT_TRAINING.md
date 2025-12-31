# Agent-Based Training with Norm Enforcement

This document describes how to run agent-based training with balanced and strict norm enforcement in NORA.

## Overview

NORA supports three **execution modes** that can be combined with three **enforcement regimes**:

### Execution Modes
- **pipeline**: Standard training without agent (baseline)
- **agentic**: Reactive agent for training optimization
- **norm_aware**: Agent with reproducibility norm enforcement (NORA focus)

### Enforcement Regimes
- **strict**: Zero tolerance - halt on any violation, no auto-remediation
- **balanced**: Moderate - auto-remediate up to 1 low-severity violation
- **exploratory**: Very tolerant - log-only, no remediation

## Quick Start

### Run Agent-Based Training (Balanced Regime)
```bash
python -m nora train \
    modes=norm_aware \
    regime=balanced \
    train.epochs=50
```

**Expected behavior:**
- Agent monitors for reproducibility violations
- Automatically remediates violations (missing seed, non-determinism, etc.)
- Continues training after remediation
- Saves agent decisions to `runs/*/norms/agent_decisions.jsonl`

### Run Strict Norm Enforcement
```bash
python -m nora train \
    modes=norm_aware \
    regime=strict \
    violation=missing_seed \
    train.epochs=50
```

**Expected behavior:**
- Agent detects missing seed violation at startup
- Halts training immediately with error
- Suggests fix in error message
- No remediation attempted

## Agent Behavior by Regime

### Balanced Regime
```yaml
regime: balanced

agent:
  halt_on_violations: false      # Don't halt
  auto_fix: true                  # Auto-remediate
  violation_tolerance: 1          # Allow 1 low-severity issue
  max_retries: 3

repro:
  deterministic: true
  cudnn_benchmark: false
```

**Violations detected at startup:**
1. Missing seed → Sets seed=42, enables determinism
2. Non-determinism → Disables cuDNN benchmark
3. Untracked config → Enables config persistence

**Result:** Training continues with fixes applied

### Strict Regime
```yaml
regime: strict

agent:
  halt_on_violations: true       # Halt immediately
  auto_fix: false                 # No remediation
  violation_tolerance: 0          # Zero tolerance
  max_retries: 1

repro:
  deterministic: true
  cudnn_benchmark: false
```

**Violations detected at startup:**
1. Any violation → Training halts with error
2. Error message suggests fix but doesn't apply it
3. User must fix manually and re-run

**Result:** Training blocked until violations resolved

## Testing Agent Enforcement

### Single GPU Test
```bash
bash scripts/test_agent_enforcement.sh
```

Runs 3 sequential tests:
1. Balanced + nondeterminism (should auto-fix)
2. Strict + missing seed (should halt)
3. Agentic + config tracking (should auto-fix)

### Multi-GPU Test (SLURM)
```bash
sbatch scripts/submit_agent_enforcement_test.sh
```

Runs 4 tests in parallel:
1. Balanced + nondeterminism
2. Strict + missing seed (halts)
3. Balanced + config tracking
4. Strict + clean (succeeds)

Results saved to: `runs/agent_test_YYYY-MM-DD_HH-MM-SS/`

## Understanding Agent Decisions

Each run generates three metrics files:

### 1. agent_decisions.jsonl
Contains all agent decisions with evidence:
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "event_type": "AGENT_DECISION",
  "agent_action": "auto_fix",
  "violation_name": "nondeterminism_enabled",
  "violation_severity": "high",
  "regime": "balanced",
  "message": "BALANCED: Auto-fixing nondeterminism_enabled, will validate",
  "suggested_fix": {
    "action": "restore_determinism",
    "parameters": {
      "deterministic": true,
      "cudnn_benchmark": false
    }
  }
}
```

### 2. agent_metrics.json
Summary statistics:
```json
{
  "total_decisions": 3,
  "auto_fixes": 2,
  "halts": 0,
  "warnings": 1,
  "remediation_success_count": 2,
  "regime": "balanced",
  "mode": "norm_aware"
}
```

### 3. reproducibility_metrics.json
Reproducibility monitoring metrics:
```json
{
  "violations_detected": 1,
  "detection_rate_pct": 100.0,
  "avg_time_to_detection_sec": 0.05,
  "severity_breakdown": {
    "high": 1
  }
}
```

## Configuration Examples

### Example 1: Balanced with Violation Injection
```bash
python -m nora train \
    modes=norm_aware \
    regime=balanced \
    violation=nondeterminism_enabled \
    seed=42 \
    train.epochs=10
```

Agent will:
- Detect non-determinism enabled
- Disable cuDNN benchmark
- Continue training with determinism restored

### Example 2: Strict with Multiple Seeds
```bash
python -m nora train \
    modes=norm_aware \
    regime=strict \
    seed=42,100,200 \
    train.epochs=10
```

Runs 3 seeds with strict enforcement - any violation halts all runs.

### Example 3: Agentic Mode (Performance-Focused)
```bash
python -m nora train \
    modes=agentic \
    regime=balanced \
    train.epochs=50 \
    hydra/launcher=submitit
```

Agent optimizes training performance with balanced tolerance.

## Comparing Regimes

Run all 3 regimes with same violation:
```bash
python -m nora train \
    --multirun \
    modes=norm_aware \
    regime=balanced,strict,exploratory \
    violation=missing_seed \
    seed=42 \
    train.epochs=1
```

Expected outcomes:
- **balanced**: Auto-fixes seed, trains successfully
- **strict**: Halts immediately
- **exploratory**: Logs warning, trains successfully

## Understanding Violations

Supported reproducibility violations:

| Violation | Severity | Balanced Fix | Strict |
|-----------|----------|-------------|--------|
| `missing_seed` | medium | Sets seed=42 | Halts |
| `nondeterminism_enabled` | high | Disables benchmark | Halts |
| `untracked_config_change` | high | Enables persistence | Halts |
| `amp_nan` | high | Enables GradScaler | Halts |
| `eval_mode_bug` | high | Forces eval mode | Halts |
| `aug_leak` | medium | Disables aug on eval | Halts |
| `checkpoint_incomplete` | high | Saves full state | Halts |
| `none` | null | (no violation) | Succeeds |

## Agent Decision Flow

```
Training Startup
    ↓
Reproducibility Monitor
    ├─ Detect missing seed? ─→ Record violation
    ├─ Detect non-determinism? ─→ Record violation
    └─ Detect config change? ─→ Record violation
    ↓
Agent Decision
    ├─ Strict: Halt? ──→ YES ──→ Raise Error
    │
    ├─ Balanced: Auto-fix? ──→ YES ──→ Apply Fix ──→ Continue
    │
    └─ Exploratory: Log only ──→ Continue
    ↓
Training Loop
    ↓
Save Metrics
    ├─ agent_decisions.jsonl
    ├─ agent_metrics.json
    └─ reproducibility_metrics.json
```

## Integration with Hydra

The agent system is fully integrated with Hydra composition:

```yaml
defaults:
  - base
  - modes: norm_aware          # Selects norm_aware.yaml
  - regime: balanced           # Selects balanced.yaml
  - violations: none           # Selects none.yaml
```

Compose at runtime:
```bash
python -m nora train \
    modes=norm_aware \
    regime=strict \
    violation=nondeterminism_enabled
```

## Debugging Agent Behavior

### View Agent Decisions
```bash
cat runs/default/norms/agent_decisions.jsonl | python -m json.tool
```

### View Remediation Details
```bash
cat runs/default/norms/agent_decisions.jsonl \
  | grep -E '"action"|"suggested_fix"'
```

### Compare Regimes
```bash
for d in runs/multirun_*/*/; do
  echo "=== $(basename $(dirname $d)) ==="
  cat "$d/norms/agent_metrics.json" | python -m json.tool
done
```

## Next Steps

1. **Run single test**: `bash scripts/test_agent_enforcement.sh`
2. **Run SLURM tests**: `sbatch scripts/submit_agent_enforcement_test.sh`
3. **Run full matrix**: `sbatch scripts/submit_full_matrix_2node.sh`
4. **Analyze results**: `python -c "import json; ..."`

See [README.md](../README.md) for more information on NORA training.
