# Generic Baseline Run Guide

Run from repo root:

```bash
cd /Users/bensonlee/dev/nanochat
```

## 0) One-time setup

```bash
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export BASELINE_WORK_DIR="$NANOCHAT_BASE_DIR/baseline_$(date +%Y%m%d_%H%M%S)"
export BASELINE_RESULTS_DIR="$BASELINE_WORK_DIR/results"

# Baseline profile defaults (override as needed)
# Defaults align with README "Research" quick experimentation profile (d12).
export BASELINE_DEPTH=12
export BASELINE_HEAD_DIM=128
export BASELINE_MAX_SEQ_LEN=2048
export BASELINE_DEVICE_BATCH_SIZE=32
export BASELINE_TOTAL_BATCH_SIZE=-1
export BASELINE_NUM_ITERATIONS=-1
export BASELINE_EVAL_EVERY=250
export BASELINE_EVAL_TOKENS=20971520
export BASELINE_CORE_METRIC_EVERY=999999
export BASELINE_DEFAULT_RATIO=10.5
export BASELINE_TARGET_METRIC="val_bpb"
export BASELINE_SEEDS="41,42,43"
export BASELINE_REQUIRED_SEEDS="$BASELINE_SEEDS"

# Threshold derivation rule
export BASELINE_MIN_SUCCESS=3
export BASELINE_THRESHOLD_OFFSET=0.02

# Where to export the threshold for downstream experiments
export BASELINE_EXPORT_THRESHOLD_VAR="TARGET_THRESHOLD"
export BASELINE_EXPORT_PREFIX="BASELINE"
```

## 1) Run true baseline seeds

```bash
bash runs/baseline_step1_run_true_baselines.sh
```

Outputs:
- `$BASELINE_RESULTS_DIR/results.csv`
- per-seed logs in `$BASELINE_RESULTS_DIR`

## 2) Derive threshold from successful baseline runs

```bash
bash runs/baseline_step2_compute_threshold.sh
source "$BASELINE_THRESHOLD_ENV"
```

Outputs:
- `$BASELINE_THRESHOLD_JSON`
- `$BASELINE_THRESHOLD_ENV`

Rule:
- `recommended_threshold = median(min_val_bpb over required seeds) + BASELINE_THRESHOLD_OFFSET`

Example:
- if `BASELINE_EXPORT_THRESHOLD_VAR=TARGET_THRESHOLD`, env file exports `TARGET_THRESHOLD=<value>`

## Run all baseline steps

```bash
bash runs/baseline_run_all.sh
source "$BASELINE_THRESHOLD_ENV"
```

## Optional: NaN debug matrix (MPS)

Only runs if baseline settings match the README Research quick-experiment profile.

```bash
bash runs/debug_nan_matrix_mps.sh
```

Outputs:
- `$NAN_DEBUG_WORK_DIR/manifest.csv`
- `$NAN_DEBUG_WORK_DIR/summary.csv`
- per-run logs in `$NAN_DEBUG_WORK_DIR/logs`
