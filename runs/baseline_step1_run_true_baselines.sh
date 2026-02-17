#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/baseline_common.sh"
baseline_init

echo "Step 1: running true baseline seeds ..."
baseline_info

any_failed=0
IFS=',' read -r -a BASELINE_SEED_LIST <<< "$BASELINE_SEEDS"
for seed in "${BASELINE_SEED_LIST[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi

  run_label="${BASELINE_RUN_LABEL_PREFIX}_s${seed}"
  wandb_run="$BASELINE_WANDB_RUN_BASE"
  if [ "$wandb_run" != "dummy" ]; then
    wandb_run="${wandb_run}_${run_label}"
  fi

  echo "Running baseline: seed=$seed, run_label=$run_label"
  set +e
  SEED="$seed" \
  RUN_LABEL="$run_label" \
  WANDB_RUN="$wandb_run" \
  RESULTS_DIR="$BASELINE_RESULTS_DIR" \
  DEPTH="$BASELINE_DEPTH" \
  HEAD_DIM="$BASELINE_HEAD_DIM" \
  MAX_SEQ_LEN="$BASELINE_MAX_SEQ_LEN" \
  DEVICE_BATCH_SIZE="$BASELINE_DEVICE_BATCH_SIZE" \
  TOTAL_BATCH_SIZE="$BASELINE_TOTAL_BATCH_SIZE" \
  NUM_ITERATIONS="$BASELINE_NUM_ITERATIONS" \
  EVAL_EVERY="$BASELINE_EVAL_EVERY" \
  EVAL_TOKENS="$BASELINE_EVAL_TOKENS" \
  CORE_METRIC_EVERY="$BASELINE_CORE_METRIC_EVERY" \
  TARGET_PARAM_DATA_RATIO="$BASELINE_DEFAULT_RATIO" \
  TARGET_METRIC="$BASELINE_TARGET_METRIC" \
  TARGET_THRESHOLD="-1" \
    bash "$SCRIPT_DIR/speedrun_theory.sh"
  run_exit=$?
  set -e

  if [ "$run_exit" -ne 0 ]; then
    any_failed=1
    echo "WARNING: baseline run failed for seed=$seed (exit=$run_exit)"
  fi
done

echo "Baseline runs finished. Results CSV: $BASELINE_RESULTS_CSV"
if [ "$any_failed" -ne 0 ]; then
  echo "At least one baseline run failed. Re-run this step for missing seeds before threshold computation."
  exit 2
fi
