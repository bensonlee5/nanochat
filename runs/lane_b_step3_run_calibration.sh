#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 3: running calibration runs ..."
echo "Calibration iterations: $LANE_B_CALIB_ITERS"

IFS=',' read -r -a CALIB_ITERS <<< "$LANE_B_CALIB_ITERS"
for it in "${CALIB_ITERS[@]}"; do
  it="${it//[[:space:]]/}"
  if [ -z "$it" ]; then
    continue
  fi

  RUN_ID="laneb_calib_${it}"
  LOG_PATH="$LANE_B_LOG_DIR/${RUN_ID}.log"
  WANDB_RUN="$LANE_B_WANDB_RUN_BASE"
  if [ "$WANDB_RUN" != "dummy" ]; then
    WANDB_RUN="${WANDB_RUN}_${RUN_ID}"
  fi

  echo "Running calibration: iter=$it, seed=$LANE_B_CALIB_SEED, log=$LOG_PATH"

  set +e
  "$PYTHON_BIN" -m scripts.base_train \
    --depth "$LANE_B_DEPTH" \
    --head-dim "$LANE_B_HEAD_DIM" \
    --window-pattern "$LANE_B_WINDOW_PATTERN" \
    --max-seq-len "$LANE_B_MAX_SEQ_LEN" \
    --device-batch-size "$LANE_B_DEVICE_BATCH_SIZE" \
    --total-batch-size "$LANE_B_TOTAL_BATCH_SIZE" \
    --eval-every "$LANE_B_EVAL_EVERY" \
    --eval-tokens "$LANE_B_EVAL_TOKENS" \
    --core-metric-every -1 \
    --sample-every -1 \
    --save-every -1 \
    --num-iterations "$it" \
    --time-to-target-metric "$LANE_B_TARGET_METRIC" \
    --time-to-target-threshold "$LANE_B_TARGET_THRESHOLD" \
    --target-param-data-ratio "$LANE_B_DEFAULT_RATIO" \
    --seed "$LANE_B_CALIB_SEED" \
    --run "$WANDB_RUN" 2>&1 | tee "$LOG_PATH"
  TRAIN_EXIT=${PIPESTATUS[0]}
  set -e

  if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Calibration run failed: $RUN_ID"
    exit "$TRAIN_EXIT"
  fi
done

echo "Calibration runs completed. Logs in: $LANE_B_LOG_DIR"
