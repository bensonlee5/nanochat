#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 6 (fallback): running baseline ratio candidates ..."
echo "Using fallback ratio: $LANE_B_DEFAULT_RATIO"

EXTRAPOLATION_ARGS=(
  --time-to-target-extrapolation "$LANE_B_TIME_TO_TARGET_EXTRAPOLATION"
)
if [ "$LANE_B_TIME_TO_TARGET_EXTRAPOLATION" = "power_law_recent_eval" ]; then
  if [ ! -f "$LANE_B_INFER_JSON" ]; then
    echo "Missing $LANE_B_INFER_JSON required for power-law extrapolation."
    echo "Run step 5 first or set LANE_B_TIME_TO_TARGET_EXTRAPOLATION=linear_recent_eval."
    exit 1
  fi

  infer_env="$("$PYTHON_BIN" - "$LANE_B_INFER_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
alpha = d.get("inferred_data_scaling_exponent")
if alpha is None:
    raise SystemExit("inferred_data_scaling_exponent missing from inference JSON")
print(f"ALPHA_DATA={alpha!r}")
calib_l_inf = d.get("calib_L_inf")
if calib_l_inf is not None:
    print(f"CALIB_L_INF={calib_l_inf!r}")
PY
)"
  if [ $? -ne 0 ]; then
    echo "Failed to parse extrapolation fields from $LANE_B_INFER_JSON."
    exit 1
  fi
  eval "$infer_env"

  : "${ALPHA_DATA:?missing inferred_data_scaling_exponent for power-law extrapolation}"
  EXTRAPOLATION_ARGS+=(
    --time-to-target-power-law-alpha "$ALPHA_DATA"
    --time-to-target-power-law-min-points "$LANE_B_TIME_TO_TARGET_POWER_LAW_MIN_POINTS"
    --time-to-target-power-law-max-points "$LANE_B_TIME_TO_TARGET_POWER_LAW_MAX_POINTS"
    --time-to-target-power-law-fit-r2-min "$LANE_B_TIME_TO_TARGET_POWER_LAW_FIT_R2_MIN"
  )
  if [ -n "${CALIB_L_INF:-}" ]; then
    EXTRAPOLATION_ARGS+=(--time-to-target-power-law-l-inf "$CALIB_L_INF")
  fi
fi

echo "run_id,seed,mode,chosen_ratio,status,log_path" > "$LANE_B_MANIFEST_CSV"

IFS=',' read -r -a RUN_SEEDS <<< "$LANE_B_RUN_SEEDS"
for seed in "${RUN_SEEDS[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi

  mode="fallback"
  ratio="$LANE_B_DEFAULT_RATIO"
  run_id="laneb_${mode}_s${seed}"
  log_path="$LANE_B_LOG_DIR/${run_id}.log"
  wandb_run="$LANE_B_WANDB_RUN_BASE"
  if [ "$wandb_run" != "dummy" ]; then
    wandb_run="${wandb_run}_${run_id}"
  fi

  echo "Running $run_id (ratio=$ratio)"
  set +e
  "$PYTHON_BIN" -m scripts.base_train \
    --depth "$LANE_B_DEPTH" \
    --aspect-ratio "$LANE_B_ASPECT_RATIO" \
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
    --time-to-target-metric "$LANE_B_TARGET_METRIC" \
    --time-to-target-threshold "$LANE_B_TARGET_THRESHOLD" \
    "${EXTRAPOLATION_ARGS[@]}" \
    --target-param-data-ratio "$ratio" \
    --seed "$seed" \
    --run "$wandb_run" 2>&1 | tee "$log_path"
  train_exit=${PIPESTATUS[0]}
  set -e

  status="ok"
  if [ "$train_exit" -ne 0 ]; then
    status="failed"
    echo "WARNING: training failed for $run_id (exit=$train_exit)"
  fi

  echo "$run_id,$seed,$mode,$ratio,$status,$log_path" >> "$LANE_B_MANIFEST_CSV"
done

echo "Wrote fallback manifest: $LANE_B_MANIFEST_CSV"
