#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 6: running inferred + confirmation candidates ..."

if [ ! -f "$LANE_B_INFER_JSON" ]; then
  echo "Missing $LANE_B_INFER_JSON. Run step 5 first."
  exit 1
fi

if ! infer_env="$("$PYTHON_BIN" - "$LANE_B_INFER_JSON" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if d.get("solve_status") != "ok":
    raise SystemExit(f"Cannot run candidates: solve_status={d.get('solve_status')}")
if d.get("alpha_data_plausible") != "yes":
    raise SystemExit(
        "Cannot run candidates: alpha_data_plausible="
        f"{d.get('alpha_data_plausible')}"
    )
print(f"INFERRED_RATIO={d['inferred_target_param_data_ratio']}")
print(f"CONFIRMATION_RATIO={d['confirmation_ratio']}")
print(f"ALPHA_DATA={d['inferred_data_scaling_exponent']}")
print(f"INFERRED_TOKENS={d['inferred_target_tokens']}")
PY
)"; then
  echo "ERROR: inference solve is not usable; skipping candidate runs."
  echo "Recovery options:"
  echo "  1) Run fallback baseline path: bash runs/lane_b_step6_run_fallback_baseline.sh"
  echo "  2) Improve stats/calibration settings and rerun from step 2 or step 5"
  exit 2
fi

eval "$infer_env"

: "${INFERRED_RATIO:?missing INFERRED_RATIO from inference JSON}"
: "${CONFIRMATION_RATIO:?missing CONFIRMATION_RATIO from inference JSON}"

echo "inferred_ratio=$INFERRED_RATIO"
echo "confirmation_ratio=$CONFIRMATION_RATIO"

echo "run_id,seed,mode,chosen_ratio,status,log_path" > "$LANE_B_MANIFEST_CSV"

IFS=',' read -r -a RUN_SEEDS <<< "$LANE_B_RUN_SEEDS"
for seed in "${RUN_SEEDS[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi

  for mode in inferred confirmation; do
    if [ "$mode" = "inferred" ]; then
      ratio="$INFERRED_RATIO"
    else
      ratio="$CONFIRMATION_RATIO"
    fi

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
done

echo "Wrote: $LANE_B_MANIFEST_CSV"
