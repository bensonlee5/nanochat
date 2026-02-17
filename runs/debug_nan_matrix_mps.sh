#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/baseline_common.sh"
baseline_init

echo "NaN debug matrix (MPS) ..."

require_profile_match=1
if [ "${NAN_DEBUG_REQUIRE_README_PROFILE:-1}" != "1" ]; then
  require_profile_match=0
fi

if [ "$require_profile_match" -eq 1 ]; then
  mismatched=0
  check_var() {
    local name="$1"
    local expected="$2"
    local actual="${!name}"
    if [ "$actual" != "$expected" ]; then
      echo "Profile mismatch: $name=$actual (expected $expected)"
      mismatched=1
    fi
  }

  # README.md -> Research quick experimentation profile
  check_var BASELINE_DEPTH "12"
  check_var BASELINE_HEAD_DIM "128"
  check_var BASELINE_MAX_SEQ_LEN "2048"
  check_var BASELINE_DEVICE_BATCH_SIZE "32"
  check_var BASELINE_TOTAL_BATCH_SIZE "-1"
  check_var BASELINE_NUM_ITERATIONS "-1"
  check_var BASELINE_EVAL_EVERY "250"
  check_var BASELINE_EVAL_TOKENS "20971520"
  check_var BASELINE_CORE_METRIC_EVERY "999999"
  check_var BASELINE_DEFAULT_RATIO "10.5"

  if [ "$mismatched" -ne 0 ]; then
    echo "Skipping NaN debug matrix because baseline settings do not match README Research defaults."
    echo "Set NAN_DEBUG_REQUIRE_README_PROFILE=0 to bypass this guard."
    exit 2
  fi
fi

NAN_DEBUG_WORK_DIR="${NAN_DEBUG_WORK_DIR:-$NANOCHAT_BASE_DIR/nan_debug_$(date +%Y%m%d_%H%M%S)}"
LOGS_DIR="$NAN_DEBUG_WORK_DIR/logs"
MANIFEST_CSV="$NAN_DEBUG_WORK_DIR/manifest.csv"
SUMMARY_CSV="$NAN_DEBUG_WORK_DIR/summary.csv"
RESULTS_FILE="$LOGS_DIR/results.csv"
NAN_DEBUG_DEVICE_TYPE="${NAN_DEBUG_DEVICE_TYPE:-mps}"
NAN_DEBUG_VARIANTS="${NAN_DEBUG_VARIANTS:-control,no_muon,no_compile}"

mkdir -p "$LOGS_DIR"
echo "run_id,seed,variant,exit_code,log_path" > "$MANIFEST_CSV"

IFS=',' read -r -a seeds <<< "$BASELINE_SEEDS"
IFS=',' read -r -a variants <<< "$NAN_DEBUG_VARIANTS"

any_failed=0

for variant in "${variants[@]}"; do
  variant="${variant//[[:space:]]/}"
  if [ -z "$variant" ]; then
    continue
  fi

  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    if [ -z "$seed" ]; then
      continue
    fi

    run_id="nan_${variant}_s${seed}"
    log_path="$LOGS_DIR/${run_id}_train.log"
    matrix_lr=""
    compile_disable=""

    case "$variant" in
      control)
        ;;
      no_muon)
        matrix_lr="0"
        ;;
      no_compile)
        compile_disable="1"
        ;;
      *)
        echo "Unknown variant '$variant' (allowed: control,no_muon,no_compile)"
        exit 1
        ;;
    esac

    echo "Running variant=$variant seed=$seed ..."

    set +e
    SEED="$seed" \
    RUN_LABEL="$run_id" \
    WANDB_RUN="dummy" \
    RESULTS_DIR="$LOGS_DIR" \
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
    DEVICE_TYPE="$NAN_DEBUG_DEVICE_TYPE" \
    MATRIX_LR="$matrix_lr" \
    TORCH_COMPILE_DISABLE="$compile_disable" \
      bash "$SCRIPT_DIR/speedrun_theory.sh"
    run_exit=$?
    set -e

    echo "${run_id},${seed},${variant},${run_exit},${log_path}" >> "$MANIFEST_CSV"
    if [ "$run_exit" -ne 0 ]; then
      any_failed=1
    fi
  done
done

"$PYTHON_BIN" -m scripts.debug_nan_report \
  --manifest "$MANIFEST_CSV" \
  --output-csv "$SUMMARY_CSV"

echo ""
echo "Artifacts:"
echo "  manifest: $MANIFEST_CSV"
echo "  summary:  $SUMMARY_CSV"
echo "  logs:     $LOGS_DIR"
echo "  speedrun results.csv: $RESULTS_FILE"

if [ "$any_failed" -ne 0 ]; then
  echo "One or more runs failed (non-zero exit). See summary + logs."
  exit 2
fi
