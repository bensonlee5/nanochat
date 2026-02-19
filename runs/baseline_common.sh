#!/bin/bash

# Shared defaults/helpers for generic baseline runs.

baseline_init() {
  set -euo pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [ ! -d ".venv" ]; then
    echo "Missing .venv in repo root. Create it first (e.g. uv venv && uv sync --extra cpu|gpu)."
    exit 1
  fi

  # shellcheck disable=SC1091
  source .venv/bin/activate

  export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

  local speedrun_ratio_default
  speedrun_ratio_default="$("$PYTHON_BIN" - "$script_dir/speedrun.sh" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8", errors="replace").read()
match = re.search(r"--target-param-data-ratio=([0-9]*\.?[0-9]+)", text)
print(match.group(1) if match else "8.25")
PY
)"
  export BASELINE_SPEEDRUN_DEFAULT_RATIO="${BASELINE_SPEEDRUN_DEFAULT_RATIO:-$speedrun_ratio_default}"

  export BASELINE_WORK_DIR="${BASELINE_WORK_DIR:-$NANOCHAT_BASE_DIR/baseline_run}"
  export BASELINE_RESULTS_DIR="${BASELINE_RESULTS_DIR:-$BASELINE_WORK_DIR/results}"
  mkdir -p "$BASELINE_WORK_DIR" "$BASELINE_RESULTS_DIR"
  export BASELINE_RESULTS_CSV="${BASELINE_RESULTS_CSV:-$BASELINE_RESULTS_DIR/results.csv}"

  # Model/profile defaults
  export BASELINE_DEPTH="${BASELINE_DEPTH:-12}"
  export BASELINE_ASPECT_RATIO="${BASELINE_ASPECT_RATIO:-64}"
  export BASELINE_HEAD_DIM="${BASELINE_HEAD_DIM:-128}"
  export BASELINE_MAX_SEQ_LEN="${BASELINE_MAX_SEQ_LEN:-2048}"

  # Run defaults
  export BASELINE_DEVICE_BATCH_SIZE="${BASELINE_DEVICE_BATCH_SIZE:-32}"
  export BASELINE_TOTAL_BATCH_SIZE="${BASELINE_TOTAL_BATCH_SIZE:--1}"
  export BASELINE_NUM_ITERATIONS="${BASELINE_NUM_ITERATIONS:--1}"
  export BASELINE_EVAL_EVERY="${BASELINE_EVAL_EVERY:-250}"
  export BASELINE_EVAL_TOKENS="${BASELINE_EVAL_TOKENS:-20971520}"
  export BASELINE_CORE_METRIC_EVERY="${BASELINE_CORE_METRIC_EVERY:-999999}"
  export BASELINE_DEFAULT_RATIO="${BASELINE_DEFAULT_RATIO:-$BASELINE_SPEEDRUN_DEFAULT_RATIO}"
  export BASELINE_TARGET_METRIC="${BASELINE_TARGET_METRIC:-val_bpb}"
  export BASELINE_WANDB_RUN_BASE="${BASELINE_WANDB_RUN_BASE:-dummy}"
  export BASELINE_RUN_LABEL_PREFIX="${BASELINE_RUN_LABEL_PREFIX:-baseline_d12}"
  export BASELINE_SEEDS="${BASELINE_SEEDS:-41,42,43}"
  export BASELINE_REQUIRED_SEEDS="${BASELINE_REQUIRED_SEEDS:-$BASELINE_SEEDS}"

  # Threshold defaults
  export BASELINE_MIN_SUCCESS="${BASELINE_MIN_SUCCESS:-3}"
  export BASELINE_THRESHOLD_OFFSET="${BASELINE_THRESHOLD_OFFSET:-0.02}"
  export BASELINE_THRESHOLD_JSON="${BASELINE_THRESHOLD_JSON:-$BASELINE_WORK_DIR/threshold.json}"
  export BASELINE_THRESHOLD_ENV="${BASELINE_THRESHOLD_ENV:-$BASELINE_WORK_DIR/threshold.env}"
  export BASELINE_EXPORT_THRESHOLD_VAR="${BASELINE_EXPORT_THRESHOLD_VAR:-TARGET_THRESHOLD}"
  export BASELINE_EXPORT_PREFIX="${BASELINE_EXPORT_PREFIX:-BASELINE}"
}


baseline_info() {
  echo "BASELINE_WORK_DIR=$BASELINE_WORK_DIR"
  echo "BASELINE_RESULTS_CSV=$BASELINE_RESULTS_CSV"
  echo "BASELINE_SEEDS=$BASELINE_SEEDS"
  echo "BASELINE_NUM_ITERATIONS=$BASELINE_NUM_ITERATIONS"
  echo "BASELINE_TOTAL_BATCH_SIZE=$BASELINE_TOTAL_BATCH_SIZE"
  echo "BASELINE_ASPECT_RATIO=$BASELINE_ASPECT_RATIO"
  echo "BASELINE_CORE_METRIC_EVERY=$BASELINE_CORE_METRIC_EVERY"
  echo "BASELINE_TARGET_METRIC=$BASELINE_TARGET_METRIC"
}
