#!/bin/bash

# Shared defaults/helpers for Lane B step scripts.

lane_b_init() {
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
  export LANE_B_SPEEDRUN_DEFAULT_RATIO="${LANE_B_SPEEDRUN_DEFAULT_RATIO:-$speedrun_ratio_default}"

  export LANE_B_WORK_DIR="${LANE_B_WORK_DIR:-$NANOCHAT_BASE_DIR/lane_b_run}"
  export LANE_B_LOG_DIR="${LANE_B_LOG_DIR:-$LANE_B_WORK_DIR/logs}"
  mkdir -p "$LANE_B_WORK_DIR" "$LANE_B_LOG_DIR"

  # Model/profile defaults
  export LANE_B_DEPTH="${LANE_B_DEPTH:-12}"
  export LANE_B_ASPECT_RATIO="${LANE_B_ASPECT_RATIO:-64}"
  export LANE_B_HEAD_DIM="${LANE_B_HEAD_DIM:-128}"
  export LANE_B_N_KV_HEAD="${LANE_B_N_KV_HEAD:-}"
  export LANE_B_MAX_SEQ_LEN="${LANE_B_MAX_SEQ_LEN:-2048}"
  export LANE_B_WINDOW_PATTERN="${LANE_B_WINDOW_PATTERN:-L}"

  # Run defaults
  export LANE_B_DEVICE_BATCH_SIZE="${LANE_B_DEVICE_BATCH_SIZE:-4}"
  export LANE_B_TOTAL_BATCH_SIZE="${LANE_B_TOTAL_BATCH_SIZE:--1}"
  export LANE_B_EVAL_EVERY="${LANE_B_EVAL_EVERY:-100}"
  export LANE_B_EVAL_TOKENS="${LANE_B_EVAL_TOKENS:-524288}"
  export LANE_B_NUM_ITERATIONS="${LANE_B_NUM_ITERATIONS:-1500}"
  export LANE_B_TARGET_METRIC="${LANE_B_TARGET_METRIC:-val_bpb}"
  export LANE_B_TARGET_THRESHOLD="${LANE_B_TARGET_THRESHOLD:-0.85}"
  export LANE_B_DEFAULT_RATIO="${LANE_B_DEFAULT_RATIO:-$LANE_B_SPEEDRUN_DEFAULT_RATIO}"
  export LANE_B_BASELINE_NUM_ITERATIONS="${LANE_B_BASELINE_NUM_ITERATIONS:-1500}"
  export LANE_B_BASELINE_SEEDS="${LANE_B_BASELINE_SEEDS:-41,42,43}"
  export LANE_B_BASELINE_MIN_SUCCESS="${LANE_B_BASELINE_MIN_SUCCESS:-3}"
  export LANE_B_THRESHOLD_OFFSET="${LANE_B_THRESHOLD_OFFSET:-0.02}"

  # Lane B metadata defaults
  export LANE_B_DATASET_ID="${LANE_B_DATASET_ID:-fineweb_edu_100b_train}"
  export LANE_B_SCHEMA_PATH="${LANE_B_SCHEMA_PATH:-ideas/lane_b_inference_schema.csv}"
  export LANE_B_RUN_SEEDS="${LANE_B_RUN_SEEDS:-41,42}"
  export LANE_B_WANDB_RUN_BASE="${LANE_B_WANDB_RUN_BASE:-dummy}"

  # Lane B stats defaults
  export LANE_B_STATS_MAX_DOCS="${LANE_B_STATS_MAX_DOCS:-5000}"
  export LANE_B_STATS_MAX_TOKENS="${LANE_B_STATS_MAX_TOKENS:-2097152}"
  export LANE_B_STATS_TOKENIZER_THREADS="${LANE_B_STATS_TOKENIZER_THREADS:-4}"
  export LANE_B_CORR_MIN_SEP="${LANE_B_CORR_MIN_SEP:-1}"
  export LANE_B_CORR_MAX_SEP="${LANE_B_CORR_MAX_SEP:-64}"
  export LANE_B_ENT_MIN_CTX="${LANE_B_ENT_MIN_CTX:-1}"
  export LANE_B_ENT_MAX_CTX="${LANE_B_ENT_MAX_CTX:-64}"
  export LANE_B_ENT_NUM_POINTS="${LANE_B_ENT_NUM_POINTS:-8}"
  export LANE_B_ENT_MIN_USABLE_POINTS="${LANE_B_ENT_MIN_USABLE_POINTS:-4}"
  export LANE_B_ENT_UNIQUENESS_THRESHOLD="${LANE_B_ENT_UNIQUENESS_THRESHOLD:-0.9}"
  export LANE_B_FIT_OPTUNA_SEED="${LANE_B_FIT_OPTUNA_SEED:-42}"
  export LANE_B_FIT_OPTUNA_TRIALS="${LANE_B_FIT_OPTUNA_TRIALS:-200}"
  export LANE_B_CORR_R2_WARN_THRESHOLD="${LANE_B_CORR_R2_WARN_THRESHOLD:-0.90}"
  export LANE_B_ENTROPY_R2_WARN_THRESHOLD="${LANE_B_ENTROPY_R2_WARN_THRESHOLD:-0.90}"
  export LANE_B_ALPHA_FALLBACK_MODE="${LANE_B_ALPHA_FALLBACK_MODE:-baseline_assisted}"
  export LANE_B_BASELINE_ALPHA_GRID_MIN="${LANE_B_BASELINE_ALPHA_GRID_MIN:-0.1}"
  export LANE_B_BASELINE_ALPHA_GRID_MAX="${LANE_B_BASELINE_ALPHA_GRID_MAX:-0.8}"
  export LANE_B_BASELINE_ALPHA_GRID_STEP="${LANE_B_BASELINE_ALPHA_GRID_STEP:-0.01}"
  export LANE_B_BASELINE_ALPHA_MIN_R2="${LANE_B_BASELINE_ALPHA_MIN_R2:-0.97}"
  export LANE_B_L_INF_LOWER_BOUND_FROM_STATS_KEY="${LANE_B_L_INF_LOWER_BOUND_FROM_STATS_KEY:-entropy_h_inf_bits}"
  export LANE_B_TIME_TO_TARGET_EXTRAPOLATION="${LANE_B_TIME_TO_TARGET_EXTRAPOLATION:-linear_recent_eval}"
  export LANE_B_TIME_TO_TARGET_POWER_LAW_MIN_POINTS="${LANE_B_TIME_TO_TARGET_POWER_LAW_MIN_POINTS:-3}"
  export LANE_B_TIME_TO_TARGET_POWER_LAW_MAX_POINTS="${LANE_B_TIME_TO_TARGET_POWER_LAW_MAX_POINTS:-5}"
  export LANE_B_TIME_TO_TARGET_POWER_LAW_FIT_R2_MIN="${LANE_B_TIME_TO_TARGET_POWER_LAW_FIT_R2_MIN:-0.0}"
  export LANE_B_ALLOW_LOW_QUALITY_STATS="${LANE_B_ALLOW_LOW_QUALITY_STATS:-0}"

  if [ "$LANE_B_TARGET_METRIC" != "val_bpb" ]; then
    echo "Lane B currently supports only LANE_B_TARGET_METRIC=val_bpb."
    echo "Got: LANE_B_TARGET_METRIC=$LANE_B_TARGET_METRIC"
    echo "Use baseline/core workflows outside Lane B for non-bpb target metrics."
    exit 1
  fi

  # Artifact paths
  export LANE_B_SCALING_JSON="${LANE_B_SCALING_JSON:-$LANE_B_WORK_DIR/scaling_params.json}"
  export LANE_B_SCALING_TXT="${LANE_B_SCALING_TXT:-$LANE_B_WORK_DIR/n_scaling_params.txt}"
  export LANE_B_STATS_JSON="${LANE_B_STATS_JSON:-$LANE_B_WORK_DIR/stats.json}"
  export LANE_B_CALIB_JSON="${LANE_B_CALIB_JSON:-$LANE_B_WORK_DIR/calibration.json}"
  export LANE_B_CALIB_ENV="${LANE_B_CALIB_ENV:-$LANE_B_WORK_DIR/calibration.env}"
  export LANE_B_INFER_JSON="${LANE_B_INFER_JSON:-$LANE_B_WORK_DIR/infer.json}"
  export LANE_B_MANIFEST_CSV="${LANE_B_MANIFEST_CSV:-$LANE_B_WORK_DIR/candidate_runs.csv}"

  # Baseline threshold artifacts
  export LANE_B_BASELINE_WORK_DIR="${LANE_B_BASELINE_WORK_DIR:-$NANOCHAT_BASE_DIR/lane_b_baseline}"
  export LANE_B_BASELINE_RESULTS_DIR="${LANE_B_BASELINE_RESULTS_DIR:-$LANE_B_BASELINE_WORK_DIR/results}"
  export LANE_B_BASELINE_RESULTS_CSV="${LANE_B_BASELINE_RESULTS_CSV:-$LANE_B_BASELINE_RESULTS_DIR/results.csv}"
  export LANE_B_BASELINE_THRESHOLD_JSON="${LANE_B_BASELINE_THRESHOLD_JSON:-$LANE_B_BASELINE_WORK_DIR/baseline_threshold.json}"
  export LANE_B_BASELINE_THRESHOLD_ENV="${LANE_B_BASELINE_THRESHOLD_ENV:-$LANE_B_BASELINE_WORK_DIR/baseline_threshold.env}"
  export LANE_B_BASELINE_LOG_GLOB="${LANE_B_BASELINE_LOG_GLOB:-$LANE_B_BASELINE_RESULTS_DIR/baseline_d*_s*_train.log}"

  # Source baseline threshold env if it exists (provides LANE_B_TARGET_THRESHOLD)
  if [ -f "$LANE_B_BASELINE_THRESHOLD_ENV" ]; then
    # shellcheck disable=SC1090
    source "$LANE_B_BASELINE_THRESHOLD_ENV"
  fi
}


lane_b_info() {
  echo "LANE_B_WORK_DIR=$LANE_B_WORK_DIR"
  echo "LANE_B_LOG_DIR=$LANE_B_LOG_DIR"
  echo "LANE_B_TARGET_THRESHOLD=$LANE_B_TARGET_THRESHOLD"
  echo "LANE_B_TOTAL_BATCH_SIZE=$LANE_B_TOTAL_BATCH_SIZE"
  echo "LANE_B_N_KV_HEAD=${LANE_B_N_KV_HEAD:-<default_n_head>}"
  echo "LANE_B_BASELINE_RESULTS_CSV=$LANE_B_BASELINE_RESULTS_CSV"
}
