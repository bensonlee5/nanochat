#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

if [ -f "$LANE_B_BASELINE_THRESHOLD_ENV" ]; then
  # shellcheck disable=SC1090
  source "$LANE_B_BASELINE_THRESHOLD_ENV"
  export LANE_B_TARGET_THRESHOLD
  echo "Loaded baseline-derived threshold from $LANE_B_BASELINE_THRESHOLD_ENV"
  echo "LANE_B_TARGET_THRESHOLD=$LANE_B_TARGET_THRESHOLD"
else
  echo "No baseline threshold env found at $LANE_B_BASELINE_THRESHOLD_ENV"
  echo "Using current LANE_B_TARGET_THRESHOLD=$LANE_B_TARGET_THRESHOLD"
  echo "NOTE: lane_b_run_all.sh does not run baseline step 0 automatically."
  echo "Run baselining first for a grounded threshold:"
  echo "  bash runs/lane_b_step0_run_true_baselines.sh"
  echo "  bash runs/lane_b_step0_compute_threshold.sh"
  echo "  source \"$LANE_B_BASELINE_THRESHOLD_ENV\""
fi

"$SCRIPT_DIR/lane_b_step1_get_scaling_params.sh"
"$SCRIPT_DIR/lane_b_step2_measure_stats.sh"
"$SCRIPT_DIR/lane_b_step3_run_calibration.sh"
"$SCRIPT_DIR/lane_b_step4_extract_calibration.sh"
"$SCRIPT_DIR/lane_b_step5_infer_ratio.sh"

if ! .venv/bin/python - <<'PY'
import json
import os
import sys

base_dir = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
work_dir = os.environ.get("LANE_B_WORK_DIR", os.path.join(base_dir, "lane_b_run"))
infer_path = os.environ.get("LANE_B_INFER_JSON", os.path.join(work_dir, "infer.json"))
if not os.path.exists(infer_path):
    raise SystemExit(f"Missing inference JSON: {infer_path}")
d = json.load(open(infer_path, "r", encoding="utf-8"))
if d.get("solve_status") != "ok":
    raise SystemExit(f"unsolved:{d.get('solve_status')}")
if d.get("alpha_data_plausible") != "yes":
    raise SystemExit(f"alpha_implausible:{d.get('alpha_data_plausible')}")
PY
then
  echo "Lane B inference is not promotable; stopping full pipeline after step 5."
  echo "Recovery options:"
  echo "  1) Run fallback baseline path:"
  echo "     bash runs/lane_b_step6_run_fallback_baseline.sh"
  echo "     bash runs/lane_b_step7_log_candidates.sh"
  echo "     bash runs/lane_b_step8_summary.sh"
  echo "  2) Adjust stats/calibration/threshold and rerun (step 2 or step 5)."
  exit 2
fi

"$SCRIPT_DIR/lane_b_step6_run_candidates.sh"
"$SCRIPT_DIR/lane_b_step7_log_candidates.sh"
"$SCRIPT_DIR/lane_b_step8_summary.sh"
