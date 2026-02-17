#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 0b (Lane B wrapper): computing threshold from generic baseline runs ..."

export BASELINE_RESULTS_CSV="$LANE_B_BASELINE_RESULTS_CSV"
export BASELINE_REQUIRED_SEEDS="$LANE_B_BASELINE_SEEDS"
export BASELINE_MIN_SUCCESS="$LANE_B_BASELINE_MIN_SUCCESS"
export BASELINE_THRESHOLD_OFFSET="$LANE_B_THRESHOLD_OFFSET"
export BASELINE_TARGET_METRIC="$LANE_B_TARGET_METRIC"
export BASELINE_THRESHOLD_JSON="$LANE_B_BASELINE_THRESHOLD_JSON"
export BASELINE_THRESHOLD_ENV="$LANE_B_BASELINE_THRESHOLD_ENV"
export BASELINE_EXPORT_THRESHOLD_VAR="LANE_B_TARGET_THRESHOLD"
export BASELINE_EXPORT_PREFIX="LANE_B_BASELINE"

bash "$SCRIPT_DIR/baseline_step2_compute_threshold.sh"
