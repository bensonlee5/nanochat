#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 4: extracting calibration tokens/metrics from logs ..."

"$PYTHON_BIN" -m scripts.lane_b_extract_calibration \
  --iterations "$LANE_B_CALIB_ITERS" \
  --total-batch-size "$LANE_B_TOTAL_BATCH_SIZE" \
  --log-dir "$LANE_B_LOG_DIR" \
  --log-prefix "laneb_calib_" \
  --log-suffix ".log" \
  --output-json "$LANE_B_CALIB_JSON" \
  --output-env "$LANE_B_CALIB_ENV"

echo "Wrote: $LANE_B_CALIB_JSON"
echo "Wrote: $LANE_B_CALIB_ENV"
