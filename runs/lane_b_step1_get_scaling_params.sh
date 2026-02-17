#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

lane_b_info
echo "Step 1: computing n_scaling_params ..."

N_SCALING_PARAMS="$("$PYTHON_BIN" -m scripts.lane_b_get_scaling_params \
  --depth "$LANE_B_DEPTH" \
  --aspect-ratio "$LANE_B_ASPECT_RATIO" \
  --head-dim "$LANE_B_HEAD_DIM" \
  --max-seq-len "$LANE_B_MAX_SEQ_LEN" \
  --window-pattern "$LANE_B_WINDOW_PATTERN" \
  --output-json "$LANE_B_SCALING_JSON")"

echo "$N_SCALING_PARAMS" > "$LANE_B_SCALING_TXT"
echo "n_scaling_params=$N_SCALING_PARAMS"
echo "Wrote: $LANE_B_SCALING_JSON"
echo "Wrote: $LANE_B_SCALING_TXT"
