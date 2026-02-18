#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 5: inferring target ratio from stats + calibration ..."

if [ ! -f "$LANE_B_SCALING_TXT" ]; then
  echo "Missing $LANE_B_SCALING_TXT. Run step 1 first."
  exit 1
fi
if [ ! -f "$LANE_B_STATS_JSON" ]; then
  echo "Missing $LANE_B_STATS_JSON. Run step 2 first."
  exit 1
fi
if [ ! -f "$LANE_B_CALIB_ENV" ]; then
  echo "Missing $LANE_B_CALIB_ENV. Run step 4 first."
  exit 1
fi

N_SCALING_PARAMS="$(cat "$LANE_B_SCALING_TXT")"
# shellcheck disable=SC1090
source "$LANE_B_CALIB_ENV"

LANE_B_PAPER_MAPPING_ID="${LANE_B_PAPER_MAPPING_ID:-paper_v1}"
LANE_B_PAPER_MAPPING_NOTES="${LANE_B_PAPER_MAPPING_NOTES:-alpha_data = gamma/(2*beta) from arXiv:2602.07488}"
LANE_B_ALPHA_MIN="${LANE_B_ALPHA_MIN:-0.2}"
LANE_B_ALPHA_MAX="${LANE_B_ALPHA_MAX:-0.8}"
LANE_B_CORR_WEIGHT="${LANE_B_CORR_WEIGHT:-1.0}"
LANE_B_ENTROPY_WEIGHT="${LANE_B_ENTROPY_WEIGHT:-1.0}"
LANE_B_UNREACHABLE_MULT="${LANE_B_UNREACHABLE_MULT:-10.0}"

"$PYTHON_BIN" -m scripts.lane_b_infer_ratio \
  --stats-json "$LANE_B_STATS_JSON" \
  --paper-mapping-id "$LANE_B_PAPER_MAPPING_ID" \
  --paper-mapping-notes "$LANE_B_PAPER_MAPPING_NOTES" \
  --corr-weight "$LANE_B_CORR_WEIGHT" \
  --entropy-weight "$LANE_B_ENTROPY_WEIGHT" \
  --alpha-min "$LANE_B_ALPHA_MIN" \
  --alpha-max "$LANE_B_ALPHA_MAX" \
  --calib-tokens "$CALIB_TOKENS" \
  --calib-metrics "$CALIB_METRICS" \
  --target-metric-threshold "$LANE_B_TARGET_THRESHOLD" \
  --target-metric-direction lower_is_better \
  --n-scaling-params "$N_SCALING_PARAMS" \
  --default-ratio "$LANE_B_DEFAULT_RATIO" \
  --unreachable-multiplier "$LANE_B_UNREACHABLE_MULT" \
  --output-json "$LANE_B_INFER_JSON"

"$PYTHON_BIN" - "$LANE_B_INFER_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
print(f"inferred_data_scaling_exponent={d.get('inferred_data_scaling_exponent')}")
print(f"inferred_target_tokens={d.get('inferred_target_tokens')}")
print(f"inferred_target_param_data_ratio={d.get('inferred_target_param_data_ratio')}")
print(f"confirmation_ratio={d.get('confirmation_ratio')}")
print(f"solve_status={d.get('solve_status')}")
print(f"feasibility_flag={d.get('feasibility_flag')}")
PY

echo "Wrote: $LANE_B_INFER_JSON"
