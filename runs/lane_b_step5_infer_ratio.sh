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
if [ ! -f "$LANE_B_BASELINE_RESULTS_CSV" ]; then
  echo "Missing $LANE_B_BASELINE_RESULTS_CSV."
  echo "Run baseline step 0 first: bash runs/lane_b_step0_run_true_baselines.sh"
  exit 1
fi

N_SCALING_PARAMS="$(cat "$LANE_B_SCALING_TXT")"

LANE_B_PAPER_MAPPING_ID="${LANE_B_PAPER_MAPPING_ID:-paper_v1}"
LANE_B_PAPER_MAPPING_NOTES="${LANE_B_PAPER_MAPPING_NOTES:-alpha_data = gamma/(2*beta) from arXiv:2602.07488}"
LANE_B_ALPHA_MIN="${LANE_B_ALPHA_MIN:-0.2}"
LANE_B_ALPHA_MAX="${LANE_B_ALPHA_MAX:-0.8}"
LANE_B_CORR_WEIGHT="${LANE_B_CORR_WEIGHT:-1.0}"
LANE_B_ENTROPY_WEIGHT="${LANE_B_ENTROPY_WEIGHT:-1.0}"
LANE_B_UNREACHABLE_MULT="${LANE_B_UNREACHABLE_MULT:-10.0}"
LANE_B_ALLOW_LOW_QUALITY_STATS="${LANE_B_ALLOW_LOW_QUALITY_STATS:-0}"
LANE_B_ALPHA_FALLBACK_MODE="${LANE_B_ALPHA_FALLBACK_MODE:-baseline_assisted}"

if ! "$PYTHON_BIN" - "$LANE_B_STATS_JSON" "$LANE_B_ALLOW_LOW_QUALITY_STATS" "$LANE_B_ALPHA_FALLBACK_MODE" <<'PY'
import json
import sys

stats_path = sys.argv[1]
allow_override = str(sys.argv[2]).strip() == "1"
alpha_fallback_mode = str(sys.argv[3]).strip()
stats = json.load(open(stats_path, "r", encoding="utf-8"))

required = [
    "corr_decay_r2",
    "corr_decay_r2_warn_threshold",
    "corr_decay_r2_low_quality",
    "entropy_decay_r2",
    "entropy_decay_r2_warn_threshold",
    "entropy_decay_r2_low_quality",
]
missing = [k for k in required if k not in stats]
if missing:
    print("ERROR: Lane B stats JSON is missing required fit-quality keys:")
    for key in missing:
        print(f"  - {key}")
    print("Rerun step 2 with the current scripts to regenerate stats JSON.")
    raise SystemExit(2)

corr_low = str(stats.get("corr_decay_r2_low_quality", "no")).lower() == "yes"
ent_low = str(stats.get("entropy_decay_r2_low_quality", "no")).lower() == "yes"
if allow_override or not (corr_low or ent_low):
    if allow_override and (corr_low or ent_low):
        print("WARNING: Proceeding despite low-quality stats fits because LANE_B_ALLOW_LOW_QUALITY_STATS=1")
    raise SystemExit(0)

if alpha_fallback_mode == "baseline_assisted":
    print(
        "WARNING: stats fit-quality gate failed, but proceeding because "
        "LANE_B_ALPHA_FALLBACK_MODE=baseline_assisted"
    )
    raise SystemExit(0)

print("ERROR: Lane B stats fit-quality gate failed.")
print(
    f"  corr_decay_r2={stats['corr_decay_r2']} "
    f"(warn threshold={stats['corr_decay_r2_warn_threshold']}, low_quality={stats['corr_decay_r2_low_quality']})"
)
print(
    f"  entropy_decay_r2={stats['entropy_decay_r2']} "
    f"(warn threshold={stats['entropy_decay_r2_warn_threshold']}, low_quality={stats['entropy_decay_r2_low_quality']})"
)
print("Recovery options:")
print("  1) Increase LANE_B_STATS_MAX_TOKENS and rerun step 2")
print("  2) Adjust R^2 warn thresholds")
print("  3) Override gate with LANE_B_ALLOW_LOW_QUALITY_STATS=1")
raise SystemExit(2)
PY
then
  echo "Stopping step 5 due to stats fit-quality gate."
  exit 2
fi

L_INF_ARGS=()
if [ -n "${LANE_B_L_INF_LOWER_BOUND_FROM_STATS_KEY:-}" ]; then
  L_INF_ARGS=(
    --l-inf-lower-bound-from-stats-key "$LANE_B_L_INF_LOWER_BOUND_FROM_STATS_KEY"
  )
fi

"$PYTHON_BIN" -m scripts.lane_b_infer_ratio \
  --stats-json "$LANE_B_STATS_JSON" \
  --paper-mapping-id "$LANE_B_PAPER_MAPPING_ID" \
  --paper-mapping-notes "$LANE_B_PAPER_MAPPING_NOTES" \
  --corr-weight "$LANE_B_CORR_WEIGHT" \
  --entropy-weight "$LANE_B_ENTROPY_WEIGHT" \
  --alpha-min "$LANE_B_ALPHA_MIN" \
  --alpha-max "$LANE_B_ALPHA_MAX" \
  --calib-from-baseline-results-csv "$LANE_B_BASELINE_RESULTS_CSV" \
  --calib-from-baseline-results-dir "$LANE_B_BASELINE_RESULTS_DIR" \
  --calib-from-baseline-required-seeds "$LANE_B_BASELINE_SEEDS" \
  --target-metric-threshold "$LANE_B_TARGET_THRESHOLD" \
  --target-metric-direction lower_is_better \
  "${L_INF_ARGS[@]}" \
  --n-scaling-params "$N_SCALING_PARAMS" \
  --default-ratio "$LANE_B_DEFAULT_RATIO" \
  --unreachable-multiplier "$LANE_B_UNREACHABLE_MULT" \
  --alpha-fallback-mode "$LANE_B_ALPHA_FALLBACK_MODE" \
  --baseline-results-csv "$LANE_B_BASELINE_RESULTS_CSV" \
  --baseline-results-dir "$LANE_B_BASELINE_RESULTS_DIR" \
  --baseline-required-seeds "$LANE_B_BASELINE_SEEDS" \
  --baseline-alpha-grid-min "$LANE_B_BASELINE_ALPHA_GRID_MIN" \
  --baseline-alpha-grid-max "$LANE_B_BASELINE_ALPHA_GRID_MAX" \
  --baseline-alpha-grid-step "$LANE_B_BASELINE_ALPHA_GRID_STEP" \
  --baseline-alpha-min-r2 "$LANE_B_BASELINE_ALPHA_MIN_R2" \
  --output-json "$LANE_B_INFER_JSON"

"$PYTHON_BIN" - "$LANE_B_INFER_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
print(f"inferred_data_scaling_exponent={d.get('inferred_data_scaling_exponent')}")
print(f"alpha_source={d.get('alpha_source')}")
print(f"inferred_target_tokens={d.get('inferred_target_tokens')}")
print(f"inferred_target_param_data_ratio={d.get('inferred_target_param_data_ratio')}")
print(f"confirmation_ratio={d.get('confirmation_ratio')}")
print(f"solve_status={d.get('solve_status')}")
print(f"feasibility_flag={d.get('feasibility_flag')}")
PY

echo "Wrote: $LANE_B_INFER_JSON"
