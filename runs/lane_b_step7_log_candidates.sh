#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 7: appending candidate runs into Lane B schema ..."

if [ ! -f "$LANE_B_MANIFEST_CSV" ]; then
  echo "Missing $LANE_B_MANIFEST_CSV. Run step 6 first."
  exit 1
fi
if [ ! -f "$LANE_B_SCALING_TXT" ]; then
  echo "Missing $LANE_B_SCALING_TXT. Run step 1 first."
  exit 1
fi

N_SCALING_PARAMS="$(cat "$LANE_B_SCALING_TXT")"
ALPHA_DATA=""
INFERRED_TOKENS=""
INFERRED_RATIO=""
CONFIRMATION_RATIO=""
SOLVE_STATUS=""

while IFS='=' read -r key value; do
  case "$key" in
    ALPHA_DATA) ALPHA_DATA="$value" ;;
    INFERRED_TOKENS) INFERRED_TOKENS="$value" ;;
    INFERRED_RATIO) INFERRED_RATIO="$value" ;;
    CONFIRMATION_RATIO) CONFIRMATION_RATIO="$value" ;;
    SOLVE_STATUS) SOLVE_STATUS="$value" ;;
  esac
done < <("$PYTHON_BIN" - "$N_SCALING_PARAMS" "$LANE_B_DEFAULT_RATIO" "$LANE_B_INFER_JSON" <<'PY'
import json
import sys
n_scaling_params = float(sys.argv[1])
default_ratio = float(sys.argv[2])
infer_path = sys.argv[3]

alpha_data = 0.0
inferred_tokens = n_scaling_params * default_ratio
inferred_ratio = default_ratio
confirmation_ratio = default_ratio * 1.05
solve_status = "missing"

try:
    with open(infer_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    solve_status = d.get("solve_status", "missing")
    alpha_data = float(d.get("inferred_data_scaling_exponent", alpha_data))
    if d.get("inferred_target_tokens") is not None:
        inferred_tokens = float(d["inferred_target_tokens"])
    if d.get("inferred_target_param_data_ratio") is not None:
        inferred_ratio = float(d["inferred_target_param_data_ratio"])
    if d.get("confirmation_ratio") is not None:
        confirmation_ratio = float(d["confirmation_ratio"])
except FileNotFoundError:
    pass

print(f"ALPHA_DATA={alpha_data}")
print(f"INFERRED_TOKENS={inferred_tokens}")
print(f"INFERRED_RATIO={inferred_ratio}")
print(f"CONFIRMATION_RATIO={confirmation_ratio}")
print(f"SOLVE_STATUS={solve_status}")
PY
)

if [ -z "$ALPHA_DATA" ] || [ -z "$INFERRED_TOKENS" ] || [ -z "$INFERRED_RATIO" ] || [ -z "$CONFIRMATION_RATIO" ] || [ -z "$SOLVE_STATUS" ]; then
  echo "Failed to parse inference fields from $LANE_B_INFER_JSON"
  exit 1
fi

LANE_B_SKIP_EXISTING="${LANE_B_SKIP_EXISTING:-1}"

while IFS=$'\t' read -r run_id seed mode chosen_ratio status log_path; do
  if [ -z "$run_id" ]; then
    continue
  fi
  if [ "$LANE_B_SKIP_EXISTING" = "1" ] && [ -f "$LANE_B_SCHEMA_PATH" ]; then
    if awk -F, -v id="$run_id" 'NR>1 && $3==id {found=1; exit} END{exit !found}' "$LANE_B_SCHEMA_PATH"; then
      echo "Skipping existing run_id in schema: $run_id"
      continue
    fi
  fi

  log_args=()
  if [ -f "$log_path" ]; then
    log_args=(--log-path "$log_path")
  else
    echo "WARNING: missing log file for $run_id: $log_path"
    if [ "$status" = "ok" ]; then
      status="missing_log"
    fi
  fi

  "$PYTHON_BIN" -m scripts.lane_b_log_run \
    --schema-path "$LANE_B_SCHEMA_PATH" \
    "${log_args[@]}" \
    --run-id "$run_id" \
    --seed "$seed" \
    --dataset-id "$LANE_B_DATASET_ID" \
    --target-metric-name "$LANE_B_TARGET_METRIC" \
    --target-metric-threshold "$LANE_B_TARGET_THRESHOLD" \
    --n-scaling-params "$N_SCALING_PARAMS" \
    --inferred-data-scaling-exponent "$ALPHA_DATA" \
    --inferred-target-tokens "$INFERRED_TOKENS" \
    --inferred-target-param-data-ratio "$INFERRED_RATIO" \
    --confirmation-ratio "$CONFIRMATION_RATIO" \
    --chosen-ratio "$chosen_ratio" \
    --lane-b-decision "$mode" \
    --status "$status" \
    --notes "lane_b_mode=$mode solve_status=$SOLVE_STATUS work_dir=$LANE_B_WORK_DIR"
done < <("$PYTHON_BIN" - "$LANE_B_MANIFEST_CSV" <<'PY'
import csv
import sys

manifest_path = sys.argv[1]
fields = ["run_id", "seed", "mode", "chosen_ratio", "status", "log_path"]
with open(manifest_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vals = [(row.get(k, "") or "").replace("\t", " ").replace("\n", " ") for k in fields]
        print("\t".join(vals))
PY
)

echo "Updated schema: $LANE_B_SCHEMA_PATH"
tail -n 10 "$LANE_B_SCHEMA_PATH"
