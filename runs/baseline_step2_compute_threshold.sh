#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/baseline_common.sh"
baseline_init

echo "Step 2: computing threshold from baseline runs ..."
echo "Using results CSV: $BASELINE_RESULTS_CSV"
echo "Required seeds: $BASELINE_REQUIRED_SEEDS"
echo "Rule: median(min_val_bpb) + $BASELINE_THRESHOLD_OFFSET"

"$PYTHON_BIN" -m scripts.baseline_compute_threshold \
  --results-csv "$BASELINE_RESULTS_CSV" \
  --required-seeds "$BASELINE_REQUIRED_SEEDS" \
  --min-successful-runs "$BASELINE_MIN_SUCCESS" \
  --threshold-offset "$BASELINE_THRESHOLD_OFFSET" \
  --target-metric-name "$BASELINE_TARGET_METRIC" \
  --export-threshold-var "$BASELINE_EXPORT_THRESHOLD_VAR" \
  --export-prefix "$BASELINE_EXPORT_PREFIX" \
  --output-json "$BASELINE_THRESHOLD_JSON" \
  --output-env "$BASELINE_THRESHOLD_ENV"

echo "Wrote: $BASELINE_THRESHOLD_JSON"
echo "Wrote: $BASELINE_THRESHOLD_ENV"
echo "Load threshold:"
echo "  source \"$BASELINE_THRESHOLD_ENV\""
