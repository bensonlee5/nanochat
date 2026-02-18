#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 2: measuring dataset statistics ..."

"$PYTHON_BIN" -m scripts.lane_b_stats \
  --split train \
  --dataset-id "$LANE_B_DATASET_ID" \
  --max-docs "$LANE_B_STATS_MAX_DOCS" \
  --max-tokens "$LANE_B_STATS_MAX_TOKENS" \
  --tokenizer-threads "$LANE_B_STATS_TOKENIZER_THREADS" \
  --correlation-min-sep "$LANE_B_CORR_MIN_SEP" \
  --correlation-max-sep "$LANE_B_CORR_MAX_SEP" \
  --entropy-min-ctx "$LANE_B_ENT_MIN_CTX" \
  --entropy-max-ctx "$LANE_B_ENT_MAX_CTX" \
  --entropy-num-points "$LANE_B_ENT_NUM_POINTS" \
  --fit-optuna-seed "$LANE_B_FIT_OPTUNA_SEED" \
  --fit-optuna-trials "$LANE_B_FIT_OPTUNA_TRIALS" \
  --corr-r2-warn-threshold "$LANE_B_CORR_R2_WARN_THRESHOLD" \
  --entropy-r2-warn-threshold "$LANE_B_ENTROPY_R2_WARN_THRESHOLD" \
  --output-json "$LANE_B_STATS_JSON"

echo "Wrote: $LANE_B_STATS_JSON"
