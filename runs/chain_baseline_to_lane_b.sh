#!/bin/bash
set -euo pipefail

# Chain Lane B after baseline completion.
# 1) Poll baseline results until all 3 seeds succeed
# 2) Bridge threshold from baseline format to lane B format
# 3) Launch lane B with baseline-matching config overrides

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# --- Baseline artifacts ---
BASELINE_WORK_DIR="$NANOCHAT_BASE_DIR/baseline_run"
BASELINE_RESULTS_CSV="$BASELINE_WORK_DIR/results/results.csv"
BASELINE_THRESHOLD_ENV="$BASELINE_WORK_DIR/threshold.env"
BASELINE_THRESHOLD_JSON="$BASELINE_WORK_DIR/threshold.json"

# --- Lane B artifacts ---
LANE_B_BASELINE_DIR="$NANOCHAT_BASE_DIR/lane_b_baseline"
LANE_B_BASELINE_THRESHOLD_ENV="$LANE_B_BASELINE_DIR/baseline_threshold.env"

REQUIRED_SEEDS=3
POLL_INTERVAL=300  # 5 minutes

echo "=== Chain: Baseline -> Lane B ==="
echo "Started at: $(date)"
echo "Polling baseline results every ${POLL_INTERVAL}s"
echo "Baseline results CSV: $BASELINE_RESULTS_CSV"
echo "Baseline threshold env: $BASELINE_THRESHOLD_ENV"
echo ""

# ============================================================
# Step 1: Wait for baseline completion
# ============================================================
echo "--- Step 1: Waiting for baseline to complete ---"

while true; do
    if [ -f "$BASELINE_THRESHOLD_ENV" ] && [ -f "$BASELINE_THRESHOLD_JSON" ]; then
        # threshold files exist → step2 completed successfully
        if [ -f "$BASELINE_RESULTS_CSV" ]; then
            ok_count=$(grep -c ',ok,' "$BASELINE_RESULTS_CSV" 2>/dev/null) || ok_count=0
            echo "[$(date '+%H:%M:%S')] Baseline done: threshold files exist, $ok_count seeds with status=ok"
            if [ "$ok_count" -ge "$REQUIRED_SEEDS" ]; then
                echo "All $REQUIRED_SEEDS seeds succeeded."
                break
            else
                echo "WARNING: Only $ok_count/$REQUIRED_SEEDS seeds succeeded, but threshold was computed."
                echo "Proceeding anyway (threshold script has its own min-success check)."
                break
            fi
        else
            echo "[$(date '+%H:%M:%S')] Threshold files exist but no results CSV? Proceeding."
            break
        fi
    fi

    # Check partial progress
    if [ -f "$BASELINE_RESULTS_CSV" ]; then
        ok_count=$(grep -c ',ok,' "$BASELINE_RESULTS_CSV" 2>/dev/null) || ok_count=0
        total_rows=$(wc -l < "$BASELINE_RESULTS_CSV" 2>/dev/null) || total_rows=0
        echo "[$(date '+%H:%M:%S')] Waiting... results CSV has $total_rows rows, $ok_count with status=ok"
    else
        echo "[$(date '+%H:%M:%S')] Waiting... no results CSV yet"
    fi

    sleep "$POLL_INTERVAL"
done

echo ""
echo "Baseline threshold.env contents:"
cat "$BASELINE_THRESHOLD_ENV"
echo ""

# ============================================================
# Step 2: Bridge threshold to lane B format
# ============================================================
echo "--- Step 2: Bridging threshold to lane B format ---"

# Source the baseline threshold to get the values
# shellcheck disable=SC1090
source "$BASELINE_THRESHOLD_ENV"

# Extract values (baseline exports: TARGET_THRESHOLD, BASELINE_MEDIAN_MIN_VAL_BPB, BASELINE_SUCCESSFUL_RUNS)
THRESHOLD_VALUE="${TARGET_THRESHOLD:?threshold.env did not set TARGET_THRESHOLD}"
MEDIAN_VALUE="${BASELINE_MEDIAN_MIN_VAL_BPB:?threshold.env did not set BASELINE_MEDIAN_MIN_VAL_BPB}"
SUCCESSFUL_RUNS="${BASELINE_SUCCESSFUL_RUNS:?threshold.env did not set BASELINE_SUCCESSFUL_RUNS}"

mkdir -p "$LANE_B_BASELINE_DIR"

cat > "$LANE_B_BASELINE_THRESHOLD_ENV" <<EOF
export LANE_B_TARGET_THRESHOLD=$THRESHOLD_VALUE
export LANE_B_BASELINE_MEDIAN_MIN_VAL_BPB=$MEDIAN_VALUE
export LANE_B_BASELINE_SUCCESSFUL_RUNS=$SUCCESSFUL_RUNS
EOF

echo "Wrote: $LANE_B_BASELINE_THRESHOLD_ENV"
cat "$LANE_B_BASELINE_THRESHOLD_ENV"
echo ""

# ============================================================
# Step 3: Run lane B with baseline-matching config
# ============================================================
echo "--- Step 3: Launching lane B with baseline-matching config ---"
echo "Config overrides:"
echo "  LANE_B_MAX_SEQ_LEN=2048"
echo "  LANE_B_TOTAL_BATCH_SIZE=-1 (auto)"
echo "  LANE_B_DEVICE_BATCH_SIZE=4"
echo "  LANE_B_EVAL_EVERY=250"
echo "  LANE_B_EVAL_TOKENS=20971520"
echo "  LANE_B_BASELINE_NUM_ITERATIONS=-1 (auto)"
echo ""

LANE_B_MAX_SEQ_LEN=2048 \
LANE_B_TOTAL_BATCH_SIZE=-1 \
LANE_B_DEVICE_BATCH_SIZE=4 \
LANE_B_EVAL_EVERY=250 \
LANE_B_EVAL_TOKENS=20971520 \
LANE_B_BASELINE_NUM_ITERATIONS=-1 \
bash "$SCRIPT_DIR/lane_b_run_all.sh"

echo ""
echo "=== Chain complete at $(date) ==="
