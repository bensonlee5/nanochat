#!/bin/bash

set -u

# Theory-first speedrun baseline harness
# Usage:
#   bash runs/speedrun_theory.sh
# Optional env overrides:
#   SERIES_NAME, RUN_LABEL, WANDB_RUN, NANOCHAT_BASE_DIR
#   DEPTH, HEAD_DIM, MAX_SEQ_LEN, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE
#   NUM_ITERATIONS, EVAL_EVERY, EVAL_TOKENS, TARGET_PARAM_DATA_RATIO
#   WARMUP_RATIO, WARMDOWN_RATIO, WARMDOWN_SHAPE, FINAL_LR_FRAC
#   TARGET_METRIC, TARGET_THRESHOLD

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ ! -d ".venv" ]; then
    echo "Missing .venv; create it first (e.g. uv venv && uv sync --extra cpu|gpu)."
    exit 1
fi
source .venv/bin/activate

SERIES_NAME="${SERIES_NAME:-theory_$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
RUN_LABEL="${RUN_LABEL:-baseline_d12}"
WANDB_RUN="${WANDB_RUN:-dummy}"

DEPTH="${DEPTH:-12}"
HEAD_DIM="${HEAD_DIM:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16384}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-10.5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.5}"
WARMDOWN_SHAPE="${WARMDOWN_SHAPE:-linear}"
FINAL_LR_FRAC="${FINAL_LR_FRAC:-0.0}"
TARGET_METRIC="${TARGET_METRIC:-val_bpb}"
TARGET_THRESHOLD="${TARGET_THRESHOLD:--1}"

RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_BASE_DIR/${SERIES_NAME}_speedrun_theory_results}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"
LOG_FILE="$RESULTS_DIR/${RUN_LABEL}_train.log"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "timestamp,git_commit,experiment_name,seed,depth,max_seq_len,device_batch_size,total_batch_size,target_param_data_ratio,warmdown_ratio,warmdown_shape,target_metric_name,target_metric_threshold,time_to_target_sec_measured,time_to_target_sec_extrapolated,extrapolation_method,tok_per_sec_median,dt_median_ms,wall_time_sec,final_train_loss,min_val_bpb,peak_mem_mib,status" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting ${RUN_LABEL} in $RESULTS_DIR"
START_TIME=$(date +%s)

CMD=(
    python -m scripts.base_train
    --depth="$DEPTH"
    --head-dim="$HEAD_DIM"
    --window-pattern=L
    --max-seq-len="$MAX_SEQ_LEN"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --total-batch-size="$TOTAL_BATCH_SIZE"
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
    --warmup-ratio="$WARMUP_RATIO"
    --warmdown-ratio="$WARMDOWN_RATIO"
    --warmdown-shape="$WARMDOWN_SHAPE"
    --final-lr-frac="$FINAL_LR_FRAC"
    --eval-every="$EVAL_EVERY"
    --eval-tokens="$EVAL_TOKENS"
    --core-metric-every=-1
    --sample-every=-1
    --save-every=-1
    --num-iterations="$NUM_ITERATIONS"
    --time-to-target-metric="$TARGET_METRIC"
    --time-to-target-threshold="$TARGET_THRESHOLD"
    --run="$WANDB_RUN"
)

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

END_TIME=$(date +%s)
WALL_TIME_SEC=$((END_TIME - START_TIME))

if [ $TRAIN_EXIT -eq 0 ]; then
    STATUS="ok"
else
    STATUS="failed"
fi

PARSED_METRICS=$(python - "$LOG_FILE" <<'PY'
import re
import statistics
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8", errors="replace").read()

def last_float(pattern):
    vals = re.findall(pattern, text)
    return vals[-1] if vals else ""

tok_vals = [int(x.replace(",", "")) for x in re.findall(r"tok/sec:\s*([\d,]+)", text)]
dt_vals = [float(x) for x in re.findall(r"dt:\s*([0-9.]+)ms", text)]
loss_vals = [float(x) for x in re.findall(r"loss:\s*([0-9.]+)", text)]

tok_median = f"{statistics.median(tok_vals):.2f}" if tok_vals else ""
dt_median = f"{statistics.median(dt_vals):.2f}" if dt_vals else ""
final_loss = f"{loss_vals[-1]:.6f}" if loss_vals else ""
min_val_bpb = last_float(r"Minimum validation bpb:\s*([0-9.]+)")
peak_mem = last_float(r"Peak memory usage:\s*([0-9.]+)MiB")

tt_measured_min = last_float(r"Time-to-target \([^)]+\):\s*([0-9.]+)m")
tt_measured_sec = f"{float(tt_measured_min) * 60:.2f}" if tt_measured_min else ""

extrap_matches = re.findall(r"Extrapolated time-to-target \([^)]+,\s*([^)]+)\):\s*([0-9.]+)m", text)
if extrap_matches:
    method, mins = extrap_matches[-1]
    tt_extrap_sec = f"{float(mins) * 60:.2f}"
    extrap_method = method.strip()
else:
    tt_extrap_sec = ""
    extrap_method = ""

print(f"TOK_MEDIAN={tok_median}")
print(f"DT_MEDIAN={dt_median}")
print(f"FINAL_LOSS={final_loss}")
print(f"MIN_VAL_BPB={min_val_bpb}")
print(f"PEAK_MEM_MIB={peak_mem}")
print(f"TT_MEASURED_SEC={tt_measured_sec}")
print(f"TT_EXTRAP_SEC={tt_extrap_sec}")
print(f"EXTRAP_METHOD={extrap_method}")
PY
)
eval "$PARSED_METRICS"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
SEED=42

echo "${TIMESTAMP},${GIT_COMMIT},${RUN_LABEL},${SEED},${DEPTH},${MAX_SEQ_LEN},${DEVICE_BATCH_SIZE},${TOTAL_BATCH_SIZE},${TARGET_PARAM_DATA_RATIO},${WARMDOWN_RATIO},${WARMDOWN_SHAPE},${TARGET_METRIC},${TARGET_THRESHOLD},${TT_MEASURED_SEC},${TT_EXTRAP_SEC},${EXTRAP_METHOD},${TOK_MEDIAN},${DT_MEDIAN},${WALL_TIME_SEC},${FINAL_LOSS},${MIN_VAL_BPB},${PEAK_MEM_MIB},${STATUS}" >> "$RESULTS_FILE"

log "Run status: ${STATUS}"
log "Log: $LOG_FILE"
log "Results CSV: $RESULTS_FILE"

if [ $TRAIN_EXIT -ne 0 ]; then
    exit $TRAIN_EXIT
fi
