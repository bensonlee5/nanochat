# Lane B Step-by-Step Run Guide

Run from repo root:

```bash
cd /Users/bensonlee/dev/nanochat
```

## 0) One-time setup for this run

Set a work directory and any overrides. Defaults are defined in `runs/lane_b_common.sh`.

```bash
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export LANE_B_WORK_DIR="$NANOCHAT_BASE_DIR/lane_b_$(date +%Y%m%d_%H%M%S)"
export LANE_B_BASELINE_WORK_DIR="$NANOCHAT_BASE_DIR/lane_b_baseline_$(date +%Y%m%d_%H%M%S)"
export LANE_B_BASELINE_RESULTS_DIR="$LANE_B_BASELINE_WORK_DIR/results"

# Optional overrides
export LANE_B_DATASET_ID="fineweb_edu_100b_train"
export LANE_B_RUN_SEEDS="41,42"
export LANE_B_CALIB_ITERS="300,600,900"
export LANE_B_BASELINE_SEEDS="41,42,43"
export LANE_B_BASELINE_NUM_ITERATIONS=1500
export LANE_B_THRESHOLD_OFFSET=0.02
export LANE_B_WANDB_RUN_BASE="dummy"
```

## 1) Run true baseline seeds

```bash
bash runs/lane_b_step0_run_true_baselines.sh
```

Note:
- this is a Lane B wrapper around `runs/baseline_step1_run_true_baselines.sh`.

Outputs:
- `$LANE_B_BASELINE_RESULTS_DIR/results.csv`
- per-seed logs in `$LANE_B_BASELINE_RESULTS_DIR`

## 2) Compute threshold from true baselines

```bash
bash runs/lane_b_step0_compute_threshold.sh
source "$LANE_B_BASELINE_THRESHOLD_ENV"
```

Note:
- this is a Lane B wrapper around `runs/baseline_step2_compute_threshold.sh`.

Outputs:
- `$LANE_B_BASELINE_THRESHOLD_JSON`
- `$LANE_B_BASELINE_THRESHOLD_ENV`

Rule:
- `LANE_B_TARGET_THRESHOLD = median(min_val_bpb over required baseline seeds) + LANE_B_THRESHOLD_OFFSET`

## 3) Compute `n_scaling_params`

```bash
bash runs/lane_b_step1_get_scaling_params.sh
```

Outputs:
- `$LANE_B_WORK_DIR/scaling_params.json`
- `$LANE_B_WORK_DIR/n_scaling_params.txt`

## 4) Measure dataset statistics

```bash
bash runs/lane_b_step2_measure_stats.sh
```

Output:
- `$LANE_B_WORK_DIR/stats.json`

## 5) Run calibration jobs

```bash
bash runs/lane_b_step3_run_calibration.sh
```

Outputs:
- calibration logs in `$LANE_B_WORK_DIR/logs/laneb_calib_<iters>.log`

## 6) Extract calibration points from logs

```bash
bash runs/lane_b_step4_extract_calibration.sh
```

Outputs:
- `$LANE_B_WORK_DIR/calibration.json`
- `$LANE_B_WORK_DIR/calibration.env`

## 7) Infer Lane B ratio

```bash
bash runs/lane_b_step5_infer_ratio.sh
```

Output:
- `$LANE_B_WORK_DIR/infer.json`

## 8) Run inferred + confirmation candidates

```bash
bash runs/lane_b_step6_run_candidates.sh
```

Outputs:
- candidate logs in `$LANE_B_WORK_DIR/logs/laneb_<mode>_s<seed>.log`
- manifest: `$LANE_B_WORK_DIR/candidate_runs.csv`

If Step 5 produced `solve_status != ok`, use fallback baseline instead:

```bash
bash runs/lane_b_step6_run_fallback_baseline.sh
```

This creates a fallback manifest with `lane_b_decision=fallback` at ratio `10.5`.

## 9) Append candidate rows to Lane B schema CSV

```bash
bash runs/lane_b_step7_log_candidates.sh
```

Output:
- appended rows in `ideas/lane_b_inference_schema.csv`

Note:
- by default this step skips `run_id`s that already exist in the schema (`LANE_B_SKIP_EXISTING=1`).

## 10) Summarize inferred vs confirmation

```bash
bash runs/lane_b_step8_summary.sh
```

This prints per-seed winner and a recommended `lane_b_decision`.

## Run all steps at once

```bash
bash runs/lane_b_run_all.sh
```

Behavior note:
- run `lane_b_step0_run_true_baselines.sh` and `lane_b_step0_compute_threshold.sh` first.
- for a lane-agnostic baseline workflow, use `runs/baseline_steps.md`.
- `runs/lane_b_run_all.sh` loads `LANE_B_BASELINE_THRESHOLD_ENV` automatically when present.
- `runs/lane_b_run_all.sh` does not execute step 0 baselining for you.
- `runs/lane_b_run_all.sh` stops after Step 5 when inference is unsolved and prints fallback instructions.

## Useful knobs

Set before running steps:

- `LANE_B_DEPTH`, `LANE_B_ASPECT_RATIO`, `LANE_B_HEAD_DIM`, `LANE_B_MAX_SEQ_LEN`
- `LANE_B_DEVICE_BATCH_SIZE`, `LANE_B_TOTAL_BATCH_SIZE`, `LANE_B_NUM_ITERATIONS`
- `LANE_B_TARGET_METRIC`, `LANE_B_TARGET_THRESHOLD`, `LANE_B_THRESHOLD_OFFSET`
- `LANE_B_STATS_MAX_DOCS`, `LANE_B_STATS_MAX_TOKENS`
- `LANE_B_FIT_OPTUNA_SEED`, `LANE_B_FIT_OPTUNA_TRIALS`
- `LANE_B_RUN_SEEDS`, `LANE_B_CALIB_ITERS`
- `LANE_B_BASELINE_SEEDS`, `LANE_B_BASELINE_NUM_ITERATIONS`
- `LANE_B_BASELINE_WORK_DIR`, `LANE_B_BASELINE_RESULTS_DIR`
- `LANE_B_WANDB_RUN_BASE`
