# Nanochat Speedrun Execution Plan (Data-Inferred Horizon + Tactical Fusion)

## 1) Objective

Primary objective:
- Minimize wall-clock time to GPT-2 capability target.

Cycle policy:
- Prioritize parameter settings with theory-backed expected optima.
- Avoid broad sweeps that are likely to overfit a single dataset realization.
- Keep diffs small and maintainable.

Primary references:
- `dev/LOG.md`
- `dev/LEADERBOARD.md`
- `scripts/base_train.py`
- `ideas/lane_b_data_horizon_theory.md`
- `ideas/lane_b_inference_worksheet.md`
- `ideas/lane_b_inference_schema.csv`
- `scripts/lane_b_stats.py`
- `scripts/lane_b_infer_ratio.py`
- `scripts/lane_b_log_run.py`
- `ideas/lane_c_fusion_theory.md`

## 2) Locked Invariants (No Sweep This Cycle)

The following are fixed for this cycle:

- Architecture shape defaults:
- `--aspect-ratio=64`
- `--head-dim=128`

- Batch behavior:
- keep auto `Bopt` scaling as default
- no broad batch sweep lane

- Optimizer family:
- no broad LR/WD optimizer-grid lane

- Lane A policy:
- skip Lane A entirely in this cycle
- no warmdown/final-lr tuning experiments for promotion decisions

## 3) Baseline Protocol (d12 local)

Use one fixed d12 baseline profile for comparisons and calibration:

```bash
python -m scripts.base_train \
  --depth=12 \
  --head-dim=128 \
  --window-pattern=L \
  --max-seq-len=512 \
  --device-batch-size=16 \
  --total-batch-size=16384 \
  --eval-every=100 \
  --eval-tokens=524288 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --num-iterations=1500 \
  --run=dummy
```

Seed policy:
- minimum 2 seeds for baseline-vs-candidate decisions
- identical seed set across compared candidates

Required metrics:
- `train/dt`
- `train/tok_per_sec`
- wall-clock train time
- `val_bpb` (if enabled)
- measured/extrapolated time-to-target

## 4) Active Lanes (Only These)

## Lane B: Data-Inferred Horizon (Primary)

Goal:
- infer `target_param_data_ratio` from dataset characteristics and scaling theory, instead of broad empirical sweeps

Method:
1. Estimate dataset statistics on training corpus:
- decay of pairwise token correlations with token separation
- decay of next-token conditional entropy gap with context length
2. Use the closed-form mapping in `arXiv:2602.07488` to infer a data-limited scaling exponent for this dataset family.
3. Use a minimal calibration fit for constants in the local loss-vs-tokens law.
4. Solve for target tokens needed to hit the chosen target proxy.
5. Convert to:
- `target_param_data_ratio = inferred_target_tokens / num_scaling_params`
6. Run one primary inferred ratio and one conservative neighbor only (no large grid).

Constraints:
- no broad architecture or optimizer sweeps inside Lane B
- treat inferred ratio as primary setting, not sweep center

## Lane C: Tactical RMSNorm/CE A/B (Secondary, Time-Boxed)

Goal:
- quick non-hardware throughput lane with strict net-quality gate

Candidates:
- baseline
- `--fused-rmsnorm`
- `--chunked-ce`
- both

Rules:
- scope to minimal A/B matrix (no open-ended kernel work)
- promote only on net time-to-target improvement with quality parity
- if no robust win across 2 seeds, stop lane

## 5) Execution Order

1. Phase 0: lock baseline and target metric/threshold
2. Phase 1: run Lane B inference workflow and evaluate inferred ratio + one neighbor
3. Phase 2: run Lane C tactical matrix
4. Phase 3: run `runs/miniseries.sh` on best candidate only

## 6) Acceptance Gates

A candidate is promotable only if all pass:

1. Primary:
- lower time-to-target (measured if reached, extrapolated otherwise) across 2 seeds

2. Secondary quality floor:
- no meaningful quality regression (`val_bpb` and loss trajectory)

3. Stability:
- no instability, compile-path breakage, or checkpoint/resume regressions

4. Generalization:
- survives miniseries gate without systematic cross-depth regression

If any gate fails:
- keep optional or reject
- do not promote defaults

## 7) Artifacts and Reporting

Each experiment row should include at least:
- `timestamp`
- `git_commit`
- `experiment_name`
- `seed`
- `depth`
- `target_metric_name`
- `target_metric_threshold`
- `time_to_target_sec_measured`
- `time_to_target_sec_extrapolated`
- `tok_per_sec_median`
- `dt_median_ms`
- `wall_time_sec`
- `min_val_bpb`
- `status`

Lane B inference fields (minimal decision+outcome set):
- `dataset_id`
- `n_scaling_params`
- `inferred_data_scaling_exponent`
- `inferred_target_tokens`
- `inferred_target_param_data_ratio`
- `confirmation_ratio`
- `chosen_ratio`
- `lane_b_decision`
- `status`
- `notes`

Lane B schema:
- use `ideas/lane_b_inference_schema.csv` as the minimal fixed column order for Lane B records

## 8) Explicitly Out of Scope This Cycle

- Lane A schedule-tail tuning (`warmdown_ratio`, `warmdown_shape`, `final_lr_frac`)
- broad ratio sweeps around hand-picked centers
- broad shape sweeps over `aspect-ratio` and `head-dim`
- skip-AdamW cadence revisits
- Based / ThunderKittens architecture migration
- broad custom-kernel and distributed overlap redesigns

## 9) Change Log

For each completed experiment:
- date
- exact change
- measured result vs baseline
- decision (`promote`, `optional`, `reject`)

This document is the canonical plan for the current cycle and intentionally prioritizes data-inferred parameter selection over exploratory sweep-heavy tuning.
