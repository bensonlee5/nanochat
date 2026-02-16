# Nanochat Speedrun Execution Plan (Theory/Data/Shape First, Based Deferred)

## 1) Scope

This runbook defines how to improve nanochat speedruns while preserving the repo's core constraint: accessibility and simplicity first, performance second.

Primary objectives:
- Minimize wall-clock time to reach GPT2-level quality.
- Prioritize theory-guided schedule and scaling transfer first.
- Prioritize data/horizon retuning second.
- Prioritize model shape reallocation at fixed budget third.
- Keep code changes minimal and readable.
- Use local (M1 Max) iteration for most iteration loops.
- Require cross-depth robustness before promoting defaults.

Explicitly out of scope for this cycle:
- Based architecture integration.
- ThunderKittens architecture rewrites.
- Broad custom-kernel architecture changes.

Primary references in-repo:
- `dev/LOG.md`
- `runs/runcpu.sh`
- `runs/miniseries.sh`
- `scripts/base_train.py`
- `nanochat/gpt.py`

## 2) Constraints and Design Principles

Hard constraints:
- Local iteration baseline is `d12`.
- Robustness gate is `runs/miniseries.sh`.
- Keep sweeps narrow, hypothesis-driven, and theory-constrained.
- Every cycle declares one explicit GPT2-target proxy metric and threshold before running experiments.

Design principles:
1. Small diffs: prefer localized edits to `scripts/base_train.py` and narrow run templates.
2. Default-safe behavior: new features behind flags until robust.
3. Reproducibility: fixed seeds and fixed run shapes for fair A/B comparisons.
4. Cross-depth validity: no changes accepted on single-depth wins alone.
5. Cost discipline: develop on M1 first, run expensive GPU validation only at milestone gates.
6. Invariant-first tuning: preserve coupled transfer laws (batch/LR/WD/horizon) before adding complexity.

## 3) Baseline Protocol (Local d12 on M1 Max)

### 3.1 Baseline command profile

Use a d12 local profile that is stable on M1 and avoids known SDPA sliding-window pitfalls by forcing full attention:

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

Notes:
- This is a benchmarking profile, not a target-quality profile.
- Keep shape fixed across candidates unless the phase explicitly tests shape transfer.
- Keep `head-dim=128` for transferability to speedrun/miniseries defaults.

### 3.2 Seed policy

- Minimum 2 seeds for baseline-vs-candidate decisions.
- Use identical seed sets for baseline and candidate.

### 3.3 Metrics to collect

Required:
- `train/dt`
- `train/tok_per_sec`
- wall-clock training time
- loss-vs-step trajectory
- peak memory
- compile overhead (first-step anomalies)
- measured or extrapolated time-to-target

Optional:
- `train/mfu` (if meaningful on runtime)

### 3.4 Result artifact format

Store each experiment row in CSV with at least:
- `timestamp`
- `git_commit`
- `experiment_name`
- `seed`
- `depth`
- `max_seq_len`
- `device_batch_size`
- `total_batch_size`
- `target_param_data_ratio`
- `warmdown_ratio`
- `warmdown_shape`
- `target_metric_name`
- `target_metric_threshold`
- `time_to_target_sec_measured`
- `time_to_target_sec_extrapolated`
- `extrapolation_method`
- `tok_per_sec_median`
- `dt_median_ms`
- `wall_time_sec`
- `final_train_loss`
- `min_val_bpb`
- `peak_mem_mib`
- `status`

## 4) Prior-Effort Triage

### 4.1 Adopt now (primary and secondary lanes)

1. Warmdown micro-sweep:
- narrow sweep on ratio and shape.
- file touchpoint: `scripts/base_train.py`.

2. Batch-centered transfer around predicted `Bopt`:
- compact neighborhood only (e.g. `0.7x, 1.0x, 1.4x`).
- file touchpoint: `scripts/base_train.py`.

3. Joint LR+WD transfer:
- couple updates when changing batch/horizon.
- file touchpoint: `scripts/base_train.py`.

4. Data/horizon retuning:
- narrow adjustments to `target_param_data_ratio`, `final_lr_frac`, warmdown settings.
- file touchpoint: `scripts/base_train.py`.

5. Shape reallocation under fixed budget:
- test depth/width/head-dim tradeoffs at matched wall-time or matched FLOPs budget.
- touchpoints: run templates + `scripts/base_train.py` flags already present.

### 4.2 Tactical exception (time-boxed)

1. RMSNorm fusion quick check:
- treat as 1-day tactical A/B check, not main lane.
- run baseline, `--fused-rmsnorm`, `--chunked-ce`, and both.
- promote only if gain is robust across 2 seeds and quality is preserved.

### 4.3 Conditional support lane (only after Phase 1-3 plateau)

1. Deferred loss logging:
- reduce CPU/GPU sync pressure from immediate `.item()` in hot path.

2. Inductor cache reuse:
- persist compile caches across repeated benchmark runs.

3. Chunked/fused cross-entropy path:
- keep behind flag and promote only on robust wins.

### 4.4 Defer (this cycle)

1. Based architecture integration.
2. ThunderKittens architecture migration.
3. Gradient checkpointing + compile interplay refactor.
4. Distributed comm/compute overlap redesign.
5. Broad custom-kernel rewrites.

## 5) Theory-Guided Sweep Framework (Primary)

Use compact hypothesis-driven sweeps as the first optimization lane for time-to-target.

### 5.1 Batch size law and critical batch size

Guiding references:
- Power Laws for Batch Size, Model Size, and Training Horizon: https://arxiv.org/abs/2505.13738
- The Batch Size-Critical Batch Size Myth: https://arxiv.org/abs/2505.23971

Operational use:
1. Start from current scaling law in `scripts/base_train.py`.
2. Sweep only a narrow neighborhood around predicted `Bopt`.
3. Reject candidates with instability or quality collapse.

### 5.2 WD-LR-horizon coupling

Guiding reference:
- How to set AdamW's weight decay as you scale model and dataset size: https://arxiv.org/abs/2405.13698

Operational use:
1. Treat `tau = B / (eta * lambda * D)` as the controlling invariant.
2. For each batch/horizon change, compute nominal LR and WD first.
3. Sweep only a small multiplicative band around nominal transfer.

### 5.3 Data/horizon track

Operational use:
1. Sweep a narrow band around current `target_param_data_ratio`.
2. Couple ratio changes with schedule adjustments (`warmdown_ratio`, `warmdown_shape`, `final_lr_frac`).
3. Prefer small local moves over broad Cartesian grids.

## 6) Phased Execution Plan

## Phase 0: Baseline + Target Lock

Deliverables:
1. Finalize one d12 baseline command profile.
2. Lock one target metric and threshold for the full cycle.
3. Capture baseline runs for two seeds.
4. Save baseline CSV/logs and baseline variance summary.

Exit criteria:
- Baseline variance understood enough to set practical thresholds.
- Target metric/threshold explicitly recorded.

## Phase 0.5: Tactical RMSNorm Check (Time-Boxed)

Work items:
1. Run 4-candidate A/B set:
- baseline
- `--fused-rmsnorm` (if implemented)
- `--chunked-ce` (if implemented)
- both flags
2. Keep all other knobs and seeds fixed.

Exit criteria:
- If no robust win, stop tactical lane and continue primary phases.

## Phase 1: Theory Transfer Micro-Sweeps (Primary)

Work items:
1. Warmdown ratio/shape micro-sweep.
2. Batch-centered micro-sweep around predicted `Bopt`.
3. Joint LR+WD micro-grid around nominal transfer.
4. Track measured/extrapolated time-to-target.

Acceptance criteria:
- Faster time-to-target or improved extrapolated time-to-target at matched quality.
- No instability across two seeds.

## Phase 2: Data/Horizon Retuning (Second Priority)

Work items:
1. Narrow sweep around `target_param_data_ratio`.
2. Couple with `final_lr_frac` and warmdown settings.
3. Keep sweep cardinality small and theory-constrained.

Acceptance criteria:
- Improves or preserves best Phase 1 candidate on time-to-target.
- No broad quality degradation.

## Phase 3: Shape Reallocation at Fixed Budget (Third Priority)

Work items:
1. Small grid over depth/aspect/head-dim combinations.
2. Enforce matched budget per candidate (matched wall-time or matched FLOPs).
3. Re-apply narrow LR/WD transfer around each shape point.

Acceptance criteria:
- Best candidate beats current best on time-to-target under matched budget.
- Stable across two seeds.

## Phase 4: Support Lane (Hardware/System)

Only run if Phase 1-3 plateau.

Work items:
1. Deferred loss logging.
2. Inductor cache reuse.
3. Chunked/fused CE (flagged).
4. Optional fused RMSNorm (flagged).

Acceptance criteria:
- >=3% median wall-time gain at matched quality vs current best candidate.
- No compile/resume regressions.

## Phase 5: Robustness Gate via Miniseries

Run `runs/miniseries.sh` baseline vs candidate with matched conditions.

Decision rule:
- Promote only if gains persist across depths and quality does not regress systematically.

## 7) Acceptance Gates, Plateau Definition, and Rollback

### 7.1 Local acceptance gate (d12)

Must pass all:
1. Primary metric: lower time-to-target across 2 seeds (measured if reached, extrapolated otherwise).
2. Proxy fallback: if target is not reached in-budget, require >=3% wall-time improvement at matched quality.
3. No training instability.
4. No checkpoint/resume regressions.
5. Quality floor: no worse than +1% relative final smoothed train loss vs baseline.
6. If val eval is enabled, no worse than +0.005 absolute `val_bpb` vs baseline median.

### 7.2 Plateau definition

Plateau is declared only if:
1. Two full cycles through Phases 1-5 are completed.
2. Each cycle yields <2% robust improvement.
3. No candidate passes miniseries promotion.

### 7.3 Based re-entry gate (strict defer policy)

Based architecture may be reconsidered only after plateau is declared by Section 7.2.

### 7.4 Rollback policy

Immediate rollback to default-off if:
1. Cross-depth regressions appear.
2. Complexity cost is high relative to gain.
3. Behavior becomes harder to reason about for contributors.

## 8) Concrete Sweep Templates

### Template A: Theory time-to-target slice

Grid:
- baseline
- warmdown update
- batch-centered update
- joint LR+WD update
- best combined theory bundle

Fixed:
- model shape, seeds, eval cadence

Outputs:
- measured/extrapolated time-to-target
- wall-time and throughput summary
- loss and val-quality comparison

### Template B: Warmdown micro-sweep

Grid:
- `warmdown_ratio`: `0.2, 0.3, 0.4, 0.5`
- `warmdown_shape`: `linear`, `cosine`

### Template C: Batch-centered micro-sweep

Grid:
- batch multipliers around predicted `Bopt`: `0.7x, 1.0x, 1.4x`

Coupled update:
- adjust LR and WD from nominal scaling rule before micro-tuning.

### Template D: Joint LR+WD micro-grid

Grid:
- LR multiplier: `{0.9, 1.0, 1.1}`
- WD multiplier: `{0.7, 1.0, 1.4}`

### Template E: Data/horizon retune

Grid:
- `target_param_data_ratio`: small local band around default
- `final_lr_frac`: small set near default
- paired warmdown settings

### Template F: Shape reallocation under fixed budget

Grid:
- small depth/aspect/head-dim combinations

Constraint:
- matched wall-time or matched FLOPs budget per candidate.

### Template G: Tactical RMSNorm A/B (time-boxed)

Grid:
- baseline
- `--fused-rmsnorm`
- `--chunked-ce`
- both

Constraint:
- 1-day maximum effort, then return to primary phases.

## 9) Minimal Implementation Backlog

Priority order:
1. Baseline harness and target-metric declaration fields.
2. Warmdown shape and batch/WD multiplier hooks in `scripts/base_train.py`.
3. Theory micro-sweep templates (warmdown, batch-centered, LR+WD).
4. Data/horizon retune template and runner.
5. Shape reallocation fixed-budget template.
6. Tactical RMSNorm A/B harness.
7. Support-lane hardware/system tweaks.

Explicitly deferred backlog:
- Based architecture integration.
- ThunderKittens architecture integration.
- broad custom-kernel architecture rewrites.

## 10) Execution Checklist

1. Baseline:
- lock target metric + threshold
- run baseline with 2 seeds
- archive CSV and logs
2. Tactical exception:
- run 1-day RMSNorm/CE A/B set and decide quickly
3. Primary phase:
- run theory transfer micro-sweeps
4. Secondary phase:
- run data/horizon retune
5. Third phase:
- run shape reallocation under fixed budget
6. Support lane (only on plateau):
- run hardware/system support candidates
7. Robustness:
- run `runs/miniseries.sh`
- compare baseline vs candidate across depths
8. Decision:
- promote defaults only when all gates pass
- otherwise keep candidate behind opt-in knobs

## 11) Change Log (keep updated during execution)

For each experiment:
- date
- candidate change
- measured gains/losses
- decision (promote, keep optional, reject)

---

This document is the canonical execution plan for the current optimization cycle. It intentionally prioritizes theory/data/shape transfer wins and defers Based architecture until strict plateau criteria are met.
