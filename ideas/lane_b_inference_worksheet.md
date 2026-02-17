# Lane B Inference Worksheet

## Purpose
Set `target_param_data_ratio` from data-characteristics inference with a minimal, auditable decision-and-outcome log.

Use this worksheet with:
- `ideas/lane_b_data_horizon_theory.md`
- `ideas/lane_b_inference_schema.csv`
- `scripts/lane_b_stats.py`
- `scripts/lane_b_infer_ratio.py`
- `scripts/lane_b_log_run.py`

Schema contract:
- `ideas/lane_b_inference_schema.csv` is the required fixed column order for Lane B records.
- extra diagnostics are optional and should go in `notes` (or a separate lab notebook), not new CSV columns.

Automation notes:
- run stats extraction via `python -m scripts.lane_b_stats ...`
- infer ratio via `python -m scripts.lane_b_infer_ratio ...`
- append each run row via `python -m scripts.lane_b_log_run ...`
- base training now supports explicit seeds via `python -m scripts.base_train --seed=<int> ...`

## Inputs
Required inputs (must be represented in schema fields):
- `dataset_id`
- `target_metric_name` (for example `val_bpb`)
- `target_metric_threshold` (numeric)
- `n_scaling_params` (from current model setup)

Working inputs (optional notes):
- `tokenizer_id`
- `target_metric_direction` (`lower_is_better` or `higher_is_better`)
- `paper_mapping_id` and exact equation notes from `2602.07488`
- fit windows and residual diagnostics

## Step 1: Measure Dataset Statistics
Fit power-law decays on a fixed corpus sample.

1. Correlation decay fit:
- fit `C(delta) = a_corr * delta^(-p_corr)` over a fixed separation window

2. Entropy-gap decay fit:
- define entropy gap `G(L) = H_inf - H(L)`
- fit `G(L) = a_ent * L^(-p_ent)` over a fixed context window

Optional notes (not required schema columns):
- correlation fit window, exponent, and `R^2`
- entropy fit window, exponent, and `R^2`

## Step 2: Infer Data-Limited Scaling Exponent
Using the selected mapping from `2602.07488`, compute:
- `inferred_data_scaling_exponent = alpha_data`

Record in schema:
- `inferred_data_scaling_exponent`

Optional notes:
- mapping identifier/equation and a short mapping rationale

## Step 2b: Plausibility Gate
Before proceeding, check that `alpha_data` falls in a plausible range:
- expected range based on scaling literature: **0.2 – 0.8**
- if `alpha_data` is outside this range, flag for manual review before calibration/runs

Optional note:
- `alpha_data_plausible` (`yes` or `no`)

## Step 3: Calibrate Local Loss-vs-Tokens Constants
Use calibration points from controlled runs (3 recommended, 2 is the hard minimum):
- `(D1, L1)`, `(D2, L2)`, and optionally `(D3, L3)`
- `D` is training tokens seen
- `L` is value of `target_metric_name` at corresponding token count

Assume local form:
- `L(D) = L_inf + A * D^(-alpha_data)`

With 2 points, solve:
- `A = (L1 - L2) / (D1^(-alpha_data) - D2^(-alpha_data))`
- `L_inf = L1 - A * D1^(-alpha_data)`

With 3+ points, fit via least-squares and compute residual/`R^2`.

Optional notes:
- calibration points and metric values
- `calib_fit_r2`, `calib_A`, `calib_L_inf`

## Step 4: Solve Inferred Target Tokens
Solve for `inferred_target_tokens = D_target` using threshold.

For `lower_is_better` metric:
- require `target_metric_threshold > L_inf`
- `D_target = (A / (target_metric_threshold - L_inf))^(1 / alpha_data)`

For `higher_is_better` metric:
- convert to equivalent monotonic loss proxy before solving, or define a mirrored fit form and document it in notes.

Unreachable-threshold guard:
- if `D_target` exceeds 10x the baseline token budget (i.e., `10 * default_ratio * n_scaling_params` where `default_ratio = 10.5`), mark run as likely unreachable

Record in schema:
- `inferred_target_tokens`
- `status` (`likely_unreachable` when this guard triggers)

## Step 5: Convert to Ratio
Compute:
- `inferred_target_param_data_ratio = inferred_target_tokens / n_scaling_params`

Set one conservative confirmation neighbor:
- `confirmation_ratio = inferred_target_param_data_ratio * 1.05`

Choose run ratio:
- `chosen_ratio` is either inferred ratio or conservative neighbor after one confirmation run
- set `lane_b_decision` to `inferred`, `confirmation`, or `fallback`

Explicit fallback:
- if both inferred and confirmation ratios fail acceptance, `chosen_ratio` falls back to the current default in `scripts/base_train.py` (currently `target_param_data_ratio = 10.5`)
- set `lane_b_decision = fallback`

Record in schema:
- `inferred_target_param_data_ratio`
- `confirmation_ratio`
- `chosen_ratio`
- `lane_b_decision`

## Step 6: Run and Record Outcomes
For each run, append one row to `ideas/lane_b_inference_schema.csv` in this exact order:

1. `timestamp_utc`
2. `git_commit`
3. `run_id`
4. `seed`
5. `dataset_id`
6. `target_metric_name`
7. `target_metric_threshold`
8. `n_scaling_params`
9. `inferred_data_scaling_exponent`
10. `inferred_target_tokens`
11. `inferred_target_param_data_ratio`
12. `confirmation_ratio`
13. `chosen_ratio`
14. `lane_b_decision`
15. `time_to_target_sec_measured`
16. `time_to_target_sec_extrapolated`
17. `min_val_bpb`
18. `tok_per_sec_median`
19. `dt_median_ms`
20. `status`
21. `notes`

## Guardrails
- Do not run broad ratio grids in this cycle.
- If both inferred and confirmation ratios fail, mark `lane_b_decision=fallback` and stop Lane B expansion.
- Recompute inference when dataset distribution changes.
