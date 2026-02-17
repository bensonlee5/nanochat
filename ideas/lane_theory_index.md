# Lane Theory Index

## Purpose
Map the active execution plan to theory justification documents.

Primary runbook:
- `ideas/codex_speedrun_execution_plan.md`

Lane theory docs:
- `ideas/lane_b_data_horizon_theory.md` (active primary lane)
- `ideas/lane_b_inference_worksheet.md` (active worksheet)
- `ideas/lane_b_inference_schema.csv` (minimal decision+outcome schema)
- `ideas/lane_c_fusion_theory.md` (active secondary lane)
- `ideas/lane_a_schedule_tail_theory.md` (archived/skipped this cycle)

## Lane Summary
Lane B (active, primary):
- Hypothesis: data characteristics determine scaling behavior; infer `target_param_data_ratio` from measured corpus statistics.
- Ideal figure: data-statistics-to-exponent map, then tokens-to-threshold solve.
- Expected gain mode: fewer wasted steps from better horizon setting.
- Effort: low-to-medium, with minimal calibration and one neighbor confirmation.

Lane C (active, secondary):
- Hypothesis: CE/RMSNorm path optimization reduces per-step wall time.
- Ideal figure: systems throughput shift plus earlier threshold crossing.
- Expected gain mode: throughput -> net time-to-target improvement (if quality preserved).
- Effort: medium due to implementation/compiler stability risk.

Lane A (skipped):
- Not active this cycle because objective is hitting-time and tail tuning is less relevant before diminishing-return regime.

## How To Use
1. Apply Lane B inferred ratio workflow first.
2. Apply Lane C tactical matrix second.
3. Run miniseries on best candidate.
4. Revisit Lane A only if explicit re-entry conditions are met.

## Scope Guard
This cycle prefers theory-inferred settings over broad exploratory sweeps.
