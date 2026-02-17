# Lane B Theory: Inferring `target_param_data_ratio` From Data Statistics

## Goal
Set `target_param_data_ratio` from dataset-structure priors, not broad sweep fitting.

Primary objective:
- minimize threshold hitting time while preserving generality beyond one empirical sweep.

## Motivation
Core concern:
- horizon tuning can overfit the current training dataset if treated as pure local search.

Theory response:
- infer data-limited scaling behavior from measurable dataset statistics, then derive the ratio from that inferred law.

Primary reference:
- Deriving Neural Scaling Laws from the statistics of natural language (`arXiv:2602.07488`): https://www.arxiv.org/pdf/2602.07488

## Evidence vs Inference
Evidence from repo:
- `target_param_data_ratio` is first-class in `scripts/base_train.py`.
- speedrun success historically depends strongly on horizon choices.

Inference from literature:
- scaling exponents depend on data characteristics (not a universal constant across all corpora).
- therefore, preferred ratio should be inferred from measured corpus statistics.

## Theory Core
Use a two-step decomposition:

1. Infer a data-limited exponent from dataset statistics:
- statistic A: decay of pairwise token correlation with token separation
- statistic B: decay of conditional entropy gap vs context length
- use mapping from `2602.07488` to infer a dataset-specific scaling exponent

2. Convert exponent into an inferred target token budget:
- fit local constants in a simple power-law form using minimal calibration points
- solve for token budget at target proxy
- divide by scaling parameter count to get inferred `target_param_data_ratio`

## Ideal Figure To Motivate Setting
Figure 1 (data stats -> exponent):
- x-axis: context length / token separation
- y-axis: measured correlation and entropy-decay curves
- outputs: inferred data-limited exponent

Figure 2 (tokens -> target crossing):
- x-axis: training tokens
- y-axis: target proxy gap to threshold
- intersection gives inferred required tokens

ASCII sketch:

```text
(1) data statistics                      (2) target token solve
metric ^                                 proxy gap ^
       |\  corr decay                              |\
       | \ entropy decay                           | \__  solve D_target here
       +-------------> separation/context          +----------------------> tokens D
```

## Parameter-Setting Procedure (Decision Complete)
1. Compute dataset statistics on the actual training corpus split used for speedrun.
2. Infer dataset-specific scaling exponent via `2602.07488` mapping.
3. Calibrate constants with minimal points (small number of short controlled runs).
4. Solve for `inferred_target_tokens` at chosen target proxy.
5. Set:
- `inferred_target_param_data_ratio = inferred_target_tokens / num_scaling_params`
6. Execute one confirmation neighbor only:
- `confirmation_ratio = inferred_target_param_data_ratio * 1.05`

No wide ratio grid in this cycle.

Operational worksheet:
- use `ideas/lane_b_inference_worksheet.md` for exact computation steps
- log every run using the minimal decision+outcome schema in `ideas/lane_b_inference_schema.csv`
- use automation helpers: `scripts/lane_b_stats.py`, `scripts/lane_b_infer_ratio.py`, and `scripts/lane_b_log_run.py`

## Acceptance And Falsification
Accept if:
- inferred ratio beats baseline on time-to-target across 2 seeds
- quality floor and stability gates hold
- miniseries sanity passes

Reject if:
- inferred ratio underperforms and conservative neighbor also fails
- calibration is too unstable to produce consistent inferred tokens

Fallback on rejection:
- revert to current default `target_param_data_ratio` from `scripts/base_train.py` (currently **10.5**)

## Practical Notes
- This lane deliberately trades broad experimentation for a theory-driven prior + minimal confirmation.
- If dataset distribution changes materially, recompute inference rather than reusing old ratio.

## References
- `2602.07488`: https://www.arxiv.org/pdf/2602.07488
- Chinchilla compute-optimal framing: https://arxiv.org/abs/2203.15556
- Power lines batch/horizon context: https://arxiv.org/abs/2505.13738
