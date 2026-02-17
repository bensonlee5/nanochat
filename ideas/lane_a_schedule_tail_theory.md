# Lane A Theory: Schedule Tail Retuning (Skipped This Cycle)

## Status
Lane A is intentionally skipped in the current cycle.

Reason:
- speedrun objective is threshold hitting time
- if threshold is typically reached before the diminishing-return tail regime, warmdown-tail tuning has low leverage

## Why It Is Skipped Now
The objection is accepted:
- schedule-tail optimization is often endpoint-oriented
- current cycle prioritizes parameters with stronger expectation of objective-relevant optimality
- Lane B provides a stronger theory path via data-informed horizon inference

## Retained For Future Re-entry
Lane A can be reconsidered only if one of these holds:
- threshold crossing consistently occurs in/near warmdown region
- Lane B and Lane C plateau with no robust gains
- new theory suggests a nontrivial anytime-optimal tail schedule for this objective

## If Re-activated Later
Use a strict gate first:
- estimate threshold-crossing step vs warmdown start
- only run tail experiments if crossing occurs inside or near warmdown

Otherwise:
- keep default schedule

## References
- SGDR: https://arxiv.org/abs/1608.03983
- Don’t decay LR, increase batch size: https://arxiv.org/abs/1711.00489
