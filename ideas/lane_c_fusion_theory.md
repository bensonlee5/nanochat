# Lane C Theory: Tactical Fusion (RMSNorm and CE Path)

## Goal
Lane C targets wall-clock reduction per step through compute-path simplification:
- `--fused-rmsnorm`
- `--chunked-ce`

Status note:
- these flags are planned in execution docs but are pending implementation wiring in current code.

Primary objective:
- improve net time-to-target, not just raw throughput.

## What Is Evidence vs Inference
Evidence from repo:
- model currently uses functional RMSNorm (`F.rms_norm`) in `nanochat/gpt.py`.
- plan includes tactical Lane C A/B matrix.

Inference from systems theory:
- normalization and CE/logits paths are often memory-bandwidth limited.
- fusion/chunking can reduce kernel launches, memory traffic, and synchronization overhead.
- net win depends on preserving step quality and compiler stability.

Important caveat:
- `torch.compile` may already fuse `F.rms_norm` into surrounding ops. Before writing any custom fused RMSNorm kernel, profile a compiled baseline run to check whether the compiler already fuses it. If it does, skip `--fused-rmsnorm` and focus effort on chunked CE instead.

## Theory Core
Roofline-style argument:
- if a path is memory-bound, reducing bytes moved often yields better speed than increasing arithmetic throughput.
- fused kernels can increase effective arithmetic intensity and lower launch overhead.

For CE/logits:
- chunking/fusing can reduce peak activation/logit materialization pressure.
- this may improve throughput and memory headroom.

For RMSNorm:
- fusion can reduce read/write passes over activations.

## Ideal Figure To Motivate Search
Panel A (systems):
- x-axis: arithmetic intensity
- y-axis: achieved throughput
- baseline point vs fused point (fused shifts up/right)

Panel B (training):
- x-axis: wall-clock time
- y-axis: quality proxy
- if quality per step is preserved, fused curve crosses target earlier.

ASCII sketch:

```text
Panel A (roofline-ish)               Panel B (time-to-target)
throughput ^                         quality proxy ^
           |      fused *                         |\ baseline
           |   baseline *                         | \__ fused (earlier crossing)
           +-------------> intensity             +-----------------> time
```

## Pre-Implementation Check: torch.compile Baseline
Before implementing any fused kernels:
1. Run a compiled baseline with `torch.compile` and profile the generated graph.
2. Check whether `F.rms_norm` is already fused by the compiler into surrounding ops.
3. If yes: skip `--fused-rmsnorm` entirely and focus on `--chunked-ce`.
4. If no: proceed with manual fused RMSNorm implementation.

## Recommended Trial Set
Minimal A/B matrix:
- baseline
- CE only (`--chunked-ce`)
- RMSNorm only (`--fused-rmsnorm`) — only if pre-implementation check shows compiler does not already fuse it
- both (if applicable)

Fixed controls:
- same seeds, eval cadence, and batch setup
- identical target metric threshold

Required metrics:
- `tok_per_sec`, `dt`
- time-to-target measured/extrapolated
- quality floor checks (`val_bpb`, loss trajectory)
- compile stability and checkpoint/resume behavior

Compile-stability verification (required for each fused/chunked variant):
- verify no graph breaks after introducing fused/chunked ops (`torch._dynamo.explain()` or compile log)
- verify checkpoint save/load round-trips work correctly (save, reload, run a few steps, compare loss)

## Acceptance And Falsification
Accept only if:
- threshold crossing time improves robustly across 2 seeds
- no meaningful quality regression
- no compile/resume regressions

Reject if:
- throughput improves but threshold-time does not
- wins are brittle or implementation complexity is disproportionate

## Practical Notes
- expected confidence is relatively high for raw throughput effect.
- engineering effort is medium because path wiring and compile interactions are the main risk.

## References
- RMSNorm: https://arxiv.org/abs/1910.07467
- PyTorch cross-entropy reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- Roofline model (original): https://doi.org/10.1145/1498765.1498785
- FlashAttention (memory-traffic perspective): https://arxiv.org/abs/2205.14135
