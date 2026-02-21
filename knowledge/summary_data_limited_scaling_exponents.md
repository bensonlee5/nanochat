# Paper Summary: Deriving Neural Scaling Laws from the Statistics of Natural Language

**arXiv:** 2602.07488
**Authors:** Cagnetta, Raventos, Ganguli, Wyart (SISSA, Stanford, JHU, EPFL)
**Venue:** ICML 2026 submission

## Core Result

The paper derives the **data-limited scaling exponent** from two measurable dataset statistics alone, with **no free parameters**:

$$\alpha_D = \gamma / (2\beta)$$

where:
- **gamma (γ):** exponent of decay of next-token conditional entropy with context length: `H_n - H_inf ~ n^{-γ}`
- **beta (β):** exponent of decay of token-token correlation strength with temporal separation: `||C(n)||_op ~ n^{-β}`

The autoregressive loss then scales as:

$$L_AR(P) - H_inf ~ P^{-γ/(2β)}$$

in the **horizon-limited regime** (where within-horizon learning is fast, i.e. δ > γ/(2β)).

## Key Concept: n*(P) — the Data-Dependent Prediction Time Horizon

**n\*(P) ≈ P^{1/(2β)}**

This is the maximum context window that the model can beneficially leverage given P training tokens. It comes from a signal-to-noise argument: the model can detect token correlations at lag n only when:

```
||C(n)||_op > c / sqrt(P)
```

Solving for n with the power-law ansatz ||C(n)||_op ~ n^{-β} gives n*(P) ~ P^{1/(2β)}.

### Physical interpretation
- At dataset size P, the model can only effectively use tokens up to n*(P) steps in the past
- As P increases, the model "unlocks" progressively longer context
- The loss reduction comes primarily from extending this horizon, not from better predictions within it

## Loss Decomposition

The autoregressive loss decomposes into:

```
L_AR(P) ≈ H_{n*(P)} + Σ_{n=1}^{n*(P)} E_n(P)
```

1. **H_{n\*(P)}**: Entropy at the prediction horizon boundary — irreducible given finite context
2. **Σ E_n(P)**: Excess loss from suboptimal use of tokens *within* the horizon

When within-horizon learning is fast (δ > γ/(2β)), the first term dominates and the scaling exponent is purely γ/(2β).

## Empirical Values

| Dataset     | γ     | β    | Predicted α_D | Observed α_D |
|-------------|-------|------|---------------|--------------|
| TinyStories | 0.34  | 0.88 | 0.19          | ~0.19        |
| WikiText    | 0.27  | 0.94 | 0.14          | ~0.14        |

## How to Measure γ and β

### Measuring γ (entropy decay):
1. Train a sufficiently large model on the full corpus
2. Compute per-position n-gram losses L_n for varying context lengths n
3. Fit power law to the decay of L_n with n (using the well-converged small-n region)
4. γ is architecture-independent — it's a property of the data distribution

### Measuring β (correlation decay):
1. Compute the token-token covariance matrix C(n) for varying lag n directly from corpus counts
2. Take the operator norm (top singular value) of C(n)
3. Fit power law to ||C(n)||_op vs n
4. Pure corpus statistic — no model needed

## Connection to nanochat

### Current Lane B Pipeline
The nanochat Lane B pipeline (`scripts/lane_b_infer_ratio.py`) already uses this paper's formula:
- `alpha_data = gamma / (2 * beta)` (line 321-325, `paper_v1` mapping)
- `scripts/lane_b_stats.py` measures γ and β from the training corpus
- The inferred α_data is used to fit a calibration curve `L(D) = L_inf + A * D^{-α_data}`
- Then solves for D_target (training tokens needed to hit a target BPB threshold)
- Converts to `inferred_target_param_data_ratio = D_target / n_scaling_params`

### What the paper gives vs. what nanochat still needs

**Paper provides directly:**
- α_D = γ/(2β) — the data scaling exponent, from corpus statistics alone

**Paper does NOT provide (nanochat still needs calibration runs for):**
- The prefactors A and L_inf in `L(D) = L_inf + A * D^{-α_D}`
- These are non-universal and depend on model architecture, tokenizer, optimizer, etc.
- The paper explicitly states that architectures in the same "universality class" share exponents but differ in prefactors

**Paper does NOT address:**
- Model scaling exponent α_N (how loss scales with parameter count N at fixed data)
- Compute-optimal allocation between N and D (Chinchilla-style)
- The paper is purely about the data-limited regime (infinite model capacity, finite data)

### Estimating n*(P) for nanochat

Given nanochat's training setup, we can estimate n*(P):
- Need to measure β for the FineWeb-Edu dataset (the actual training corpus)
- `scripts/lane_b_stats.py` already does this
- Then n*(P) = P^{1/(2β)}

For example, with β ≈ 0.9 (typical for natural language):
- P = 786M tokens (current baseline at ratio 8.25): n* ≈ 786M^{1/1.8} ≈ 786M^{0.556} ≈ ~14,000 tokens
- P = 1.16B tokens (ratio 10.5): n* ≈ 1.16B^{0.556} ≈ ~20,000 tokens

These are much larger than the model's context window (T=2048), which means **the model is context-limited, not data-limited** in the sense of n*(P). The paper's formula applies when T >> n*(P), i.e., when the context window is not the bottleneck.

**Important caveat:** The n*(P) estimate above may be unrealistically large because:
1. The signal-to-noise argument gives an upper bound on the usable horizon
2. With T=2048, the model literally cannot use context beyond 2048 tokens
3. The paper tests at academic scales (P up to ~100M tokens, T up to 512)

### Practical implications for nanochat speedrun optimization

1. **α_D tells us the diminishing returns of more data**: A lower α_D (like WikiText's 0.14) means slower improvement per additional token — more reason to keep training horizons short.

2. **The universality class assumption**: If the paper's theory holds for nanochat's model+data combination, then the data scaling exponent should be predictable from corpus statistics alone, saving many calibration runs.

3. **Prefactors still matter**: Even with known α_D, the actual loss at a given P depends on A and L_inf, which require at least 2-3 calibration points. This is exactly what the Lane B pipeline does.

4. **Context window regime matters**: The paper's clean theory applies when T >> n*(P). For nanochat with T=2048, we need to verify which regime we're in.

## Key Equations Reference

| Equation | Formula | Meaning |
|----------|---------|---------|
| Entropy decay | H_n - H_inf ~ n^{-γ} | Conditional entropy drops as power law with context |
| Correlation decay | \|\|C(n)\|\|_op ~ n^{-β} | Token correlations decay as power law with lag |
| Prediction horizon | n*(P) = P^{1/(2β)} | Max usable context given P training tokens |
| Data threshold | P*_n = n^{2β} | Min data to use context of length n |
| Scaling law | L(P) - H_inf ~ P^{-γ/(2β)} | Loss scaling with data (horizon-limited) |
| n-gram collapse | L_n(P) = n^{-γ} ℓ(P/n^{2β}) | All n-gram curves collapse to one master curve |
| General exponent | α = min(δ, γ/(2β)) | δ governs within-horizon learning speed |
