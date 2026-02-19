"""
Estimate Lane B dataset statistics from the active pretraining corpus.

Outputs correlation-decay and entropy-gap decay fits that are used by
`scripts/lane_b_infer_ratio.py`.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import powerlaw  # noqa: E402

MASK64 = (1 << 64) - 1
HASH_BASE = 11400714819323198485  # odd 64-bit constant


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_power_law_inputs(xs, ys, min_points):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    xs, ys = xs[valid], ys[valid]
    if xs.size < min_points:
        raise ValueError(f"Need at least {min_points} positive finite points for power-law fit")

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if np.unique(xs).size < 2:
        raise ValueError("Need at least two distinct x-values for power-law fit")
    return xs, ys


def _power_law_basis(xs, exponent, xmin=None, xmax=None):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        raise ValueError("xs must be non-empty")
    if xmin is None:
        xmin = float(np.min(xs))
    if xmax is None:
        xmax = float(np.max(xs))
    if xmax <= xmin:
        xmax = xmin * (1.0 + 1e-9)

    model = powerlaw.Power_Law(
        xmin=float(xmin),
        xmax=float(xmax),
        discrete=False,
        parameters=[float(exponent)],
        fit_method="Likelihood",
    )
    basis = np.asarray(model.pdf(xs), dtype=np.float64)
    if (not np.all(np.isfinite(basis))) or np.any(basis <= 0):
        raise ValueError("Non-finite or non-positive power-law basis values")
    return basis


def fit_power_law_simple(xs, ys, optuna_seed=42, optuna_trials=200):
    """Fit y = a * x^{-exponent} using powerlaw package basis + Optuna."""
    if int(optuna_trials) < 1:
        raise ValueError("optuna_trials must be >= 1")
    xs, ys = _sanitize_power_law_inputs(xs, ys, min_points=2)
    xmin = float(np.min(xs))
    xmax = float(np.max(xs))

    def objective(trial):
        exponent = trial.suggest_float("exponent", 1e-3, 6.0)
        try:
            basis = _power_law_basis(xs, exponent, xmin=xmin, xmax=xmax)
        except ValueError:
            return float("inf")

        denom = float(np.dot(basis, basis))
        if denom <= 0:
            return float("inf")
        a = float(np.dot(basis, ys) / denom)
        if (not np.isfinite(a)) or a <= 0:
            return float("inf")

        y_pred = a * basis
        mse = float(np.mean((ys - y_pred) ** 2))
        return mse if np.isfinite(mse) else float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(optuna_seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(optuna_trials), n_jobs=1, show_progress_bar=False)
    if (study.best_params is None) or (not np.isfinite(study.best_value)):
        raise ValueError("Power-law fit failed to find finite parameters")

    exponent = float(study.best_params["exponent"])
    basis = _power_law_basis(xs, exponent, xmin=xmin, xmax=xmax)
    denom = float(np.dot(basis, basis))
    if denom <= 0:
        raise ValueError("Degenerate basis in fit_power_law_simple")
    a = float(np.dot(basis, ys) / denom)
    if (not np.isfinite(a)) or a <= 0:
        raise ValueError("Non-finite or non-positive amplitude in fit_power_law_simple")

    y_pred = a * basis
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    if not np.isfinite(a) or not np.isfinite(exponent) or not np.isfinite(r2):
        raise ValueError("Non-finite fit result in fit_power_law_simple")

    return float(a), float(exponent), float(r2)


def fit_power_law(xs, ys, optuna_seed=42, optuna_trials=200):
    """Backward-compatible power-law API used by Lane B tests and callers."""
    return fit_power_law_simple(xs, ys, optuna_seed=optuna_seed, optuna_trials=optuna_trials)


def fit_shifted_power_law(xs, ys, optuna_seed=42, optuna_trials=200):
    """Fit y = A * x^{-gamma} + B by optimizing B via Optuna."""
    if int(optuna_trials) < 1:
        raise ValueError("optuna_trials must be >= 1")
    xs, ys = _sanitize_power_law_inputs(xs, ys, min_points=2)

    simple_a, simple_gamma, simple_r2 = fit_power_law_simple(
        xs, ys, optuna_seed=optuna_seed, optuna_trials=optuna_trials
    )
    if xs.size < 3:
        # Not enough points for a stable asymptote fit.
        return simple_a, simple_gamma, 0.0, simple_r2

    min_y = float(np.min(ys))
    y_span = float(np.max(ys) - min_y)
    margin = max(1e-12, 1e-6 * max(abs(min_y), y_span, 1.0))
    b_upper = min_y - margin
    if b_upper <= 0:
        # If the asymptote range collapses, use the simpler model.
        return simple_a, simple_gamma, 0.0, simple_r2

    xmin = float(np.min(xs))
    xmax = float(np.max(xs))

    def objective(trial):
        b = trial.suggest_float("b", 0.0, b_upper)
        gamma = trial.suggest_float("gamma", 1e-3, 6.0)

        y_shifted = ys - b
        if np.any(y_shifted <= 0):
            return float("inf")

        try:
            basis = _power_law_basis(xs, gamma, xmin=xmin, xmax=xmax)
        except ValueError:
            return float("inf")

        denom = float(np.dot(basis, basis))
        if denom <= 0:
            return float("inf")
        a = float(np.dot(basis, y_shifted) / denom)
        if (not np.isfinite(a)) or a <= 0:
            return float("inf")

        y_pred = a * basis + b
        mse = float(np.mean((ys - y_pred) ** 2))
        return float(mse) if np.isfinite(mse) else float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(optuna_seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(optuna_trials), n_jobs=1, show_progress_bar=False)
    if not np.isfinite(study.best_value):
        return simple_a, simple_gamma, 0.0, simple_r2

    best_b = float(study.best_params.get("b", 0.0))
    best_gamma = float(study.best_params.get("gamma", simple_gamma))
    if not (0.0 <= best_b < min_y):
        return simple_a, simple_gamma, 0.0, simple_r2

    # Re-compute parameters with best B and gamma
    y_shifted = ys - best_b
    if np.any(y_shifted <= 0):
        return simple_a, simple_gamma, 0.0, simple_r2
    try:
        basis = _power_law_basis(xs, best_gamma, xmin=xmin, xmax=xmax)
    except ValueError:
        return simple_a, simple_gamma, 0.0, simple_r2

    denom = float(np.dot(basis, basis))
    if denom <= 0:
        return simple_a, simple_gamma, 0.0, simple_r2
    gamma = best_gamma
    a = float(np.dot(basis, y_shifted) / denom)
    if not np.isfinite(a):
        return simple_a, simple_gamma, 0.0, simple_r2

    # Calculate R2 on the original scale
    y_pred = a * basis + best_b
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    if (
        (not np.isfinite(a))
        or (not np.isfinite(gamma))
        or (not np.isfinite(r2))
        or gamma <= 0
        or a <= 0
        or (r2 + 1e-9 < simple_r2)
    ):
        return simple_a, simple_gamma, 0.0, simple_r2

    return float(a), float(gamma), float(best_b), float(r2)


def build_context_lengths(min_ctx, max_ctx, num_points):
    if min_ctx < 1:
        raise ValueError("--entropy-min-ctx must be >= 1")
    if max_ctx < min_ctx:
        raise ValueError("--entropy-max-ctx must be >= --entropy-min-ctx")
    if num_points < 2:
        return [min_ctx, max_ctx] if min_ctx != max_ctx else [min_ctx]

    grid = np.geomspace(min_ctx, max_ctx, num=num_points)
    values = {int(round(v)) for v in grid}
    values.add(min_ctx)
    values.add(max_ctx)
    return sorted(v for v in values if v >= 1)


def collect_token_sample(split, max_docs, max_tokens, tokenizer_threads):
    from nanochat.dataset import parquets_iter_batched
    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    token_buffer = []
    docs_seen = 0

    for text_batch in parquets_iter_batched(split):
        if max_docs > 0 and docs_seen >= max_docs:
            break
        if max_tokens > 0 and len(token_buffer) >= max_tokens:
            break

        if max_docs > 0:
            remaining_docs = max_docs - docs_seen
            text_batch = text_batch[:remaining_docs]

        encoded = tokenizer.encode(text_batch, prepend=bos, num_threads=tokenizer_threads)
        for doc_tokens in encoded:
            token_buffer.extend(doc_tokens)
            docs_seen += 1
            if max_docs > 0 and docs_seen >= max_docs:
                break
            if max_tokens > 0 and len(token_buffer) >= max_tokens:
                break

    if max_tokens > 0 and len(token_buffer) > max_tokens:
        token_buffer = token_buffer[:max_tokens]

    tokens = np.asarray(token_buffer, dtype=np.int64)
    return tokens, docs_seen, tokenizer.get_vocab_size()


def compute_correlation_curve(tokens, min_sep, max_sep):
    """Compute ||C(n)||_F for each lag n using sparse joint-count accumulation."""
    if min_sep < 1:
        raise ValueError("--correlation-min-sep must be >= 1")
    if max_sep < min_sep:
        raise ValueError("--correlation-max-sep must be >= --correlation-min-sep")
    if tokens.size <= max_sep:
        raise ValueError("Not enough tokens for requested correlation separation window")

    # Marginal distribution: P(mu) for each token mu
    counts = np.bincount(tokens)
    probs = counts.astype(np.float64) / tokens.size
    # sum P(mu)^2 — used for the Frobenius norm correction term
    sum_p_sq = float(np.dot(probs, probs))
    vocab_size = int(counts.size)
    max_pair_id = (vocab_size * vocab_size) - 1
    pair_key_dtype = np.uint32 if max_pair_id <= np.iinfo(np.uint32).max else np.uint64
    pair_vocab_mult = pair_key_dtype(vocab_size)
    pair_tokens = tokens.astype(pair_key_dtype, copy=False)
    eps = 1e-12

    curve = []
    for sep in range(min_sep, max_sep + 1):
        n_pairs = tokens.size - sep
        left = tokens[:-sep]
        right = tokens[sep:]

        # Encode token pairs into scalar keys and count each distinct pair in vectorized NumPy code.
        pair_ids = pair_tokens[:-sep] * pair_vocab_mult + pair_tokens[sep:]
        _, pair_counts = np.unique(pair_ids, return_counts=True)
        pair_counts = pair_counts.astype(np.float64, copy=False)
        n_pairs_f = float(n_pairs)
        sum_pjoint_sq = float(np.dot(pair_counts, pair_counts) / (n_pairs_f * n_pairs_f))

        # cross_term = E_{(mu,nu)~P(mu,nu|n)}[P(mu)P(nu)]
        cross_term = float(np.mean(probs[left] * probs[right]))

        # ||C(n)||_F^2 = sum_{mu,nu} [P(mu,nu|n) - P(mu)*P(nu)]^2
        # Expand: sum P(mu,nu)^2 - 2*sum P(mu,nu)*P(mu)*P(nu) + sum P(mu)^2 * sum P(nu)^2
        # The last term = sum_p_sq^2 (marginals are the same distribution)
        frob_sq = sum_pjoint_sq - 2 * cross_term + sum_p_sq * sum_p_sq
        frob_norm = max(math.sqrt(max(frob_sq, 0.0)), eps)
        curve.append(
            {
                "separation": sep,
                "frobenius_norm": frob_norm,
            }
        )
    return curve, sum_p_sq


def conditional_entropy_bits(tokens, context_len, uniqueness_threshold=0.9):
    """Estimate H(X_{n+1} | X_1..X_n) in bits.

    Returns (entropy_bits, num_unique_contexts, is_data_starved).
    is_data_starved is True when >uniqueness_threshold of contexts appear only once,
    meaning the entropy estimate is unreliable (biased toward 0).
    """
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    if tokens.size <= context_len:
        raise ValueError("Not enough tokens for requested context length")

    if context_len == 1:
        pow_l = 1
    else:
        pow_l = pow(HASH_BASE, context_len - 1, 1 << 64)

    h = 0
    for t in tokens[:context_len]:
        h = ((h * HASH_BASE) + int(t) + 1) & MASK64

    totals = defaultdict(int)
    next_counts = defaultdict(lambda: defaultdict(int))
    n = int(tokens.size)
    total_positions = n - context_len

    for idx in range(context_len, n):
        nxt = int(tokens[idx])
        totals[h] += 1
        next_counts[h][nxt] += 1

        if idx + 1 >= n:
            break

        old = int(tokens[idx - context_len]) + 1
        new = int(tokens[idx]) + 1
        h = (h - old * pow_l) & MASK64
        h = ((h * HASH_BASE) + new) & MASK64

    num_unique = len(totals)
    num_singletons = sum(1 for c in totals.values() if c == 1)
    is_data_starved = (num_unique > 0) and (num_singletons / num_unique > uniqueness_threshold)

    entropy_nats = 0.0
    for ctx_hash, ctx_total in totals.items():
        p_ctx = ctx_total / total_positions
        h_ctx = 0.0
        for c in next_counts[ctx_hash].values():
            p = c / ctx_total
            h_ctx -= p * math.log(p)
        entropy_nats += p_ctx * h_ctx

    entropy_bits = entropy_nats / math.log(2)
    return float(entropy_bits), num_unique, is_data_starved


def build_entropy_curve(tokens, context_lengths, uniqueness_threshold):
    entropy_curve = []
    skipped_data_starved = 0
    for ctx_len in context_lengths:
        if ctx_len >= tokens.size - 1:
            continue
        h_bits, unique_ctx, is_data_starved = conditional_entropy_bits(
            tokens,
            ctx_len,
            uniqueness_threshold=uniqueness_threshold,
        )
        if is_data_starved:
            skipped_data_starved += 1
            continue
        entropy_curve.append(
            {
                "context_len": ctx_len,
                "conditional_entropy_bits": h_bits,
                "unique_context_hashes": unique_ctx,
            }
        )
    return entropy_curve, skipped_data_starved


def build_entropy_curve_adaptive(
    tokens,
    min_ctx,
    max_ctx,
    num_points,
    min_usable_points,
    uniqueness_threshold,
):
    if min_usable_points < 2:
        raise ValueError("--entropy-min-usable-points must be >= 2")

    requested_lengths = build_context_lengths(min_ctx, max_ctx, num_points)
    requested_points = len(requested_lengths)

    effective_max_ctx = int(max_ctx)
    adaptive_reductions = 0
    last_entropy_curve = []
    last_skipped = 0

    while True:
        context_lengths = build_context_lengths(min_ctx, effective_max_ctx, num_points)
        entropy_curve, skipped_data_starved = build_entropy_curve(
            tokens=tokens,
            context_lengths=context_lengths,
            uniqueness_threshold=uniqueness_threshold,
        )
        last_entropy_curve = entropy_curve
        last_skipped = skipped_data_starved

        if len(entropy_curve) >= min_usable_points:
            return entropy_curve, skipped_data_starved, requested_points, adaptive_reductions

        if effective_max_ctx <= min_ctx:
            break
        next_max_ctx = max(min_ctx, effective_max_ctx // 2)
        if next_max_ctx >= effective_max_ctx:
            next_max_ctx = effective_max_ctx - 1
        if next_max_ctx < min_ctx:
            next_max_ctx = min_ctx
        if next_max_ctx == effective_max_ctx:
            break

        adaptive_reductions += 1
        effective_max_ctx = next_max_ctx

    raise ValueError(
        "Need at least "
        f"{min_usable_points} usable entropy points after adaptive context-window shrink; "
        f"only found {len(last_entropy_curve)} "
        f"(min_ctx={min_ctx}, max_ctx={max_ctx}, skipped_data_starved={last_skipped}). "
        "Increase --max-tokens or relax --entropy-uniqueness-threshold."
    )


def main():
    parser = argparse.ArgumentParser(description="Estimate Lane B dataset decay statistics.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="dataset split")
    parser.add_argument("--dataset-id", type=str, default="", help="dataset identifier to store in output JSON")
    parser.add_argument("--tokenizer-id", type=str, default="default", help="tokenizer identifier to store in output JSON")
    parser.add_argument("--max-docs", type=int, default=5000, help="max number of documents to sample (-1 for all available)")
    parser.add_argument("--max-tokens", type=int, default=2097152, help="max number of tokens to sample (-1 for no explicit cap)")
    parser.add_argument("--tokenizer-threads", type=int, default=4, help="threads used by tokenizer batch encoding")
    parser.add_argument("--correlation-min-sep", type=int, default=1, help="minimum token separation for correlation fit")
    parser.add_argument("--correlation-max-sep", type=int, default=64, help="maximum token separation for correlation fit")
    parser.add_argument("--entropy-min-ctx", type=int, default=1, help="minimum context length for entropy-gap fit")
    parser.add_argument("--entropy-max-ctx", type=int, default=64, help="maximum context length for entropy-gap fit")
    parser.add_argument("--entropy-num-points", type=int, default=8, help="number of log-spaced context lengths")
    parser.add_argument("--entropy-min-usable-points", type=int, default=4, help="minimum non-starved entropy points required after adaptive context-window shrink")
    parser.add_argument("--entropy-uniqueness-threshold", type=float, default=0.9, help="data-starvation threshold: skip a context length when singleton-context ratio exceeds this value")
    parser.add_argument("--fit-optuna-seed", type=int, default=42, help="fixed Optuna sampler seed for deterministic power-law fits")
    parser.add_argument("--fit-optuna-trials", type=int, default=200, help="Optuna trial count per power-law fit")
    parser.add_argument("--corr-r2-warn-threshold", type=float, default=0.90, help="warn when corr power-law fit R^2 falls below this value")
    parser.add_argument("--entropy-r2-warn-threshold", type=float, default=0.90, help="warn when entropy power-law fit R^2 falls below this value")
    parser.add_argument("--output-json", type=str, required=True, help="path to output JSON")
    args = parser.parse_args()

    if args.entropy_min_usable_points < 2:
        raise ValueError("--entropy-min-usable-points must be >= 2")
    if not 0.0 < args.entropy_uniqueness_threshold < 1.0:
        raise ValueError("--entropy-uniqueness-threshold must be in (0, 1)")

    max_docs = args.max_docs
    max_tokens = args.max_tokens
    if max_docs == -1:
        max_docs = 0
    if max_tokens == -1:
        max_tokens = 0

    tokens, docs_seen, vocab_size = collect_token_sample(
        split=args.split,
        max_docs=max_docs,
        max_tokens=max_tokens,
        tokenizer_threads=args.tokenizer_threads,
    )
    if tokens.size < 1000:
        raise ValueError(f"Token sample too small ({tokens.size}); increase --max-docs/--max-tokens")

    corr_curve, sum_p_sq = compute_correlation_curve(
        tokens=tokens,
        min_sep=args.correlation_min_sep,
        max_sep=args.correlation_max_sep,
    )
    corr_x = [x["separation"] for x in corr_curve]
    corr_y = [x["frobenius_norm"] for x in corr_curve]
    
    _, corr_exp, corr_r2 = fit_power_law(
        corr_x,
        corr_y,
        optuna_seed=args.fit_optuna_seed,
        optuna_trials=args.fit_optuna_trials,
    )

    entropy_curve, skipped_data_starved, entropy_points_requested, entropy_fit_adaptive_reductions = build_entropy_curve_adaptive(
        tokens=tokens,
        min_ctx=args.entropy_min_ctx,
        max_ctx=args.entropy_max_ctx,
        num_points=args.entropy_num_points,
        min_usable_points=args.entropy_min_usable_points,
        uniqueness_threshold=args.entropy_uniqueness_threshold,
    )
    if skipped_data_starved > 0:
        print(
            "Skipped "
            f"{skipped_data_starved} context length(s) due to data starvation "
            f"(singleton-context ratio > {args.entropy_uniqueness_threshold:.2f})."
        )
    if entropy_fit_adaptive_reductions > 0:
        print(
            "Adaptive entropy fit reduced context window "
            f"{entropy_fit_adaptive_reductions} time(s) to reach "
            f"{len(entropy_curve)} usable point(s)."
        )

    # For entropy, we fit y = A * x^-gamma + B (where B is H_inf)
    ent_x = [x["context_len"] for x in entropy_curve]
    ent_y = [x["conditional_entropy_bits"] for x in entropy_curve]
    
    _, ent_exp, h_inf_bits, ent_r2 = fit_shifted_power_law(
        ent_x,
        ent_y,
        optuna_seed=args.fit_optuna_seed,
        optuna_trials=args.fit_optuna_trials,
    )

    corr_r2_low_quality = corr_r2 < args.corr_r2_warn_threshold
    entropy_r2_low_quality = ent_r2 < args.entropy_r2_warn_threshold
    if corr_r2_low_quality:
        print(
            "WARNING: corr power-law fit appears weak "
            f"(R^2={corr_r2:.4f} < {args.corr_r2_warn_threshold:.4f}). "
            "Consider increasing --max-tokens."
        )
    if entropy_r2_low_quality:
        print(
            "WARNING: entropy power-law fit appears weak "
            f"(R^2={ent_r2:.4f} < {args.entropy_r2_warn_threshold:.4f}). "
            "Consider increasing --max-tokens."
        )

    # Post-calculate gap bits for visualization/logging using the fitted H_inf
    eps = 1e-12
    for item in entropy_curve:
        item["entropy_gap_bits"] = max(item["conditional_entropy_bits"] - h_inf_bits, eps)

    dataset_id = args.dataset_id if args.dataset_id else f"fineweb_edu_100b_{args.split}"
    result = {
        "timestamp_utc": utc_timestamp(),
        "dataset_id": dataset_id,
        "tokenizer_id": args.tokenizer_id,
        "split": args.split,
        "num_docs_sampled": int(docs_seen),
        "num_tokens_sampled": int(tokens.size),
        "vocab_size": int(vocab_size),
        "corr_fit_min_sep": int(args.correlation_min_sep),
        "corr_fit_max_sep": int(args.correlation_max_sep),
        "fit_optuna_seed": int(args.fit_optuna_seed),
        "fit_optuna_trials": int(args.fit_optuna_trials),
        "corr_decay_exponent": float(corr_exp),
        "corr_decay_r2": float(corr_r2),
        "corr_decay_r2_warn_threshold": float(args.corr_r2_warn_threshold),
        "corr_decay_r2_low_quality": "yes" if corr_r2_low_quality else "no",
        "entropy_fit_min_ctx": int(args.entropy_min_ctx),
        "entropy_fit_max_ctx": int(args.entropy_max_ctx),
        "entropy_num_points_requested": int(entropy_points_requested),
        "entropy_num_points_used": int(len(entropy_curve)),
        "entropy_fit_adaptive_reductions": int(entropy_fit_adaptive_reductions),
        "entropy_min_usable_points": int(args.entropy_min_usable_points),
        "entropy_uniqueness_threshold": float(args.entropy_uniqueness_threshold),
        "entropy_decay_exponent": float(ent_exp),
        "entropy_decay_r2": float(ent_r2),
        "entropy_decay_r2_warn_threshold": float(args.entropy_r2_warn_threshold),
        "entropy_decay_r2_low_quality": "yes" if entropy_r2_low_quality else "no",
        "entropy_h_inf_bits": float(h_inf_bits),
        "entropy_skipped_data_starved": skipped_data_starved,
        "token_unigram_collision_prob": float(sum_p_sq),
        "corr_curve": corr_curve,
        "entropy_curve": entropy_curve,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote Lane B stats to {args.output_json}")
    print(f"corr_decay_exponent={corr_exp:.6f}, entropy_decay_exponent={ent_exp:.6f}, h_inf={h_inf_bits:.4f}")


if __name__ == "__main__":
    main()
