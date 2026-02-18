"""
Estimate Lane B dataset statistics from the active pretraining corpus.

Outputs correlation-decay and entropy-gap decay fits that are used by
`scripts/lane_b_infer_ratio.py`.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np  # noqa: E402
import optuna  # noqa: E402

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


def fit_power_law_simple(xs, ys, optuna_seed=42, optuna_trials=200):
    """Fit y = a * x^{-exponent} using linear regression on log-log data."""
    if int(optuna_trials) < 1:
        raise ValueError("optuna_trials must be >= 1")
    xs, ys = _sanitize_power_law_inputs(xs, ys, min_points=2)

    log_x = np.log(xs)
    log_y = np.log(ys)

    # Linear regression: log_y = -exponent * log_x + log_a
    slope, intercept = np.polyfit(log_x, log_y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("Non-finite linear fit parameters in fit_power_law_simple")
    exponent = -slope
    a = np.exp(intercept)

    # Calculate R2
    y_pred = a * np.power(xs, -exponent)
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

    log_x = np.log(xs)

    def objective(trial):
        b = trial.suggest_float("b", 0.0, b_upper)

        y_shifted = ys - b
        if np.any(y_shifted <= 0):
            return float("inf")

        # Fit simple power law to shifted data: y - b = A * x^-gamma
        log_y_shifted = np.log(y_shifted)
        slope, intercept = np.polyfit(log_x, log_y_shifted, 1)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return float("inf")

        # We want to minimize the squared error of the linear fit in log-log space
        # This roughly corresponds to maximizing the likelihood of the power law
        y_fit_log = slope * log_x + intercept
        residual = log_y_shifted - y_fit_log
        mse = np.mean(residual ** 2)
        return float(mse) if np.isfinite(mse) else float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(optuna_seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(optuna_trials), n_jobs=1, show_progress_bar=False)
    if not np.isfinite(study.best_value):
        return simple_a, simple_gamma, 0.0, simple_r2

    best_b = float(study.best_params.get("b", 0.0))
    if not (0.0 <= best_b < min_y):
        return simple_a, simple_gamma, 0.0, simple_r2

    # Re-compute parameters with best B
    y_shifted = ys - best_b
    if np.any(y_shifted <= 0):
        return simple_a, simple_gamma, 0.0, simple_r2
    log_x = np.log(xs)
    log_y_shifted = np.log(y_shifted)
    slope, intercept = np.polyfit(log_x, log_y_shifted, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return simple_a, simple_gamma, 0.0, simple_r2

    gamma = -slope
    a = np.exp(intercept)

    # Calculate R2 on the original scale
    y_pred = a * np.power(xs, -gamma) + best_b
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
    sum_p_sq = float(np.sum(probs * probs))
    eps = 1e-12

    curve = []
    for sep in range(min_sep, max_sep + 1):
        n_pairs = tokens.size - sep
        # Accumulate sparse joint counts count(mu, nu) for lag=sep
        joint = Counter(zip(tokens[:-sep].tolist(), tokens[sep:].tolist()))
        # ||C(n)||_F^2 = sum_{mu,nu} [P(mu,nu|n) - P(mu)*P(nu)]^2
        # Expand: sum P(mu,nu)^2 - 2*sum P(mu,nu)*P(mu)*P(nu) + sum P(mu)^2 * sum P(nu)^2
        # The last term = sum_p_sq^2 (marginals are the same distribution)
        sum_pjoint_sq = 0.0
        cross_term = 0.0
        for (mu, nu), c in joint.items():
            p_joint = c / n_pairs
            sum_pjoint_sq += p_joint * p_joint
            cross_term += p_joint * probs[mu] * probs[nu]
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
    parser.add_argument("--fit-optuna-seed", type=int, default=42, help="fixed Optuna sampler seed for deterministic power-law fits")
    parser.add_argument("--fit-optuna-trials", type=int, default=200, help="Optuna trial count per power-law fit")
    parser.add_argument("--corr-r2-warn-threshold", type=float, default=0.90, help="warn when corr power-law fit R^2 falls below this value")
    parser.add_argument("--entropy-r2-warn-threshold", type=float, default=0.90, help="warn when entropy power-law fit R^2 falls below this value")
    parser.add_argument("--output-json", type=str, required=True, help="path to output JSON")
    args = parser.parse_args()

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

    context_lengths = build_context_lengths(args.entropy_min_ctx, args.entropy_max_ctx, args.entropy_num_points)
    entropy_curve = []
    skipped_data_starved = 0
    for ctx_len in context_lengths:
        if ctx_len >= tokens.size - 1:
            continue
        h_bits, unique_ctx, is_data_starved = conditional_entropy_bits(tokens, ctx_len)
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
    if skipped_data_starved > 0:
        print(f"Skipped {skipped_data_starved} context length(s) due to data starvation (>90% unique contexts)")
    if len(entropy_curve) < 2:
        raise ValueError("Need at least two entropy points; increase token sample or reduce context window")

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
