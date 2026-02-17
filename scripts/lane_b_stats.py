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
import powerlaw  # noqa: E402

MASK64 = (1 << 64) - 1
HASH_BASE = 11400714819323198485  # odd 64-bit constant


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fit_power_law(xs, ys, optuna_seed=42, optuna_trials=200):
    """Fit y = a * x^{-exponent} using powerlaw PDF basis + optuna search."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    valid = (xs > 0) & (ys > 0)
    xs, ys = xs[valid], ys[valid]
    if xs.size < 2:
        raise ValueError("Need at least two positive points for power-law fit")

    ly = np.log(ys)
    xmin, xmax = float(np.min(xs)), float(np.max(xs))

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 5.0)
        try:
            dist = powerlaw.Power_Law(parameters=[alpha], xmin=xmin, xmax=xmax, discrete=False)
            phi = np.asarray(dist.pdf(xs), dtype=np.float64)
        except KeyboardInterrupt:
            raise
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return float("inf")
        if np.any(phi <= 0) or np.any(~np.isfinite(phi)):
            return float("inf")
        ln_phi = np.log(phi)
        log_scale = float(np.mean(ly - ln_phi))
        residual = ly - (log_scale + ln_phi)
        return float(np.sum(residual * residual))

    if int(optuna_trials) < 1:
        raise ValueError("optuna_trials must be >= 1")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(optuna_seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(optuna_trials), n_jobs=1, show_progress_bar=False)

    best_alpha = study.best_params["alpha"]
    dist = powerlaw.Power_Law(parameters=[best_alpha], xmin=xmin, xmax=xmax, discrete=False)
    phi = np.asarray(dist.pdf(xs), dtype=np.float64)
    ln_phi = np.log(phi)
    log_scale = float(np.mean(ly - ln_phi))

    exponent = float(best_alpha)
    norm_const = float(np.mean(phi * np.power(xs, exponent)))
    a = float(math.exp(log_scale) * norm_const)

    ss_res = float(study.best_value)
    ly_mean = float(np.mean(ly))
    ss_tot = float(np.sum((ly - ly_mean) ** 2))
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return a, exponent, r2


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
    parser.add_argument("--max-tokens", type=int, default=500000, help="max number of tokens to sample (-1 for no explicit cap)")
    parser.add_argument("--tokenizer-threads", type=int, default=4, help="threads used by tokenizer batch encoding")
    parser.add_argument("--correlation-min-sep", type=int, default=1, help="minimum token separation for correlation fit")
    parser.add_argument("--correlation-max-sep", type=int, default=64, help="maximum token separation for correlation fit")
    parser.add_argument("--entropy-min-ctx", type=int, default=1, help="minimum context length for entropy-gap fit")
    parser.add_argument("--entropy-max-ctx", type=int, default=64, help="maximum context length for entropy-gap fit")
    parser.add_argument("--entropy-num-points", type=int, default=8, help="number of log-spaced context lengths")
    parser.add_argument("--fit-optuna-seed", type=int, default=42, help="fixed Optuna sampler seed for deterministic power-law fits")
    parser.add_argument("--fit-optuna-trials", type=int, default=200, help="Optuna trial count per power-law fit")
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

    h_inf_bits = min(item["conditional_entropy_bits"] for item in entropy_curve)
    eps = 1e-12
    for item in entropy_curve:
        item["entropy_gap_bits"] = max(item["conditional_entropy_bits"] - h_inf_bits, eps)

    ent_x = [x["context_len"] for x in entropy_curve]
    ent_y = [x["entropy_gap_bits"] for x in entropy_curve]
    _, ent_exp, ent_r2 = fit_power_law(
        ent_x,
        ent_y,
        optuna_seed=args.fit_optuna_seed,
        optuna_trials=args.fit_optuna_trials,
    )

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
        "entropy_fit_min_ctx": int(args.entropy_min_ctx),
        "entropy_fit_max_ctx": int(args.entropy_max_ctx),
        "entropy_decay_exponent": float(ent_exp),
        "entropy_decay_r2": float(ent_r2),
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
    print(f"corr_decay_exponent={corr_exp:.6f}, entropy_decay_exponent={ent_exp:.6f}")


if __name__ == "__main__":
    main()
