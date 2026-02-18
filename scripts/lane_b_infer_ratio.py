"""
Infer Lane B target_param_data_ratio from measured dataset statistics + calibration points.
"""

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_csv_floats(raw, arg_name):
    parts = [p.strip() for p in raw.split(",")]
    vals = [float(p) for p in parts if p]
    if not vals:
        raise ValueError(f"{arg_name} must provide at least one numeric value")
    return vals


def metric_value_to_proxy(value, metric_direction):
    value = float(value)
    if metric_direction == "lower_is_better":
        return value
    if metric_direction == "higher_is_better":
        return -value
    raise ValueError(f"Unknown metric direction: {metric_direction}")


def validate_calibration_points(tokens, metrics, metric_direction):
    pairs = sorted(zip(tokens, metrics), key=lambda p: p[0])
    if len(pairs) < 2:
        raise ValueError("Need at least 2 calibration points")

    sorted_tokens = np.asarray([p[0] for p in pairs], dtype=np.float64)
    sorted_metrics = np.asarray([p[1] for p in pairs], dtype=np.float64)

    if np.any(sorted_tokens <= 0):
        raise ValueError("Calibration token counts must all be > 0")
    if np.any(np.diff(sorted_tokens) <= 0):
        raise ValueError("Calibration token counts must be strictly increasing")

    metric_span = float(np.max(sorted_metrics) - np.min(sorted_metrics))
    if metric_span <= 0:
        raise ValueError("Calibration metric values must vary across points")

    trend_corr = float(np.corrcoef(sorted_tokens, sorted_metrics)[0, 1])
    if not math.isfinite(trend_corr):
        raise ValueError("Calibration trend correlation is non-finite")

    if metric_direction == "lower_is_better":
        net_improvement = float(sorted_metrics[0] - sorted_metrics[-1])
        if trend_corr >= 0 or net_improvement <= 0:
            raise ValueError(
                "Calibration trend is non-improving for lower_is_better "
                "(expect metric to decrease with more tokens)"
            )
    elif metric_direction == "higher_is_better":
        net_improvement = float(sorted_metrics[-1] - sorted_metrics[0])
        if trend_corr <= 0 or net_improvement <= 0:
            raise ValueError(
                "Calibration trend is non-improving for higher_is_better "
                "(expect metric to increase with more tokens)"
            )
    else:
        raise ValueError(f"Unknown metric direction: {metric_direction}")

    return (
        sorted_tokens.tolist(),
        sorted_metrics.tolist(),
        trend_corr,
        metric_span,
        net_improvement,
    )


def infer_alpha(stats, paper_mapping_id, corr_weight, entropy_weight, alpha_override):
    if alpha_override is not None:
        return float(alpha_override), "alpha_override"

    corr_exp = float(stats["corr_decay_exponent"])
    ent_exp = float(stats["entropy_decay_exponent"])

    if paper_mapping_id == "paper_v1":
        # arXiv:2602.07488 closed-form: alpha_D = gamma / (2 * beta)
        # where gamma = entropy decay exponent, beta = correlation decay exponent
        if corr_exp <= 0:
            raise ValueError("corr_decay_exponent must be > 0 for paper_v1 mapping")
        alpha = ent_exp / (2 * corr_exp)
    elif paper_mapping_id == "weighted_mean_v1":
        if corr_weight <= 0 or entropy_weight <= 0:
            raise ValueError("corr/entropy weights must both be > 0 for weighted_mean_v1")
        alpha = (corr_weight * corr_exp + entropy_weight * ent_exp) / (corr_weight + entropy_weight)
    elif paper_mapping_id == "corr_only_v1":
        alpha = corr_exp
    elif paper_mapping_id == "entropy_only_v1":
        alpha = ent_exp
    else:
        raise ValueError(f"Unknown paper_mapping_id: {paper_mapping_id}")

    return float(alpha), paper_mapping_id


def fit_calibration(
    tokens,
    metrics_proxy,
    alpha,
    l_inf_prior=None,
    l_inf_prior_weight=1.0,
    l_inf_lower_bound=None,
    l_inf_upper_bound=None,
):
    tokens = np.asarray(tokens, dtype=np.float64)
    y = np.asarray(metrics_proxy, dtype=np.float64)
    if tokens.size < 2:
        raise ValueError("Need at least 2 calibration points")
    if np.any(tokens <= 0):
        raise ValueError("Calibration token counts must all be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    if l_inf_prior is not None:
        l_inf_prior = float(l_inf_prior)
        if not math.isfinite(l_inf_prior):
            raise ValueError("l_inf_prior must be finite")
        l_inf_prior_weight = float(l_inf_prior_weight)
        if l_inf_prior_weight <= 0:
            raise ValueError("l_inf_prior_weight must be > 0 when l_inf_prior is provided")
    else:
        l_inf_prior_weight = 0.0

    if l_inf_lower_bound is not None:
        l_inf_lower_bound = float(l_inf_lower_bound)
        if not math.isfinite(l_inf_lower_bound):
            raise ValueError("l_inf_lower_bound must be finite")
    if l_inf_upper_bound is not None:
        l_inf_upper_bound = float(l_inf_upper_bound)
        if not math.isfinite(l_inf_upper_bound):
            raise ValueError("l_inf_upper_bound must be finite")
    if (
        l_inf_lower_bound is not None
        and l_inf_upper_bound is not None
        and l_inf_lower_bound > l_inf_upper_bound
    ):
        raise ValueError("l_inf_lower_bound cannot be greater than l_inf_upper_bound")

    x = np.power(tokens, -alpha)
    X = np.column_stack([x, np.ones_like(x)])
    y_aug = y
    X_aug = X
    if l_inf_prior is not None:
        prior_scale = math.sqrt(l_inf_prior_weight)
        X_aug = np.vstack([X_aug, np.asarray([0.0, prior_scale], dtype=np.float64)])
        y_aug = np.concatenate([y_aug, np.asarray([prior_scale * l_inf_prior], dtype=np.float64)])

    coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    A = float(coeffs[0])
    L_inf = float(coeffs[1])

    L_inf_clamped = L_inf
    if l_inf_lower_bound is not None:
        L_inf_clamped = max(L_inf_clamped, l_inf_lower_bound)
    if l_inf_upper_bound is not None:
        L_inf_clamped = min(L_inf_clamped, l_inf_upper_bound)
    if L_inf_clamped != L_inf:
        denom = float(np.dot(x, x))
        if denom <= 0:
            raise ValueError("Degenerate calibration inputs for constrained fit")
        L_inf = float(L_inf_clamped)
        A = float(np.dot(x, y - L_inf) / denom)

    y_hat = L_inf + A * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return A, L_inf, r2


def solve_target_tokens(A, L_inf, alpha, threshold_proxy):
    if alpha <= 0:
        return None, "invalid_alpha"
    if A <= 0:
        return None, "non_positive_A"

    gap = threshold_proxy - L_inf
    if gap <= 0:
        return None, "threshold_not_above_asymptote"

    value = A / gap
    if value <= 0:
        return None, "non_positive_ratio_inside_power"

    d_target = float(value ** (1.0 / alpha))
    if not math.isfinite(d_target) or d_target <= 0:
        return None, "non_finite_target_tokens"
    return d_target, "ok"


def main():
    parser = argparse.ArgumentParser(description="Infer Lane B ratio from stats + calibration points.")
    parser.add_argument("--stats-json", type=str, required=True, help="output JSON from scripts/lane_b_stats.py")
    parser.add_argument("--paper-mapping-id", type=str, default="paper_v1", choices=["paper_v1", "weighted_mean_v1", "corr_only_v1", "entropy_only_v1"], help="mapping used for corr/entropy exponent -> alpha (default: paper_v1)")
    parser.add_argument("--paper-mapping-notes", type=str, default="alpha_data = gamma / (2 * beta) from arXiv:2602.07488", help="free-form notes for the mapping choice")
    parser.add_argument("--corr-weight", type=float, default=1.0, help="weight on corr exponent for weighted_mean_v1")
    parser.add_argument("--entropy-weight", type=float, default=1.0, help="weight on entropy exponent for weighted_mean_v1")
    parser.add_argument("--alpha-override", type=float, default=None, help="optional direct alpha override")
    parser.add_argument("--alpha-min", type=float, default=0.2, help="plausibility lower bound")
    parser.add_argument("--alpha-max", type=float, default=0.8, help="plausibility upper bound")
    parser.add_argument("--calib-tokens", type=str, required=True, help="comma-separated token counts, e.g. 300000,600000,900000")
    parser.add_argument("--calib-metrics", type=str, required=True, help="comma-separated metric values aligned with calib-tokens")
    parser.add_argument("--target-metric-threshold", type=float, required=True, help="target threshold in metric space")
    parser.add_argument("--target-metric-direction", type=str, default="lower_is_better", choices=["lower_is_better", "higher_is_better"], help="metric direction")
    parser.add_argument("--l-inf-lower-bound", type=float, default=None, help="optional lower bound for asymptote L_inf in metric space")
    parser.add_argument("--l-inf-lower-bound-from-stats-key", type=str, default="", help="optional stats-json field name to use as L_inf lower bound in metric space (e.g. entropy_h_inf_bits)")
    parser.add_argument("--l-inf-prior", type=float, default=None, help="optional prior target for asymptote L_inf in metric space")
    parser.add_argument("--l-inf-prior-weight", type=float, default=1.0, help="strength of the optional L_inf prior term")
    parser.add_argument("--l-inf-min", type=float, default=None, help="deprecated alias of --l-inf-lower-bound")
    parser.add_argument("--n-scaling-params", type=float, required=True, help="scaling parameter count")
    parser.add_argument("--default-ratio", type=float, default=10.5, help="fallback/default ratio for feasibility guard")
    parser.add_argument("--unreachable-multiplier", type=float, default=10.0, help="multiplier for unreachable-threshold guard")
    parser.add_argument("--output-json", type=str, required=True, help="path to output JSON")
    args = parser.parse_args()

    with open(args.stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if args.l_inf_min is not None and args.l_inf_lower_bound is not None:
        raise ValueError("Use only one of --l-inf-min or --l-inf-lower-bound")
    if args.l_inf_lower_bound is not None and args.l_inf_lower_bound_from_stats_key:
        raise ValueError(
            "Use at most one lower-bound source: --l-inf-lower-bound or --l-inf-lower-bound-from-stats-key"
        )

    l_inf_lower_bound_metric = args.l_inf_lower_bound
    l_inf_lower_bound_source = "arg"
    if l_inf_lower_bound_metric is None and args.l_inf_min is not None:
        l_inf_lower_bound_metric = args.l_inf_min
        l_inf_lower_bound_source = "arg_legacy_l_inf_min"
    if args.l_inf_lower_bound_from_stats_key:
        key = args.l_inf_lower_bound_from_stats_key
        if key not in stats:
            raise ValueError(f"Stats JSON missing key requested by --l-inf-lower-bound-from-stats-key: {key}")
        l_inf_lower_bound_metric = float(stats[key])
        if not math.isfinite(l_inf_lower_bound_metric):
            raise ValueError(f"Stats key {key} produced non-finite lower bound")
        l_inf_lower_bound_source = f"stats:{key}"

    alpha_data, mapping_used = infer_alpha(
        stats=stats,
        paper_mapping_id=args.paper_mapping_id,
        corr_weight=args.corr_weight,
        entropy_weight=args.entropy_weight,
        alpha_override=args.alpha_override,
    )
    alpha_data_plausible = args.alpha_min <= alpha_data <= args.alpha_max

    calib_tokens = parse_csv_floats(args.calib_tokens, "--calib-tokens")
    calib_metrics = parse_csv_floats(args.calib_metrics, "--calib-metrics")
    if len(calib_tokens) != len(calib_metrics):
        raise ValueError("--calib-tokens and --calib-metrics must have identical lengths")

    (
        calib_tokens,
        calib_metrics,
        calib_trend_corr,
        calib_metric_span,
        calib_net_improvement,
    ) = validate_calibration_points(
        tokens=calib_tokens,
        metrics=calib_metrics,
        metric_direction=args.target_metric_direction,
    )

    if args.target_metric_direction == "lower_is_better":
        metrics_proxy = calib_metrics
        threshold_proxy = args.target_metric_threshold
        l_inf_lower_bound_proxy = l_inf_lower_bound_metric
        l_inf_upper_bound_proxy = None
    else:
        metrics_proxy = [-m for m in calib_metrics]
        threshold_proxy = -args.target_metric_threshold
        l_inf_lower_bound_proxy = None
        l_inf_upper_bound_proxy = (
            None if l_inf_lower_bound_metric is None else -float(l_inf_lower_bound_metric)
        )

    l_inf_prior_proxy = None
    if args.l_inf_prior is not None:
        l_inf_prior_proxy = metric_value_to_proxy(args.l_inf_prior, args.target_metric_direction)

    calib_A, calib_L_inf, calib_fit_r2 = fit_calibration(
        calib_tokens,
        metrics_proxy,
        alpha_data,
        l_inf_prior=l_inf_prior_proxy,
        l_inf_prior_weight=args.l_inf_prior_weight,
        l_inf_lower_bound=l_inf_lower_bound_proxy,
        l_inf_upper_bound=l_inf_upper_bound_proxy,
    )
    inferred_target_tokens, solve_status = solve_target_tokens(
        A=calib_A,
        L_inf=calib_L_inf,
        alpha=alpha_data,
        threshold_proxy=threshold_proxy,
    )

    inferred_ratio = None
    confirmation_ratio = None
    if inferred_target_tokens is not None:
        inferred_ratio = inferred_target_tokens / args.n_scaling_params
        confirmation_ratio = inferred_ratio * 1.05

    feasibility_limit = args.unreachable_multiplier * args.default_ratio * args.n_scaling_params
    if inferred_target_tokens is None or inferred_target_tokens > feasibility_limit:
        feasibility_flag = "likely_unreachable"
    else:
        feasibility_flag = "feasible"

    l_inf_bound_active = False
    if l_inf_lower_bound_proxy is not None and calib_L_inf <= l_inf_lower_bound_proxy + 1e-12:
        l_inf_bound_active = True
    if l_inf_upper_bound_proxy is not None and calib_L_inf >= l_inf_upper_bound_proxy - 1e-12:
        l_inf_bound_active = True

    result = {
        "timestamp_utc": utc_timestamp(),
        "dataset_id": stats.get("dataset_id"),
        "tokenizer_id": stats.get("tokenizer_id"),
        "paper_mapping_id": mapping_used,
        "paper_mapping_notes": args.paper_mapping_notes,
        "inferred_data_scaling_exponent": float(alpha_data),
        "alpha_data_plausible": "yes" if alpha_data_plausible else "no",
        "calib_tokens": calib_tokens,
        "calib_metrics": calib_metrics,
        "calib_trend_correlation": float(calib_trend_corr),
        "calib_metric_span": float(calib_metric_span),
        "calib_net_improvement": float(calib_net_improvement),
        "calib_fit_metric_name": "target_metric_proxy",
        "calib_fit_r2": float(calib_fit_r2),
        "calib_A": float(calib_A),
        "calib_L_inf": float(calib_L_inf),
        "calib_L_inf_lower_bound_metric": None if l_inf_lower_bound_metric is None else float(l_inf_lower_bound_metric),
        "calib_L_inf_lower_bound_source": l_inf_lower_bound_source if l_inf_lower_bound_metric is not None else None,
        "calib_L_inf_lower_bound_proxy": None if l_inf_lower_bound_proxy is None else float(l_inf_lower_bound_proxy),
        "calib_L_inf_upper_bound_proxy": None if l_inf_upper_bound_proxy is None else float(l_inf_upper_bound_proxy),
        "calib_L_inf_prior_metric": None if args.l_inf_prior is None else float(args.l_inf_prior),
        "calib_L_inf_prior_proxy": None if l_inf_prior_proxy is None else float(l_inf_prior_proxy),
        "calib_L_inf_prior_weight": float(args.l_inf_prior_weight) if args.l_inf_prior is not None else 0.0,
        "calib_L_inf_bound_active": "yes" if l_inf_bound_active else "no",
        "target_metric_threshold": float(args.target_metric_threshold),
        "target_metric_direction": args.target_metric_direction,
        "target_metric_threshold_proxy": float(threshold_proxy),
        "n_scaling_params": float(args.n_scaling_params),
        "inferred_target_tokens": None if inferred_target_tokens is None else float(inferred_target_tokens),
        "inferred_target_param_data_ratio": None if inferred_ratio is None else float(inferred_ratio),
        "confirmation_ratio": None if confirmation_ratio is None else float(confirmation_ratio),
        "feasibility_limit_tokens": float(feasibility_limit),
        "feasibility_flag": feasibility_flag,
        "solve_status": solve_status,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote Lane B inference to {args.output_json}")
    print(f"alpha_data={alpha_data:.6f}, solve_status={solve_status}, feasibility={feasibility_flag}")


if __name__ == "__main__":
    main()
