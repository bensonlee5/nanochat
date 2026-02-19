"""
Infer Lane B target_param_data_ratio from measured dataset statistics + calibration points.
"""

import argparse
import csv
import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


STEP_VAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s*([0-9.]+)")
TOTAL_BATCH_RE = re.compile(r"Total batch size\s+([0-9,]+)\s+=>")
LOG_SEED_RE = re.compile(r"(?:^|[_-])s(\d+)(?:[_-]|$)")


def parse_csv_floats(raw, arg_name):
    parts = [p.strip() for p in raw.split(",")]
    vals = [float(p) for p in parts if p]
    if not vals:
        raise ValueError(f"{arg_name} must provide at least one numeric value")
    return vals


def parse_seed_list(raw, arg_name):
    if not raw:
        return []
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{arg_name} must contain integer seeds; got {part!r}") from exc
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one seed")
    dedup = []
    seen = set()
    for seed in vals:
        if seed in seen:
            continue
        seen.add(seed)
        dedup.append(seed)
    return dedup


def filter_paths_by_required_seeds(log_paths, required_seeds, source_name):
    if not required_seeds:
        return sorted(set(log_paths))
    required_set = set(required_seeds)
    selected_by_seed = {}
    for path in sorted(set(log_paths)):
        stem = Path(path).stem
        m = LOG_SEED_RE.search(stem)
        if m is None:
            continue
        seed = int(m.group(1))
        if seed in required_set:
            selected_by_seed[seed] = str(path)
    missing = [seed for seed in required_seeds if seed not in selected_by_seed]
    if missing:
        missing_csv = ",".join(str(s) for s in missing)
        raise ValueError(
            f"Missing baseline logs for required seeds ({missing_csv}) in {source_name}"
        )
    return [selected_by_seed[seed] for seed in required_seeds]


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


def _latest_monotonic_step_segment(step_metric_pairs):
    if not step_metric_pairs:
        return []
    start_idx = 0
    prev_step = -1
    for i, (step, _) in enumerate(step_metric_pairs):
        if step < prev_step:
            start_idx = i
        prev_step = step
    return step_metric_pairs[start_idx:]


def extract_val_curve_from_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    m = TOTAL_BATCH_RE.search(text)
    if m is None:
        raise ValueError(f"Missing total batch size in log: {log_path}")
    total_batch_size = int(m.group(1).replace(",", ""))
    if total_batch_size <= 0:
        raise ValueError(f"Invalid total batch size in log: {log_path}")

    raw_pairs = [(int(m.group(1)), float(m.group(2))) for m in STEP_VAL_RE.finditer(text)]
    if not raw_pairs:
        raise ValueError(f"Missing validation curve points in log: {log_path}")

    segment = _latest_monotonic_step_segment(raw_pairs)
    dedup = {}
    for step, metric in segment:
        if step <= 0:
            continue
        dedup[step] = metric
    if len(dedup) < 3:
        raise ValueError(f"Need at least 3 positive-step validation points in log: {log_path}")

    steps = sorted(dedup.keys())
    tokens = [float(step * total_batch_size) for step in steps]
    metrics = [float(dedup[step]) for step in steps]
    return tokens, metrics


def baseline_logs_from_results_csv(
    results_csv, results_dir, require_status_ok=True, required_seeds=None
):
    paths = []
    required_seeds = required_seeds or []
    required_set = set(required_seeds)
    selected_by_seed = {}
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if require_status_ok and status != "ok":
                continue
            seed_raw = (row.get("seed") or "").strip()
            seed = None
            if seed_raw:
                try:
                    seed = int(seed_raw)
                except ValueError:
                    seed = None
            if required_set and seed not in required_set:
                continue
            run_label = (row.get("experiment_name") or "").strip()
            if not run_label:
                continue
            log_path = Path(results_dir) / f"{run_label}_train.log"
            if log_path.exists():
                if required_set and seed is not None:
                    # Keep the last successful row for each required seed.
                    selected_by_seed[seed] = str(log_path)
                else:
                    paths.append(str(log_path))
    if required_set:
        missing = [seed for seed in required_seeds if seed not in selected_by_seed]
        if missing:
            missing_csv = ",".join(str(s) for s in missing)
            raise ValueError(
                "Missing successful baseline rows in results CSV for required seeds: "
                f"{missing_csv}"
            )
        return [selected_by_seed[seed] for seed in required_seeds]
    return sorted(set(paths))


def infer_alpha_from_baseline_logs(
    log_paths,
    threshold_metric,
    metric_direction,
    alpha_min,
    alpha_max,
    alpha_step,
    min_median_r2,
):
    if alpha_step <= 0:
        raise ValueError("baseline alpha grid step must be > 0")
    if alpha_max < alpha_min:
        raise ValueError("baseline alpha grid max must be >= min")

    threshold_proxy = metric_value_to_proxy(threshold_metric, metric_direction)
    series = []
    for path in sorted(set(log_paths)):
        try:
            tokens, metrics = extract_val_curve_from_log(path)
        except ValueError:
            continue
        metrics_proxy = [metric_value_to_proxy(m, metric_direction) for m in metrics]
        series.append((path, tokens, metrics_proxy))
    if not series:
        raise ValueError("No usable baseline log curves for alpha fallback")

    n_steps = int(math.floor((alpha_max - alpha_min) / alpha_step)) + 1
    alpha_grid = [alpha_min + i * alpha_step for i in range(max(1, n_steps))]
    if alpha_grid[-1] < alpha_max - 1e-12:
        alpha_grid.append(alpha_max)

    center = 0.5 * (alpha_min + alpha_max)
    best = None
    for alpha in alpha_grid:
        r2_values = []
        feasible_count = 0
        for _, tokens, metrics_proxy in series:
            A, L_inf, r2 = fit_calibration(tokens, metrics_proxy, alpha)
            _, status = solve_target_tokens(A, L_inf, alpha, threshold_proxy)
            if status == "ok":
                feasible_count += 1
                r2_values.append(float(r2))
        if feasible_count != len(series):
            continue
        median_r2 = float(np.median(r2_values))
        score = (median_r2, -abs(alpha - center), -alpha)
        if best is None or score > best["score"]:
            best = {
                "alpha": float(alpha),
                "median_r2": median_r2,
                "num_seeds": len(series),
                "score": score,
            }

    if best is None:
        raise ValueError(
            "No alpha on baseline grid yielded feasible fits across all baseline logs"
        )
    if best["median_r2"] < min_median_r2:
        raise ValueError(
            "Baseline alpha fallback fit quality too low: "
            f"median_r2={best['median_r2']:.6f} < {min_median_r2:.6f}"
        )

    return best["alpha"], {
        "baseline_alpha_fit_median_r2": best["median_r2"],
        "baseline_alpha_fit_num_seeds": best["num_seeds"],
    }


def calibration_points_from_baseline_logs(log_paths):
    by_tokens = {}
    for path in sorted(set(log_paths)):
        try:
            tokens, metrics = extract_val_curve_from_log(path)
        except ValueError:
            continue
        for tok, metric in zip(tokens, metrics):
            by_tokens.setdefault(float(tok), []).append(float(metric))
    if not by_tokens:
        raise ValueError("No calibration points extracted from baseline logs")

    calib_tokens = sorted(by_tokens.keys())
    calib_metrics = [float(np.median(by_tokens[tok])) for tok in calib_tokens]
    if len(calib_tokens) < 3:
        raise ValueError(
            "Need at least 3 aggregated baseline calibration points; "
            f"only found {len(calib_tokens)}"
        )
    return calib_tokens, calib_metrics


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
    parser.add_argument("--calib-tokens", type=str, default="", help="optional comma-separated token counts, e.g. 300000,600000,900000")
    parser.add_argument("--calib-metrics", type=str, default="", help="optional comma-separated metric values aligned with calib-tokens")
    parser.add_argument("--calib-from-baseline-log-glob", type=str, default="", help="optional baseline log glob used to derive calibration points from validation curves")
    parser.add_argument("--calib-from-baseline-results-csv", type=str, default="", help="optional baseline results.csv used to locate baseline logs for calibration extraction")
    parser.add_argument("--calib-from-baseline-results-dir", type=str, default="", help="optional baseline results dir containing *_train.log files when using --calib-from-baseline-results-csv")
    parser.add_argument("--calib-from-baseline-required-seeds", type=str, default="", help="optional comma-separated baseline seeds required for calibration extraction, e.g. 41,42,43")
    parser.add_argument("--target-metric-threshold", type=float, required=True, help="target threshold in metric space")
    parser.add_argument("--target-metric-direction", type=str, default="lower_is_better", choices=["lower_is_better", "higher_is_better"], help="metric direction")
    parser.add_argument("--l-inf-lower-bound", type=float, default=None, help="optional lower bound for asymptote L_inf in metric space")
    parser.add_argument("--l-inf-lower-bound-from-stats-key", type=str, default="", help="optional stats-json field name to use as L_inf lower bound in metric space (e.g. entropy_h_inf_bits)")
    parser.add_argument("--l-inf-prior", type=float, default=None, help="optional prior target for asymptote L_inf in metric space")
    parser.add_argument("--l-inf-prior-weight", type=float, default=1.0, help="strength of the optional L_inf prior term")
    parser.add_argument("--l-inf-min", type=float, default=None, help="deprecated alias of --l-inf-lower-bound")
    parser.add_argument("--n-scaling-params", type=float, required=True, help="scaling parameter count")
    parser.add_argument("--default-ratio", type=float, default=8.25, help="fallback/default ratio for feasibility guard")
    parser.add_argument("--unreachable-multiplier", type=float, default=10.0, help="multiplier for unreachable-threshold guard")
    parser.add_argument("--alpha-fallback-mode", type=str, default="none", choices=["none", "baseline_assisted"], help="when stats alpha is weak/implausible, optionally derive alpha from baseline log curves")
    parser.add_argument("--baseline-log-glob", type=str, default="", help="optional baseline log glob used for alpha fallback")
    parser.add_argument("--baseline-results-csv", type=str, default="", help="optional baseline results.csv used to locate baseline logs for alpha fallback")
    parser.add_argument("--baseline-results-dir", type=str, default="", help="optional baseline results dir containing *_train.log files when using --baseline-results-csv")
    parser.add_argument("--baseline-required-seeds", type=str, default="", help="optional comma-separated baseline seeds required for alpha fallback, e.g. 41,42,43")
    parser.add_argument("--baseline-alpha-grid-min", type=float, default=0.1, help="minimum alpha value searched in baseline-assisted fallback")
    parser.add_argument("--baseline-alpha-grid-max", type=float, default=0.8, help="maximum alpha value searched in baseline-assisted fallback")
    parser.add_argument("--baseline-alpha-grid-step", type=float, default=0.01, help="alpha grid step for baseline-assisted fallback")
    parser.add_argument("--baseline-alpha-min-r2", type=float, default=0.97, help="minimum median R^2 required to accept baseline-assisted alpha fallback")
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

    baseline_required_seeds = parse_seed_list(
        args.baseline_required_seeds, "--baseline-required-seeds"
    )
    calib_required_seeds = parse_seed_list(
        args.calib_from_baseline_required_seeds,
        "--calib-from-baseline-required-seeds",
    )

    def collect_baseline_log_paths(results_csv, results_dir, log_glob, label, required_seeds):
        paths = []
        if results_csv:
            if not results_dir:
                raise ValueError(
                    f"--{label}-results-dir is required when --{label}-results-csv is set"
                )
            paths.extend(
                baseline_logs_from_results_csv(
                    results_csv=results_csv,
                    results_dir=results_dir,
                    require_status_ok=True,
                    required_seeds=required_seeds,
                )
            )
        if log_glob:
            paths.extend(glob.glob(log_glob))
        return filter_paths_by_required_seeds(
            paths, required_seeds, source_name=f"{label} log discovery"
        )

    alpha_data_stats_raw, mapping_used = infer_alpha(
        stats=stats,
        paper_mapping_id=args.paper_mapping_id,
        corr_weight=args.corr_weight,
        entropy_weight=args.entropy_weight,
        alpha_override=args.alpha_override,
    )
    alpha_data = float(alpha_data_stats_raw)
    alpha_source = "stats"
    alpha_data_baseline_fallback = None
    alpha_fallback_reason = None
    baseline_alpha_fit_median_r2 = None
    baseline_alpha_fit_num_seeds = 0

    corr_r2_low_quality = str(stats.get("corr_decay_r2_low_quality", "no")).lower() == "yes"
    ent_r2_low_quality = str(stats.get("entropy_decay_r2_low_quality", "no")).lower() == "yes"
    if "corr_decay_r2_low_quality" not in stats:
        if "corr_decay_r2" in stats and "corr_decay_r2_warn_threshold" in stats:
            corr_r2_low_quality = float(stats["corr_decay_r2"]) < float(
                stats["corr_decay_r2_warn_threshold"]
            )
    if "entropy_decay_r2_low_quality" not in stats:
        if "entropy_decay_r2" in stats and "entropy_decay_r2_warn_threshold" in stats:
            ent_r2_low_quality = float(stats["entropy_decay_r2"]) < float(
                stats["entropy_decay_r2_warn_threshold"]
            )
    stats_fit_low_quality = corr_r2_low_quality or ent_r2_low_quality
    alpha_data_plausible_stats = args.alpha_min <= alpha_data <= args.alpha_max

    if args.alpha_fallback_mode == "baseline_assisted":
        fallback_needed = (args.alpha_override is None) and (
            (not alpha_data_plausible_stats) or stats_fit_low_quality
        )
        if fallback_needed:
            fallback_log_paths = collect_baseline_log_paths(
                results_csv=args.baseline_results_csv,
                results_dir=args.baseline_results_dir,
                log_glob=args.baseline_log_glob,
                label="baseline",
                required_seeds=baseline_required_seeds,
            )
            if not fallback_log_paths:
                raise ValueError(
                    "alpha fallback requested but no baseline logs were found. "
                    "Provide --baseline-log-glob or --baseline-results-csv/--baseline-results-dir."
                )

            alpha_data, fallback_meta = infer_alpha_from_baseline_logs(
                log_paths=fallback_log_paths,
                threshold_metric=args.target_metric_threshold,
                metric_direction=args.target_metric_direction,
                alpha_min=args.baseline_alpha_grid_min,
                alpha_max=args.baseline_alpha_grid_max,
                alpha_step=args.baseline_alpha_grid_step,
                min_median_r2=args.baseline_alpha_min_r2,
            )
            alpha_source = "baseline_assisted"
            alpha_data_baseline_fallback = float(alpha_data)
            baseline_alpha_fit_median_r2 = float(
                fallback_meta["baseline_alpha_fit_median_r2"]
            )
            baseline_alpha_fit_num_seeds = int(
                fallback_meta["baseline_alpha_fit_num_seeds"]
            )
            reasons = []
            if not alpha_data_plausible_stats:
                reasons.append("stats_alpha_implausible")
            if stats_fit_low_quality:
                reasons.append("stats_fit_low_quality")
            alpha_fallback_reason = ",".join(reasons) if reasons else "fallback_requested"
            mapping_used = "baseline_assisted"

    alpha_data_plausible = args.alpha_min <= alpha_data <= args.alpha_max
    if alpha_source == "baseline_assisted":
        # Baseline-assisted alpha already passed explicit fit-quality and feasibility checks.
        alpha_data_plausible = True

    calib_tokens = []
    calib_metrics = []
    if args.calib_tokens or args.calib_metrics:
        if not args.calib_tokens or not args.calib_metrics:
            raise ValueError(
                "Provide both --calib-tokens and --calib-metrics, or provide neither."
            )
        calib_tokens = parse_csv_floats(args.calib_tokens, "--calib-tokens")
        calib_metrics = parse_csv_floats(args.calib_metrics, "--calib-metrics")
        if len(calib_tokens) != len(calib_metrics):
            raise ValueError("--calib-tokens and --calib-metrics must have identical lengths")
    else:
        calib_log_paths = collect_baseline_log_paths(
            results_csv=args.calib_from_baseline_results_csv,
            results_dir=args.calib_from_baseline_results_dir,
            log_glob=args.calib_from_baseline_log_glob,
            label="calib-from-baseline",
            required_seeds=calib_required_seeds,
        )
        if not calib_log_paths:
            raise ValueError(
                "No explicit calibration points provided and no baseline logs found for calibration extraction. "
                "Provide --calib-tokens/--calib-metrics or --calib-from-baseline-log-glob "
                "or --calib-from-baseline-results-csv/--calib-from-baseline-results-dir."
            )
        calib_tokens, calib_metrics = calibration_points_from_baseline_logs(calib_log_paths)

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

    l_inf_lower_bound_disabled_reason = None
    if (
        args.target_metric_direction == "lower_is_better"
        and l_inf_lower_bound_proxy is not None
        and l_inf_lower_bound_proxy >= threshold_proxy
    ):
        l_inf_lower_bound_disabled_reason = "lower_bound_not_below_threshold"
        l_inf_lower_bound_metric = None
        l_inf_lower_bound_proxy = None
        l_inf_lower_bound_source = None

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
        "alpha_source": alpha_source,
        "alpha_data_stats_raw": float(alpha_data_stats_raw),
        "alpha_data_stats_raw_plausible": "yes" if alpha_data_plausible_stats else "no",
        "alpha_data_baseline_fallback": None if alpha_data_baseline_fallback is None else float(alpha_data_baseline_fallback),
        "alpha_fallback_reason": alpha_fallback_reason,
        "baseline_alpha_fit_median_r2": None if baseline_alpha_fit_median_r2 is None else float(baseline_alpha_fit_median_r2),
        "baseline_alpha_fit_num_seeds": int(baseline_alpha_fit_num_seeds),
        "baseline_required_seeds": baseline_required_seeds,
        "calib_from_baseline_required_seeds": calib_required_seeds,
        "stats_fit_low_quality": "yes" if stats_fit_low_quality else "no",
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
        "calib_L_inf_lower_bound_disabled_reason": l_inf_lower_bound_disabled_reason,
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
    print(
        f"alpha_data={alpha_data:.6f} (source={alpha_source}), "
        f"solve_status={solve_status}, feasibility={feasibility_flag}"
    )


if __name__ == "__main__":
    main()
