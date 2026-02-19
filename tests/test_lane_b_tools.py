import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_baseline_log(path, steps, metrics, total_batch_size=16384):
    lines = [f"Total batch size {total_batch_size:,} => gradient accumulation steps: 1"]
    for step, metric in zip(steps, metrics):
        lines.append(f"Step {int(step):05d} | Validation bpb: {float(metric):.6f}")
    lines.append(f"Minimum validation bpb: {float(min(metrics)):.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


lane_b_infer_ratio = _load_module("scripts/lane_b_infer_ratio.py", "lane_b_infer_ratio")
lane_b_log_run = _load_module("scripts/lane_b_log_run.py", "lane_b_log_run")
lane_b_stats = _load_module("scripts/lane_b_stats.py", "lane_b_stats")
lane_b_extract_calibration = _load_module("scripts/lane_b_extract_calibration.py", "lane_b_extract_calibration")
lane_b_get_scaling_params = _load_module("scripts/lane_b_get_scaling_params.py", "lane_b_get_scaling_params")


def test_fit_power_law_recovers_exponent():
    xs = [1, 2, 4, 8, 16]
    ys = [3.0 * (x ** -0.7) for x in xs]
    _, exponent, r2 = lane_b_stats.fit_power_law(xs, ys)
    assert abs(exponent - 0.7) < 0.05
    assert r2 > 0.99


def test_fit_power_law_is_deterministic_for_fixed_seed():
    xs = [1, 2, 4, 8, 16]
    ys = [3.0 * (x ** -0.7) for x in xs]
    _, exp_1, _ = lane_b_stats.fit_power_law(xs, ys, optuna_seed=123, optuna_trials=64)
    _, exp_2, _ = lane_b_stats.fit_power_law(xs, ys, optuna_seed=123, optuna_trials=64)
    assert abs(exp_1 - exp_2) < 1e-12


def test_fit_shifted_power_law_recovers_asymptote():
    xs = [1, 2, 4, 8, 16, 32, 64]
    ys = [2.5 * (x ** -0.55) + 0.7 for x in xs]
    _, exponent, offset, r2 = lane_b_stats.fit_shifted_power_law(xs, ys, optuna_seed=7, optuna_trials=200)
    assert abs(exponent - 0.55) < 0.08
    assert abs(offset - 0.7) < 0.12
    assert r2 > 0.99


def test_build_context_lengths_is_sorted_and_includes_bounds():
    ctx = lane_b_stats.build_context_lengths(1, 64, 8)
    assert ctx[0] == 1
    assert ctx[-1] == 64
    assert ctx == sorted(ctx)


def test_entropy_curve_adaptive_shrinks_window_when_needed():
    rng = np.random.default_rng(123)
    tokens = rng.integers(0, 8, size=20000, dtype=np.int64)
    entropy_curve, skipped_data_starved, points_requested, adaptive_reductions = lane_b_stats.build_entropy_curve_adaptive(
        tokens=tokens,
        min_ctx=1,
        max_ctx=64,
        num_points=8,
        min_usable_points=4,
        uniqueness_threshold=0.9,
    )
    assert points_requested >= 4
    assert len(entropy_curve) >= 4
    assert adaptive_reductions >= 1
    assert skipped_data_starved >= 0


def test_compute_correlation_curve_matches_naive_reference():
    tokens = np.asarray([0, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 2], dtype=np.int64)
    min_sep, max_sep = 1, 4

    curve, sum_p_sq = lane_b_stats.compute_correlation_curve(tokens, min_sep, max_sep)

    counts = np.bincount(tokens)
    probs = counts.astype(np.float64) / tokens.size
    sum_p_sq_ref = float(np.dot(probs, probs))
    eps = 1e-12
    curve_ref = []
    for sep in range(min_sep, max_sep + 1):
        n_pairs = tokens.size - sep
        joint = {}
        for mu, nu in zip(tokens[:-sep], tokens[sep:]):
            key = (int(mu), int(nu))
            joint[key] = joint.get(key, 0) + 1

        sum_pjoint_sq = 0.0
        cross_term = 0.0
        for (mu, nu), c in joint.items():
            p_joint = c / n_pairs
            sum_pjoint_sq += p_joint * p_joint
            cross_term += p_joint * probs[mu] * probs[nu]
        frob_sq = sum_pjoint_sq - 2 * cross_term + sum_p_sq_ref * sum_p_sq_ref
        curve_ref.append(
            {
                "separation": sep,
                "frobenius_norm": max(math.sqrt(max(frob_sq, 0.0)), eps),
            }
        )

    assert abs(sum_p_sq - sum_p_sq_ref) < 1e-15
    assert len(curve) == len(curve_ref)
    for got, expected in zip(curve, curve_ref):
        assert got["separation"] == expected["separation"]
        assert abs(got["frobenius_norm"] - expected["frobenius_norm"]) < 1e-12


def test_infer_alpha_from_baseline_logs_recovers_expected_alpha(tmp_path):
    steps = np.asarray([100, 200, 400, 800, 1200], dtype=np.int64)
    total_batch_size = 16384.0
    tokens = steps.astype(np.float64) * total_batch_size
    alpha_true = 0.45
    l_inf = 0.80
    a1 = 70.0
    a2 = 71.4
    metrics_1 = l_inf + a1 * np.power(tokens, -alpha_true)
    metrics_2 = l_inf + a2 * np.power(tokens, -alpha_true)

    log_1 = tmp_path / "baseline_s41.log"
    log_2 = tmp_path / "baseline_s42.log"
    _write_baseline_log(log_1, steps, metrics_1, total_batch_size=int(total_batch_size))
    _write_baseline_log(log_2, steps, metrics_2, total_batch_size=int(total_batch_size))

    alpha, meta = lane_b_infer_ratio.infer_alpha_from_baseline_logs(
        log_paths=[str(log_1), str(log_2)],
        threshold_metric=0.86,
        metric_direction="lower_is_better",
        alpha_min=0.2,
        alpha_max=0.8,
        alpha_step=0.01,
        min_median_r2=0.95,
    )
    assert abs(alpha - alpha_true) < 0.08
    assert meta["baseline_alpha_fit_num_seeds"] == 2
    assert meta["baseline_alpha_fit_median_r2"] > 0.99


def test_calibration_points_from_baseline_logs_uses_median_curve(tmp_path):
    steps = np.asarray([100, 200, 400, 800], dtype=np.int64)
    total_batch_size = 16384.0
    tokens = steps.astype(np.float64) * total_batch_size
    alpha = 0.4
    l_inf = 0.81
    a1 = 50.0
    a2 = 55.0
    metrics_1 = l_inf + a1 * np.power(tokens, -alpha)
    metrics_2 = l_inf + a2 * np.power(tokens, -alpha)

    log_1 = tmp_path / "baseline_s41.log"
    log_2 = tmp_path / "baseline_s42.log"
    _write_baseline_log(log_1, steps, metrics_1, total_batch_size=int(total_batch_size))
    _write_baseline_log(log_2, steps, metrics_2, total_batch_size=int(total_batch_size))

    calib_tokens, calib_metrics = lane_b_infer_ratio.calibration_points_from_baseline_logs(
        [str(log_1), str(log_2)]
    )
    assert calib_tokens == sorted(calib_tokens)
    assert len(calib_tokens) == len(steps)
    expected_first = float(np.median([metrics_1[0], metrics_2[0]]))
    assert abs(calib_metrics[0] - expected_first) < 1e-6


def test_infer_ratio_baseline_logs_from_results_csv_filters_required_seeds(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed in (41, 42, 43, 99):
        (results_dir / f"baseline_d12_s{seed}_train.log").write_text("ok", encoding="utf-8")

    csv_path = results_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "experiment_name", "status"])
        writer.writeheader()
        writer.writerows(
            [
                {"seed": "41", "experiment_name": "baseline_d12_s41", "status": "ok"},
                {"seed": "42", "experiment_name": "baseline_d12_s42", "status": "ok"},
                {"seed": "43", "experiment_name": "baseline_d12_s43", "status": "ok"},
                {"seed": "99", "experiment_name": "baseline_d12_s99", "status": "ok"},
            ]
        )

    paths = lane_b_infer_ratio.baseline_logs_from_results_csv(
        str(csv_path),
        str(results_dir),
        required_seeds=[43, 41, 42],
    )
    assert [Path(p).name for p in paths] == [
        "baseline_d12_s43_train.log",
        "baseline_d12_s41_train.log",
        "baseline_d12_s42_train.log",
    ]


def test_extract_calibration_results_csv_enforces_required_seeds(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for seed in (41, 42):
        (results_dir / f"baseline_d12_s{seed}_train.log").write_text("ok", encoding="utf-8")

    csv_path = results_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "experiment_name", "status"])
        writer.writeheader()
        writer.writerows(
            [
                {"seed": "41", "experiment_name": "baseline_d12_s41", "status": "ok"},
                {"seed": "42", "experiment_name": "baseline_d12_s42", "status": "ok"},
            ]
        )

    try:
        lane_b_extract_calibration.baseline_log_paths_from_results_csv(
            str(csv_path),
            str(results_dir),
            required_seeds=[41, 42, 43],
        )
        assert False, "expected missing required seed to fail"
    except ValueError as exc:
        assert "43" in str(exc)


def test_infer_ratio_reachable_and_unreachable_paths():
    stats = {"corr_decay_exponent": 0.4, "entropy_decay_exponent": 0.6}
    alpha, mapping = lane_b_infer_ratio.infer_alpha(
        stats=stats,
        paper_mapping_id="weighted_mean_v1",
        corr_weight=1.0,
        entropy_weight=1.0,
        alpha_override=None,
    )
    assert mapping == "weighted_mean_v1"
    assert abs(alpha - 0.5) < 1e-9

    calib_tokens = [1_000_000, 2_000_000, 4_000_000]
    calib_metrics = [1.0, 0.93, 0.89]
    A, L_inf, _ = lane_b_infer_ratio.fit_calibration(calib_tokens, calib_metrics, alpha)
    assert A > 0
    assert math.isfinite(L_inf)

    d_target, status = lane_b_infer_ratio.solve_target_tokens(A, L_inf, alpha, threshold_proxy=0.86)
    assert status == "ok"
    assert d_target is not None and d_target > 0

    d_target_bad, status_bad = lane_b_infer_ratio.solve_target_tokens(A, L_inf, alpha, threshold_proxy=L_inf)
    assert d_target_bad is None
    assert status_bad == "threshold_not_above_asymptote"


def test_fit_calibration_respects_linf_lower_bound():
    calib_tokens = [1_000_000, 2_000_000, 4_000_000]
    calib_metrics = [1.0, 0.93, 0.89]
    alpha = 0.5

    _, l_inf_unconstrained, _ = lane_b_infer_ratio.fit_calibration(calib_tokens, calib_metrics, alpha)
    _, l_inf_bounded, _ = lane_b_infer_ratio.fit_calibration(
        calib_tokens,
        calib_metrics,
        alpha,
        l_inf_lower_bound=0.90,
    )

    assert l_inf_unconstrained < 0.90
    assert abs(l_inf_bounded - 0.90) < 1e-9


def test_fit_calibration_prior_pulls_linf_toward_prior():
    calib_tokens = [1_000_000, 2_000_000, 4_000_000]
    calib_metrics = [1.0, 0.93, 0.89]
    alpha = 0.5

    _, l_inf_unconstrained, _ = lane_b_infer_ratio.fit_calibration(calib_tokens, calib_metrics, alpha)
    _, l_inf_prior, _ = lane_b_infer_ratio.fit_calibration(
        calib_tokens,
        calib_metrics,
        alpha,
        l_inf_prior=0.95,
        l_inf_prior_weight=100.0,
    )

    assert abs(l_inf_prior - 0.95) < abs(l_inf_unconstrained - 0.95)


def test_validate_calibration_points_sorts_and_enforces_direction():
    tokens = [4_000_000, 1_000_000, 2_000_000]
    metrics = [0.89, 1.0, 0.93]
    sorted_tokens, sorted_metrics, trend_corr, metric_span, net_improvement = lane_b_infer_ratio.validate_calibration_points(
        tokens=tokens,
        metrics=metrics,
        metric_direction="lower_is_better",
    )
    assert sorted_tokens == [1_000_000.0, 2_000_000.0, 4_000_000.0]
    assert sorted_metrics == [1.0, 0.93, 0.89]
    assert trend_corr < 0
    assert metric_span > 0
    assert net_improvement > 0


def test_validate_calibration_points_rejects_non_improving_series():
    try:
        lane_b_infer_ratio.validate_calibration_points(
            tokens=[1_000_000, 2_000_000, 4_000_000],
            metrics=[0.90, 0.91, 0.92],
            metric_direction="lower_is_better",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "non-improving" in str(e)


def test_log_schema_creation_and_metric_parsing(tmp_path):
    schema_path = tmp_path / "lane_b_schema.csv"
    lane_b_log_run.ensure_schema(schema_path)

    with schema_path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == lane_b_log_run.EXPECTED_COLUMNS

    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "tok/sec: 50,000 dt: 20.0ms",
                "tok/sec: 60,000 dt: 25.0ms",
                "Minimum validation bpb: 0.8123",
                "Time-to-target (val_bpb): 12.50m",
                "Extrapolated time-to-target (val_bpb, linear_recent_eval): 15.00m",
            ]
        ),
        encoding="utf-8",
    )
    parsed = lane_b_log_run.parse_metrics_from_log(log_path)
    assert abs(parsed["tok_per_sec_median"] - 55000) < 1e-9
    assert abs(parsed["dt_median_ms"] - 22.5) < 1e-9
    assert abs(parsed["min_val_bpb"] - 0.8123) < 1e-9
    assert abs(parsed["time_to_target_sec_measured"] - 750.0) < 1e-9
    assert abs(parsed["time_to_target_sec_extrapolated"] - 900.0) < 1e-9


def test_log_metric_parsing_uses_latest_run_segment_after_restart(tmp_path):
    log_path = tmp_path / "train_restart.log"
    log_path.write_text(
        "\n".join(
            [
                "step 00010/01500 (0.67%) | loss: 3.000000 | lrm: 1.00 | dt: 50.0ms | tok/sec: 10,000 | bf16_mfu: 10.00 | epoch: 0 | total time: 1.00m",
                "Minimum validation bpb: 0.9500",
                "Time-to-target (val_bpb): 30.00m",
                "step 00000/01500 (0.00%) | loss: 2.000000 | lrm: 1.00 | dt: 20.0ms | tok/sec: 60,000 | bf16_mfu: 20.00 | epoch: 0 | total time: 0.00m",
                "step 00001/01500 (0.07%) | loss: 1.900000 | lrm: 1.00 | dt: 22.0ms | tok/sec: 70,000 | bf16_mfu: 21.00 | epoch: 0 | total time: 0.01m",
                "Minimum validation bpb: 0.8123",
                "Time-to-target (val_bpb): 12.50m",
                "Extrapolated time-to-target (val_bpb, linear_recent_eval): 15.00m",
            ]
        ),
        encoding="utf-8",
    )
    parsed = lane_b_log_run.parse_metrics_from_log(log_path)
    assert abs(parsed["tok_per_sec_median"] - 65000) < 1e-9
    assert abs(parsed["dt_median_ms"] - 21.0) < 1e-9
    assert abs(parsed["min_val_bpb"] - 0.8123) < 1e-9
    assert abs(parsed["time_to_target_sec_measured"] - 750.0) < 1e-9
    assert abs(parsed["time_to_target_sec_extrapolated"] - 900.0) < 1e-9


def test_extract_calibration_metric(tmp_path):
    log_path = tmp_path / "calib.log"
    log_path.write_text(
        "\n".join(
            [
                "Step 00100 | Validation bpb: 0.900000",
                "Minimum validation bpb: 0.8123",
                "Minimum validation bpb: 0.7999",
            ]
        ),
        encoding="utf-8",
    )
    value = lane_b_extract_calibration.extract_min_val_bpb(log_path)
    assert abs(value - 0.7999) < 1e-9


def test_extract_calibration_uses_latest_segment_after_restart(tmp_path):
    log_path = tmp_path / "calib_restart.log"
    log_path.write_text(
        "\n".join(
            [
                "Step 00100 | Validation bpb: 2.200000",
                "Step 00200 | Validation bpb: 2.100000",
                "Minimum validation bpb: 2.100000",
                "Step 00000 | Validation bpb: 3.100000",
                "Step 00100 | Validation bpb: 2.000000",
                "Step 00200 | Validation bpb: 1.950000",
                "Minimum validation bpb: 1.950000",
            ]
        ),
        encoding="utf-8",
    )
    value = lane_b_extract_calibration.extract_min_val_bpb(log_path)
    assert abs(value - 1.95) < 1e-9


def test_extract_calibration_fails_on_missing_summary(tmp_path):
    log_path = tmp_path / "calib_incomplete.log"
    log_path.write_text(
        "\n".join(
            [
                "Step 00100 | Validation bpb: 0.900000",
                "Step 00200 | Validation bpb: 0.800000",
            ]
        ),
        encoding="utf-8",
    )
    try:
        lane_b_extract_calibration.extract_min_val_bpb(log_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "incomplete" in str(e)


def test_compute_scaling_params_accounts_for_gqa():
    mha = lane_b_get_scaling_params.compute_scaling_params(
        depth=4,
        aspect_ratio=64,
        head_dim=128,
        n_kv_head=None,
        max_seq_len=128,
        window_pattern="L",
    )
    gqa = lane_b_get_scaling_params.compute_scaling_params(
        depth=4,
        aspect_ratio=64,
        head_dim=128,
        n_kv_head=1,
        max_seq_len=128,
        window_pattern="L",
    )

    assert mha["n_head"] == 2
    assert mha["n_kv_head"] == mha["n_head"]
    assert gqa["n_kv_head"] == 1
    assert gqa["n_scaling_params"] < mha["n_scaling_params"]
    assert gqa["param_counts"]["transformer_matrices"] < mha["param_counts"]["transformer_matrices"]


def test_step1_scaling_script_wires_optional_n_kv_head():
    content = (ROOT / "runs/lane_b_step1_get_scaling_params.sh").read_text(encoding="utf-8")
    assert "N_KV_HEAD_ARGS=()" in content
    assert "--n-kv-head" in content
    assert '"${N_KV_HEAD_ARGS[@]}"' in content


def test_step3_inference_script_has_low_quality_hard_stop_gate():
    content = (ROOT / "runs/lane_b_step3_infer_ratio.sh").read_text(encoding="utf-8")
    assert "LANE_B_ALLOW_LOW_QUALITY_STATS" in content
    assert "Lane B stats fit-quality gate failed." in content
    assert "Stopping step 3 due to stats fit-quality gate." in content


def test_step3_inference_script_uses_baseline_calibration_and_alpha_fallback():
    content = (ROOT / "runs/lane_b_step3_infer_ratio.sh").read_text(encoding="utf-8")
    assert "--calib-from-baseline-results-csv" in content
    assert "--calib-from-baseline-results-dir" in content
    assert "--calib-from-baseline-required-seeds" in content
    assert "--alpha-fallback-mode" in content
    assert "--baseline-required-seeds" in content


def test_run_all_references_renumbered_steps():
    content = (ROOT / "runs/lane_b_run_all.sh").read_text(encoding="utf-8")
    assert "lane_b_step3_infer_ratio.sh" in content
    assert "lane_b_step4_run_candidates.sh" in content
    assert "lane_b_step5_log_candidates.sh" in content
    assert "lane_b_step6_summary.sh" in content


def test_lane_b_common_exposes_new_defaults():
    content = (ROOT / "runs/lane_b_common.sh").read_text(encoding="utf-8")
    assert 'LANE_B_SPEEDRUN_DEFAULT_RATIO="${LANE_B_SPEEDRUN_DEFAULT_RATIO:-$speedrun_ratio_default}"' in content
    assert 'LANE_B_N_KV_HEAD="${LANE_B_N_KV_HEAD:-}"' in content
    assert 'LANE_B_ALLOW_LOW_QUALITY_STATS="${LANE_B_ALLOW_LOW_QUALITY_STATS:-0}"' in content
    assert 'LANE_B_ENT_MIN_USABLE_POINTS="${LANE_B_ENT_MIN_USABLE_POINTS:-4}"' in content
    assert 'LANE_B_ALPHA_FALLBACK_MODE="${LANE_B_ALPHA_FALLBACK_MODE:-baseline_assisted}"' in content


def test_baseline_common_uses_speedrun_ratio_default():
    content = (ROOT / "runs/baseline_common.sh").read_text(encoding="utf-8")
    assert 'BASELINE_SPEEDRUN_DEFAULT_RATIO="${BASELINE_SPEEDRUN_DEFAULT_RATIO:-$speedrun_ratio_default}"' in content
    assert 'BASELINE_DEFAULT_RATIO="${BASELINE_DEFAULT_RATIO:-$BASELINE_SPEEDRUN_DEFAULT_RATIO}"' in content
