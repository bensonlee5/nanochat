import csv
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_fallback_script_uses_same_extrapolation_arg_family():
    content = (ROOT / "runs/lane_b_step6_run_fallback_baseline.sh").read_text(encoding="utf-8")
    assert "--time-to-target-extrapolation" in content
    assert "--time-to-target-power-law-alpha" in content
    assert "--time-to-target-power-law-fit-r2-min" in content
    assert '"${EXTRAPOLATION_ARGS[@]}"' in content


def test_step5_inference_script_has_low_quality_hard_stop_gate():
    content = (ROOT / "runs/lane_b_step5_infer_ratio.sh").read_text(encoding="utf-8")
    assert "LANE_B_ALLOW_LOW_QUALITY_STATS" in content
    assert "Lane B stats fit-quality gate failed." in content
    assert "Stopping step 5 due to stats fit-quality gate." in content


def test_lane_b_common_exposes_new_defaults():
    content = (ROOT / "runs/lane_b_common.sh").read_text(encoding="utf-8")
    assert 'LANE_B_N_KV_HEAD="${LANE_B_N_KV_HEAD:-}"' in content
    assert 'LANE_B_ALLOW_LOW_QUALITY_STATS="${LANE_B_ALLOW_LOW_QUALITY_STATS:-0}"' in content
