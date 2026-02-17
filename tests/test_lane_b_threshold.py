import csv
import importlib.util
from pathlib import Path

import pytest


def _load_module(relpath, name):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline_threshold = _load_module(
    "scripts/baseline_compute_threshold.py", "baseline_compute_threshold"
)


HEADER = [
    "timestamp",
    "git_commit",
    "experiment_name",
    "seed",
    "depth",
    "max_seq_len",
    "device_batch_size",
    "total_batch_size",
    "target_param_data_ratio",
    "warmdown_ratio",
    "warmdown_shape",
    "target_metric_name",
    "target_metric_threshold",
    "time_to_target_sec_measured",
    "time_to_target_sec_extrapolated",
    "extrapolation_method",
    "tok_per_sec_median",
    "dt_median_ms",
    "wall_time_sec",
    "final_train_loss",
    "min_val_bpb",
    "peak_mem_mib",
    "status",
]


def _write_results_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def test_threshold_uses_median_of_required_seed_runs(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [
            ["t", "g", "baseline_d12_s41", "41", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2.030000", "", "ok"],
            ["t", "g", "baseline_d12_s42", "42", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "1.990000", "", "ok"],
            ["t", "g", "baseline_d12_s43", "43", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2.010000", "", "ok"],
        ],
    )

    rows = baseline_threshold.collect_successful_rows(csv_path, ["41", "42", "43"])
    min_vals = [r["min_val_bpb"] for r in rows]
    median_min, threshold = baseline_threshold.compute_threshold(min_vals, 0.02)

    assert abs(median_min - 2.01) < 1e-9
    assert abs(threshold - 2.03) < 1e-9


def test_collect_successful_rows_rejects_missing_required_seeds(tmp_path):
    csv_path = tmp_path / "results.csv"
    _write_results_csv(
        csv_path,
        [
            ["t", "g", "baseline_d12_s41", "41", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2.010000", "", "ok"],
            ["t", "g", "baseline_d12_s42", "42", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2.020000", "", "failed"],
            ["t", "g", "baseline_d12_s43", "43", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "ok"],
        ],
    )

    with pytest.raises(
        ValueError, match="Missing successful baseline rows for required seeds: 42,43"
    ):
        baseline_threshold.collect_successful_rows(csv_path, ["41", "42", "43"])


def test_compute_threshold_rejects_empty_min_values():
    with pytest.raises(ValueError, match="No successful baseline min_val_bpb values found"):
        baseline_threshold.compute_threshold([], 0.02)


def test_write_env_file_uses_custom_export_names(tmp_path):
    env_path = tmp_path / "threshold.env"
    baseline_threshold.write_env_file(
        output_env=env_path,
        recommended_threshold=2.04,
        median_min=2.02,
        successful_runs_count=3,
        export_threshold_var="LANE_X_TARGET_THRESHOLD",
        export_prefix="LANE_X_BASELINE",
    )
    text = env_path.read_text(encoding="utf-8")
    assert "export LANE_X_TARGET_THRESHOLD=2.040000" in text
    assert "export LANE_X_BASELINE_MEDIAN_MIN_VAL_BPB=2.020000" in text
    assert "export LANE_X_BASELINE_SUCCESSFUL_RUNS=3" in text
