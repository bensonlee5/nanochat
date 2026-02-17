"""
Append one Lane B run row to ideas/lane_b_inference_schema.csv.
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COLUMNS = [
    "timestamp_utc",
    "git_commit",
    "run_id",
    "seed",
    "dataset_id",
    "target_metric_name",
    "target_metric_threshold",
    "n_scaling_params",
    "inferred_data_scaling_exponent",
    "inferred_target_tokens",
    "inferred_target_param_data_ratio",
    "confirmation_ratio",
    "chosen_ratio",
    "lane_b_decision",
    "time_to_target_sec_measured",
    "time_to_target_sec_extrapolated",
    "min_val_bpb",
    "tok_per_sec_median",
    "dt_median_ms",
    "status",
    "notes",
]


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def last_float(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def parse_metrics_from_log(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    tok_vals = [int(x.replace(",", "")) for x in re.findall(r"tok/sec:\s*([\d,]+)", text)]
    dt_vals = [float(x) for x in re.findall(r"dt:\s*([0-9.]+)ms", text)]

    measured_min = last_float(r"Time-to-target \([^)]+\):\s*([0-9.]+)m", text)
    measured_sec = measured_min * 60.0 if measured_min is not None else None

    extrap_matches = re.findall(
        r"Extrapolated time-to-target \([^)]+,\s*([^)]+)\):\s*([0-9.]+)m",
        text,
    )
    if extrap_matches:
        extrap_sec = float(extrap_matches[-1][1]) * 60.0
    else:
        extrap_sec = None

    return {
        "time_to_target_sec_measured": measured_sec,
        "time_to_target_sec_extrapolated": extrap_sec,
        "min_val_bpb": last_float(r"Minimum validation bpb:\s*([0-9.]+)", text),
        "tok_per_sec_median": statistics.median(tok_vals) if tok_vals else None,
        "dt_median_ms": statistics.median(dt_vals) if dt_vals else None,
    }


def normalize_scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def ensure_schema(schema_path):
    path = Path(schema_path)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header is None:
            raise ValueError(f"Schema file {schema_path} is empty")
        if header != EXPECTED_COLUMNS:
            raise ValueError(
                f"Schema header mismatch in {schema_path}.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {header}"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Append a Lane B run row to the schema CSV.")
    parser.add_argument("--schema-path", type=str, default="ideas/lane_b_inference_schema.csv", help="path to Lane B schema CSV")
    parser.add_argument("--log-path", type=str, default="", help="optional base_train log to parse metrics from")
    parser.add_argument("--timestamp-utc", type=str, default="", help="timestamp override (default now UTC)")
    parser.add_argument("--git-commit", type=str, default="", help="git commit override (default short HEAD)")

    parser.add_argument("--run-id", type=str, required=True, help="run identifier")
    parser.add_argument("--seed", type=int, required=True, help="run seed")
    parser.add_argument("--dataset-id", type=str, required=True, help="dataset identifier")
    parser.add_argument("--target-metric-name", type=str, required=True, help="target metric name")
    parser.add_argument("--target-metric-threshold", type=float, required=True, help="target metric threshold")
    parser.add_argument("--n-scaling-params", type=float, required=True, help="scaling parameter count")
    parser.add_argument("--inferred-data-scaling-exponent", type=float, required=True, help="inferred alpha_data")
    parser.add_argument("--inferred-target-tokens", type=float, required=True, help="inferred target token budget")
    parser.add_argument("--inferred-target-param-data-ratio", type=float, required=True, help="inferred target param:data ratio")
    parser.add_argument("--confirmation-ratio", type=float, required=True, help="confirmation ratio (usually inferred*1.05)")
    parser.add_argument("--chosen-ratio", type=float, required=True, help="chosen ratio for this run")
    parser.add_argument("--lane-b-decision", type=str, required=True, choices=["inferred", "confirmation", "fallback"], help="Lane B decision label")

    parser.add_argument("--time-to-target-sec-measured", type=float, default=None, help="measured time-to-target seconds")
    parser.add_argument("--time-to-target-sec-extrapolated", type=float, default=None, help="extrapolated time-to-target seconds")
    parser.add_argument("--min-val-bpb", type=float, default=None, help="minimum validation bpb")
    parser.add_argument("--tok-per-sec-median", type=float, default=None, help="median throughput")
    parser.add_argument("--dt-median-ms", type=float, default=None, help="median dt in ms")
    parser.add_argument("--status", type=str, default="ok", help="status label")
    parser.add_argument("--notes", type=str, default="", help="free-form notes")
    parser.add_argument("--inference-json", type=str, default="", help="optional JSON from scripts/lane_b_infer_ratio.py")
    args = parser.parse_args()

    ensure_schema(args.schema_path)

    parsed_metrics = {}
    if args.log_path:
        parsed_metrics = parse_metrics_from_log(args.log_path)

    inference_data = {}
    if args.inference_json:
        inference_data = json.loads(Path(args.inference_json).read_text(encoding="utf-8"))

    values = {
        "timestamp_utc": args.timestamp_utc or utc_timestamp(),
        "git_commit": args.git_commit or short_git_commit(),
        "run_id": args.run_id,
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "target_metric_name": args.target_metric_name,
        "target_metric_threshold": args.target_metric_threshold,
        "n_scaling_params": args.n_scaling_params,
        "inferred_data_scaling_exponent": args.inferred_data_scaling_exponent,
        "inferred_target_tokens": args.inferred_target_tokens,
        "inferred_target_param_data_ratio": args.inferred_target_param_data_ratio,
        "confirmation_ratio": args.confirmation_ratio,
        "chosen_ratio": args.chosen_ratio,
        "lane_b_decision": args.lane_b_decision,
        "time_to_target_sec_measured": args.time_to_target_sec_measured,
        "time_to_target_sec_extrapolated": args.time_to_target_sec_extrapolated,
        "min_val_bpb": args.min_val_bpb,
        "tok_per_sec_median": args.tok_per_sec_median,
        "dt_median_ms": args.dt_median_ms,
        "status": args.status,
        "notes": args.notes,
    }

    # Fill empty metric fields from parsed log values.
    for key in (
        "time_to_target_sec_measured",
        "time_to_target_sec_extrapolated",
        "min_val_bpb",
        "tok_per_sec_median",
        "dt_median_ms",
    ):
        if values[key] is None and key in parsed_metrics:
            values[key] = parsed_metrics[key]

    # Optional consistency checks if inference JSON is provided.
    if inference_data:
        if inference_data.get("inferred_target_param_data_ratio") is not None:
            expected = float(inference_data["inferred_target_param_data_ratio"])
            if abs(expected - float(values["inferred_target_param_data_ratio"])) > 1e-9:
                raise ValueError("inferred_target_param_data_ratio does not match inference JSON")
        if inference_data.get("confirmation_ratio") is not None:
            expected = float(inference_data["confirmation_ratio"])
            if abs(expected - float(values["confirmation_ratio"])) > 1e-9:
                raise ValueError("confirmation_ratio does not match inference JSON")

    row = [normalize_scalar(values.get(col)) for col in EXPECTED_COLUMNS]
    with Path(args.schema_path).open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"Appended Lane B row to {args.schema_path}")
    print(f"run_id={args.run_id}, lane_b_decision={args.lane_b_decision}, status={args.status}")


if __name__ == "__main__":
    main()
