"""
Compute a recommended target threshold from baseline runs.
"""

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_seed_csv(raw):
    return [s.strip() for s in raw.split(",") if s.strip()]


def collect_successful_rows(results_csv, required_seeds):
    path = Path(results_csv)
    if not path.exists():
        raise ValueError(f"Missing baseline results CSV: {results_csv}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Results CSV has no header: {results_csv}")

        required_columns = {"seed", "experiment_name", "min_val_bpb", "status"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Results CSV missing required columns {sorted(missing)}: {results_csv}"
            )

        all_ok_rows = []
        latest_any_by_seed = {}
        latest_ok_by_seed = {}
        for idx, row in enumerate(reader, start=2):
            seed = (row.get("seed") or "").strip()
            if seed:
                latest_any_by_seed[seed] = {
                    "line": idx,
                    "status": (row.get("status") or "").strip().lower(),
                    "experiment_name": (row.get("experiment_name") or "").strip(),
                }

            status = (row.get("status") or "").strip().lower()
            if status != "ok":
                continue

            min_raw = (row.get("min_val_bpb") or "").strip()
            if not min_raw:
                continue
            try:
                min_val_bpb = float(min_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid min_val_bpb at line {idx} in {results_csv}: {min_raw}"
                ) from exc
            if not math.isfinite(min_val_bpb):
                continue

            record = {
                "seed": seed,
                "experiment_name": (row.get("experiment_name") or "").strip(),
                "min_val_bpb": min_val_bpb,
                "_line": idx,
            }
            all_ok_rows.append(record)
            latest_ok_by_seed[seed] = record

    if required_seeds:
        missing_seeds = []
        selected_rows = []
        for seed in required_seeds:
            latest_any = latest_any_by_seed.get(seed)
            latest_ok = latest_ok_by_seed.get(seed)
            if latest_any is None or latest_ok is None or latest_ok["_line"] != latest_any["line"]:
                missing_seeds.append(seed)
                continue

            selected_rows.append(
                {
                    "seed": latest_ok["seed"],
                    "experiment_name": latest_ok["experiment_name"],
                    "min_val_bpb": latest_ok["min_val_bpb"],
                }
            )

        if missing_seeds:
            raise ValueError(
                "Missing successful baseline rows for required seeds: "
                + ",".join(missing_seeds)
            )
        return selected_rows

    return [
        {
            "seed": r["seed"],
            "experiment_name": r["experiment_name"],
            "min_val_bpb": r["min_val_bpb"],
        }
        for r in all_ok_rows
    ]


def compute_threshold(min_vals, threshold_offset):
    if threshold_offset < 0:
        raise ValueError("threshold_offset must be >= 0")
    if not min_vals:
        raise ValueError("No successful baseline min_val_bpb values found")

    median_min = float(statistics.median(min_vals))
    recommended = float(round(median_min + threshold_offset, 6))
    return median_min, recommended


def write_env_file(
    output_env,
    recommended_threshold,
    median_min,
    successful_runs_count,
    export_threshold_var,
    export_prefix,
):
    out_env = Path(output_env)
    out_env.parent.mkdir(parents=True, exist_ok=True)
    with out_env.open("w", encoding="utf-8") as f:
        f.write(f"export {export_threshold_var}={recommended_threshold:.6f}\n")
        f.write(f"export {export_prefix}_MEDIAN_MIN_VAL_BPB={median_min:.6f}\n")
        f.write(f"export {export_prefix}_SUCCESSFUL_RUNS={successful_runs_count}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute target threshold from baseline runs."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="results CSV produced by runs/speedrun_theory.sh baseline runs",
    )
    parser.add_argument(
        "--required-seeds",
        type=str,
        default="",
        help="comma-separated seeds that must each have a successful baseline row",
    )
    parser.add_argument(
        "--min-successful-runs",
        type=int,
        default=3,
        help="minimum successful baseline rows required",
    )
    parser.add_argument(
        "--threshold-offset",
        type=float,
        default=0.02,
        help="offset added to median(min_val_bpb) to define threshold",
    )
    parser.add_argument(
        "--target-metric-name",
        type=str,
        default="val_bpb",
        help="target metric name for output metadata",
    )
    parser.add_argument(
        "--export-threshold-var",
        type=str,
        default="TARGET_THRESHOLD",
        help="env var name to export threshold into",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default="BASELINE",
        help="prefix for extra exported env vars",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="optional output JSON path",
    )
    parser.add_argument(
        "--output-env",
        type=str,
        default="",
        help="optional output env file path",
    )
    args = parser.parse_args(argv)

    required_seeds = parse_seed_csv(args.required_seeds)
    selected_rows = collect_successful_rows(args.results_csv, required_seeds)
    if len(selected_rows) < args.min_successful_runs:
        raise ValueError(
            f"Need at least {args.min_successful_runs} successful baseline rows; "
            f"found {len(selected_rows)}"
        )

    if not args.export_threshold_var.strip():
        raise ValueError("--export-threshold-var must be non-empty")
    if not args.export_prefix.strip():
        raise ValueError("--export-prefix must be non-empty")

    min_vals = [float(r["min_val_bpb"]) for r in selected_rows]
    median_min, recommended_threshold = compute_threshold(min_vals, args.threshold_offset)

    result = {
        "timestamp_utc": utc_timestamp(),
        "results_csv": str(Path(args.results_csv).resolve()),
        "target_metric_name": args.target_metric_name,
        "threshold_formula": "median(min_val_bpb) + threshold_offset",
        "threshold_offset": float(args.threshold_offset),
        "required_seeds": required_seeds,
        "min_successful_runs": int(args.min_successful_runs),
        "successful_runs_count": len(selected_rows),
        "successful_runs": selected_rows,
        "median_min_val_bpb": float(median_min),
        "recommended_target_threshold": float(recommended_threshold),
        "export_threshold_var": args.export_threshold_var,
        "export_prefix": args.export_prefix,
    }

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")

    if args.output_env:
        write_env_file(
            output_env=args.output_env,
            recommended_threshold=recommended_threshold,
            median_min=median_min,
            successful_runs_count=len(selected_rows),
            export_threshold_var=args.export_threshold_var,
            export_prefix=args.export_prefix,
        )

    print(f"baseline_rows={len(selected_rows)}")
    print(f"median_min_val_bpb={median_min:.6f}")
    print(f"recommended_target_threshold={recommended_threshold:.6f}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc))
