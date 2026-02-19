"""
Extract Lane B calibration metric points from baseline-run logs.
"""

import argparse
import csv
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOG_SEED_RE = re.compile(r"(?:^|[_-])s(\d+)(?:[_-]|$)")


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def extract_min_val_bpb(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    step_re = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s*([0-9.]+)")
    min_re = re.compile(r"Minimum validation bpb:\s*([0-9.]+)")

    step_matches = list(step_re.finditer(text))
    if not step_matches:
        raise ValueError(
            f"Missing calibration eval points in {log_path}; log may be incomplete"
        )

    # If logs were appended across restarts, step counters reset (e.g. 900 -> 0).
    # Keep only the final monotonic step segment.
    start_idx = 0
    prev_step = -1
    for i, m in enumerate(step_matches):
        step = int(m.group(1))
        if step < prev_step:
            start_idx = i
        prev_step = step

    segment = step_matches[start_idx:]
    segment_start_pos = segment[0].start()
    min_val_from_steps = min(float(m.group(2)) for m in segment)

    summary_matches = [
        float(m.group(1)) for m in min_re.finditer(text) if m.start() >= segment_start_pos
    ]
    if not summary_matches:
        raise ValueError(
            f"Missing run-complete summary in {log_path}; calibration log may be incomplete"
        )

    min_val_from_summary = min(summary_matches)
    return float(min(min_val_from_steps, min_val_from_summary))


def extract_total_batch_size(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Total batch size\s+([0-9,]+)\s+=>", text)
    if m is None:
        raise ValueError(f"Missing total batch size in {log_path}")
    return int(m.group(1).replace(",", ""))


def extract_val_curve_points(log_path):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    step_re = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s*([0-9.]+)")
    step_pairs = [(int(m.group(1)), float(m.group(2))) for m in step_re.finditer(text)]
    if not step_pairs:
        raise ValueError(f"Missing validation curve in {log_path}")

    start_idx = 0
    prev_step = -1
    for i, (step, _) in enumerate(step_pairs):
        if step < prev_step:
            start_idx = i
        prev_step = step
    segment = step_pairs[start_idx:]

    dedup = {}
    for step, val in segment:
        if step <= 0:
            continue
        dedup[step] = val
    if len(dedup) < 3:
        raise ValueError(f"Need at least 3 positive-step validation points in {log_path}")

    total_batch_size = extract_total_batch_size(log_path)
    points = [(float(step * total_batch_size), float(val)) for step, val in sorted(dedup.items())]
    return points


def _filter_paths_by_required_seeds(log_paths, required_seeds, source_name):
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


def baseline_log_paths_from_results_csv(results_csv, results_dir, required_seeds=None):
    paths = []
    required_seeds = required_seeds or []
    required_set = set(required_seeds)
    selected_by_seed = {}
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status != "ok":
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


def extract_baseline_calibration(log_paths):
    by_tokens = {}
    for log_path in sorted(set(log_paths)):
        try:
            points = extract_val_curve_points(log_path)
        except ValueError:
            continue
        for tokens, metric in points:
            by_tokens.setdefault(tokens, []).append(metric)
    if not by_tokens:
        raise ValueError("No usable baseline validation curves found for calibration extraction")

    calib_tokens = sorted(by_tokens.keys())
    calib_metrics = [float(np.median(by_tokens[tok])) for tok in calib_tokens]
    if len(calib_tokens) < 3:
        raise ValueError(
            "Need at least 3 aggregated calibration points from baseline logs; "
            f"only found {len(calib_tokens)}"
        )
    return calib_tokens, calib_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract Lane B calibration points from baseline logs."
    )
    parser.add_argument("--baseline-results-csv", type=str, default="", help="baseline results CSV with experiment_name/status columns")
    parser.add_argument("--baseline-results-dir", type=str, default="", help="directory containing baseline *_train.log files")
    parser.add_argument("--baseline-log-glob", type=str, default="", help="optional glob for baseline *_train.log files")
    parser.add_argument("--baseline-required-seeds", type=str, default="", help="optional comma-separated baseline seeds to require and use in order, e.g. 41,42,43")
    parser.add_argument("--output-json", type=str, default="", help="optional output JSON path")
    parser.add_argument("--output-env", type=str, default="", help="optional output .env snippet path")
    args = parser.parse_args()

    if not args.baseline_results_csv and not args.baseline_log_glob:
        raise ValueError(
            "Provide baseline logs via --baseline-results-csv/--baseline-results-dir "
            "or --baseline-log-glob."
        )

    required_seeds = parse_seed_list(
        args.baseline_required_seeds, "--baseline-required-seeds"
    )

    log_paths = []
    if args.baseline_results_csv:
        if not args.baseline_results_dir:
            raise ValueError(
                "--baseline-results-dir is required when --baseline-results-csv is set"
            )
        log_paths.extend(
            baseline_log_paths_from_results_csv(
                results_csv=args.baseline_results_csv,
                results_dir=args.baseline_results_dir,
                required_seeds=required_seeds,
            )
        )
    if args.baseline_log_glob:
        log_paths.extend(glob.glob(args.baseline_log_glob))

    log_paths = _filter_paths_by_required_seeds(
        log_paths, required_seeds, source_name="baseline log discovery"
    )
    if not log_paths:
        raise ValueError("No baseline logs found for calibration extraction")
    calib_tokens, calib_metrics = extract_baseline_calibration(log_paths)

    result = {
        "timestamp_utc": utc_timestamp(),
        "mode": "baseline_logs",
        "calib_tokens": calib_tokens,
        "calib_metrics": calib_metrics,
        "calib_tokens_csv": ",".join(str(x) for x in calib_tokens),
        "calib_metrics_csv": ",".join(f"{x:.10g}" for x in calib_metrics),
        "baseline_required_seeds": required_seeds,
    }

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")

    if args.output_env:
        with open(args.output_env, "w", encoding="utf-8") as f:
            f.write(f"CALIB_TOKENS={result['calib_tokens_csv']}\n")
            f.write(f"CALIB_METRICS={result['calib_metrics_csv']}\n")

    print(f"CALIB_TOKENS={result['calib_tokens_csv']}")
    print(f"CALIB_METRICS={result['calib_metrics_csv']}")


if __name__ == "__main__":
    main()
