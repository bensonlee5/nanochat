"""
Extract calibration metric points from Lane B calibration run logs.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def main():
    parser = argparse.ArgumentParser(description="Extract Lane B calibration points from log files.")
    parser.add_argument("--iterations", type=str, required=True, help="comma-separated iteration counts, e.g. 300,600,900")
    parser.add_argument("--total-batch-size", type=int, required=True, help="total batch size in tokens")
    parser.add_argument("--log-dir", type=str, required=True, help="directory containing calibration logs")
    parser.add_argument("--log-prefix", type=str, default="laneb_calib_", help="prefix for calibration log filenames")
    parser.add_argument("--log-suffix", type=str, default=".log", help="suffix for calibration log filenames")
    parser.add_argument("--output-json", type=str, default="", help="optional output JSON path")
    parser.add_argument("--output-env", type=str, default="", help="optional output .env snippet path")
    args = parser.parse_args()

    iters = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
    if len(iters) < 2:
        raise ValueError("Need at least two calibration iteration counts")

    calib_tokens = []
    calib_metrics = []
    log_dir = Path(args.log_dir)
    for it in iters:
        log_path = log_dir / f"{args.log_prefix}{it}{args.log_suffix}"
        metric = extract_min_val_bpb(log_path)
        calib_tokens.append(int(args.total_batch_size * it))
        calib_metrics.append(float(metric))

    result = {
        "timestamp_utc": utc_timestamp(),
        "iterations": iters,
        "total_batch_size": int(args.total_batch_size),
        "calib_tokens": calib_tokens,
        "calib_metrics": calib_metrics,
        "calib_tokens_csv": ",".join(str(x) for x in calib_tokens),
        "calib_metrics_csv": ",".join(f"{x:.10g}" for x in calib_metrics),
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
