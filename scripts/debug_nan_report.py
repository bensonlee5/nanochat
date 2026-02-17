"""
Summarize NaN diagnostics from debug matrix logs.
"""

import argparse
import csv
import math
import re
from pathlib import Path


LOSS_STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s+\([^)]+\)\s+\|\s+loss:\s+([^\s|]+)",
    re.IGNORECASE,
)
VAL_STEP_RE = re.compile(
    r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([^\s|]+)",
    re.IGNORECASE,
)
MIN_VAL_RE = re.compile(r"Minimum validation bpb:\s+([^\s|]+)", re.IGNORECASE)


def _parse_float_token(raw):
    token = raw.strip().lower()
    if token in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(token)
    except ValueError:
        return None


def parse_log_metrics(path):
    text = path.read_text(encoding="utf-8", errors="replace")

    first_nan_step = None
    last_finite_step = None

    for match in LOSS_STEP_RE.finditer(text):
        step = int(match.group(1))
        loss_val = _parse_float_token(match.group(2))
        if loss_val is None:
            continue
        if math.isnan(loss_val):
            if first_nan_step is None:
                first_nan_step = step
        elif math.isfinite(loss_val):
            last_finite_step = step

    for match in VAL_STEP_RE.finditer(text):
        step = int(match.group(1))
        val_bpb = _parse_float_token(match.group(2))
        if val_bpb is None:
            continue
        if math.isnan(val_bpb) and first_nan_step is None:
            first_nan_step = step

    min_val_bpb = ""
    min_match = MIN_VAL_RE.findall(text)
    if min_match:
        parsed = _parse_float_token(min_match[-1])
        if parsed is not None and math.isfinite(parsed):
            min_val_bpb = f"{parsed:.6f}"

    return {
        "nan_detected": "yes" if first_nan_step is not None else "no",
        "first_nan_step": "" if first_nan_step is None else str(first_nan_step),
        "last_finite_step": "" if last_finite_step is None else str(last_finite_step),
        "min_val_bpb": min_val_bpb,
    }


def load_manifest(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize NaN debug matrix logs.")
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest from debug run script")
    parser.add_argument("--output-csv", type=str, default="", help="optional output summary CSV")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest_rows = load_manifest(manifest_path)
    summary_rows = []

    for row in manifest_rows:
        log_path = Path(row["log_path"])
        parsed = {
            "nan_detected": "unknown",
            "first_nan_step": "",
            "last_finite_step": "",
            "min_val_bpb": "",
        }
        if log_path.exists():
            parsed = parse_log_metrics(log_path)

        summary_rows.append(
            {
                "variant": row.get("variant", ""),
                "seed": row.get("seed", ""),
                "nan_detected": parsed["nan_detected"],
                "first_nan_step": parsed["first_nan_step"],
                "last_finite_step": parsed["last_finite_step"],
                "exit_code": row.get("exit_code", ""),
                "min_val_bpb": parsed["min_val_bpb"],
                "log_path": str(log_path),
            }
        )

    variant_order = {"control": 0, "no_muon": 1, "no_compile": 2}
    summary_rows.sort(
        key=lambda r: (
            variant_order.get(r["variant"], 99),
            int(r["seed"]) if str(r["seed"]).isdigit() else 999999,
        )
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "variant",
                    "seed",
                    "nan_detected",
                    "first_nan_step",
                    "last_finite_step",
                    "exit_code",
                    "min_val_bpb",
                    "log_path",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print("| variant | seed | nan_detected | first_nan_step | last_finite_step | exit_code | min_val_bpb |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['variant']} | {row['seed']} | {row['nan_detected']} | "
            f"{row['first_nan_step'] or ''} | {row['last_finite_step'] or ''} | "
            f"{row['exit_code'] or ''} | {row['min_val_bpb'] or ''} |"
        )


if __name__ == "__main__":
    main()
