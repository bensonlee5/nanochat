"""
Backward-compatible Lane B wrapper around the generic baseline threshold script.
"""

from scripts.baseline_compute_threshold import (  # re-export for older imports/tests
    collect_successful_rows,
    compute_threshold,
    main as _generic_main,
    parse_seed_csv,
    utc_timestamp,
    write_env_file,
)


def main(argv=None):
    lane_b_defaults = [
        "--export-threshold-var",
        "LANE_B_TARGET_THRESHOLD",
        "--export-prefix",
        "LANE_B_BASELINE",
    ]
    merged = []
    if argv is None:
        import sys

        merged = list(sys.argv[1:])
    else:
        merged = list(argv)
    merged.extend(lane_b_defaults)
    _generic_main(merged)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(str(exc))
