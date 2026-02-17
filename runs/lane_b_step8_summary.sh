#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lane_b_common.sh"
lane_b_init

echo "Step 8: summarizing Lane B manifest runs ..."

if [ ! -f "$LANE_B_MANIFEST_CSV" ]; then
  echo "Missing $LANE_B_MANIFEST_CSV. Run step 6 first."
  exit 1
fi
if [ ! -f "$LANE_B_SCHEMA_PATH" ]; then
  echo "Missing $LANE_B_SCHEMA_PATH. Run step 7 first."
  exit 1
fi

"$PYTHON_BIN" - "$LANE_B_MANIFEST_CSV" "$LANE_B_SCHEMA_PATH" <<'PY'
import csv
import sys
from collections import defaultdict

manifest_path, schema_path = sys.argv[1], sys.argv[2]

run_meta = {}
with open(manifest_path, "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        run_meta[row["run_id"]] = row

rows = []
with open(schema_path, "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["run_id"] in run_meta:
            rows.append(row)

if not rows:
    print("No matching rows found in schema for current manifest.")
    raise SystemExit(0)

by_seed = defaultdict(dict)
mode_order = ["inferred", "confirmation", "fallback"]
present_modes = set()
for r in rows:
    seed = r["seed"]
    mode = r["lane_b_decision"]
    present_modes.add(mode)
    measured = float(r["time_to_target_sec_measured"]) if r["time_to_target_sec_measured"] else None
    extrap = float(r["time_to_target_sec_extrapolated"]) if r["time_to_target_sec_extrapolated"] else None
    score = measured if measured is not None else extrap
    by_seed[seed][mode] = score

ordered_modes = [m for m in mode_order if m in present_modes]
extra_modes = sorted(m for m in present_modes if m not in mode_order)
ordered_modes.extend(extra_modes)

print("Per-seed time-to-target (sec; measured preferred, fallback extrapolated):")
print("seed," + ",".join(ordered_modes) + ",winner")
wins = {m: 0 for m in ordered_modes}
wins["tie_or_missing"] = 0
for seed in sorted(by_seed, key=lambda x: int(x)):
    scores = {mode: by_seed[seed].get(mode) for mode in ordered_modes}
    present_scores = {k: v for k, v in scores.items() if v is not None}
    if not present_scores:
        winner = "tie_or_missing"
    elif len(present_scores) == 1:
        winner = next(iter(present_scores.keys()))
    else:
        best_mode = min(present_scores, key=present_scores.get)
        best_val = present_scores[best_mode]
        ties = [m for m, v in present_scores.items() if v == best_val]
        winner = best_mode if len(ties) == 1 else "tie_or_missing"
    wins[winner] += 1
    cells = [seed]
    for mode in ordered_modes:
        val = scores.get(mode)
        cells.append("" if val is None else f"{val:.2f}")
    cells.append(winner)
    print(",".join(cells))

print("")
print("Win tally:")
print(", ".join([f"{mode}={wins[mode]}" for mode in ordered_modes]) + f", tie_or_missing={wins['tie_or_missing']}")

best_mode = None
best_count = -1
best_ties = []
for mode in ordered_modes:
    if wins[mode] > best_count:
        best_mode = mode
        best_count = wins[mode]
        best_ties = [mode]
    elif wins[mode] == best_count:
        best_ties.append(mode)

if best_mode is None:
    print("Recommended lane_b_decision: fallback")
else:
    if len(best_ties) > 1:
        print(
            "Top-win tie detected among: "
            + ", ".join(best_ties)
            + f". Selecting {best_mode} by mode precedence."
        )
    print(f"Recommended lane_b_decision: {best_mode}")
PY
