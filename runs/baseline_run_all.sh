#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/baseline_step1_run_true_baselines.sh"
"$SCRIPT_DIR/baseline_step2_compute_threshold.sh"
