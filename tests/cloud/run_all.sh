#!/bin/bash
# Runs the six suites and prints one line per scenario. Output in
# $WORK/run_all.out (default ${TMPDIR:-/tmp}/af3-cloud-tests/run_all.out).
# About 15 minutes on a laptop. Creates no cloud resources (see common.sh).
. "$(dirname "$0")/common.sh"
OUT="$WORK/run_all.out"
for s in suite_1_basics suite_2_resilience suite_3_zones_gpu_plans suite_4_tpu_stack suite_5_pilot suite_6_connection; do
  echo ">> $s" >&2
  bash "$KIT/$s.sh"
done > "$OUT" 2>&1
python3 "$KIT/summarize.py" "$OUT"
