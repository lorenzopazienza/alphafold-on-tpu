#!/usr/bin/env bash
# Runs ON the VM, detached (cloud/af3_run.sh starts it through
# cloud/lib_detached.sh, which writes OUT/EXIT_CODE when it ends). Sets the VM
# up (cloud/vm_af3_setup.sh), then runs the plan (harness/run_plan.py). OUT is
# results/af3/<session> inside the checkout; everything in it is fetched.
#
#   PLATFORM= PLAN= WEIGHTS= TARGETS= SESSION= bash cloud/vm_af3_job.sh OUT
#
# Exit status: setup's (3 inputs, 4 weights, 5 device, other: install), or
# the plan's (0 all runs exited 0, 1 some failed and were recorded).
set -uo pipefail
OUT=${1:?output folder}
: "${PLATFORM:?}" "${PLAN:?}" "${WEIGHTS:?}" "${TARGETS:?}" "${SESSION:?}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
cd "$ROOT"

echo ">> [$(date -u +%H:%M:%S)] job: $PLATFORM, plan $PLAN, session $SESSION, weights $WEIGHTS"
bash cloud/vm_af3_setup.sh "$OUT"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "!! [$(date -u +%H:%M:%S)] setup failed with exit code $rc; no AlphaFold3 run started"
  exit "$rc"
fi

echo ">> [$(date -u +%H:%M:%S)] job: running the plan"
python3 harness/run_plan.py --plan "$PLAN" --platform "$PLATFORM" --session "$SESSION" \
  --model_dir "${AF3_WEIGHTS_DIR:-$HOME/af3_weights_run}"
rc=$?
# Derived inputs (frozen input with one seed, MSA included) are large and
# reproducible; each run.json keeps their SHA-256. Not fetched.
find "$OUT" -path '*/work/input_seeds.json' -delete 2> /dev/null || true
echo ">> [$(date -u +%H:%M:%S)] job: finished, exit code $rc"
exit "$rc"
