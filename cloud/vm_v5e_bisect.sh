#!/usr/bin/env bash
# Runs ON the v5e VM, detached (cloud/v5e_bisect.sh starts it through
# cloud/lib_detached.sh, which writes OUT/EXIT_CODE when it ends). Sets the VM
# up exactly as the probe did (cloud/vm_af3_setup.sh: AlphaFold3 with
# af3_tpu.patch, jax[tpu]==0.10.2, random weights), then runs the bisection
# variants (cloud/vm_v5e_bisect.py). OUT is results/af3/<session> inside the
# checkout; everything in it is fetched.
#
#   SESSION= BISECT_BUDGET_MIN= VARIANT_TIMEOUT_MIN= [RESULTS_URI=] [BISECT_PLAN=] bash cloud/vm_v5e_bisect.sh OUT
#
# BISECT_PLAN: optional JSON plan (path in the checkout) passed to
# vm_v5e_bisect.py --plan; without it the built-in variants V0 to V9 run.
#
# The budget counts from this script's start, setup included.
# Exit status: setup's (3 inputs, 4 weights, 5 device, other: install), or
# vm_v5e_bisect.py's (0 the bisection ran, 2 internal error).
set -uo pipefail
OUT=${1:?output folder}
: "${SESSION:?}" "${BISECT_BUDGET_MIN:?}" "${VARIANT_TIMEOUT_MIN:?}"
RESULTS_URI="${RESULTS_URI:-}"
BISECT_PLAN="${BISECT_PLAN:-}"
JOB_START=$(date +%s)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bisect.log") 2>&1
cd "$ROOT"

upload_out() {
  [ -n "$RESULTS_URI" ] || return 0
  if gcloud storage rsync --recursive "$OUT" "$RESULTS_URI" > "$OUT/upload_final.log" 2>&1; then
    echo ">> [$(date -u +%H:%M:%S)] bisect: uploaded the session to the results bucket"
  else
    echo "!! [$(date -u +%H:%M:%S)] bisect: upload to the results bucket failed (upload_final.log)"
  fi
}

echo ">> [$(date -u +%H:%M:%S)] bisect: session $SESSION, budget $BISECT_BUDGET_MIN min, $VARIANT_TIMEOUT_MIN min per variant"
PLATFORM=v5e WEIGHTS=random TARGETS=7U3J,7D5C bash cloud/vm_af3_setup.sh "$OUT"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "!! [$(date -u +%H:%M:%S)] setup failed with exit code $rc; no variant ran"
  upload_out
  exit "$rc"
fi

python3 cloud/vm_v5e_bisect.py --out "$OUT" --session "$SESSION" --job_start "$JOB_START" \
  --budget_min "$BISECT_BUDGET_MIN" --timeout_min "$VARIANT_TIMEOUT_MIN" ${BISECT_PLAN:+--plan "$BISECT_PLAN"}
rc=$?
find "$OUT" -path '*/work/input_seeds.json' -delete 2> /dev/null || true
echo ">> [$(date -u +%H:%M:%S)] bisect: finished, exit code $rc"
upload_out
exit "$rc"
