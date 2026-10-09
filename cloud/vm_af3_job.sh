#!/usr/bin/env bash
# Runs ON the VM, detached (cloud/af3_run.sh starts it through
# cloud/lib_detached.sh, which writes OUT/EXIT_CODE when it ends). Sets the VM
# up (cloud/vm_af3_setup.sh), then runs the plan (harness/run_plan.py). OUT is
# results/af3/<session> inside the checkout; everything in it is fetched.
#
#   PLATFORM= PLAN= WEIGHTS= TARGETS= SESSION= [RESULTS_URI=] [LIBTPU_VERSION=] bash cloud/vm_af3_job.sh OUT
#
# LIBTPU_VERSION (TPU): the libtpu setup installs (cloud/vm_tpu_stack.sh);
# a plan with several libtpu versions switches between them itself.
#
# RESULTS_URI (optional): gs:// folder in the results bucket for this session.
# The plan uploads each target's folder there as soon as it finishes; this
# script uploads OUT once more at the end, also after a failed setup.
#
# Exit status: setup's (3 inputs, 4 weights, 5 device or TPU stack, other:
# install), or the plan's (0 all runs exited 0, 1 some failed and were
# recorded, 5 a libtpu switch failed).
set -uo pipefail
OUT=${1:?output folder}
: "${PLATFORM:?}" "${PLAN:?}" "${WEIGHTS:?}" "${TARGETS:?}" "${SESSION:?}"
RESULTS_URI="${RESULTS_URI:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
cd "$ROOT"

# Final upload of OUT to the results bucket; a failure is reported, never fatal.
# The derived inputs (work/input_seeds.json) are never uploaded. OUT/.uploaded
# marks a complete upload: without it, cloud/lib_vm.sh's light fetch brings
# everything, since the bucket may lack the large files.
upload_out() {
  [ -n "$RESULTS_URI" ] || return 0
  rm -f "$OUT/.uploaded"
  if gcloud storage rsync --recursive --exclude='.*/work/input_seeds\.json$' "$OUT" "$RESULTS_URI" \
      > "$OUT/upload_final.log" 2>&1; then
    touch "$OUT/.uploaded"
    echo ">> [$(date -u +%H:%M:%S)] job: uploaded the session to the results bucket"
  else
    echo "!! [$(date -u +%H:%M:%S)] job: final upload to the results bucket failed (upload_final.log)"
  fi
}

echo ">> [$(date -u +%H:%M:%S)] job: $PLATFORM, plan $PLAN, session $SESSION, weights $WEIGHTS"
bash cloud/vm_af3_setup.sh "$OUT"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "!! [$(date -u +%H:%M:%S)] setup failed with exit code $rc; no AlphaFold3 run started"
  upload_out
  exit "$rc"
fi

echo ">> [$(date -u +%H:%M:%S)] job: running the plan"
python3 harness/run_plan.py --plan "$PLAN" --platform "$PLATFORM" --session "$SESSION" \
  --model_dir "${AF3_WEIGHTS_DIR:-$HOME/af3_weights_run}" ${RESULTS_URI:+--upload_uri "$RESULTS_URI"}
rc=$?
# Derived inputs (frozen input with one seed, MSA included) are large and
# reproducible; each run.json keeps their SHA-256. Not fetched.
find "$OUT" -path '*/work/input_seeds.json' -delete 2> /dev/null || true
echo ">> [$(date -u +%H:%M:%S)] job: finished, exit code $rc"
upload_out
exit "$rc"
