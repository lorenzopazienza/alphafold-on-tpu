#!/usr/bin/env bash
# AlphaFold3 smoke test on a single-chip Spot Cloud TPU. Creates the VM, arms a
# self-delete watchdog, runs af3_tpu/vm_smoke.sh there (patched AlphaFold3,
# jax[tpu], random weights made on the VM, toy input once), copies the results
# to results/af3_smoke/<timestamp>_<accel>/ and deletes the VM, also on error
# or Ctrl-C. Not a measurement.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   bash cloud/af3_tpu_smoke.sh                                    # v5e
#   ACCEL=v6e-1 RUNTIME=v2-alpha-tpuv6e ZONE=europe-west4-a \
#     bash cloud/af3_tpu_smoke.sh                                  # v6e
set -euo pipefail

die() { echo "!! $*" >&2; exit 1; }

# cloud/env.sh exports ACCEL=v5litepod-8, ZONE, TPU_NAME and MAX_HOURS for the
# multi-chip AF2 sessions; inheriting them here would create the wrong VM.
[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell; this script has its own defaults."

PROJECT="${PROJECT:-af2-tpu-benchmark}"
ACCEL="${ACCEL:-v5litepod-1}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
ZONE="${ZONE:-europe-west4-b}"
SPOT="${SPOT:-1}"
MAX_HOURS="${MAX_HOURS:-2}"
TPU_NAME="${TPU_NAME:-af3-smoke-$ACCEL}"

[[ "$ACCEL" == *-1 ]] || die "ACCEL=$ACCEL is not a single-chip type (v5litepod-1 or v6e-1)."
command -v gcloud >/dev/null || die "gcloud not found."

cd "$(git rev-parse --show-toplevel)"
source af3_tpu/pin.sh
BENCH_COMMIT=$(git rev-parse HEAD)
BENCH_DIRTY=$([ -z "$(git status --porcelain -- af3_tpu cloud/af3_tpu_smoke.sh)" ] && echo false || echo true)
SESSION="$(date -u +%Y%m%dT%H%M%SZ)_$ACCEL"
EXPIRES=$(( $(date +%s) + MAX_HOURS * 3600 ))

GC=(--project="$PROJECT" --zone="$ZONE")
SSH=(gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${GC[@]}" --ssh-flag=-oServerAliveInterval=30 --command)

# Never adopt (and later delete) a VM this run did not create.
if gcloud compute tpus tpu-vm describe "$TPU_NAME" "${GC[@]}" >/dev/null 2>&1; then
  die "$TPU_NAME already exists in $ZONE. Delete it or set TPU_NAME."
fi

# From here on the VM is deleted however the script ends. Results are copied
# back first, so a failed run still leaves its logs in results/af3_smoke/.
cleanup() {
  local rc=$?
  [ "$rc" -ne 0 ] || [ "${SMOKE_DONE:-0}" = "1" ] || rc=1
  trap - EXIT INT TERM
  set +e
  echo ">> Copying results to results/af3_smoke/$SESSION"
  mkdir -p results/af3_smoke
  gcloud compute tpus tpu-vm scp --recurse "$TPU_NAME:af3_smoke/$SESSION" results/af3_smoke/ "${GC[@]}"
  echo ">> Deleting $TPU_NAME"
  gcloud compute tpus tpu-vm delete "$TPU_NAME" "${GC[@]}" --quiet
  echo ">> TPU VMs left in $ZONE (anything listed here is billing):"
  gcloud compute tpus tpu-vm list "${GC[@]}" --format="table(name,acceleratorType,state)"
  echo ">> exit code $rc; session: results/af3_smoke/$SESSION/session.json"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

SPOT_FLAG=()
[ "$SPOT" = "1" ] && SPOT_FLAG=(--spot)

echo ">> Creating $TPU_NAME ($ACCEL, $RUNTIME) in $ZONE, spot=$SPOT, session $SESSION"
gcloud compute tpus tpu-vm create "$TPU_NAME" "${GC[@]}" \
  --accelerator-type="$ACCEL" --version="$RUNTIME" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --labels="purpose=af3-smoke,owner=lorenzo,expires=$EXPIRES" \
  ${SPOT_FLAG[@]+"${SPOT_FLAG[@]}"}

echo ">> Arming watchdog: VM self-deletes in ${MAX_HOURS} h"
"${SSH[@]}" "set -e
  G=\$(command -v gcloud || true)
  if [ -z \"\$G\" ]; then echo 'NO GCLOUD ON VM: watchdog NOT armed' >&2; exit 3; fi
  sudo systemd-run --unit=af3-watchdog --on-active=${MAX_HOURS}h \
    \"\$G\" compute tpus tpu-vm delete $TPU_NAME --project=$PROJECT --zone=$ZONE --quiet
  systemctl list-timers af3-watchdog* --no-pager"

echo ">> Copying af3_tpu/ (commit $BENCH_COMMIT, dirty=$BENCH_DIRTY)"
TGZ=$(mktemp -t af3-tpu.XXXXXX)
tar czf "$TGZ" --exclude=__pycache__ af3_tpu
gcloud compute tpus tpu-vm scp "$TGZ" "$TPU_NAME:af3_tpu.tgz" "${GC[@]}"
rm -f "$TGZ"
"${SSH[@]}" "mkdir -p ~/alphafold-on-tpu ~/af3_smoke && tar xzf ~/af3_tpu.tgz -C ~/alphafold-on-tpu"

echo ">> Running af3_tpu/vm_smoke.sh on the VM"
"${SSH[@]}" "cd ~/alphafold-on-tpu && \
  ACCEL=$ACCEL RUNTIME=$RUNTIME ZONE=$ZONE PROJECT=$PROJECT TPU_NAME=$TPU_NAME SPOT=$SPOT \
  BENCH_COMMIT=$BENCH_COMMIT BENCH_DIRTY=$BENCH_DIRTY \
  bash af3_tpu/vm_smoke.sh ~/af3_smoke/$SESSION"
echo ">> Smoke test passed"
SMOKE_DONE=1
