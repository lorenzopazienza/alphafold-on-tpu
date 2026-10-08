#!/usr/bin/env bash
# AlphaFold3 smoke test on a single-chip TPU v6e, created through the Compute
# Engine API with the Flex-start provisioning model. The request waits in a
# queue until capacity is free (at most REQUEST_VALID_FOR), then the VM runs
# af3_tpu/vm_smoke.sh unchanged, results are copied to
# results/af3_smoke/<timestamp>_<machine-type>/ and the VM is deleted. Not a
# measurement.
#
# Two independent ways the VM ends up deleted:
#   1. Google deletes it MAX_RUN_DURATION after it starts running
#      (--max-run-duration with --instance-termination-action=DELETE), even if
#      this Mac goes offline;
#   2. the EXIT trap below deletes it when this script ends, on error and on
#      Ctrl-C.
#
# vm_smoke.sh runs detached on the VM and is followed with short SSH polls
# (cloud/lib_detached.sh), so a dropped connection does not kill the run.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   bash cloud/af3_tpu_smoke_gce.sh
#   ZONE=us-east5-a bash cloud/af3_tpu_smoke_gce.sh        # another v6e zone
set -euo pipefail

die() { echo "!! $*" >&2; exit 1; }

# cloud/env.sh exports ZONE and other settings for the multi-chip AF2 sessions;
# inheriting them here would create the VM in the wrong place.
[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell; this script has its own defaults."

PROJECT="${PROJECT:-af2-tpu-benchmark}"
ZONE="${ZONE:-europe-west4-a}"
MACHINE_TYPE="${MACHINE_TYPE:-ct6e-standard-1t}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-}"            # empty: Compute Engine's default
MAX_RUN_DURATION="${MAX_RUN_DURATION:-2h}"      # Google deletes the VM this long after it starts
REQUEST_VALID_FOR="${REQUEST_VALID_FOR:-2h}"    # how long the request may wait in the queue

[[ "$MACHINE_TYPE" == *-1t ]] || die "MACHINE_TYPE=$MACHINE_TYPE is not a single-chip type (ct6e-standard-1t)."
command -v gcloud >/dev/null || die "gcloud not found."

# "1d2h3m4s" -> seconds, for the deadline and the expires label.
to_seconds() {
  echo "$1" | grep -Eq '^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s)?$' && [ -n "$1" ] \
    || die "Not a duration like 2h or 90m: '$1'"
  echo $(( $(echo "$1" | sed -E 's/([0-9]+)d/\1*86400+/; s/([0-9]+)h/\1*3600+/; s/([0-9]+)m/\1*60+/; s/([0-9]+)s/\1+/; s/\+$//') ))
}
RUN_S=$(to_seconds "$MAX_RUN_DURATION")
WAIT_S=$(to_seconds "$REQUEST_VALID_FOR")

cd "$(git rev-parse --show-toplevel)"
source af3_tpu/pin.sh
source cloud/lib_detached.sh
BENCH_COMMIT=$(git rev-parse HEAD)
BENCH_DIRTY=$([ -z "$(git status --porcelain -- af3_tpu cloud/af3_tpu_smoke_gce.sh cloud/lib_detached.sh)" ] && echo false || echo true)
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="${STAMP}_$MACHINE_TYPE"
VM_NAME="${VM_NAME:-af3-smoke-v6e-$(echo "$STAMP" | tr 'A-Z' 'a-z')}"
EXPIRES=$(( $(date +%s) + WAIT_S + RUN_S ))

# No bash arrays: macOS bash 3.2 fails on empty ones under set -u. None of
# these values can contain spaces, so plain unquoted strings are safe.
GC="--project=$PROJECT --zone=$ZONE"
DISK_TYPE_FLAG=""
[ -z "$BOOT_DISK_TYPE" ] || DISK_TYPE_FLAG="--boot-disk-type=$BOOT_DISK_TYPE"
vm_ssh() { gcloud compute ssh "$VM_NAME" $GC --ssh-flag=-oServerAliveInterval=30 --ssh-flag=-oConnectTimeout=30 --command "$1"; }

# Status of this VM's insert operation: PENDING, RUNNING, DONE, or empty.
insert_op() {
  gcloud compute operations list --project="$PROJECT" --zones="$ZONE" \
    --filter="targetLink~/instances/$VM_NAME\$ AND operationType=insert" \
    --format="value($1)" 2>/dev/null | head -1
}

# Never adopt (and later delete) a VM this run did not create.
if gcloud compute instances describe "$VM_NAME" $GC >/dev/null 2>&1; then
  die "$VM_NAME already exists in $ZONE. Delete it or set VM_NAME."
fi

# From here on the VM is deleted however the script ends. Results are copied
# back first, so a failed run still leaves its logs in results/af3_smoke/.
cleanup() {
  local rc=$?
  [ "$rc" -ne 0 ] || [ "${SMOKE_DONE:-0}" = "1" ] || rc=1
  trap - EXIT INT TERM
  set +e
  if [ "${COPIED:-0}" = "1" ]; then
    echo ">> Copying results to results/af3_smoke/$SESSION"
    mkdir -p results/af3_smoke
    gcloud compute scp --recurse "$VM_NAME:af3_smoke/$SESSION" results/af3_smoke/ $GC
  fi
  echo ">> Deleting $VM_NAME"
  if ! gcloud compute instances delete "$VM_NAME" $GC --quiet; then
    local op
    op=$(insert_op status)
    if [ "$op" = "PENDING" ] || [ "$op" = "RUNNING" ]; then
      echo "!! The Flex-start request for $VM_NAME is still queued and could not be cancelled."
      echo "!! If it is granted, Google deletes the VM after $MAX_RUN_DURATION. To delete it sooner:"
      echo "!!   gcloud compute instances delete $VM_NAME --project=$PROJECT --zone=$ZONE --quiet"
    fi
  fi
  # Listed project-wide and filtered here: a gcloud --filter that matches
  # nothing prints a "filter keys were not present" warning.
  echo ">> af3-smoke VMs left in $PROJECT (anything listed here is billing or queued):"
  local all left
  if all=$(gcloud compute instances list --project="$PROJECT" \
      --format="value(name,zone.basename(),machineType.basename(),status)"); then
    left=$(echo "$all" | awk '$1 ~ /^af3-smoke-/')
    echo "${left:-   none}"
  else
    echo "!! Could not list instances; check the Cloud console."
  fi
  echo ">> exit code $rc"
  [ "${COPIED:-0}" != "1" ] || echo ">> session: results/af3_smoke/$SESSION/session.json"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo ">> Requesting $VM_NAME ($MACHINE_TYPE, $IMAGE_FAMILY) in $ZONE, Flex-start"
echo "   queue up to $REQUEST_VALID_FOR, run at most $MAX_RUN_DURATION, session $SESSION"
gcloud compute instances create "$VM_NAME" $GC \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$BOOT_DISK_SIZE" $DISK_TYPE_FLAG \
  --provisioning-model=FLEX_START \
  --max-run-duration="$MAX_RUN_DURATION" \
  --request-valid-for-duration="$REQUEST_VALID_FOR" \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --reservation-affinity=none \
  --scopes=cloud-platform \
  --labels="purpose=af3-smoke,owner=lorenzo,provisioning=flex-start,expires=$EXPIRES" \
  --async

echo ">> Waiting for capacity (status printed at most once a minute)"
START=$(date +%s)
LAST_PRINT=0
while :; do
  NOW=$(date +%s)
  STATUS=$(gcloud compute instances describe "$VM_NAME" $GC --format="value(status)" 2>/dev/null || true)
  case "$STATUS" in
    RUNNING) echo ">> $VM_NAME is RUNNING after $(( (NOW - START) / 60 )) min"; break ;;
    STOPPING|STOPPED|SUSPENDING|SUSPENDED|TERMINATED)
      die "$VM_NAME reached status $STATUS before running." ;;
  esac
  if [ "$(insert_op status)" = "DONE" ]; then
    ERR=$(insert_op "error.errors[0].message")
    [ -z "$ERR" ] || die "Flex-start request failed: $ERR"
  fi
  [ $(( NOW - START )) -le $(( WAIT_S + 600 )) ] \
    || die "Still not running $(( (NOW - START) / 60 )) min after the request (REQUEST_VALID_FOR=$REQUEST_VALID_FOR)."
  if [ $(( NOW - LAST_PRINT )) -ge 60 ]; then
    echo "   $(date -u +%H:%M:%SZ) status=${STATUS:-not created yet} request=$(insert_op status)"
    LAST_PRINT=$NOW
  fi
  sleep 15
done

echo ">> Waiting for SSH"
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if gcloud compute ssh "$VM_NAME" $GC --quiet --command=true >/dev/null 2>&1; then break; fi
  [ "$i" -lt 20 ] || die "SSH to $VM_NAME not reachable after 5 min."
  sleep 15
done

echo ">> Copying af3_tpu/ (commit $BENCH_COMMIT, dirty=$BENCH_DIRTY)"
TGZ=$(mktemp -t af3-tpu.XXXXXX)
tar czf "$TGZ" --exclude=__pycache__ af3_tpu
gcloud compute scp "$TGZ" "$VM_NAME:af3_tpu.tgz" $GC
rm -f "$TGZ"
vm_ssh "mkdir -p ~/alphafold-on-tpu ~/af3_smoke/$SESSION && tar xzf ~/af3_tpu.tgz -C ~/alphafold-on-tpu"
COPIED=1

# vm_smoke.sh records ACCEL and RUNTIME in session.json; here they are the
# machine type and the image family. SPOT=0: Flex-start is not a Spot VM.
echo ">> Starting af3_tpu/vm_smoke.sh detached on the VM"
detached_start "af3_smoke/$SESSION" \
  "ACCEL=$MACHINE_TYPE RUNTIME=$IMAGE_FAMILY ZONE=$ZONE PROJECT=$PROJECT TPU_NAME=$VM_NAME SPOT=0 BENCH_COMMIT=$BENCH_COMMIT BENCH_DIRTY=$BENCH_DIRTY" \
  || die "Could not start vm_smoke.sh on the VM."
detached_wait "af3_smoke/$SESSION" || die "No result from vm_smoke.sh."
if [ "$DETACHED_RC" != "0" ]; then
  echo "!! vm_smoke.sh failed with exit code $DETACHED_RC; its logs are copied below"
  exit "$DETACHED_RC"
fi
echo ">> Smoke test passed"
SMOKE_DONE=1
