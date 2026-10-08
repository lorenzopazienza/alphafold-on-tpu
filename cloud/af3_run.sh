#!/usr/bin/env bash
# Runs a plan (harness/plans/*.yaml) on one cloud platform, with the same
# frozen inputs everywhere. Creates one single-device VM, uploads only what
# the plan needs, runs cloud/vm_af3_job.sh there detached (setup, then the
# plan), follows it with short SSH polls, fetches results/af3/<session>/ and
# deletes the VM, also on error, deadline or Ctrl-C.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   PLATFORM=v5e bash cloud/af3_run.sh                  # the probe (default plan)
#   PLATFORM=v6e PLAN=harness/plans/pilot.yaml WEIGHTS_GCS_URI=gs://BUCKET/af3.bin.zst \
#     bash cloud/af3_run.sh
#
# PLATFORM  API              default                       provisioning
# cpu       Compute Engine   n2-highmem-16 (Ice Lake)      Spot (SPOT=0: on-demand)
# l4        Compute Engine   g2-standard-8 (1 x L4)        Spot (SPOT=0: on-demand,
#                                                          PROVISIONING=flex_start: Flex-start)
# v5e       legacy TPU API   v5litepod-1                   Spot (SPOT=0: on-demand)
# v6e       Compute Engine   ct6e-standard-1t              Flex-start
#
# Overrides: PROJECT ZONE MACHINE_TYPE SPOT PROVISIONING (spot, standard, or
# flex_start for l4 and v6e; wins over SPOT) IMAGE_PROJECT IMAGE_FAMILY RUNTIME
# (v5e) BOOT_DISK_SIZE MIN_CPU_PLATFORM (cpu) REQUEST_VALID_FOR (Flex-start
# queue, 90s to 2h, default 2h) DETACHED_DEADLINE_MIN MAX_RUN_DURATION WEIGHTS
# (random or gcs; default: the plan's) WEIGHTS_GCS_URI (gcs only; never
# printed) VM_NAME YES=1 (no prompt).
#
# Flex-start (v6e, and l4 with PROVISIONING=flex_start): the create request
# queues up to REQUEST_VALID_FOR and is not billed while queued; the VM gets
# --max-run-duration and --instance-termination-action=DELETE like every
# Compute Engine VM here. Flags as documented in `gcloud compute instances
# create --help` and https://cloud.google.com/compute/docs/instances/create-flex-start-vm
#
# Results bucket (optional): RESULTS_GCS_URI=gs://BUCKET[/PREFIX], a private
# bucket separate from the weights bucket. The VM uploads each target's
# folder as soon as it finishes, so a preempted or unreachable VM loses at
# most the target in progress; objects go to RESULTS_GCS_URI/af3/<session>/.
# If the final SSH fetch fails, the laptop fetches from the bucket instead.
# FETCH=light (default) brings records, logs, *_summary_confidences.json,
# *_ranking_scores.csv and the mmCIFs to the laptop and leaves *_data.json
# and the full *_confidences.json in the bucket; FETCH=full brings
# everything. Without RESULTS_GCS_URI, or if the VM's final upload failed,
# the fetch is full, since the VM and its copy are deleted afterwards.
#
# Preemption: when an SSH poll fails, the launcher asks the API for the VM's
# state; a preempted, stopped or deleted VM ends the run at once and goes
# straight to cleanup (bucket fetch if configured, delete, leftover check).
#
# Price: the worst case uses cloud/prices.csv for the zone's region; if the
# region (or machine type) has no row there, it says which list price it shows
# instead. Without a published price for the provisioning model (G2
# Flex-start), it uses the on-demand price as the upper bound and says so.
#
# The VM is deleted in every case: by the EXIT trap (results first), and
# independently by Google: --max-run-duration with
# --instance-termination-action=DELETE on every Compute Engine VM (Spot and
# on-demand too), the systemd watchdog on the legacy TPU VM. The script never
# adopts or deletes a VM it did not create, and lists leftovers at the end.
# macOS bash 3.2: no arrays.
set -euo pipefail

die() { echo "!! $*" >&2; exit 1; }

# cloud/env.sh exports ZONE, TPU_NAME and other settings for the multi-chip AF2
# sessions; inheriting them here would create the wrong VM.
[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell; this script has its own defaults."
: "${PLATFORM:?set PLATFORM to cpu, l4, v5e or v6e}"
command -v gcloud > /dev/null || die "gcloud not found."
command -v python3 > /dev/null || die "python3 not found."

PLAN="${PLAN:-harness/plans/probe.yaml}"
# Read before cloud/lib_detached.sh is sourced: it sets its own default of 60.
DEADLINE_OVERRIDE="${DETACHED_DEADLINE_MIN:-}"
PROJECT="${PROJECT:-af2-tpu-benchmark}"
SPOT="${SPOT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
BOOT_DISK_TYPE=""
RUNTIME=""
MIN_CPU_PLATFORM_DEFAULT=""
case "$PLATFORM" in
  cpu)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-n2-highmem-16}"
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
    BOOT_DISK_TYPE=pd-balanced
    case "$MACHINE_TYPE" in n2-*) MIN_CPU_PLATFORM_DEFAULT="Intel Ice Lake" ;; esac
    case "$MACHINE_TYPE" in
      a2-* | a3-* | a4-* | g2-* | g4-* | ct* | *tpu*) die "PLATFORM=cpu takes a CPU-only machine type, not $MACHINE_TYPE." ;;
    esac ;;
  l4)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-580}"
    BOOT_DISK_TYPE=pd-balanced
    case "$MACHINE_TYPE" in
      g2-standard-4 | g2-standard-8 | g2-standard-12 | g2-standard-16 | g2-standard-32) ;;
      *) die "PLATFORM=l4 takes a single-L4 type (g2-standard-4/8/12/16/32), not $MACHINE_TYPE." ;;
    esac ;;
  v5e)
    VM_API=tpu
    MACHINE_TYPE="${MACHINE_TYPE:-v5litepod-1}"
    RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
    ZONE="${ZONE:-europe-west4-b}"
    [ "$MACHINE_TYPE" = "v5litepod-1" ] || die "PLATFORM=v5e takes v5litepod-1 only (single chip), not $MACHINE_TYPE." ;;
  v6e)
    VM_API=gce
    MACHINE_TYPE="${MACHINE_TYPE:-ct6e-standard-1t}"
    ZONE="${ZONE:-europe-west4-a}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e}"
    [ "$MACHINE_TYPE" = "ct6e-standard-1t" ] || die "PLATFORM=v6e takes ct6e-standard-1t only (single chip), not $MACHINE_TYPE." ;;
  *) die "PLATFORM must be cpu, l4, v5e or v6e." ;;
esac
MIN_CPU_PLATFORM="${MIN_CPU_PLATFORM-$MIN_CPU_PLATFORM_DEFAULT}"
case "${PROVISIONING:-}" in
  "")
    case "$PLATFORM" in
      v6e) PROVISIONING=flex_start ;;
      *) if [ "$SPOT" = "1" ]; then PROVISIONING=spot; else PROVISIONING=standard; fi ;;
    esac ;;
  flex_start)
    case "$PLATFORM" in
      l4 | v6e) ;;
      *) die "PROVISIONING=flex_start is supported here for PLATFORM=l4 and v6e only." ;;
    esac ;;
  spot | standard)
    [ "$PLATFORM" != "v6e" ] || die "PLATFORM=v6e runs with Flex-start only (PROVISIONING=flex_start)." ;;
  *) die "PROVISIONING must be spot, standard or flex_start." ;;
esac
REGION="${ZONE%-*}"

# "1d2h3m4s" -> seconds.
to_seconds() {
  echo "$1" | grep -Eq '^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s)?$' && [ -n "$1" ] \
    || die "Not a duration like 2h or 90m: '$1'"
  echo $(( $(echo "$1" | sed -E 's/([0-9]+)d/\1*86400+/; s/([0-9]+)h/\1*3600+/; s/([0-9]+)m/\1*60+/; s/([0-9]+)s/\1+/; s/\+$//') ))
}

cd "$(git rev-parse --show-toplevel)"
source cloud/lib_vm.sh
source cloud/lib_detached.sh

# The plan, expanded for this platform (targets, runs, deadline, weights).
DESC=$(python3 harness/run_plan.py --plan "$PLAN" --platform "$PLATFORM" --describe) \
  || die "Could not read the plan $PLAN."
jget() { echo "$DESC" | python3 -c "import json,sys; v=json.load(sys.stdin)$1; print(v)"; }
TARGETS=$(jget "['targets']" | python3 -c "import ast,sys; print(','.join(ast.literal_eval(sys.stdin.read())))")
[ -n "$TARGETS" ] || die "The plan has no targets for $PLATFORM."
PLAN_NAME=$(basename "$PLAN" .yaml | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')
PLAN_WEIGHTS=$(jget "['weights']")
PLAN_DEADLINE=$(jget "['deadline_min']")
WEIGHTS="${WEIGHTS:-$PLAN_WEIGHTS}"
case "$WEIGHTS" in
  random) [ -z "${WEIGHTS_GCS_URI:-}" ] || echo "   note: WEIGHTS=random, so WEIGHTS_GCS_URI is ignored" ;;
  gcs)
    case "${WEIGHTS_GCS_URI:-}" in gs://?*/?*) ;; *) die "WEIGHTS=gcs needs WEIGHTS_GCS_URI=gs://BUCKET/OBJECT (a private bucket in $PROJECT)." ;; esac ;;
  *) die "WEIGHTS must be random or gcs." ;;
esac
[ "$WEIGHTS" = "$PLAN_WEIGHTS" ] || echo "   note: the plan asks for weights=$PLAN_WEIGHTS; running with WEIGHTS=$WEIGHTS"

# Results bucket and fetch mode.
RESULTS_GCS_URI="${RESULTS_GCS_URI:-}"
FETCH="${FETCH:-light}"
case "$FETCH" in light | full) ;; *) die "FETCH must be light or full." ;; esac
if [ -n "$RESULTS_GCS_URI" ]; then
  RESULTS_GCS_URI="${RESULTS_GCS_URI%/}"
  echo "$RESULTS_GCS_URI" | grep -Eq '^gs://[a-z0-9][a-z0-9._-]*[a-z0-9](/[A-Za-z0-9._-]+)*$' \
    || die "RESULTS_GCS_URI must be gs://BUCKET or gs://BUCKET/PREFIX (letters, digits, . _ - /)."
  RESULTS_BUCKET=$(echo "$RESULTS_GCS_URI" | cut -d/ -f3)
  if [ "$WEIGHTS" = "gcs" ] && [ "$RESULTS_BUCKET" = "$(echo "$WEIGHTS_GCS_URI" | cut -d/ -f3)" ]; then
    die "RESULTS_GCS_URI must be a different bucket from the weights bucket."
  fi
  FETCH_EFFECTIVE=$FETCH
else
  FETCH_EFFECTIVE=full
fi

DETACHED_DEADLINE_MIN="${DEADLINE_OVERRIDE:-$PLAN_DEADLINE}"
[ "$DETACHED_DEADLINE_MIN" -gt 0 ] 2> /dev/null \
  || die "The plan's deadline for $PLATFORM is not set (0). Size it from the probe, or set DETACHED_DEADLINE_MIN."
MAX_RUN_DURATION="${MAX_RUN_DURATION:-$(( DETACHED_DEADLINE_MIN + 30 ))m}"
MAX_RUN_S=$(to_seconds "$MAX_RUN_DURATION")
[ "$MAX_RUN_S" -ge $(( DETACHED_DEADLINE_MIN * 60 + 600 )) ] \
  || die "MAX_RUN_DURATION ($MAX_RUN_DURATION) must exceed the deadline ($DETACHED_DEADLINE_MIN min) by 10 min or more."
WATCHDOG_HOURS=$(( (MAX_RUN_S + 3599) / 3600 ))
if [ "$PROVISIONING" = "flex_start" ]; then
  # Google accepts 90 s to 2 h for a zonal Flex-start request.
  REQUEST_VALID_FOR="${REQUEST_VALID_FOR:-2h}"
  REQUEST_VALID_S=$(to_seconds "$REQUEST_VALID_FOR")
  [ "$REQUEST_VALID_S" -ge 90 ] && [ "$REQUEST_VALID_S" -le 7200 ] \
    || die "REQUEST_VALID_FOR ($REQUEST_VALID_FOR) must be between 90s and 2h for a Flex-start request in one zone."
  WAIT_S=$(( REQUEST_VALID_S + 600 ))
else
  WAIT_S=900
fi

# Frozen inputs must match the manifest before anything is created.
echo ">> Checking the plan's frozen inputs against inputs/manifest.csv"
python3 harness/verify_inputs.py --targets "$TARGETS" \
  || die "Frozen inputs do not match inputs/manifest.csv; nothing was created."

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="${STAMP}_${PLATFORM}_${PLAN_NAME}"
[ ! -e "results/af3/$SESSION" ] || die "results/af3/$SESSION exists; sessions are never overwritten."
VM_NAME="${VM_NAME:-af3-run-$PLATFORM-$(echo "$STAMP" | tr 'A-Z' 'a-z')}"
OUT_REL="alphafold-on-tpu/results/af3/$SESSION"
BENCH_COMMIT=$(git rev-parse HEAD)
EXPIRES=$(( $(date +%s) + MAX_RUN_S + WAIT_S ))
RESULTS_SESSION_URI="${RESULTS_GCS_URI:+$RESULTS_GCS_URI/af3/$SESSION}"
vm_init

# What will run and what it can cost, before anything is created. The price
# is the row for this platform, provisioning, machine type and the zone's
# region; any fallback is printed (PRICE_NOTES).
PRICE_OUT=$(python3 - "$PLATFORM" "$PROVISIONING" "$MACHINE_TYPE" "$REGION" <<'EOF'
import csv, sys
platform, prov, machine, region = sys.argv[1:5]
rows = [r for r in csv.DictReader(open('cloud/prices.csv')) if r['platform'] == platform]

def pick(provisioning):
  cands = [r for r in rows if r['provisioning'] == provisioning]
  for ok in (lambda r: r['machine_type'] == machine and r['region'] == region,
             lambda r: r['region'] == region,
             lambda r: r['machine_type'] == machine and r['region'] == 'europe-west4',
             lambda r: r['region'] == 'europe-west4',
             lambda r: True):
    for r in cands:
      if ok(r):
        return r
  return None

notes = []
row = pick(prov)
if row is None:
  sys.exit(0)
if not row['usd_per_hour'].strip():
  std = pick('standard')
  if std is None or not std['usd_per_hour'].strip():
    sys.exit(0)
  notes.append(f"no published {prov} price for {row['machine_type']} in {row['region']} "
               f"({row['source_url']}, read {row['read_on']}); the worst case uses the "
               f"on-demand price as an upper bound")
  row = std
if row['region'] != region:
  notes.append(f"no {platform} {prov} price for region {region} in cloud/prices.csv: "
               f"the price shown is the {row['region']} list price, not {region}'s")
if row['machine_type'] != machine:
  notes.append(f"no price for {machine} in cloud/prices.csv: the price shown is for {row['machine_type']}")
print(row['usd_per_hour'], row['read_on'], row['source_url'])
for n in notes:
  print(n)
EOF
)
PRICE_LINE=$(echo "$PRICE_OUT" | head -1)
PRICE_NOTES=$(echo "$PRICE_OUT" | tail -n +2)
[ -n "$PRICE_LINE" ] || die "No $PLATFORM $PROVISIONING price in cloud/prices.csv."
set -- $PRICE_LINE
PRICE=$1 PRICE_DATE=$2 PRICE_URL=$3
if [ "$VM_API" = "tpu" ]; then LIFETIME_H=$WATCHDOG_HOURS; else LIFETIME_H=$(python3 -c "print(round($MAX_RUN_S/3600, 2))"); fi
WORST=$(python3 -c "print(f'{$PRICE * $LIFETIME_H:.2f}')")

echo
echo "$DESC" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\">> Plan {d['plan']} ({d['label']}): {len(d['runs'])} harness runs, {d['processes']} AlphaFold3 processes\")
print('   targets:', ', '.join(f'{t} ({d[\"tokens\"][t]} tokens)' for t in d['targets']))
print(f\"   seeds {d['seeds']}, fresh-cache reps {d['fresh_reps']}, warm rerun of seed: {d['warm_rerun_seed'] or 'none'}\")
print(f\"   recycles {d['num_recycles'] or 'AF3 default'}, diffusion samples {d['num_diffusion_samples'] or 'AF3 default'}\")
print('   runs:', ', '.join(r['run'] for r in d['runs']))"
echo ">> Platform $PLATFORM: $MACHINE_TYPE${RUNTIME:+ (runtime $RUNTIME)}, zone $ZONE, provisioning $PROVISIONING"
[ "$VM_API" = "tpu" ] || echo "   image $IMAGE_PROJECT/$IMAGE_FAMILY${MIN_CPU_PLATFORM:+, min CPU platform $MIN_CPU_PLATFORM}"
if [ "$WEIGHTS" = "gcs" ]; then echo "   weights: gcs (copied on the VM from your bucket; URI not printed)"; else echo "   weights: random (generated on the VM)"; fi
echo "   follow deadline: $DETACHED_DEADLINE_MIN min"
if [ "$VM_API" = "tpu" ]; then
  echo "   VM lifetime limit: watchdog deletes it after $WATCHDOG_HOURS h"
else
  echo "   VM lifetime limit: --max-run-duration=$MAX_RUN_DURATION, then Google deletes it"
  [ "$PROVISIONING" != "flex_start" ] || echo "   Flex-start queue: up to $REQUEST_VALID_FOR (not billed while queued)"
fi
echo "   worst-case cost: \$$WORST = \$$PRICE/h x $LIFETIME_H h, list price read $PRICE_DATE"
echo "     ($PRICE_URL; excludes boot disk and network, a few cents)"
if [ -n "$PRICE_NOTES" ]; then echo "$PRICE_NOTES" | sed 's/^/   !! price note: /'; fi
if [ -n "$RESULTS_GCS_URI" ]; then
  echo "   results bucket: $RESULTS_SESSION_URI/ (each target uploaded when it finishes); fetch: $FETCH_EFFECTIVE"
else
  echo "   results bucket: none (RESULTS_GCS_URI not set); fetch: full over SSH (FETCH=light needs the bucket)"
fi
echo "   session: results/af3/$SESSION/   VM: $VM_NAME"
if [ "${YES:-0}" != "1" ]; then
  printf 'Proceed? [y/N] '
  read -r ANSWER || ANSWER=""
  case "$ANSWER" in y | Y | yes | YES) ;; *) echo ">> Not started; nothing was created."; exit 0 ;; esac
fi

# Never adopt (and later delete) a VM this run did not create.
if vm_exists; then die "$VM_NAME already exists in $ZONE. Delete it or set VM_NAME."; fi

CREATE_ISSUED=0 STARTED=0 RUN_DONE=0
cleanup() {
  local rc=$? op
  [ "$rc" -ne 0 ] || [ "$RUN_DONE" = "1" ] || rc=1
  trap - EXIT INT TERM
  set +e
  if [ "$CREATE_ISSUED" = "1" ]; then
    if [ "$STARTED" = "1" ]; then
      results_fetch "$OUT_REL" results/af3 "$FETCH_EFFECTIVE" "$RESULTS_SESSION_URI"
    fi
    echo ">> Deleting $VM_NAME"
    if ! vm_delete && [ "$VM_API" = "gce" ]; then
      op=$(vm_insert_op status)
      if [ "$op" = "PENDING" ] || [ "$op" = "RUNNING" ]; then
        echo "!! The create request for $VM_NAME is still queued and could not be cancelled."
        echo "!! If it is granted, Google deletes the VM after $MAX_RUN_DURATION. To delete it sooner:"
        echo "!!   gcloud compute instances delete $VM_NAME --project=$PROJECT --zone=$ZONE --quiet"
      fi
    fi
  fi
  vm_leftovers
  echo ">> exit code $rc"
  [ ! -d "results/af3/$SESSION" ] || echo ">> results: results/af3/$SESSION/"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LABELS="purpose=af3-run,platform=$PLATFORM,plan=$PLAN_NAME,owner=lorenzo,expires=$EXPIRES"
CREATE_ISSUED=1
if [ "$VM_API" = "tpu" ]; then
  echo ">> Creating $VM_NAME ($MACHINE_TYPE, $RUNTIME) in $ZONE, $PROVISIONING"
  SPOT_FLAG=""
  [ "$PROVISIONING" != "spot" ] || SPOT_FLAG="--spot"
  vm_create_tpu "--accelerator-type=$MACHINE_TYPE --version=$RUNTIME \
--scopes=https://www.googleapis.com/auth/cloud-platform --labels=$LABELS $SPOT_FLAG"
  vm_wait_running "$WAIT_S" || exit 1
  echo ">> Arming watchdog: $VM_NAME deletes itself in $WATCHDOG_HOURS h"
  vm_arm_watchdog "$WATCHDOG_HOURS"
else
  echo ">> Requesting $VM_NAME ($MACHINE_TYPE) in $ZONE, $PROVISIONING"
  FLAGS="--machine-type=$MACHINE_TYPE --image-family=$IMAGE_FAMILY --image-project=$IMAGE_PROJECT \
--boot-disk-size=$BOOT_DISK_SIZE${BOOT_DISK_TYPE:+ --boot-disk-type=$BOOT_DISK_TYPE} \
--provisioning-model=$(echo "$PROVISIONING" | tr 'a-z' 'A-Z') --max-run-duration=$MAX_RUN_DURATION \
--instance-termination-action=DELETE --maintenance-policy=TERMINATE --reservation-affinity=none \
--scopes=cloud-platform --labels=$LABELS"
  [ "$PROVISIONING" != "flex_start" ] || FLAGS="$FLAGS --request-valid-for-duration=$REQUEST_VALID_FOR"
  vm_create_gce "$FLAGS"
  vm_wait_running "$WAIT_S" || exit 1
fi
echo ">> Waiting for SSH"
vm_wait_ssh || exit 1

echo ">> Uploading the plan's files (commit $BENCH_COMMIT)"
TGZ=$(mktemp -t af3-run.XXXXXX)
FILES="af3_tpu harness targets inputs/manifest.csv cloud/vm_af3_setup.sh cloud/vm_af3_weights.sh cloud/vm_af3_job.sh"
for t in $(echo "$TARGETS" | tr ',' ' '); do FILES="$FILES data/inputs/$t.json"; done
tar czf "$TGZ" --exclude=__pycache__ $FILES
vm_upload "$TGZ" af3_run.tgz
rm -f "$TGZ"
vm_ssh "mkdir -p ~/alphafold-on-tpu && tar xzf ~/af3_run.tgz -C ~/alphafold-on-tpu && rm -f ~/af3_run.tgz"
if [ "$WEIGHTS" = "gcs" ]; then
  URI_FILE=$(mktemp -t af3-uri.XXXXXX)
  chmod 600 "$URI_FILE"
  printf '%s' "$WEIGHTS_GCS_URI" > "$URI_FILE"
  vm_upload "$URI_FILE" .af3_weights_uri
  rm -f "$URI_FILE"
  vm_ssh "chmod 600 ~/.af3_weights_uri"
fi

echo ">> Starting cloud/vm_af3_job.sh detached on the VM (setup, then the plan)"
DETACHED_SCRIPT=cloud/vm_af3_job.sh
DETACHED_LOG=job.log
STARTED=1
detached_start "$OUT_REL" \
  "PLATFORM=$PLATFORM PLAN=$PLAN WEIGHTS=$WEIGHTS TARGETS=$TARGETS SESSION=$SESSION RESULTS_URI=$RESULTS_SESSION_URI" \
  || die "Could not start the job on the VM."
if ! detached_wait "$OUT_REL"; then
  [ "$DETACHED_VM_GONE" != "1" ] || die "$VM_NAME was preempted or deleted during the run; cleaning up."
  die "No result from the job."
fi
if [ "$DETACHED_RC" != "0" ]; then
  echo "!! The job exited with $DETACHED_RC (setup: 3 inputs, 4 weights, 5 device;" \
       "plan: 1 some runs failed, all recorded); results are fetched below"
  exit "$DETACHED_RC"
fi
echo ">> Plan finished: every harness run exited 0"
RUN_DONE=1
