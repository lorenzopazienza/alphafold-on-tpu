#!/usr/bin/env bash
# One-VM bisection of the v5e compile segfault (probe of 2026-10-08: SIGSEGV
# about 1 s into the XLA compile of jit(apply_fn) on v5litepod-1, while the
# smoke test passed on the same VM type). Creates ONE on-demand v5litepod-1,
# arms a 2 h self-delete watchdog, uploads only what the bisection needs, runs
# cloud/vm_v5e_bisect.sh there detached (setup as in the probe, then the
# variants V0 to V9 of cloud/vm_v5e_bisect.py, each in a fresh process),
# follows it with short SSH polls, fetches results/af3/<session>/ and deletes
# the VM, also on error, preemption, deadline or Ctrl-C.
#
# Run from the repo root, in a shell where cloud/env.sh is NOT sourced:
#   bash cloud/v5e_bisect.sh
#
# Overrides: PROJECT ZONE (europe-west4-b) RUNTIME (v2-alpha-tpuv5-lite, as in
# the probe and the smoke test) SPOT (0) BISECT_BUDGET_MIN (80: VM-side time
# for setup and variants; later variants are skipped once it runs out)
# VARIANT_TIMEOUT_MIN (15) DETACHED_DEADLINE_MIN (100: how long the laptop
# follows) RESULTS_GCS_URI (optional results bucket: the VM uploads the
# session at the end, and the laptop falls back to it if the SSH fetch fails)
# VM_NAME YES=1 (no prompt).
#
# The VM is deleted by the EXIT trap (results first) and, independently, by
# the watchdog after 2 h whatever happens to the laptop: the worst case is
# 2 h of on-demand v5e. The script never adopts or deletes a VM it did not
# create, and lists leftovers at the end. macOS bash 3.2: no arrays.
set -euo pipefail

die() { echo "!! $*" >&2; exit 1; }

[ -z "${AF2_COMMIT:-}" ] || die "cloud/env.sh is sourced in this shell. Open a fresh shell; this script has its own defaults."
command -v gcloud > /dev/null || die "gcloud not found."
command -v python3 > /dev/null || die "python3 not found."

PLATFORM=v5e
VM_API=tpu
MACHINE_TYPE=v5litepod-1
PROJECT="${PROJECT:-af2-tpu-benchmark}"
ZONE="${ZONE:-europe-west4-b}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
SPOT="${SPOT:-0}"
WATCHDOG_HOURS=2
BISECT_BUDGET_MIN="${BISECT_BUDGET_MIN:-80}"
VARIANT_TIMEOUT_MIN="${VARIANT_TIMEOUT_MIN:-15}"
# Read before cloud/lib_detached.sh is sourced: it sets its own default of 60.
DETACHED_DEADLINE_MIN="${DETACHED_DEADLINE_MIN:-100}"
RESULTS_GCS_URI="${RESULTS_GCS_URI:-}"
if [ "$SPOT" = "1" ]; then PROVISIONING=spot; else PROVISIONING=standard; fi

for n in BISECT_BUDGET_MIN VARIANT_TIMEOUT_MIN DETACHED_DEADLINE_MIN; do
  eval "v=\$$n"
  [ "$v" -gt 0 ] 2> /dev/null || die "$n must be a whole number of minutes, not '$v'."
done
[ $(( BISECT_BUDGET_MIN + 10 )) -le "$DETACHED_DEADLINE_MIN" ] \
  || die "DETACHED_DEADLINE_MIN ($DETACHED_DEADLINE_MIN) must exceed BISECT_BUDGET_MIN ($BISECT_BUDGET_MIN) by 10 min or more."
[ $(( DETACHED_DEADLINE_MIN + 15 )) -le $(( WATCHDOG_HOURS * 60 )) ] \
  || die "DETACHED_DEADLINE_MIN ($DETACHED_DEADLINE_MIN) must leave 15 min before the ${WATCHDOG_HOURS} h watchdog for the fetch."
if [ -n "$RESULTS_GCS_URI" ]; then
  RESULTS_GCS_URI="${RESULTS_GCS_URI%/}"
  echo "$RESULTS_GCS_URI" | grep -Eq '^gs://[a-z0-9][a-z0-9._-]*[a-z0-9](/[A-Za-z0-9._-]+)*$' \
    || die "RESULTS_GCS_URI must be gs://BUCKET or gs://BUCKET/PREFIX (letters, digits, . _ - /)."
fi

cd "$(git rev-parse --show-toplevel)"
source cloud/lib_vm.sh
source cloud/lib_detached.sh

echo ">> Checking the frozen inputs (7U3J, 7D5C) against inputs/manifest.csv"
python3 harness/verify_inputs.py --targets 7U3J,7D5C \
  || die "Frozen inputs do not match inputs/manifest.csv; nothing was created."
[ -f af3_tpu/inputs/toy_118.json ] || die "af3_tpu/inputs/toy_118.json is missing."

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="${STAMP}_v5e_bisect"
[ ! -e "results/af3/$SESSION" ] || die "results/af3/$SESSION exists; sessions are never overwritten."
# The af3-run- prefix keeps it in vm_leftovers' list.
VM_NAME="${VM_NAME:-af3-run-v5e-bisect-$(echo "$STAMP" | tr 'A-Z' 'a-z')}"
OUT_REL="alphafold-on-tpu/results/af3/$SESSION"
RESULTS_SESSION_URI="${RESULTS_GCS_URI:+$RESULTS_GCS_URI/af3/$SESSION}"
BENCH_COMMIT=$(git rev-parse HEAD)
EXPIRES=$(( $(date +%s) + WATCHDOG_HOURS * 3600 + 900 ))
vm_init

PRICE_LINE=$(python3 - "$PROVISIONING" "${ZONE%-*}" <<'EOF'
import csv, sys
prov, region = sys.argv[1:3]
rows = [r for r in csv.DictReader(open('cloud/prices.csv'))
        if r['platform'] == 'v5e' and r['machine_type'] == 'v5litepod-1' and r['provisioning'] == prov
        and r['usd_per_hour'].strip()]
row = next((r for r in rows if r['region'] == region), rows[0] if rows else None)
if row:
  print(row['usd_per_hour'], row['read_on'], row['region'], row['source_url'])
EOF
)
[ -n "$PRICE_LINE" ] || die "No v5e $PROVISIONING price in cloud/prices.csv."
set -- $PRICE_LINE
PRICE=$1 PRICE_DATE=$2 PRICE_REGION=$3 PRICE_URL=$4
WORST=$(python3 -c "print(f'{$PRICE * $WATCHDOG_HOURS:.2f}')")
TYPICAL=$(python3 -c "print(f'{$PRICE * 40 / 60:.2f} to \${$PRICE * 1.5:.2f}')")

echo
echo ">> v5e bisection: one $MACHINE_TYPE ($RUNTIME) in $ZONE, $PROVISIONING"
echo "   variants, in order (each AlphaFold3 run in a fresh process, ${VARIANT_TIMEOUT_MIN} min timeout):"
python3 cloud/vm_v5e_bisect.py --list | sed 's/^/     /'
echo "   libtpu verbose logging (TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0) in V1 to V9; HLO dump in V3/V3p"
echo "   per variant: exit code, signal, last 50 log lines, dmesg tail, /tmp/tpu_logs (passing ones too)"
echo "   budget on the VM: $BISECT_BUDGET_MIN min from job start, setup included (4.5 min in the probe);"
echo "     later variants are skipped once it runs out"
echo "   follow deadline: $DETACHED_DEADLINE_MIN min; VM lifetime limit: watchdog deletes it after $WATCHDOG_HOURS h"
echo "   worst-case cost: \$$WORST = \$$PRICE/h x $WATCHDOG_HOURS h, list price read $PRICE_DATE"
echo "     ($PRICE_URL; excludes boot disk and network, a few cents)"
[ "$PRICE_REGION" = "${ZONE%-*}" ] || echo "   !! price note: no v5e price for ${ZONE%-*} in cloud/prices.csv: the price shown is the $PRICE_REGION list price"
echo "   expected: about 40 to 90 min of VM time (crashing variants take about 1 min each), \$$TYPICAL"
if [ -n "$RESULTS_GCS_URI" ]; then echo "   results bucket: $RESULTS_SESSION_URI/ (uploaded at the end)"; else echo "   results bucket: none (fetch over SSH)"; fi
echo "   session: results/af3/$SESSION/   VM: $VM_NAME"
if [ "${YES:-0}" != "1" ]; then
  printf 'Proceed? [y/N] '
  read -r ANSWER || ANSWER=""
  case "$ANSWER" in y | Y | yes | YES) ;; *) echo ">> Not started; nothing was created."; exit 0 ;; esac
fi

if vm_exists; then die "$VM_NAME already exists in $ZONE. Delete it or set VM_NAME."; fi

CREATE_ISSUED=0 STARTED=0 RUN_DONE=0
cleanup() {
  local rc=$?
  [ "$rc" -ne 0 ] || [ "$RUN_DONE" = "1" ] || rc=1
  trap - EXIT INT TERM
  set +e
  if [ "$CREATE_ISSUED" = "1" ]; then
    if [ "$STARTED" = "1" ]; then
      results_fetch "$OUT_REL" results/af3 full "$RESULTS_SESSION_URI"
    fi
    echo ">> Deleting $VM_NAME"
    vm_delete
  fi
  vm_leftovers
  if [ -f "results/af3/$SESSION/bisect_summary.txt" ]; then
    echo ">> Bisection summary (results/af3/$SESSION/bisect_summary.txt):"
    sed 's/^/   /' "results/af3/$SESSION/bisect_summary.txt"
  fi
  echo ">> exit code $rc"
  [ ! -d "results/af3/$SESSION" ] || echo ">> results: results/af3/$SESSION/"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LABELS="purpose=af3-bisect,platform=v5e,owner=lorenzo,expires=$EXPIRES"
CREATE_ISSUED=1
echo ">> Creating $VM_NAME ($MACHINE_TYPE, $RUNTIME) in $ZONE, $PROVISIONING"
SPOT_FLAG=""
[ "$PROVISIONING" != "spot" ] || SPOT_FLAG="--spot"
vm_create_tpu "--accelerator-type=$MACHINE_TYPE --version=$RUNTIME \
--scopes=https://www.googleapis.com/auth/cloud-platform --labels=$LABELS $SPOT_FLAG"
vm_wait_running 900 || exit 1
echo ">> Arming watchdog: $VM_NAME deletes itself in $WATCHDOG_HOURS h"
vm_arm_watchdog "$WATCHDOG_HOURS"
echo ">> Waiting for SSH"
vm_wait_ssh || exit 1

echo ">> Uploading the bisection's files (commit $BENCH_COMMIT)"
TGZ=$(mktemp -t af3-bisect.XXXXXX)
tar czf "$TGZ" --exclude=__pycache__ af3_tpu harness targets inputs/manifest.csv \
  cloud/vm_af3_setup.sh cloud/vm_af3_weights.sh cloud/vm_v5e_bisect.sh cloud/vm_v5e_bisect.py \
  data/inputs/7U3J.json data/inputs/7D5C.json
vm_upload "$TGZ" af3_bisect.tgz
rm -f "$TGZ"
vm_ssh "mkdir -p ~/alphafold-on-tpu && tar xzf ~/af3_bisect.tgz -C ~/alphafold-on-tpu && rm -f ~/af3_bisect.tgz"

echo ">> Starting cloud/vm_v5e_bisect.sh detached on the VM (setup, then the variants)"
DETACHED_SCRIPT=cloud/vm_v5e_bisect.sh
DETACHED_LOG=bisect.log
STARTED=1
detached_start "$OUT_REL" \
  "SESSION=$SESSION BISECT_BUDGET_MIN=$BISECT_BUDGET_MIN VARIANT_TIMEOUT_MIN=$VARIANT_TIMEOUT_MIN RESULTS_URI=$RESULTS_SESSION_URI" \
  || die "Could not start the bisection on the VM."
if ! detached_wait "$OUT_REL"; then
  [ "$DETACHED_VM_GONE" != "1" ] || die "$VM_NAME was preempted or deleted during the bisection; cleaning up."
  die "No result from the bisection."
fi
if [ "$DETACHED_RC" != "0" ]; then
  echo "!! The bisection job exited with $DETACHED_RC (setup: 3 inputs, 4 weights, 5 device;" \
       "2 internal error); results are fetched below"
  exit "$DETACHED_RC"
fi
echo ">> Bisection finished"
RUN_DONE=1
