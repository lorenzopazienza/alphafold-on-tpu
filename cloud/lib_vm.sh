# Shared VM lifecycle for cloud/af3_run.sh: create, wait, SSH, copy, delete
# and the leftover check, for Compute Engine VMs (VM_API=gce: CPU, L4, v6e)
# and legacy Cloud TPU VMs (VM_API=tpu: v5e). Sourced after setting VM_API,
# VM_NAME, PROJECT and ZONE, then calling vm_init. Defines vm_ssh, which
# cloud/lib_detached.sh needs. macOS bash 3.2: no arrays; flag strings hold
# no values with spaces (the one that does, --min-cpu-platform, is quoted
# separately).

vm_init() {
  GC="--project=$PROJECT --zone=$ZONE"
  case "$VM_API" in
    gce) VM_RES="compute instances"; VM_SSH_GROUP="compute" ;;
    tpu) VM_RES="compute tpus tpu-vm"; VM_SSH_GROUP="compute tpus tpu-vm" ;;
    *) echo "!! VM_API must be gce or tpu" >&2; return 1 ;;
  esac
}

vm_exists() { gcloud $VM_RES describe "$VM_NAME" $GC > /dev/null 2>&1; }

vm_ssh() {
  gcloud $VM_SSH_GROUP ssh "$VM_NAME" $GC --ssh-flag=-oServerAliveInterval=30 \
    --ssh-flag=-oConnectTimeout=30 --command "$1"
}

# vm_upload LOCAL_FILE REMOTE_PATH (remote path relative to the VM home)
vm_upload() { gcloud $VM_SSH_GROUP scp "$1" "$VM_NAME:$2" $GC; }

# vm_fetch REMOTE_DIR LOCAL_PARENT: copies the remote folder into LOCAL_PARENT.
vm_fetch() { gcloud $VM_SSH_GROUP scp --recurse "$VM_NAME:$1" "$2" $GC; }

vm_delete() { gcloud $VM_RES delete "$VM_NAME" $GC --quiet; }

# Status of the Compute Engine insert operation: PENDING, RUNNING, DONE or empty.
vm_insert_op() {
  gcloud compute operations list --project="$PROJECT" --zones="$ZONE" \
    --filter="targetLink~/instances/$VM_NAME\$ AND operationType=insert" \
    --format="value($1)" 2> /dev/null | head -1
}

# vm_create_gce "FLAGS": Compute Engine create, asynchronous (vm_wait_running
# follows). MIN_CPU_PLATFORM, if set, is passed as its own quoted flag.
vm_create_gce() {
  if [ -n "${MIN_CPU_PLATFORM:-}" ]; then
    gcloud compute instances create "$VM_NAME" $GC $1 --min-cpu-platform="$MIN_CPU_PLATFORM" --async
  else
    gcloud compute instances create "$VM_NAME" $GC $1 --async
  fi
}

# vm_create_tpu "FLAGS": legacy TPU VM create (returns when the VM is ready).
vm_create_tpu() { gcloud compute tpus tpu-vm create "$VM_NAME" $GC $1; }

# vm_wait_running MAX_WAIT_S: until the VM runs. Status printed at most once a
# minute. Returns 1 with a message if the request fails or times out.
vm_wait_running() {
  local max="$1" start now status err last=0
  start=$(date +%s)
  while :; do
    now=$(date +%s)
    if [ "$VM_API" = "tpu" ]; then
      status=$(gcloud compute tpus tpu-vm describe "$VM_NAME" $GC --format="value(state)" 2> /dev/null || true)
      case "$status" in
        READY) echo ">> $VM_NAME is READY"; return 0 ;;
        PREEMPTED | TERMINATED | STOPPED) echo "!! $VM_NAME reached state $status"; return 1 ;;
      esac
    else
      status=$(gcloud compute instances describe "$VM_NAME" $GC --format="value(status)" 2> /dev/null || true)
      case "$status" in
        RUNNING) echo ">> $VM_NAME is RUNNING after $(( (now - start) / 60 )) min"; return 0 ;;
        STOPPING | STOPPED | SUSPENDING | SUSPENDED | TERMINATED)
          echo "!! $VM_NAME reached status $status before running"; return 1 ;;
      esac
      if [ "$(vm_insert_op status)" = "DONE" ]; then
        err=$(vm_insert_op "error.errors[0].message")
        if [ -n "$err" ]; then echo "!! create request failed: $err"; return 1; fi
      fi
    fi
    if [ $(( now - start )) -gt "$max" ]; then
      echo "!! $VM_NAME not running $(( (now - start) / 60 )) min after the request"; return 1
    fi
    if [ $(( now - last )) -ge 60 ]; then
      echo "   $(date -u +%H:%M:%SZ) status=${status:-not created yet}"
      last=$now
    fi
    sleep 15
  done
}

vm_wait_ssh() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if gcloud $VM_SSH_GROUP ssh "$VM_NAME" $GC --quiet --command=true > /dev/null 2>&1; then return 0; fi
    sleep 15
  done
  echo "!! SSH to $VM_NAME not reachable after 5 min"
  return 1
}

# vm_arm_watchdog HOURS (legacy TPU VMs): the VM deletes itself after HOURS,
# whatever happens to the laptop (as in cloud/af3_tpu_smoke.sh).
vm_arm_watchdog() {
  vm_ssh "set -e
  G=\$(command -v gcloud || true)
  if [ -z \"\$G\" ]; then echo 'NO GCLOUD ON VM: watchdog NOT armed' >&2; exit 3; fi
  sudo systemd-run --unit=af3-watchdog --on-active=${1}h \
    \"\$G\" compute tpus tpu-vm delete $VM_NAME --project=$PROJECT --zone=$ZONE --quiet
  systemctl list-timers af3-watchdog* --no-pager"
}

# Lists every af3-run VM still in the project (Compute Engine, all zones) or
# zone (legacy TPU), or "none". A gcloud --filter that matches nothing prints a
# warning, so names are filtered here instead.
vm_leftovers() {
  local all left
  echo ">> af3-run VMs left (anything listed here is billing or queued):"
  if [ "$VM_API" = "tpu" ]; then
    all=$(gcloud compute tpus tpu-vm list $GC --format="value(name,acceleratorType,state)") || {
      echo "!! could not list TPU VMs; check the Cloud console"; return 0; }
  else
    all=$(gcloud compute instances list --project="$PROJECT" \
      --format="value(name,zone.basename(),machineType.basename(),status)") || {
      echo "!! could not list instances; check the Cloud console"; return 0; }
  fi
  left=$(echo "$all" | awk '$1 ~ /^af3-run-/')
  echo "${left:-   none}"
}
