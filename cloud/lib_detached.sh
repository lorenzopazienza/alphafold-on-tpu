# Run af3_tpu/vm_smoke.sh detached on the VM and follow it with short SSH
# calls, so a dropped connection no longer kills the run. Sourced by
# cloud/af3_tpu_smoke.sh and cloud/af3_tpu_smoke_gce.sh, which must define
#
#   vm_ssh "<command>"    run one command on the VM over SSH, return its status
#
# OUT arguments are paths relative to the VM user's home directory.
# Settings (environment):
#   DETACHED_POLL_S         seconds between polls                   (60)
#   DETACHED_MAX_SSH_FAILS  consecutive failed polls before giving up (10)
#   DETACHED_DEADLINE_MIN   minutes before polling stops            (60)
#   DETACHED_SCRIPT         script started on the VM, relative to ~/alphafold-on-tpu
#                           (af3_tpu/vm_smoke.sh)
#   DETACHED_LOG            log in ~/OUT whose last line each poll prints (vm_smoke.log)
# Written for macOS bash 3.2: no arrays, no associative arrays.

DETACHED_POLL_S="${DETACHED_POLL_S:-60}"
DETACHED_MAX_SSH_FAILS="${DETACHED_MAX_SSH_FAILS:-10}"
DETACHED_DEADLINE_MIN="${DETACHED_DEADLINE_MIN:-60}"
DETACHED_SCRIPT="${DETACHED_SCRIPT:-af3_tpu/vm_smoke.sh}"
DETACHED_LOG="${DETACHED_LOG:-vm_smoke.log}"
DETACHED_RC=""

# detached_start OUT "VAR=value ...": start `VAR=value ... bash
# af3_tpu/vm_smoke.sh ~/OUT` in its own session on the VM, from
# ~/alphafold-on-tpu. A wrapper writes ~/OUT/EXIT_CODE when vm_smoke.sh ends,
# whatever the reason, and ~/OUT/PID holds the wrapper's PID. The SSH call
# returns within seconds. Retrying is safe: nothing new is started while a
# run is alive or has finished.
detached_start() {
  local out="$1" env="$2" try
  local cmd="O=\$HOME/$out; mkdir -p \$O && cd \$HOME/alphafold-on-tpu || exit 1
if [ -f \$O/EXIT_CODE ]; then echo \"   already finished\"; exit 0; fi
if [ -f \$O/PID ] && kill -0 \$(cat \$O/PID) 2>/dev/null; then echo \"   already running, pid \$(cat \$O/PID)\"; exit 0; fi
nohup setsid bash -c '$env bash $DETACHED_SCRIPT \"\$0\"; echo \$? > \"\$0/EXIT_CODE.tmp\"; mv \"\$0/EXIT_CODE.tmp\" \"\$0/EXIT_CODE\"' \$O > \$O/nohup.out 2>&1 < /dev/null &
echo \$! > \$O/PID
echo \"   started, pid \$!\""
  for try in 1 2 3; do
    if vm_ssh "$cmd"; then return 0; fi
    echo "   starting over SSH failed (try $try/3)"
    [ "$try" -eq 3 ] || sleep 15
  done
  return 1
}

# detached_wait OUT: poll every DETACHED_POLL_S until ~/OUT/EXIT_CODE exists,
# printing the last line of ~/OUT/vm_smoke.log meanwhile. On success, returns
# 0 with the exit code of vm_smoke.sh in DETACHED_RC. Returns 1 if the run
# vanished without an exit code, after DETACHED_MAX_SSH_FAILS consecutive
# failed polls, or at the deadline.
detached_wait() {
  local out="$1" start now fails=0 reply state line
  local probe="O=\$HOME/$out
if [ ! -f \$O/EXIT_CODE ] && [ -f \$O/PID ] && kill -0 \$(cat \$O/PID) 2>/dev/null; then
  echo '@@ RUNNING'
elif [ -f \$O/EXIT_CODE ]; then
  echo \"@@ EXIT \$(cat \$O/EXIT_CODE)\"
else
  echo '@@ GONE'
fi
tail -n 1 \$O/$DETACHED_LOG 2>/dev/null | cut -c1-160"
  start=$(date +%s)
  echo ">> Following $DETACHED_SCRIPT: a poll every ${DETACHED_POLL_S}s, deadline ${DETACHED_DEADLINE_MIN} min"
  while :; do
    sleep "$DETACHED_POLL_S"
    now=$(date +%s)
    if reply=$(vm_ssh "$probe" 2>/dev/null); then
      state=$(echo "$reply" | grep '^@@ ' | head -1 || true)
      line=$(echo "$reply" | grep -v '^@@ ' | tail -1 || true)
      case "$state" in
        "@@ EXIT "*)
          DETACHED_RC="${state#@@ EXIT }"
          case "$DETACHED_RC" in
            ''|*[!0-9]*) echo "!! Unreadable EXIT_CODE on the VM: '$DETACHED_RC'"; return 1 ;;
          esac
          echo ">> $DETACHED_SCRIPT finished with exit code $DETACHED_RC"
          return 0 ;;
        "@@ RUNNING")
          fails=0
          echo "   $(date -u +%H:%M:%SZ) running | $line" ;;
        "@@ GONE")
          echo "!! $DETACHED_SCRIPT is no longer running on the VM and left no EXIT_CODE"
          return 1 ;;
        *)
          fails=$((fails + 1))
          echo "   $(date -u +%H:%M:%SZ) unreadable poll reply ($fails/$DETACHED_MAX_SSH_FAILS)" ;;
      esac
    else
      fails=$((fails + 1))
      echo "   $(date -u +%H:%M:%SZ) SSH poll failed ($fails/$DETACHED_MAX_SSH_FAILS in a row); the run continues on the VM"
    fi
    if [ "$fails" -ge "$DETACHED_MAX_SSH_FAILS" ]; then
      echo "!! Lost SSH to the VM: $fails polls in a row failed"
      return 1
    fi
    if [ $((now - start)) -ge $((DETACHED_DEADLINE_MIN * 60)) ]; then
      echo "!! Deadline: $DETACHED_SCRIPT still not finished after $DETACHED_DEADLINE_MIN min"
      return 1
    fi
  done
}
