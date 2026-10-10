# Run af3_tpu/vm_smoke.sh detached on the VM and follow it with short SSH
# calls, so a dropped connection no longer kills the run. Sourced by
# cloud/af3_tpu_smoke.sh and cloud/af3_tpu_smoke_gce.sh, which must define
#
#   vm_ssh "<command>"    run one command on the VM over SSH, return its status
#
# OUT arguments are paths relative to the VM user's home directory.
# Settings (environment):
#   DETACHED_POLL_S         seconds between polls                   (60)
#   DETACHED_MAX_BACKOFF_S  longest wait between polls while SSH or the
#                           network is down (300; never past the deadline)
#   DETACHED_MAX_SSH_FAILS  consecutive failed polls before giving up, only
#                           when the caller cannot ask the API (10; see below)
#   DETACHED_DEADLINE_MIN   minutes before polling stops            (60)
#   DETACHED_SCRIPT         script started on the VM, relative to ~/alphafold-on-tpu
#                           (af3_tpu/vm_smoke.sh)
#   DETACHED_LOG            log in ~/OUT whose last line each poll prints (vm_smoke.log)
#   DETACHED_EVENTS         optional local file: one timestamped line per
#                           state change (connection lost and back, VM gone,
#                           job finished, deadline), for the session's record
#
# Failed polls: if the caller also defines vm_state and vm_state_is_gone
# (cloud/lib_vm.sh does), every failed poll asks the API for the VM's state
# and tells three cases apart:
#   - the API cannot be reached (vm_state UNKNOWN: this laptop has no network,
#     or Google's API is unreachable): nothing is known about the VM, so
#     detached_wait keeps retrying with backoff until the deadline;
#   - the VM is preempted, stopped, being deleted or gone: detached_wait
#     returns 1 at once with DETACHED_VM_GONE=1;
#   - the VM is up but SSH fails: the job goes on, so detached_wait keeps
#     retrying with backoff until the deadline; if the caller defines
#     detached_exit_elsewhere (prints the job's exit code read another way,
#     for example from the results bucket, or nothing), a finished job is
#     detected without SSH.
# It prints one line when the state changes, not one per retry. Only callers
# without vm_state give up after DETACHED_MAX_SSH_FAILS failed polls, as before.
# Return status: 0 when the job finished (DETACHED_RC, DETACHED_RC_VIA ssh or
# elsewhere); 1 otherwise, with DETACHED_END set to deadline, vm_gone,
# job_gone (no process and no EXIT_CODE, seen twice), unreadable or ssh_lost.
# Written for macOS bash 3.2: no arrays, no associative arrays.

DETACHED_POLL_S="${DETACHED_POLL_S:-60}"
DETACHED_MAX_BACKOFF_S="${DETACHED_MAX_BACKOFF_S:-300}"
DETACHED_MAX_SSH_FAILS="${DETACHED_MAX_SSH_FAILS:-10}"
DETACHED_DEADLINE_MIN="${DETACHED_DEADLINE_MIN:-60}"
DETACHED_SCRIPT="${DETACHED_SCRIPT:-af3_tpu/vm_smoke.sh}"
DETACHED_LOG="${DETACHED_LOG:-vm_smoke.log}"
DETACHED_RC=""
DETACHED_RC_VIA=""
DETACHED_END=""
DETACHED_VM_GONE=0

# detached_event TEXT: appends "UTC-time  TEXT" to DETACHED_EVENTS (if set).
detached_event() {
  [ -n "${DETACHED_EVENTS:-}" ] || return 0
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  $*" >> "$DETACHED_EVENTS"
}

# detached_classify: after a failed poll, prints "CLASS STATE": net (the API
# cannot be reached), gone (the VM is preempted, stopped, being deleted or
# missing), up (the API reports it in any other state) or none (the caller
# has no vm_state), followed by the VM's state from the API.
DETACHED_VM_STATE=""
detached_classify() {
  local s
  type vm_state > /dev/null 2>&1 && type vm_state_is_gone > /dev/null 2>&1 || { echo "none -"; return; }
  s=$(vm_state)
  if [ "$s" = "UNKNOWN" ]; then echo "net $s"
  elif vm_state_is_gone "$s"; then echo "gone $s"
  else echo "up $s"
  fi
}

# detached_start OUT "VAR=value ...": start `VAR=value ... bash
# af3_tpu/vm_smoke.sh ~/OUT` in its own session on the VM, from
# ~/alphafold-on-tpu. A wrapper writes ~/OUT/EXIT_CODE when vm_smoke.sh ends,
# whatever the reason, and ~/OUT/PID holds the wrapper's PID. If the VM also
# has cloud/vm_job_finish.sh, the wrapper then runs it with the same
# variables (it writes final_status.json and copies both files to the results
# bucket); ~/OUT/.wrapper_done marks the end of all that. The SSH call
# returns within seconds. Retrying is safe: nothing new is started while a
# run is alive or has finished.
detached_start() {
  local out="$1" env="$2" try
  local cmd="O=\$HOME/$out; mkdir -p \$O && cd \$HOME/alphafold-on-tpu || exit 1
if [ -f \$O/EXIT_CODE ]; then echo \"   already finished\"; exit 0; fi
if [ -f \$O/PID ] && kill -0 \$(cat \$O/PID) 2>/dev/null; then echo \"   already running, pid \$(cat \$O/PID)\"; exit 0; fi
nohup setsid bash -c '$env bash $DETACHED_SCRIPT \"\$0\"; echo \$? > \"\$0/EXIT_CODE.tmp\"; mv \"\$0/EXIT_CODE.tmp\" \"\$0/EXIT_CODE\"; [ ! -f cloud/vm_job_finish.sh ] || $env bash cloud/vm_job_finish.sh \"\$0\"; touch \"\$0/.wrapper_done\"' \$O > \$O/nohup.out 2>&1 < /dev/null &
echo \$! > \$O/PID
echo \"   started, pid \$!\""
  for try in 1 2 3; do
    if vm_ssh "$cmd"; then return 0; fi
    echo "   starting over SSH failed (try $try/3)"
    [ "$try" -eq 3 ] || sleep 15
  done
  return 1
}

# detached_wait OUT: poll until ~/OUT/EXIT_CODE exists, printing the last
# line of ~/OUT/$DETACHED_LOG meanwhile (see the header for failed polls).
detached_wait() {
  local out="$1" start now fails=0 reply state line link=ok since since_txt wait_s left gone_seen=0 cls rc
  # Finished once the wrapper is done (EXIT_CODE written and, where there is
  # one, the finish step over); a VM started by an older wrapper has no
  # .wrapper_done, and its EXIT_CODE counts once the wrapper has exited.
  local probe="O=\$HOME/$out
if [ ! -f \$O/.wrapper_done ] && [ -f \$O/PID ] && kill -0 \$(cat \$O/PID) 2>/dev/null; then
  echo '@@ RUNNING'
elif [ -f \$O/EXIT_CODE ]; then
  echo \"@@ EXIT \$(cat \$O/EXIT_CODE)\"
else
  echo '@@ GONE'
fi
tail -n 1 \$O/$DETACHED_LOG 2>/dev/null | cut -c1-160"
  start=$(date +%s)
  since=$start since_txt=$(date -u +%H:%M:%SZ)
  wait_s=$DETACHED_POLL_S
  DETACHED_END="" DETACHED_RC="" DETACHED_RC_VIA=""
  echo ">> Following $DETACHED_SCRIPT: a poll every ${DETACHED_POLL_S}s, deadline ${DETACHED_DEADLINE_MIN} min"
  detached_event "follow: started, deadline $DETACHED_DEADLINE_MIN min"
  while :; do
    left=$(( start + DETACHED_DEADLINE_MIN * 60 - $(date +%s) ))
    [ "$left" -gt 0 ] || left=0
    [ "$wait_s" -le "$left" ] || wait_s=$left
    sleep "$wait_s"
    now=$(date +%s)
    state=""
    if reply=$(vm_ssh "$probe" 2>/dev/null); then
      state=$(echo "$reply" | grep '^@@ ' | head -1 || true)
      line=$(echo "$reply" | grep -v '^@@ ' | tail -1 || true)
    fi
    case "$state" in
      "@@ EXIT "* | "@@ RUNNING" | "@@ GONE")
        if [ "$link" != ok ]; then
          echo ">> $(date -u +%H:%M:%SZ) connection to the VM is back after $(( (now - since + 30) / 60 )) min ($link); following again"
          detached_event "connection: back after $(( now - since )) s ($link down since $since_txt)"
          link=ok since=$now since_txt=$(date -u +%H:%M:%SZ)
        fi
        fails=0 wait_s=$DETACHED_POLL_S ;;
    esac
    case "$state" in
      "@@ EXIT "*)
        DETACHED_RC="${state#@@ EXIT }"
        case "$DETACHED_RC" in
          ''|*[!0-9]*) echo "!! Unreadable EXIT_CODE on the VM: '$DETACHED_RC'"; DETACHED_END=unreadable
            detached_event "job: unreadable EXIT_CODE '$DETACHED_RC'"; return 1 ;;
        esac
        DETACHED_RC_VIA=ssh
        echo ">> $DETACHED_SCRIPT finished with exit code $DETACHED_RC"
        detached_event "job: finished, exit code $DETACHED_RC (read over SSH)"
        return 0 ;;
      "@@ RUNNING")
        gone_seen=0
        echo "   $(date -u +%H:%M:%SZ) running | $line" ;;
      "@@ GONE")
        if [ "$gone_seen" = 1 ]; then
          echo "!! $DETACHED_SCRIPT is no longer running on the VM and left no EXIT_CODE"
          DETACHED_END=job_gone
          detached_event "job: not running and no EXIT_CODE (seen on two polls)"
          return 1
        fi
        gone_seen=1 ;;
      *)
        # No usable reply: tell the cases apart (header).
        fails=$((fails + 1))
        cls=$(detached_classify)
        DETACHED_VM_STATE=${cls#* }
        cls=${cls%% *}
        case "$cls" in
          gone)
            echo "!! The VM is $DETACHED_VM_STATE (preempted, stopped or deleted): the run cannot go on; stopping now"
            detached_event "vm: $DETACHED_VM_STATE according to the API; following stops"
            DETACHED_VM_GONE=1 DETACHED_END=vm_gone
            return 1 ;;
          none)
            echo "   $(date -u +%H:%M:%SZ) SSH poll failed ($fails/$DETACHED_MAX_SSH_FAILS in a row); the run continues on the VM"
            if [ "$fails" -ge "$DETACHED_MAX_SSH_FAILS" ]; then
              echo "!! Lost SSH to the VM: $fails polls in a row failed"
              DETACHED_END=ssh_lost
              detached_event "follow: gave up after $fails failed SSH polls (no API state available)"
              return 1
            fi ;;
          net | up)
            if [ "$cls" = up ] && type detached_exit_elsewhere > /dev/null 2>&1; then
              rc=$(detached_exit_elsewhere 2> /dev/null || true)
              case "$rc" in
                '' | *[!0-9]*) ;;
                *)
                  DETACHED_RC=$rc DETACHED_RC_VIA=elsewhere
                  echo ">> $DETACHED_SCRIPT finished with exit code $rc (read from the results bucket; SSH is down)"
                  detached_event "job: finished, exit code $rc (read from the results bucket; SSH down)"
                  return 0 ;;
              esac
            fi
            if [ "$cls" = net ]; then cls=network; else cls=ssh; fi
            if [ "$link" != "$cls" ]; then
              if [ "$cls" = network ]; then
                echo "!! $(date -u +%H:%M:%SZ) no network or Google API unreachable from this laptop: the VM is left alone;" \
                     "retrying every ${DETACHED_POLL_S}s to ${DETACHED_MAX_BACKOFF_S}s until the deadline"
                detached_event "connection: lost (no network or API unreachable from the laptop)"
              else
                echo "!! $(date -u +%H:%M:%SZ) SSH to the VM fails while the API reports it $DETACHED_VM_STATE: the job goes on;" \
                     "retrying every ${DETACHED_POLL_S}s to ${DETACHED_MAX_BACKOFF_S}s until the deadline"
                detached_event "connection: SSH fails, VM $DETACHED_VM_STATE according to the API"
              fi
              if [ "$link" = ok ]; then since=$now since_txt=$(date -u +%H:%M:%SZ); fi
              link=$cls
            fi
            wait_s=$(( wait_s * 2 ))
            [ "$wait_s" -le "$DETACHED_MAX_BACKOFF_S" ] || wait_s=$DETACHED_MAX_BACKOFF_S ;;
        esac ;;
    esac
    if [ $((now - start)) -ge $((DETACHED_DEADLINE_MIN * 60)) ]; then
      echo "!! Deadline: $DETACHED_SCRIPT still not finished after $DETACHED_DEADLINE_MIN min"
      DETACHED_END=deadline
      detached_event "follow: deadline of $DETACHED_DEADLINE_MIN min reached$([ "$link" = ok ] || echo " ($link down since $since_txt)")"
      return 1
    fi
  done
}
