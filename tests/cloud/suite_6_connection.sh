#!/bin/bash
# Suite 6 (first written 2026-10-10): connection loss and resume, after the
# 2026-10-09 incident (the laptop's network dropped for about 40 min; one
# launcher deleted its VM in the middle of a run). The launcher must tell a
# laptop without network from a VM that is gone and from a VM whose SSH
# fails; never delete a running VM whose job has not finished, except at the
# deadline or on Ctrl-C; print one line per state change; write
# launcher_events.txt; put EXIT_CODE and final_status.json in the results
# bucket; and re-attach to a running VM (resume mode) only if it is the
# session's VM. The network loss is scaled down: seconds stand for minutes.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
B=gs://pilot-results/study
events() {   # the session's launcher_events.txt, and what the fake refused during the loss
  local f
  f=$(ls "$WORK/state_$1/fetched/results/af3/"*/launcher_events.txt 2> /dev/null | head -1)
  [ -n "$f" ] || f=$(ls "$WORK/state_$1/fetched/results/af3/.launcher/"*/launcher_events.txt 2> /dev/null | head -1)
  if [ -n "$f" ]; then echo "   launcher_events.txt:"; sed 's/^/   | /' "$f"; else echo "   (no launcher_events.txt)"; fi
  if [ -f "$WORK/state_$1/net_refused" ]; then
    echo "   laptop gcloud calls refused during the loss: $(grep -c refused: "$WORK/state_$1/net_refused")," \
         "of which delete: $(grep -c 'refused: gcloud .* delete ' "$WORK/state_$1/net_refused")"
  fi
  echo "   state-change lines printed: $(grep -cE '^!! [0-9:]+Z (no network|SSH to the VM fails)|connection to the VM is back' "$WORK/state_$1/out")," \
       "per-retry lines: $(grep -c 'SSH poll failed' "$WORK/state_$1/out")"
}
bucket() {   # EXIT_CODE and final_status.json in the session's bucket folder
  local d
  d=$(ls -d "$WORK/state_$1/bucket/pilot-results/study/af3/"*/ 2> /dev/null | head -1)
  [ -n "$d" ] || { echo "   (no bucket folder)"; return; }
  echo "   bucket: EXIT_CODE=$(cat "$d/EXIT_CODE" 2> /dev/null || echo missing);" \
       "final_status.json: $(cat "$d/final_status.json" 2> /dev/null || echo missing)"
  echo "   bucket: launcher records: $(ls "$d" | grep -E '^(launch_record|launcher_events|uploaded_files)\.txt$' | tr '\n' ' ')"
}
echo "======== network loss while the VM runs: no deletion, the follow resumes"
bash "$R" net_loss_running ok PLATFORM=v5e FAKE_PLATFORM=v5e FAKE_RUN_S=8 FAKE_NET_DOWN=3:12 \
  DETACHED_MAX_BACKOFF_S=4 RESULTS_GCS_URI=$B FETCH=full
events net_loss_running
echo "======== network loss past the deadline: the deadline is the only reason to delete; it waits for the API"
bash "$R" net_loss_deadline ok PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_RUN_S=60 FAKE_NET_DOWN=2:75 \
  DETACHED_DEADLINE_MIN=1 DETACHED_MAX_BACKOFF_S=8 CLEANUP_WAIT_MIN=2 RESULTS_GCS_URI=$B
events net_loss_deadline
echo "======== Google deletes the VM during the loss: seen as gone when the network is back; bucket fetch"
bash "$R" net_loss_vm_gone ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONE=us-east4-a FAKE_RUN_S=6 FAKE_NET_DOWN=9:8 FAKE_NET_GONE=1 \
  DETACHED_MAX_BACKOFF_S=4 RESULTS_GCS_URI=$B FETCH=full
events net_loss_vm_gone
echo "======== EXIT_CODE in the bucket; with SSH lost while the VM is up, the end is read from the bucket"
bash "$R" exit_code_bucket ok PLATFORM=cpu FAKE_PLATFORM=cpu RESULTS_GCS_URI=$B FETCH=full
bucket exit_code_bucket
bash "$R" exit_code_bucket_ssh_lost lost PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_RUN_S=3 DETACHED_MAX_BACKOFF_S=4 \
  RESULTS_GCS_URI=$B FETCH=full
bucket exit_code_bucket_ssh_lost
events exit_code_bucket_ssh_lost
echo "======== resume mode: the launcher killed (SIGKILL) while the job runs, then RESUME_VM"
python3 "$KIT/resume.py" resume_record v6e record FAKE_RUN_S=4 RESULTS_GCS_URI=$B FETCH=full
python3 "$KIT/resume.py" resume_legacy v5e legacy FAKE_RUN_S=4 RESULTS_GCS_URI=$B FETCH=full
