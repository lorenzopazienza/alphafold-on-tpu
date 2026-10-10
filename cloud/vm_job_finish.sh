#!/usr/bin/env bash
# Runs on the VM when the detached job has ended (cloud/lib_detached.sh's
# wrapper, right after it wrote OUT/EXIT_CODE), with the job's variables:
#   [RESULTS_URI=gs://...] [SESSION=...] [PLATFORM=...] bash cloud/vm_job_finish.sh OUT
# Writes OUT/final_status.json and, if RESULTS_URI is set, copies
# final_status.json, nohup.out and EXIT_CODE (last, so its presence in the
# bucket means the others are there) to RESULTS_URI/. The launcher reads
# EXIT_CODE from the bucket when it cannot reach the VM over SSH, and a
# person can see from the bucket alone how the job ended. Three tries per
# copy; a failure goes to OUT/finish.log and never changes EXIT_CODE.
# macOS bash 3.2 compatible (the fake VM of tests/cloud runs it there).
set -u
OUT="$1"
RESULTS_URI="${RESULTS_URI:-}"
rc=$(cat "$OUT/EXIT_CODE" 2> /dev/null || true)
case "$rc" in '' | *[!0-9]*) rc_json="null" ;; *) rc_json=$rc ;; esac
uploaded=false
[ ! -f "$OUT/.uploaded" ] || uploaded=true
printf '{"session": "%s", "platform": "%s", "exit_code": %s, "finished_utc": "%s", "final_upload_complete": %s}\n' \
  "${SESSION:-}" "${PLATFORM:-}" "$rc_json" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$uploaded" > "$OUT/final_status.json"
[ -n "$RESULTS_URI" ] || exit 0

TO=""
command -v timeout > /dev/null && TO="timeout 120"
copy() {   # copy FILE: to RESULTS_URI/, three tries
  local try
  for try in 1 2 3; do
    if $TO gcloud storage cp "$OUT/$1" "$RESULTS_URI/$1" >> "$OUT/finish.log" 2>&1; then return 0; fi
    sleep 5
  done
  echo "!! could not copy $1 to the results bucket (finish.log)" >> "$OUT/finish.log"
  return 1
}
copy final_status.json
copy nohup.out
copy EXIT_CODE
