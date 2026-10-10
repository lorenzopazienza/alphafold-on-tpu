#!/bin/bash
# Suite 1 (first written 2026-10-08): cloud/af3_run.sh on every platform,
# OOM, dropped and lost SSH, deadline, GCS weights, input mismatches,
# machine-type refusals, the prompt, and Ctrl-C during setup and during the plan.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
for p in cpu l4 v5e v6e; do bash "$R" probe_$p ok PLATFORM=$p FAKE_PLATFORM=$p; done
bash "$R" oom_v5e oom PLATFORM=v5e FAKE_PLATFORM=v5e
bash "$R" drop_v6e drop PLATFORM=v6e FAKE_PLATFORM=v6e
# SSH lost while the API reports the VM running: since 2026-10-10 the launcher
# follows on to the deadline (1 min here) instead of deleting after 3 polls.
bash "$R" lost_l4 lost PLATFORM=l4 FAKE_PLATFORM=l4 ZONE=europe-west4-a DETACHED_MAX_SSH_FAILS=3 DETACHED_DEADLINE_MIN=1 \
  DETACHED_MAX_BACKOFF_S=8
bash "$R" deadline_l4 hang PLATFORM=l4 FAKE_PLATFORM=l4 ZONE=europe-west4-a DETACHED_DEADLINE_MIN=1
bash "$R" gcs_v6e ok PLATFORM=v6e FAKE_PLATFORM=v6e WEIGHTS=gcs WEIGHTS_GCS_URI=gs://private-weights/af3.bin.zst
bash "$R" mismatch_remote tamper_remote PLATFORM=v6e FAKE_PLATFORM=v6e
cp "$WORK/repo/data/inputs/7D5C.json" "$WORK/7D5C.bak"; echo ' ' >> "$WORK/repo/data/inputs/7D5C.json"
bash "$R" mismatch_local ok PLATFORM=v6e FAKE_PLATFORM=v6e
cp "$WORK/7D5C.bak" "$WORK/repo/data/inputs/7D5C.json"
for pm in cpu:a2-highgpu-1g cpu:ct6e-standard-1t l4:a2-highgpu-1g l4:g2-standard-24 v5e:v5litepod-4 v6e:ct6e-standard-4t; do
  bash "$R" refuse_${pm%%:*}_${pm#*:} ok PLATFORM=${pm%%:*} MACHINE_TYPE=${pm#*:}
done
bash "$R" prompt_no ok PLATFORM=v5e ANSWER=n
python3 "$KIT/ctrlc.py" ctrlc_setup v6e "setup: fake install" FAKE_SETUP_S=20
python3 "$KIT/ctrlc.py" ctrlc_run l4 "running the plan" FAKE_RUN_S=20 ZONE=europe-west4-a
