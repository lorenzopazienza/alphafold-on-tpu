#!/bin/bash
# Suite 2 (first written 2026-10-08): preemption, results bucket, FETCH
# light/full, L4 Flex-start, price by region, provisioning refusals, and the
# v5e bisection (V0 to V9) on a fake VM, with LIBTPU_VERSION=0.0.42.1 (the
# stack that segfaults; the fake reproduces that).
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
B=gs://af3-results-test/runs
polls() { echo "   SSH polls that failed: $(grep -c 'SSH poll failed' "$WORK/state_$1/out")"; }
bash "$KIT/mkrepo.sh"
echo "======== preemption"
bash "$R" preempt_v5e preempt PLATFORM=v5e FAKE_PLATFORM=v5e FAKE_RUN_S=5; polls preempt_v5e
bash "$R" preempt_gone_l4_bucket preempt_gone PLATFORM=l4 ZONE=europe-west4-a FAKE_PLATFORM=l4 FAKE_RUN_S=2 FAKE_PREEMPT_POLL=6 RESULTS_GCS_URI=$B; polls preempt_gone_l4_bucket
bash "$R" preempt_v6e preempt PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_RUN_S=5; polls preempt_v6e
echo "======== results bucket and FETCH"
bash "$R" bucket_light_ssh ok PLATFORM=v5e FAKE_PLATFORM=v5e RESULTS_GCS_URI=$B
bash "$R" bucket_full_ssh ok PLATFORM=v5e FAKE_PLATFORM=v5e RESULTS_GCS_URI=$B FETCH=full
bash "$R" fetchfail_light fetchfail PLATFORM=v6e FAKE_PLATFORM=v6e RESULTS_GCS_URI=$B/ FETCH=light
bash "$R" fetchfail_full fetchfail PLATFORM=v6e FAKE_PLATFORM=v6e RESULTS_GCS_URI=$B FETCH=full
bash "$R" fetchfail_nobucket fetchfail PLATFORM=v6e FAKE_PLATFORM=v6e
bash "$R" light_without_bucket ok PLATFORM=cpu FAKE_PLATFORM=cpu FETCH=light
bash "$R" uploadfail uploadfail PLATFORM=l4 ZONE=europe-west4-a FAKE_PLATFORM=l4 RESULTS_GCS_URI=$B
bash "$R" refuse_fetch ok PLATFORM=l4 FETCH=some
bash "$R" refuse_bucket_uri ok PLATFORM=l4 RESULTS_GCS_URI=s3://x
bash "$R" refuse_same_bucket ok PLATFORM=l4 WEIGHTS=gcs WEIGHTS_GCS_URI=gs://af3-results-test/w.bin.zst RESULTS_GCS_URI=$B
echo "======== L4 Flex-start, price by region, provisioning refusals"
bash "$R" l4_flex ok PLATFORM=l4 ZONE=europe-west4-a FAKE_PLATFORM=l4 PROVISIONING=flex_start
echo "   create: $(tr ' ' '\n' < "$WORK/state_l4_flex/create_args" | grep -E 'provisioning|valid-for|max-run|termination|maintenance|reservation|machine-type' | tr '\n' ' ')"
bash "$R" l4_flex_us ok PLATFORM=l4 FAKE_PLATFORM=l4 PROVISIONING=flex_start ZONE=us-central1-a REQUEST_VALID_FOR=90m
bash "$R" l4_spot_us ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONE=us-central1-b
bash "$R" l4_spot_west1 ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONE=europe-west1-b
bash "$R" l4_g2_4 ok PLATFORM=l4 ZONE=europe-west4-a FAKE_PLATFORM=l4 MACHINE_TYPE=g2-standard-4 SPOT=0
bash "$R" v6e_default ok PLATFORM=v6e FAKE_PLATFORM=v6e
echo "   create: $(tr ' ' '\n' < "$WORK/state_v6e_default/create_args" | grep -E 'provisioning|valid-for' | tr '\n' ' ')"
for pv in v5e:flex_start cpu:flex_start v6e:spot l4:preemptible; do
  bash "$R" refuse_prov_${pv%%:*}_${pv#*:} ok PLATFORM=${pv%%:*} PROVISIONING=${pv#*:}
done
bash "$R" refuse_valid_for ok PLATFORM=l4 PROVISIONING=flex_start REQUEST_VALID_FOR=3h
echo "======== v5e bisection (V0 to V9) on a fake VM"
bash "$R" bisect bisect SCRIPT=cloud/v5e_bisect.sh LIBTPU_VERSION=0.0.42.1 FAKE_PLATFORM=v5e BISECT_TPU_LOG_DIR=$WORK/state_bisect/tpu_logs \
  BISECT_DUMP_MAX_MB=0.05 FAKE_DUMP_BYTES=20000
grep -E '^   (id|V[0-9]|reading)' "$WORK/state_bisect/out"
bash "$R" bisect_budget bisect SCRIPT=cloud/v5e_bisect.sh LIBTPU_VERSION=0.0.42.1 FAKE_PLATFORM=v5e BISECT_TPU_LOG_DIR=$WORK/state_bisect_budget/tpu_logs \
  BISECT_BUDGET_MIN=1 DETACHED_DEADLINE_MIN=20 BISECT_MIN_START_S=57
bash "$R" bisect_preempt preempt SCRIPT=cloud/v5e_bisect.sh LIBTPU_VERSION=0.0.42.1 FAKE_PLATFORM=v5e FAKE_RUN_S=5 BISECT_TPU_LOG_DIR=$WORK/state_bisect_preempt/tpu_logs; polls bisect_preempt
bash "$R" bisect_refuse ok SCRIPT=cloud/v5e_bisect.sh BISECT_BUDGET_MIN=95
