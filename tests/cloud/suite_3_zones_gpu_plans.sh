#!/bin/bash
# Suite 3 (first written 2026-10-09): L4 zone fallback with the real insert
# errors in fixtures/, JAX_PLATFORMS on the GPU path (fails if any GPU path
# sets a value containing gpu or rocm), and the v5e samples plan.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
attempts() { grep -E '>> Zone attempt|zone .*: no capacity|is running in|No zone|Not a capacity|zone .*: ' "$WORK/state_$1/out" | sed 's/^/   | /'; }
bash "$KIT/mkrepo.sh"
echo "======== zone fallback"
bash "$R" zones_suggest ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="europe-west4-a us-central1-b us-east4-c us-west1-a us-east4-a" \
  FAKE_ZONES="europe-west4-a:sync us-central1-b:none us-east4-c:suggest"
bash "$R" zones_staging_gone ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-c us-central1-a" FAKE_ZONES="us-east4-c:suggest"
grep -E "STOPPING|STAGING" "$WORK/state_zones_staging_gone/out" | sed 's/^/   | /' | head -3
bash "$R" zones_leftover ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="europe-west4-a europe-west4-b" FAKE_ZONES="europe-west4-a:leftover"
bash "$R" zones_all_out ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="europe-west4-a us-central1-b us-east4-c" \
  FAKE_ZONES="europe-west4-a:sync us-central1-b:none us-east4-c:none"
grep -A1 "af3-run VMs left" "$WORK/state_zones_all_out/out" | sed 's/^/   | /'
bash "$R" zones_all_out_suggested ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-c" FAKE_ZONES="us-east4-c:suggest us-east4-a:sync"
bash "$R" zones_quota ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="europe-west4-a europe-west4-b" FAKE_ZONES="europe-west4-a:quota"
bash "$R" zones_default_l4 ok PLATFORM=l4 FAKE_PLATFORM=l4 FAKE_ZONES="europe-west4-a:sync europe-west4-b:sync europe-west4-c:none"
bash "$R" zones_flex ok PLATFORM=l4 FAKE_PLATFORM=l4 PROVISIONING=flex_start ZONES="europe-west4-a us-east4-a" FAKE_ZONES="europe-west4-a:leftover"
bash "$R" zones_v6e ok PLATFORM=v6e FAKE_PLATFORM=v6e ZONES="europe-west4-a us-east5-b" FAKE_ZONES="europe-west4-a:sync"
bash "$R" zones_refuse_v5e ok PLATFORM=v5e ZONES="europe-west4-b us-central1-a"
bash "$R" zones_refuse_bad ok PLATFORM=l4 ZONES="europe-west4-a nowhere"

echo "======== JAX_PLATFORMS on the GPU path"
bash "$R" l4_probe_cuda ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-a"
bash "$R" l4_devfail devfail PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-a"
echo "   static check: JAX_PLATFORMS assignments containing gpu or rocm in cloud/ and harness/ (must print none):"
(cd "$REPO" && grep -rnIE "JAX_PLATFORMS *[=:] *['\"]?[^ #,]*(gpu|rocm)|JAX_PLATFORMS=[^ ]*(gpu|rocm)" cloud harness) | sed 's/^/   | /'
echo "   (end of static check)"
echo "   regression 1: the old setup check (JAX_PLATFORMS=gpu) must be caught"
sed -i '' 's/  l4) BACKEND=gpu JAXP=cuda ;;/  l4) BACKEND=gpu JAXP=gpu ;;/' "$WORK/repo/cloud/vm_device_check.sh"
bash "$R" l4_regress_setup ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-a"
echo "   regression 2: a GPU config with JAX_PLATFORMS: gpu must be caught"
bash "$KIT/mkrepo.sh" > /dev/null
python3 - "$WORK/repo/harness/configs.yaml" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
i = s.index('\nl4_xla:'); j = s.index('JAX_PLATFORMS: cuda', i)
p.write_text(s[:j] + 'JAX_PLATFORMS: gpu' + s[j + len('JAX_PLATFORMS: cuda'):])
PYEOF
bash "$R" l4_regress_config ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES="us-east4-a"
bash "$KIT/mkrepo.sh" > /dev/null

echo "======== v5e samples plan on a fake VM"
plan_ref() {
  local refdir="$WORK/repo/results/af3/20261008T165218Z_v6e_probe/tpu_xla_rep1_seed1_fresh/7U3J/af3_output"
  [ -d "$WORK/ref/7U3J" ] || { echo "   (no v6e reference structures on this machine: the prefix comparison is skipped)"; return; }
  mkdir -p "$refdir" && cp -R "$WORK/ref/7U3J" "$refdir/"
}
plan_ref
bash "$R" plan_samples bisect SCRIPT=cloud/v5e_bisect.sh BISECT_PLAN=cloud/plans/v5e_bisect_samples.json FAKE_PLATFORM=v5e \
  BISECT_TPU_LOG_DIR=$WORK/state_plan_samples/tpu_logs FAKE_REF_DIR=$WORK/ref/7U3J
sed -n '/Sample-prefix comparison/,/Bisection summary/p' "$WORK/state_plan_samples/out" | head -16
grep -E '^   (id|S[0-9]|reading)' "$WORK/state_plan_samples/out"
plan_ref
bash "$R" plan_samples_venvfail bisect SCRIPT=cloud/v5e_bisect.sh BISECT_PLAN=cloud/plans/v5e_bisect_samples.json FAKE_PLATFORM=v5e \
  BISECT_TPU_LOG_DIR=$WORK/state_plan_samples_venvfail/tpu_logs FAKE_UV_FAIL=libtpu==0.0.43.2
grep -E "^   S5l" "$WORK/state_plan_samples_venvfail/out"
bash "$R" plan_refuse ok SCRIPT=cloud/v5e_bisect.sh BISECT_PLAN=cloud/plans/missing.json
bash "$R" plan_prompt_no ok SCRIPT=cloud/v5e_bisect.sh BISECT_PLAN=cloud/plans/v5e_bisect_samples.json ANSWER=n
bash "$R" l4_prompt_no ok PLATFORM=l4 ZONES=us-east4-a ANSWER=n
