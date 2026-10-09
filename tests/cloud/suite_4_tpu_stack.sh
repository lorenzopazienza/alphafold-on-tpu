#!/bin/bash
# Suite 4 (first written 2026-10-09): the TPU stack (jax/jaxlib 0.10.2 with
# libtpu LIBTPU_VERSION, default 0.0.43.2): install and exit-5 check
# (cloud/vm_tpu_stack.sh), tpu_runtime in setup.json and session.json, the
# stack_validation plan (two libtpu versions on one v6e VM, switch failure,
# comparison with the v6e reference after the fetch), and the fake's crash
# rule (5 samples segfault on v5e with 0.0.42.1 only).
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
REFRUN="$WORK/repo/results/af3/20261008T165218Z_v6e_probe/tpu_xla_rep1_seed1_fresh"
ref() {   # the v6e reference run in the scratch repository (run.sh removes results/ after each run)
  [ -d "$WORK/ref/7U3J" ] || { echo "   (no v6e reference structures on this machine: comparisons are skipped)"; return; }
  mkdir -p "$REFRUN/7U3J/af3_output" && cp -R "$WORK/ref/7U3J" "$REFRUN/7U3J/af3_output/"
}
records() {   # tpu_runtime of setup.json and of each harness session.json, libtpu installs, plan_runs
  local s
  s=$(ls -d "$WORK/state_$1/fetched/results/af3/"*/ 2> /dev/null | grep -v 20261008T165218Z_v6e_probe | tail -1)
  [ -n "$s" ] || { echo "   (nothing fetched)"; return; }
  python3 - "$s" <<'PYEOF'
import json, pathlib, sys
s = pathlib.Path(sys.argv[1])
def show(label, block):
  if block is None:
    print(f'   {label}: no tpu_runtime'); return
  print(f"   {label}: libtpu {block.get('libtpu_version')}, {block.get('build_label')}, built {block.get('build_date')}"
        + (f", error: {block['error']}" if block.get('error') else ''))
if (s / 'setup.json').exists():
  show('setup.json', json.loads((s / 'setup.json').read_text()).get('tpu_runtime'))
for run in sorted(p for p in s.iterdir() if (p / 'session.json').exists()):
  show(f'{run.name}/session.json', json.loads((run / 'session.json').read_text()).get('tpu_runtime'))
if (s / 'plan_runs.jsonl').exists():
  for line in (s / 'plan_runs.jsonl').read_text().splitlines():
    r = json.loads(line)
    print('   plan_runs:', {k: r.get(k) for k in ('run', 'libtpu', 'harness_exit_code', 'libtpu_switch', 'ok', 'installed') if k in r})
PYEOF
  grep -oE "uv pip install --python [^ ]+ .*" "$WORK/state_$1/log" | sed -E 's#--python [^ ]+#--python <venv>#; s/^/   /' | sort -u
}

echo "======== stack on single runs"
bash "$R" stack_v5e_default ok PLATFORM=v5e FAKE_PLATFORM=v5e; records stack_v5e_default
bash "$R" stack_v5e_0421 ok PLATFORM=v5e FAKE_PLATFORM=v5e LIBTPU_VERSION=0.0.42.1; records stack_v5e_0421
bash "$R" stack_wrong ok PLATFORM=v6e FAKE_PLATFORM=v6e FAKE_LIBTPU_WRONG=0.0.42.1
grep -h "TPU stack check" "$WORK/state_stack_wrong/fetched/results/af3/"*/job.log 2> /dev/null | sed 's/^/   /'
bash "$R" stack_bad_version ok PLATFORM=v5e LIBTPU_VERSION=0.0.43.x
bash "$R" stack_ignored_l4 ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES=us-east4-a LIBTPU_VERSION=0.0.43.2
echo "   libtpu installs on L4 (must be none): $(grep -c 'libtpu==' "$WORK/state_stack_ignored_l4/log")"

echo "======== stack_validation plan"
ref; bash "$R" stack_plan_v6e ok PLATFORM=v6e FAKE_PLATFORM=v6e PLAN=harness/plans/stack_validation.yaml FAKE_REF_DIR=$WORK/ref/7U3J
records stack_plan_v6e
sed -n '/>> Comparing with/,/af3-run VMs left/p' "$WORK/state_stack_plan_v6e/out" | grep -v "VMs left"
ref; bash "$R" stack_plan_v5e ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/stack_validation.yaml FAKE_REF_DIR=$WORK/ref/7U3J
records stack_plan_v5e
bash "$R" stack_plan_conflict ok PLATFORM=v6e PLAN=harness/plans/stack_validation.yaml LIBTPU_VERSION=0.0.42.1
ref; bash "$R" stack_switch_fail ok PLATFORM=v6e FAKE_PLATFORM=v6e PLAN=harness/plans/stack_validation.yaml FAKE_UV_FAIL=libtpu==0.0.42.1
records stack_switch_fail

echo "======== bisection launcher on the default stack (nothing should segfault)"
bash "$R" stack_bisect_default bisect SCRIPT=cloud/v5e_bisect.sh FAKE_PLATFORM=v5e BISECT_TPU_LOG_DIR=$WORK/state_stack_bisect_default/tpu_logs
grep -E '^   (V[0-9]|reading)' "$WORK/state_stack_bisect_default/out" | head -14

echo "======== launch summaries (prompt answered n)"
bash "$R" stack_prompt_v5e ok PLATFORM=v5e SPOT=0 PLAN=harness/plans/stack_validation.yaml ANSWER=n
sed -n '/>> Plan /,/Proceed/p' "$WORK/state_stack_prompt_v5e/out" | sed 's/^/   | /'
bash "$R" stack_prompt_v6e ok PLATFORM=v6e PLAN=harness/plans/stack_validation.yaml ANSWER=n
sed -n '/>> Plan /,/Proceed/p' "$WORK/state_stack_prompt_v6e/out" | sed 's/^/   | /'
