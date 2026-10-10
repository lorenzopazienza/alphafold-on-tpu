#!/bin/bash
# Suite 5 (first written 2026-10-09): the pilot launches of
# harness/plans/pilot.yaml and its split plans, exactly as they will be run
# (official weights from a bucket, results bucket, FETCH=full), on fake VMs:
# the run names and warm rerun on each VM, the deadline and lifetime the
# launcher derives, and that the weights URI is never printed. The whole plan
# on l4 or cpu (deadline 0) must be refused.
. "$(dirname "$0")/common.sh"
R="$KIT/run.sh"
bash "$KIT/mkrepo.sh"
W=gs://private-weights/af3.bin.zst
B=gs://pilot-results/study
pilot() {   # NAME: harness runs, processes and cache modes fetched, deadline, lifetime, weights URI leaks
  local s
  grep -E "follow deadline|VM lifetime limit|worst-case cost" "$WORK/state_$1/out" | sed 's/^/   | /'
  echo "   weights URI printed: $(grep -c 'private-weights' "$WORK/state_$1/out") time(s)"
  s=$(ls -d "$WORK/state_$1/fetched/results/af3/"*/ 2> /dev/null | grep -v 20261008T165218Z_v6e_probe | tail -1)
  [ -n "$s" ] || { echo "   (nothing fetched)"; return; }
  python3 - "$s" <<'PYEOF'
import json, pathlib, sys
s = pathlib.Path(sys.argv[1])
runs = [json.loads(l) for l in (s / 'plan_runs.jsonl').read_text().splitlines()] if (s / 'plan_runs.jsonl').exists() else []
procs = sorted(s.glob('*/*/run.json'))
modes = {}
for p in procs:
  r = json.loads(p.read_text())
  modes[r['compile_cache']['mode']] = modes.get(r['compile_cache']['mode'], 0) + 1
print(f"   harness runs: {', '.join(r.get('run', '?') + ' exit ' + str(r.get('harness_exit_code')) for r in runs)}")
print(f"   processes fetched: {len(procs)}; cache modes: {modes}")
# Cache evidence (harness/run_af3.py): every fresh process starts from an empty cache and compiles
# jit_apply_fn; every warm process loads an entry its source fresh process (same target) wrote.
problems, warm = [], 0
written = {}
for p in procs:
  r = json.loads(p.read_text())
  c, cc = r.get('compile') or {}, r['compile_cache']
  if cc['mode'] == 'fresh':
    if not cc.get('started_empty') or c.get('model_cache_hit'):
      problems.append(f"{p.parent.parent.name}/{r['pdb_id']}: fresh run did not start empty or hit the cache")
    if cc.get('saved_to'):
      written[r['pdb_id']] = set(cc.get('model_entries_written') or [])
for p in procs:
  r = json.loads(p.read_text())
  c, cc = r.get('compile') or {}, r['compile_cache']
  if cc['mode'] == 'warm':
    warm += 1
    if not c.get('model_cache_hit') or c.get('model_cache_key') not in written.get(r['pdb_id'], set()):
      problems.append(f"{r['pdb_id']}: warm run loaded {c.get('model_cache_key')}, not its source executable")
print(f"   cache check: {len(problems)} problem(s); {warm} warm process(es) loaded their source executable"
      if not problems else f"   cache check: {len(problems)} problem(s): {'; '.join(problems[:3])}")
PYEOF
}
echo "======== pilot launches (official weights from a bucket, results bucket, FETCH=full)"
bash "$R" pilot_v5e ok PLATFORM=v5e FAKE_PLATFORM=v5e SPOT=0 PLAN=harness/plans/pilot.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FETCH=full
pilot pilot_v5e
bash "$R" pilot_v6e ok PLATFORM=v6e FAKE_PLATFORM=v6e PLAN=harness/plans/pilot.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FETCH=full
pilot pilot_v6e
bash "$R" pilot_l4_1 ok PLATFORM=l4 FAKE_PLATFORM=l4 SPOT=0 PLAN=harness/plans/pilot_l4_1.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FETCH=full FAKE_ZONES="europe-west4-a:sync europe-west4-b:none"
pilot pilot_l4_1
bash "$R" pilot_cpu_2 ok PLATFORM=cpu FAKE_PLATFORM=cpu SPOT=0 PLAN=harness/plans/pilot_cpu_2.yaml \
  WEIGHTS_GCS_URI=$W RESULTS_GCS_URI=$B FETCH=full
pilot pilot_cpu_2
echo "======== the whole plan on a split platform is refused (deadline 0)"
bash "$R" pilot_refuse_l4 ok PLATFORM=l4 SPOT=0 PLAN=harness/plans/pilot.yaml WEIGHTS_GCS_URI=$W
bash "$R" pilot_refuse_cpu ok PLATFORM=cpu SPOT=0 PLAN=harness/plans/pilot.yaml WEIGHTS_GCS_URI=$W
