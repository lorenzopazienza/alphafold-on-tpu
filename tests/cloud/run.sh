#!/bin/bash
# run.sh NAME SCENARIO [VAR=value ...]: one launcher run against the fake gcloud.
#
# The launcher runs from the scratch repository $WORK/repo (built by
# mkrepo.sh) with FAKE_SCENARIO=SCENARIO and the given variables; its output
# and the fake VM's home are in $WORK/state_NAME/. VAR=value pairs may include
#   SCRIPT=cloud/v5e_bisect.sh  run the bisection launcher (default cloud/af3_run.sh)
#   ANSWER=n                    answer the Proceed prompt instead of YES=1
# Prints a summary: exit code, VM created/deleted/left, zones tried,
# JAX_PLATFORMS values seen on the fake VM, what was fetched.
. "$(dirname "$0")/common.sh"
NAME=$1 SC=$2; shift 2
D=$WORK/state_$NAME; rm -rf "$D"; mkdir -p "$D/home"
SCRIPT=cloud/af3_run.sh ANSWER=""
for kv in "$@"; do
  case "$kv" in SCRIPT=*) SCRIPT=${kv#SCRIPT=} ;; ANSWER=*) ANSWER=${kv#ANSWER=} ;; esac
done
[ -d "$WORK/repo/cloud" ] || { echo "!! no scratch repository: run mkrepo.sh first" >&2; exit 2; }
cd "$WORK/repo"
T0=$(date +%s)
if [ -n "$ANSWER" ]; then
  (unset YES; echo "$ANSWER" | fake_env "$D" FAKE_SCENARIO="$SC" "$@" /bin/bash "$SCRIPT" > "$D/out" 2>&1)
else
  fake_env "$D" FAKE_SCENARIO="$SC" YES=1 "$@" /bin/bash "$SCRIPT" > "$D/out" 2>&1
fi
rc=$?
echo "################ $NAME (scenario $SC) $*"
echo "  exit code $rc in $(( $(date +%s) - T0 )) s | VM created: $([ -f $D/create_args ] && echo "yes, $(cat $D/created_zone 2>/dev/null)" || echo no) | deleted: $([ -f $D/deleted ] && echo yes || echo no) | still exists: $([ -f $D/created ] && echo YES || echo no)$(ls $D/zone_* > /dev/null 2>&1 && echo ' | LEFT IN A ZONE: YES')"
[ -f "$D/zones_tried" ] && echo "  zones tried: $(tr '\n' ' ' < $D/zones_tried)$([ -f $D/zones_deleted ] && echo "| deleted in: $(tr '\n' ' ' < $D/zones_deleted)")"
if [ -f "$D/jax_platforms.log" ]; then
  bad=$(grep -cE 'JAX_PLATFORMS=[^ ]*(gpu|rocm)' "$D/jax_platforms.log")
  echo "  JAX_PLATFORMS seen: $(sed 's/.*JAX_PLATFORMS=//' $D/jax_platforms.log | sort | uniq -c | tr -s ' ' | tr '\n' ';') violations (gpu/rocm): $bad"
fi
S=$(ls -d results/af3/*/ 2>/dev/null | grep -v '20261008T165218Z_v6e_probe/$' | tail -1)
echo "  fetched: ${S:-nothing}"
if [ -n "$S" ]; then
  echo "  fetched files: $(find "$S" -type f | wc -l | tr -d ' '); heavy (_data.json, full _confidences.json): $(find "$S" -type f \( -name '*_data.json' -o \( -name '*_confidences.json' ! -name '*_summary_confidences.json' \) \) | wc -l | tr -d ' '); mmCIF: $(find "$S" -name '*.cif' | wc -l | tr -d ' ')"
  for f in $(find "$S" -name run.json | sort); do python3 -c "
import json; r=json.load(open('$f'))
print('   ', r['session'].split('/')[-1], r['pdb_id'], 'exit', r['exit_code'], 'failure', r['failure'], 'compile', (r['compile'] or {}).get('model_seconds'), 'cache', r['compile_cache']['mode'], 'outputs', len(r['outputs']))"; done
fi
[ -d "$D/bucket" ] && echo "  bucket objects: $(find "$D/bucket" -type f | wc -l | tr -d ' ')"
grep -E '^!!|price note|results bucket|preempted|is gone|Fetching|Zone attempt|zone .*:|is running in|No zone|Not started' "$D/out" | sed 's/^/  | /' | head -24
pgrep -f "$D/home" > /dev/null && { echo "  (fake VM job still running: killed)"; pkill -f "$D/home"; }
mkdir -p "$D/fetched" && [ -d results ] && cp -R results "$D/fetched/"
rm -rf "$WORK/repo/results"
exit 0
