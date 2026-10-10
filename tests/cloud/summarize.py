"""summarize.py OUT: one line per scenario of a run_all.sh output.

A scenario fails the check if its exit code differs from the expected one
(EXPECTED, as validated on 2026-10-09), if a VM is still there (or left in a
zone), or if a JAX_PLATFORMS value containing gpu or rocm was seen outside
the two regression scenarios that inject one on purpose. A scenario missing
from the output also fails.
"""
import re
import sys

EXPECT_VIOLATION = {'l4_regress_setup', 'l4_regress_config'}
EXPECTED = {
    'probe_cpu': 0, 'probe_l4': 0, 'probe_v5e': 0, 'probe_v6e': 0, 'oom_v5e': 1, 'drop_v6e': 0,
    'lost_l4': 1, 'deadline_l4': 1, 'gcs_v6e': 0, 'mismatch_remote': 3, 'mismatch_local': 1,
    'refuse_cpu_a2-highgpu-1g': 1, 'refuse_cpu_ct6e-standard-1t': 1, 'refuse_l4_a2-highgpu-1g': 1,
    'refuse_l4_g2-standard-24': 1, 'refuse_v5e_v5litepod-4': 1, 'refuse_v6e_ct6e-standard-4t': 1,
    'prompt_no': 0, 'ctrlc_setup': 130, 'ctrlc_run': 130,
    'preempt_v5e': 1, 'preempt_gone_l4_bucket': 1, 'preempt_v6e': 1, 'bucket_light_ssh': 0,
    'bucket_full_ssh': 0, 'fetchfail_light': 0, 'fetchfail_full': 0, 'fetchfail_nobucket': 0,
    'light_without_bucket': 0, 'uploadfail': 0, 'refuse_fetch': 1, 'refuse_bucket_uri': 1,
    'refuse_same_bucket': 1, 'l4_flex': 0, 'l4_flex_us': 0, 'l4_spot_us': 0, 'l4_spot_west1': 0,
    'l4_g2_4': 0, 'v6e_default': 0, 'refuse_prov_v5e_flex_start': 1, 'refuse_prov_cpu_flex_start': 1,
    'refuse_prov_v6e_spot': 1, 'refuse_prov_l4_preemptible': 1, 'refuse_valid_for': 1, 'bisect': 0,
    'bisect_budget': 0, 'bisect_preempt': 1, 'bisect_refuse': 1,
    'zones_suggest': 0, 'zones_staging_gone': 0, 'zones_leftover': 0, 'zones_all_out': 1,
    'zones_all_out_suggested': 1, 'zones_quota': 1, 'zones_default_l4': 0, 'zones_flex': 0,
    'zones_v6e': 0, 'zones_refuse_v5e': 1, 'zones_refuse_bad': 1, 'l4_probe_cuda': 0, 'l4_devfail': 5,
    'l4_regress_setup': 5, 'l4_regress_config': 1, 'plan_samples': 0, 'plan_samples_venvfail': 0,
    'plan_refuse': 1, 'plan_prompt_no': 0, 'l4_prompt_no': 0,
    'stack_v5e_default': 0, 'stack_v5e_0421': 1, 'stack_wrong': 5, 'stack_bad_version': 1,
    'stack_ignored_l4': 0, 'stack_plan_v6e': 0, 'stack_plan_v5e': 0, 'stack_plan_conflict': 1,
    'stack_switch_fail': 5, 'stack_bisect_default': 0, 'stack_prompt_v5e': 0, 'stack_prompt_v6e': 0,
    'pilot_v5e': 0, 'pilot_v6e': 0, 'pilot_l4_1': 0, 'pilot_cpu_2': 0, 'pilot_refuse_l4': 1,
    'pilot_refuse_cpu': 1,
    'net_loss_running': 0, 'net_loss_deadline': 1, 'net_loss_vm_gone': 1, 'exit_code_bucket': 0,
    'exit_code_bucket_ssh_lost': 0, 'resume_record': 0, 'resume_legacy': 0,
}
rows, cur = [], None
for line in open(sys.argv[1]).read().splitlines():
  m = re.match(r'################ (\S+)(:?)', line)
  if m:
    cur = {'name': m.group(1).rstrip(':'), 'line': line[17:]}
    rows.append(cur)
    continue
  if cur is None:
    continue
  m = re.match(r'  exit code (\d+) in \d+ s \| VM created: ([^|]+)\| deleted: (\w+) \| still exists: (\w+)(.*)', line)
  if m:
    cur.update(rc=m.group(1), created=m.group(2).strip(), deleted=m.group(3), exists=m.group(4),
               left='LEFT' in m.group(5))
  m = re.search(r'violations \(gpu/rocm\): (\d+)', line)
  if m:
    cur['viol'] = int(m.group(1))
  if line.startswith('  fetched: '):
    cur['fetched'] = 'no' if 'nothing' in line else 'yes'
bad = 0
for r in rows:
  if 'rc' not in r:
    m = re.search(r'exit code (\d+)', r['line'])
    ok = ('still exists: True' not in r['line'] and m is not None
          and int(m.group(1)) == EXPECTED.get(r['name']))
    print(f"{'ok ' if ok else 'BAD'} {r['line'][:150]}")
    bad += not ok
    continue
  viol = r.get('viol', 0)
  ok = (r['exists'] == 'no' and not r['left'] and (viol == 0) != (r['name'] in EXPECT_VIOLATION)
        and int(r['rc']) == EXPECTED.get(r['name']))
  bad += not ok
  print(f"{'ok ' if ok else 'BAD'} {r['name']:28} exit {r['rc']:>3}  created {r['created']:22} deleted {r['deleted']:4}"
        f" left {'YES' if r['exists'] != 'no' or r['left'] else 'no ':3} fetched {r.get('fetched', '-'):3}"
        f" gpu/rocm {r.get('viol', '-')}")
missing = sorted(set(EXPECTED) - {r['name'] for r in rows})
for name in missing:
  print(f'BAD {name}: missing from the output')
bad += len(missing)
print(f'{len(rows)} scenarios, {bad} failing the check')
sys.exit(1 if bad else 0)
