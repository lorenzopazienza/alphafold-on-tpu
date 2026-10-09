"""Runs a plan (harness/plans/*.yaml) on one platform through harness/run_af3.py.

    python3 harness/run_plan.py --plan harness/plans/probe.yaml --platform v5e --describe
    python3 harness/run_plan.py --plan harness/plans/probe.yaml --platform v5e \\
        --session 20261010T090000Z_v5e_probe --model_dir ~/af3_weights_run

A plan gives targets (a list, or "pilot" for the manifest's pilot rows, with
an optional per-platform token cap), seeds, fresh-cache repetitions, an
optional warm-cache rerun of one seed (both per platform if given as a
mapping), recycles and samples ("default" means AlphaFold3's defaults), the
harness configuration per platform and the polling deadline per platform.
See harness/plans/probe.yaml.

Every (config, rep, seed) is one harness session results/af3/<session>/<run>/
in which each target runs in its own fresh process. Fresh-cache runs start
from an empty JAX compilation cache. If the plan asks for a warm rerun of seed
S, the rep 1 run of seed S keeps its compiled executables
(--save_fresh_cache) and the rerun loads them (--compile_cache warm:...), so
it can be compared with its source run. A failed run or target never stops the
plan; every harness exit code goes to results/af3/<session>/plan_runs.jsonl.
Exit status: 0 if every harness run exited 0, 1 otherwise, 5 if a libtpu
switch failed.

TPU stacks (optional plan key "libtpu", per platform, a list of libtpu
versions; TPU platforms only): every run is repeated for each version, in the
order given, and the run names carry it (tpu_xla_libtpu-0.0.43.2_rep1_seed1_fresh).
Setup installs the first version (cloud/af3_run.sh passes it as
LIBTPU_VERSION); before the first run of each version the plan makes sure it
is the installed one, switching with cloud/vm_tpu_stack.sh --switch (jax and
jaxlib unchanged). A failed switch stops the plan with exit status 5. Without
the key, nothing is switched and run names are as before.

compare_reference (optional, per platform): a harness run folder of an
earlier session; cloud/af3_run.sh compares the fetched session with it
(harness/compare_runs.py, bit for bit and by RMSD).

--describe prints the expanded plan as JSON and runs nothing; cloud/af3_run.sh
uses it on the laptop to choose uploads and to size the VM's lifetime.
--upload_uri gs://FOLDER (optional, the session's folder in the results
bucket): each harness run uploads its folder after every target, and the whole
session folder (plan_runs.jsonl, setup records, logs) is uploaded after every
harness run. An upload failure is printed and never stops the plan.
Standard library only.
"""

import argparse
import csv
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'harness'))
from run_af3 import read_configs, upload  # noqa: E402

PLATFORMS = ('cpu', 'l4', 'v5e', 'v6e')


def as_list(value):
  value = value.strip()
  if value.startswith('[') and value.endswith(']'):
    return [v.strip() for v in value[1:-1].split(',') if v.strip()]
  return [value]


def per_platform(value, platform):
  """A value or a mapping with per-platform keys and an optional default."""
  if isinstance(value, dict):
    if platform in value:
      return value[platform]
    if 'default' in value:
      return value['default']
    raise SystemExit(f'plan has no value for platform {platform}: {value}')
  return value


def optional_int(value):
  return None if value in (None, 'default', 'none') else int(value)


def expand(plan_path, platform, manifest_path):
  if platform not in PLATFORMS:
    raise SystemExit(f'--platform must be one of {", ".join(PLATFORMS)}')
  plan = read_configs(pathlib.Path(plan_path))
  with open(manifest_path) as f:
    manifest = {r['pdb_id']: r for r in csv.DictReader(f)}

  spec = as_list(per_platform(plan['targets'], platform))
  if spec == ['pilot']:
    targets = [pid for pid, r in sorted(manifest.items()) if r['pilot'] == 'yes']
  else:
    targets = spec
  missing = [t for t in targets if t not in manifest]
  if missing:
    raise SystemExit(f'Targets not in the manifest: {missing}')
  cap = optional_int(per_platform(plan.get('max_tokens', 'none'), platform))
  if cap is not None:
    targets = [t for t in targets if int(manifest[t]['num_tokens']) <= cap]
  targets = sorted(targets, key=lambda t: (int(manifest[t]['num_tokens']), t))

  configs = as_list(per_platform(plan['configs'], platform))
  seeds = [int(s) for s in as_list(plan['seeds'])]
  reps = int(per_platform(plan.get('fresh_reps', '1'), platform))
  warm_seed = optional_int(per_platform(plan.get('warm_rerun_seed', 'none'), platform))
  if warm_seed is not None and warm_seed not in seeds:
    raise SystemExit(f'warm_rerun_seed {warm_seed} is not one of the seeds {seeds}')

  libtpu = [v for v in as_list(per_platform(plan.get('libtpu', 'none'), platform)) if v != 'none']
  for v in libtpu:
    if not re.fullmatch(r'[0-9]+(\.[0-9]+)+', v):
      raise SystemExit(f'plan: libtpu {v!r} is not a version like 0.0.43.2')
  if libtpu and platform not in ('v5e', 'v6e'):
    raise SystemExit(f'plan: libtpu versions are for TPU platforms, not {platform}')
  if len(set(libtpu)) != len(libtpu):
    raise SystemExit(f'plan: libtpu versions repeat: {libtpu}')

  runs = []
  for version in libtpu or [None]:
    tag = f'_libtpu-{version}' if version else ''
    for config in configs:
      name = f'{config}{tag}'
      for rep in range(1, reps + 1):
        for seed in seeds:
          runs.append({'run': f'{name}_rep{rep}_seed{seed}_fresh', 'config': config, 'rep': rep,
                       'seed': seed, 'cache': 'fresh', 'libtpu': version,
                       'save_cache': warm_seed == seed and rep == 1})
      if warm_seed is not None:
        runs.append({'run': f'{name}_seed{warm_seed}_warm', 'config': config, 'rep': None,
                     'seed': warm_seed, 'cache': 'warm', 'save_cache': False, 'libtpu': version,
                     'source_run': f'{name}_rep1_seed{warm_seed}_fresh'})
  reference = per_platform(plan.get('compare_reference', 'none'), platform)
  return {
      'plan': str(pathlib.Path(plan_path)), 'platform': platform,
      'description': plan.get('description', ''), 'label': plan.get('label', 'measurement'),
      'weights': plan.get('weights', 'random'),
      'targets': targets, 'tokens': {t: int(manifest[t]['num_tokens']) for t in targets},
      'configs': configs, 'seeds': seeds, 'fresh_reps': reps, 'warm_rerun_seed': warm_seed,
      'num_recycles': optional_int(plan.get('num_recycles', 'default')),
      'num_diffusion_samples': optional_int(plan.get('num_diffusion_samples', 'default')),
      'run_timeout_min': float(plan.get('run_timeout_min', '180')),
      'deadline_min': int(per_platform(plan['deadline_min'], platform)),
      'libtpu': libtpu, 'compare_reference': None if reference == 'none' else reference,
      'runs': runs, 'processes': len(runs) * len(targets),
  }


AF3_PY = REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python'


def installed_libtpu():
  out = subprocess.run([str(AF3_PY), '-c', "import importlib.metadata as m; print(m.version('libtpu'))"],
                       capture_output=True, text=True)
  return out.stdout.strip() if out.returncode == 0 else None


def ensure_libtpu(version):
  """Makes libtpu VERSION the installed one (cloud/vm_tpu_stack.sh --switch); True on success."""
  have = installed_libtpu()
  if have == version:
    print(f'>> libtpu {version} is installed', flush=True)
    return True
  print(f'>> switching libtpu {have} -> {version}', flush=True)
  env = dict(os.environ, PY=str(AF3_PY), LIBTPU_VERSION=version,
             PATH=f'{pathlib.Path.home() / ".local" / "bin"}:{os.environ.get("PATH", "")}')
  rc = subprocess.run(['bash', str(REPO / 'cloud' / 'vm_tpu_stack.sh'), '--switch'], cwd=REPO, env=env).returncode
  return rc == 0 and installed_libtpu() == version


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--plan', required=True)
  ap.add_argument('--platform', required=True, choices=PLATFORMS)
  ap.add_argument('--manifest', default='inputs/manifest.csv')
  ap.add_argument('--describe', action='store_true', help='print the expanded plan as JSON')
  ap.add_argument('--session', help='results/af3/<session>/ holds every run of this plan')
  ap.add_argument('--model_dir', default='~/af3_weights')
  ap.add_argument('--upload_uri', help='gs:// folder of this session in the results bucket')
  args = ap.parse_args()

  desc = expand(REPO / args.plan, args.platform, REPO / args.manifest)
  desc['plan'] = args.plan
  if args.describe:
    print(json.dumps(desc, indent=2))
    return
  if not args.session or not re.fullmatch(r'[A-Za-z0-9_.-]+', args.session):
    raise SystemExit('--session NAME (letters, digits, _ . -) is required to run')

  log = REPO / 'results' / 'af3' / args.session / 'plan_runs.jsonl'
  log.parent.mkdir(parents=True, exist_ok=True)
  failed, current = 0, None
  for i, run in enumerate(desc['runs'], 1):
    if run.get('libtpu') and run['libtpu'] != current:
      if not ensure_libtpu(run['libtpu']):
        with open(log, 'a') as f:
          f.write(json.dumps({'libtpu_switch': run['libtpu'], 'ok': False, 'installed': installed_libtpu(),
                              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')})
                  + '\n')
        print(f'!! could not make libtpu {run["libtpu"]} the installed version; stopping the plan', flush=True)
        if args.upload_uri:
          upload(log.parent, args.upload_uri.rstrip('/'))
        sys.exit(5)
      current = run['libtpu']
    cache_name = f'{args.session}_{run["run"].split("_rep")[0].split("_seed")[0]}'
    cmd = [sys.executable, str(REPO / 'harness' / 'run_af3.py'),
           '--config', run['config'], '--targets', ','.join(desc['targets']),
           '--seeds', str(run['seed']), '--session', f'{args.session}/{run["run"]}',
           '--label', desc['label'], '--manifest', args.manifest, '--model_dir', args.model_dir,
           '--timeout_min', str(desc['run_timeout_min']), '--order', 'tokens',
           '--entry', '--log_compiles']
    if run['cache'] == 'warm':
      cmd += ['--compile_cache', f'warm:{cache_name}']
    else:
      cmd += ['--compile_cache', 'fresh']
      if run['save_cache']:
        cmd += ['--save_fresh_cache', cache_name]
    if desc['num_recycles'] is not None:
      cmd += ['--num_recycles', str(desc['num_recycles'])]
    if desc['num_diffusion_samples'] is not None:
      cmd += ['--num_diffusion_samples', str(desc['num_diffusion_samples'])]
    if args.upload_uri:
      cmd += ['--upload_uri', f'{args.upload_uri.rstrip("/")}/{run["run"]}']
    print(f'>> [{i}/{len(desc["runs"])}] {run["run"]}: {len(desc["targets"])} targets', flush=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    rc = subprocess.run(cmd, cwd=REPO).returncode
    failed += rc != 0
    with open(log, 'a') as f:
      f.write(json.dumps(dict(run, start_utc=start, harness_exit_code=rc,
                              session=f'{args.session}/{run["run"]}')) + '\n')
    if args.upload_uri:
      upload(log.parent, args.upload_uri.rstrip('/'))
  print(f'>> Plan done: {len(desc["runs"]) - failed}/{len(desc["runs"])} harness runs exited 0',
        flush=True)
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
