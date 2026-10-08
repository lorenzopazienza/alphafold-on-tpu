"""Runs the patched AlphaFold3 on frozen inputs, one fresh process per target.

    python3 harness/run_af3.py --config tpu_xla --targets pilot --session <name>
    python3 harness/run_af3.py --config cpu_xla --targets pilot --order tokens \\
        --limit 2 --num_recycles 1 --num_diffusion_samples 1 --seeds 1 --label smoke

Standard library only, so any python3 can drive it; run_alphafold.py itself
runs with the AlphaFold3 venv's python (never `uv run`, which would re-sync
the environment; see af3_tpu/README.md).

For every target the harness starts run_alphafold.py from third_party/alphafold3
in a new process, with the flags of the chosen configuration in
harness/configs.yaml, --run_data_pipeline=false and the frozen input from
inputs/manifest.csv. If --seeds selects fewer seeds than the frozen input has,
a derived input with only those seeds is written to the target's work/ folder
and both hashes are recorded.

Compile cache (--compile_cache):
  fresh       a new, empty JAX persistent compilation cache directory for every
              process, deleted after its size is recorded (default);
  warm:NAME   the named directory data/jax_cache/NAME, reused across processes
              and sessions; entries before and after each run are recorded.

Output: results/af3/<session>/ with session.json (configuration, machine,
packages, device, weights and manifest hashes), runs.jsonl (one line per
target), and per target command.txt, run_alphafold.log, run.json and
af3_output/. An existing session is never overwritten; results/ is gitignored.

Records hold no absolute paths and no hostname. Paths are relative to the
repository root; the recorded command is the one that ran, from the repository
root, except that the weights folder (outside the repository by rule) appears
as <model_dir>; session.json gives the weights file name and SHA-256. The
machine is described by platform, CPU, and on Google Cloud the machine and
accelerator type from the metadata server.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIGS = REPO / 'harness' / 'configs.yaml'


def utc_now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sha256_file(path, chunk=1 << 24):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    while block := f.read(chunk):
      h.update(block)
  return h.hexdigest()


def read_configs(path):
  """Reads the YAML subset described in configs.yaml into nested dicts."""
  root = {}
  stack = [(-1, root)]
  for n, raw in enumerate(path.read_text().splitlines(), 1):
    line = raw.split(' #')[0].rstrip() if not raw.lstrip().startswith('#') else ''
    if not line.strip():
      continue
    indent = len(line) - len(line.lstrip(' '))
    key, sep, value = line.strip().partition(':')
    if not sep or indent % 2:
      raise SystemExit(f'{path}:{n}: not a "key: value" line with 2-space indentation')
    while stack[-1][0] >= indent:
      stack.pop()
    parent = stack[-1][1]
    if value.strip():
      parent[key] = value.strip()
    else:
      parent[key] = {}
      stack.append((indent, parent[key]))
  return root


def git(path, *args):
  return subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True).stdout.strip()


def dir_stats(path):
  files = [p for p in pathlib.Path(path).rglob('*') if p.is_file()] if pathlib.Path(path).exists() else []
  return {'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}


PROBE = r'''
import importlib.metadata as md, json, platform
import jax
pk = {}
for p in ('alphafold3', 'jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
          'tokamax', 'dm-haiku', 'numpy', 'rdkit'):
  try: pk[p] = md.version(p)
  except md.PackageNotFoundError: pk[p] = None
d = jax.devices()
print(json.dumps({'python': platform.python_version(), 'packages': pk,
                  'default_backend': jax.default_backend(), 'devices': [str(x) for x in d],
                  'device_kind': d[0].device_kind}))
'''


def probe(python, env, backend):
  """Package versions and devices, from a separate process of the AF3 python."""
  env = dict(env, JAX_PLATFORMS=backend)
  out = subprocess.run([python, '-c', PROBE], capture_output=True, text=True, env=env, timeout=600)
  if out.returncode != 0:
    raise SystemExit(f'Environment probe failed:\n{out.stderr[-2000:]}')
  return json.loads(out.stdout.strip().splitlines()[-1])


def gce_metadata(key):
  """A Google Cloud metadata value (last path component), or None elsewhere."""
  req = urllib.request.Request(f'http://metadata.google.internal/computeMetadata/v1/instance/{key}',
                               headers={'Metadata-Flavor': 'Google'})
  try:
    with urllib.request.urlopen(req, timeout=1) as r:
      return r.read().decode().rsplit('/', 1)[-1] or None
  except OSError:
    return None


def host_info():
  """The machine, without its hostname."""
  cpu = platform.processor()
  try:
    for line in open('/proc/cpuinfo'):
      if line.startswith('model name'):
        cpu = line.split(':', 1)[1].strip()
        break
  except OSError:
    if sys.platform == 'darwin':
      cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                           capture_output=True, text=True).stdout.strip()
  return {'platform': platform.platform(), 'machine': platform.machine(), 'cpu': cpu,
          'cpu_count': os.cpu_count(),
          'gce_machine_type': gce_metadata('machine-type'),
          'gce_accelerator_type': gce_metadata('attributes/accelerator-type'),
          'gce_zone': gce_metadata('zone')}


def rel(path):
  """A path inside the repository, relative to its root."""
  return os.path.relpath(pathlib.Path(path).resolve(), REPO)


def scrub(obj):
  """Last line of defence: no repository or home prefix in written records."""
  if isinstance(obj, dict):
    return {k: scrub(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [scrub(v) for v in obj]
  if isinstance(obj, str):
    return obj.replace(str(REPO) + os.sep, '').replace(str(REPO), '.').replace(
        str(pathlib.Path.home()), '~')
  return obj


def write_json(path, obj):
  path.write_text(json.dumps(scrub(obj), indent=2) + '\n')


LOG_PATTERNS = {
    'featurisation_seconds': re.compile(r'Featurising data with \d+ seed\(s\) took ([\d.]+) seconds'),
    'inference_seconds_by_seed': re.compile(r'Running model inference with seed (\d+) took ([\d.]+) seconds'),
    'extraction_seconds_by_seed': re.compile(r'Extracting \d+ inference samples with seed (\d+) took ([\d.]+) seconds'),
    'inference_and_extraction_seconds': re.compile(
        r'Running model inference and extracting output structures? with \d+ seed\(s\) took ([\d.]+) seconds'),
    'device_line': re.compile(r'(Found local .* devices: .*)'),
    'bucket_line': re.compile(r'(Got bucket size \d+ for input with \d+ tokens.*)'),
}


def parse_log(text):
  out = {}
  for key, pat in LOG_PATTERNS.items():
    if key.endswith('_by_seed'):
      out[key] = {m.group(1): float(m.group(2)) for m in pat.finditer(text)}
    else:
      m = pat.search(text)
      out[key] = (float(m.group(1)) if key.endswith('seconds') else m.group(1).strip()) if m else None
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--config', required=True)
  ap.add_argument('--targets', default='pilot', help='pilot, all, or comma-separated PDB IDs')
  ap.add_argument('--session', help='default: <UTC timestamp>_<config>')
  ap.add_argument('--label', default='measurement', help='for example smoke or measurement')
  ap.add_argument('--manifest', default='inputs/manifest.csv')
  ap.add_argument('--af3_dir', default='third_party/alphafold3', help='must be inside the repository')
  ap.add_argument('--python', help='default: <af3_dir>/.venv/bin/python')
  ap.add_argument('--model_dir', default='~/af3_weights', help='outside the repository')
  ap.add_argument('--compile_cache', default='fresh', help='fresh or warm:NAME')
  ap.add_argument('--num_recycles', type=int)
  ap.add_argument('--num_diffusion_samples', type=int)
  ap.add_argument('--seeds', help='comma-separated subset of the frozen seeds, for example 1')
  ap.add_argument('--order', choices=('manifest', 'tokens'), default='manifest')
  ap.add_argument('--limit', type=int, help='run only the first N targets after ordering')
  ap.add_argument('--timeout_min', type=float, default=180)
  args = ap.parse_args()

  configs = read_configs(CONFIGS)
  if args.config not in configs:
    raise SystemExit(f'Unknown config {args.config}; known: {", ".join(configs)}')
  cfg = configs[args.config]
  if args.compile_cache != 'fresh' and not re.fullmatch(r'warm:[A-Za-z0-9_.-]+', args.compile_cache):
    raise SystemExit('--compile_cache must be fresh or warm:NAME')
  af3_dir = (REPO / args.af3_dir).resolve()
  if not af3_dir.is_relative_to(REPO):
    raise SystemExit('--af3_dir must be inside the repository, so records can use relative paths.')
  # Absolute but not resolved: a venv's bin/python is a symlink to the base
  # interpreter, and following it would leave the venv.
  python = (pathlib.Path(os.path.abspath(os.path.expanduser(args.python))) if args.python
            else af3_dir / '.venv' / 'bin' / 'python')
  # Run-time form of python: relative to af3_dir when inside the repository.
  python_arg = os.path.relpath(python, af3_dir) if python.is_relative_to(REPO) else str(python)
  python_shown = python_arg if python.is_relative_to(REPO) else f'<python: {python.name}>'
  model_dir = pathlib.Path(args.model_dir).expanduser().resolve()
  if model_dir.is_relative_to(REPO):
    raise SystemExit('--model_dir must be outside the repository: weights never go in it.')
  weights = sorted(model_dir.glob('*.bin*'))
  if len(weights) != 1:
    raise SystemExit(f'Expected exactly one weights file in --model_dir, found {len(weights)}')
  manifest_path = (REPO / args.manifest).resolve()

  with open(manifest_path) as f:
    manifest = list(csv.DictReader(f))
  if args.targets == 'pilot':
    rows = [r for r in manifest if r['pilot'] == 'yes']
  elif args.targets == 'all':
    rows = manifest
  else:
    wanted = args.targets.split(',')
    rows = [r for r in manifest if r['pdb_id'] in wanted]
    missing = set(wanted) - {r['pdb_id'] for r in rows}
    if missing:
      raise SystemExit(f'Not in the manifest: {sorted(missing)}')
  if args.order == 'tokens':
    rows = sorted(rows, key=lambda r: (int(r['num_tokens']), r['pdb_id']))
  if args.limit:
    rows = rows[:args.limit]
  seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None

  session = args.session or f'{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}_{args.config}'
  out_dir = REPO / 'results' / 'af3' / session
  if out_dir.exists():
    raise SystemExit(f'{rel(out_dir)} exists; sessions are never overwritten.')
  out_dir.mkdir(parents=True)

  env = dict(os.environ, PYTHONUNBUFFERED='1')
  env.update(cfg.get('env', {}))
  warm_dir = None
  if args.compile_cache.startswith('warm:'):
    warm_dir = REPO / 'data' / 'jax_cache' / args.compile_cache.split(':', 1)[1]
    warm_dir.mkdir(parents=True, exist_ok=True)

  print(f'>> Session {session}: {args.config}, {len(rows)} targets, label {args.label}', flush=True)
  print('>> Probing the environment and hashing the weights', flush=True)
  env_probe = probe(str(python), env, cfg['jax_backend'])
  patch_sha = sha256_file(REPO / 'af3_tpu' / 'af3_tpu.patch')
  session_info = {
      'session': session, 'label': args.label, 'created_utc': utc_now(),
      'config_name': args.config, 'config': cfg,
      'args': dict(vars(args), af3_dir=rel(af3_dir), manifest=rel(manifest_path),
                   python=(os.path.relpath(python, REPO) if python.is_relative_to(REPO)
                           else f'<python: {python.name}>'),
                   model_dir='<model_dir>'),
      'targets': [r['pdb_id'] for r in rows],
      'compile_cache': {'mode': 'fresh' if warm_dir is None else 'warm',
                        'warm_dir': rel(warm_dir) if warm_dir else None,
                        'warm_at_start': dir_stats(warm_dir) if warm_dir else None},
      'host': host_info(), 'probe': env_probe,
      'env_overrides': cfg.get('env', {}),
      'xla_env': {k: v for k, v in env.items() if k.startswith(('XLA_', 'JAX_', 'TF_'))},
      'repo': {'commit': git(REPO, 'rev-parse', 'HEAD'),
               'dirty': bool(git(REPO, 'status', '--porcelain', '--', 'harness', 'inputs', 'af3_tpu'))},
      'alphafold3': {'dir': rel(af3_dir), 'commit': git(af3_dir, 'rev-parse', 'HEAD'),
                     'patch_sha256': patch_sha,
                     'run_alphafold_sha256': sha256_file(af3_dir / 'run_alphafold.py')},
      'weights': {'file': weights[0].name, 'sha256': sha256_file(weights[0]),
                  'bytes': weights[0].stat().st_size},
      'manifest': {'path': rel(manifest_path), 'sha256': sha256_file(manifest_path)},
  }
  write_json(out_dir / 'session.json', session_info)
  print(f'   device: {env_probe["device_kind"]} ({env_probe["default_backend"]}), '
        f'jax {env_probe["packages"]["jax"]}', flush=True)

  failures = 0
  for i, row in enumerate(rows, 1):
    tdir = out_dir / row['pdb_id']
    work = tdir / 'work'
    work.mkdir(parents=True)
    frozen = REPO / row['input_path']
    frozen_sha = sha256_file(frozen)
    if frozen_sha != row['input_sha256']:
      raise SystemExit(f'{rel(frozen)}: SHA-256 differs from the manifest')
    run_input = frozen
    if seeds:
      data = json.loads(frozen.read_text())
      if not set(seeds) <= set(data['modelSeeds']):
        raise SystemExit(f'--seeds {seeds} not all in the frozen input seeds {data["modelSeeds"]}')
      data['modelSeeds'] = seeds
      run_input = work / 'input_seeds.json'
      run_input.write_text(json.dumps(data, indent=2) + '\n')
    cache_dir = warm_dir if warm_dir else work / 'jax_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_before = dir_stats(cache_dir)

    def from_af3(path):
      return os.path.relpath(path, af3_dir)

    cmd = [python_arg, 'run_alphafold.py',
           f'--json_path={from_af3(run_input)}',
           f'--output_dir={from_af3(tdir / "af3_output")}',
           f'--model_dir={model_dir}',
           f'--jax_backend={cfg["jax_backend"]}',
           f'--flash_attention_implementation={cfg["flash_attention_implementation"]}',
           '--run_data_pipeline=false',
           f'--jax_compilation_cache_dir={from_af3(cache_dir)}']
    if args.num_recycles is not None:
      cmd.append(f'--num_recycles={args.num_recycles}')
    if args.num_diffusion_samples is not None:
      cmd.append(f'--num_diffusion_samples={args.num_diffusion_samples}')
    env_prefix = ' '.join(f'{k}={shlex.quote(v)}' for k, v in cfg.get('env', {}).items())
    shown = [python_shown] + [a if not a.startswith('--model_dir=') else '--model_dir=<model_dir>'
                              for a in cmd[1:]]
    command = f'cd {shlex.quote(rel(af3_dir))} && {env_prefix + " " if env_prefix else ""}{shlex.join(shown)}'
    (tdir / 'command.txt').write_text(command + '\n')

    print(f'[{i}/{len(rows)}] {row["pdb_id"]} ({row["num_tokens"]} tokens, bucket {row["bucket"]})',
          flush=True)
    start_utc, t0, timed_out = utc_now(), time.monotonic(), False
    with open(tdir / 'run_alphafold.log', 'w') as log:
      proc = subprocess.Popen(cmd, cwd=af3_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
      try:
        rc = proc.wait(timeout=args.timeout_min * 60)
      except subprocess.TimeoutExpired:
        proc.kill()
        rc, timed_out = proc.wait(), True
    wall = time.monotonic() - t0
    parsed = parse_log((tdir / 'run_alphafold.log').read_text(errors='replace'))
    outputs = sorted(str(p.relative_to(tdir)) for p in (tdir / 'af3_output').rglob('*')
                     if p.is_file() and (p.suffix == '.cif' or p.name.endswith('summary_confidences.json')))
    cache_after = dir_stats(cache_dir)
    if warm_dir is None:
      shutil.rmtree(cache_dir, ignore_errors=True)
    record = {
        'session': session, 'config': args.config, 'label': args.label, 'pdb_id': row['pdb_id'],
        'num_tokens': int(row['num_tokens']), 'bucket': int(row['bucket']),
        'start_utc': start_utc, 'end_utc': utc_now(), 'wall_seconds': round(wall, 2),
        'exit_code': rc, 'timed_out': timed_out, **parsed,
        'device_kind': env_probe['device_kind'],
        'command': command, 'input_path': row['input_path'], 'input_sha256': frozen_sha,
        'run_input_sha256': sha256_file(run_input), 'seeds': seeds or json.loads(frozen.read_text())['modelSeeds'],
        'weights_sha256': session_info['weights']['sha256'],
        'compile_cache': {'mode': session_info['compile_cache']['mode'], 'dir': rel(cache_dir),
                          'before': cache_before, 'after': cache_after},
        'output_dir': str((tdir / 'af3_output').relative_to(REPO)), 'outputs': outputs,
    }
    record = scrub(record)
    write_json(tdir / 'run.json', record)
    with open(out_dir / 'runs.jsonl', 'a') as f:
      f.write(json.dumps(record) + '\n')
    ok = rc == 0 and any(o.endswith('.cif') for o in outputs) and any(o.endswith('.json') for o in outputs)
    failures += not ok
    inf = parsed['inference_seconds_by_seed']
    print(f'    exit {rc}{" (timed out)" if timed_out else ""}, wall {wall:.1f} s, '
          f'inference by seed {inf}, outputs {len(outputs)}', flush=True)

  print(f'>> Done: {len(rows) - failures}/{len(rows)} targets ok. results/af3/{session}/', flush=True)
  sys.exit(1 if failures else 0)


if __name__ == '__main__':
  main()
