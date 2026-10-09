"""Runs ON a v5e VM after setup: one-VM bisection of the v5e compile segfault.

    python3 cloud/vm_v5e_bisect.py --out results/af3/<session> --session <session> \\
        --job_start EPOCH_SECONDS [--budget_min 80] [--timeout_min 15]
    python3 cloud/vm_v5e_bisect.py --list      # the variants, one per line
    python3 cloud/vm_v5e_bisect.py --plan cloud/plans/v5e_bisect_samples.json ...

Without --plan the built-in variants V0 to V9 below run. --plan FILE (JSON)
replaces them with the plan's own list; see cloud/plans/v5e_bisect_samples.json:
  variants   the same fields as below (id, kind, target, what, recycles,
             samples, entry, log_compiles, cache, only_if_crashed), plus
             env          extra environment variables for that run (for
                          example LIBTPU_INIT_ARGS),
             venv         the name of a venv from "venvs" to run with,
             compare_to   a variant this one changes one thing against; the
                          readings say whether the change avoids its crash,
             series       "samples": part of the diffusion-samples series
                          summarised in the readings;
  venvs      name -> {"pip": [requirements], "find_links": URL (optional)}:
             a copy of AlphaFold3's venv (third_party/alphafold3/.venv_NAME)
             with those packages installed on top, made the first time a
             variant needs it (log and pip freeze in OUT/venv_NAME*.txt). If
             that fails, its variants are skipped with the reason.

Started by cloud/vm_v5e_bisect.sh (itself started by cloud/v5e_bisect.sh) once
cloud/vm_af3_setup.sh has installed AlphaFold3, jax[tpu] and random weights.

Background (2026-10-08): the probe (harness, AF3 defaults of 10 recycles and
5 diffusion samples, persistent compile cache, af3_entry.py wrapper,
JAX_LOG_COMPILES=1) died with SIGSEGV about 1 s into XLA compilation of
jit(apply_fn) on v5litepod-1 for 7U3J and 7D5C, while the smoke test
(run_alphafold.py called directly on toy_118 with 1 recycle and 1 sample, no
cache) passed on the same VM type, and the v6e compiled the same shapes.

Variants run in this order, each AlphaFold3 run in a fresh process with its
own timeout (--timeout_min, capped by what is left of the budget):

  V0   exact smoke command on toy_118 (1 recycle, 1 sample); expected to pass
  V1   toy_118 through the harness with the probe's settings
  V2   7U3J, smoke-style direct command (1 recycle, 1 sample; no harness,
       wrapper, cache or JAX_LOG_COMPILES)
  V3   7U3J direct with AF3 defaults (10 recycles, 5 samples), XLA HLO dump
       at the start and end of optimisation
  V3p  only if V3 crashed: V3 with an HLO dump after every compiler pass; the
       last pass dumped before the crash names the pass that crashed
  V4   7U3J through the harness, probe settings but 1 diffusion sample
  V5   7U3J through the harness, probe settings but 1 recycle
  V6   7U3J through the harness, probe settings without the persistent cache
  V7   7U3J through the harness, probe settings without af3_entry.py
  V8   7U3J through the harness, probe settings without JAX_LOG_COMPILES
  V9   7D5C (1023 tokens, the 16 GB test) through the harness with the first
       setting that passed among V2 (1 recycle, 1 sample), V4 (1 sample) and
       V5 (1 recycle); skipped if none passed

The budget (--budget_min, counted from --job_start, setup included) skips
the remaining variants once less than BISECT_MIN_START_S (180) seconds are
left. Every variant except V0 (kept exactly as the smoke ran) has libtpu's
verbose logging on: TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0
TF_CPP_MIN_LOG_LEVEL=0, the settings in Google's "A Developer's Guide to
Debugging JAX on Cloud TPUs" (developers.googleblog.com, 2026-01-05), which
also gives /tmp/tpu_logs as libtpu's log folder. HLO dumps use
XLA_FLAGS=--xla_dump_to=DIR (and --xla_dump_hlo_pass_re=.* for V3p), as in
https://openxla.org/xla/hlo_dumps.

Per variant, in OUT/<id>/: variant.json (command, exit code, signal, timeout,
wall time, the last 50 lines of the AlphaFold3 log, kernel segfault lines),
last50.txt, dmesg_tail.txt, tpu_logs.tgz (libtpu's logs for that variant,
passing ones too; each file cut to its last 20000 lines if the archive
passes BISECT_TPU_LOG_MAX_MB, 100), and for V3/V3p xla_dump_files.txt (every
dump file with its size, in write order) plus xla_dump.tgz if it is at most
BISECT_DUMP_MAX_MB (200) compressed. Harness variants also hold the harness's
usual records (session.json, runs.jsonl, run.json). OUT/host.txt records the
host (ulimit, memory, TPU/LIBTPU/XLA/JAX environment); OUT/setup_tpu_logs.tgz
the logs of setup's device check. OUT/bisect_summary.json and .txt are
rewritten after every variant, so a partial fetch still has them.

Exit status 0 when the bisection ran (whatever the variants did), 2 on an
internal error. Standard library only.
"""

import argparse
import csv
import io
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'harness'))
from run_af3 import scrub, sha256_file, utc_now  # noqa: E402

AF3 = REPO / 'third_party' / 'alphafold3'
PY = AF3 / '.venv' / 'bin' / 'python'
WEIGHTS = pathlib.Path(os.environ.get('AF3_WEIGHTS_DIR', '~/af3_weights_run')).expanduser()
TOY = REPO / 'af3_tpu' / 'inputs' / 'toy_118.json'
TPU_LOG_DIR = pathlib.Path(os.environ.get('BISECT_TPU_LOG_DIR', '/tmp/tpu_logs'))
DUMP_MAX_MB = float(os.environ.get('BISECT_DUMP_MAX_MB', '200'))
TPU_LOG_MAX_MB = float(os.environ.get('BISECT_TPU_LOG_MAX_MB', '100'))
MIN_START_S = float(os.environ.get('BISECT_MIN_START_S', '180'))
VERBOSE = {'TPU_STDERR_LOG_LEVEL': '0', 'TPU_MIN_LOG_LEVEL': '0', 'TF_CPP_MIN_LOG_LEVEL': '0'}

# kind: direct (run_alphafold.py, as the smoke test calls it) or harness
# (harness/run_af3.py --config tpu_xla, one target, seed 1). For harness runs
# the probe's settings are entry, log_compiles and a fresh cache; each
# variant names what it changes.
PROBE = {'entry': True, 'log_compiles': True, 'cache': 'fresh'}
VARIANTS = [
    {'id': 'V0', 'kind': 'direct', 'target': 'TOY118', 'recycles': 1, 'samples': 1, 'verbose': False,
     'what': 'exact smoke command on toy_118 (1 recycle, 1 sample); expected to pass'},
    {'id': 'V1', 'kind': 'harness', 'target': 'TOY118', **PROBE,
     'what': "toy_118 through the harness, probe settings"},
    {'id': 'V2', 'kind': 'direct', 'target': '7U3J', 'recycles': 1, 'samples': 1,
     'what': '7U3J, smoke-style direct command (1 recycle, 1 sample; no harness, wrapper, cache)'},
    {'id': 'V3', 'kind': 'direct', 'target': '7U3J', 'dump': 'default',
     'what': '7U3J direct, AF3 defaults (10 recycles, 5 samples), HLO dump'},
    {'id': 'V3p', 'kind': 'direct', 'target': '7U3J', 'dump': 'passes', 'only_if_crashed': 'V3',
     'what': 'V3 with an HLO dump after every pass (runs only if V3 crashed)'},
    {'id': 'V4', 'kind': 'harness', 'target': '7U3J', **PROBE, 'samples': 1,
     'what': '7U3J harness, probe settings but 1 diffusion sample'},
    {'id': 'V5', 'kind': 'harness', 'target': '7U3J', **PROBE, 'recycles': 1,
     'what': '7U3J harness, probe settings but 1 recycle'},
    {'id': 'V6', 'kind': 'harness', 'target': '7U3J', **dict(PROBE, cache='none'),
     'what': '7U3J harness, probe settings without the persistent compile cache'},
    {'id': 'V7', 'kind': 'harness', 'target': '7U3J', **dict(PROBE, entry=False),
     'what': '7U3J harness, probe settings without the af3_entry.py wrapper'},
    {'id': 'V8', 'kind': 'harness', 'target': '7U3J', **dict(PROBE, log_compiles=False),
     'what': '7U3J harness, probe settings without JAX_LOG_COMPILES'},
    {'id': 'V9', 'kind': 'harness', 'target': '7D5C', **PROBE, 'from_first_pass': ['V2', 'V4', 'V5'],
     'what': '7D5C (1023 tokens) harness with the first passing setting among V2, V4, V5'},
]
# The recycles/samples setting each V9 source stands for.
V9_SETTINGS = {'V2': {'recycles': 1, 'samples': 1}, 'V4': {'samples': 1}, 'V5': {'recycles': 1}}
PLAN = {}  # set from --plan


def venv_python(name):
  return AF3 / f'.venv_{name}' / 'bin' / 'python' if name else PY


def prepare_venv(name, spec, out, cache={}):
  """Copies AlphaFold3's venv to .venv_NAME and installs spec['pip'] on top.

  Returns None on success, else the reason. Done once per name.
  """
  if name in cache:
    return cache[name]
  dest = AF3 / f'.venv_{name}'
  logf = out / f'venv_{name}.txt'
  uv = shutil.which('uv') or str(pathlib.Path.home() / '.local' / 'bin' / 'uv')
  steps = []
  if not dest.exists():
    steps.append(['cp', '-a', str(AF3 / '.venv'), str(dest)])
  install = [uv, 'pip', 'install', '--python', str(dest / 'bin' / 'python')] + list(spec['pip'])
  if spec.get('find_links'):
    install += ['-f', spec['find_links']]
  steps += [install]
  reason = None
  with open(logf, 'w') as f:
    for cmd in steps:
      f.write('$ ' + ' '.join(cmd) + '\n')
      f.flush()
      try:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
      except (OSError, subprocess.TimeoutExpired) as e:
        rc, reason = None, f'{type(e).__name__}: {e}'
      if rc != 0:
        reason = reason or f'{cmd[0]} {cmd[1]} exited {rc}'
        break
  if reason is None:
    freeze = subprocess.run([uv, 'pip', 'freeze', '--python', str(dest / 'bin' / 'python')],
                            capture_output=True, text=True)
    (out / f'venv_{name}_freeze.txt').write_text(freeze.stdout)
  cache[name] = reason
  log(f'venv {name}: {"ready" if reason is None else "FAILED: " + reason}')
  return reason


def log(msg):
  print(f'>> [{time.strftime("%H:%M:%S", time.gmtime())}] bisect: {msg}', flush=True)


def tail_lines(path, n):
  try:
    with open(path, errors='replace') as f:
      return f.read().splitlines()[-n:]
  except OSError as e:
    return [f'(no log: {e})']


def sh(cmd, timeout=60):
  """Output of a shell command (stdout and stderr), never raises."""
  try:
    out = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=timeout)
    return out.stdout + out.stderr
  except (OSError, subprocess.TimeoutExpired) as e:
    return f'({type(e).__name__}: {e})'


def archive_tpu_logs(dest_tgz):
  """Moves libtpu's logs into dest_tgz and empties the log folder."""
  if not TPU_LOG_DIR.is_dir():
    return f'no {TPU_LOG_DIR}'
  files = sorted(p for p in TPU_LOG_DIR.rglob('*') if p.is_file() and not p.is_symlink())
  with tarfile.open(dest_tgz, 'w:gz') as tar:
    for p in files:
      tar.add(p, arcname=f'tpu_logs/{p.relative_to(TPU_LOG_DIR)}')
  note = f'{len(files)} files'
  if dest_tgz.stat().st_size > TPU_LOG_MAX_MB * 1e6:
    with tarfile.open(dest_tgz, 'w:gz') as tar:
      for p in files:
        data = '\n'.join(tail_lines(p, 20000)).encode()
        info = tarfile.TarInfo(f'tpu_logs/{p.relative_to(TPU_LOG_DIR)}.last20000')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    note += f'; over {TPU_LOG_MAX_MB:g} MB compressed, kept the last 20000 lines of each'
  for p in TPU_LOG_DIR.iterdir():
    if p.is_dir() and not p.is_symlink():
      shutil.rmtree(p, ignore_errors=True)
    else:
      p.unlink(missing_ok=True)
  return note


def archive_dump(dump_dir, vdir):
  """xla_dump_files.txt always; xla_dump.tgz only up to DUMP_MAX_MB."""
  if not dump_dir.is_dir():
    return {'dump': 'no dump folder was written'}
  files = sorted((p for p in dump_dir.rglob('*') if p.is_file()), key=lambda p: p.stat().st_mtime_ns)
  total = sum(p.stat().st_size for p in files)
  with open(vdir / 'xla_dump_files.txt', 'w') as f:
    f.write(f'# {len(files)} files, {total} bytes, in write order: bytes name\n')
    for p in files:
      f.write(f'{p.stat().st_size} {p.relative_to(dump_dir)}\n')
  tgz = vdir / 'xla_dump.tgz'
  with tarfile.open(tgz, 'w:gz') as tar:
    tar.add(dump_dir, arcname='xla_dump')
  size = tgz.stat().st_size
  kept = size <= DUMP_MAX_MB * 1e6
  if not kept:
    tgz.unlink()
  shutil.rmtree(dump_dir, ignore_errors=True)
  return {'dump_files': len(files), 'dump_bytes': total, 'dump_tgz_bytes': size,
          'dump_kept': kept, 'dump_last_file': str(files[-1].relative_to(dump_dir)) if files else None}


def make_inputs(out):
  """The bisection's manifest (7U3J, 7D5C, TOY118) and 7U3J cut to seed 1."""
  with open(REPO / 'inputs' / 'manifest.csv') as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, [r for r in reader if r['pdb_id'] in ('7U3J', '7D5C')]
  for r in rows:
    if sha256_file(REPO / r['input_path']) != r['input_sha256']:
      raise SystemExit(f'{r["input_path"]}: SHA-256 differs from the manifest')
  toy = dict.fromkeys(fields, '')
  toy.update(pdb_id='TOY118', pilot='no', input_path=os.path.relpath(TOY, REPO),
             input_sha256=sha256_file(TOY), input_bytes=str(TOY.stat().st_size),
             num_tokens='118', bucket='256', num_seeds='1')
  manifest = out / 'bisect_manifest.csv'
  with open(manifest, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows + [toy])
  inputs = out / 'inputs'
  inputs.mkdir(exist_ok=True)
  data = json.loads((REPO / 'data' / 'inputs' / '7U3J.json').read_text())
  data['modelSeeds'] = [1]
  (inputs / '7U3J_seed1.json').write_text(json.dumps(data, indent=2) + '\n')
  return manifest, {'TOY118': TOY, '7U3J': inputs / '7U3J_seed1.json'}


def run_direct(v, vdir, timeout_s, direct_inputs):
  vdir.mkdir(parents=True)
  cmd = [str(venv_python(v.get('venv'))), 'run_alphafold.py', f'--json_path={direct_inputs[v["target"]]}',
         f'--output_dir={vdir / "af3_output"}', f'--model_dir={WEIGHTS}', '--jax_backend=tpu',
         '--flash_attention_implementation=xla', '--run_data_pipeline=false']
  if v.get('recycles'):
    cmd.append(f'--num_recycles={v["recycles"]}')
  if v.get('samples'):
    cmd.append(f'--num_diffusion_samples={v["samples"]}')
  extra = dict(VERBOSE) if v.get('verbose', True) else {}
  extra.update(v.get('env', {}))
  if v.get('dump'):
    extra['XLA_FLAGS'] = f'--xla_dump_to={vdir / "xla_dump"}' + (
        ' --xla_dump_hlo_pass_re=.*' if v['dump'] == 'passes' else '')
  env = dict(os.environ, PYTHONUNBUFFERED='1', **extra)
  logf = vdir / 'run_alphafold.log'
  t0, timed_out = time.monotonic(), False
  with open(logf, 'w') as f:
    proc = subprocess.Popen(cmd, cwd=AF3, env=env, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
      rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
      os.killpg(proc.pid, signal.SIGKILL)
      rc, timed_out = proc.wait(), True
  outputs = [p for p in (vdir / 'af3_output').rglob('*.cif')] if (vdir / 'af3_output').exists() else []
  shown = [os.path.relpath(cmd[0], AF3)] + cmd[1:]
  return {'command': f'cd {AF3} && ' + ''.join(f'{k}={val} ' for k, val in extra.items())
                     + ' '.join(shown),
          'exit_code': rc, 'timed_out': timed_out, 'wall_seconds': round(time.monotonic() - t0, 1),
          'has_mmcif': bool(outputs), 'af3_log': logf}


def run_harness(v, vdir, timeout_s, session, manifest):
  setting = {k: v.get(k) for k in ('recycles', 'samples')}
  cmd = [sys.executable, 'harness/run_af3.py', '--config', 'tpu_xla', '--targets', v['target'],
         '--session', f'{session}/{v["id"]}', '--label', 'bisect',
         '--manifest', os.path.relpath(manifest, REPO), '--model_dir', str(WEIGHTS),
         '--seeds', '1', '--order', 'tokens', '--compile_cache', v['cache'],
         # The AlphaFold3 process gets the variant's time minus the harness's
         # own start-up (environment probe, weights hash).
         '--timeout_min', f'{max(60.0, timeout_s - 120) / 60:.2f}']
  if v['entry']:
    cmd.append('--entry')
  if v['log_compiles']:
    cmd.append('--log_compiles')
  if setting['recycles']:
    cmd += ['--num_recycles', str(setting['recycles'])]
  if setting['samples']:
    cmd += ['--num_diffusion_samples', str(setting['samples'])]
  if v.get('venv'):
    cmd += ['--python', str(venv_python(v['venv']))]
  extra = dict(VERBOSE, **v.get('env', {}))
  env = dict(os.environ, **extra)
  hlog = vdir.parent / f'{v["id"]}.harness.log'
  t0, timed_out = time.monotonic(), False
  with open(hlog, 'w') as f:
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
      hrc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
      os.killpg(proc.pid, signal.SIGKILL)
      hrc, timed_out = proc.wait(), True
  vdir.mkdir(parents=True, exist_ok=True)
  shutil.move(str(hlog), vdir / 'harness.log')
  run = vdir / v['target'] / 'run.json'
  rec = json.loads(run.read_text()) if run.exists() else None
  return {'command': f'cd {REPO} && ' + ''.join(f'{k}={val} ' for k, val in extra.items())
                     + ' '.join(['python3'] + cmd[1:]),
          'harness_exit_code': hrc,
          'exit_code': rec['exit_code'] if rec else None,
          'timed_out': timed_out or bool(rec and rec['timed_out']),
          'wall_seconds': round(time.monotonic() - t0, 1),
          'has_mmcif': bool(rec and any(o.endswith('.cif') for o in rec['outputs'])),
          'harness_failure': rec['failure'] if rec else 'no run.json (harness did not finish)',
          'compile': rec['compile'] if rec else None, 'memory': rec.get('memory') if rec else None,
          'af3_log': vdir / v['target'] / 'run_alphafold.log' if rec else vdir / 'harness.log'}


def result_of(r):
  if r.get('skipped'):
    return f'skipped: {r["skipped"]}'
  if r.get('timed_out'):
    return 'timeout'
  rc = r.get('exit_code')
  if rc is None:
    return f'error: {r.get("harness_failure") or r.get("error")}'
  if rc < 0:
    return f'signal {-rc}'
  if rc == 0 and r.get('has_mmcif'):
    return 'pass'
  return f'exit {rc}'


def readings(by_id):
  """Plain conclusions from the pattern of results; None where it says nothing."""
  res = {k: result_of(v) for k, v in by_id.items()}
  crashed = lambda k: res.get(k, '').startswith('signal')  # noqa: E731
  passed = lambda k: res.get(k) == 'pass'  # noqa: E731
  out = []
  if 'V0' in res and not passed('V0'):
    out.append('V0 (the smoke command) did not pass: the VM or setup is suspect; the rest is not conclusive.')
  if crashed('V1') and passed('V2'):
    out.append('Not the input: toy_118 crashes with the probe settings and 7U3J passes with the smoke settings.')
  if crashed('V3'):
    out.append('Harness, wrapper, cache and JAX_LOG_COMPILES are not needed: the direct call with AF3 defaults crashes.')
  if passed('V4') and crashed('V5'):
    out.append('The 5 diffusion samples trigger it (1 sample passes, 1 recycle with 5 samples crashes).')
  if passed('V5') and crashed('V4'):
    out.append('The 10 recycles trigger it (1 recycle passes, 1 sample with 10 recycles crashes).')
  if passed('V4') and passed('V5'):
    out.append('It needs both 5 samples and 10 recycles (each alone at 1 passes).')
  if crashed('V4') and crashed('V5'):
    out.append('Neither change alone avoids it (1 sample crashes, 1 recycle crashes).')
  for k, what in (('V6', 'the persistent cache'), ('V7', 'the af3_entry.py wrapper'),
                  ('V8', 'JAX_LOG_COMPILES')):
    if crashed('V1') and passed(k):
      out.append(f'Turning off {what} avoids it ({k} passes).')
  return out


def plan_readings(by_id):
  """Readings for a --plan run: the samples series and each compare_to pair."""
  res = {k: result_of(v) for k, v in by_id.items()}
  crashed = lambda k: res.get(k, '').startswith('signal')  # noqa: E731
  passed = lambda k: res.get(k) == 'pass'  # noqa: E731
  out = []
  series = [(v.get('samples') or 5, v['id']) for v in PLAN.get('variants', [])
            if v.get('series') == 'samples' and v['id'] in res and not res[v['id']].startswith('skipped')]
  if series:
    ok = sorted(n for n, k in series if passed(k))
    bad = sorted(n for n, k in series if crashed(k))
    other = sorted(f'{n} ({res[k]})' for n, k in series if not passed(k) and not crashed(k))
    out.append(f'Diffusion samples that compile and run: {ok or "none"}; that segfault: {bad or "none"}'
               + (f'; other: {", ".join(other)}' if other else '') + '.')
  for v in PLAN.get('variants', []):
    ref, k = v.get('compare_to'), v['id']
    if not ref or k not in res or ref not in res or res[k].startswith('skipped'):
      continue
    if crashed(ref) and passed(k):
      out.append(f'{k} avoids the crash: {v["what"]} passes where {ref} segfaults.')
    elif crashed(ref) and crashed(k):
      out.append(f'{k} does not avoid it: {v["what"]} segfaults like {ref}.')
    elif crashed(ref):
      out.append(f'{k}: {res[k]} ({v["what"]}); see its harness.log / last50.txt.')
  return out


def write_summary(out, records):
  by_id = {r['id']: r for r in records}
  slim = [{k: v for k, v in r.items() if k not in ('last_50_lines', 'af3_log')} for r in records]
  for r in slim:
    r['result'] = result_of(r)
  summary = {'updated_utc': utc_now(), 'plan': PLAN.get('name'), 'variants': slim,
             'readings': plan_readings(by_id) if PLAN else readings(by_id)}
  (out / 'bisect_summary.json').write_text(json.dumps(scrub(summary), indent=2) + '\n')
  w = max([4] + [len(r['id']) for r in slim])
  lines = [f'{"id":{w}} {"result":38} {"wall_s":>7}  what']
  for r in slim:
    lines.append(f'{r["id"]:{w}} {r["result"][:38]:38} {r.get("wall_seconds") or "":>7}  {r["what"]}')
  lines += [''] + [f'reading: {x}' for x in summary['readings']]
  (out / 'bisect_summary.txt').write_text('\n'.join(lines) + '\n')
  return lines


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--out', help='results/af3/<session>, inside the checkout')
  ap.add_argument('--session')
  ap.add_argument('--job_start', type=float, help='epoch seconds; the budget counts from here')
  ap.add_argument('--budget_min', type=float, default=80)
  ap.add_argument('--timeout_min', type=float, default=15)
  ap.add_argument('--list', action='store_true', help='print the variants and exit')
  ap.add_argument('--plan', help='JSON plan with its own variants (default: built-in V0 to V9)')
  args = ap.parse_args()
  variants = VARIANTS
  if args.plan:
    PLAN.update(json.loads((REPO / args.plan).read_text()))
    PLAN.setdefault('name', pathlib.Path(args.plan).stem)
    variants = PLAN['variants']
    for v in variants:
      if v.get('venv') and v['venv'] not in PLAN.get('venvs', {}):
        raise SystemExit(f'plan: variant {v["id"]} uses venv {v["venv"]}, not in "venvs"')
  if args.list:
    w = max(len(v['id']) for v in variants)
    for v in variants:
      print(f'{v["id"]:{w}} {v["what"]}')
    return
  if not (args.out and args.session and args.job_start):
    ap.error('--out, --session and --job_start are required')
  out = (REPO / args.out).resolve() if not os.path.isabs(args.out) else pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  deadline = args.job_start + args.budget_min * 60

  (out / 'host.txt').write_text(sh(
      'echo "## uname"; uname -srvmo; echo "## ulimit -a"; ulimit -a; echo "## nproc"; nproc; '
      'echo "## free -g"; free -g; echo "## environment (TPU, LIBTPU, XLA, JAX)"; '
      'env | grep -E "^(TPU|LIBTPU|XLA|JAX)" | sort; '
      f'echo "## {TPU_LOG_DIR}"; ls -la {TPU_LOG_DIR} 2>&1 | head -20'))
  log(f'setup device-check logs: {archive_tpu_logs(out / "setup_tpu_logs.tgz")}')
  manifest, direct_inputs = make_inputs(out)
  log(f'budget {args.budget_min:g} min from job start ({(deadline - time.time()) / 60:.1f} min left), '
      f'{args.timeout_min:g} min per variant')

  records, by_id = [], {}
  for v in variants:
    rec = {'id': v['id'], 'what': v['what'], 'kind': v['kind'], 'target': v['target']}
    vdir = out / v['id']
    left = deadline - time.time()
    if v.get('only_if_crashed') and not result_of(by_id.get(v['only_if_crashed'], {})).startswith('signal'):
      rec['skipped'] = f'{v["only_if_crashed"]} did not crash'
    elif v.get('from_first_pass'):
      src = next((k for k in v['from_first_pass'] if result_of(by_id.get(k, {})) == 'pass'), None)
      if src is None:
        rec['skipped'] = f'none of {", ".join(v["from_first_pass"])} passed'
      else:
        v = dict(v, **V9_SETTINGS[src])
        rec['setting_from'] = src
    if 'skipped' not in rec and left < MIN_START_S:
      rec['skipped'] = f'budget ({left / 60:.1f} min left)'
    if 'skipped' not in rec and v.get('venv'):
      why = prepare_venv(v['venv'], PLAN['venvs'][v['venv']], out)
      if why:
        rec['skipped'] = f'venv {v["venv"]} not ready: {why}'[:200]
      left = deadline - time.time()
      if 'skipped' not in rec and left < MIN_START_S:
        rec['skipped'] = f'budget ({left / 60:.1f} min left)'
    if 'skipped' in rec:
      log(f'{v["id"]}: skipped ({rec["skipped"]})')
      records.append(rec)
      by_id[v['id']] = rec
      write_summary(out, records)
      continue
    timeout_s = min(args.timeout_min * 60, left)
    rec.update(timeout_seconds=round(timeout_s), recycles=v.get('recycles'), samples=v.get('samples'),
               env=v.get('env'), venv=v.get('venv'), start_utc=utc_now())
    log(f'{v["id"]}: {v["what"]} (timeout {timeout_s / 60:.1f} min)')
    try:
      if v['kind'] == 'direct':
        rec.update(run_direct(v, vdir, timeout_s, direct_inputs))
      else:
        rec.update(run_harness(v, vdir, timeout_s, args.session, manifest))
    except Exception as e:  # record and go on with the next variant
      vdir.mkdir(parents=True, exist_ok=True)
      rec.update(error=f'{type(e).__name__}: {e}'[:500], exit_code=None)
    rec['end_utc'] = utc_now()
    rc = rec.get('exit_code')
    rec['signal'] = -rc if isinstance(rc, int) and rc < 0 else None
    last = tail_lines(rec['af3_log'], 50) if rec.get('af3_log') else []
    rec['last_50_lines'] = last
    (vdir / 'last50.txt').write_text('\n'.join(last) + '\n')
    dmesg = sh('sudo -n dmesg -T 2>/dev/null || sudo -n dmesg 2>&1')
    (vdir / 'dmesg_tail.txt').write_text('\n'.join(dmesg.splitlines()[-50:]) + '\n')
    rec['kernel_segfault_lines'] = [x for x in dmesg.splitlines() if 'segfault' in x][-5:]
    rec['tpu_logs'] = archive_tpu_logs(vdir / 'tpu_logs.tgz')
    if v.get('dump'):
      rec.update(archive_dump(vdir / 'xla_dump', vdir))
    rec_out = {k: (str(val) if isinstance(val, pathlib.Path) else val) for k, val in rec.items()}
    (vdir / 'variant.json').write_text(json.dumps(scrub(rec_out), indent=2) + '\n')
    records.append(rec)
    by_id[v['id']] = rec
    log(f'{v["id"]}: {result_of(rec)} after {rec.get("wall_seconds")} s')
    write_summary(out, records)

  shutil.rmtree(out / 'inputs', ignore_errors=True)  # derived input, reproducible
  for line in write_summary(out, records):
    print(line, flush=True)
  log('done')


if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    raise
  except Exception as e:
    print(f'!! bisect: internal error: {type(e).__name__}: {e}', flush=True)
    sys.exit(2)
