"""Builds the tidy per-sample table of all AlphaFold3 runs (notes/analysis_plan.md, sections 4 and 6).

    analysis/.venv/bin/python analysis/collect.py [--root results/af3] [--root COPY_OF_BUCKET/af3] \\
        [--out results/analysis/<name>] [--jobs 4] [--no_m1]

Walks every harness run folder (one with session.json and runs.jsonl) under
the roots: results/af3/ by default, or a local copy of the results bucket
(gcloud storage rsync gs://BUCKET/af3 DIR; this script reads local files
only). One row per (platform, attention, libtpu, target, seed, repetition,
sample), with:
  - condition: platform code (V5, V6, G, GT, C; plan section 3), attention,
    libtpu version and build label, device kind, zone and machine type, and
    the default matmul precision (JAX_DEFAULT_MATMUL_PRECISION as recorded
    in session.json xla_env, empty if unset). The precision-matched arms of
    plan section 11 (configs tpu_xla_highest and l4_xla_highest) set it to
    highest and get the code with an "h": V6h, Gh. Any other value is kept
    apart as CODE[value], never merged with the default arm;
  - repetition: r1, r2, ... (fresh-cache repetitions) or w (warm rerun), from
    the run name harness/run_plan.py gives; num_recycles and
    num_diffusion_samples (AlphaFold3 defaults 10 and 5 when not set);
  - per sample: ranking_score, ptm, iptm, has_clash, fraction_disordered
    (summary confidences), mean pLDDT (mean of the per-atom pLDDT AlphaFold3
    writes in the mmCIF B-factor column), the model path, and M1 (ligand RMSD,
    PoseBusters validity, success, error; analysis/score_ligand.py);
  - per run: exit code, failure, wall, inference and compile seconds, compiled
    memory (peak, temp, arguments, needed), allocator peak reservation and
    peak in use, the device memory limit, max RSS;
  - weights file and SHA-256, input SHA-256 (frozen and derived), session id.
A run that failed (no model files) still gets one row per seed, with the
failure and empty sample columns (plan section 8: never dropped silently).

Outputs in --out (default results/analysis/collect, gitignored like all of
results/, so nothing produced with official weights reaches git):
samples.csv and m1_cache.jsonl (M1 by model-file SHA-256, reused on reruns).
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))

PLATFORM_BY_KIND = {'TPU v5 lite': 'V5', 'TPU v6 lite': 'V6', 'cpu': 'C'}
COLUMNS = [
    'session', 'run', 'label', 'platform', 'attention', 'matmul_precision', 'device_kind', 'libtpu',
    'libtpu_build', 'jax',
    'zone', 'machine_type', 'target', 'num_tokens', 'bucket', 'seed', 'repetition', 'cache', 'sample',
    'num_recycles', 'num_diffusion_samples', 'exit_code', 'failure', 'ranking_score', 'ptm', 'iptm',
    'mean_plddt', 'has_clash', 'fraction_disordered', 'm1_ligand_rmsd', 'm1_pb_valid', 'm1_success',
    'm1_pb_failed', 'm1_error', 'model_path', 'model_sha256', 'wall_s', 'inference_s', 'compile_s',
    'compiled_peak_bytes', 'compiled_temp_bytes', 'compiled_args_bytes', 'compiled_needed_bytes',
    'alloc_peak_reserved_bytes', 'alloc_peak_in_use_bytes', 'device_bytes_limit', 'max_rss_bytes', 'weights_file',
    'weights_sha256', 'input_sha256', 'run_input_sha256',
]


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def platform_of(kind, attention, matmul_precision=None):
  """Condition code: section 3's platform code, with "h" for the section 11 arms (highest precision)."""
  if 'L4' in kind:
    code = 'GT' if attention == 'triton' else 'G'
  else:
    code = PLATFORM_BY_KIND.get(kind, kind)
  if not matmul_precision:
    return code
  # No "_" in the code: contrasts.py --condition platform_libtpu splits the condition at the first "_".
  return f'{code}h' if matmul_precision == 'highest' else f'{code}[{matmul_precision.replace("_", "-")}]'


def repetition_of(run_name, cache_mode):
  m = re.search(r'_rep(\d+)_seed\d+_fresh$', run_name)
  if m:
    return f'r{m.group(1)}'
  if re.search(r'_seed\d+_warm$', run_name) or cache_mode == 'warm':
    return 'w'
  return 'r1'


def mean_plddt(cif):
  """Mean of the B-factor column (AlphaFold3 writes per-atom pLDDT there) of the mmCIF atom_site loop."""
  lines = pathlib.Path(cif).read_text().splitlines()
  i = next(n for n, l in enumerate(lines) if l.startswith('_atom_site.'))
  fields = []
  while lines[i].startswith('_atom_site.'):
    fields.append(lines[i].split('.', 1)[1].strip())
    i += 1
  col = fields.index('B_iso_or_equiv')
  vals = []
  while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', 'loop_', '_')):
    vals.append(float(lines[i].split()[col]))
    i += 1
  return round(sum(vals) / len(vals), 4) if vals else None


def run_folders(roots):
  for root in roots:
    for ses in sorted(pathlib.Path(root).rglob('session.json')):
      if (ses.parent / 'runs.jsonl').exists():
        yield ses.parent


def _m1(args):
  target, path = args
  import score_ligand
  return score_ligand.score(target, path)


def rows_of_run(folder, manifest):
  session = json.loads((folder / 'session.json').read_text())
  probe = session.get('probe') or {}
  cfg = session.get('config') or {}
  sargs = session.get('args') or {}
  attention = cfg.get('flash_attention_implementation')
  kind = probe.get('device_kind', '')
  # What the run's environment held (run_af3.py records XLA_, JAX_ and TF_ variables).
  precision = (session.get('xla_env') or {}).get('JAX_DEFAULT_MATMUL_PRECISION') or None
  tpu = session.get('tpu_runtime') or {}
  host = session.get('host') or {}
  base = {
      'session': str(folder.relative_to(REPO)) if folder.resolve().is_relative_to(REPO) else folder.name,
      'run': folder.name, 'label': session.get('label'), 'platform': platform_of(kind, attention, precision),
      'attention': attention, 'matmul_precision': precision, 'device_kind': kind,
      'libtpu': (probe.get('packages') or {}).get('libtpu'),
      'libtpu_build': tpu.get('build_label'), 'jax': (probe.get('packages') or {}).get('jax'),
      'zone': host.get('gce_zone'), 'machine_type': host.get('gce_machine_type') or host.get('gce_accelerator_type'),
      'num_recycles': sargs.get('num_recycles') or 10,
      'num_diffusion_samples': sargs.get('num_diffusion_samples') or 5,
      'cache': (session.get('compile_cache') or {}).get('mode'),
      'weights_file': (session.get('weights') or {}).get('file'),
      'weights_sha256': (session.get('weights') or {}).get('sha256'),
  }
  base['repetition'] = repetition_of(folder.name, base['cache'])
  out, todo = [], []
  for line in (folder / 'runs.jsonl').read_text().splitlines():
    r = json.loads(line)
    t = r['pdb_id']
    m = r.get('memory') or {}
    stats = r.get('entry_stats') or {}
    compiled = [c for c in (stats.get('compiled_memory') or []) if 'error' not in c]
    c = compiled[-1] if compiled else {}
    dev = stats.get('device_memory_stats') or {}
    run_cols = dict(base, target=t, num_tokens=r.get('num_tokens'), bucket=r.get('bucket'),
                    exit_code=r.get('exit_code'), failure=r.get('failure'), wall_s=r.get('wall_seconds'),
                    compile_s=(r.get('compile') or {}).get('model_seconds'),
                    compiled_peak_bytes=c.get('peak_memory_in_bytes'), compiled_temp_bytes=c.get('temp_size_in_bytes'),
                    compiled_args_bytes=c.get('argument_size_in_bytes'),
                    compiled_needed_bytes=c.get('device_bytes_needed'),
                    alloc_peak_reserved_bytes=dev.get('peak_bytes_reserved', m.get('peak_bytes_reserved')),
                    alloc_peak_in_use_bytes=dev.get('peak_bytes_in_use', m.get('peak_bytes_in_use')),
                    device_bytes_limit=dev.get('bytes_limit', m.get('bytes_limit')),
                    max_rss_bytes=stats.get('max_rss_bytes', m.get('max_rss_bytes')),
                    input_sha256=r.get('input_sha256'), run_input_sha256=r.get('run_input_sha256'))
    outdir = REPO / r['output_dir'] / t if not pathlib.Path(r['output_dir']).is_absolute() else pathlib.Path(r['output_dir']) / t
    if not outdir.exists():
      outdir = folder / t / 'af3_output' / t
    for seed in r.get('seeds') or []:
      inf = (r.get('inference_seconds_by_seed') or {}).get(str(seed))
      samples = sorted(outdir.glob(f'seed-{seed}_sample-*')) if outdir.exists() else []
      if not samples:
        out.append(dict(run_cols, seed=seed, inference_s=inf, sample=None))
        continue
      for d in samples:
        k = int(d.name.rsplit('-', 1)[1])
        row = dict(run_cols, seed=seed, inference_s=inf, sample=k)
        summ = next((p for p in d.glob('*_summary_confidences.json')), None)
        if summ:
          s = json.loads(summ.read_text())
          row.update(ranking_score=s.get('ranking_score'), ptm=s.get('ptm'), iptm=s.get('iptm'),
                     has_clash=s.get('has_clash'), fraction_disordered=s.get('fraction_disordered'))
        cif = next((p for p in d.glob('*_model.cif')), None)
        if cif:
          row['model_path'] = str(cif.resolve().relative_to(REPO)) if cif.resolve().is_relative_to(REPO) else str(cif)
          row['model_sha256'] = sha256(cif)
          row['mean_plddt'] = mean_plddt(cif)
          todo.append((len(out), t, cif))
        out.append(row)
  return out, todo


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--root', action='append', help='folder holding sessions (default results/af3)')
  ap.add_argument('--out', default='results/analysis/collect')
  ap.add_argument('--jobs', type=int, default=4)
  ap.add_argument('--no_m1', action='store_true', help='skip M1 (quick look)')
  args = ap.parse_args()
  roots = [REPO / r if not pathlib.Path(r).is_absolute() else pathlib.Path(r) for r in (args.root or ['results/af3'])]
  out_dir = REPO / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  manifest = {r['pdb_id']: r for r in csv.DictReader(open(REPO / 'inputs' / 'manifest.csv'))}

  rows, todo = [], []
  for folder in run_folders(roots):
    r, t = rows_of_run(folder, manifest)
    todo += [(len(rows) + i, target, cif) for i, target, cif in t]
    rows += r
  print(f'>> {len(rows)} rows from {len({r["session"] for r in rows})} harness runs; '
        f'{len(todo)} models to score for M1', flush=True)

  if not args.no_m1 and todo:
    cache_path = out_dir / 'm1_cache.jsonl'
    cache = {}
    if cache_path.exists():
      for line in cache_path.read_text().splitlines():
        c = json.loads(line)
        cache[(c['model_sha256'], c['target'])] = c
    need = [(i, t, cif) for i, t, cif in todo if (rows[i]['model_sha256'], t) not in cache]
    if need:
      with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_m1, [(t, str(cif)) for _, t, cif in need], chunksize=4))
      with open(cache_path, 'a') as f:
        for (i, t, _), res in zip(need, results):
          res['model_sha256'] = rows[i]['model_sha256']
          cache[(res['model_sha256'], t)] = res
          f.write(json.dumps(res) + '\n')
    for i, t, _ in todo:
      res = cache[(rows[i]['model_sha256'], t)]
      rows[i].update(m1_ligand_rmsd=res['ligand_rmsd'], m1_pb_valid=res['pb_valid'], m1_success=res['success'],
                     m1_pb_failed=';'.join(res['pb_failed'] or []), m1_error=res['error'])
    print(f'>> M1: {len(todo) - len(need)} from the cache, {len(need)} scored now', flush=True)

  with open(out_dir / 'samples.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
  print(f'>> wrote {(out_dir / "samples.csv").relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
