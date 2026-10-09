"""Compares AlphaFold3 per-sample structures of runs against a reference run.

    python3 harness/compare_samples.py --ref REF_DIR [--ref_label TEXT] \\
        --run LABEL=RUN_DIR [--run LABEL=RUN_DIR ...] [--bitwise] [--json OUT.json]

Each DIR is an AlphaFold3 output folder of one target (the one holding
seed-<s>_sample-<k>/ folders). For every sample k of every run, and every
sample j of the reference with the same seed, it prints the coordinate RMSD
in Angstrom over all atoms (same atoms, same order required; no
superposition, since the runs share the input frame).

Why: with JAX's default partitionable threefry PRNG (jax 0.10.2), a run with
k diffusion samples draws exactly the noise and keys of samples 0..k-1 of a
5-sample run (AlphaFold3 diffusion_head.sample: jax.random.normal over
(num_samples, ...) and jax.random.split(key, num_samples)). So sample k of a
smaller run should match reference sample k up to hardware numerics (v5e 1
sample against v6e sample 0: 0.94 A, 2026-10-09) and differ from the others
by tens of Angstrom. "prefix holds" means every judged sample k is closest to
reference sample k, and that by a margin (nearest other sample at least 10
times farther). A sample k with no reference sample k (a 1-sample reference
against a 5-sample run) is not judged: it shows "n/a".

--bitwise also compares, per sample with a reference sample of the same
index, the mmCIF coordinates as written (text of Cartn_x/y/z) and the
sample's *_confidences.json and *_summary_confidences.json: identical,
differing (how many values, largest absolute difference) or missing (for
example left in the results bucket by FETCH=light). Standard library only.
"""

import argparse
import json
import math
import pathlib
import re
import sys


def read_coords(path):
  """(atom keys, xyz floats, xyz as written) from an mmCIF _atom_site loop."""
  lines = pathlib.Path(path).read_text().splitlines()
  i = next(n for n, l in enumerate(lines) if l.startswith('_atom_site.'))
  fields = []
  while lines[i].startswith('_atom_site.'):
    fields.append(lines[i].split('.', 1)[1].strip())
    i += 1
  col = {f: n for n, f in enumerate(fields)}
  keys, xyz, text = [], [], []
  while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', 'loop_', '_')):
    r = lines[i].split()
    keys.append((r[col['label_asym_id']], r[col['label_seq_id']], r[col['label_atom_id']]))
    raw = (r[col['Cartn_x']], r[col['Cartn_y']], r[col['Cartn_z']])
    text.append(raw)
    xyz.append(tuple(float(v) for v in raw))
    i += 1
  return keys, xyz, text


def samples(folder):
  """{(seed, sample): sample folder} of an AlphaFold3 output folder (with a model.cif)."""
  out = {}
  for d in pathlib.Path(folder).glob('seed-*_sample-*'):
    m = re.fullmatch(r'seed-(\d+)_sample-(\d+)', d.name)
    if m and sorted(d.glob('*_model.cif')):
      out[(int(m.group(1)), int(m.group(2)))] = d
  return out


def model_cif(folder):
  return sorted(pathlib.Path(folder).glob('*_model.cif'))[0]


def rmsd(a, b):
  return math.sqrt(sum((x - y) ** 2 for p, q in zip(a, b) for x, y in zip(p, q)) / len(a))


def numbers(obj):
  """All numeric leaves of a JSON value, in a fixed order."""
  if isinstance(obj, bool) or obj is None:
    return []
  if isinstance(obj, (int, float)):
    return [float(obj)]
  if isinstance(obj, list):
    return [x for v in obj for x in numbers(v)]
  if isinstance(obj, dict):
    return [x for k in sorted(obj) for x in numbers(obj[k])]
  return []


def compare_json(a_dir, b_dir, suffix):
  """identical / differs ... / missing, for the sample's file ending in suffix."""
  pick = lambda d: [p for p in pathlib.Path(d).glob(f'*{suffix}')  # noqa: E731
                    if suffix != '_confidences.json' or not p.name.endswith('_summary_confidences.json')]
  a, b = pick(a_dir), pick(b_dir)
  if not a or not b:
    return {'status': 'missing', 'missing_in': 'run' if not a else 'reference'}
  if a[0].read_bytes() == b[0].read_bytes():
    return {'status': 'identical'}
  try:
    va, vb = numbers(json.loads(a[0].read_text())), numbers(json.loads(b[0].read_text()))
  except ValueError:
    return {'status': 'differs', 'note': 'not both valid JSON'}
  if len(va) != len(vb):
    return {'status': 'differs', 'note': f'{len(va)} against {len(vb)} values'}
  diffs = [abs(x - y) for x, y in zip(va, vb) if x != y]
  return {'status': 'differs' if diffs else 'identical values (formatting differs)',
          'values_differing': len(diffs), 'values': len(va), 'max_abs_diff': max(diffs) if diffs else 0.0}


def compare(ref_dir, run_dir, bitwise=False):
  ref = {k: (d, read_coords(model_cif(d))) for k, d in samples(ref_dir).items()}
  rows = []
  for (seed, k), folder in sorted(samples(run_dir).items()):
    keys, xyz, text = read_coords(model_cif(folder))
    dists = {}
    for (rseed, j), (_, (rkeys, rxyz, _)) in sorted(ref.items()):
      if rseed != seed:
        continue
      if rkeys != keys:
        raise SystemExit(f'{folder}: atoms differ from the reference sample {j}')
      dists[j] = round(rmsd(xyz, rxyz), 3)
    if not dists:
      continue
    nearest = min(dists, key=dists.get)
    if k in dists:
      others = [d for j, d in dists.items() if j != k]
      ok = nearest == k and (not others or min(others) >= 10 * max(dists[k], 0.01))
    else:
      ok = None  # no reference sample with this index: not judged
    row = {'seed': seed, 'sample': k, 'rmsd_to_ref': dists, 'nearest_ref_sample': nearest,
           'matches_same_index': ok}
    if bitwise and (seed, k) in ref:
      rdir, (_, rxyz, rtext) = ref[(seed, k)]
      differ = [n for n, (p, q) in enumerate(zip(text, rtext)) if p != q]
      row['bitwise'] = {
          'coordinates': 'identical' if not differ else 'differ',
          'atoms_differing': len(differ), 'atoms': len(text),
          'max_abs_diff': max((max(abs(a - b) for a, b in zip(xyz[n], rxyz[n])) for n in differ), default=0.0),
          'confidences': compare_json(folder, rdir, '_confidences.json'),
          'summary_confidences': compare_json(folder, rdir, '_summary_confidences.json')}
    rows.append(row)
  return rows


def verdict(rows):
  judged = [r for r in rows if r['matches_same_index'] is not None]
  if not judged:
    return 'n/a (no sample has a reference sample of the same index)'
  held = 'yes' if all(r['matches_same_index'] for r in judged) else 'NO'
  if len(judged) < len(rows):
    held += f' ({len(judged)} of {len(rows)} samples judged; the reference has no sample of the others\' index)'
  return held


def bitwise_line(b):
  conf = lambda c: c['status'] + (f" ({c['values_differing']}/{c['values']} values, max {c['max_abs_diff']:.3g})"  # noqa: E731
                                   if c['status'] == 'differs' and 'values' in c else '')
  coords = ('identical' if b['coordinates'] == 'identical'
            else f"{b['atoms_differing']}/{b['atoms']} atoms differ, max {b['max_abs_diff']:.3g} A")
  return f'coordinates {coords}; confidences {conf(b["confidences"])}; summary {conf(b["summary_confidences"])}'


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ref', required=True)
  ap.add_argument('--ref_label', default='reference')
  ap.add_argument('--run', action='append', default=[], help='LABEL=DIR, repeatable')
  ap.add_argument('--bitwise', action='store_true', help='also compare coordinates and confidences exactly')
  ap.add_argument('--json', help='also write the comparison as JSON')
  args = ap.parse_args()
  if not samples(args.ref):
    raise SystemExit(f'no seed-*_sample-* folders with an mmCIF in {args.ref}')
  report = {'reference': args.ref, 'reference_label': args.ref_label, 'runs': {}}
  print(f'Per-sample RMSD (A) against {args.ref_label}:')
  for spec in args.run:
    label, _, folder = spec.partition('=')
    if not samples(folder):
      print(f'  {label}: no per-sample mmCIFs (run did not finish?)')
      report['runs'][label] = None
      continue
    rows = compare(args.ref, folder, args.bitwise)
    report['runs'][label] = rows
    print(f'  {label}: {len(rows)} sample(s); prefix holds: {verdict(rows)}')
    for r in rows:
      cells = '  '.join(f'ref{j}={d:.2f}' for j, d in r['rmsd_to_ref'].items())
      mark = '' if r['matches_same_index'] is not None else '  (n/a: no reference sample ' + str(r['sample']) + ')'
      print(f'    seed {r["seed"]} sample {r["sample"]}: {cells}{mark}')
      if 'bitwise' in r:
        print(f'      bitwise: {bitwise_line(r["bitwise"])}')
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(report, indent=2) + '\n')
  return 0


if __name__ == '__main__':
  sys.exit(main())
