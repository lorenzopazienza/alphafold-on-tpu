"""Compares AlphaFold3 per-sample structures of runs against a reference run.

    python3 harness/compare_samples.py --ref REF_DIR [--ref_label TEXT] \\
        --run LABEL=RUN_DIR [--run LABEL=RUN_DIR ...] [--json OUT.json]

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
by tens of Angstrom. "prefix holds" means every sample k is closest to
reference sample k, and that by a margin (nearest other sample at least
10 times farther). Standard library only.
"""

import argparse
import json
import math
import pathlib
import re
import sys


def read_coords(path):
  """(atom keys, xyz list) from an mmCIF _atom_site loop."""
  lines = pathlib.Path(path).read_text().splitlines()
  i = next(n for n, l in enumerate(lines) if l.startswith('_atom_site.'))
  fields = []
  while lines[i].startswith('_atom_site.'):
    fields.append(lines[i].split('.', 1)[1].strip())
    i += 1
  col = {f: n for n, f in enumerate(fields)}
  keys, xyz = [], []
  while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', 'loop_', '_')):
    r = lines[i].split()
    keys.append((r[col['label_asym_id']], r[col['label_seq_id']], r[col['label_atom_id']]))
    xyz.append((float(r[col['Cartn_x']]), float(r[col['Cartn_y']]), float(r[col['Cartn_z']])))
    i += 1
  return keys, xyz


def samples(folder):
  """{(seed, sample): model.cif path} of an AlphaFold3 output folder."""
  out = {}
  for d in pathlib.Path(folder).glob('seed-*_sample-*'):
    m = re.fullmatch(r'seed-(\d+)_sample-(\d+)', d.name)
    cif = sorted(d.glob('*_model.cif'))
    if m and cif:
      out[(int(m.group(1)), int(m.group(2)))] = cif[0]
  return out


def rmsd(a, b):
  return math.sqrt(sum((x - y) ** 2 for p, q in zip(a, b) for x, y in zip(p, q)) / len(a))


def compare(ref_dir, run_dir):
  ref = {k: read_coords(p) for k, p in samples(ref_dir).items()}
  rows = []
  for (seed, k), path in sorted(samples(run_dir).items()):
    keys, xyz = read_coords(path)
    dists = {}
    for (rseed, j), (rkeys, rxyz) in sorted(ref.items()):
      if rseed != seed:
        continue
      if rkeys != keys:
        raise SystemExit(f'{path}: atoms differ from the reference sample {j}')
      dists[j] = round(rmsd(xyz, rxyz), 3)
    if not dists:
      continue
    nearest = min(dists, key=dists.get)
    others = [d for j, d in dists.items() if j != k]
    ok = (nearest == k and k in dists and (not others or min(others) >= 10 * max(dists[k], 0.01)))
    rows.append({'seed': seed, 'sample': k, 'rmsd_to_ref': dists, 'nearest_ref_sample': nearest,
                 'matches_same_index': ok})
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--ref', required=True)
  ap.add_argument('--ref_label', default='reference')
  ap.add_argument('--run', action='append', default=[], help='LABEL=DIR, repeatable')
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
    rows = compare(args.ref, folder)
    report['runs'][label] = rows
    held = all(r['matches_same_index'] for r in rows)
    print(f'  {label}: {len(rows)} sample(s); prefix holds: {"yes" if held else "NO"}')
    for r in rows:
      cells = '  '.join(f'ref{j}={d:.2f}' for j, d in r['rmsd_to_ref'].items())
      print(f'    seed {r["seed"]} sample {r["sample"]}: {cells}')
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(report, indent=2) + '\n')
  return 0


if __name__ == '__main__':
  sys.exit(main())
