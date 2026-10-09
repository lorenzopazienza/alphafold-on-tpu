"""Compares every harness run of a plan session, per target and sample, bit for bit and by RMSD.

    python3 harness/compare_runs.py --session results/af3/<session> \\
        [--reference results/af3/<other session>/<run>] [--out NAME]

For each harness run of the session (a folder with session.json) and each
target in it (<run>/<target>/af3_output/<target>):
  - against the reference run, if given and it has the target;
  - the session's first run against each later run (for example the same plan
    on two libtpu versions on one VM).
Each comparison is harness/compare_samples.py with --bitwise: per sample,
RMSD to every reference sample, whether sample k matches reference sample k,
and whether coordinates and confidences are bit-identical. Prints a table and
writes <session>/<NAME>.txt and .json (NAME default compare_runs). Used by
cloud/af3_run.sh after the fetch when the plan has a compare_reference.
Bit-for-bit needs the full *_confidences.json: with FETCH=light they stay
in the results bucket and show as "missing". Standard library only.
"""

import argparse
import io
import json
import pathlib
import sys
from contextlib import redirect_stdout

from compare_samples import compare, samples, verdict


def runs_of(session):
  return sorted((p for p in pathlib.Path(session).iterdir() if (p / 'session.json').is_file()),
                key=lambda p: (json.loads((p / 'session.json').read_text()).get('created_utc', ''), p.name))


def target_dirs(run):
  out = {}
  for t in sorted(p for p in run.iterdir() if (p / 'af3_output').is_dir()):
    d = t / 'af3_output' / t.name
    if samples(d):
      out[t.name] = d
  return out


def summarize(rows):
  """One line: prefix verdict, RMSD of same-index samples, bitwise counts."""
  same = [r['rmsd_to_ref'][r['sample']] for r in rows if r['sample'] in r['rmsd_to_ref']]
  bits = [r['bitwise'] for r in rows if 'bitwise' in r]
  ident = sum(b['coordinates'] == 'identical' for b in bits)
  conf = [b['confidences']['status'] for b in bits]
  summ = [b['summary_confidences']['status'] for b in bits]
  count = lambda xs, v: sum(x == v for x in xs)  # noqa: E731
  return {
      'prefix_holds': verdict(rows),
      'same_index_rmsd_min': min(same) if same else None, 'same_index_rmsd_max': max(same) if same else None,
      'samples_compared_bitwise': len(bits), 'coordinates_identical': ident,
      'confidences_identical': count(conf, 'identical'), 'confidences_missing': count(conf, 'missing'),
      'summary_identical': count(summ, 'identical'), 'summary_missing': count(summ, 'missing'),
  }


def line(label, s):
  rng = ('-' if s['same_index_rmsd_min'] is None
         else f"{s['same_index_rmsd_min']:.3f}-{s['same_index_rmsd_max']:.3f}")
  n = s['samples_compared_bitwise']
  return (f'  {label:58} RMSD same sample {rng:>13} A | bit-identical: coordinates {s["coordinates_identical"]}/{n},'
          f' confidences {s["confidences_identical"]}/{n}'
          f'{" (" + str(s["confidences_missing"]) + " missing)" if s["confidences_missing"] else ""},'
          f' summary {s["summary_identical"]}/{n} | prefix {s["prefix_holds"].split(" ")[0]}')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--session', required=True)
  ap.add_argument('--reference', help='a harness run folder of another session')
  ap.add_argument('--out', default='compare_runs')
  args = ap.parse_args()
  session = pathlib.Path(args.session)
  runs = runs_of(session)
  if not runs:
    raise SystemExit(f'no harness runs (session.json) in {session}')
  ref = pathlib.Path(args.reference) if args.reference else None
  report, lines = {'session': str(session), 'reference': args.reference, 'comparisons': []}, []

  def add(label, ref_dir, run_dir):
    with redirect_stdout(io.StringIO()):
      rows = compare(ref_dir, run_dir, bitwise=True)
    s = summarize(rows)
    report['comparisons'].append({'label': label, 'reference': str(ref_dir), 'run': str(run_dir),
                                  'summary': s, 'samples': rows})
    lines.append(line(label, s))

  if ref is not None:
    lines.append(f'Against the reference {ref}:')
    ref_targets = target_dirs(ref) if ref.is_dir() else {}
    if not ref_targets:
      lines.append('  (reference has no target outputs on this machine)')
    for run in runs:
      for t, d in target_dirs(run).items():
        if t in ref_targets:
          add(f'{run.name} {t}', ref_targets[t], d)
  if len(runs) > 1:
    lines.append(f'Against the session\'s first run {runs[0].name}:')
    base = target_dirs(runs[0])
    for run in runs[1:]:
      for t, d in target_dirs(run).items():
        if t in base:
          add(f'{run.name} {t}', base[t], d)
  if not report['comparisons']:
    lines.append('  (nothing to compare: no runs with per-sample outputs)')
  text = '\n'.join(lines) + '\n'
  print(text, end='')
  (session / f'{args.out}.txt').write_text(text)
  (session / f'{args.out}.json').write_text(json.dumps(report, indent=2) + '\n')
  return 0


if __name__ == '__main__':
  sys.exit(main())
