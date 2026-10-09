"""Contrast statistics (analysis/contrasts.py) on a synthetic table with known answers."""

import argparse

import numpy as np
import pandas as pd
import pytest

import contrasts


def table(flip_targets_hw=(), flip_targets_rc=(), success=None):
  """3 targets x seeds 1, 2 x 5 samples on V6 (r1, r2, w) and G (r1); no model files (M4 not computed).

  In HW the top-1 sample of the targets in flip_targets_hw moves from 0 to 1;
  in RC(V6) that happens for flip_targets_rc. success: {(platform, target, sample): bool}.
  """
  success = success or {}
  rows = []
  for plat, reps in (('V6', ('r1', 'r2', 'w')), ('G', ('r1',))):
    for rep in reps:
      for t in ('T1', 'T2', 'T3'):
        for seed in (1, 2):
          for k in range(5):
            score = 0.9 - 0.1 * k
            flip = (plat == 'G' and rep == 'r1' and t in flip_targets_hw) or \
                (plat == 'V6' and rep == 'r2' and t in flip_targets_rc)
            if flip and k == 1:
              score = 0.95
            rows.append({'session': f'{plat}_{rep}', 'label': 'pilot', 'platform': plat, 'libtpu': None,
                         'target': t, 'seed': seed, 'repetition': rep, 'sample': k, 'num_recycles': 10,
                         'num_diffusion_samples': 5, 'exit_code': 0, 'failure': None, 'ranking_score': score,
                         'mean_plddt': 80.0, 'iptm': 0.8, 'm1_success': success.get((plat, t, k), k == 0),
                         'model_path': None})
  return pd.DataFrame(rows)


def run(df, tmp_path, replicates=500):
  path = tmp_path / 'samples.csv'
  df.to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=None, include=None, exclude=None, condition='platform',
                            replicates=replicates, seed=2026)
  d, ok = contrasts.load(args)
  seed_pairs, sample_pairs, cons, excluded = contrasts.build_pairs(ok)
  return args, d, ok, seed_pairs, sample_pairs, cons, excluded


def stat(summary, name, statistic):
  return summary[(summary.contrast == name) & (summary.statistic == statistic)].iloc[0]


def test_contrasts_found(tmp_path):
  _, _, _, _, _, cons, excluded = run(table(), tmp_path)
  assert set(cons) == {'RC(V6)', 'W(V6)', 'SD(V6)', 'SD(G)', 'HW(V6,G)'}
  assert cons['HW(V6,G)'] == ('HW', True)
  assert (excluded.excluded_only_a == 0).all() and (excluded.excluded_only_b == 0).all()


def test_flip_rates_and_flips_that_matter(tmp_path):
  # HW: T1 top-1 moves to sample 1, which is not a success (sample 0 is): a flip that matters.
  # T2 top-1 moves to sample 1, which is also a success on G: a flip that does not matter.
  df = table(flip_targets_hw=('T1', 'T2'), success={('G', 'T2', 1): True})
  args, _, _, sp, sm, cons, _ = run(df, tmp_path)
  s = contrasts.summarize(sp, sm, cons, args)
  assert stat(s, 'HW(V6,G)', 'flip_top1').estimate == pytest.approx(4 / 6)
  assert stat(s, 'HW(V6,G)', 'flip_success').estimate == pytest.approx(2 / 6)
  assert stat(s, 'HW(V6,G)', 'flip_matters').estimate == pytest.approx(2 / 6)
  assert stat(s, 'RC(V6)', 'flip_top1').estimate == 0
  assert stat(s, 'W(V6)', 'flip_top1').estimate == 0


def test_bootstrap_is_deterministic_and_brackets_the_estimate(tmp_path):
  args, _, _, sp, sm, cons, _ = run(table(flip_targets_hw=('T1',)), tmp_path)
  a = contrasts.summarize(sp, sm, cons, args)
  b = contrasts.summarize(sp, sm, cons, args)
  pd.testing.assert_frame_equal(a, b)
  r = stat(a, 'HW(V6,G)', 'flip_top1')
  assert r.ci_low <= r.estimate <= r.ci_high
  assert r.ci_low == 0 and r.ci_high == 1  # 1 of 3 targets flips both seeds: resamples give 0 to 3 of 3


def test_hw_minus_rc(tmp_path):
  args, _, _, sp, sm, cons, _ = run(table(flip_targets_hw=('T1', 'T2'), flip_targets_rc=('T1',)), tmp_path)
  h = contrasts.hw_minus_rc(sp, cons, args)
  r = h[(h.hardware == 'HW(V6,G)') & (h.reference == 'RC(P)') & (h.statistic == 'flip_top1')].iloc[0]
  assert r.recompilation == 'RC(V6)'
  assert r.estimate == pytest.approx(4 / 6 - 2 / 6)
  assert r.ci_low <= r.estimate <= r.ci_high


def test_missing_pairs_are_counted_not_dropped(tmp_path):
  df = table()
  df = df[~((df.platform == 'G') & (df.target == 'T3'))]
  _, _, _, sp, _, _, excluded = run(df, tmp_path)
  e = excluded.set_index('contrast').loc['HW(V6,G)']
  assert e.pairs_used == 4 and e.excluded_only_a == 2 and e.excluded_only_b == 0
  assert set(sp[sp.contrast == 'HW(V6,G)'].target) == {'T1', 'T2'}


def test_duplicate_sessions_are_refused(tmp_path):
  df = table()
  dup = df[df.platform == 'G'].copy()
  dup['session'] = 'G_r1_again'
  with pytest.raises(SystemExit, match='several sessions'):
    run(pd.concat([df, dup]), tmp_path)


def test_q3_seed_curve(tmp_path):
  args, _, ok, *_ = run(table(), tmp_path)
  rows, answer = contrasts.q3(ok, args)
  assert sorted(rows.n_seeds.unique()) == [1, 2]
  rates = rows[rows.statistic == 'success_rate']
  assert (rates.estimate == 1.0).all()
  assert answer == 1


def test_bit_identity_compares_coordinates_as_written(tmp_path):
  src = contrasts.REPO / 'data' / 'rcsb' / '7U3J.cif'
  if not src.exists():
    pytest.skip('data/rcsb/ not present')
  text = src.read_text()
  a, b, c = tmp_path / 'a.cif', tmp_path / 'b.cif', tmp_path / 'c.cif'
  a.write_text(text)
  b.write_text(text.replace('_atom_site.', '_atom_site.', 1))
  line = next(l for l in text.splitlines() if l.startswith('ATOM'))
  x = line.split()
  changed = line.replace(x[10], f'{float(x[10]) + 0.001:.3f}', 1)
  c.write_text(text.replace(line, changed, 1))
  assert contrasts.coords_text(str(a)) == contrasts.coords_text(str(b))
  assert contrasts.coords_text(str(a)) != contrasts.coords_text(str(c))


def test_q3_pairs_use_their_common_targets(tmp_path):
  # C-like subset: G runs only T1 and T2; V6 keeps its 3 targets, the pair uses 2.
  df = table()
  df = df[~((df.platform == 'G') & (df.target == 'T3'))]
  args, _, ok, *_ = run(df, tmp_path)
  rows, _ = contrasts.q3(ok, args)
  one = rows[rows.n_seeds == 2].set_index(['condition', 'statistic'])
  assert one.loc[('V6', 'success_rate')].n_targets == 3
  assert one.loc[('G', 'success_rate')].n_targets == 2
  assert one.loc[('V6-G', 'success_difference')].n_targets == 2
