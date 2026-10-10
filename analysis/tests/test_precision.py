"""Section 11 of notes/analysis_plan.md on synthetic data: the V6h and Gh conditions (analysis/collect.py)
and the PR and HWh contrasts (analysis/contrasts.py), computed as the section 5 contrasts."""

import argparse
import json

import pandas as pd
import pytest

import collect
import contrasts
import cost

TARGETS = ('T1', 'T2', 'T3')


def table(conditions, flips=()):
  """TARGETS x seeds 1, 2 x 5 samples for each (condition, repetitions) of conditions; no model files.

  The top-1 sample is 0, except for (condition, repetition, target) in flips, where it is sample 1.
  Sample 0 is the only success.
  """
  rows = []
  for cond, reps in conditions:
    for rep in reps:
      for t in TARGETS:
        for seed in (1, 2):
          for k in range(5):
            score = 0.95 if (cond, rep, t) in flips and k == 1 else 0.9 - 0.1 * k
            rows.append({'session': f'{cond}_{rep}', 'label': 'pilot', 'platform': cond, 'libtpu': None,
                         'target': t, 'seed': seed, 'repetition': rep, 'sample': k, 'num_recycles': 10,
                         'num_diffusion_samples': 5, 'exit_code': 0, 'failure': None, 'ranking_score': score,
                         'mean_plddt': 80.0, 'iptm': 0.8, 'm1_success': k == 0, 'model_path': None})
  return pd.DataFrame(rows)


PILOT = [('V6', ('r1', 'r2', 'w')), ('G', ('r1', 'r2', 'w')), ('C', ('r1',))]
MATCHED = [('V6h', ('r1',)), ('Gh', ('r1',))]


def run(df, tmp_path, condition='platform', replicates=300):
  path = tmp_path / 'samples.csv'
  df.to_csv(path, index=False)
  args = argparse.Namespace(samples=str(path), labels=None, include=None, exclude=None, condition=condition,
                            replicates=replicates, seed=2026)
  d, ok = contrasts.load(args)
  sp, sm, cons, excluded = contrasts.build_pairs(ok)
  return args, d, ok, sp, sm, cons, excluded


def test_condition_codes():
  assert collect.platform_of('TPU v6 lite', 'xla') == 'V6'
  assert collect.platform_of('TPU v6 lite', 'xla', 'highest') == 'V6h'
  assert collect.platform_of('NVIDIA L4', 'xla', 'highest') == 'Gh'
  assert collect.platform_of('NVIDIA L4', 'xla', None) == 'G'
  assert collect.platform_of('cpu', 'xla', '') == 'C'
  # Any other precision is kept apart from the default arm, with no "_" in the code.
  assert collect.platform_of('NVIDIA L4', 'xla', 'BF16_BF16_F32') == 'G[BF16-BF16-F32]'


def test_collect_reads_the_precision_from_session_json(tmp_path):
  for name, xla_env in (('default', {'JAX_PLATFORMS': 'cuda'}),
                        ('matched', {'JAX_PLATFORMS': 'cuda', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest'})):
    run_dir = tmp_path / 'ses' / f'l4_xla_rep1_seed1_fresh_{name}'
    run_dir.mkdir(parents=True)
    (run_dir / 'session.json').write_text(json.dumps({
        'label': 'pilot', 'config': {'flash_attention_implementation': 'xla'},
        'probe': {'device_kind': 'NVIDIA L4', 'packages': {'jax': '0.10.2'}}, 'xla_env': xla_env,
        'compile_cache': {'mode': 'fresh'}}))
    (run_dir / 'runs.jsonl').write_text(json.dumps({'pdb_id': 'T1', 'output_dir': str(run_dir / 'none'),
                                                    'seeds': [1], 'exit_code': 1, 'failure': 'error'}) + '\n')
    rows, _ = collect.rows_of_run(run_dir, {})
    assert len(rows) == 1 and rows[0]['sample'] is None
    want = ('Gh', 'highest') if name == 'matched' else ('G', None)
    assert (rows[0]['platform'], rows[0]['matmul_precision']) == want
  assert 'matmul_precision' in collect.COLUMNS


def test_precision_conditions_have_prices():
  prices = cost.load_prices()
  for arm, base in (('V6h', 'V6'), ('Gh', 'G')):
    assert cost.price_for(prices, arm, 'europe-west4-a', 'standard') == \
        cost.price_for(prices, base, 'europe-west4-a', 'standard')


def test_section_11_contrasts_and_only_those(tmp_path):
  _, _, _, _, _, cons, _ = run(table(PILOT + MATCHED), tmp_path)
  section5 = {'RC(V6)', 'RC(G)', 'W(V6)', 'W(G)', 'SD(V6)', 'SD(G)', 'SD(C)', 'HW(V6,G)', 'HW(V6,C)', 'HW(G,C)'}
  section11 = {'PR(V6)': 'PR', 'PR(G)': 'PR', 'HWh(V6,G)': 'HWh', 'HWh(G,C)': 'HWh'}
  assert set(cons) == section5 | set(section11)
  for name, kind in section11.items():
    assert cons[name] == (kind, False), 'section 11 contrasts are secondary'


def test_matched_arms_leave_section_5_unchanged(tmp_path):
  flips = {('G', 'r1', 'T1'), ('V6', 'r2', 'T2'), ('V6h', 'r1', 'T3'), ('Gh', 'r1', 'T1'), ('Gh', 'r1', 'T2')}
  args, d0, ok0, sp0, sm0, cons0, ex0 = run(table(PILOT, flips), tmp_path)
  args, d1, ok1, sp1, sm1, cons1, ex1 = run(table(PILOT + MATCHED, flips), tmp_path)
  s0 = contrasts.summarize(sp0, sm0, cons0, args)
  s1 = contrasts.summarize(sp1, sm1, cons1, args)
  pd.testing.assert_frame_equal(s0, s1[s1.contrast.isin(cons0)].reset_index(drop=True))
  h0, h1 = contrasts.hw_minus_rc(sp0, cons0, args), contrasts.hw_minus_rc(sp1, cons1, args)
  pd.testing.assert_frame_equal(h0, h1[h1.hardware.isin(cons0)].reset_index(drop=True))
  q0, q1 = contrasts.q3(ok0, args), contrasts.q3(ok1, args)
  pd.testing.assert_frame_equal(q0[0], q1[0])
  assert q0[1] == q1[1]
  c1 = contrasts.checks(d1, ok1, sm1, args)
  assert set(c1[c1.check.str.startswith('3 ')].condition) == {'V6'}
  assert {'V6h', 'Gh'} <= set(c1[c1.check == '1 exit codes'].condition)


def test_pr_is_computed_as_a_section_5_contrast(tmp_path):
  # V6h carries exactly G's data, so PR(V6) (V6 vs V6h) must equal HW(V6,G) (V6 vs G) in every
  # statistic and interval: same pairing, same statistics, same bootstrap with seed 2026.
  flips = {('G', 'r1', 'T1'), ('G', 'r1', 'T3'), ('V6h', 'r1', 'T1'), ('V6h', 'r1', 'T3')}
  args, _, _, sp, sm, cons, _ = run(table(PILOT + MATCHED, flips), tmp_path)
  s = contrasts.summarize(sp, sm, cons, args)
  cols = ['statistic', 'n', 'n_targets', 'estimate', 'ci_low', 'ci_high']
  pr = s[s.contrast == 'PR(V6)'][cols].reset_index(drop=True)
  hw = s[s.contrast == 'HW(V6,G)'][cols].reset_index(drop=True)
  pd.testing.assert_frame_equal(pr, hw)
  assert pr.set_index('statistic').loc['flip_top1'].estimate == pytest.approx(4 / 6)


def test_flip_rates_of_the_matched_contrasts(tmp_path):
  # V6h: top-1 moves on T1. Gh: on T1 and T2. C: never.
  flips = {('V6h', 'r1', 'T1'), ('Gh', 'r1', 'T1'), ('Gh', 'r1', 'T2')}
  args, _, _, sp, sm, cons, _ = run(table(PILOT + MATCHED, flips), tmp_path)
  s = contrasts.summarize(sp, sm, cons, args).set_index(['contrast', 'statistic'])
  assert s.loc[('PR(V6)', 'flip_top1')].estimate == pytest.approx(2 / 6)
  assert s.loc[('PR(G)', 'flip_top1')].estimate == pytest.approx(4 / 6)
  assert s.loc[('HWh(V6,G)', 'flip_top1')].estimate == pytest.approx(2 / 6)   # T2 only
  assert s.loc[('HWh(G,C)', 'flip_top1')].estimate == pytest.approx(4 / 6)
  assert s.loc[('HWh(G,C)', 'flip_matters')].estimate == pytest.approx(4 / 6)
  r = s.loc[('HWh(G,C)', 'flip_top1')]
  assert r.ci_low <= r.estimate <= r.ci_high


def test_hwh_minus_rc_uses_the_default_arms(tmp_path):
  flips = {('V6', 'r2', 'T1'), ('V6h', 'r1', 'T2'), ('Gh', 'r1', 'T1'), ('Gh', 'r1', 'T2')}
  args, _, _, sp, sm, cons, _ = run(table(PILOT + MATCHED, flips), tmp_path)
  h = contrasts.hw_minus_rc(sp, cons, args)
  r = h[(h.hardware == 'HWh(V6,G)') & (h.reference == 'RC(P)') & (h.statistic == 'flip_top1')].iloc[0]
  assert r.recompilation == 'RC(V6)' and r.primary == False  # noqa: E712
  assert r.estimate == pytest.approx(2 / 6 - 2 / 6)           # HWh flips on T1; RC(V6) on T1
  q = h[(h.hardware == 'HWh(G,C)')]
  assert set(q.recompilation) == {'RC(G)'}                    # no RC(C): C runs r1 only


def test_missing_matched_pairs_are_counted(tmp_path):
  df = table(PILOT + MATCHED)
  df = df[~((df.platform == 'Gh') & (df.target == 'T3'))]
  _, _, _, _, _, _, excluded = run(df, tmp_path)
  e = excluded.set_index('contrast')
  assert (e.loc['PR(G)'].pairs_used, e.loc['PR(G)'].excluded_only_a) == (4, 2)
  assert (e.loc['HWh(V6,G)'].pairs_used, e.loc['HWh(V6,G)'].excluded_only_a) == (4, 2)


def test_platform_libtpu_pairs_pr_within_one_stack(tmp_path):
  df = table([('V6', ('r1', 'r2')), ('V6h', ('r1',)), ('G', ('r1',)), ('Gh', ('r1',))])
  df.loc[df.platform.isin(['V6', 'V6h']), 'libtpu'] = '0.0.43.2'
  old = table([('V6', ('r1',))])
  old['libtpu'], old['session'] = '0.0.42.1', 'V6_old'
  _, _, _, _, _, cons, _ = run(pd.concat([df, old]), tmp_path, condition='platform_libtpu')
  assert {c for c in cons if c.startswith(('PR', 'HWh'))} == \
      {'PR(V6_0.0.43.2)', 'PR(G_none)', 'HWh(V6_0.0.43.2,G_none)'}
