"""harness/plans/pilot.yaml against notes/analysis_plan.md section 6, and the split plans against pilot.yaml."""

import csv
import pathlib

import pytest

from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
PILOT = sorted(r['pdb_id'] for r in csv.DictReader(open(MANIFEST)) if r['pilot'] == 'yes')
TOKENS = {r['pdb_id']: int(r['num_tokens']) for r in csv.DictReader(open(MANIFEST))}
SPLITS = {'l4': ['pilot_l4_1.yaml', 'pilot_l4_2.yaml'],
          'cpu': ['pilot_cpu_1.yaml', 'pilot_cpu_2.yaml', 'pilot_cpu_3.yaml', 'pilot_cpu_4.yaml']}


def cells(desc):
  """{(target, seed, repetition)} of an expanded plan; repetition r1, r2 or w."""
  out = set()
  for run in desc['runs']:
    rep = 'w' if run['cache'] == 'warm' else f'r{run["rep"]}'
    out |= {(t, run['seed'], rep) for t in desc['targets']}
  return out


def test_pilot_targets():
  assert len(PILOT) == 10
  assert len([t for t in PILOT if TOKENS[t] <= 512]) == 6


@pytest.mark.parametrize('platform', ['v5e', 'v6e', 'l4'])
def test_accelerators_run_the_full_design(platform):
  d = expand(PLANS / 'pilot.yaml', platform, MANIFEST)
  assert sorted(d['targets']) == PILOT
  assert d['weights'] == 'gcs' and d['label'] == 'pilot'
  assert d['num_recycles'] is None and d['num_diffusion_samples'] is None
  assert cells(d) == {(t, s, r) for t in PILOT for s in (1, 2) for r in ('r1', 'r2')} | {(t, 1, 'w') for t in PILOT}
  warm = [r for r in d['runs'] if r['cache'] == 'warm']
  assert [r['source_run'] for r in warm] == [next(r['run'] for r in d['runs'] if r['save_cache'])]


def test_cpu_runs_r1_on_the_small_targets():
  d = expand(PLANS / 'pilot.yaml', 'cpu', MANIFEST)
  small = sorted(t for t in PILOT if TOKENS[t] <= 512)
  assert sorted(d['targets']) == small
  assert cells(d) == {(t, s, 'r1') for t in small for s in (1, 2)}


@pytest.mark.parametrize('platform', ['l4', 'cpu'])
def test_split_plans_cover_the_pilot_exactly(platform):
  whole = cells(expand(PLANS / 'pilot.yaml', platform, MANIFEST))
  parts = [expand(PLANS / name, platform, MANIFEST) for name in SPLITS[platform]]
  union = set()
  for p in parts:
    assert p['weights'] == 'gcs' and p['label'] == 'pilot' and p['deadline_min'] > 0
    assert p['num_recycles'] is None and p['num_diffusion_samples'] is None
    c = cells(p)
    assert not (union & c), 'two split plans run the same (target, seed, repetition)'
    union |= c
    for run in p['runs']:
      if run['cache'] == 'warm':
        assert run['source_run'] in [r['run'] for r in p['runs']], 'a warm rerun needs its source run on the same VM'
  assert union == whole


def test_whole_plan_cannot_launch_unsplit_platforms():
  for platform in SPLITS:
    assert expand(PLANS / 'pilot.yaml', platform, MANIFEST)['deadline_min'] == 0
  for platform in ('v5e', 'v6e'):
    assert expand(PLANS / 'pilot.yaml', platform, MANIFEST)['deadline_min'] > 0
