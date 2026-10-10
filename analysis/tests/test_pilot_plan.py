"""harness/plans/pilot.yaml against notes/analysis_plan.md section 6, the split plans against pilot.yaml, and
the precision-matched arms (pilot_v6h.yaml, pilot_l4h.yaml and its parts) against section 11."""

import csv
import pathlib

import pytest

from collect import platform_of
from run_af3 import read_configs
from run_plan import expand

REPO = pathlib.Path(__file__).resolve().parents[2]
PLANS = REPO / 'harness' / 'plans'
MANIFEST = REPO / 'inputs' / 'manifest.csv'
PILOT = sorted(r['pdb_id'] for r in csv.DictReader(open(MANIFEST)) if r['pilot'] == 'yes')
TOKENS = {r['pdb_id']: int(r['num_tokens']) for r in csv.DictReader(open(MANIFEST))}
SPLITS = {'l4': ['pilot_l4_1.yaml', 'pilot_l4_2.yaml'],
          'cpu': ['pilot_cpu_1.yaml', 'pilot_cpu_2.yaml', 'pilot_cpu_3.yaml', 'pilot_cpu_4.yaml']}
CONFIGS = read_configs(REPO / 'harness' / 'configs.yaml')
# Section 11 arms: plan, platform, the config it must run, the default-precision config it is matched to,
# the device kind the harness records on that platform, and the condition code analysis/collect.py gives.
MATCHED = {'V6h': ('pilot_v6h.yaml', 'v6e', 'tpu_xla_highest', 'tpu_xla', 'TPU v6 lite'),
           'Gh': ('pilot_l4h.yaml', 'l4', 'l4_xla_highest', 'l4_xla', 'NVIDIA L4')}
MATCHED_SPLITS = {'Gh': ['pilot_l4h_1.yaml', 'pilot_l4h_2.yaml']}


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
  assert d['weights'] == 'gcs' and d['label'] == 'pilot2'
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
    assert p['weights'] == 'gcs' and p['label'] == 'pilot2' and p['deadline_min'] > 0
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


@pytest.mark.parametrize('code', sorted(MATCHED))
def test_matched_config_is_the_default_plus_highest(code):
  _, _, config, default, kind = MATCHED[code]
  cfg, base = CONFIGS[config], CONFIGS[default]
  assert {k: v for k, v in cfg.items() if k not in ('description', 'env')} == \
      {k: v for k, v in base.items() if k not in ('description', 'env')}
  assert cfg['env'] == dict(base.get('env', {}), JAX_DEFAULT_MATMUL_PRECISION='highest')
  assert platform_of(kind, cfg['flash_attention_implementation'], cfg['env']['JAX_DEFAULT_MATMUL_PRECISION']) == code


@pytest.mark.parametrize('code', sorted(MATCHED))
def test_matched_arm_follows_section_11(code):
  plan, platform, config, _, _ = MATCHED[code]
  d = expand(PLANS / plan, platform, MANIFEST)
  assert sorted(d['targets']) == PILOT
  assert d['weights'] == 'gcs' and d['label'] == 'pilot2'
  assert d['num_recycles'] is None and d['num_diffusion_samples'] is None
  assert d['configs'] == [config]
  assert d['warm_rerun_seed'] is None and not any(r['save_cache'] for r in d['runs'])
  assert cells(d) == {(t, s, 'r1') for t in PILOT for s in (1, 2)}
  for other in ('cpu', 'l4', 'v5e', 'v6e'):
    if other != platform:
      with pytest.raises(SystemExit):
        expand(PLANS / plan, other, MANIFEST)


def test_v6h_runs_on_one_vm():
  assert expand(PLANS / MATCHED['V6h'][0], 'v6e', MANIFEST)['deadline_min'] > 0


@pytest.mark.parametrize('code', sorted(MATCHED_SPLITS))
def test_matched_split_plans_cover_the_arm_exactly(code):
  plan, platform, config, _, _ = MATCHED[code]
  whole = expand(PLANS / plan, platform, MANIFEST)
  assert whole['deadline_min'] == 0, 'the unsplit plan must not be launchable'
  union = set()
  for name in MATCHED_SPLITS[code]:
    p = expand(PLANS / name, platform, MANIFEST)
    assert (p['weights'], p['label'], p['configs']) == ('gcs', 'pilot2', [config])
    assert p['num_recycles'] is None and p['num_diffusion_samples'] is None
    assert 0 < p['deadline_min'] and p['deadline_min'] + 30 <= 270, 'each VM within about 4.5 h'
    c = cells(p)
    assert not (union & c), 'two split plans run the same (target, seed, repetition)'
    union |= c
  assert union == cells(whole)
