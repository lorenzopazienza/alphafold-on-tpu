"""The persistent-cache fix of 2026-10-11 (harness/run_af3.py) and the check 2 preconditions (contrasts.cache_checks).

The pilot of 2026-10-09/10 showed that no warm rerun loaded its source run's executable: jax 0.10.2 puts
<cache dir>/xla_gpu_per_fusion_autotune_cache_dir into the compile options, which the cache key hashes.
"""

import os
import subprocess

import pandas as pd
import pytest

import contrasts
import run_af3

AF3_PY = run_af3.REPO / 'third_party' / 'alphafold3' / '.venv' / 'bin' / 'python'
KEY_SCRIPT = r'''
import sys, numpy as np, jax, jax.numpy as jnp
from jax._src import compiler, compilation_cache
jax.config.update('jax_compilation_cache_dir', sys.argv[1])
mod = jax.jit(lambda x: jnp.tanh(x @ x.T).sum()).lower(jnp.ones((8, 8), jnp.float32)).compiler_ir('stablehlo')
backend = jax.devices()[0].client
opts = compiler.get_compile_options(num_replicas=1, num_partitions=1, backend=backend)
print(compilation_cache.get_cache_key(mod, np.array(jax.devices()[:1]), opts, backend))
'''


def key_in(tmp_path, folder, extra_env):
  env = {k: v for k, v in os.environ.items() if k != 'JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES'}
  env.update(JAX_PLATFORMS='cpu', **extra_env)
  out = subprocess.run([str(AF3_PY), '-c', KEY_SCRIPT, str(tmp_path / folder)], env=env, capture_output=True,
                       text=True, timeout=300)
  assert out.returncode == 0, out.stderr[-2000:]
  return out.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not AF3_PY.exists(), reason='AlphaFold3 venv (jax 0.10.2) not installed')
def test_jax_key_depends_on_the_cache_dir_unless_xla_caches_are_off(tmp_path):
  # The premise of the fix, on the installed jax: two processes, two cache folders.
  assert key_in(tmp_path, 'a', {}) != key_in(tmp_path, 'b', {})
  assert key_in(tmp_path, 'a', run_af3.CACHE_ENV) == key_in(tmp_path, 'b', run_af3.CACHE_ENV)


def test_cache_env_turns_off_the_path_dependent_xla_caches():
  assert run_af3.CACHE_ENV == {'JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES': 'none'}


def test_parse_compiles_records_the_loaded_key():
  key = 'jit_apply_fn-' + 'ab' * 32
  log = (f"W pxla.py] Persistent compilation cache hit for 'jit_stage' with key 'jit_stage-00'\n"
         f"W compiler.py] Persistent compilation cache hit for 'jit_apply_fn' with key '{key}'\n"
         'W dispatch.py] Finished XLA compilation of jit(apply_fn) in 0.4 sec\n')
  c = run_af3.parse_compiles(log)
  assert c['model_cache_hit'] is True and c['model_cache_key'] == key and c['cache_hits'] == 2
  miss = run_af3.parse_compiles('W dispatch.py] Finished XLA compilation of jit(apply_fn) in 61.2 sec\n')
  assert miss['model_cache_hit'] is False and miss['model_cache_key'] is None
  # Logs written before keys were parsed (no "with key") still give the hit.
  old = run_af3.parse_compiles("Persistent compilation cache hit for 'jit_apply_fn'\n"
                               'Finished XLA compilation of jit(apply_fn) in 0.4 sec\n')
  assert old['model_cache_hit'] is True and old['model_cache_key'] is None


def test_model_entries_lists_jit_apply_fn_executables_only(tmp_path):
  for name in ('jit_apply_fn-0a1b-cache', 'jit_apply_fn-0a1b-atime', 'jit_stage-99-cache', 'jit_apply_fn-ff-cache'):
    (tmp_path / name).write_text('x')
  (tmp_path / 'xla_gpu_per_fusion_autotune_cache_dir').mkdir()
  assert run_af3.model_entries(tmp_path) == ['jit_apply_fn-0a1b', 'jit_apply_fn-ff']
  assert run_af3.model_entries(tmp_path / 'missing') == [] and run_af3.model_entries(None) == []


def evidence(warm_key='K1', fresh_before=0, fresh_hit=False):
  """V6, 2 targets: r1 seed 1 (wrote K1/K2), r2 seed 1, and the warm rerun of seed 1."""
  rows = []
  for t, written in (('T1', 'K1'), ('T2', 'K2')):
    rows.append({'session': 's/r1', 'condition': 'V6', 'target': t, 'seed': 1, 'repetition': 'r1',
                 'cache_files_before': fresh_before, 'model_cache_hit': fresh_hit, 'model_cache_key': None,
                 'model_cache_written': written})
    rows.append({'session': 's/r2', 'condition': 'V6', 'target': t, 'seed': 1, 'repetition': 'r2',
                 'cache_files_before': 0, 'model_cache_hit': False, 'model_cache_key': None,
                 'model_cache_written': 'K9'})
    rows.append({'session': 's/w', 'condition': 'V6', 'target': t, 'seed': 1, 'repetition': 'w',
                 'cache_files_before': 2, 'model_cache_hit': True,
                 'model_cache_key': warm_key if t == 'T1' else 'K2', 'model_cache_written': None})
  return pd.DataFrame(rows)


def by_check(rows):
  return {r['check'][:2]: r for r in rows}


def test_cache_checks_pass_when_each_warm_run_loads_its_source():
  r = by_check(contrasts.cache_checks(evidence()))
  assert r['2a']['value'] == '4/4 processes' and r['2a']['detail'] == 'pass'
  assert r['2b']['value'] == '2/2 processes' and r['2b']['detail'] == 'pass'


def test_cache_checks_fail_when_a_warm_run_loads_another_executable():
  # T1's warm run loaded T2's entry (same bucket, same key before the per-target folders): not its source.
  r = by_check(contrasts.cache_checks(evidence(warm_key='K2')))
  assert r['2b']['value'] == '1/2 processes' and r['2b']['detail'] == 'FAIL: T1'


def test_cache_checks_fail_when_a_fresh_run_did_not_start_empty():
  r = by_check(contrasts.cache_checks(evidence(fresh_before=3)))
  assert r['2a']['value'] == '2/4 processes' and r['2a']['detail'].startswith('FAIL')
  r = by_check(contrasts.cache_checks(evidence(fresh_hit=True)))
  assert r['2a']['detail'].startswith('FAIL')


def test_cache_checks_on_records_without_keys_fail_rather_than_pass():
  # The pilot of 2026-10-09/10 recorded no keys: check 2b cannot pass on it.
  d = evidence()
  d['model_cache_key'] = None
  d['model_cache_written'] = None
  assert by_check(contrasts.cache_checks(d))['2b']['detail'] == 'FAIL: T1, T2'


def test_diag_plan_loads_one_executable_twice_per_config():
  from run_plan import expand
  plans = run_af3.REPO / 'harness' / 'plans'
  d = expand(plans / 'diag_l4_determinism.yaml', 'l4', run_af3.REPO / 'inputs' / 'manifest.csv')
  assert d['label'] == 'diag' and d['weights'] == 'gcs' and d['deadline_min'] > 0
  assert d['configs'] == ['l4_xla', 'l4_xla_det', 'l4_xla_noautotune'] and d['processes'] == 24
  for c in d['configs']:
    runs = [r for r in d['runs'] if r['config'] == c]
    assert [r['run'] for r in runs] == [f'{c}_rep1_seed1_fresh', f'{c}_rep2_seed1_fresh', f'{c}_seed1_warm',
                                        f'{c}_seed1_warm2']
    assert [r['save_cache'] for r in runs] == [True, False, False, False]
    assert {r.get('source_run') for r in runs if r['cache'] == 'warm'} == {f'{c}_rep1_seed1_fresh'}
  # Plans without the key keep one warm rerun.
  pilot = expand(plans / 'pilot.yaml', 'v6e', run_af3.REPO / 'inputs' / 'manifest.csv')
  assert [r['run'] for r in pilot['runs'] if r['cache'] == 'warm'] == ['tpu_xla_seed1_warm']
