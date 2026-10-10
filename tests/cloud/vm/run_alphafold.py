"""Test double for AlphaFold3's run_alphafold.py (absl flags, fake outputs).

Writes AlphaFold3's output layout: per target <name>_data.json and the full
<name>_confidences.json (the large files FETCH=light leaves in the bucket),
<name>_summary_confidences.json, <name>_ranking_scores.csv, <name>_model.cif,
and per sample folder the same three per-sample files. With FAKE_REF_DIR
(an AlphaFold3 output folder), per-sample mmCIFs are copied from its
seed-S_sample-K folders, so harness/compare_samples.py has real structures.
As found on v5e (2026-10-08/09): on v5e (v6e is fine), num_diffusion_samples > 1
crashes with SIGSEGV when the venv's libtpu (fake_libtpu, written by bin/uv; missing means
0.0.42.1) is 0.0.42.1, whatever LIBTPU_INIT_ARGS says; a venv with jaxlib
0.11.2 fails at import like AlphaFold3's pinned flax does.
FAKE_SCENARIO=bisect: also writes fake libtpu logs to $BISECT_TPU_LOG_DIR and
fake HLO dumps where XLA_FLAGS --xla_dump_to points.
Persistent cache, as jax 0.10.2 keys it: one jit_apply_fn executable per
token bucket, and the key also hashes the cache directory path unless
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none (jax/_src/compiler.py puts
<dir>/xla_gpu_per_fusion_autotune_cache_dir into the compile options). Entries
are <key>-cache files; a hit is logged with its key, like JAX_LOG_COMPILES.
"""
import glob, hashlib, json, os, shutil, signal, sys, time
from absl import app, flags
_JSON_PATH = flags.DEFINE_string('json_path', None, '')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, '')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, '')
_JAX_BACKEND = flags.DEFINE_string('jax_backend', 'gpu', '')
_GPU_DEVICE = flags.DEFINE_integer('gpu_device', 0, '')
flags.DEFINE_string('flash_attention_implementation', 'triton', '')
flags.DEFINE_bool('run_data_pipeline', True, '')
_CACHE = flags.DEFINE_string('jax_compilation_cache_dir', None, '')
_RECYCLES = flags.DEFINE_integer('num_recycles', 10, '')
_SAMPLES = flags.DEFINE_integer('num_diffusion_samples', 5, '')


def venv_file(name, default):
  venv = os.path.dirname(os.path.dirname(os.environ.get('FAKE_VENV', '')))
  try:
    return open(os.path.join(venv, name)).read().strip()
  except OSError:
    return default


def main(_):
  sc = os.environ.get('FAKE_SCENARIO', '')
  if venv_file('fake_jaxlib', '0.10.2') == '0.11.2':
    print('AttributeError: jax.core.Effect  was deprecated in JAX v0.10.0 and removed in JAX v0.11.0. '
          'Use jax.extend.core.Effect.', flush=True)
    sys.exit(1)
  data = json.load(open(_JSON_PATH.value)); name = data['name']; seeds = data['modelSeeds']
  assert os.listdir(_MODEL_DIR.value), 'no weights'
  print(f'Found local {_JAX_BACKEND.value.upper()} devices: [FakeDevice(id=0)], using device 0: fake:0', flush=True)
  print(f'Featurising data with {len(seeds)} seed(s) took 2.00 seconds.', flush=True)
  if sc == 'oom' and name == '7D5C':
    print('jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17.2GiB.', flush=True)
    sys.exit(1)
  if sc == 'bisect':
    logs = os.environ.get('BISECT_TPU_LOG_DIR')
    if logs:
      os.makedirs(logs, exist_ok=True)
      with open(os.path.join(logs, f'tpu_driver.fake.log.INFO.{os.getpid()}'), 'w') as f:
        f.write(f'I fake libtpu log, verbose={os.environ.get("TPU_MIN_LOG_LEVEL")}, '
                f'samples={_SAMPLES.value}, recycles={_RECYCLES.value}, '
                f'LIBTPU_INIT_ARGS={os.environ.get("LIBTPU_INIT_ARGS")}\n' * 50)
    for flag in os.environ.get('XLA_FLAGS', '').split():
      if flag.startswith('--xla_dump_to='):
        d = flag.split('=', 1)[1]; os.makedirs(d, exist_ok=True)
        passes = '--xla_dump_hlo_pass_re' in os.environ.get('XLA_FLAGS', '')
        names = ['module_0001.jit_apply_fn.before_optimizations.txt']
        if passes:
          names += [f'module_0001.jit_apply_fn.{i:04d}.fake-pass-{i}.after_pass.txt' for i in range(1, 6)]
        size = int(os.environ.get('FAKE_DUMP_BYTES', '20000'))
        for n in names:
          with open(os.path.join(d, n), 'wb') as f:
            f.write(os.urandom(size))
          time.sleep(0.01)
    print('Finished jaxpr to MLIR module conversion jit(apply_fn) in 0.8 sec', flush=True)
  if (os.environ.get('FAKE_PLATFORM') == 'v5e' and _SAMPLES.value > 1
      and venv_file('fake_libtpu', '0.0.42.1') == '0.0.42.1'):
    print('Fatal Python error: Segmentation fault', flush=True)
    os.kill(os.getpid(), signal.SIGSEGV)
  time.sleep(float(os.environ.get('FAKE_RUN_S', '0.5')))
  if sc == 'hang':
    time.sleep(600)
  cache = _CACHE.value
  key = path = None
  if cache:
    os.makedirs(cache, exist_ok=True)
    residues = sum(len(s[k]['sequence']) for s in data['sequences'] for k in s if 'sequence' in s[k])
    bucket = next(b for b in (256, 512, 768, 1024, 1280, 1536, 2048, 4096) if residues <= b)
    path_part = '' if os.environ.get('JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES') == 'none' else cache
    key = 'jit_apply_fn-' + hashlib.sha256(f'{bucket}|{path_part}'.encode()).hexdigest()[:32]
    path = os.path.join(cache, f'{key}-cache')
  hit = bool(path and os.path.exists(path))
  if path and not hit:
    open(path, 'w').write('x' * 1000)
  if os.environ.get('JAX_LOG_COMPILES') == '1':
    if hit:
      print(f"Persistent compilation cache hit for 'jit_apply_fn' with key '{key}'", flush=True)
      print('Finished XLA compilation of jit(apply_fn) in 0.4 sec', flush=True)
    else:
      print('Finished XLA compilation of jit(apply_fn) in 61.2 sec', flush=True)
    print('Finished XLA compilation of jit(stage) in 0.01 sec', flush=True)
  for s in seeds:
    print(f'Running model inference with seed {s} took 70.50 seconds.', flush=True)
    print(f'Extracting {_SAMPLES.value} inference samples with seed {s} took 0.20 seconds.', flush=True)
  print(f'Running model inference and extracting output structures with {len(seeds)} seed(s) took 70.70 seconds.', flush=True)
  out = os.path.join(_OUTPUT_DIR.value, name); os.makedirs(out, exist_ok=True)
  big = json.dumps({'atom_plddts': [0.5] * 5000, 'pae': [[1.0] * 50] * 50})
  files = {f'{name}_model.cif': 'fake', f'{name}_summary_confidences.json': '{}',
           f'{name}_ranking_scores.csv': 'seed,sample,ranking_score\n', f'{name}_data.json': big,
           f'{name}_confidences.json': big, 'TERMS_OF_USE.md': 'terms'}
  for f, text in files.items():
    open(os.path.join(out, f), 'w').write(text)
  ref = os.environ.get('FAKE_REF_DIR')
  for s in seeds:
    for k in range(_SAMPLES.value):
      sd = os.path.join(out, f'seed-{s}_sample-{k}'); os.makedirs(sd, exist_ok=True)
      for suffix, text in (('model.cif', 'fake'), ('summary_confidences.json', '{}'), ('confidences.json', big)):
        open(os.path.join(sd, f'{name}_seed-{s}_sample-{k}_{suffix}'), 'w').write(text)
      src = glob.glob(os.path.join(ref, f'seed-{s}_sample-{k}', '*_model.cif')) if ref else []
      if src:
        shutil.copy(src[0], os.path.join(sd, f'{name}_seed-{s}_sample-{k}_model.cif'))
  print('Done running 1 fold jobs.', flush=True)


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
