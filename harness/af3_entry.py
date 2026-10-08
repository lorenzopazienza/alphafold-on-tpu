"""Runs run_alphafold.py's main() unchanged and records memory at exit.

Started by harness/run_af3.py --entry with third_party/alphafold3 as the
working directory, with the same flags run_alphafold.py would get. It imports
run_alphafold and does exactly what that file's __main__ block does
(mark --output_dir required, absl app.run(main)), so AlphaFold3 itself is not
changed.

At interpreter exit, also after an exception, it writes the JSON file named
by $AF3_ENTRY_STATS with:
  device_memory_stats  jax memory_stats() of the device AlphaFold3 used
                       (GPU and TPU), None on CPU: the device allocator's
                       counters (xla/tsl/framework/allocator.h AllocatorStats):
                       bytes_in_use / peak_bytes_in_use are live buffers,
                       bytes_reserved / peak_bytes_reserved memory the
                       allocator set aside. On TPU the model's temporary
                       buffers do not show in peak_bytes_in_use (v6e probe
                       2026-10-08, 1023 tokens: peak_bytes_in_use 1.5 GB,
                       about the weights and inputs, peak_bytes_reserved
                       6.3 GB), hence compiled_memory below.
  compiled_memory      a list, one entry per compiled (or cache-loaded) model
                       executable jit(apply_fn), with XLA's memory analysis of
                       it: get_compiled_memory_stats(), the same object as
                       jax's Compiled.memory_analysis(). Fields per
                       xla/pjrt/compiled_memory_stats.h (CompiledMemoryStats):
                       argument_size_in_bytes, output_size_in_bytes,
                       alias_size_in_bytes ("How much argument is reused for
                       output"), temp_size_in_bytes, generated_code_size_in_bytes,
                       peak_memory_in_bytes (not documented there), host_*
                       for host memory. device_bytes_needed is the sum that
                       header gives for the on-device memory needed:
                       generated code + arguments + outputs - aliased + temp.
  compiled_memory_error  why compiled_memory is missing or partial, or None
  jax_version          the JAX that ran
  max_rss_bytes        resource.getrusage(RUSAGE_SELF) max RSS of this process
  seconds_in_process   wall time from import to exit
A process killed by a signal (for example the kernel OOM killer) writes
nothing; the harness then records entry_stats as null.

How compiled_memory is obtained: this wrapper replaces
jax._src.compiler.compile_or_get_cached, a PRIVATE JAX function, with a
wrapper that calls it unchanged and reads get_compiled_memory_stats() from
the executable it returns when the module is jit_apply_fn. jax's pxla module
looks the function up at call time, so the wrapper sees the model compile and
persistent-cache hits alike. Checked against jax==0.10.2 (AlphaFold3's
uv.lock pin) only: a JAX upgrade can break it. If installing the wrapper or
reading the statistics fails, AlphaFold3 runs on unchanged and the error is
written to compiled_memory_error.
"""

import atexit
import json
import os
import resource
import sys
import time

_T0 = time.time()
sys.path.insert(0, os.getcwd())

from absl import app  # noqa: E402
from absl import flags  # noqa: E402
import run_alphafold  # noqa: E402

_MODEL_MODULE = 'jit_apply_fn'
_COMPILED = []        # one dict per jit_apply_fn executable
_HOOK = {'error': None}


def _compiled_stats(executable):
  """XLA's memory analysis of one executable, as a dict of ints."""
  stats = executable.get_compiled_memory_stats()
  out = {}
  for name in dir(stats):
    if name.startswith('_') or name.startswith('serialized'):
      continue
    value = getattr(stats, name)
    if isinstance(value, int) and not isinstance(value, bool):
      out[name] = value
  needed = ('generated_code_size_in_bytes', 'argument_size_in_bytes', 'output_size_in_bytes',
            'alias_size_in_bytes', 'temp_size_in_bytes')
  if all(k in out for k in needed):
    out['device_bytes_needed'] = (out['generated_code_size_in_bytes'] + out['argument_size_in_bytes']
                                  + out['output_size_in_bytes'] - out['alias_size_in_bytes']
                                  + out['temp_size_in_bytes'])
  return out


def _install_compile_hook():
  """Wraps jax._src.compiler.compile_or_get_cached (private; jax 0.10.2)."""
  try:
    if 'jax' not in sys.modules:
      raise RuntimeError('run_alphafold did not import jax')
    from jax._src import compiler
    from jax._src.lib.mlir import ir
    original = compiler.compile_or_get_cached

    def compile_or_get_cached(backend, computation, *args, **kwargs):
      executable = original(backend, computation, *args, **kwargs)
      try:
        name = ir.StringAttr(computation.operation.attributes['sym_name']).value
        if name == _MODEL_MODULE:
          _COMPILED.append(dict(_compiled_stats(executable), module=name))
      except Exception as e:  # never let a record break AlphaFold3
        if len(_COMPILED) < 5:
          _COMPILED.append({'error': f'{type(e).__name__}: {e}'[:300]})
      return executable

    compile_or_get_cached.__wrapped__ = original
    compiler.compile_or_get_cached = compile_or_get_cached
  except Exception as e:
    _HOOK['error'] = f'hook not installed: {type(e).__name__}: {e}'[:300]


def _device_memory_stats():
  try:
    backend = str(run_alphafold._JAX_BACKEND.value)
    index = run_alphafold._GPU_DEVICE.value
  except Exception:  # flags were never parsed
    return None, 'flags not parsed'
  if backend == 'cpu':
    return None, None
  try:
    import jax  # already imported by AlphaFold3
    stats = jax.local_devices(backend=backend)[index].memory_stats()
    return ({k: int(v) for k, v in stats.items()} if stats else None), None
  except Exception as e:  # report, never mask AlphaFold3's own exit status
    return None, f'{type(e).__name__}: {e}'[:300]


def _write_stats():
  path = os.environ.get('AF3_ENTRY_STATS')
  if not path:
    return
  maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  memory, error = _device_memory_stats()
  compiled_error = _HOOK['error']
  if compiled_error is None:
    errors = [c['error'] for c in _COMPILED if 'error' in c]
    if errors:
      compiled_error = errors[0]
    elif not _COMPILED:
      compiled_error = f'no {_MODEL_MODULE} executable was compiled or loaded'
  jax_module = sys.modules.get('jax')
  out = {
      'device_memory_stats': memory,
      'device_memory_error': error,
      'compiled_memory': _COMPILED,
      'compiled_memory_error': compiled_error,
      'jax_version': getattr(jax_module, '__version__', None),
      # Linux reports kilobytes, macOS bytes.
      'max_rss_bytes': maxrss if sys.platform == 'darwin' else maxrss * 1024,
      'seconds_in_process': round(time.time() - _T0, 2),
  }
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(out, f, indent=2)
  os.replace(tmp, path)


atexit.register(_write_stats)

if __name__ == '__main__':
  _install_compile_hook()
  flags.mark_flags_as_required(['output_dir'])
  app.run(run_alphafold.main)
