"""Runs run_alphafold.py's main() unchanged and records memory at exit.

Started by harness/run_af3.py --entry with third_party/alphafold3 as the
working directory, with the same flags run_alphafold.py would get. It imports
run_alphafold and does exactly what that file's __main__ block does
(mark --output_dir required, absl app.run(main)), so AlphaFold3 itself is not
changed.

At interpreter exit, also after an exception, it writes the JSON file named
by $AF3_ENTRY_STATS with:
  device_memory_stats  jax memory_stats() of the device AlphaFold3 used
                       (GPU and TPU; peak_bytes_in_use is the peak), None on CPU
  max_rss_bytes        resource.getrusage(RUSAGE_SELF) max RSS of this process
  seconds_in_process   wall time from import to exit
A process killed by a signal (for example the kernel OOM killer) writes
nothing; the harness then records entry_stats as null.
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
  out = {
      'device_memory_stats': memory,
      'device_memory_error': error,
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
  flags.mark_flags_as_required(['output_dir'])
  app.run(run_alphafold.main)
