#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_setup.sh (its last step). Checks that
# AlphaFold3's JAX sees the expected device and writes OUT/setup.json; a
# failure explains itself in OUT/device_check.txt and in the job log.
#
#   PLATFORM=cpu|l4|v5e|v6e bash cloud/vm_device_check.sh OUT AF3_COMMIT
#
# JAX_PLATFORMS per platform: cpu, cuda (L4: never "gpu", which JAX 0.10.2
# expands to cuda, rocm and oneapi and then fails because rocm is not
# installed; jax/_src/xla_bridge.py, expand_platform_alias and backends()),
# tpu. JAX still reports the default backend of a CUDA device as "gpu".
#
# device_check.txt holds: the JAX_PLATFORMS used, the environment variables
# that steer JAX and CUDA, nvidia-smi and /dev/nvidia* (L4), and on failure
# the full Python traceback, JAX's per-backend initialisation errors and its
# own NVIDIA-device visibility check. Exit 5 if JAX does not see the device.
set -uo pipefail
OUT=${1:?output folder} AF3_COMMIT=${2:?AlphaFold3 commit}
: "${PLATFORM:?}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/third_party/alphafold3/.venv/bin/python"
REPORT="$OUT/device_check.txt"

case "$PLATFORM" in
  cpu) BACKEND=cpu JAXP=cpu ;;
  l4) BACKEND=gpu JAXP=cuda ;;
  v5e | v6e) BACKEND=tpu JAXP=tpu ;;
  *) echo "!! device check: unknown PLATFORM $PLATFORM" >&2; exit 5 ;;
esac

{
  echo "## device check for PLATFORM=$PLATFORM: JAX_PLATFORMS=$JAXP, expected backend $BACKEND"
  echo "## environment (JAX, XLA, CUDA, NVIDIA, LD_LIBRARY_PATH, TPU)"
  env | grep -E '^(JAX_|XLA_|CUDA|NVIDIA|LD_LIBRARY_PATH|TPU_|LIBTPU_)' | sort
  if [ "$PLATFORM" = "l4" ]; then
    echo "## nvidia-smi"
    nvidia-smi 2>&1 || echo "(nvidia-smi failed with exit code $?)"
    echo "## /dev/nvidia*"
    ls -l /dev/nvidia* 2>&1
  fi
} > "$REPORT" 2>&1
cat "$REPORT"

JAX_PLATFORMS=$JAXP "$PY" - "$OUT/setup.json" "$PLATFORM" "$BACKEND" "$AF3_COMMIT" "$REPORT" <<'EOF'
import importlib.metadata as md
import json
import sys
import traceback

out, platform, backend, commit, report = sys.argv[1:6]


def note(text):
  print(text, flush=True)
  with open(report, 'a') as f:
    f.write(text + '\n')


packages = {}
for name in ('alphafold3', 'jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
             'tokamax', 'dm-haiku', 'numpy', 'rdkit'):
  try:
    packages[name] = md.version(name)
  except md.PackageNotFoundError:
    packages[name] = None
note('## packages: ' + json.dumps(packages))

import jax  # noqa: E402

try:
  devices = jax.devices()
except Exception:
  note('## jax.devices() failed:')
  note(traceback.format_exc())
  try:
    from jax._src import hardware_utils, xla_bridge
    note('## JAX backend initialisation errors: ' + json.dumps(dict(xla_bridge._backend_errors)))
    note('## JAX sees an NVIDIA device node: ' + str(hardware_utils.has_visible_nvidia_gpu()))
  except Exception as e:  # private API; the traceback above is the main record
    note(f'## (could not read JAX backend details: {type(e).__name__}: {e})')
  sys.exit(f'JAX found no {backend} device; see device_check.txt')

kind = devices[0].device_kind
with open(out.replace('setup.json', 'weights.json')) as f:
  weights = json.load(f)
info = {'platform': platform, 'jax_backend': jax.default_backend(), 'device_kind': kind,
        'devices': [str(d) for d in devices], 'alphafold3_commit': commit,
        'packages': packages, 'weights': weights}
with open(out, 'w') as f:
  json.dump(info, f, indent=2)
note(f'   backend {info["jax_backend"]}, {len(devices)} device(s), kind {kind}')
for name in ('jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'tokamax'):
  note(f'   {name:18s} {packages[name]}')
if jax.default_backend() != backend:
  sys.exit(f'JAX backend is {jax.default_backend()}, expected {backend}')
if platform == 'l4' and 'L4' not in kind:
  sys.exit(f'expected an NVIDIA L4, JAX sees {kind}')
EOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "!! device check failed (exit $rc); details in device_check.txt" | tee -a "$REPORT"
  exit 5
fi
