#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_job.sh. Checks the uploaded inputs against
# inputs/manifest.csv, installs AlphaFold3 at the pinned commit with
# af3_tpu.patch (the same steps as af3_tpu/vm_smoke.sh, copied here, not
# called), sets up JAX for the platform, runs build_data, puts the weights in
# place (cloud/vm_af3_weights.sh) and checks that JAX sees the right device.
#
#   PLATFORM=cpu|l4|v5e|v6e WEIGHTS=random|gcs TARGETS=ID,ID bash cloud/vm_af3_setup.sh OUT
#
# JAX per platform: TPU swaps AlphaFold3's CUDA JAX for jax[tpu]==0.10.2 (as
# the smoke tests do); L4 keeps AlphaFold3's pinned CUDA JAX and must show an
# L4; CPU keeps the default install (the cpu_xla config sets JAX_PLATFORMS=cpu).
# Writes OUT/setup.json, OUT/pip_freeze.txt and OUT/weights.json. Exit 3 if
# an input does not match the manifest, 4 for a weights problem, 5 if JAX does
# not see the expected device.
set -euo pipefail
OUT=${1:?output folder}
: "${PLATFORM:?}" "${WEIGHTS:?}" "${TARGETS:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/af3_tpu/pin.sh"
AF3="$ROOT/third_party/alphafold3"
PY="$AF3/.venv/bin/python"
WEIGHTS_DIR="${AF3_WEIGHTS_DIR:-$HOME/af3_weights_run}"   # outside the checkout; never fetched
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
step() { echo; echo ">> [$(date -u +%H:%M:%S)] setup: $*"; }

step "inputs against inputs/manifest.csv"
python3 "$ROOT/harness/verify_inputs.py" --targets "$TARGETS" || exit 3

step "system packages (the ones AlphaFold3's docker/Dockerfile installs)"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git gcc g++ make zlib1g-dev zstd

step "uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "AlphaFold3 $AF3_TAG ($AF3_COMMIT) with af3_tpu.patch"
[ -d "$AF3/.git" ] || git clone -q "$AF3_REPO" "$AF3"
git -C "$AF3" -c advice.detachedHead=false checkout -q "$AF3_COMMIT"
bash "$ROOT/af3_tpu/apply.sh" "$AF3"

step "install from uv.lock (uv sync --frozen, as in docker/Dockerfile)"
cd "$AF3"
uv sync --frozen --python 3.12

# From here on, never `uv run`: it re-syncs to uv.lock and would undo the TPU
# swap. Call the venv's executables directly.
case "$PLATFORM" in
  v5e | v6e)
    step "swap CUDA JAX for jax[tpu]==0.10.2"
    CUDA_PKGS=$(uv pip freeze --python "$PY" | grep -o '^jax-cuda12-[a-z0-9-]*' || true)
    echo "removing: ${CUDA_PKGS:-none found}"
    [ -z "$CUDA_PKGS" ] || uv pip uninstall --python "$PY" $CUDA_PKGS
    uv pip install --python "$PY" "jax[tpu]==0.10.2" \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    if uv pip freeze --python "$PY" | grep '^jax-cuda12-'; then
      echo "jax-cuda12 packages are still installed" >&2; exit 1
    fi ;;
  l4)
    step "NVIDIA driver (from the image); AlphaFold3's pinned CUDA JAX is kept"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader ;;
  cpu)
    step "CPU: default install kept; cpu_xla runs with JAX_PLATFORMS=cpu" ;;
esac

step "build_data"
"$AF3/.venv/bin/build_data"

step "weights ($WEIGHTS)"
cd "$ROOT"
WEIGHTS="$WEIGHTS" bash "$ROOT/cloud/vm_af3_weights.sh" "$WEIGHTS_DIR" "$OUT" || exit 4

step "devices and versions"
uv pip freeze --python "$PY" > "$OUT/pip_freeze.txt"
case "$PLATFORM" in cpu) BACKEND=cpu ;; l4) BACKEND=gpu ;; *) BACKEND=tpu ;; esac
JAX_PLATFORMS=$BACKEND "$PY" - "$OUT/setup.json" "$PLATFORM" "$BACKEND" "$AF3_COMMIT" <<'EOF' || exit 5
import importlib.metadata as md
import json
import sys

import jax

out, platform, backend, commit = sys.argv[1:5]
packages = {}
for name in ('alphafold3', 'jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'jax-cuda12-pjrt',
             'tokamax', 'dm-haiku', 'numpy', 'rdkit'):
  try:
    packages[name] = md.version(name)
  except md.PackageNotFoundError:
    packages[name] = None
devices = jax.devices()
kind = devices[0].device_kind
with open(out.replace('setup.json', 'weights.json')) as f:
  weights = json.load(f)
info = {'platform': platform, 'jax_backend': jax.default_backend(), 'device_kind': kind,
        'devices': [str(d) for d in devices], 'alphafold3_commit': commit,
        'packages': packages, 'weights': weights}
with open(out, 'w') as f:
  json.dump(info, f, indent=2)
print(f'   backend {info["jax_backend"]}, {len(devices)} device(s), kind {kind}')
for name in ('jax', 'jaxlib', 'libtpu', 'jax-cuda12-plugin', 'tokamax'):
  print(f'   {name:18s} {packages[name]}')
if jax.default_backend() != backend:
  sys.exit(f'JAX backend is {jax.default_backend()}, expected {backend}')
if platform == 'l4' and 'L4' not in kind:
  sys.exit(f'expected an NVIDIA L4, JAX sees {kind}')
EOF
echo ">> [$(date -u +%H:%M:%S)] setup: done"
