#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_job.sh. Checks the uploaded inputs against
# inputs/manifest.csv, installs AlphaFold3 at the pinned commit with
# af3_tpu.patch (the same steps as af3_tpu/vm_smoke.sh, copied here, not
# called), sets up JAX for the platform, runs build_data, puts the weights in
# place (cloud/vm_af3_weights.sh) and checks that JAX sees the right device.
#
#   PLATFORM=cpu|l4|v5e|v6e WEIGHTS=random|gcs TARGETS=ID,ID bash cloud/vm_af3_setup.sh OUT
#
# JAX per platform: TPU swaps AlphaFold3's CUDA JAX for jax/jaxlib 0.10.2 with
# libtpu LIBTPU_VERSION (default 0.0.43.2; cloud/vm_tpu_stack.sh explains why
# not the 0.0.42.* that jax[tpu]==0.10.2 pins); L4 keeps AlphaFold3's pinned CUDA JAX and must show an
# L4; CPU keeps the default install (the cpu_xla config sets JAX_PLATFORMS=cpu).
# The device check is cloud/vm_device_check.sh (JAX_PLATFORMS=cuda on L4; it
# writes OUT/device_check.txt, which explains any failure).
# Writes OUT/setup.json (on TPU with a tpu_runtime block: libtpu version,
# build label and build date), OUT/pip_freeze.txt and OUT/weights.json. Exit 3
# if an input does not match the manifest, 4 for a weights problem, 5 if JAX
# does not see the expected device or the installed libtpu is not LIBTPU_VERSION.
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
    step "swap CUDA JAX for jax/jaxlib 0.10.2 with libtpu ${LIBTPU_VERSION:-0.0.43.2}"
    PY="$PY" LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.43.2}" bash "$ROOT/cloud/vm_tpu_stack.sh" ;;
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
# JAX_PLATFORMS is cpu, cuda or tpu there, never "gpu" (see that script).
PLATFORM="$PLATFORM" bash "$ROOT/cloud/vm_device_check.sh" "$OUT" "$AF3_COMMIT" || exit 5
echo ">> [$(date -u +%H:%M:%S)] setup: done"
