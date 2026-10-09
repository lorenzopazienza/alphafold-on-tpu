#!/usr/bin/env bash
# Runs ON a TPU VM (v5e, v6e): puts the TPU JAX stack into AlphaFold3's venv.
#
#   PY=.../.venv/bin/python LIBTPU_VERSION=0.0.43.2 bash cloud/vm_tpu_stack.sh          # setup
#   PY=.../.venv/bin/python LIBTPU_VERSION=0.0.42.1 bash cloud/vm_tpu_stack.sh --switch # libtpu only
#
# The stack is jax and jaxlib 0.10.2 (AlphaFold3's uv.lock pin) with libtpu
# LIBTPU_VERSION (default 0.0.43.2). It installs exactly what jax[tpu]==0.10.2
# would (jaxlib==0.10.2, libtpu, requests), except libtpu's version:
# jax[tpu]==0.10.2 pins libtpu==0.0.42.*, and with 0.0.42.1 AlphaFold3 at its
# default 5 diffusion samples segfaults while compiling on TPU v5e; with
# 0.0.43.2 it compiles and runs, on the same jaxlib (bisections of 2026-10-08
# and 2026-10-09, notes/upstream_v5e_segfault.md). That combination is outside
# the jax[tpu] pin and not documented as supported; set LIBTPU_VERSION=0.0.42.1
# for the pinned one.
#
# Setup mode removes AlphaFold3's CUDA JAX plugins first. --switch replaces
# only libtpu (a second stack on the same VM, harness/run_plan.py). Either way
# it then checks the installed libtpu: exit 5 if it is not LIBTPU_VERSION,
# 1 if the install itself fails. Needs uv on PATH.
set -euo pipefail
: "${PY:?python of the AlphaFold3 venv}"
LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.43.2}"
JAX_VERSION=0.10.2
LIBTPU_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html
echo "$LIBTPU_VERSION" | grep -Eq '^[0-9]+(\.[0-9]+)+$' \
  || { echo "!! LIBTPU_VERSION must be a version like 0.0.43.2, not '$LIBTPU_VERSION'" >&2; exit 5; }

if [ "${1:-}" = "--switch" ]; then
  echo "   switching libtpu to $LIBTPU_VERSION (jax/jaxlib $JAX_VERSION unchanged)"
  uv pip install --python "$PY" "libtpu==$LIBTPU_VERSION" -f "$LIBTPU_LINKS"
else
  CUDA_PKGS=$(uv pip freeze --python "$PY" | grep -o '^jax-cuda12-[a-z0-9-]*' || true)
  echo "removing: ${CUDA_PKGS:-none found}"
  [ -z "$CUDA_PKGS" ] || uv pip uninstall --python "$PY" $CUDA_PKGS
  echo "   installing jax==$JAX_VERSION jaxlib==$JAX_VERSION libtpu==$LIBTPU_VERSION requests"
  uv pip install --python "$PY" "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" "libtpu==$LIBTPU_VERSION" requests \
    -f "$LIBTPU_LINKS"
  if uv pip freeze --python "$PY" | grep '^jax-cuda12-'; then
    echo "jax-cuda12 packages are still installed" >&2; exit 1
  fi
fi

GOT=$("$PY" -c "import importlib.metadata as m; print(m.version('libtpu'))" 2> /dev/null || echo none)
JAXLIB=$("$PY" -c "import importlib.metadata as m; print(m.version('jaxlib'))" 2> /dev/null || echo none)
if [ "$GOT" != "$LIBTPU_VERSION" ] || [ "$JAXLIB" != "$JAX_VERSION" ]; then
  echo "!! TPU stack check: libtpu $GOT (expected $LIBTPU_VERSION), jaxlib $JAXLIB (expected $JAX_VERSION)" >&2
  exit 5
fi
echo "   TPU stack: jax/jaxlib $JAX_VERSION, libtpu $GOT"
