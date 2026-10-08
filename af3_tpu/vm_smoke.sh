#!/usr/bin/env bash
# Runs ON the TPU VM, started by cloud/af3_tpu_smoke.sh. Installs AlphaFold3 at
# the pinned commit with af3_tpu.patch, swaps the CUDA JAX for jax[tpu],
# generates random weights on the VM, runs the toy input once on the TPU and
# writes session.json. Smoke test only: its timing is not a measurement.
#
#   bash af3_tpu/vm_smoke.sh OUT_DIR
#
# Reads ACCEL RUNTIME ZONE PROJECT TPU_NAME SPOT BENCH_COMMIT BENCH_DIRTY from
# the environment. Exit code 0 only if inference exits 0 and wrote an mmCIF
# and a summary_confidences.json.
set -euo pipefail
OUT=${1:?output dir}
: "${ACCEL:?}" "${RUNTIME:?}" "${ZONE:?}" "${PROJECT:?}" "${TPU_NAME:?}" "${SPOT:?}"
: "${BENCH_COMMIT:?}" "${BENCH_DIRTY:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/af3_tpu/pin.sh"
AF3="$ROOT/third_party/alphafold3"
PY="$AF3/.venv/bin/python"
WEIGHTS_DIR="$HOME/af3_weights"   # outside the repo copy; never copied back
AF3_OUT="$HOME/af3_out"
INPUT="$ROOT/af3_tpu/inputs/toy_118.json"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/vm_smoke.log") 2>&1
export PYTHONUNBUFFERED=1
step() { echo; echo ">> [$(date -u +%H:%M:%S)] $*"; }

step "System packages (the ones AlphaFold3's docker/Dockerfile installs)"
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

step "Install from uv.lock (uv sync --frozen, as in docker/Dockerfile)"
cd "$AF3"
uv sync --frozen --python 3.12

# From here on, never `uv run`: it re-syncs to uv.lock and would put the CUDA
# JAX back. Call the venv's executables directly.
step "Swap CUDA JAX for jax[tpu]==0.10.2"
CUDA_PKGS=$(uv pip freeze --python "$PY" | grep -o '^jax-cuda12-[a-z0-9-]*' || true)
echo "removing: ${CUDA_PKGS:-none found}"
[ -z "$CUDA_PKGS" ] || uv pip uninstall --python "$PY" $CUDA_PKGS
uv pip install --python "$PY" "jax[tpu]==0.10.2" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
if uv pip freeze --python "$PY" | grep '^jax-cuda12-'; then
  echo "jax-cuda12 packages are still installed" >&2; exit 1
fi

step "build_data"
"$AF3/.venv/bin/build_data"

step "Random weights (generated on the VM, never copied off it)"
"$PY" "$ROOT/af3_tpu/make_random_params.py" \
  --output "$WEIGHTS_DIR/random_weights.bin.zst" | tee "$OUT/random_params.txt"

step "Devices and versions"
uv pip freeze --python "$PY" > "$OUT/pip_freeze.txt"
"$PY" - "$OUT/versions.json" <<'EOF'
import importlib.metadata as md
import json
import sys

import jax

packages = {}
for name in ('jax', 'jaxlib', 'libtpu', 'tokamax', 'dm-haiku', 'numpy', 'alphafold3'):
  try:
    packages[name] = md.version(name)
  except md.PackageNotFoundError:
    packages[name] = None
devices = jax.devices()
print('jax.devices():', devices)
for name, version in packages.items():
  print(f'{name:12s} {version}')
with open(sys.argv[1], 'w') as f:
  json.dump({
      'packages': packages,
      'default_backend': jax.default_backend(),
      'devices': [str(d) for d in devices],
      'device_kind': devices[0].device_kind,
  }, f, indent=2)
if jax.default_backend() != 'tpu':
  sys.exit(f'JAX default backend is {jax.default_backend()!r}, not tpu')
EOF

step "Inference on the toy input (smoke test, not a measurement)"
CMD=("$PY" run_alphafold.py
  --json_path="$INPUT"
  --output_dir="$AF3_OUT"
  --model_dir="$WEIGHTS_DIR"
  --jax_backend=tpu
  --flash_attention_implementation=xla
  --run_data_pipeline=false
  --num_recycles=1
  --num_diffusion_samples=1)
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; echo >> "$OUT/command.txt"
cat "$OUT/command.txt"
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s.%N)
set +e
"${CMD[@]}" 2>&1 | tee "$OUT/run_alphafold.log"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s.%N); END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "run_alphafold.py exit code: $RC"

step "Collect mmCIF and summary_confidences.json"
mkdir -p "$OUT/af3_output"
if [ -d "$AF3_OUT" ]; then
  (cd "$AF3_OUT" && find . -type f \( -name '*.cif' -o -name '*summary_confidences.json' \) \
    -exec cp --parents {} "$OUT/af3_output/" \;)
fi
find "$OUT/af3_output" -type f | sort

step "session.json"
export RC START_UTC END_UTC T0 T1 OUT INPUT AF3 WEIGHTS_DIR ROOT
"$PY" - <<'EOF'
import glob
import hashlib
import json
import os
import platform
import subprocess

env = os.environ
out = env['OUT']

def sha256(path):
  with open(path, 'rb') as f:
    return hashlib.sha256(f.read()).hexdigest()

def cpu_model():
  for line in open('/proc/cpuinfo'):
    if line.startswith('model name'):
      return line.split(':', 1)[1].strip()
  return None

outputs = sorted(os.path.relpath(p, out) for p in
                 glob.glob(os.path.join(out, 'af3_output', '**', '*'), recursive=True)
                 if os.path.isfile(p))
has_cif = any(p.endswith('.cif') for p in outputs)
has_summary = any(p.endswith('summary_confidences.json') for p in outputs)
weights = glob.glob(os.path.join(env['WEIGHTS_DIR'], '*.bin.zst'))
with open(os.path.join(out, 'versions.json')) as f:
  versions = json.load(f)

session = {
    'kind': 'af3_tpu_smoke',
    'note': 'Smoke test with random weights. Not a measurement.',
    'session': os.path.basename(out.rstrip('/')),
    'start_utc': env['START_UTC'],
    'end_utc': env['END_UTC'],
    'inference_wall_seconds': round(float(env['T1']) - float(env['T0']), 1),
    'exit_code': int(env['RC']),
    'passed': int(env['RC']) == 0 and has_cif and has_summary,
    'command': open(os.path.join(out, 'command.txt')).read().strip(),
    'accelerator': env['ACCEL'],
    'runtime': env['RUNTIME'],
    'zone': env['ZONE'],
    'project': env['PROJECT'],
    'tpu_name': env['TPU_NAME'],
    'spot': env['SPOT'] == '1',
    'host_cpu': cpu_model(),
    'host_kernel': platform.release(),
    'python': platform.python_version(),
    'alphafold3': {
        'repo': env['AF3_REPO'],
        'tag': env['AF3_TAG'],
        'commit': subprocess.check_output(
            ['git', '-C', env['AF3'], 'rev-parse', 'HEAD'], text=True).strip(),
        'patch_sha256': sha256(os.path.join(env['ROOT'], 'af3_tpu/af3_tpu.patch')),
        'patched_run_alphafold_sha256': sha256(os.path.join(env['AF3'], 'run_alphafold.py')),
    },
    'bench_repo': {'commit': env['BENCH_COMMIT'], 'dirty': env['BENCH_DIRTY'] == 'true'},
    'input': {'path': 'af3_tpu/inputs/toy_118.json', 'sha256': sha256(env['INPUT'])},
    'weights': {
        'kind': 'random uniform(-1, 1), make_random_params.py --seed 0',
        'sha256': sha256(weights[0]) if len(weights) == 1 else None,
    },
    'packages': versions['packages'],
    'jax_default_backend': versions['default_backend'],
    'jax_devices': versions['devices'],
    'device_kind': versions['device_kind'],
    'outputs': outputs,
}
with open(os.path.join(out, 'session.json'), 'w') as f:
  json.dump(session, f, indent=2)
print(json.dumps({k: session[k] for k in
                  ('passed', 'exit_code', 'inference_wall_seconds', 'device_kind')}))
raise SystemExit(0 if session['passed'] else 1)
EOF
