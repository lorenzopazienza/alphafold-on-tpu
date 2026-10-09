#!/bin/bash
# Test double for cloud/vm_af3_setup.sh: real input check, weights script and
# device check (cloud/vm_device_check.sh); fake install.
set -euo pipefail
OUT=$1; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo ">> setup: inputs against inputs/manifest.csv"
python3 "$ROOT/harness/verify_inputs.py" --targets "$TARGETS" || exit 3
echo ">> setup: fake install of AlphaFold3 (test double)"
sleep "${FAKE_SETUP_S:-0.5}"
mkdir -p "$ROOT/third_party/alphafold3/.venv/bin"
cp "$FAKE_KIT/vm/run_alphafold.py" "$ROOT/third_party/alphafold3/run_alphafold.py"
cp "$FAKE_KIT/vm/python" "$ROOT/third_party/alphafold3/.venv/bin/python"; chmod +x "$ROOT/third_party/alphafold3/.venv/bin/python"
echo ">> setup: weights ($WEIGHTS)"
WEIGHTS="$WEIGHTS" bash "$ROOT/cloud/vm_af3_weights.sh" "${AF3_WEIGHTS_DIR:-$HOME/af3_weights_run}" "$OUT" || exit 4
echo ">> setup: devices and versions (real cloud/vm_device_check.sh)"
PLATFORM="$PLATFORM" bash "$ROOT/cloud/vm_device_check.sh" "$OUT" fakecommit || exit 5
echo ">> setup: done"
