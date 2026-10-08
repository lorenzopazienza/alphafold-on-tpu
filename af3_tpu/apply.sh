#!/usr/bin/env bash
# Apply af3_tpu.patch to the pinned AlphaFold3 checkout.
#
#   bash af3_tpu/apply.sh [AF3_DIR]     (default: third_party/alphafold3)
#
# Fails if the checkout is not at AF3_COMMIT or the patch does not apply
# cleanly. Running it again on an already patched checkout is a no-op.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pin.sh"
AF3_DIR="${1:-$HERE/../third_party/alphafold3}"
PATCH="$HERE/af3_tpu.patch"

die() { echo "apply.sh: ERROR: $*" >&2; exit 1; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

[ -d "$AF3_DIR/.git" ] || die "$AF3_DIR is not a git checkout. Run:
  git clone $AF3_REPO $AF3_DIR && git -C $AF3_DIR checkout $AF3_TAG"

head="$(git -C "$AF3_DIR" rev-parse HEAD)"
[ "$head" = "$AF3_COMMIT" ] || die "$AF3_DIR is at $head, expected $AF3_COMMIT ($AF3_TAG)."

if git -C "$AF3_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "apply.sh: patch already applied in $AF3_DIR"
  exit 0
fi

git -C "$AF3_DIR" apply --check "$PATCH" \
  || die "af3_tpu.patch does not apply cleanly to $AF3_DIR (git output above)."
git -C "$AF3_DIR" apply "$PATCH"
echo "apply.sh: applied af3_tpu.patch (sha256 $(sha256 "$PATCH")) to $AF3_DIR at $AF3_COMMIT"
