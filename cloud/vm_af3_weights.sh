#!/usr/bin/env bash
# Runs ON the VM, from cloud/vm_af3_setup.sh. Puts exactly one AlphaFold3
# weights file into DEST, a folder outside the repository checkout, and reports
# only its file name, size and SHA-256 (also to OUT/weights.json).
#
#   WEIGHTS=random bash cloud/vm_af3_weights.sh DEST OUT
#   WEIGHTS=gcs bash cloud/vm_af3_weights.sh DEST OUT
#
# random  generated on the VM by af3_tpu/make_random_params.py (seed 0).
# gcs     copied from the private bucket object whose gs:// URI is in
#         WEIGHTS_URI_FILE (default ~/.af3_weights_uri, uploaded by
#         cloud/af3_run.sh). The URI, the destination path and gcloud's own
#         output are never printed, and the URI file is deleted once read.
set -euo pipefail
DEST=${1:?destination folder} OUT=${2:?output folder}
: "${WEIGHTS:?random or gcs}"
WEIGHTS_URI_FILE="${WEIGHTS_URI_FILE:-$HOME/.af3_weights_uri}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/third_party/alphafold3/.venv/bin/python"
die() { echo "!! $*" >&2; exit 4; }

case "$DEST" in "$ROOT" | "$ROOT"/*) die "the weights folder must be outside the repository checkout" ;; esac
mkdir -p "$DEST" && chmod 700 "$DEST"
[ -z "$(ls -A "$DEST")" ] || die "the weights folder is not empty"

case "$WEIGHTS" in
  random)
    "$PY" "$ROOT/af3_tpu/make_random_params.py" --output "$DEST/random_weights.bin.zst" \
      | grep -vE '^(wrote|schema):' ;;
  gcs)
    [ -s "$WEIGHTS_URI_FILE" ] || die "no weights URI on the VM"
    URI=$(cat "$WEIGHTS_URI_FILE")
    rm -f "$WEIGHTS_URI_FILE"
    case "$URI" in gs://?*/?*) ;; *) die "the weights URI is not a gs://bucket/object URI" ;; esac
    if ! gcloud storage cp "$URI" "$DEST/" > /dev/null 2>&1; then
      die "could not copy the weights from the bucket (URI and gcloud output not shown);" \
          "check that the VM's service account can read the object"
    fi
    unset URI ;;
  *) die "WEIGHTS must be random or gcs" ;;
esac

COUNT=$(find "$DEST" -maxdepth 1 -type f | wc -l | tr -d ' ')
FILE=$(find "$DEST" -maxdepth 1 -type f | head -1)
[ "$COUNT" = "1" ] || die "expected one weights file, found $COUNT"
NAME=$(basename "$FILE")
case "$NAME" in *.bin | *.bin.zst) ;; *) die "weights file name must end in .bin or .bin.zst" ;; esac
BYTES=$(wc -c < "$FILE" | tr -d ' ')
SHA=$( (sha256sum "$FILE" 2>/dev/null || shasum -a 256 "$FILE") | cut -d' ' -f1)
echo "   weights ($WEIGHTS): $NAME, $BYTES bytes, sha256 $SHA"
printf '{"mode": "%s", "file": "%s", "bytes": %s, "sha256": "%s"}\n' \
  "$WEIGHTS" "$NAME" "$BYTES" "$SHA" > "$OUT/weights.json"
