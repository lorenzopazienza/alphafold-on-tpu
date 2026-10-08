"""Checks frozen input JSONs against inputs/manifest.csv before any run.

    python3 harness/verify_inputs.py --targets 7U3J,7D5C [--manifest inputs/manifest.csv]

For every listed target, data/inputs/<pdb_id>.json must exist and its SHA-256
must equal input_sha256 in the manifest. Prints one line per target and exits
2 on any missing file or mismatch, 0 otherwise. Standard library only; run by
cloud/af3_run.sh on the laptop before creating a VM and by
cloud/vm_af3_setup.sh on the VM before any AlphaFold3 run.
"""

import argparse
import csv
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 24), b''):
      h.update(block)
  return h.hexdigest()


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--targets', required=True, help='comma-separated PDB IDs')
  ap.add_argument('--manifest', default='inputs/manifest.csv')
  args = ap.parse_args()

  with open(REPO / args.manifest) as f:
    manifest = {r['pdb_id']: r for r in csv.DictReader(f)}
  bad = 0
  for pdb_id in args.targets.split(','):
    row = manifest.get(pdb_id)
    if row is None:
      print(f'  {pdb_id}: not in the manifest')
      bad += 1
      continue
    path = REPO / row['input_path']
    if not path.exists():
      print(f'  {pdb_id}: {row["input_path"]} missing')
      bad += 1
      continue
    digest = sha256_file(path)
    ok = digest == row['input_sha256']
    bad += not ok
    print(f'  {pdb_id}: {"ok" if ok else "SHA-256 MISMATCH"} {digest[:16]}')
  if bad:
    print(f'!! {bad} input(s) do not match inputs/manifest.csv; nothing will run.')
    sys.exit(2)
  print('   all inputs match inputs/manifest.csv')


if __name__ == '__main__':
  main()
