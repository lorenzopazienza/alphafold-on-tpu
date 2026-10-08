"""Frozen AlphaFold3 input JSONs for the selected targets, validated by AlphaFold3.

Run inside the pinned AlphaFold3 environment, after targets/ and the MSAs:

    cd third_party/alphafold3
    .venv/bin/python ../../inputs/build_inputs.py

For each row of targets/posebusters_subset.csv, writes data/inputs/<pdb_id>.json
(dialect alphafold3, version 1): the protein chain with the cached unpaired MSA
from data/msa/, empty paired MSA, no templates; the ligand by CCD code;
modelSeeds [1, 2, 3, 4, 5]. Each written file is then loaded with the loader
run_alphafold.py uses for --json_path and featurised with AlphaFold3's own
pipeline (first seed, no inference). Its token count must equal the one in the
targets CSV.

Writes inputs/manifest.csv and inputs/manifest.json (build environment).
Refuses to change an existing input file whose content would differ: frozen
inputs stay frozen. Delete the file by hand to rebuild it on purpose.
"""

import csv
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import af3_featurise  # noqa: E402

REPO = af3_featurise.REPO_ROOT
SUBSET = REPO / 'targets' / 'posebusters_subset.csv'
MSA_DIR = REPO / 'data' / 'msa'
INPUT_DIR = REPO / 'data' / 'inputs'
SEEDS = (1, 2, 3, 4, 5)


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def git_head(path) -> str:
  return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()


def main():
  INPUT_DIR.mkdir(parents=True, exist_ok=True)
  with open(SUBSET) as f:
    targets = list(csv.DictReader(f))
  ids = [t['pdb_id'] for t in targets]
  if len(set(ids)) != len(ids):
    raise SystemExit('Duplicate PDB IDs in the subset; input file names would collide.')

  rows, problems = [], []
  for n, t in enumerate(targets, 1):
    seq = t['sequence']
    key = sha256(seq.encode())
    if key != t['sequence_sha256']:
      raise SystemExit(f'{t["pdb_id"]}: sequence hash does not match the targets CSV')
    a3m_path, side_path = MSA_DIR / f'{key}.a3m', MSA_DIR / f'{key}.json'
    if not (a3m_path.exists() and side_path.exists()):
      problems.append(f'{t["pdb_id"]}: no cached MSA ({key[:12]}); run inputs/build_msas.py')
      continue
    a3m = a3m_path.read_text()
    side = json.loads(side_path.read_text())
    if sha256(a3m.encode()) != side['a3m_sha256']:
      raise SystemExit(f'{a3m_path}: SHA-256 does not match its sidecar')

    text = af3_featurise.protein_ligand_json(t['pdb_id'], seq, t['ligand_ccd'], SEEDS, a3m) + '\n'
    path = INPUT_DIR / f'{t["pdb_id"]}.json'
    if path.exists() and path.read_text() != text:
      raise SystemExit(f'{path} exists with different content; frozen inputs are not overwritten.')
    if not path.exists():
      path.write_text(text)

    t0 = time.time()
    try:
      result = af3_featurise.featurise_path(path)
    except Exception as e:  # AlphaFold3 rejects the input
      problems.append(f'{t["pdb_id"]}: AlphaFold3 rejected the input: {type(e).__name__}: {e}')
      continue
    seconds = time.time() - t0
    if result['num_tokens'] != int(t['num_tokens']) or result['num_seeds'] != len(SEEDS):
      problems.append(f'{t["pdb_id"]}: featurised {result}, targets CSV says {t["num_tokens"]} tokens')
      continue
    rows.append({
        'pdb_id': t['pdb_id'], 'ligand_ccd': t['ligand_ccd'], 'pilot': t['pilot'],
        'input_path': str(path.relative_to(REPO)), 'input_sha256': sha256(text.encode()),
        'input_bytes': len(text.encode()), 'sequence_sha256': key,
        'msa_sha256': side['a3m_sha256'], 'msa_rows': side['a3m_rows'],
        'msa_server': side['server'], 'msa_mode': side['mode'],
        'msa_query_utc': side['submit_utc'], 'num_tokens': result['num_tokens'],
        'bucket': result['bucket'], 'num_seeds': result['num_seeds'],
        'validated': 'af3_featurised', 'featurise_seconds': round(seconds, 1),
    })
    print(f'[{n}/{len(targets)}] {t["pdb_id"]}: {result["num_tokens"]} tokens, bucket '
          f'{result["bucket"]}, {side["a3m_rows"]} MSA rows, featurised in {seconds:.1f} s', flush=True)

  if problems:
    print('\n'.join(['Problems:'] + problems))
    raise SystemExit(1)

  with open(REPO / 'inputs' / 'manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
  packages = {}
  for name in ('alphafold3', 'jax', 'jaxlib', 'numpy', 'rdkit', 'dm-haiku', 'tokamax'):
    try:
      packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      packages[name] = None
  (REPO / 'inputs' / 'manifest.json').write_text(json.dumps({
      'built_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'built_by': 'inputs/build_inputs.py',
      'repo_commit': git_head(REPO),
      'alphafold3_commit': git_head(af3_featurise.AF3_DIR),
      'af3_buckets': list(af3_featurise.BUCKETS),
      'af3_ref_max_modified_date': af3_featurise.REF_MAX_MODIFIED_DATE.isoformat(),
      'validation': 'folding_input.load_fold_inputs_from_path + featurisation.featurise_input '
                    '(first seed, no inference)',
      'seeds': list(SEEDS), 'python': platform.python_version(), 'packages': packages,
      'targets_csv_sha256': sha256(SUBSET.read_bytes()), 'inputs': len(rows),
  }, indent=2) + '\n')
  print(f'Wrote inputs/manifest.csv with {len(rows)} validated inputs.')


if __name__ == '__main__':
  main()
