"""Selects the protein-ligand target set from PoseBusters Benchmark V2.

Run inside the pinned AlphaFold3 environment (token counts come from
AlphaFold3's own featurisation):

    cd third_party/alphafold3
    .venv/bin/python ../../targets/build_posebusters_subset.py

Sources (downloaded once into data/, never committed):
  * PoseBusters paper data, Zenodo record 8278563 (CC BY 4.0): 428 complexes,
    MD5 checked against the Zenodo API.
  * PoseBusters V2 identifiers (308 PDB_CCD pairs, the subset used in the
    journal paper and in the AlphaFold3 paper), linked from the Zenodo record.
  * One mmCIF per candidate from the RCSB PDB (files.rcsb.org), for the full
    polymer sequence.

Selection rule (a pure function of those files and AlphaFold3 v3.0.4):
  1. Start from the 308 V2 identifiers. PDB ID and ligand CCD code come from
     the identifier (PDBID_CCD).
  2. Keep complexes whose PoseBusters protein file has exactly one protein
     chain (a chain with at least one standard amino acid) and no nucleic
     acid.
  3. The protein sequence is the canonical one-letter sequence
     (_entity_poly.pdbx_seq_one_letter_code_can) of the RCSB entity that
     contains that chain. The chain is matched to the entity's author chain
     IDs (_entity_poly.pdbx_strand_id) exactly or, because PDB-format files
     hold one character, to the only author chain ID that starts with it
     (for example A for AAA). Exclude it if there is no such match, the
     entity is not polypeptide(L) or the sequence has letters outside the 20
     standard amino acids.
  4. Exclude it if the ligand CCD code is not in AlphaFold3's bundled CCD.
  5. Build the AlphaFold3 input (that protein chain plus one copy of the
     ligand by CCD code, empty MSA, no templates) and featurise it with
     AlphaFold3's own pipeline. Cofactors and other ligands in the PoseBusters
     file are not part of the input. Exclude it if featurisation fails or the
     token count is above 1024.
  6. Bins by token count: <=256, 257-512, 513-768, 769-1024.
  7. Sort the eligible complexes by identifier. Allocate 60 targets to the
     bins as evenly as possible: visiting bins from fewest to most eligible
     (ties in bin order), each takes min(eligible, remaining // bins left).
     Within each bin, in bin order, draw that many without replacement with
     numpy.random.default_rng(SAMPLE_SEED).
  8. Pilot: shuffle the 60 (sorted by identifier) with
     numpy.random.default_rng(seed) and take the first 10, where seed is the
     smallest integer >= PILOT_SEED for which those 10 cover every bin. The
     seed used is recorded in sources.json.

Outputs: targets/posebusters_subset.csv (the 60), targets/posebusters_v2_screen.csv
(all 308 with the reason for each exclusion) and targets/sources.json.
"""

import collections
import csv
import datetime
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'inputs'))
import af3_featurise  # noqa: E402
from http_util import fetch  # noqa: E402
from alphafold3.cpp import cif_dict  # noqa: E402

REPO = af3_featurise.REPO_ROOT
DATA = REPO / 'data'
OUT_DIR = REPO / 'targets'

ZENODO_RECORD = 'https://zenodo.org/api/records/8278563'
ZIP_NAME = 'posebusters_paper_data.zip'
V2_IDS_URL = 'https://github.com/maabuu/posebusters/files/14516485/posebusters_pdb_ccd_ids.txt'
RCSB_URL = 'https://files.rcsb.org/download/{pdb}.cif'

MAX_TOKENS = 1024
BINS = (('le256', 1, 256), ('257-512', 257, 512), ('513-768', 513, 768), ('769-1024', 769, 1024))
N_TARGETS = 60
N_PILOT = 10
SAMPLE_SEED = 2026
PILOT_SEED = 2027

AA = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
NA = set('A C G U DA DC DG DT DI'.split())
STANDARD_LETTERS = set('ACDEFGHIKLMNPQRSTVWY')


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def cached(path: pathlib.Path, url: str) -> bytes:
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch(url)
    path.with_suffix(path.suffix + '.partial').write_bytes(data)
    path.with_suffix(path.suffix + '.partial').rename(path)
  return path.read_bytes()


def protein_chains(pdb_text: str):
  """Protein chain IDs and whether any nucleic acid is present."""
  chains, has_na = set(), False
  for line in pdb_text.splitlines():
    if line.startswith(('ATOM', 'HETATM')):
      res = line[17:20].strip()
      if res in AA:
        chains.add(line[21])
      elif res in NA:
        has_na = True
  return sorted(chains), has_na


def rcsb_entity(cif_text: str, chain: str):
  cif = cif_dict.from_string(cif_text)
  revisions = cif.get('_pdbx_audit_revision_history.revision_date', [])
  info = {
      'initial_release': revisions[0] if revisions else '',
      'latest_revision': revisions[-1] if revisions else '',
  }
  entities = [([s.strip() for s in strands.split(',')], kind, ''.join(seq.split()))
              for strands, kind, seq in zip(cif['_entity_poly.pdbx_strand_id'],
                                            cif['_entity_poly.type'],
                                            cif['_entity_poly.pdbx_seq_one_letter_code_can'])]
  for match in (lambda c: c == chain, lambda c: len(c) > 1 and c.startswith(chain)):
    hits = [(kind, seq) for chains, kind, seq in entities for c in chains if match(c)]
    if len(hits) == 1:
      return hits[0][0], hits[0][1], info
  return None, None, info


def bin_of(tokens: int) -> str:
  for name, lo, hi in BINS:
    if lo <= tokens <= hi:
      return name
  raise ValueError(tokens)


def allocate(counts: dict, total: int) -> dict:
  order = sorted(counts, key=lambda b: (counts[b], [n for n, _, _ in BINS].index(b)))
  alloc, remaining = {}, total
  for i, b in enumerate(order):
    alloc[b] = min(counts[b], remaining // (len(order) - i))
    remaining -= alloc[b]
  if remaining:
    raise SystemExit(f'Only {total - remaining} eligible targets for {total} slots.')
  return alloc


def main():
  started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
  pb_dir = DATA / 'posebusters'

  record = json.loads(fetch(ZENODO_RECORD))
  zip_meta = next(f for f in record['files'] if f['key'] == ZIP_NAME)
  zip_bytes = cached(pb_dir / ZIP_NAME, zip_meta['links']['self'])
  md5 = hashlib.md5(zip_bytes).hexdigest()
  if f'md5:{md5}' != zip_meta['checksum']:
    raise SystemExit(f'{ZIP_NAME}: MD5 {md5} does not match Zenodo {zip_meta["checksum"]}')
  ids_bytes = cached(pb_dir / 'posebusters_pdb_ccd_ids.txt', V2_IDS_URL)
  v2_ids = sorted(l.strip() for l in ids_bytes.decode().splitlines() if l.strip())
  print(f'PoseBusters V2: {len(v2_ids)} identifiers; zip MD5 verified')

  zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
  v1_ids = {l.strip() for l in zf.read('posebusters_benchmark_set_ids.txt').decode().splitlines() if l.strip()}
  ccd = af3_featurise.ccd()
  cache_path = DATA / 'posebusters' / 'token_counts.json'
  token_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

  screen = []
  for n, pb_id in enumerate(v2_ids, 1):
    pdb, lig = pb_id.split('_')
    row = {'posebusters_id': pb_id, 'pdb_id': pdb, 'ligand_ccd': lig, 'status': 'excluded',
           'reason': '', 'n_protein_chains': '', 'num_tokens': '', 'bucket': ''}
    screen.append(row)
    if pb_id not in v1_ids:
      row['reason'] = 'not in Zenodo set'
      continue
    member = f'posebusters_benchmark_set/{pb_id}/{pb_id}_protein.pdb'
    pdb_bytes = zf.read(member)
    chains, has_na = protein_chains(pdb_bytes.decode())
    row['n_protein_chains'] = len(chains)
    row.update(source_file=f'{ZIP_NAME}:{member}', source_sha256=sha256(pdb_bytes))
    if len(chains) != 1 or has_na:
      row['reason'] = 'nucleic acid present' if has_na else f'{len(chains)} protein chains'
      continue
    url = RCSB_URL.format(pdb=pdb)
    cif_path = DATA / 'rcsb' / f'{pdb}.cif'
    fresh = not cif_path.exists()
    cif_bytes = cached(cif_path, url)
    if fresh:
      time.sleep(0.2)
    kind, seq, info = rcsb_entity(cif_bytes.decode(), chains[0])
    row.update(protein_chain_id=chains[0], rcsb_url=url, rcsb_sha256=sha256(cif_bytes),
               rcsb_initial_release=info['initial_release'],
               rcsb_latest_revision=info['latest_revision'])
    if kind is None:
      row['reason'] = f'chain {chains[0]} not found in RCSB entity_poly'
      continue
    if kind != 'polypeptide(L)':
      row['reason'] = f'entity type {kind}'
      continue
    if set(seq) - STANDARD_LETTERS:
      row['reason'] = 'non-standard letters in sequence: ' + ''.join(sorted(set(seq) - STANDARD_LETTERS))
      continue
    if lig not in ccd:
      row['reason'] = 'ligand not in AlphaFold3 CCD'
      continue
    row.update(sequence=seq, sequence_length=len(seq), sequence_sha256=sha256(seq.encode()))
    input_json = af3_featurise.protein_ligand_json(pb_id, seq, lig, seeds=[1])
    key = sha256(input_json.encode())
    if key not in token_cache:
      t0 = time.time()
      try:
        token_cache[key] = af3_featurise.featurise(input_json)
      except Exception as e:  # AlphaFold3 rejects the input
        token_cache[key] = {'error': f'{type(e).__name__}: {e}'[:200]}
      cache_path.write_text(json.dumps(token_cache, indent=1))
      print(f'  [{n}/{len(v2_ids)}] {pb_id}: {token_cache[key]} ({time.time() - t0:.1f} s)', flush=True)
    result = token_cache[key]
    if 'error' in result:
      row['reason'] = 'AlphaFold3 featurisation failed: ' + result['error']
      continue
    row.update(num_tokens=result['num_tokens'], bucket=result['bucket'])
    if result['num_tokens'] > MAX_TOKENS:
      row['reason'] = f'{result["num_tokens"]} tokens > {MAX_TOKENS}'
      continue
    row.update(status='eligible', bin=bin_of(result['num_tokens']))

  eligible = [r for r in screen if r['status'] == 'eligible']  # already sorted by identifier
  by_bin = collections.OrderedDict((b, [r for r in eligible if r['bin'] == b]) for b, _, _ in BINS)
  alloc = allocate({b: len(v) for b, v in by_bin.items()}, N_TARGETS)
  rng = np.random.default_rng(SAMPLE_SEED)
  chosen = []
  for b, rows in by_bin.items():
    idx = sorted(rng.choice(len(rows), size=alloc[b], replace=False)) if alloc[b] else []
    chosen += [rows[i] for i in idx]
  chosen.sort(key=lambda r: r['posebusters_id'])
  for r in chosen:
    r['status'] = 'selected'
  needed_bins = {b for b in by_bin if alloc[b]}
  for pilot_seed in range(PILOT_SEED, PILOT_SEED + 1000):
    pilot_order = np.random.default_rng(pilot_seed).permutation(len(chosen))
    pilot = {chosen[i]['posebusters_id'] for i in pilot_order[:N_PILOT]}
    if {r['bin'] for r in chosen if r['posebusters_id'] in pilot} == needed_bins:
      break
  else:
    raise SystemExit('No pilot seed in 1000 tries covers every bin.')

  subset_cols = ['pdb_id', 'ligand_ccd', 'posebusters_id', 'n_protein_chains', 'n_input_chains',
                 'protein_chain_id', 'sequence_length', 'num_tokens', 'bucket', 'bin', 'pilot',
                 'sequence_sha256', 'source_file', 'source_sha256', 'rcsb_url', 'rcsb_sha256',
                 'rcsb_initial_release', 'rcsb_latest_revision', 'sequence']
  with open(OUT_DIR / 'posebusters_subset.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, subset_cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    for r in chosen:
      w.writerow(dict(r, n_input_chains=2, pilot='yes' if r['posebusters_id'] in pilot else 'no'))
  screen_cols = ['posebusters_id', 'pdb_id', 'ligand_ccd', 'status', 'reason', 'n_protein_chains',
                 'num_tokens', 'bucket', 'bin', 'source_sha256', 'rcsb_sha256']
  with open(OUT_DIR / 'posebusters_v2_screen.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, screen_cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    w.writerows(screen)

  sources = {
      'built_utc': started,
      'posebusters': {
          'zenodo_record': ZENODO_RECORD.replace('/api/records/', '/records/'),
          'zenodo_title': record['metadata']['title'],
          'zenodo_publication_date': record['metadata'].get('publication_date'),
          'licence': record['metadata'].get('license', {}).get('id'),
          'file': ZIP_NAME, 'url': zip_meta['links']['self'], 'md5_verified': zip_meta['checksum'],
          'sha256': sha256(zip_bytes), 'complexes_in_zip': len(v1_ids),
          'v2_ids_url': V2_IDS_URL, 'v2_ids_sha256': sha256(ids_bytes), 'v2_ids_count': len(v2_ids),
      },
      'rcsb': {'url_template': RCSB_URL, 'licence': 'PDB data: CC0 1.0 (wwPDB usage policy)',
               'entries_downloaded': sum(1 for r in screen if r.get('rcsb_url'))},
      'alphafold3': {'commit': subprocess.check_output(
                         ['git', '-C', str(af3_featurise.AF3_DIR), 'rev-parse', 'HEAD'], text=True).strip(),
                     'buckets': list(af3_featurise.BUCKETS),
                     'ref_max_modified_date': af3_featurise.REF_MAX_MODIFIED_DATE.isoformat()},
      'rule': {'max_tokens': MAX_TOKENS, 'bins': [list(b) for b in BINS], 'n_targets': N_TARGETS,
               'n_pilot': N_PILOT, 'sample_seed': SAMPLE_SEED, 'pilot_seed_start': PILOT_SEED,
               'pilot_seed_used': pilot_seed,
               'allocation': alloc},
      'counts': {
          'v2': len(v2_ids),
          'by_status': dict(collections.Counter(r['status'] for r in screen)),
          'eligible_by_bin': {b: len(v) for b, v in by_bin.items()},
          'exclusion_reasons': dict(collections.Counter(
              r['reason'].split(':')[0] if 'tokens >' not in r['reason'] else 'more than 1024 tokens'
              for r in screen if r['status'] == 'excluded')),
      },
  }
  (OUT_DIR / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
  print(json.dumps(sources['counts'], indent=2))
  print('allocation:', alloc, f'| pilot seed {pilot_seed}:', sorted(pilot))


if __name__ == '__main__':
  main()
