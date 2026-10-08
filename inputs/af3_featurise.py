"""AlphaFold3's own input parsing and featurisation, without inference.

Shared by targets/build_posebusters_subset.py (token counts) and
inputs/build_inputs.py (validation). Must run inside the pinned AlphaFold3
environment, because it imports alphafold3 and run_alphafold.py from
third_party/alphafold3:

    cd third_party/alphafold3 && .venv/bin/python ../../<script>.py

The bucket list and the reference date for CCD model coordinates are read from
run_alphafold.py's own flag defaults, so counts match what run_alphafold.py
does with default flags.
"""

import dataclasses
import datetime
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
AF3_DIR = REPO_ROOT / 'third_party' / 'alphafold3'
sys.path.insert(0, str(AF3_DIR))

from alphafold3.common import folding_input  # noqa: E402
from alphafold3.constants import chemical_components  # noqa: E402
from alphafold3.data import featurisation  # noqa: E402
import run_alphafold  # noqa: E402

BUCKETS = tuple(int(b) for b in run_alphafold._BUCKETS.default)
REF_MAX_MODIFIED_DATE = datetime.date.fromisoformat(
    run_alphafold._MAX_TEMPLATE_DATE.default)

_CCD = None


def ccd():
  """AlphaFold3's chemical components dictionary (built by build_data)."""
  global _CCD
  if _CCD is None:
    _CCD = chemical_components.Ccd()
  return _CCD


def featurise(input_json: str) -> dict:
  """Parses an AlphaFold3 input JSON string and featurises its first seed."""
  return _featurise(folding_input.Input.from_json(input_json))


def featurise_path(path) -> dict:
  """Loads an input file with the loader run_alphafold.py uses for
  --json_path and featurises its first seed."""
  fold_inputs = list(folding_input.load_fold_inputs_from_path(path))
  if len(fold_inputs) != 1:
    raise ValueError(f'{path}: expected one fold input, got {len(fold_inputs)}')
  return _featurise(fold_inputs[0])


def _featurise(fold_input) -> dict:
  """Featurises the first seed of a parsed input.

  Returns the number of tokens, the padded bucket and the number of seeds.
  Raises whatever AlphaFold3 raises on an invalid input. Token count and bucket
  do not depend on the seed, so one seed is enough.
  """
  first_seed = dataclasses.replace(fold_input, rng_seeds=[fold_input.rng_seeds[0]])
  batch = featurisation.featurise_input(
      first_seed,
      ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
      buckets=BUCKETS,
      ref_max_modified_date=REF_MAX_MODIFIED_DATE,
  )[0]
  mask = batch['seq_mask']
  return {
      'num_tokens': int(mask.sum()),
      'bucket': int(mask.shape[0]),
      'num_seeds': len(fold_input.rng_seeds),
  }


def protein_ligand_json(name, sequence, ccd_code, seeds, unpaired_msa=''):
  """AlphaFold3 input (dialect alphafold3, version 1, like the toy input) for
  one protein chain and one ligand given by CCD code. Empty paired MSA, no
  templates."""
  return json.dumps({
      'name': name,
      'modelSeeds': list(seeds),
      'sequences': [
          {'protein': {
              'id': 'A',
              'sequence': sequence,
              'unpairedMsa': unpaired_msa,
              'pairedMsa': '',
              'templates': [],
          }},
          {'ligand': {'id': 'B', 'ccdCodes': [ccd_code]}},
      ],
      'dialect': 'alphafold3',
      'version': 1,
  }, indent=2)
