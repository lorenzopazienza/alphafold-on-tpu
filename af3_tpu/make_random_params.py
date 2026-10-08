"""Writes random AlphaFold3 parameters for performance work.

Follows docs/model_parameters.md of the pinned AlphaFold3 checkout: every
tensor in the published schema gets uniform(-1, 1) values (never all zeros,
so accelerators cannot shortcut the arithmetic), and the identifier is zeros.
The values are meaningless; predictions made with them are not structures.

Run inside the AlphaFold3 environment, because it uses alphafold3's own
record encoder:

    cd third_party/alphafold3
    uv run python ../../af3_tpu/make_random_params.py [--output PATH] [--seed N]

The output must stay outside this repository; the script refuses otherwise.
--model_dir for run_alphafold.py is the folder that holds the file.
"""

import argparse
import hashlib
import math
import os
import pathlib
import re

import ml_dtypes
import numpy as np
import zstandard

from alphafold3.model import params

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO_ROOT / 'third_party/alphafold3/docs/model_parameters.md'
DEFAULT_OUTPUT = pathlib.Path('~/af3_weights/random_weights.bin.zst')
IDENTIFIER = '__meta__:__identifier__'

_LINE = re.compile(r'^name=(\S+)\s+dtype=(\S+)\s+shape=\(([^)]*)\)\s*$')
_DTYPES = {
    'float32': np.dtype(np.float32),
    'bfloat16': np.dtype(ml_dtypes.bfloat16),
    'uint8': np.dtype(np.uint8),
}


def read_schema(path):
  """Returns [(name, shape, dtype)] from the 'Parameters Schema' section."""
  text = pathlib.Path(path).read_text()
  section = text.split('## Parameters Schema', 1)
  if len(section) != 2:
    raise ValueError(f'No "## Parameters Schema" section in {path}')
  schema = []
  for line in section[1].splitlines():
    if not line.startswith('name='):
      continue
    m = _LINE.match(line)
    if not m:
      raise ValueError(f'Unparsable schema line: {line!r}')
    name, dtype, shape = m.groups()
    if dtype not in _DTYPES:
      raise ValueError(f'Unknown dtype {dtype!r} for {name}')
    shape = tuple(int(d) for d in shape.split(',') if d.strip())
    schema.append((name, shape, _DTYPES[dtype]))
  if not schema or schema[0][0] != IDENTIFIER:
    raise ValueError(f'Schema in {path} does not start with {IDENTIFIER}')
  return schema


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--output', type=pathlib.Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--schema', type=pathlib.Path, default=DEFAULT_SCHEMA)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  output = args.output.expanduser().resolve()
  if output.is_relative_to(REPO_ROOT):
    raise SystemExit(f'Refusing to write weights inside the repository: {output}')
  if not output.name.endswith('.bin.zst'):
    raise SystemExit(f'Output must end in .bin.zst: {output}')
  output.parent.mkdir(parents=True, exist_ok=True)

  schema = read_schema(args.schema)
  rng = np.random.default_rng(args.seed)
  n_values = 0
  n_bytes = 0
  partial = output.with_name(output.name + '.partial')
  with zstandard.open(partial, 'wb') as compressed:
    for name, shape, dtype in schema:
      if name == IDENTIFIER:
        arr = np.zeros(shape=shape, dtype=dtype)
      else:
        arr = rng.uniform(low=-1, high=1, size=shape).astype(dtype)
        n_values += math.prod(shape)
      n_bytes += arr.nbytes
      compressed.write(params.encode_record(*name.split(':'), arr))
  os.replace(partial, output)

  digest = hashlib.sha256(output.read_bytes()).hexdigest()
  print(f'schema:      {args.schema} ({len(schema)} tensors)')
  print(f'seed:        {args.seed}')
  print(f'parameters:  {n_values:,} (excluding the {IDENTIFIER} bytes)')
  print(f'array bytes: {n_bytes:,}')
  print(f'file bytes:  {output.stat().st_size:,} (zstd)')
  print(f'sha256:      {digest}')
  print(f'wrote:       {output}')


if __name__ == '__main__':
  main()
