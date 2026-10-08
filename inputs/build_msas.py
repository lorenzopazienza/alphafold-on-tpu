"""Frozen unpaired MSAs from the public ColabFold MMseqs2 server.

    python3 inputs/build_msas.py [--limit N]

For every unique protein sequence in targets/posebusters_subset.csv, asks the
ColabFold MSA server (https://api.colabfold.com, mode "env": UniRef30 plus the
ColabFold environmental databases, the ColabFold default) for an unpaired MSA
and caches it as

    data/msa/<sha256 of sequence>.a3m      combined A3M, query first
    data/msa/<sha256 of sequence>.json     provenance sidecar
    data/msa/<sha256 of sequence>.tar.gz   the server's raw result

The combined A3M is uniref.a3m followed by bfd.mgnify30.metaeuk30.smag30.a3m
without its repeated query, as ColabFold itself combines them.

Server etiquette (ColabFold README: queries must be serial and from a single
IP; large-scale work should use local databases):
  * one request at a time, never in parallel;
  * a sequence already cached is never queried again;
  * DELAY_BETWEEN_QUERIES_S between queries, POLL_S plus jitter between status
    polls (the reference client polls every 5 to 10 s);
  * on RATELIMIT, MAINTENANCE, HTTP 429 or 5xx: exponential backoff from 60 s
    up to 15 min; after MAX_BACKOFFS in a row the script stops, and a rerun
    resumes from the cache.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import pathlib
import random
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import http_util  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
SUBSET = REPO / 'targets' / 'posebusters_subset.csv'
MSA_DIR = REPO / 'data' / 'msa'
HOST = 'https://api.colabfold.com'
MODE = 'env'
DELAY_BETWEEN_QUERIES_S = 30
POLL_S = 10
MAX_BACKOFFS = 8
UNIREF_A3M = 'uniref.a3m'
ENV_A3M = 'bfd.mgnify30.metaeuk30.smag30.a3m'


def utc_now():
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


class Backoff:
  """Exponential backoff shared by every call to the server."""

  def __init__(self):
    self.count = 0
    self.total_s = 0

  def wait(self, reason):
    self.count += 1
    if self.count > MAX_BACKOFFS:
      raise SystemExit(f'Server still refusing after {MAX_BACKOFFS} backoffs ({reason}). '
                       'Stopping; rerun later to resume from the cache.')
    delay = min(60 * 2 ** (self.count - 1), 900)
    print(f'    backoff {self.count}/{MAX_BACKOFFS}: {reason}; waiting {delay} s', flush=True)
    self.total_s += delay
    time.sleep(delay)

  def reset(self):
    self.count = 0


def call(url, backoff, data=None):
  """One server call that returns parsed JSON or raw bytes, with backoff."""
  while True:
    try:
      status, headers, body = http_util.request(url, data=data, timeout=60)
    except OSError as e:
      backoff.wait(f'network error {e}')
      continue
    if status == 429 or status >= 500:
      backoff.wait(f'HTTP {status}')
      continue
    if status != 200:
      raise SystemExit(f'HTTP {status} from {url}: {body[:300]!r}')
    return headers, body


def query(seq, backoff, log):
  """Submits one sequence and returns (tar.gz bytes, provenance dict)."""
  prov = {'submit_utc': utc_now(), 'polls': 0}
  while True:
    headers, body = call(f'{HOST}/ticket/msa', backoff, data={'q': f'>101\n{seq}\n', 'mode': MODE})
    out = json.loads(body)
    log.append({'utc': utc_now(), 'call': 'submit', 'response': out})
    if out.get('status') in ('UNKNOWN', 'RATELIMIT', 'MAINTENANCE'):
      backoff.wait(f'submit status {out.get("status")}')
      continue
    if out.get('status') == 'ERROR':
      raise SystemExit(f'Server returned ERROR for a sequence: {out}')
    break
  prov.update(ticket=out.get('id'), submit_response=out, submit_headers={
      k: v for k, v in headers.items() if k.lower() in ('date', 'server')})
  while out.get('status') in ('UNKNOWN', 'RUNNING', 'PENDING'):
    time.sleep(POLL_S + random.randint(0, 5))
    _, body = call(f'{HOST}/ticket/{prov["ticket"]}', backoff)
    out = json.loads(body)
    prov['polls'] += 1
  log.append({'utc': utc_now(), 'call': 'final status', 'response': out})
  if out.get('status') != 'COMPLETE':
    raise SystemExit(f'Ticket {prov["ticket"]} ended with {out}')
  prov['complete_utc'] = utc_now()
  prov['final_response'] = out
  _, tar_bytes = call(f'{HOST}/result/download/{prov["ticket"]}', backoff)
  backoff.reset()
  return tar_bytes, prov


def combine(tar_bytes, seq):
  """Combined A3M (uniref then environmental, query once) and per-file info."""
  members = {}
  with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
    for m in tar.getmembers():
      if m.isfile():
        members[m.name] = tar.extractfile(m).read()
  texts = []
  for name in (UNIREF_A3M, ENV_A3M):
    if name not in members:
      raise SystemExit(f'{name} missing from result tar ({sorted(members)})')
    texts.append(members[name].decode().replace('\x00', '').strip())
  uniref, env = texts
  first = uniref.split('\n', 2)
  if len(first) < 2 or first[1].strip() != seq:
    raise SystemExit('Query is not the first sequence of uniref.a3m')
  env_records = env.split('\n>', 1)
  combined = uniref + ('\n>' + env_records[1] if len(env_records) == 2 else '') + '\n'
  info = {name: {'sha256': sha256(data), 'bytes': len(data),
                 'rows': data.decode(errors='replace').count('\n>') + data.startswith(b'>')}
          for name, data in sorted(members.items()) if name.endswith('.a3m')}
  return combined, info


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--limit', type=int, default=0, help='query at most N new sequences')
  args = parser.parse_args()
  MSA_DIR.mkdir(parents=True, exist_ok=True)
  repo_commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                               capture_output=True, text=True).stdout.strip()

  with open(SUBSET) as f:
    seqs = sorted({row['sequence'] for row in csv.DictReader(f)}, key=lambda s: sha256(s.encode()))
  todo = []
  for seq in seqs:
    key = sha256(seq.encode())
    a3m, side = MSA_DIR / f'{key}.a3m', MSA_DIR / f'{key}.json'
    if a3m.exists() and side.exists() and json.loads(side.read_text())['a3m_sha256'] == sha256(a3m.read_bytes()):
      continue
    todo.append(seq)
  print(f'{len(seqs)} unique sequences, {len(seqs) - len(todo)} cached, {len(todo)} to query', flush=True)
  if args.limit:
    todo = todo[:args.limit]

  backoff, started, queried = Backoff(), time.time(), 0
  for i, seq in enumerate(todo, 1):
    if i > 1:
      time.sleep(DELAY_BETWEEN_QUERIES_S)
    key = sha256(seq.encode())
    print(f'[{i}/{len(todo)}] {key[:12]} ({len(seq)} aa)', flush=True)
    log = []
    t0 = time.time()
    tar_bytes, prov = query(seq, backoff, log)
    combined, members = combine(tar_bytes, seq)
    (MSA_DIR / f'{key}.tar.gz').write_bytes(tar_bytes)
    (MSA_DIR / f'{key}.a3m').write_text(combined)
    sidecar = {
        'sequence_sha256': key, 'sequence_length': len(seq), 'sequence': seq,
        'server': HOST, 'endpoint': '/ticket/msa', 'mode': MODE,
        'user_agent': http_util.USER_AGENT, **prov,
        'query_seconds': round(time.time() - t0, 1),
        'tar_sha256': sha256(tar_bytes), 'tar_bytes': len(tar_bytes), 'members': members,
        'a3m_sha256': sha256(combined.encode()), 'a3m_rows': combined.count('\n>') + 1,
        'combine_rule': f'{UNIREF_A3M} + {ENV_A3M} without its query record',
        'server_databases': 'see https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History',
        'built_by': 'inputs/build_msas.py', 'repo_commit': repo_commit, 'call_log': log,
    }
    (MSA_DIR / f'{key}.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    queried += 1
    print(f'    {sidecar["a3m_rows"]} rows, {prov["polls"]} polls, {sidecar["query_seconds"]} s', flush=True)

  print(f'Done: {queried} queries in {time.time() - started:.0f} s, '
        f'{backoff.total_s} s spent in backoff', flush=True)


if __name__ == '__main__':
  main()
