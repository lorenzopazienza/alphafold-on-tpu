"""Fake `gcloud storage rsync [--recursive] [--exclude=R1,R2] SRC DST` on a local bucket folder.

gs://B/P maps to $FAKE_DIR/bucket/B/P. Excludes are Python regexes matched
(re.match) against the path relative to SRC, as gcloud documents.
"""
import os, re, shutil, sys

args = sys.argv[3:]
excludes, paths = [], []
for a in args:
  if a.startswith('--exclude='):
    excludes += a.split('=', 1)[1].split(',')
  elif a.startswith('--'):
    pass
  else:
    paths.append(a)
src, dst = paths
bucket = os.path.join(os.environ['FAKE_DIR'], 'bucket')
def local(p):
  return os.path.join(bucket, p[len('gs://'):]) if p.startswith('gs://') else p
if dst.startswith('gs://') and os.environ.get('FAKE_SCENARIO') == 'uploadfail':
  print('ERROR: (gcloud.storage.rsync) HTTPError 403: no storage.objects.create permission', file=sys.stderr)
  sys.exit(1)
s, d = local(src), local(dst)
if not os.path.isdir(s):
  print(f'ERROR: (gcloud.storage.rsync) source not found: {src}', file=sys.stderr)
  sys.exit(1)
n = 0
for root, _, files in os.walk(s):
  for f in files:
    rel = os.path.relpath(os.path.join(root, f), s)
    if any(re.match(x, rel) for x in excludes):
      continue
    os.makedirs(os.path.join(d, os.path.dirname(rel)), exist_ok=True)
    shutil.copy2(os.path.join(root, f), os.path.join(d, rel))
    n += 1
with open(os.environ['FAKE_LOG'], 'a') as log:
  log.write(f'   fake rsync copied {n} files {src} -> {dst} (excludes {excludes})\n')
