"""The libtpu build that a TPU process actually loaded, from libtpu's own log.

    python3 harness/tpu_runtime.py --add_to results/af3/<session>/setup.json
    python3 harness/tpu_runtime.py --show [--log_dir /tmp/tpu_logs]

libtpu writes a log per process to /tmp/tpu_logs (AF3_TPU_LOG_DIR overrides
where it is read from; Google's "A Developer's Guide to Debugging JAX on
Cloud TPUs" gives that folder). Its first lines carry the build, for example

    ... b295d63588a.cc:884] Built on Aug 24 2026 14:55:32 (1787608532)
    ... b295d63588a.cc:894] Build label: libtpu_lts_20260630_b_RC06

tpu_runtime() reads the newest INFO log that has a build label, so call it
right after a TPU process ran (the setup's device check, the harness's
environment probe). The pip version is not in the log; callers pass it.

--add_to FILE adds a "tpu_runtime" block to a setup.json written by
cloud/vm_device_check.sh, taking libtpu_version from its "packages".
Standard library only.
"""

import argparse
import datetime
import json
import os
import pathlib
import re

LABEL = re.compile(r'Build label: (\S+)')
BUILT = re.compile(r'Built on (.+?) \((\d+)\)')


def log_dir():
  return pathlib.Path(os.environ.get('AF3_TPU_LOG_DIR', '/tmp/tpu_logs'))


def tpu_runtime(libtpu_version=None, directory=None):
  """{libtpu_version, build_label, build_date (UTC ISO), build_epoch, log}, plus 'error' if not found."""
  out = {'libtpu_version': libtpu_version, 'build_label': None, 'build_date': None,
         'build_epoch': None, 'log': None}
  d = pathlib.Path(directory) if directory else log_dir()
  if not d.is_dir():
    out['error'] = f'no libtpu log folder {d}'
    return out
  logs = sorted((p for p in d.iterdir() if p.is_file() and not p.is_symlink() and '.INFO.' in p.name),
                key=lambda p: p.stat().st_mtime, reverse=True)
  for p in logs:
    try:
      with open(p, errors='replace') as f:
        head = f.read(200_000)
    except OSError:
      continue
    m = LABEL.search(head)
    if not m:
      continue
    # The file name holds the host and user name; records keep only INFO.<date>.<pid>.
    out['build_label'], out['log'] = m.group(1), p.name[p.name.find('.INFO.') + 1:]
    b = BUILT.search(head)
    if b:
      epoch = int(b.group(2))
      out['build_epoch'] = epoch
      out['build_date'] = datetime.datetime.fromtimestamp(
          epoch, datetime.timezone.utc).isoformat(timespec='seconds')
    return out
  out['error'] = f'no libtpu log with a build label in {d}'
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--add_to', help='setup.json to add the tpu_runtime block to')
  ap.add_argument('--show', action='store_true', help='print the block')
  ap.add_argument('--log_dir')
  args = ap.parse_args()
  if args.add_to:
    path = pathlib.Path(args.add_to)
    setup = json.loads(path.read_text())
    setup['tpu_runtime'] = tpu_runtime((setup.get('packages') or {}).get('libtpu'), args.log_dir)
    path.write_text(json.dumps(setup, indent=2) + '\n')
    print('   tpu_runtime: ' + json.dumps(setup['tpu_runtime']))
  if args.show:
    print(json.dumps(tpu_runtime(None, args.log_dir), indent=2))


if __name__ == '__main__':
  main()
