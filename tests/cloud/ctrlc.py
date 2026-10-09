"""ctrlc.py NAME PLATFORM MARKER [VAR=value ...]: Ctrl-C test for cloud/af3_run.sh.

Starts the launcher against the fake gcloud from the scratch repository, and
sends SIGINT to its process group once MARKER appears in the fake VM's job
log. Same guard and fake environment as common.sh: refuses to run unless
every fake in bin/ is executable and is the gcloud on the test PATH, and
gives gcloud an empty configuration with a nonexistent project.
"""
import glob, os, shutil, signal, subprocess, sys, time

KIT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(KIT))
WORK = (os.environ.get('CLOUD_TEST_WORK') or os.path.join(os.environ.get('TMPDIR', '/tmp'), 'af3-cloud-tests')).rstrip('/')
for f in glob.glob(f'{KIT}/bin/*'):
  if not os.access(f, os.X_OK):
    sys.exit(f'!! {f} is not executable: refusing to run')
PATH = f'{KIT}/bin:' + os.environ['PATH']
if shutil.which('gcloud', path=PATH) != f'{KIT}/bin/gcloud':
  sys.exit('!! gcloud on the test PATH is not the fake: refusing to run')

name, platform, marker = sys.argv[1:4]
D = f'{WORK}/state_{name}'
shutil.rmtree(D, ignore_errors=True)
os.makedirs(f'{D}/home'); os.makedirs(f'{D}/cloudsdk')
env = {k: v for k, v in os.environ.items() if k not in (
    'AF2_COMMIT', 'JAX_PLATFORMS', 'GOOGLE_APPLICATION_CREDENTIALS', 'CLOUDSDK_AUTH_ACCESS_TOKEN_FILE',
    'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE', 'CLOUDSDK_ACTIVE_CONFIG_NAME')}
env.update(CLOUDSDK_CONFIG=f'{D}/cloudsdk', CLOUDSDK_CORE_PROJECT='fake-project-no-access', PATH=PATH,
           FAKE_SCENARIO='ok', FAKE_LOG=f'{D}/log', FAKE_DIR=D, FAKE_HOME=f'{D}/home', FAKE_KIT=KIT,
           FAKE_REAL_PY=f'{REPO}/third_party/alphafold3/.venv/bin/python', DETACHED_POLL_S='1',
           VM_POLL_S='0', YES='1', PLATFORM=platform, FAKE_PLATFORM=platform)
for kv in sys.argv[4:]:
  k, v = kv.split('=', 1); env[k] = v


def pre():
  os.setpgrp(); signal.signal(signal.SIGINT, signal.SIG_DFL)


p = subprocess.Popen(['/bin/bash', 'cloud/af3_run.sh'], cwd=f'{WORK}/repo', env=env, preexec_fn=pre,
                     stdout=open(f'{D}/out', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while time.time() - t0 < 120:
  logs = glob.glob(f'{D}/home/alphafold-on-tpu/results/af3/*/job.log')
  if logs and marker in open(logs[0]).read():
    break
  time.sleep(0.3)
time.sleep(1.5)
os.killpg(p.pid, signal.SIGINT)
rc = p.wait()
fetched = len(glob.glob(f'{WORK}/repo/results/af3/*/'))
print(f'################ {name}: SIGINT after "{marker}" -> exit code {rc} | deleted: {os.path.exists(D + "/deleted")}'
      f' | still exists: {os.path.exists(D + "/created")} | fetched sessions: {fetched}')
subprocess.run(['pkill', '-f', f'{D}/home'])
shutil.rmtree(f'{WORK}/repo/results', ignore_errors=True)
