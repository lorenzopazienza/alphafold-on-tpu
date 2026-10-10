"""resume.py NAME PLATFORM MODE [VAR=value ...]: resume-mode test for cloud/af3_run.sh.

Starts the launcher against the fake gcloud from the scratch repository and
kills it with SIGKILL (no cleanup, as when the laptop dies) once the fake
VM's job is running, so the VM is left running. Then, with the same fake VM:
  1. RESUME_VM set to a VM that is not in the session's records: refused,
     nothing deleted;
  2. RESUME_VM set to the session's VM: follows, fetches, deletes.
MODE record: the session's launch record is used (a launch with this
launcher). MODE legacy: the record is removed first, as for a session
launched before records existed; the resume then needs ZONE as well.
Prints the second resume in run.sh's format (one "################" line and
the exit/created/deleted line), so summarize.py checks it. Same guard and
fake environment as common.sh.
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

name, platform, mode = sys.argv[1:4]
D = f'{WORK}/state_{name}'
R = f'{WORK}/repo'
shutil.rmtree(D, ignore_errors=True)
os.makedirs(f'{D}/home'); os.makedirs(f'{D}/cloudsdk')
env = {k: v for k, v in os.environ.items() if k not in (
    'AF2_COMMIT', 'JAX_PLATFORMS', 'GOOGLE_APPLICATION_CREDENTIALS', 'CLOUDSDK_AUTH_ACCESS_TOKEN_FILE',
    'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE', 'CLOUDSDK_ACTIVE_CONFIG_NAME', 'RESUME_VM', 'SESSION')}
env.update(CLOUDSDK_CONFIG=f'{D}/cloudsdk', CLOUDSDK_CORE_PROJECT='fake-project-no-access', PATH=PATH,
           FAKE_SCENARIO='ok', FAKE_LOG=f'{D}/log', FAKE_DIR=D, FAKE_HOME=f'{D}/home', FAKE_KIT=KIT,
           FAKE_REAL_PY=f'{REPO}/third_party/alphafold3/.venv/bin/python', DETACHED_POLL_S='1',
           VM_POLL_S='0', VM_OP_WAIT_S='5', VM_CLEAR_SLEEP_S='0', YES='1', PLATFORM=platform,
           FAKE_PLATFORM=platform, AF3_TPU_LOG_DIR=f'{D}/libtpu_logs')
for kv in sys.argv[4:]:
  k, v = kv.split('=', 1); env[k] = v


def pre():
  os.setpgrp()


# 1. Launch, and kill the launcher (SIGKILL: no trap runs) while the job runs.
p = subprocess.Popen(['/bin/bash', 'cloud/af3_run.sh'], cwd=R, env=env, preexec_fn=pre,
                     stdout=open(f'{D}/out_launch', 'w'), stderr=subprocess.STDOUT)
t0 = time.time()
while time.time() - t0 < 120:
  logs = glob.glob(f'{D}/home/alphafold-on-tpu/results/af3/*/job.log')
  if logs and 'running the plan' in open(logs[0]).read():
    break
  time.sleep(0.3)
time.sleep(1.5)
os.killpg(p.pid, signal.SIGKILL)
p.wait()
records = glob.glob(f'{R}/results/af3/.launcher/*/launch_record.txt')
session = os.path.basename(os.path.dirname(records[0])) if records else ''
vm = ''
if records:
  for line in open(records[0]):
    if line.startswith('VM_NAME='):
      vm = line.strip().split('=', 1)[1]
print(f'   launcher killed (SIGKILL) while the job ran: VM still exists: {os.path.exists(D + "/created")};'
      f' session {session}; VM {vm}')
if mode == 'legacy':
  shutil.rmtree(f'{R}/results/af3/.launcher/{session}')
  print('   launch record removed (a session launched before records existed)')
zone = {'v5e': 'europe-west4-b'}.get(platform, 'europe-west4-a')
extra = {'ZONE': zone} if mode == 'legacy' else {}


def resume(vm_name, out):
  e = dict(env, RESUME_VM=vm_name, SESSION=session, **extra)
  e.pop('PLATFORM')   # resume mode takes it from the session, not from the shell
  t = time.time()
  rc = subprocess.run(['/bin/bash', 'cloud/af3_run.sh'], cwd=R, env=e, stdout=open(f'{D}/{out}', 'w'),
                      stderr=subprocess.STDOUT).returncode
  return rc, int(time.time() - t)


# 2. A VM that is not in the session's records: refused, nothing touched.
wrong = vm[:-1] + ('x' if vm[-1] != 'x' else 'y')
rc, _ = resume(wrong, 'out_resume_wrong')
refusal = [l.strip() for l in open(f'{D}/out_resume_wrong') if l.startswith('!!')]
print(f'   resume with RESUME_VM={wrong} (not in the records): exit {rc}, VM still exists:'
      f' {os.path.exists(D + "/created")}, deleted: {os.path.exists(D + "/deleted")}')
for l in refusal[:2]:
  print(f'   | {l}')

# 3. The session's VM: follows, fetches, deletes.
rc, secs = resume(vm, 'out')
fetched = sorted(glob.glob(f'{R}/results/af3/*/'))
print(f'################ {name} (resume, mode {mode}) PLATFORM={platform} RESUME_VM={vm} SESSION={session}')
print(f'  exit code {rc} in {secs} s | VM created: yes, {zone} | deleted: {"yes" if os.path.exists(D + "/deleted") else "no"}'
      f' | still exists: {"YES" if os.path.exists(D + "/created") else "no"}')
print(f'  fetched: {os.path.relpath(fetched[-1], R) + "/" if fetched else "nothing"}')
for l in open(f'{D}/out'):
  if l.startswith(('!!', '>> Following', '>> Deleting', '   RESUME', '>> Plan finished', '   (SSH')):
    print(f'  | {l.rstrip()}')
ev = glob.glob(f'{R}/results/af3/*/launcher_events.txt')
if ev:
  print('  launcher_events.txt:')
  for l in open(ev[0]):
    print(f'    {l.rstrip()}')
subprocess.run(['pkill', '-f', f'{D}/home/'])
os.makedirs(f'{D}/fetched', exist_ok=True)
if os.path.isdir(f'{R}/results'):
  shutil.copytree(f'{R}/results', f'{D}/fetched/results', dirs_exist_ok=True)
shutil.rmtree(f'{R}/results', ignore_errors=True)
