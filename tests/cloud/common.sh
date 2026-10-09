# Shared setup for the fake-gcloud tests in tests/cloud/. Sourced by run.sh,
# mkrepo.sh and the suites; ctrlc.py repeats the same guard in Python.
#
#   KIT    this folder (fakes in bin/, VM test doubles in vm/, fixtures/)
#   REPO   the repository root (the code under test is copied from here)
#   WORK   where every test writes: the scratch repository, one state_<name>/
#          folder per scenario, the reference structures. Outside the
#          repository: $CLOUD_TEST_WORK, default ${TMPDIR:-/tmp}/af3-cloud-tests
#
# Safety: the tests must never reach the real gcloud. cloud_test_guard stops
# unless every fake in bin/ is executable and `gcloud` on the test PATH is the
# fake, and fake_env (used for every launcher run) also points gcloud at an
# empty configuration folder with a nonexistent project and clears credential
# variables, so even a real gcloud reached by mistake has no account.
# macOS bash 3.2.

KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT/../.." && pwd)
WORK="${CLOUD_TEST_WORK:-${TMPDIR:-/tmp}/af3-cloud-tests}"
WORK="${WORK%/}"
mkdir -p "$WORK"

cloud_test_guard() {
  local f
  for f in "$KIT"/bin/*; do
    [ -x "$f" ] || { echo "!! $f is not executable: refusing to run (chmod +x it)" >&2; exit 99; }
  done
  [ "$(PATH="$KIT/bin:$PATH" command -v gcloud)" = "$KIT/bin/gcloud" ] \
    || { echo "!! gcloud on the test PATH is not $KIT/bin/gcloud: refusing to run" >&2; exit 99; }
}

# fake_env STATE_DIR [VAR=value ...] COMMAND ...: runs COMMAND in the fake
# environment of one scenario (state in STATE_DIR). Every value is passed
# quoted, so paths with spaces (also in PATH) are safe.
fake_env() {
  local d="$1"
  shift
  mkdir -p "$d/cloudsdk"
  env $FAKE_UNSET CLOUDSDK_CONFIG="$d/cloudsdk" CLOUDSDK_CORE_PROJECT=fake-project-no-access \
    PATH="$KIT/bin:$PATH" FAKE_LOG="$d/log" FAKE_DIR="$d" FAKE_HOME="$d/home" FAKE_KIT="$KIT" \
    FAKE_REAL_PY="$REPO/third_party/alphafold3/.venv/bin/python" DETACHED_POLL_S=1 VM_POLL_S=0 \
    VM_OP_WAIT_S=5 VM_CLEAR_SLEEP_S=0 "$@"
}

# Variables a test run must not inherit from the caller's shell.
FAKE_UNSET="-u AF2_COMMIT -u JAX_PLATFORMS -u GOOGLE_APPLICATION_CREDENTIALS \
-u CLOUDSDK_AUTH_ACCESS_TOKEN_FILE -u CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE -u CLOUDSDK_ACTIVE_CONFIG_NAME"

cloud_test_guard
