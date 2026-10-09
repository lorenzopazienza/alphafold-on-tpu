# tests/cloud: fake-gcloud tests for the cloud launchers

These tests run `cloud/af3_run.sh` and `cloud/v5e_bisect.sh` end to end against a fake
`gcloud`. The fake runs every SSH command for real in a local folder that stands in for
the VM, and injects failures on request. No test creates, reads or deletes anything in
Google Cloud.

```bash
bash tests/cloud/run_all.sh          # all four suites (80 scenarios), about 8 minutes
bash tests/cloud/mkrepo.sh           # or one scenario at a time:
bash tests/cloud/run.sh probe_l4 ok PLATFORM=l4 FAKE_PLATFORM=l4 ZONES=us-east4-a
```

Everything is written under `$CLOUD_TEST_WORK` (default `${TMPDIR:-/tmp}/af3-cloud-tests`),
outside the repository. That folder holds the scratch copy of the code under test, plus
one `state_<scenario>/` folder per scenario: the launcher output in `out`, the fake VM's
home in `home/`, the fake bucket in `bucket/` and the fetched results in `fetched/`.

## Safety

The tests must never reach the real `gcloud`. On 2026-10-09 a fake that was not
executable let `PATH` fall through to the real one, and a test created a real VM, which
was deleted after about 3.5 minutes. So:

- `common.sh` (sourced by every script) and `ctrlc.py` refuse to run unless every file
  in `bin/` is executable and `gcloud` on the test `PATH` is `bin/gcloud`;
- every launcher run gets an empty `CLOUDSDK_CONFIG` folder, a nonexistent project, and
  no credential variables. A real `gcloud` reached by mistake therefore has no account
  ("You do not currently have an active account selected").

After adding or editing a file in `bin/`, `chmod +x` it.

## Files

| File | Purpose |
|---|---|
| `common.sh` | Paths (`KIT`, `REPO`, `WORK`), the guard, the fake environment |
| `run.sh` | One launcher run: `run.sh NAME SCENARIO [VAR=value ...]`, summary printed |
| `mkrepo.sh` | Scratch copy of the code under test in `$WORK/repo`, plus the v6e reference structures if present |
| `ctrlc.py` | Ctrl-C during setup or during the plan |
| `suite_1_basics.sh` | Every platform, OOM, SSH drop and loss, deadline, GCS weights, input mismatches, refusals, prompt, Ctrl-C |
| `suite_2_resilience.sh` | Preemption, results bucket, `FETCH` light/full, L4 Flex-start, prices by region, the V0 to V9 bisection |
| `suite_3_zones_gpu_plans.sh` | L4 zone fallback, `JAX_PLATFORMS` on the GPU path (fails on `gpu`/`rocm`), the v5e samples plan |
| `suite_4_tpu_stack.sh` | TPU stack: `LIBTPU_VERSION`, the exit-5 check, `tpu_runtime` records, the `stack_validation` plan (two libtpu versions on one VM, switch failure, comparison after the fetch) |
| `run_all.sh`, `summarize.py` | All suites; a scenario fails the check if its exit code is not the expected one, a VM is left, or a GPU path sets `gpu`/`rocm` |
| `bin/gcloud` | The fake: create, describe, ssh, scp, delete, operations, `storage rsync`; scenarios and zone behaviours listed in its header |
| `bin/fake_rsync.py`, `bin/uv`, `bin/nvidia-smi`, `bin/setsid` | Fake bucket copy, venv installs (records the libtpu and jaxlib versions per venv), GPU listing, `setsid` for macOS |
| `vm/` | Test doubles on the fake VM: setup (it runs the real `cloud/vm_tpu_stack.sh` and `cloud/vm_device_check.sh`), the venv `python` (it records `JAX_PLATFORMS`, fails like JAX 0.10.2 on `gpu`/`rocm`, and writes libtpu logs with the real build labels), `run_alphafold.py` (on v5e, more than one sample segfaults only with libtpu 0.0.42.1, as measured) |
| `fixtures/` | Real insert-operation errors from 2026-10-08/09 (project and IDs redacted): us-east4-c suggesting us-east4-a, us-central1-b without a suggestion, a Flex-start stockout after a 32-minute queue |

`FAKE_REAL_PY` is the local AlphaFold3 venv's python (`third_party/alphafold3/.venv`). It
runs the test doubles and the real JAX device check for `PLATFORM=cpu`. The samples-plan
test compares structures only if `results/af3/20261008T165218Z_v6e_probe/` exists locally.
