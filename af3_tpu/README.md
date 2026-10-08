# af3_tpu: AlphaFold3 inference on Cloud TPU, minimal port

The smallest change that lets DeepMind's official AlphaFold3 inference run on a
Cloud TPU, plus what is needed to try it without the official weights. Nothing
here is a measurement yet.

## Pinned AlphaFold3

| | |
|---|---|
| Repository | https://github.com/google-deepmind/alphafold3 |
| Tag | `v3.0.4` |
| Commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` |

The pin lives in `pin.sh`, which every script reads. AlphaFold3's source is not
copied into this repository: it is cloned into `third_party/alphafold3/`
(gitignored), and our change is kept as `af3_tpu.patch`.

```bash
git clone https://github.com/google-deepmind/alphafold3 third_party/alphafold3
git -C third_party/alphafold3 checkout v3.0.4
bash af3_tpu/apply.sh
```

`apply.sh` stops with an error if the checkout is not at the pinned commit or
the patch does not apply cleanly, and does nothing if it is already applied.

## What the patch changes, and why

It changes only `run_alphafold.py`, in two places (7 added lines, nothing
removed):

1. **`tpu` becomes a valid `--jax_backend`.** Upstream accepts only `gpu`,
   `cpu` and `mps`, and its device check ends in "Unsupported JAX backend" for
   anything else. The patch adds `TPU` to the `JaxBackend` list.
2. **On TPU, `--flash_attention_implementation=xla` is required.** The model
   passes a pair bias to `tokamax.dot_product_attention`, and Tokamax's TPU
   attention kernel rejects any bias ("Bias is not supported."). The XLA
   implementation handles the bias. The patch adds a TPU branch to the device
   check that refuses other settings, with the same kind of message upstream
   uses for CPU and MPS.

Device selection needed no change. The script already picks the device with
`jax.local_devices(backend=<value of --jax_backend>)[--gpu_device]`, and
`jax.local_devices(backend='tpu')` works, so `--jax_backend=tpu` selects TPU
chip 0 by default.

Nothing in the model code is touched. `tokamax.gated_linear_unit` has no TPU
kernel and falls back to XLA by itself; no other GPU-only code path was found.

## Files

| File | Purpose |
|---|---|
| `pin.sh` | AlphaFold3 repository, tag and commit |
| `af3_tpu.patch` | The change to `run_alphafold.py` described above |
| `apply.sh` | Applies the patch to the pinned checkout, loudly failing otherwise |
| `make_random_params.py` | Writes random weights from AlphaFold3's published parameter schema |
| `inputs/toy_118.json` | The 118-residue test input |
| `vm_smoke.sh` | Runs on the TPU VM; started by `cloud/af3_tpu_smoke.sh` |

## Random weights

The official weights are under separate non-commercial terms and are never used
or stored here. For performance work, AlphaFold3's `docs/model_parameters.md`
publishes the name, shape and dtype of every parameter and suggests generating
random ones. `make_random_params.py` follows the snippet in that document:
uniform(-1, 1) values (never all zeros, which accelerators can shortcut), a
zero identifier, and AlphaFold3's own record encoder.

```bash
cd third_party/alphafold3
uv run python ../../af3_tpu/make_random_params.py   # default: ~/af3_weights/random_weights.bin.zst
```

- Seed 0 by default (`--seed`), so the file is reproducible.
- 405 tensors, 368,384,538 parameters, 1,146,752,808 bytes of arrays,
  1,007,926,810 bytes compressed.
- SHA-256 of the file written on Lorenzo's Mac:
  `cc6da32ce59daadc2c6878e4465486a97b4072d755d35e2007bf5a8e1d1e5ced`.
- The script refuses to write inside this repository, and `.gitignore`
  excludes `*.bin`, `*.bin.zst` and `weights/`.
- `--model_dir` for `run_alphafold.py` is the folder holding the file.

With random weights the model computes the same operations as with real ones,
but the predicted structures and confidences are meaningless.

## Test input

`inputs/toy_118.json` is the summer 2026 AlphaFold3 test input: one protein
chain, empty `unpairedMsa` and `pairedMsa`, no templates, `modelSeeds: [1]`,
job name `af3_toy_test`, dialect `alphafold3` version 1 (still accepted by
v3.0.4).

The sequence is `TOY_SEQUENCE_118` from `src/make_af3_input.py` in the archived
repository (github.com/lorenzopazienza/alphafold-tpu-benchmark, commit
`a04af635303d73ab669f46eddaeeb31caae2ef0b`), which is the same sequence as
`TOY_SEQUENCE_118` in this repository's `src/spike_tpu_forward_pass.py`. The
file was written with the same dictionary and `json.dump(..., indent=2)` call
as that script with its default seed of 1.

Run it with `--run_data_pipeline=false`: no genetic search, no databases.

## Local install (macOS, Apple Silicon)

Following AlphaFold3's `docs/installation.md` (CPU-only section). HMMER is not
needed because the data pipeline is off.

```bash
cd third_party/alphafold3
uv venv --python 3.12
uv sync --frozen
uv run build_data
```

## Local smoke test

**Smoke test, not a measurement.** It shows only that the patched runner, the
random weights and the toy input work end to end. Recycles and diffusion
samples are cut to the minimum, the run includes JIT compilation, and it was
run once on a laptop.

Run on 2026-10-08 on Lorenzo's Mac (Apple M2 Max, macOS 27.0), Python 3.12.6,
jax 0.10.2, jaxlib 0.10.2, tokamax 0.0.12, dm-haiku 0.0.16, AlphaFold3 at
`85c4d20` with `af3_tpu.patch` applied:

```bash
cd third_party/alphafold3
uv run run_alphafold.py \
  --json_path=../../af3_tpu/inputs/toy_118.json \
  --output_dir=$HOME/af3_out/cpu_smoke \
  --model_dir=$HOME/af3_weights \
  --jax_backend=cpu \
  --flash_attention_implementation=xla \
  --run_data_pipeline=false \
  --num_recycles=1 \
  --num_diffusion_samples=1
```

- Exit code 0.
- Wall-clock 285.4 s for the whole command (AlphaFold3's own log: model
  inference with seed 1 took 251.56 s, compilation included). Peak memory
  10.2 GB.
- Outputs, under the job folder `af3_toy_test/`:
  `af3_toy_test_model.cif`, `af3_toy_test_summary_confidences.json`, and the
  same pair under `seed-1_sample-0/`.

The log shows one caught error, `NotImplementedError: Not supported on gpu.`,
from `tokamax.gated_linear_unit`. It is Tokamax trying its kernel, failing and
falling back to XLA, as expected. It names `gpu` because the `jax-mps` plugin
installed on macOS exposes a device of that kind; inference itself ran on
`cpu:0`. On TPU the same line should appear, naming the TPU.

## Cloud smoke test on a single-chip TPU

`cloud/af3_tpu_smoke.sh` runs the same test on a Spot `v5litepod-1` in
`europe-west4-b` (or a `v6e-1` in `europe-west4-a`), with `--jax_backend=tpu`.
Run it from the repository root in a shell where `cloud/env.sh` is not sourced:

```bash
bash cloud/af3_tpu_smoke.sh                                   # v5e
ACCEL=v6e-1 RUNTIME=v2-alpha-tpuv6e ZONE=europe-west4-a \
  bash cloud/af3_tpu_smoke.sh                                 # v6e
```

It creates the VM, arms a watchdog that deletes it after `MAX_HOURS` (default
2), copies `af3_tpu/` to it and runs `vm_smoke.sh` there, which:

1. installs AlphaFold3 at the pinned commit from its `uv.lock`, then applies
   the patch;
2. removes the `jax-cuda12-*` packages and installs `jax[tpu]==0.10.2`;
3. runs `build_data` and generates the random weights on the VM (they are
   never uploaded or copied back);
4. prints `jax.devices()` and the jax, jaxlib, libtpu and tokamax versions,
   and stops if JAX's default backend is not `tpu`;
5. runs the toy input with the flags above, but `--jax_backend=tpu`.

After the swap in step 2 the VM never calls `uv run`, because `uv run`
re-syncs the environment to `uv.lock` and would put the CUDA JAX back.

Whatever happens, the script then copies `~/af3_smoke/<session>/` back to
`results/af3_smoke/<timestamp>_<accel>/` and deletes the VM. The folder holds
`vm_smoke.log`, `run_alphafold.log`, `pip_freeze.txt`, `versions.json`,
`random_params.txt`, `command.txt`, the mmCIF and summary confidences under
`af3_output/`, and `session.json` with the AlphaFold3 commit, patch SHA-256,
package versions, accelerator, zone, exact command, exit code and wall time.

## Licences

AlphaFold3's code is under Apache 2.0 (since v3.0.3). `af3_tpu.patch` contains
modified lines of its `run_alphafold.py`. The official AlphaFold3 weights are
under separate non-commercial terms; this project does not use them.
