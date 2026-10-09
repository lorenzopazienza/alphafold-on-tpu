# Draft: TPU v5e compiler segfault (SIGSEGV in backend passes) for AlphaFold3 with 5 diffusion samples

Status: draft for jax-ml/jax, not filed. Lorenzo decides whether and where to file it
(jax-ml/jax issues; possibly also google-deepmind/alphafold3 as a heads-up).
Evidence: `results/af3/20261008T205351Z_v5e_bisect/` (first bisection, commit `29f9a3b` of
this repository), `results/af3/20261009T065048Z_v5e_bisect_samples/` (second bisection,
commit `e8b4a6e`) and `results/af3/20261008T165218Z_v6e_probe/` (v6e reference). All runs
used random weights; no official AlphaFold3 weights are needed to reproduce.

### Update 2026-10-09 (second bisection), and whether to file

- **Only 1 diffusion sample compiles** on v5e with libtpu 0.0.42.1: 2, 3, 4 and 5 samples
  all segfault (7U3J, 10 recycles), so it is not a threshold or an odd/even pattern but any
  sample batch larger than 1.
- **The scoped VMEM limit has no effect**: `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=8192`
  and `=32768` (default 16384) both still segfault with 5 samples.
- **libtpu 0.0.43.2 fixes it**: jaxlib 0.10.2 with libtpu 0.0.43.2 (build label
  `libtpu_lts_20260630_b_RC06`; 0.0.42.1 is `libtpu_lts_20260615_b_RC03`) compiles and runs
  the 5-sample program on the same v5e (compile 32.9 s; 1 sample with 0.0.42.1: 30.3 s).
  Each of its 5 samples matches the v6e 0.0.42.1 sample of the same index at 0.92 to 1.19 A
  coordinate RMSD (random weights), the other pairings being 53 to 100 A apart. 0.0.43.2 is
  outside the `libtpu==0.0.42.*` pin of `jax[tpu]==0.10.2`; its PyPI description only says
  "libtpu supports JAX 0.7.1 or newer". We did not test 0.0.43 or 0.0.43.1, so the fixing
  release is not narrowed down.
- The newest stack (`jax[tpu]==0.11.2`, libtpu 0.0.48) could not be tested with AlphaFold3
  v3.0.4: its pinned flax 0.12.2 fails at import (`AttributeError: jax.core.Effect was
  deprecated in JAX v0.10.0 and removed in JAX v0.11.0`).
- No public report or release note was found: JAX changelog 0.10.2 to 0.11.2, the libtpu
  PyPI descriptions 0.0.42.1 to 0.0.43.2, and the Cloud TPU release notes (no entries after
  2026-06-01) mention no TPU compiler segfault, VMEM or LLO fix.

**Still worth filing?** Yes, but as a short, low-priority report rather than a bug hunt,
because it appears fixed in a newer libtpu. What it would still give: (1) a record that
`jax[tpu]==0.10.2` pins a libtpu with which a common model (AlphaFold3 at its default
settings, whose `uv.lock` pins jax 0.10.2) segfaults at compile time on v5e, with no Python
error, so that others find the workaround; (2) the questions only the maintainers can answer:
is libtpu 0.0.43.x supported with jaxlib 0.10.2, which change fixed it, and could a compile
failure surface as an error instead of a segfault. A heads-up on google-deepmind/alphafold3
(v5e users need `libtpu==0.0.43.2` with the pinned jax) may help more people than the JAX
issue. The report text below is updated with these results.

---

## Title

TPU v5e (v5litepod-1): process segfaults during XLA compilation (BACKEND_PASSES) of AlphaFold3's
model when 5 diffusion samples are batched; the same program compiles on v6e

## Description

Compiling AlphaFold3's inference function (`jit(apply_fn)`) on a single-chip TPU v5e kills the
process with SIGSEGV during compilation. No Python exception and no XLA error status are
raised. The crash depends only on the number of diffusion samples AlphaFold3 batches with
`hk.vmap`: 1 sample compiles and runs; 2, 3, 4 and 5 samples (5 is AlphaFold3's default)
crash every time. The same jax, jaxlib and libtpu versions compile and run the 5-sample
program on TPU v6e (Trillium). With jaxlib 0.10.2 and libtpu 0.0.43.2 instead of 0.0.42.1,
the 5-sample program compiles and runs on the same v5e.

## Versions

| | |
|---|---|
| jax / jaxlib | 0.10.2 / 0.10.2 (`pip install "jax[tpu]==0.10.2"`) |
| libtpu | 0.0.42.1 (from PyPI, the version `jax[tpu]==0.10.2` pins: `libtpu==0.0.42.*`) |
| Python | 3.12.15 |
| Other | dm-haiku 0.0.16, tokamax 0.0.12, numpy 2.4.1 (full `pip freeze` available) |
| Environment variables | none of `TPU_*`, `LIBTPU_*`, `XLA_*`, `JAX_*` set (checked with `env`) |

## Hardware

| | crashes | works |
|---|---|---|
| TPU | v5litepod-1 (`TPU v5 lite`), 1 chip | v6e, `ct6e-standard-1t`, 1 chip (`TPU v6 lite`) |
| API / image | Cloud TPU VM, runtime `v2-alpha-tpuv5-lite` | Compute Engine, image family `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` |
| Zone | europe-west4-b | europe-west4-a |
| Host | AMD EPYC 7B13, 24 vCPU, 47 GiB RAM, Linux 6.5.0-1013-gcp (Ubuntu 22.04), `ulimit -s` 8192 | |

Identical package versions on both (byte-identical `pip freeze`).

## Reproduction

AlphaFold3 v3.0.4 (`85c4d20505fd5cef05eac22b534d4e793971ae69`) with a 7-line patch that only
lets `run_alphafold.py` accept `--jax_backend=tpu` and requires
`--flash_attention_implementation=xla` on TPU (the model code is untouched). Weights: random
tensors with AlphaFold3's published parameter shapes (the crash is at compile time, so the
values do not matter).

```bash
git clone https://github.com/google-deepmind/alphafold3 && cd alphafold3
git checkout 85c4d20505fd5cef05eac22b534d4e793971ae69
git apply af3_tpu.patch                  # attached: adds JaxBackend.TPU (7 lines)
uv sync --frozen --python 3.12
uv pip uninstall --python .venv/bin/python jax-cuda12-plugin jax-cuda12-pjrt
uv pip install --python .venv/bin/python "jax[tpu]==0.10.2"
.venv/bin/build_data
python make_random_params.py --output ~/af3_weights/random_weights.bin.zst   # attached

# passes (1 diffusion sample)
.venv/bin/python run_alphafold.py --json_path=input.json --output_dir=out \
  --model_dir=~/af3_weights --jax_backend=tpu --flash_attention_implementation=xla \
  --run_data_pipeline=false --num_diffusion_samples=1

# Segmentation fault during compilation (5 samples, AlphaFold3's default)
.venv/bin/python run_alphafold.py --json_path=input.json --output_dir=out \
  --model_dir=~/af3_weights --jax_backend=tpu --flash_attention_implementation=xla \
  --run_data_pipeline=false --num_diffusion_samples=5
```

`input.json`: any input. It reproduces with a 118-residue protein with an empty MSA (attached,
bucket 256), with a 205-token protein-ligand complex (PDB 7U3J) and with a 1023-token one
(7D5C). `--num_recycles` does not matter (1 or 10).

### What we varied (one fresh process each, same VM)

| run | recycles | samples | entry | result |
|---|---|---|---|---|
| toy, 118 residues | 1 | 1 | direct call | pass |
| toy, 118 residues | 10 | 5 | our harness | SIGSEGV |
| 7U3J | 1 | 1 | direct call | pass |
| 7U3J | 10 | 5 | direct call | SIGSEGV |
| 7U3J | 10 | 1 | harness | pass |
| 7U3J | 1 | 5 | harness | SIGSEGV |
| 7U3J | 10 | 5 | harness, no persistent compilation cache | SIGSEGV |
| 7U3J | 10 | 5 | harness, without our wrapper script | SIGSEGV |
| 7U3J | 10 | 5 | harness, without `JAX_LOG_COMPILES` | SIGSEGV |
| 7D5C, 1023 tokens | 10 | 1 | harness | pass |
| 7U3J | 10 | 2 | harness | SIGSEGV |
| 7U3J | 10 | 3 | harness | SIGSEGV |
| 7U3J | 10 | 4 | harness | SIGSEGV |
| 7U3J | 10 | 5 | harness, `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=8192` | SIGSEGV |
| 7U3J | 10 | 5 | harness, `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=32768` | SIGSEGV |
| 7U3J | 10 | 5 | harness, **libtpu 0.0.43.2** (jax/jaxlib 0.10.2) | **pass** |
| 7U3J | 10 | 5 | harness, `jax[tpu]==0.11.2` (libtpu 0.0.48) | not testable: AlphaFold3's pinned flax fails to import with jax 0.11 |

The last three rows ran in copies of the same venv with only those packages changed
(`pip freeze` otherwise identical).

## What happens

Python's faulthandler output (the main thread is printed as `Thread`, not `Current thread`, so
the fault is in a thread without a Python thread state, consistent with a libtpu compiler worker
thread):

```
Fatal Python error: Segmentation fault

Thread 0x00007fead7604c80 (most recent call first):
  File ".../jax/_src/compiler.py", line 353 in backend_compile_and_load
  File ".../jax/_src/profiler.py", line 420 in wrapper
  File ".../jax/_src/compiler.py", line 737 in _compile_and_write_cache
  File ".../jax/_src/compiler.py", line 469 in compile_or_get_cached
  File ".../jax/_src/interpreters/pxla.py", line 1516 in _cached_compilation
  File ".../jax/_src/interpreters/pxla.py", line 1733 in from_hlo
  File ".../jax/_src/interpreters/pxla.py", line 1239 in compile
  File ".../jax/_src/pjit.py", line 1175 in _pjit_call_impl_python
  ...
  File ".../alphafold3/run_alphafold.py", line 495 in run_inference
```

(The same stack appears with and without `--jax_compilation_cache_dir`: in jax 0.10.2
`_compile_and_write_cache` is on the compile path whenever `jax_enable_compilation_cache` is
true, its default, and the cache write is a no-op without a directory.)

libtpu's log (`/tmp/tpu_logs`, with `TPU_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0`) for the
crashing compile of `jit_apply_fn` shows that HLO optimisation, memory scheduling and memory
space assignment complete, and the process dies about 12 s into the backend stage:

```
21:04:49.83 deepsea_compiler_hlo_passes.cc] XLA::TPU running hlo passes for 14,749 instructions, module: jit_apply_fn
21:04:58.39 hlo_memory_scheduler.cc] Chose min-memory dfs sequence: 1.94GiB
21:04:59.22 deepsea_compiler_hlo_passes.cc] HLO_PASSES stage duration: 9.383455634s
21:05:00.15 heap_simulator.h] Primary algorithm finished with size: 975618048
21:05:01.82 .. 21:05:07.20  llo_loop.cc] [copy.N] 0-iteration loop inserted, body will not be executed (several threads)
21:05:08.49 lowering_emitter.cc] Successful retry compilation of fusion.4263 after retry_count=1
21:05:09.73 .. 21:05:11.36  llo_loop.cc] [copy.N] 0-iteration loop inserted, body will not be executed
<process killed by SIGSEGV; no further libtpu output>
```

The passing 1-sample compile logs the same messages (including the same "Successful retry
compilation" of the corresponding `f32[192,32,128,16]` atom-attention fusion), then
`BACKEND_PASSES stage duration: 14.37s`, `CODE_GENERATION`, and:

```
XLA::TPU program HBM usage: 1.00G / 15.75G
XLA::TPU program VMEM usage: 127.23M / 128.00M
```

So the 1-sample program already uses 127.23 of 128 MiB of VMEM. The crash is not an HBM
problem: the 5-sample program's buffer assignment totals 2.02 GiB of HBM (dump
`after_optimizations-memory-usage-report.txt`).

A per-pass HLO dump (`XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_pass_re=.*`) of the crashing
run writes all 236 HLO passes of `jit_apply_fn` (last one: `Rename_TensorCore_Fusions`,
`rename-tensor-core-fusion-ops`) plus `after_optimizations.txt` and the buffer assignment,
then nothing. The crash is therefore after the HLO pipeline, in the backend lowering /
code-generation stage, which has no HLO dump. We can attach the optimised HLO
(`after_optimizations.txt`, 10.8 MB) and the first-pass HLO (4.0 MB), or the whole dump
(117 MB compressed). No core dump was written (`ulimit -c 0`); we can rerun with core dumps
enabled if useful.

## Expected

Either a compiled executable, as on v6e, or a Python-visible error (for example a compile-time
VMEM or resource error) instead of a segfault.

## Notes

- Kernel log: no `segfault at` line for the Python process (Python's faulthandler handles the
  signal first).
- The 5-sample program differs from the 1-sample one only by a leading batch dimension of 5
  on the diffusion head's tensors (AlphaFold3 `diffusion_head.sample`, `hk.vmap` over samples
  inside an `hk.scan` with `unroll=4`), for example `f32[5,6144,768]` and `f32[5,192,32,768]`.
- libtpu 0.0.42.1 is the newest wheel allowed by `jax[tpu]==0.10.2` (`libtpu==0.0.42.*`).
  libtpu 0.0.43.2 with jaxlib 0.10.2 fixes the crash (see the table); we found no statement
  that this combination is supported, and did not test 0.0.43 or 0.0.43.1.
- With libtpu 0.0.43.2 the 5 samples are the expected ones: AlphaFold3 draws them with JAX's
  partitionable threefry, so each sample k matches sample k of the v6e run (0.92 to 1.19 A
  RMSD with random weights) and not the others (53 to 100 A).
