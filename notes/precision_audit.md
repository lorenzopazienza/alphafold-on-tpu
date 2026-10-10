# Numeric precision audit: AlphaFold3 on CPU, NVIDIA L4 and TPU (v5e, v6e)

Scope: AlphaFold3 v3.0.4 (commit 85c4d20) with `af3_tpu/af3_tpu.patch`, jax/jaxlib 0.10.2,
tokamax 0.0.12, dm-haiku 0.0.16, libtpu 0.0.43.2, as run by `harness/run_af3.py` with
`harness/configs.yaml`. Read-only audit of source; nothing was executed, no results or weights
were opened. Paths below are relative to the repository root unless they start with
`AF3/` (= `third_party/alphafold3/src/alphafold3/`), `VENV/` (=
`third_party/alphafold3/.venv/lib/python3.12/site-packages/`) or `third_party/`. Line numbers
in the repository's own files are those of commit `47b2082`, before the section 11 configs
were added to `harness/configs.yaml`. Written 2026-10-09/10 as a draft outside the repository
and copied here unchanged apart from this paragraph and the note at the start of section 7.

## 1. Summary

**Same program, same dtypes everywhere.** Nothing in AF3, the patch or our harness changes
dtypes by platform. AF3 uses `bfloat16='all'` on every backend, CPU included. What changes by
platform is (a) how each XLA backend runs **float32 matmuls with no explicit precision**
(`precision=None`, i.e. DEFAULT), and (b) on the L4 only, Tokamax swaps in a **Triton kernel for
the gated linear units**.

| Part | Params | Activations | bf16 x bf16 dots (all platforms) | f32 dots, precision=None: CPU / L4 / TPU | f32 dots, `highest`: CPU / L4 / TPU |
|---|---|---|---|---|---|
| Input embedding: target_feat, relative/bond/template/MSA embeddings | bf16 weights, f32 LayerNorm | bf16 | exact products, f32 accumulate, bf16 out | n/a | n/a |
| Input embedding: per-atom conditioning encoder (`evoformer_conditioning`) | f32 | f32, cast to bf16 at exit | n/a | f32 / TF32 / bf16 1-pass | f32 / f32 / bf16 x6 (about f32) |
| Trunk: MSA module, Pairformer x48 | bf16 weights, f32 LayerNorm | bf16 (LayerNorm, GLU gating, attention softmax in f32) | same as above; grid attention pinned to BF16_BF16_F32 | single-attention q.k logits only: f32 / TF32 / bf16 1-pass | n/a |
| Diffusion module (conditioning, atom encoder/decoder, 24-block token transformer) | f32 (all 125 tensors) | f32 | n/a | **f32 / TF32 / bf16 1-pass** (most of its matmuls) | f32 / f32 / bf16 x6 |
| Confidence head: Pairformer x4 | bf16 weights, f32 LayerNorm | bf16 | as trunk | single-attention logits: f32 / TF32 / bf16 1-pass | n/a |
| Confidence head: PDE, PAE, pLDDT, resolved logits | f32 | f32 | n/a | f32 / TF32 / bf16 1-pass | n/a |
| Distogram head | f32 | f32 | n/a | logits: f32 / TF32 / bf16 1-pass | contact einsum: f32 / f32 / bf16 x6 |

"bf16 1-pass" = f32 operands rounded to bfloat16 (8-bit significand), f32 accumulation.
"TF32" = operands rounded to TensorFloat-32 (11-bit significand), f32 accumulation.
"bf16 x6" = six bf16 passes, close to f32 accuracy but not IEEE-identical.

## 2. Model dtypes (question 1)

**Global switch, not platform dependent.**
- `GlobalConfig.bfloat16: Literal['all','none','intermediate'] = 'all'` (AF3/model/model_config.py:34).
- `make_model_config` sets only attention implementation, samples, recycles and return flags;
  it never touches `bfloat16` (third_party/alphafold3/run_alphafold.py:432-447). It is called
  identically on every backend (run_alphafold.py:1040-1052).
- The patch adds only `JaxBackend.TPU` and a TPU branch that requires
  `--flash_attention_implementation=xla` (af3_tpu/af3_tpu.patch:9, 17-22). The checked-out diff
  equals the patch (verified with `git diff`). No dtype, precision or XLA flag change.
- The harness passes only `--jax_backend`, `--flash_attention_implementation`,
  `--run_data_pipeline=false`, the cache dir and optional recycles/samples
  (harness/run_af3.py:470-482). `harness/af3_entry.py` runs `run_alphafold.main` unchanged
  (af3_entry.py:163-166). The only `jax.config.update` is the compilation cache dir
  (run_alphafold.py:903-906). Conclusion: **bf16 is used on CPU exactly as on accelerators.**

**Weights.** Stored dtypes follow AF3's published schema (third_party/alphafold3/docs/
model_parameters.md): 138 bf16 tensors (trunk and confidence-Pairformer Linear weights), 266 f32
tensors (all LayerNorm scales/offsets, every diffusion-head tensor, the per-atom conditioning
encoder, the distogram and confidence output logits), 1 uint8 identifier. Loading is
`jnp.array(arr)` with the stored dtype (AF3/model/params.py:230); the only cast is Haiku's
`bfloat16_getter`, which casts a parameter to bf16 when a module requests bf16 and the stored
value is not bf16 (AF3/model/components/utils.py:46-57). Our random weights use the same schema
dtypes (af3_tpu/make_random_params.py:36-41, 91). The official weights used in the pilot were not
opened (rule); that they match the schema is assumed, not verified.

**Per part.**
- Input embedding: `create_target_feat_embedding` casts target features to bf16 inside
  `bfloat16_context` (AF3/model/model.py:159-178). The per-atom conditioning encoder reads f32
  reference-structure features (AF3/model/network/atom_cross_attention.py:51-72) with f32
  parameters, so it runs in f32; its token output is cast to bf16 (model.py:176-178).
- Trunk: dtype = bf16 (AF3/model/network/evoformer.py:265-269); recycled `prev` pair/single
  are cast to bf16 on entry (evoformer.py:280, 315) and the trunk outputs are cast back to f32
  after each recycle (model.py:298-299). Explicit upcasts inside the bf16 trunk:
  LayerNorm upcasts bf16 to f32 and casts back (AF3/model/components/haiku_modules.py:92-95,
  118-119); Tokamax GLU computes activation and product in f32 then casts to bf16
  (VENV/tokamax/_src/ops/gated_linear_unit/base.py:114-124); single attention casts q, k and
  bias to f32 and computes softmax in f32, then casts weights back to bf16
  (AF3/model/network/diffusion_transformer.py:160-168, called from modules.py:522-529);
  MSA attention computes `jax.nn.softmax` on bf16 logits (modules.py:115-117), which stays in
  bf16 for exp and division (VENV/jax/_src/nn/functions.py:615-623) while the sum accumulates in
  f32 (VENV/jax/_src/numpy/reductions.py:139, 234).
- Diffusion module: f32 throughout. Embeddings arrive as f32 (model.py:298-299), the token
  activations and conditioning are forced to f32 (AF3/model/network/diffusion_head.py:250,
  265-268), and every diffusion-head parameter is f32, so `bfloat16_context`
  (diffusion_head.py:221) casts nothing.
- Confidence head: Pairformer in bf16 (AF3/model/network/confidence_head.py:121-131), then pair
  activations cast to f32 (confidence_head.py:163) and single to f32 (confidence_head.py:244)
  for the f32 PDE/PAE/pLDDT/resolved logits; softmaxes in f32 (193, 221, 257, 269).
- Distogram head: f32 pair embeddings and f32 weights; logits by a DEFAULT-precision Linear,
  contact probabilities by a HIGHEST einsum (AF3/model/network/distogram_head.py:61-85).
- Host side: bf16 outputs are converted to f32 after transfer (run_alphafold.py:497-500).

## 3. Matmul precision and flags (question 2)

**`jax_default_matmul_precision`: not set anywhere.** Default `None`
(VENV/jax/_src/config.py:1755-1790), settable through the env var
`JAX_DEFAULT_MATMUL_PRECISION` (config.py:557) and part of the jit cache key (config.py:1789).
No occurrence in AF3, the patch, harness or cloud scripts (grep). AF3's `Linear` default precision
is `None` (haiku_modules.py:42, 249, 313), so unannotated dots use each backend's DEFAULT.

**Explicit `precision=` in the model (all HIGHEST, all on f32 data):**
- atom_cross_attention.py:52 (`embed_ref_pos`), :87 (`embed_pair_offsets`), :158
  (`embed_trunk_single_cond`), :188 (`atom_positions_to_features`), :228
  (`embed_trunk_pair_cond`), :292 (`embed_pair_offsets`, keys), :421
  (`atom_features_to_position_update`).
- diffusion_head.py:47 (random rotation), :81 (augmentation), :161
  (`pair_cond_initial_projection`), :185 (`single_cond_initial_projection`), :194
  (`noise_embedding_initial_projection`), :254 (`single_cond_embedding_projection`).
- distogram_head.py:84 (contact probabilities).
- AF3/jax/geometry/rotation_matrix.py:96, 102, 264, 276, 283 (template geometry helpers).
Everything else (all q/k/v/gating/output projections, transitions, triangle multiplication,
outer product mean, attention einsums at diffusion_transformer.py:164, 172, 314, 323 and
modules.py:123, 322, 405, 418) has `precision=None`.

**What DEFAULT means per backend (JAX docstring, VENV/jax/_src/lax/lax.py:2129-2150):**
"Has no impact on CPU backends ... only has an effect on float32 computations. DEFAULT: On TPU:
performs float32 computations in bfloat16. On GPU: uses tensorfloat32 if available (e.g. on A100
and H100 GPUs) ... HIGHEST: On TPU: ... 6 bfloat16 [passes]. On GPU: uses float32." Tokamax
encodes the same table (VENV/tokamax/_src/precision.py:43-64): TPU DEFAULT -> BF16_BF16_F32,
HIGHEST -> BF16_BF16_F32_X6; GPU (cc >= 8.0) DEFAULT/HIGH -> TF32_TF32_F32, HIGHEST ->
F32_F32_F32; CPU all -> F32_F32_F32.
- TPU: f32 DEFAULT dots round both operands to bf16 (single pass), accumulate in f32. When the
  inputs are already bf16 (trunk), DEFAULT is lossless on input: bf16 x bf16 products are exact
  in f32 and accumulate in f32 on the MXU, output rounded to bf16.
- L4 (Ada, cc 8.9): f32 DEFAULT dots use TF32 tensor cores. `--xla_gpu_enable_triton_gemm=false`
  (configs.yaml:39, from AF3's docker/Dockerfile:93) routes GEMMs to cuBLAS instead of Triton GEMM
  fusions; it changes the GEMM library, not the precision class. **There is no
  `--xla_gpu_enable_tf32` flag in this XLA:** it is absent from the 470 `xla_*` flag names
  compiled into VENV/jaxlib/libjax_common.dylib (macOS jaxlib, same XLA revision assumed for the
  CUDA plugin). That library does contain `BLAS_COMPUTATION_TYPE_TF32_AS_F32` and the flag
  `xla_gpu_default_to_alg_dot_bf16_bf16_f32` ("Use the dot precision algorithm
  ALG_DOT_BF16_BF16_F32 by default for f32 dots"). The cuBLAS env var `NVIDIA_TF32_OVERRIDE` is not
  set by us (grep) and is not captured by the harness's `xla_env` record (run_af3.py:427 keeps
  only `XLA_`, `JAX_`, `TF_`), but `cloud/vm_device_check.sh:37` logs `NVIDIA*` variables.
- CPU (n2-highmem-16, Ice Lake, no native bf16 matmul): f32 dots are IEEE f32 at every
  precision. bf16 dots are believed to be upcast to f32 by XLA:CPU (exact products, f32
  accumulation, bf16 output), i.e. the same numeric class as the accelerators.
- XLA_FLAGS: AF3's `run_alphafold.py` sets none; it only reads `XLA_FLAGS` to demand
  `--xla_disable_hlo_passes=custom-kernel-fusion-rewriter` on cc 7.x GPUs (run_alphafold.py:
  956-967). Ours: L4 `--xla_gpu_enable_triton_gemm=false` plus `XLA_PYTHON_CLIENT_PREALLOCATE=true`,
  `XLA_CLIENT_MEM_FRACTION=0.95` (memory only); CPU `JAX_PLATFORMS=cpu`; TPU no env at all
  (configs.yaml:26-66). No `LIBTPU_INIT_ARGS` anywhere (grep). The patch sets nothing.

## 4. Attention implementation (question 3)

Only the Pairformer/MSA-module grid attention (`GridSelfAttention`) goes through
`tokamax.dot_product_attention` with `implementation=flash_attention_implementation`
(modules.py:183-190). Single attention, MSA attention and the diffusion and atom transformers are
plain `jnp.einsum` + `jax.nn.softmax` and are identical across configs.

- `xla` (cpu_xla, l4_xla, tpu_xla): Tokamax base implementation
  (VENV/tokamax/_src/ops/attention/api.py:37-46, base.py:520-539, 601-639). With bf16 q/k/v and
  no precision argument, both dots get the explicit preset BF16_BF16_F32 on every backend
  (precision.py:148-152), so this attention is **not** subject to the backend-default difference.
  Logits: f32 accumulation, logits dtype f32 (base.py:535-536, 601-607), scale applied in f32
  after the dot, bias added in f32, masked with f32 min, softmax in f32 with max subtraction
  (base.py:662-670), normalized weights (f32) enter the P.V dot whose BF16_BF16_F32 algorithm
  rounds them to bf16 (VENV/jax/_src/lax/lax.py:2365-2370, 6082-6083), f32 accumulation, output
  cast to bf16 (base.py:633-639).
- `triton` (l4_triton = GT): Pallas Triton flash attention (attention/pallas_triton.py). Same
  presets, but q is pre-multiplied by the scale in bf16 before the dot (pallas_triton.py:173-177,
  80-81), online (blockwise) softmax with running max and rescaling, `exp2` with a log2(e) factor
  when `use_base2` (pallas_triton.py:212-217, 247-256), unnormalized p rounded to bf16 for P.V
  (pallas_triton.py:271), division by the f32 denominator at the end (pallas_triton.py:298-299).
  Same nominal dtypes, different rounding points and summation order: DU (GT vs G) is an
  algorithmic contrast on fixed hardware.
- `cudnn`: available in the flag but unused in this study.
- **Tokamax GLU differs by platform regardless of the attention flag.** AF3 calls
  `tokamax.gated_linear_unit` without `implementation` (modules.py:79, 300;
  diffusion_transformer.py:109). Default order is triton, mosaic, xla
  (VENV/tokamax/_src/ops/gated_linear_unit/api.py:31-45). Triton is skipped unless the device is a
  GPU with cc >= 8.0 (api.py:106; tokamax/_src/gpu_utils.py:75-91); Mosaic GPU needs sm90/sm100
  (gated_linear_unit/pallas_mosaic_gpu.py:45-46). So L4 uses the Pallas Triton GLU kernel
  (gated_linear_unit/pallas_triton.py:40-77: f32 accumulators, gate and projection rounded to
  the input dtype, then product in f32), while TPU and CPU use XLA einsum (base.py:114-124). For
  f32 inputs (diffusion transitions) the Triton dot with DEFAULT precision lowers to TF32
  (VENV/jax/_src/pallas/triton/lowering.py:2316, 2386-2392), consistent with XLA's TF32 on L4.

## 5. Certain vs uncertain

Certain (read in source of the installed versions): dtypes and casts per part; no platform switch
in AF3, patch or harness; no `jax_default_matmul_precision`, no TF32 or TPU flags set by us; the
list of explicit HIGHEST sites; Tokamax xla attention numerics and its BF16_BF16_F32 presets; GLU
kernel selection logic; Pallas Triton f32 DEFAULT -> TF32.

Uncertain (documented behaviour, not verifiable from source here):
1. TPU f32 DEFAULT = single-pass bf16 in libtpu 0.0.43.2 (libtpu is closed; JAX docstring and
   Tokamax table say so).
2. XLA:GPU f32 DEFAULT = TF32 on L4 in the CUDA plugin of jax 0.10.2 (JAX docstring, Tokamax
   table, `TF32_AS_F32` string; the CUDA plugin binary was not inspected).
3. Absence of a `--xla_gpu_enable_tf32` flag: checked in the CPU jaxlib's flag registry only.
4. XLA:CPU bf16 dot path (upcast to f32 vs oneDNN/XNNPACK) and whether `xla_cpu_enable_fast_math`
   is off by default (flag exists; default not read from the binary).
5. Intermediate bf16 rounding in fused elementwise chains: XLA may keep f32 between ops
   (`xla_allow_excess_precision` exists; default believed true), and backends fuse differently.
   Whether v5e and v6e run bf16 elementwise natively or via f32 is not documented here.
6. That the L4 runs actually used the Triton GLU (expected from source; check by the absence of
   Tokamax's "Failed to run implementation" GLU log line on L4 and its presence on TPU/CPU, or a
   `pallas_glu` kernel in a profile).
7. That no relevant env var (`JAX_DEFAULT_MATMUL_PRECISION`, `NVIDIA_TF32_OVERRIDE`,
   `LIBTPU_INIT_ARGS`) leaked from the VM environment: check `session.json` `xla_env` and
   `device_check.txt` of the live sessions (not opened in this audit).

## 6. Consequences per pre-specified contrast (question 4)

- **V5 vs V6 (v5e vs v6e): pure hardware contrast within one stack.** Same HLO, same dtypes,
  same libtpu/XLA:TPU compiler, same precision policy (f32 DEFAULT = bf16 1-pass on both), same
  GLU and attention implementations. Differences come from the chip and the code the compiler
  generates for it (MXU geometry, tiling, fusion, reduction order).
- **V6 vs G (v6e vs L4): hardware plus a precision difference.** Same program and dtypes, but
  the f32 DEFAULT dots (most of the diffusion module, the per-atom conditioning, every
  single-attention logit einsum in trunk and confidence Pairformers, the distogram and confidence
  output logits) take bf16 operands on TPU and TF32 operands on L4. In addition the L4 runs a
  Triton GLU kernel where TPU runs XLA. The bf16 trunk matmuls and the grid attention are
  precision-matched (BF16_BF16_F32 class on both).
- **G vs C (L4 vs CPU): hardware plus a precision difference.** The same f32 DEFAULT dots are TF32
  on L4 and IEEE f32 on CPU; the GLU is Triton on L4 and XLA on CPU. bf16 parts are matched in
  class. Note AF3 was validated on A100/H100 (third_party/alphafold3/docs/performance.md:185-191),
  where f32 DEFAULT is also TF32: G is the closest arm to the validated numerics, CPU is more
  precise and TPU less precise in the f32 parts.
- **GT vs G (DU):** same hardware and precision policy; only the attention algorithm differs.

**Proposed threats-to-validity text:**

> AlphaFold3 runs the same JAX program with the same parameter and activation dtypes on every
> platform (bfloat16 trunk and confidence Pairformer, float32 diffusion module and output heads),
> and nothing in our setup changes them; however, float32 matrix multiplications without an
> explicit precision, which include most of the diffusion module, run at each XLA backend's
> default precision, which uses single-pass bfloat16 operands on TPU, TensorFloat-32 on the L4
> and full float32 on the CPU, and on the L4 Tokamax also replaces XLA's gated linear units with a
> Triton kernel. The V5 vs V6 contrast therefore isolates the chip within one compiler and
> precision policy, whereas V6 vs G and G vs C measure the whole platform stack, including these
> precision and kernel differences.

## 7. Cheap control (proposed, not run)

Control A was adopted on 2026-10-10 as section 11 of `notes/analysis_plan.md`: arms V6h and Gh
(called V6_hi and G_hi below), configs `tpu_xla_highest` and `l4_xla_highest`, plans
`harness/plans/pilot_v6h.yaml` and `pilot_l4h*.yaml`. The cost figures below are this audit's
first estimate; the plans' own sizing supersedes them.

**Control A (recommended): matched f32 precision via `JAX_DEFAULT_MATMUL_PRECISION=highest`.**
- Mechanism: add the env var to new configs (e.g. `l4_xla_highest`, `tpu_xla_highest`) in
  `harness/configs.yaml` `env:`; the harness passes it to run_alphafold.py and records it in
  `session.json` `xla_env` (run_af3.py:397-400, 427). JAX reads it at import (config.py:557); it
  affects only dots with `precision=None`; explicit HIGHEST sites are unchanged; it enters the
  jit cache key, so no cache mixing. Tokamax also reads it (precision.py:75-77): the bf16 grid
  attention stays BF16_BF16_F32 (precision.py:148-152), the Triton GLU on f32 inputs switches
  from TF32 to IEEE f32 (lowering.py:2389-2392).
- Effect: L4 f32 dots become IEEE f32; TPU f32 dots become 6-pass bf16 (about f32); CPU is
  unchanged (already f32), so **the expensive CPU arm need not be rerun**.
- Isolates: G_hi vs C and V6_hi vs G_hi = platform stack with f32 dots matched (residual: bf16
  accumulation order, 6-pass vs IEEE, elementwise/fusion, Triton vs XLA GLU structure);
  G vs G_hi and V6 vs V6_hi on the same hardware = the size of the default-precision effect,
  directly comparable to RC and HW.
- Cost for the 10 pilot targets (buckets: 1 x 256, 5 x 512, 2 x 768, 2 x 1024), r1, per seed.
  Bucket 512/768 times are linear interpolations of the given 256 and 1024 figures:
  - L4: 126 + 5 x ~319 + 2 x ~513 + 2 x 706 = ~4,160 s = 1.16 h, ~$1.0 to $1.05 at
    $0.85 to $0.90/h. Seeds 1 and 2: ~$2.1, plus ~0.5 h setup (~$0.45).
  - v6e: 72 + 5 x ~120 + 2 x ~168 + 2 x 216 = ~1,440 s = 0.40 h, ~$0.54 at $1.35/h. Two seeds:
    ~$1.1, plus setup (~$0.7).
  - HIGHEST slows f32 matmuls (up to 6 passes on TPU, no tensor cores for f32 on L4); the slowdown
    is unknown, assume 1.5x to 3x on the whole run: **about $5 to $10 total for both platforms and
    both seeds**, roughly $3 to $5 for seed 1 only.

**Control B (optional): TPU-matched precision on the other platforms,
`JAX_DEFAULT_MATMUL_PRECISION=BF16_BF16_F32`.**
- L4 (G_bf): cheap, same cost as above without the slowdown (~$2.5 to $3 for two seeds). Makes
  V6 vs G_bf precision-matched in the f32 parts; the Triton GLU then casts its f32 operands to
  bf16 too (lowering.py:2361-2384). `XLA_FLAGS=--xla_gpu_default_to_alg_dot_bf16_bf16_f32=true`
  is an alternative for XLA dots only (it would leave the Triton GLU on TF32).
- CPU (C_bf), 6 targets <= 512 tokens: 1,609 + 5 x 3,901 = 21,114 s = 5.9 h per seed,
  ~$6.8 per seed, ~$13.5 for two seeds at $1.153824/h, plus setup and any emulation slowdown.
- Risk: XLA:CPU support for the BF16_BF16_F32 algorithm on f32-stored operands is not verified;
  a one-target compile check (CPU, bucket 256, `--num_recycles=1 --num_diffusion_samples=1`)
  should precede it.

Neither control should start before the live pilot runs end (repository rule: no edits during a
live cloud run).

## 8. Addendum (2026-10-10, checked after the audit): GLU kernel per platform

Uncertain item 6 is resolved. This was read from the AlphaFold3 logs of the **random-weight**
probe sessions only, which run the same code and configs as the pilot. No pilot log was
opened.

- **L4** (`20261009T065101Z_l4_probe`, `run_alphafold.log`): Tokamax runs
  `PallasTritonGatedLinearUnit` on "NVIDIA L4". The log shows "Autotuning cache miss" for it,
  since `tokamax/data/autotuning/nvidia_l4/` ships no entry.
- **v6e, v5e, CPU** (`20261008T165218Z_v6e_probe`, `20261009T125801Z_v5e_stack_validation`,
  `20261009T104902Z_cpu_probe`): Tokamax logs "Failed to run implementation" from
  `gated_linear_unit/api.py:116`. It then falls back to the XLA implementation, as expected.
- **No runtime autotuning on L4.** On a cache miss, Tokamax uses its heuristic config: the
  `tokamax_autotuning_cache_miss_fallback` flag defaults to `heuristics`
  (VENV/tokamax/_src/config.py:67-71; VENV/tokamax/_src/ops/op.py:426-446). So the L4's GLU
  configuration is chosen deterministically from the shapes, with no benchmarking at runtime,
  and adds no run-to-run variability to RC(G).
