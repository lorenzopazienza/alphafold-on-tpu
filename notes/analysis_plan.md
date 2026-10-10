# Pre-specified analysis plan: pilot and main study

Written 2026-10-09, **before any run with the official AlphaFold3 weights**. All earlier runs used random weights and served only engineering and sizing. Any later change to this plan is logged in section 10 with its date and reason, and the paper reports it.

## 1. Questions

- **Q1 (primary).** Under identical frozen inputs, weights, seeds and AF3 settings, does the accelerator change what AF3 returns, beyond what the seed and recompilation already change? Three levels:
  1. **Decision level:** which sample is ranked first, and whether the top-ranked prediction is a success.
  2. **Structure level:** how far the same (seed, sample) moves between platforms.
  3. **Confidence level:** how much ranking score, pLDDT and ipTM move.
- **Q2.** Time, cost and device memory per structure on each platform.
- **Q3.** How many seeds are needed before conclusions (success rate, chosen structure) stop depending on the platform.

## 2. Fixed elements

**Inputs:**
- Targets: the 60 PoseBusters V2 targets in `targets/posebusters_subset.csv`, with the 10-target pilot subset marked there.
- Inputs frozen in `inputs/manifest.csv`; SHA-256 checked on every VM before running.
- MSAs from the ColabFold server, frozen. No templates.
- Ligand by CCD code. Cofactors and extra ligands are not included.

**Model:**
- AF3 v3.0.4 (commit 85c4d20), with `af3_tpu/af3_tpu.patch`.
- Official weights: `af3.bin.zst`, copied from Google's bucket on 2026-10-09. The SHA-256 is recorded per run.
- AF3 defaults: 10 recycles, 5 diffusion samples per seed. Attention `xla` in the controlled arm.

**Software stacks:**

| Platform | Stack |
|---|---|
| TPU v5e and v6e | jax/jaxlib 0.10.2 with libtpu 0.0.43.2 (validated 2026-10-09) |
| L4 and CPU | AF3's pinned stack: jax 0.10.2, CUDA 12 wheels on L4 |

**Runs:**
- One fresh process per (platform, target, seed, repetition).
- Persistent compile cache **empty** at the start of each fresh repetition.

## 3. Conditions

| Code | Platform | Attention | Role |
|---|---|---|---|
| V5 | TPU v5e (`v5litepod-1`) | xla | controlled arm |
| V6 | TPU v6e (`ct6e-standard-1t`) | xla | controlled arm |
| G | NVIDIA L4 (`g2-standard-8`) | xla | controlled arm |
| C | CPU (`n2-highmem-16`, Ice Lake) | xla | reference, on a subset |
| GT | NVIDIA L4 | triton (AF3 GPU default) | "as a user would run it" arm |

**Repetitions:**
- **r1, r2:** two fresh-cache repetitions of the same (platform, target, seed), as separate processes, to measure recompilation variability.
- **w:** one warm-cache rerun of r1 for seed 1, expected bit-identical to r1. It checks that the cache does what AF3's docs say.

## 4. Per-run outputs and metrics

For each run and each sample k = 0..4, read from AF3's output:
- the coordinates;
- `ranking_score`, `ptm`, `iptm`, mean pLDDT, `has_clash`;
- the full confidences.

**M1. Ligand success**, against the crystal structure:
- pocket = protein residues with any heavy atom within 10 Å of a reference ligand heavy atom;
- superpose the prediction on the reference using the pocket Cα atoms;
- compute the symmetry-corrected heavy-atom RMSD of the ligand;
- **Success** = RMSD < 2 Å **and** PoseBusters-valid, using the `posebusters` package's checks for docked poses (record the package version and configuration).
- A prediction that cannot be parsed or checked counts as not a success, and is reported.

**M2. Top-1 identity:**
- per seed: the sample with the highest `ranking_score` among the seed's 5;
- overall: the (seed, sample) with the highest `ranking_score` across all seeds, as AF3 ranks them.

**M3. Ranking margin:** `ranking_score` of top-1 minus top-2 within the seed. This explains flips.

**M4. Prediction-to-prediction divergence** for the same (target, seed, sample) between two conditions:
- a bit-identical flag for coordinates;
- the ligand RMSD after the same pocket superposition as M1, using the first prediction as reference;
- the protein Cα RMSD after global superposition.

**M5. Confidence divergence:** absolute differences in `ranking_score`, mean pLDDT and `iptm` for the same (target, seed, sample).

**M6. Time and cost:**
- wall, inference and compile seconds from the run records;
- cost per structure = process wall time × hourly price, at on-demand, Spot and Flex-start list prices from `cloud/prices.csv` (with dates);
- a second variant that amortises compilation over a batch of same-bucket targets.

**M7. Memory:**
- TPU and GPU: the compiled peak plus arguments, and the allocator's peak reservation;
- CPU: max RSS;
- per token bucket, plus the largest bucket that runs on each chip.

## 5. Contrasts

All contrasts are paired on (target, seed) or (target, seed, sample).

| Contrast | Pairs | Measures |
|---|---|---|
| **Recompilation** RC(P) | r1 vs r2 on platform P | variability from compiling again on the same hardware |
| **Hardware** HW(P,Q) | r1 on P vs r1 on Q | variability from the accelerator |
| **Seed** SD(P) | seed s vs seed s' on P | ordinary sampling variability, as a yardstick |
| **Default use** DU | GT vs G | effect of AF3's GPU default attention |

**Pre-specified primary hardware pairs:**
- V6 vs G: TPU against GPU;
- V5 vs V6: two TPU generations;
- G vs C: GPU against CPU.

The other pairs are reported as secondary.

**Statistics per contrast:**
- **flip rate of top-1:** fraction of pairs where the top-1 sample differs;
- **flip rate of success:** fraction of pairs where the top-1's success differs;
- **flips that matter:** the fraction where top-1 changes **and** the two chosen structures differ in success;
- median and 90th percentile of M4 and M5;
- the bit-identical fraction.

**Uncertainty:**
- 95% confidence intervals from a **paired bootstrap over targets**: resample targets with replacement, 10,000 replicates, fixed seed 2026.
- The key quantity is the difference HW(P,Q) minus RC(P) for each flip rate, with its interval.
- No p-value hunting: every pre-specified contrast is reported, whatever it shows.

**Q3 (seeds needed):** for n = 1..5 seeds, take the overall top-1 over the first n seeds and compute the success rate per platform and its spread across platforms. The answer is the smallest n where every pairwise difference in success rate has a bootstrap interval within ±2 percentage points. If none reaches that, say so.

## 6. Pilot

**Purpose:**
- check that the full pipeline works with official weights;
- measure RC and HW on a small scale;
- size the main study.

**Design:**

| Platform | Targets | Seeds | Repetitions |
|---|---|---|---|
| V5, V6, G | the 10 pilot targets | 1, 2 | r1, r2, plus w for seed 1 |
| C | the 6 pilot targets of 512 tokens or fewer | 1, 2 | r1 only |
| GT | not in the pilot | | |

**Checks before reading any result:**
1. Every run exits 0, or its failure is recorded.
2. The w rerun is bit-identical to r1 on every platform.
3. **Success-rate sanity check:** the top-1 success on V6 over the 10 pilot targets.
   - If it is below 50%, stop and investigate the inputs before the main study (ColabFold MSAs, missing cofactors, full RCSB sequence with tags).
   - Published AF3 numbers on PoseBusters are not directly comparable because our inputs differ, so this is only a sanity threshold.

**Decision**, taken when the pilot is complete (target: 20 October 2026). The pilot decides the framing and sizing of the paper, not whether it exists:

- **Outcome A, the hardware matters at decision level.** HW shows top-1 or success flips on at least one primary pair while RC on the same platform shows fewer, or the ranking margins (M3) are small enough that flips are expected at scale.
  - Main study as in section 7, with the emphasis on decision-level effects and Q3.
- **Outcome B, the hardware moves numbers but not decisions.** HW shows nonzero M4 or M5 but no decision flips, and margins are comfortable.
  - Main study still runs: a clean null at decision level is a result.
  - The paper is framed as "AF3 inference is reproducible across accelerators at the decision level, within these bounds".
  - Budget moves towards more seeds (Q3) and towards cost and memory (Q2).
- **Outcome C, the pipeline is not trustworthy:** the sanity check fails, or w is not bit-identical.
  - Fix first, then repeat the pilot.

**Caution:** with 10 targets × 2 seeds = 20 pairs per contrast, pilot flip rates have wide intervals. They size the main study; they are not reported as findings.

## 7. Main study (after the pilot)

| Platform | Targets | Seeds | Repetitions |
|---|---|---|---|
| V5, V6, G | all 60 | 1..5 | r1, r2, plus w for seed 1 on 10 targets |
| GT | all 60 | 1..5 | r1 only |
| C | the 6 pilot targets of 512 tokens or fewer | 1, 2 | r1 only, if the budget allows |

- If the pilot gives outcome B, seeds may increase beyond 5 within budget. That change goes in section 10.
- **Frozen minimal result set**, matching the venue plan: 15 November 2026.

## 8. Missing data and exclusions

- A failed run (OOM, timeout, crash) is recorded with its cause, never silently dropped.
- A (target, seed) pair enters a contrast only if both conditions have it; the paper reports the counts excluded per contrast.
- Targets are never excluded after seeing results. A target with an input problem found later is reported and analysed both with and without it.

## 9. What the paper will not claim

- Nothing from random-weight runs, apart from the engineering facts:
  - the v5e segfault and its fix;
  - the bit-identical v6e outputs across libtpu versions;
  - memory and compile figures, labelled as such.
- No "first AF3-like model on TPU": the claim is the first measured port of DeepMind's official AF3.
- No comparison with published AF3 PoseBusters success rates as if the setups were the same.
- No statement on AF3's training hardware unless checked in the AF3 Supplementary Information.

## 10. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-09 | First version | |
| 2026-10-09 | Interpretations made while implementing the plan in `analysis/` (1 to 12 below), and the pilot schedule; recorded before any official-weights run | Points the plan leaves open, and an L4 quota constraint |
| 2026-10-10 | Provenance of the pilot sessions of 2026-10-09. The V5, V6 and C part 1 sessions (`20261009T154308Z_v5e_pilot`, `20261009T154318Z_v6e_pilot`, `20261009T154331Z_cpu_pilot_cpu_1`) were launched at 15:43 UTC from the working tree; the same files were committed unchanged about 2 minutes later, in `47b2082` (15:44:50 UTC). C parts 2 to 4 (`20261009T172653Z_cpu_pilot_cpu_2`, `20261009T172743Z_cpu_pilot_cpu_3`, `20261009T172752Z_cpu_pilot_cpu_4`) and both L4 parts were launched after that commit, from a clean tree at `47b2082`. Evidence: on every session, the AlphaFold3 patch hash and the manifest hash recorded on the VM equal those of `47b2082`; every frozen input matched the manifest at setup; no file uploaded to the VMs was modified after 15:26 UTC. The run records' own `repo.commit` is empty and `repo.dirty` false on every cloud session, because the VM receives the files without `.git`; those two fields carry no information for cloud runs. | Record the exact code of the pilot before analysis; the run records cannot show it themselves |
| 2026-10-10 | Network incident of 2026-10-09: the laptop running the launchers lost its network from about 16:43 to 17:24 UTC. The launchers stopped following after 10 failed SSH polls. **V6**: the VM went on; all 5 harness runs exited 0, the job finished at 17:30:33 UTC and uploaded the session to the results bucket, from which it was fetched. **C part 1**: likewise; its harness run exited 0, the job finished at 18:27:27 UTC and uploaded the session. **V5**: harness runs r1 and r2 of both seeds (4 of 5) exited 0 with 10 of 10 targets each; the launcher deleted the VM during the fifth run, the warm rerun `tpu_xla_seed1_warm`, which therefore holds 5 of its 10 targets (7U3J, 7NP6, 7V3N, 7BTT, 7VBU) and no `EXIT_CODE`. C parts 2 to 4 started after the incident and were not affected. | Document the incident and its effect on each session; no run was repeated or dropped because of it |
| 2026-10-10 | V5 warm-cache check (section 6, check 2): in the pilot session, w exists for 5 of the 10 targets (reason: the incident above). It is completed in a separate session on one on-demand v5e with the same stack (jax/jaxlib 0.10.2, libtpu 0.0.43.2), plan `harness/plans/v5e_warm_completion.yaml`, label `v5e_warm_check`: for each of the other 5 targets (7NPL, 7VC5, 7XQZ, 8EYE, 7D5C), a fresh-cache seed-1 run that only fills the compilation cache, then the warm rerun on the same VM, compared bit for bit with that fresh run. The fresh runs of that session are not r1 or r2 of the pilot and are excluded from every pilot contrast by their label. Check 2 for V5 is reported as the 5 targets of the pilot session plus the 5 of the completion session, each warm rerun against the fresh run of its own session. | Complete a pre-specified check that the incident cut short, without changing the pilot's repetitions |
| 2026-10-10 | L4 arm (G) of the pilot: run in two sequential parts on one VM at a time (GPU quota of 1), on-demand: `20261009T230055Z_l4_pilot_l4_1` (`pilot_l4_1.yaml`, job 23:03 to 01:29 UTC) and `20261010T013052Z_l4_pilot_l4_2` (`pilot_l4_2.yaml`, europe-west2-a, job 01:37 to 04:09 UTC). All 10 harness runs exited 0 (50 of 50 target runs). | The GPU quota allows one L4 VM; the plan's split into two parts is unchanged |
| 2026-10-10 | Numeric precision: section 11 added (precision statement, precision-matched arms V6h and Gh, contrasts PR and HWh). | A read-only audit of the AF3 source found that float32 matmuls without explicit precision run at each backend's default (single-pass bfloat16 operands on TPU, TF32 on the L4, IEEE float32 on CPU), and that on the L4 Tokamax runs a Triton kernel for the gated linear units. Added before any pilot outcome was opened |
| 2026-10-10 | Implementation of section 11 in `analysis/`, made after the OSF registration and before any analysis: (a) the h arms have r1 only, so the reading rule compares HWh(P,Q) with RC(P) of the default arm on the same platform; (b) the h arms enter only PR, HWh and pilot check 1 (exit codes), not SD, Q3 or pilot checks 2 and 3; (c) their deadlines are sized for float32 matmuls up to 3 times slower under `highest`; a run cut by the deadline is recorded as a failure (section 8), not dropped; (d) on the L4, Tokamax offers no setting to replace its Triton GLU kernel without changing AF3 code, so that kernel stays in Gh, with its float32 matmuls at `highest`. | Points section 11 leaves open |
| 2026-10-10 | Pilot checks (section 6), run before any result was read. **Check 1** passed: 207 processes recorded, all exit 0; the 5 V5 warm-rerun targets not run are those of the network incident row. **Check 3** passed at the threshold: top-1 success on V6 (overall top-1 across seeds, r1) 5 of 10 targets, 50%. **Check 2** failed: the warm rerun was bit-identical to its source run for V5 (10 of 10 targets, pilot and completion sessions) and V6 (10 of 10), but for only 5 of 10 L4 targets (7BTT, 7NP6, 7XQZ, 7V3N, 7VC5 differ). The run records show that no warm process loaded its source run's executable, on any platform: jax 0.10.2 writes the cache folder path into the compile options, which the persistent-cache key hashes, so the executables saved by the fresh runs were never found. The first warm target of each token bucket recompiled, and the later ones loaded that recompilation. Check 2 therefore tested recompilation in a new process, not the cache. Outcome C: no contrast was computed. All 177 fresh processes of the pilot started from an empty cache and compiled the model once. For check 2 on G, the per-sample full confidences of the two compared runs were copied from the results bucket (those sessions were fetched light). | Record the pre-specified checks and their outcome before any contrast |
| 2026-10-11 | Persistent-cache fix in the harness (`harness/run_af3.py`, no AlphaFold3 change). Every process that uses a persistent cache runs with `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none` (recorded in `session.json` `xla_env`), so the key no longer contains the folder path; on the L4, XLA's autotuning results are then kept in memory only. Each fresh run saves its cache in one folder per target, and the warm rerun of that target reads that folder. Each process records the model entries it found, wrote and loaded. Check 2 now has two parts, read in order. **2a and 2b, from the run records:** every fresh process started from an empty cache and compiled the model, and every warm process loaded the model from the cache with a key its source fresh process wrote. **Bit-identity:** only then, the warm rerun is compared bit for bit with its source run, as before. Verified on the Mac CPU with AlphaFold3 (toy input, random weights: before the fix a copied cache in another folder missed; after it, it hit) and in the fake-cloud tests. The earlier pilot sessions keep the label `pilot` and are referred to as pilot 0; the repeated pilot runs with the same plan files under the label `pilot2`, and every pilot analysis uses `pilot2` only. Pilot 0 is reported through this row and the previous one. | Without the fix, check 2 cannot test what section 3 says it tests (the cache does what AlphaFold3's documentation says) |
| 2026-10-11 | L4 determinism diagnostic before the repeated G and Gh arms: plan `harness/plans/diag_l4_determinism.yaml`, label `diag`, outside every contrast. On one L4, targets 7NP6 and 7BTT, seed 1, each of three configurations (AF3 defaults; `--xla_gpu_deterministic_ops=true`; `--xla_gpu_autotune_level=0`) runs two fresh compilations and two warm loads of the first one, to separate compile-time from run-time nondeterminism. The decision on how G enters the repeated pilot (keep AF3's defaults and report the effect in RC(G); add a deterministic arm; or require only 2a and 2b for G and report bit-identity) is recorded in a later row before G and Gh are repeated. The TPU and CPU arms of the repeated pilot do not depend on it and may run first. | Decide, on evidence and before the repeated G arm, how a platform that may not be run-to-run deterministic enters the contrasts |
| 2026-10-11 | Scope of the repeated pilot: under the strict reading of section 6 ("fix first, then repeat the pilot"), all arms are repeated with the fixed harness and the same plan files (V5, V6, G, C, V6h, Gh); the V5 warm completion session is superseded by a full V5 session. No earlier fresh run is invalid under the fix (all 177 started from an empty cache and compiled the model); the earlier warm reruns did not test the cache. Expected about $30, worst case about $56. | Section 6, Outcome C |

Entry of 2026-10-09, before any official-weights run. Interpretations as implemented in `analysis/`:

1. PoseBusters protein (M1): the checks run on the superposed predicted ligand with the predicted protein, moved by the same pocket superposition, as the conditioning protein (`dock` configuration). The crystal protein is available as an option (`--pb_protein crystal`) and is not the default.
2. Structure (M6): one (target, seed) prediction with its 5 samples. Cost per structure = run_alphafold process wall time / seeds in the process × hourly price; cost per sample (divided by 5) is also reported. VM setup and idle time are not included.
3. Amortised compilation (M6): wall − compile + compile / batch, the batch being the number of same-bucket targets among the 60 (19, 19, 14 and 8 for buckets 256, 512, 768 and 1024), a parameter of `analysis/cost.py`. Warm reruns are reported as measured, not amortised again.
4. Memory (M7): the compiled peak plus arguments is reported as written. XLA's compiled peak already includes the arguments on GPU (peak = temporaries + arguments + output) but not on TPU, so on GPU the sum counts the arguments twice; the compiled peak alone is reported as well. The CUDA allocator preallocates its pool and reports no reservation (0), so on GPU the allocator's peak bytes in use is reported instead. CPU: max RSS.
5. HW − RC (section 5): P is the first platform of each pair as named in section 5 (V6 for V6 vs G, V5 for V5 vs V6, G for G vs C), on the same resampled targets. HW − RC(Q) is also reported, labelled secondary. RC(C) does not exist in the pilot (C runs r1 only).
6. SD (section 5): pairs of seeds on r1. Its top-1 flip compares sample indices of different seeds, which are unrelated draws; it is reported with that note, and only its success flip and M4/M5 are read as informative.
7. Pilot check 3 (section 6): top-1 success on V6 uses the overall top-1 across seeds (as AlphaFold3 ranks), r1. The top-1 success per seed is reported as secondary.
8. Bit-identical (M4, pilot check 2): in `analysis/contrasts.py`, the mmCIF coordinates are identical as written. The bit-for-bit comparison of all outputs is `harness/compare_runs.py`.
9. Q3: each condition's success rate is computed on its own targets, and each pairwise difference on that pair's common targets (C covers 6 of the 10 pilot targets).
10. M4 reference ("first prediction"): side A of the contrast, the first-named condition. Prediction B is superposed on A by the pocket C-alpha atoms (pocket from the crystal, as in M1), and the ligand RMSD is symmetry-corrected. The RMSD does not depend on which side is the reference.
11. Pilot provisioning for G: on-demand (not Spot, and not Flex-start, which has no published price for g2-standard-8), with the launcher's default L4 zone list.
12. Correction: the earlier v5e vs v6e figure of 0.91 to 1.19 Å (random weights, libtpu 0.0.43.2, 7U3J and 7D5C, 5 samples each) was all-atom RMSD without superposition (`harness/compare_samples.py`), not RMSD after global superposition. On the same samples it reproduces as 0.907 to 1.186 Å; M4 as defined here gives a protein C-alpha RMSD after global superposition of 0.88 to 1.11 Å and a ligand RMSD after pocket superposition of 0.60 to 1.42 Å.

Pilot schedule: the L4 arm (G) of the pilot will run later than the TPU (V5, V6) and CPU (C) arms, because a GPU quota increase is pending.

## 11. Numeric precision (amendment of 2026-10-10, before any analysis)

**What the audit found** (AF3 v3.0.4 source, jax 0.10.2, tokamax 0.0.12):
- AF3 uses the same dtypes on every platform, CPU included: bfloat16 trunk and confidence Pairformer, float32 diffusion module, per-atom conditioning and output heads. Nothing in AF3, the patch or the harness changes them.
- Float32 matmuls without an explicit precision (most of the diffusion module, the per-atom conditioning, single-attention logits, the distogram and confidence logits) run at the backend's default: single-pass bfloat16 operands on TPU, TensorFloat-32 on the L4, IEEE float32 on CPU.
- On the L4, Tokamax runs a Triton kernel for the gated linear units (heuristic configuration, no runtime autotuning); TPU and CPU fall back to XLA. This holds in every L4 arm and cannot be changed without modifying AF3 code.
- The grid attention in the `xla` arm uses the same explicit precision (bfloat16 operands, float32 accumulation) on every platform.

**Consequence for the contrasts of section 5:**
- V5 vs V6 isolates the chip within one compiler and precision policy.
- V6 vs G and G vs C measure the whole platform stack as AF3 runs by default, including the precision and GLU-kernel differences. The paper says so explicitly and does not attribute these contrasts to the chip alone.

**Precision-matched arms (added):**

| Code | Platform | Setting | Role |
|---|---|---|---|
| V6h | TPU v6e | `xla`, `JAX_DEFAULT_MATMUL_PRECISION=highest` | precision-matched arm |
| Gh | NVIDIA L4 | `xla`, `JAX_DEFAULT_MATMUL_PRECISION=highest` | precision-matched arm |

- `highest` makes float32 matmuls without explicit precision run as IEEE float32 on the L4 and as six-pass bfloat16 (close to float32) on TPU. It does not change bfloat16 matmuls, the explicit-precision sites or the attention. CPU already runs these matmuls in float32, so C serves as the reference for both arms and is not rerun.
- **Pilot:** V6h and Gh on the 10 pilot targets, seeds 1 and 2, r1 only, in their own sessions with the same frozen inputs and weights.
- **Main study:** V6h and Gh on all 60 targets, seeds 1 and 2 at least; whether to extend them is decided with the pilot (section 6) and logged in section 10.

**Added contrasts** (secondary, pre-specified, paired and computed as in section 5, each reported whatever it shows):

| Contrast | Pairs | Measures |
|---|---|---|
| **Precision** PR(P) | r1 on P vs r1 on Ph, P in {V6, G} | effect of the backend's default matmul precision on the same hardware |
| **Matched hardware** HWh(V6,G) | V6h vs Gh | platform difference with float32 matmuls matched |
| **Matched hardware** HWh(G,C) | Gh vs C | as above, against the float32 CPU reference |

**Reading rule, fixed now:** for each primary pair, the paper reports HW(P,Q) next to HWh(P,Q) and PR. A difference that shrinks to the RC level under matched precision is attributed to the default matmul precision; what remains is attributed to the platform (chip, compiler and, on the L4, the GLU kernel), without separating those further. No direction is predicted.

**Residual differences not controlled:** accumulation order and fusion choices of each compiler, six-pass bfloat16 against IEEE float32, bfloat16 elementwise handling, and the L4's Triton GLU kernel. They are listed in the threats to validity.
