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
