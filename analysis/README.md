# Analysis

Implements `notes/analysis_plan.md` (pre-specified). Reads only local files; creates no cloud resources.

```bash
uv venv --python 3.12 analysis/.venv
uv pip install --python analysis/.venv/bin/python -r analysis/requirements.txt
source analysis/.venv/bin/activate

python3 analysis/collect.py --out results/analysis/pilot          # tidy table + M1 (cached)
python3 analysis/contrasts.py --samples results/analysis/pilot/samples.csv --labels pilot
python3 analysis/cost.py --samples results/analysis/pilot/samples.csv --labels pilot
python3 -m pytest analysis -q
```

| Script | Plan | Output (under `results/analysis/<name>/`) |
|---|---|---|
| `score_ligand.py` | M1 | one prediction's ligand RMSD, PoseBusters checks, success; `--self_test` scores the crystal against itself |
| `collect.py` | sections 4, 8, 11 | `samples.csv`: one row per (platform, attention, libtpu, target, seed, repetition, sample); runs with `JAX_DEFAULT_MATMUL_PRECISION=highest` are the section 11 conditions V6h and Gh |
| `contrasts.py` | M2 to M5, sections 5, 6, 8, 11 | `contrasts/`: flip rates, M4/M5 medians and p90, bootstrap intervals, HW minus RC, Q3, pilot checks, the section 11 contrasts PR and HWh, `report.md` |
| `cost.py` | M6, M7 | `cost/`: cost per structure and per sample at each price, amortised compilation, memory per bucket, `report.md` |

M1, as implemented: the pocket is every crystal residue (data/rcsb/) with a heavy atom within 10 Å of a
heavy atom of the reference ligand (the PoseBusters SDF); the prediction is superposed on the crystal by
the pocket C-alpha atoms (Kabsch), residues matched by `label_seq_id` (the AlphaFold3 input is the RCSB
entity sequence; identity checked at 95% or more); the ligand RMSD is RDKit's symmetry-corrected
heavy-atom RMSD (`rdMolAlign.CalcRMS`, no realignment); PoseBusters runs with its `dock` configuration
(physical and chemical checks, no RMSD test) on the superposed ligand with the superposed predicted
protein (`--pb_protein crystal` uses the crystal protein instead). Success: RMSD below 2 Å and every
check passed. A prediction that cannot be read or matched is not a success; the reason is in `error`.

Everything produced from official-weight outputs stays under `results/` (gitignored). A published copy of
AlphaFold3 outputs keeps AlphaFold3's `TERMS_OF_USE.md`; the outputs are never used to train a model.
