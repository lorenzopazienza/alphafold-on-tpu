"""M1 (notes/analysis_plan.md, section 4): ligand success of AlphaFold3 predictions.

    analysis/.venv/bin/python analysis/score_ligand.py --target 7U3J --prediction PATH/model.cif
    analysis/.venv/bin/python analysis/score_ligand.py --target 7U3J --self_test

Against the crystal structure, as the plan specifies:
  1. Pocket: protein residues of the target's chain in the crystal structure
     (data/rcsb/<PDB>.cif) with any heavy atom within 10 A of a heavy atom of
     the reference ligand.
  2. Superpose the prediction on the crystal structure using the pocket C-alpha
     atoms (Kabsch, residues matched by label_seq_id; AlphaFold3's input is the
     RCSB entity's canonical sequence, so residue i of the prediction is
     label_seq_id i of the crystal entity; checked per target by residue name).
  3. Symmetry-corrected heavy-atom RMSD of the ligand after that superposition
     (RDKit rdMolAlign.CalcRMS: no further alignment, all symmetry-equivalent
     atom mappings of the molecule).
  4. PoseBusters-valid: every check of the posebusters package's "dock"
     configuration (physical validity of a docked pose; it has no RMSD check)
     passes, with mol_cond the predicted protein in the same superposed frame
     (default) or the PoseBusters crystal protein (--pb_protein crystal).
  Success = RMSD < 2 A and PoseBusters-valid. A prediction that cannot be
  parsed or checked is not a success; the reason is recorded in "error".

Reference ligand: the PoseBusters V2 ligand file
(data/posebusters/posebusters_benchmark_set/<ID>/<ID>_ligand.sdf), which is
the crystal ligand instance PoseBusters evaluates, with bond orders. Its atoms
get their PDB atom names from the matching instance in data/rcsb/<PDB>.cif (by
coordinates), and predicted ligand atoms are matched by those names
(AlphaFold3 writes CCD atom names). Heavy atoms only; alternative conformers:
the first.

Also used by analysis/contrasts.py for M4 (pair_divergence): the same pocket
superposition between two predictions, their ligand RMSD, and the protein
C-alpha RMSD after global superposition.
"""

import argparse
import functools
import json
import pathlib
import sys
import tempfile

import gemmi
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

REPO = pathlib.Path(__file__).resolve().parent.parent
POCKET_RADIUS = 10.0
SUCCESS_RMSD = 2.0
MIN_RESIDUE_IDENTITY = 0.95
PB_CONFIG = 'dock'
RDLogger.DisableLog('rdApp.*')


class ScoreError(Exception):
  """A prediction or reference that cannot be scored; the message is recorded."""


def _subset_row(target):
  import csv
  for row in csv.DictReader(open(REPO / 'targets' / 'posebusters_subset.csv')):
    if row['pdb_id'] == target:
      return row
  raise ScoreError(f'{target} is not in targets/posebusters_subset.csv')


def _load_structure(path):
  st = gemmi.read_structure(str(path))
  st.setup_entities()
  st.remove_hydrogens()
  st.remove_alternative_conformations()
  return st


def _one_letter(name):
  info = gemmi.find_tabulated_residue(name)
  return info.one_letter_code.upper() if info else 'X'


def kabsch(mobile, target):
  """(R, t) minimising |mobile @ R.T + t - target|; arrays of shape (n, 3)."""
  mc, tc = mobile.mean(0), target.mean(0)
  h = (mobile - mc).T @ (target - tc)
  u, _, vt = np.linalg.svd(h)
  d = np.sign(np.linalg.det(vt.T @ u.T))
  r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
  return r, tc - mc @ r.T


def rmsd(a, b):
  return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


class Reference:
  """The crystal side of M1 for one target: pocket, C-alpha atoms, ligand."""

  def __init__(self, target):
    row = _subset_row(target)
    self.target, self.ccd, self.pb_id = target, row['ligand_ccd'], row['posebusters_id']
    self.sequence = row['sequence']
    pb_dir = REPO / 'data' / 'posebusters' / 'posebusters_benchmark_set' / self.pb_id
    self.pb_protein = pb_dir / f'{self.pb_id}_protein.pdb'
    sdf = pb_dir / f'{self.pb_id}_ligand.sdf'
    mol = Chem.MolFromMolFile(str(sdf), removeHs=False, sanitize=True)
    if mol is None:
      raise ScoreError(f'cannot read {sdf.name}')
    self.ligand = Chem.RemoveHs(mol)
    lig_xyz = self.ligand.GetConformer().GetPositions()

    st = _load_structure(REPO / 'data' / 'rcsb' / f'{target}.cif')
    model = st[0]
    chain_id = row['protein_chain_id']
    chains = [c for c in model if c.name == chain_id] or [c for c in model if c.name.startswith(chain_id)]
    poly = None
    for c in chains:
      p = c.get_polymer()
      if len(p):
        poly = p
        break
    if poly is None:
      raise ScoreError(f'{target}: protein chain {chain_id} not found in data/rcsb/{target}.cif')

    # Names for the SDF atoms: the crystal instance of the ligand at the same coordinates.
    self.ligand_names, best = None, None
    for c in model:
      for res in c:
        if res.name != self.ccd:
          continue
        names, worst = [], 0.0
        atoms = [(a.name, np.array(a.pos.tolist())) for a in res]
        for p in lig_xyz:
          name, pos = min(atoms, key=lambda x: np.linalg.norm(x[1] - p))
          worst = max(worst, float(np.linalg.norm(pos - p)))
          names.append(name)
        if worst < 0.05 and len(set(names)) == len(names) and (best is None or worst < best):
          self.ligand_names, best = names, worst
    if self.ligand_names is None:
      raise ScoreError(f'{target}: no {self.ccd} instance in the crystal matches the PoseBusters ligand')

    self.ca, self.resname, pocket = {}, {}, []
    for res in poly:
      if res.label_seq is None:
        continue
      self.resname[res.label_seq] = res.name
      ca = res.find_atom('CA', '*')
      if ca:
        self.ca[res.label_seq] = np.array(ca.pos.tolist())
      xyz = np.array([a.pos.tolist() for a in res])
      if len(xyz) and np.min(np.linalg.norm(xyz[:, None, :] - lig_xyz[None, :, :], axis=2)) <= POCKET_RADIUS:
        pocket.append(res.label_seq)
    self.pocket = sorted(s for s in pocket if s in self.ca)
    if len(self.pocket) < 3:
      raise ScoreError(f'{target}: fewer than 3 pocket residues with a C-alpha')
    # Residue numbering check: crystal residue i against the input sequence (AlphaFold3's residue i).
    same = sum(_one_letter(n) == self.sequence[i - 1] for i, n in self.resname.items() if 0 < i <= len(self.sequence))
    self.residue_identity = same / max(1, len(self.resname))
    if self.residue_identity < MIN_RESIDUE_IDENTITY:
      raise ScoreError(f'{target}: crystal label_seq_id does not follow the input sequence '
                       f'({self.residue_identity:.0%} identical)')

  def ligand_with(self, coords_by_name):
    """A copy of the reference ligand (bond orders) with the given coordinates by atom name."""
    missing = [n for n in self.ligand_names if n not in coords_by_name]
    if missing:
      raise ScoreError(f'ligand atoms missing in the prediction: {missing[:5]}')
    mol = Chem.Mol(self.ligand)
    conf = mol.GetConformer()
    for i, name in enumerate(self.ligand_names):
      conf.SetAtomPosition(i, gemmi.Position(*coords_by_name[name]).tolist())
    return mol


@functools.lru_cache(maxsize=None)
def reference(target):
  return Reference(target)


class Prediction:
  """Protein C-alpha atoms (by label_seq_id) and ligand atoms (by name) of one model."""

  def __init__(self, path, ccd, protein_chain='A', ligand_seqid=None):
    self.path = pathlib.Path(path)
    try:
      st = _load_structure(self.path)
    except Exception as e:  # noqa: BLE001 - any parse failure is recorded
      raise ScoreError(f'cannot parse {self.path.name}: {type(e).__name__}: {e}')
    self.structure, model = st, st[0]
    self.protein_chain = protein_chain
    chain = next((c for c in model if c.name == protein_chain), None)
    if chain is None:
      raise ScoreError(f'{self.path.name}: no protein chain {protein_chain}')
    self.ca, self.resname = {}, {}
    for res in chain.get_polymer():
      ca = res.find_atom('CA', '*')
      if res.label_seq is not None:
        self.resname[res.label_seq] = res.name
        if ca:
          self.ca[res.label_seq] = np.array(ca.pos.tolist())
    ligands = [r for c in model for r in c if r.name == ccd and (ligand_seqid is None or r.seqid.num == ligand_seqid)]
    if len(ligands) != 1:
      raise ScoreError(f'{self.path.name}: expected one {ccd} ligand, found {len(ligands)}')
    self.ligand = {a.name: np.array(a.pos.tolist()) for a in ligands[0]}


def _pocket_superposition(ref_ca, mobile_ca, pocket):
  common = [s for s in pocket if s in ref_ca and s in mobile_ca]
  if len(common) < 3:
    raise ScoreError(f'fewer than 3 pocket C-alpha atoms in common ({len(common)})')
  r, t = kabsch(np.array([mobile_ca[s] for s in common]), np.array([ref_ca[s] for s in common]))
  return r, t, len(common)


def _transform(coords, r, t):
  return {k: v @ r.T + t for k, v in coords.items()}


def posebusters_check(mol_pred, protein_pdb):
  """(valid, failed checks) from posebusters' dock configuration."""
  from posebusters import PoseBusters
  table = PoseBusters(config=PB_CONFIG).bust(mol_pred=mol_pred, mol_cond=str(protein_pdb))
  row = table.iloc[0]
  checks = {str(k): bool(v) for k, v in row.items() if isinstance(v, (bool, np.bool_))}
  failed = sorted(k for k, v in checks.items() if not v)
  return not failed and bool(checks), failed


def _write_protein_pdb(pred, r, t, path):
  st = pred.structure.clone()
  model = st[0]
  for name in [c.name for c in model if c.name != pred.protein_chain]:
    model.remove_chain(name)
  st.remove_ligands_and_waters()
  tr = gemmi.Transform(gemmi.Mat33(r.tolist()), gemmi.Vec3(*t.tolist()))
  model.transform_pos_and_adp(tr)
  st.setup_entities()
  st.write_pdb(str(path))


def score(target, prediction_path, pb_protein='predicted', protein_chain='A', ligand_seqid=None):
  """M1 for one predicted model; never raises, errors go to the result."""
  shown = pathlib.Path(prediction_path).resolve()
  shown = str(shown.relative_to(REPO)) if shown.is_relative_to(REPO) else shown.name
  out = {'target': target, 'prediction': shown, 'ligand_rmsd': None, 'pocket_residues': None,
         'pocket_ca_used': None, 'pb_valid': None, 'pb_failed': None, 'pb_protein': pb_protein,
         'success': False, 'error': None}
  try:
    ref = reference(target)
    pred = Prediction(prediction_path, ref.ccd, protein_chain, ligand_seqid)
    r, t, n = _pocket_superposition(ref.ca, pred.ca, ref.pocket)
    out.update(pocket_residues=len(ref.pocket), pocket_ca_used=n)
    mol = ref.ligand_with(_transform(pred.ligand, r, t))
    out['ligand_rmsd'] = round(float(rdMolAlign.CalcRMS(mol, ref.ligand)), 4)
    with tempfile.TemporaryDirectory() as tmp:
      if pb_protein == 'predicted':
        protein = pathlib.Path(tmp) / 'protein.pdb'
        _write_protein_pdb(pred, r, t, protein)
      else:
        protein = ref.pb_protein
      valid, failed = posebusters_check(mol, protein)
    out.update(pb_valid=valid, pb_failed=failed)
    out['success'] = out['ligand_rmsd'] < SUCCESS_RMSD and valid
  except ScoreError as e:
    out['error'] = str(e)
  except Exception as e:  # noqa: BLE001 - recorded, never fatal for the batch
    out['error'] = f'{type(e).__name__}: {e}'[:300]
  return out


def pair_divergence(target, path_a, path_b):
  """M4 between two predictions of the same target, prediction A as reference.

  ligand_rmsd: symmetry-corrected heavy-atom RMSD of B's ligand on A's, after
  superposing B on A by the pocket C-alpha atoms (pocket from the crystal, as
  in M1). ca_rmsd_global: protein C-alpha RMSD after global superposition.
  """
  out = {'ligand_rmsd': None, 'ca_rmsd_global': None, 'error': None}
  try:
    ref = reference(target)
    a = Prediction(path_a, ref.ccd)
    b = Prediction(path_b, ref.ccd)
    r, t, _ = _pocket_superposition(a.ca, b.ca, ref.pocket)
    mol_a, mol_b = ref.ligand_with(a.ligand), ref.ligand_with(_transform(b.ligand, r, t))
    out['ligand_rmsd'] = round(float(rdMolAlign.CalcRMS(mol_b, mol_a)), 4)
    common = sorted(set(a.ca) & set(b.ca))
    xa, xb = np.array([a.ca[s] for s in common]), np.array([b.ca[s] for s in common])
    rg, tg = kabsch(xb, xa)
    out['ca_rmsd_global'] = round(rmsd(xb @ rg.T + tg, xa), 4)
  except Exception as e:  # noqa: BLE001
    out['error'] = f'{type(e).__name__}: {e}'[:300]
  return out


def crystal_site(target):
  """(protein chain name, ligand residue number) of the reference ligand instance in data/rcsb/<target>.cif."""
  ref = reference(target)
  row = _subset_row(target)
  st = _load_structure(REPO / 'data' / 'rcsb' / f'{target}.cif')
  seqid = None
  for c in st[0]:
    for res in c:
      if res.name == ref.ccd and [a.name for a in res if a.name in ref.ligand_names]:
        pos = {a.name: np.array(a.pos.tolist()) for a in res}
        if all(n in pos for n in ref.ligand_names) and max(
            np.linalg.norm(pos[n] - ref.ligand.GetConformer().GetAtomPosition(i)) for i, n in enumerate(ref.ligand_names)) < 0.05:
          seqid = res.seqid.num
  chain = row['protein_chain_id']
  chain = next((c.name for c in st[0] if c.name == chain), next(c.name for c in st[0] if c.name.startswith(chain)))
  return chain, seqid


def self_test(target, pb_protein='crystal', path=None):
  """The crystal structure (or a copy of it at path, e.g. rigidly moved) scored as a prediction of itself."""
  chain, seqid = crystal_site(target)
  xtal = path or REPO / 'data' / 'rcsb' / f'{target}.cif'
  return score(target, xtal, pb_protein=pb_protein, protein_chain=chain, ligand_seqid=seqid)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--target', required=True)
  ap.add_argument('--prediction')
  ap.add_argument('--self_test', action='store_true', help='score the crystal structure against itself')
  ap.add_argument('--pb_protein', choices=('predicted', 'crystal'), default='predicted')
  args = ap.parse_args()
  if args.self_test:
    result = self_test(args.target)
  elif args.prediction:
    result = score(args.target, args.prediction, args.pb_protein)
  else:
    ap.error('give --prediction or --self_test')
  print(json.dumps(result, indent=2))
  return 0


if __name__ == '__main__':
  sys.exit(main())
