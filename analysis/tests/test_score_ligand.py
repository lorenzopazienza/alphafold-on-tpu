"""M1 (analysis/score_ligand.py): the crystal ligand scored against itself, rigid-motion invariance, failures."""

import pathlib

import numpy as np
import pytest

import score_ligand

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = '7U3J'
needs_data = pytest.mark.skipif(not (REPO / 'data' / 'rcsb' / f'{TARGET}.cif').exists(),
                                reason='data/rcsb/ not present (gitignored; see data/README or scripts)')


def random_rigid(seed):
  rng = np.random.default_rng(seed)
  q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
  if np.linalg.det(q) < 0:
    q[:, 0] *= -1
  return q, rng.normal(scale=20.0, size=3)


def test_kabsch_recovers_a_rigid_motion():
  rng = np.random.default_rng(0)
  x = rng.normal(size=(50, 3)) * 10
  r, t = random_rigid(1)
  y = x @ r.T + t
  r2, t2 = score_ligand.kabsch(x, y)
  assert np.allclose(r2, r, atol=1e-8) and np.allclose(t2, t, atol=1e-6)
  assert score_ligand.rmsd(x @ r2.T + t2, y) < 1e-8


@needs_data
@pytest.mark.parametrize('pb_protein', ['crystal', 'predicted'])
def test_crystal_against_itself(pb_protein):
  res = score_ligand.self_test(TARGET, pb_protein=pb_protein)
  assert res['error'] is None
  assert res['ligand_rmsd'] == 0.0
  assert res['pb_valid'] is True and res['pb_failed'] == []
  assert res['success'] is True
  assert res['pocket_ca_used'] == res['pocket_residues'] > 0


@needs_data
def test_rigidly_moved_crystal_scores_the_same(tmp_path):
  import gemmi
  st = gemmi.read_structure(str(REPO / 'data' / 'rcsb' / f'{TARGET}.cif'))
  r, t = random_rigid(7)
  st[0].transform_pos_and_adp(gemmi.Transform(gemmi.Mat33(r.tolist()), gemmi.Vec3(*t)))
  st.setup_entities()
  moved = tmp_path / 'moved.cif'
  st.make_mmcif_document().write_file(str(moved))
  res = score_ligand.self_test(TARGET, pb_protein='predicted', path=moved)
  assert res['error'] is None
  assert res['ligand_rmsd'] < 0.01
  assert res['success'] is True


@needs_data
def test_unparseable_prediction_is_not_a_success(tmp_path):
  bad = tmp_path / 'bad_model.cif'
  bad.write_text('data_x\nthis is not an mmCIF atom table\n')
  res = score_ligand.score(TARGET, bad)
  assert res['success'] is False
  assert res['error']
  assert res['ligand_rmsd'] is None


@needs_data
def test_prediction_without_the_ligand_is_not_a_success(tmp_path):
  import gemmi
  st = gemmi.read_structure(str(REPO / 'data' / 'rcsb' / f'{TARGET}.cif'))
  ccd = score_ligand.reference(TARGET).ccd
  for chain in st[0]:
    for i in reversed(range(len(chain))):
      if chain[i].name == ccd:
        del chain[i]
  st.setup_entities()
  path = tmp_path / 'no_ligand.cif'
  st.make_mmcif_document().write_file(str(path))
  res = score_ligand.self_test(TARGET, path=path)
  assert res['success'] is False and res['error']


@needs_data
def test_pair_divergence_of_a_model_with_itself():
  models = sorted((REPO / 'results' / 'af3').glob(f'*/*/{TARGET}/af3_output/{TARGET}/seed-1_sample-0/*_model.cif'))
  if not models:
    pytest.skip('no AlphaFold3 output of 7U3J under results/af3/')
  res = score_ligand.pair_divergence(TARGET, models[0], models[0])
  assert res['error'] is None
  assert res['ligand_rmsd'] == 0.0 and res['ca_rmsd_global'] < 1e-6
