"""M6 and M7 helpers (analysis/cost.py)."""

import pandas as pd
import pytest

import cost


def test_price_lookup_by_region_and_fallback():
  prices = cost.load_prices()
  price, read_on, note = cost.price_for(prices, 'G', 'us-east4-c', 'standard')
  assert price == pytest.approx(0.850829188) and read_on and note == ''
  price, _, note = cost.price_for(prices, 'V6', 'us-east5-b', 'flex_start')
  assert price == pytest.approx(1.35) and 'europe-west4' in note
  price, _, note = cost.price_for(prices, 'G', 'europe-west4-a', 'flex_start')
  assert price is None and 'no l4 flex_start price' in note
  price, _, note = cost.price_for(prices, 'C', 'europe-west4-a', 'flex_start')
  assert price is None and note


def test_cost_and_amortisation():
  p = pd.DataFrame([{'session': 's', 'label': 'x', 'platform': 'V6', 'libtpu': '0.0.43.2', 'zone': 'europe-west4-a',
                     'target': 'T', 'num_tokens': 200, 'bucket': 256, 'repetition': 'r1', 'seeds': 1,
                     'exit_code': 0, 'wall_s': 3600.0, 'inference_s': 3000.0, 'compile_s': 400.0,
                     'num_diffusion_samples': 5}])
  c = cost.cost_table(p, cost.load_prices(), lambda bucket: 4).iloc[0]
  assert c.usd_per_structure_standard == pytest.approx(2.97)
  assert c.usd_per_structure_flex_start == pytest.approx(1.35)
  assert c.amortised_wall_per_structure_s == pytest.approx(3600 - 400 + 100)
  assert c.usd_per_sample_standard == pytest.approx(2.97 / 5)


def test_failed_process_has_no_cost():
  p = pd.DataFrame([{'session': 's', 'label': 'x', 'platform': 'G', 'libtpu': None, 'zone': 'us-east4-c',
                     'target': 'T', 'num_tokens': 900, 'bucket': 1024, 'repetition': 'r1', 'seeds': 1,
                     'exit_code': 1, 'wall_s': 50.0, 'inference_s': None, 'compile_s': None,
                     'num_diffusion_samples': 5}])
  c = cost.cost_table(p, cost.load_prices(), lambda bucket: 1).iloc[0]
  assert pd.isna(c.usd_per_structure_standard)


def test_subset_batch_sizes():
  batch = cost.batch_sizes('subset')
  assert [batch(b) for b in (256, 512, 768, 1024)] == [19, 19, 14, 8]
  assert cost.batch_sizes('5')(256) == 5
