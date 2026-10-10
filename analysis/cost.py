"""M6 (time and cost) and M7 (memory) of notes/analysis_plan.md, section 4.

    analysis/.venv/bin/python analysis/cost.py --samples results/analysis/<name>/samples.csv \\
        [--out results/analysis/<name>/cost] [--labels pilot] [--include REGEX] [--exclude REGEX] \\
        [--batch subset|N] [--all_settings]

Input: the table of analysis/collect.py, reduced to one row per process (one
run_alphafold call: a target and its seeds). Only runs at AlphaFold3's
defaults (10 recycles, 5 samples) unless --all_settings.

M6, per process:
  - wall, inference and compile seconds (run records; wall is the whole
    run_alphafold process: model load, featurisation, compilation, inference
    and output; VM setup and idle time are not included);
  - cost per structure = process wall / seeds in the process x hourly price, a
    structure being one (target, seed) prediction with its 5 samples; also per
    sample (divided by the number of diffusion samples);
  - prices: cloud/prices.csv rows for the platform (V5 v5e, V6 and V6h v6e,
    G, GT and Gh l4, C cpu; V6h and Gh are the arms of plan section 11) in
    the region of the run's zone, at standard (on-demand), spot and
    flex_start; when the region has no row, the europe-west4 row is used
    and the fallback noted; a missing price (no Flex-start for L4, CPU) is
    left empty, never guessed;
  - amortised variant: wall - compile + compile / batch, the batch being the
    number of same-bucket targets of targets/posebusters_subset.csv
    (--batch subset, the default) or a fixed N (--batch N). Warm reruns
    (repetition w) are reported as measured, not amortised again.
M7, per (platform, bucket): the compiled peak plus arguments and the
allocator peak reservation (TPU and GPU) against the device limit, max RSS
(all platforms, the only figure for CPU), and the largest bucket with a
successful run per platform. Also shown: the compiled peak alone (XLA's GPU
peak already includes the arguments, its TPU peak does not) and, for GPU,
the allocator peak in use (the CUDA allocator reports no reservation).

Outputs in --out: cost_per_process.csv, cost_summary.csv, memory.csv, prices_used.csv and report.md.
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
PRICE_KEY = {'V5': 'v5e', 'V6': 'v6e', 'V6h': 'v6e', 'G': 'l4', 'GT': 'l4', 'Gh': 'l4', 'C': 'cpu'}
PROVISIONING = ['standard', 'spot', 'flex_start']
FALLBACK_REGION = 'europe-west4'
GIB = 1024 ** 3


def load_prices():
  rows = list(csv.DictReader(open(REPO / 'cloud' / 'prices.csv')))
  out = {}
  for r in rows:
    price = float(r['usd_per_hour']) if r['usd_per_hour'] else None
    out[(r['platform'], r['region'], r['provisioning'])] = (price, r['read_on'])
  return out


def price_for(prices, platform, zone, prov):
  key = PRICE_KEY.get(platform)
  region = zone.rsplit('-', 1)[0] if isinstance(zone, str) and '-' in zone else FALLBACK_REGION
  if (key, region, prov) in prices and prices[(key, region, prov)][0] is not None:
    return prices[(key, region, prov)] + ('',)
  if (key, FALLBACK_REGION, prov) in prices:
    price, read_on = prices[(key, FALLBACK_REGION, prov)]
    if price is None:
      return None, read_on, f'no {key} {prov} price published'
    note = f'{key} {prov}: no row for {region}, the {FALLBACK_REGION} price is used' if region != FALLBACK_REGION else ''
    return price, read_on, note
  return None, None, f'no {key} {prov} row'


def processes(args):
  d = pd.read_csv(args.samples)
  if not args.all_settings:
    d = d[(d.num_recycles == 10) & (d.num_diffusion_samples == 5)]
  if args.labels:
    d = d[d.label.isin(args.labels.split(','))]
  if args.include:
    d = d[d.session.str.contains(args.include, regex=True)]
  if args.exclude:
    d = d[~d.session.str.contains(args.exclude, regex=True)]
  keep = ['session', 'label', 'platform', 'attention', 'libtpu', 'zone', 'machine_type', 'target', 'num_tokens',
          'bucket', 'repetition', 'cache', 'num_recycles', 'num_diffusion_samples', 'exit_code', 'failure', 'wall_s',
          'compile_s', 'compiled_peak_bytes', 'compiled_args_bytes', 'alloc_peak_reserved_bytes', 'alloc_peak_in_use_bytes',
          'device_bytes_limit', 'max_rss_bytes']
  g = d.groupby(['session', 'target'], sort=False)
  p = g[keep].first().reset_index(drop=True)
  p['seeds'] = g.seed.nunique().values
  p['inference_s'] = g.apply(lambda x: x.drop_duplicates('seed').inference_s.sum(min_count=1), include_groups=False).values
  return p


def batch_sizes(arg):
  if arg != 'subset':
    return lambda bucket: int(arg)
  man = pd.read_csv(REPO / 'inputs' / 'manifest.csv')
  subset = set(pd.read_csv(REPO / 'targets' / 'posebusters_subset.csv').pdb_id)
  counts = man[man.pdb_id.isin(subset)].groupby('bucket').size().to_dict()
  return lambda bucket: int(counts.get(bucket, 1))


def cost_table(p, prices, batch_of):
  rows = []
  for r in p.itertuples():
    ok = r.exit_code == 0 and not pd.isna(r.wall_s)
    per_struct = r.wall_s / r.seeds if ok and r.seeds else np.nan
    compile_s = 0.0 if pd.isna(r.compile_s) else float(r.compile_s)
    warm = r.repetition == 'w'
    batch = batch_of(r.bucket) if not pd.isna(r.bucket) else 1
    amort = per_struct if warm else (r.wall_s - compile_s + compile_s / batch) / r.seeds if ok else np.nan
    row = {'session': r.session, 'label': r.label, 'platform': r.platform, 'libtpu': r.libtpu, 'zone': r.zone,
           'target': r.target, 'num_tokens': r.num_tokens, 'bucket': r.bucket, 'repetition': r.repetition,
           'seeds': r.seeds, 'exit_code': r.exit_code, 'wall_s': r.wall_s, 'inference_s': r.inference_s,
           'compile_s': r.compile_s, 'batch': np.nan if warm else batch,
           'wall_per_structure_s': per_struct, 'amortised_wall_per_structure_s': amort}
    notes = set()
    for prov in PROVISIONING:
      price, read_on, note = price_for(prices, r.platform, r.zone, prov)
      if note:
        notes.add(note)
      row[f'usd_per_hour_{prov}'] = price
      row[f'price_read_on_{prov}'] = read_on
      row[f'usd_per_structure_{prov}'] = per_struct / 3600 * price if price is not None else np.nan
      row[f'usd_per_structure_amortised_{prov}'] = amort / 3600 * price if price is not None else np.nan
      row[f'usd_per_sample_{prov}'] = row[f'usd_per_structure_{prov}'] / r.num_diffusion_samples
    row['price_notes'] = ' | '.join(sorted(notes))
    rows.append(row)
  return pd.DataFrame(rows)


def summary_table(c):
  ok = c[c.exit_code == 0]
  rows = []
  for (plat, bucket, warm), g in ok.groupby(['platform', 'bucket', ok.repetition == 'w']):
    row = {'platform': plat, 'bucket': int(bucket), 'cache': 'warm' if warm else 'fresh', 'processes': len(g),
           'targets': g.target.nunique(), 'median_wall_s': g.wall_s.median(), 'max_wall_s': g.wall_s.max(),
           'median_inference_s': g.inference_s.median(), 'median_compile_s': g.compile_s.median()}
    for prov in PROVISIONING:
      row[f'median_usd_per_structure_{prov}'] = g[f'usd_per_structure_{prov}'].median()
      if not warm:
        row[f'median_usd_per_structure_amortised_{prov}'] = g[f'usd_per_structure_amortised_{prov}'].median()
    rows.append(row)
  return pd.DataFrame(rows)


def memory_table(p):
  rows = []
  for (plat, bucket), g in p.groupby(['platform', 'bucket']):
    ok = g[g.exit_code == 0]
    peak_args = (g.compiled_peak_bytes + g.compiled_args_bytes).max()
    limit = g.device_bytes_limit.max()
    reserved = g.alloc_peak_reserved_bytes.max()
    if plat in ('G', 'GT', 'Gh') and reserved == 0:
      reserved = np.nan  # the CUDA allocator preallocates its pool and reports no reservation
    rows.append({'platform': plat, 'bucket': int(bucket), 'processes': len(g), 'succeeded': len(ok),
                 'max_tokens_run': ok.num_tokens.max() if len(ok) else np.nan,
                 'compiled_peak_gib': g.compiled_peak_bytes.max() / GIB,
                 'compiled_peak_plus_args_gib': peak_args / GIB if not pd.isna(peak_args) else np.nan,
                 'alloc_peak_reserved_gib': reserved / GIB,
                 'alloc_peak_in_use_gib': g.alloc_peak_in_use_bytes.max() / GIB if plat != 'C' else np.nan,
                 'device_limit_gib': limit / GIB if not pd.isna(limit) else np.nan,
                 'peak_plus_args_fraction_of_limit': peak_args / limit if not pd.isna(limit) and limit else np.nan,
                 'max_rss_gib': g.max_rss_bytes.max() / GIB,
                 'failures': '; '.join(sorted(set(g[g.exit_code != 0].failure.dropna().astype(str))))[:200]})
  m = pd.DataFrame(rows)
  if len(m):
    largest = m[m.succeeded > 0].groupby('platform').bucket.max()
    m['largest_bucket_run_on_platform'] = m.platform.map(largest)
  return m


def fmt(x, nd=3):
  if x is None or (isinstance(x, float) and np.isnan(x)):
    return '-'
  return f'{x:.{nd}g}' if isinstance(x, float) else str(x)


def report(c, s, m, args, out):
  lines = ['# Time, cost and memory (M6, M7)', '',
           f'Source: `{pathlib.Path(args.samples).name}`; filters: labels={args.labels or "all"}, '
           f'include={args.include or "-"}, exclude={args.exclude or "-"}; amortisation batch: {args.batch}. '
           'Prices: cloud/prices.csv (read_on dates in prices_used.csv). Wall is the run_alphafold process; '
           'VM setup and idle time are not included. A structure is one (target, seed) with its 5 samples.', '',
           '## Cost per structure (median over processes, USD)', '',
           '| platform | bucket | cache | n | wall s | compile s | on-demand | Spot | Flex-start | '
           'on-demand amortised | Spot amortised | Flex-start amortised |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
  for r in s.itertuples():
    am = [getattr(r, f'median_usd_per_structure_amortised_{p}', np.nan) for p in PROVISIONING]
    lines.append(f'| {r.platform} | {r.bucket} | {r.cache} | {r.processes} | {fmt(r.median_wall_s, 4)} | '
                 f'{fmt(r.median_compile_s)} | ' + ' | '.join(fmt(getattr(r, f'median_usd_per_structure_{p}'))
                                                          for p in PROVISIONING)
                 + ' | ' + ' | '.join(fmt(x) for x in am) + ' |')
  notes = sorted(set(' | '.join(c.price_notes.dropna()).split(' | ')) - {''})
  if notes:
    lines += ['', 'Price notes: ' + '; '.join(notes) + '.']
  lines += ['', '## Memory per bucket (M7, GiB)', '',
            '| platform | bucket | ok/runs | max tokens | compiled peak | compiled peak + args | '
            'allocator peak reserved | allocator peak in use | device limit | max RSS | largest bucket run |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
  for r in m.itertuples():
    lines.append(f'| {r.platform} | {r.bucket} | {r.succeeded}/{r.processes} | {fmt(r.max_tokens_run)} | '
                 f'{fmt(r.compiled_peak_gib)} | {fmt(r.compiled_peak_plus_args_gib)} | {fmt(r.alloc_peak_reserved_gib)} | '
                 f'{fmt(r.alloc_peak_in_use_gib)} | '
                 f'{fmt(r.device_limit_gib)} | {fmt(r.max_rss_gib)} | {fmt(r.largest_bucket_run_on_platform)} |')
  lines += ['', 'Compiled peak + args is the plan\'s M7 figure. XLA reports the compiled peak differently per '
            'backend: on TPU it excludes the arguments (it is below the temporaries), on GPU it already includes '
            'them (peak = temporaries + arguments + output), so on GPU the plan\'s sum counts the arguments twice; '
            'the compiled peak alone is shown for that reason. On GPU the allocator preallocates its pool and '
            'reports no reservation (shown as -); its peak in use is shown instead. CPU: max RSS.']
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--out')
  ap.add_argument('--labels')
  ap.add_argument('--include')
  ap.add_argument('--exclude')
  ap.add_argument('--batch', default='subset', help="'subset' (same-bucket targets of the 60) or an integer")
  ap.add_argument('--all_settings', action='store_true', help='keep runs with non-default recycles or samples')
  args = ap.parse_args()
  out = pathlib.Path(args.out or pathlib.Path(args.samples).parent / 'cost')
  out.mkdir(parents=True, exist_ok=True)
  p = processes(args)
  if p.empty:
    raise SystemExit('No process matches these filters.')
  prices = load_prices()
  c = cost_table(p, prices, batch_sizes(args.batch))
  s = summary_table(c)
  m = memory_table(p)
  c.to_csv(out / 'cost_per_process.csv', index=False)
  s.to_csv(out / 'cost_summary.csv', index=False)
  m.to_csv(out / 'memory.csv', index=False)
  pd.DataFrame([{'platform': k[0], 'region': k[1], 'provisioning': k[2], 'usd_per_hour': v[0], 'read_on': v[1]}
                for k, v in sorted(prices.items())]).to_csv(out / 'prices_used.csv', index=False)
  report(c, s, m, args, out)
  print(f'>> {len(c)} processes; report: {out / "report.md"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
