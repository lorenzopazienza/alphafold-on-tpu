"""M2 to M5 contrasts and statistics of notes/analysis_plan.md, sections 4 to 6 and 11.

    analysis/.venv/bin/python analysis/contrasts.py --samples results/analysis/<name>/samples.csv \\
        [--out results/analysis/<name>/contrasts] [--labels pilot] [--include REGEX] [--exclude REGEX] \\
        [--condition platform|platform_libtpu] [--replicates 10000] [--seed 2026]

Input: the table of analysis/collect.py. Only runs at AlphaFold3's defaults
(10 recycles, 5 diffusion samples; plan section 2) are used. A condition is
the platform code (V5, V6, G, GT, C), or platform and libtpu version with
--condition platform_libtpu. Two rows for the same (condition, target, seed,
repetition, sample) from different sessions are an error, never resolved
silently: select sessions with --labels, --include or --exclude. The
precision-matched arms of plan section 11 (V6h, Gh: analysis/collect.py) are
conditions of their own; they enter only the section 11 contrasts below,
never RC, W, SD, HW, DU or Q3.

Per (condition, repetition, target, seed):
  M2 top-1: the sample with the highest ranking_score (ties: lowest sample
     index); overall top-1 across seeds is the (seed, sample) with the highest
     ranking_score, as AlphaFold3 ranks them;
  M3 margin: top-1 minus top-2 ranking_score within the seed.
Contrasts (plan section 5), paired on (target, seed) and (target, seed, sample):
  RC(P)    r1 against r2 on platform P;
  HW(P,Q)  r1 on P against r1 on Q; primary pairs V6-G, V5-V6, G-C, others secondary;
  SD(P)    seed s against seed s' (r1) on P, as a yardstick; its top-1 flip
           compares sample indices of different seeds, which are unrelated
           draws, so only its success flip and M4/M5 are informative;
  DU       GT against G (r1);
  ST(P,P') the same platform on two software stacks (libtpu versions; only
           with --condition platform_libtpu; not a plan contrast, used for the
           stack validation);
  W(P)     r1 against the warm rerun w (pilot check 2: expected bit-identical).
Section 11 contrasts (secondary), paired and computed in the same way:
  PR(P)    r1 on P against r1 on Ph, P in {V6, G}: the default matmul precision;
  HWh(V6,G) r1 on V6h against r1 on Gh, and HWh(G,C) r1 on Gh against r1 on C:
           the platform difference with float32 matmuls matched.
Statistics per contrast: flip rate of top-1, flip rate of top-1 success,
flips that matter (top-1 changes and the two chosen structures differ in
success), median and 90th percentile of M4 (ligand RMSD after pocket
superposition, protein C-alpha RMSD after global superposition) and M5
(absolute differences of ranking_score, mean pLDDT, iptm), and the fraction
of sample pairs with bit-identical coordinates.
Uncertainty: 95% percentile intervals from a paired bootstrap over targets
(resample targets with replacement, --replicates 10000, --seed 2026), and
for each hardware pair HW(P,Q) - RC(P) of each flip rate on the same
resampled targets (also HW - RC(Q), labelled secondary). For HWh(P,Q) the
same, against RC(P) and RC(Q) of the default-precision arms (the arms with
"h" run r1 only), for section 11's reading rule.
Q3: for n = 1..N seeds, the overall top-1 over the first n seeds, its success
rate per condition (on its own targets) and the pairwise differences on each
pair's common targets, with bootstrap intervals; the answer is the smallest n
where every interval lies within +-2 percentage points, or "not reached".
Pilot checks (section 6): exit codes, W bit-identity, and the top-1 success
on V6 against the 50% sanity threshold.

Outputs in --out: contrasts.csv, hw_minus_rc.csv, pairs_seed.csv,
pairs_sample.csv, q3.csv, checks.csv, excluded.csv (section 8 counts) and report.md.
"""

import argparse
import functools
import itertools
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'harness'))

PRIMARY = [('V6', 'G'), ('V5', 'V6'), ('G', 'C')]
ORDER = ['V5', 'V6', 'G', 'GT', 'C']
# Section 11: (kind, condition A, condition B), A the first-named side (M4 reference).
PRECISION_CONTRASTS = [('PR', 'V6', 'V6h'), ('PR', 'G', 'Gh'), ('HWh', 'V6h', 'Gh'), ('HWh', 'Gh', 'C')]
FLIP_STATS = ['flip_top1', 'flip_success', 'flip_matters']
SAMPLE_STATS = ['m4_ligand_rmsd', 'm4_ca_rmsd_global', 'm5_ranking_score', 'm5_mean_plddt', 'm5_iptm']
SANITY_THRESHOLD = 0.5
Q3_BAND = 0.02


def rank(condition):
  base = condition.split('_')[0]
  return (ORDER.index(base) if base in ORDER else len(ORDER), condition)


def arm(condition):
  """The condition's code without the libtpu suffix of --condition platform_libtpu: V6h_0.0.43.2 -> V6h."""
  return condition.split('_')[0]


def precision_base(condition):
  """For a condition with a non-default matmul precision (V6h, Gh, V6[float32]), its section 3 code; else None."""
  a = arm(condition)
  if '[' in a:
    return a.split('[')[0]
  if a.endswith('h') and a[:-1] in ORDER:
    return a[:-1]
  return None


def default_precision(condition):
  return precision_base(condition) is None


def unh(condition):
  """V6h -> V6, Gh_none -> G_none; other conditions unchanged."""
  base = precision_base(condition)
  return condition if base is None else base + condition[len(arm(condition)):]


@functools.lru_cache(maxsize=None)
def coords_text(path):
  from compare_samples import read_coords
  return tuple(read_coords(REPO / path)[2])


@functools.lru_cache(maxsize=None)
def divergence(target, path_a, path_b):
  import score_ligand
  return score_ligand.pair_divergence(target, REPO / path_a, REPO / path_b)


def load(args):
  d = pd.read_csv(args.samples)
  d = d[(d.num_recycles == 10) & (d.num_diffusion_samples == 5)]
  if args.labels:
    d = d[d.label.isin(args.labels.split(','))]
  if args.include:
    d = d[d.session.str.contains(args.include, regex=True)]
  if args.exclude:
    d = d[~d.session.str.contains(args.exclude, regex=True)]
  d = d.copy()
  d['condition'] = d.platform if args.condition == 'platform' else d.platform + '_' + d.libtpu.fillna('none').astype(str)
  ok = d[d['sample'].notna()].copy()
  ok['sample'] = ok['sample'].astype(int)
  dup = ok.duplicated(['condition', 'target', 'seed', 'repetition', 'sample'], keep=False)
  if dup.any():
    clash = ok[dup].groupby(['condition', 'target', 'seed', 'repetition']).session.unique()
    raise SystemExit('Same (condition, target, seed, repetition) in several sessions; select with '
                     '--labels/--include/--exclude:\n' + clash.head(10).to_string())
  return d, ok


def seed_units(ok):
  """{(condition, repetition, target, seed): DataFrame of its samples}."""
  return {k: g.set_index('sample').sort_index() for k, g in ok.groupby(['condition', 'repetition', 'target', 'seed'])}


def top(g):
  order = g.sort_values(['ranking_score'], ascending=False, kind='mergesort')
  first = order.index[0]
  margin = float(order.ranking_score.iloc[0] - order.ranking_score.iloc[1]) if len(order) > 1 else np.nan
  return first, bool(order.m1_success.iloc[0] == True), margin  # noqa: E712 - NaN counts as not a success


def pair_rows(name, kind, a_key, b_key, units, primary, contrast_pairs, seed_rows, sample_rows):
  """Adds one (target, seed) pair and its sample pairs to the output lists."""
  ga, gb = units[a_key], units[b_key]
  ta, sa, ma = top(ga)
  tb, sb, mb = top(gb)
  target = a_key[2]
  seed_rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'target': target, 'seed_a': a_key[3],
                    'seed_b': b_key[3], 'top1_a': ta, 'top1_b': tb, 'success_a': sa, 'success_b': sb,
                    'margin_a': ma, 'margin_b': mb, 'flip_top1': ta != tb, 'flip_success': sa != sb,
                    'flip_matters': ta != tb and sa != sb})
  for k in sorted(set(ga.index) & set(gb.index)):
    ra, rb = ga.loc[k], gb.loc[k]
    ident = isinstance(ra.model_path, str) and isinstance(rb.model_path, str) and \
        coords_text(ra.model_path) == coords_text(rb.model_path)
    if ident:
      m4 = {'ligand_rmsd': 0.0, 'ca_rmsd_global': 0.0, 'error': None}
    elif isinstance(ra.model_path, str) and isinstance(rb.model_path, str):
      m4 = divergence(target, ra.model_path, rb.model_path)
    else:
      m4 = {'ligand_rmsd': None, 'ca_rmsd_global': None, 'error': 'model missing'}
    sample_rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'target': target, 'seed_a': a_key[3],
                        'seed_b': b_key[3], 'sample': k, 'bit_identical': ident,
                        'm4_ligand_rmsd': m4['ligand_rmsd'], 'm4_ca_rmsd_global': m4['ca_rmsd_global'],
                        'm4_error': m4['error'],
                        'm5_ranking_score': abs(ra.ranking_score - rb.ranking_score),
                        'm5_mean_plddt': abs(ra.mean_plddt - rb.mean_plddt),
                        'm5_iptm': abs(ra.iptm - rb.iptm)})
  contrast_pairs.setdefault(name, (kind, primary))


def build_pairs(ok):
  units = seed_units(ok)
  all_conds = sorted(ok.condition.unique())
  conds = [c for c in all_conds if default_precision(c)]
  reps = {c: sorted(ok[ok.condition == c].repetition.unique()) for c in all_conds}
  seed_rows, sample_rows, contrasts, excluded = [], [], {}, []

  def keys(c, rep):
    return {(k[2], k[3]): k for k in units if k[0] == c and k[1] == rep}

  def add(name, kind, ka, kb, primary):
    common = sorted(set(ka) & set(kb))
    excluded.append({'contrast': name, 'pairs_used': len(common),
                     'excluded_only_a': len(set(ka) - set(kb)), 'excluded_only_b': len(set(kb) - set(ka))})
    for ts in common:
      pair_rows(name, kind, ka[ts], kb[ts], units, primary, contrasts, seed_rows, sample_rows)

  for c in conds:
    if 'r1' in reps[c] and 'r2' in reps[c]:
      add(f'RC({c})', 'RC', keys(c, 'r1'), keys(c, 'r2'), True)
    if 'r1' in reps[c] and 'w' in reps[c]:
      add(f'W({c})', 'W', keys(c, 'r1'), keys(c, 'w'), True)
    if 'r1' in reps[c]:
      r1 = keys(c, 'r1')
      by_target = {}
      for (t, s), k in r1.items():
        by_target.setdefault(t, []).append(s)
      ka, kb = {}, {}
      for t, seeds in by_target.items():
        for s1, s2 in itertools.combinations(sorted(seeds), 2):
          ka[(t, (s1, s2))], kb[(t, (s1, s2))] = r1[(t, s1)], r1[(t, s2)]
      if ka:
        add(f'SD({c})', 'SD', ka, kb, True)
  for p, q in itertools.combinations(sorted(conds, key=rank), 2):
    if 'r1' in reps[p] and 'r1' in reps[q]:
      base = lambda c: c.split('_')[0]  # noqa: E731
      primary = (base(p), base(q)) in PRIMARY or (base(q), base(p)) in PRIMARY
      a, b = (p, q) if (base(p), base(q)) in PRIMARY or not primary else (q, p)
      if base(a) == base(b):
        name, kind = f'ST({a},{b})', 'ST'
      elif {base(a), base(b)} == {'G', 'GT'}:
        name, kind = 'DU', 'DU'
      else:
        name, kind = f'HW({a},{b})', 'HW'
      add(name, kind, keys(a, 'r1'), keys(b, 'r1'), primary or kind == 'DU')
  # Section 11, secondary: the precision-matched arms enter only these.
  for kind, code_a, code_b in PRECISION_CONTRASTS:
    for a in all_conds:
      for b in all_conds:
        if arm(a) != code_a or arm(b) != code_b or 'r1' not in reps[a] or 'r1' not in reps[b]:
          continue
        if kind == 'PR' and unh(b) != a:   # same platform (and libtpu, with --condition platform_libtpu)
          continue
        name = f'PR({a})' if kind == 'PR' else f'HWh({unh(a)},{unh(b)})'
        add(name, kind, keys(a, 'r1'), keys(b, 'r1'), False)
  return pd.DataFrame(seed_rows), pd.DataFrame(sample_rows), contrasts, pd.DataFrame(excluded)


def boot_ci(groups, stat, rng, reps):
  """95% percentile interval of stat over targets resampled with replacement; groups: list of arrays."""
  groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
  if not groups:
    return np.nan, np.nan
  idx = rng.integers(0, len(groups), size=(reps, len(groups)))
  vals = np.array([stat(np.concatenate([groups[i] for i in row])) for row in idx])
  vals = vals[~np.isnan(vals)]
  return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def p90(x):
  return float(np.percentile(x, 90)) if len(x) else np.nan


def med(x):
  return float(np.median(x)) if len(x) else np.nan


def mean(x):
  return float(np.mean(x)) if len(x) else np.nan


def summarize(seed_pairs, sample_pairs, contrasts, args):
  rows = []
  for name, (kind, primary) in contrasts.items():
    sp = seed_pairs[seed_pairs.contrast == name]
    sm = sample_pairs[sample_pairs.contrast == name]
    targets = sorted(sp.target.unique())
    for st in FLIP_STATS:
      if kind == 'SD' and st != 'flip_success':
        note = 'sample indices of different seeds are unrelated draws'
      else:
        note = ''
      groups = [sp[sp.target == t][st].astype(float).values for t in targets]
      lo, hi = boot_ci(groups, mean, np.random.default_rng(args.seed), args.replicates)
      rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'statistic': st, 'n': len(sp),
                   'n_targets': len(targets), 'estimate': mean(sp[st].astype(float).values),
                   'ci_low': lo, 'ci_high': hi, 'note': note})
    for st in SAMPLE_STATS:
      vals = sm[st].dropna().astype(float)
      for fn, fname in ((med, 'median'), (p90, 'p90')):
        groups = [sm[(sm.target == t)][st].dropna().astype(float).values for t in targets]
        lo, hi = boot_ci(groups, fn, np.random.default_rng(args.seed), args.replicates)
        rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'statistic': f'{st}_{fname}',
                     'n': len(vals), 'n_targets': len(targets), 'estimate': fn(vals.values),
                     'ci_low': lo, 'ci_high': hi, 'note': ''})
    groups = [sm[sm.target == t].bit_identical.astype(float).values for t in targets]
    lo, hi = boot_ci(groups, mean, np.random.default_rng(args.seed), args.replicates)
    rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'statistic': 'bit_identical_fraction',
                 'n': len(sm), 'n_targets': len(targets), 'estimate': mean(sm.bit_identical.astype(float).values),
                 'ci_low': lo, 'ci_high': hi, 'note': ''})
    m4err = sm.m4_error.dropna()
    if len(m4err):
      rows.append({'contrast': name, 'kind': kind, 'primary': primary, 'statistic': 'm4_errors', 'n': len(m4err),
                   'n_targets': len(targets), 'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                   'note': '; '.join(sorted(set(m4err)))[:300]})
  return pd.DataFrame(rows)


def hw_minus_rc(seed_pairs, contrasts, args):
  rows = []
  for name, (kind, primary) in contrasts.items():
    if kind not in ('HW', 'HWh'):
      continue
    # HWh(P,Q) names the default-precision codes, so RC(P) and RC(Q) are those of the default arms.
    p, q = re.match(r'HWh?\((.+),(.+)\)', name).groups()
    for which, rc_platform in (('RC(P)', p), ('RC(Q), secondary', q)):
      rc = f'RC({rc_platform})'
      if rc not in contrasts:
        continue
      hw_sp, rc_sp = seed_pairs[seed_pairs.contrast == name], seed_pairs[seed_pairs.contrast == rc]
      targets = sorted(set(hw_sp.target) & set(rc_sp.target))
      for st in FLIP_STATS:
        hw_g = [hw_sp[hw_sp.target == t][st].astype(float).values for t in targets]
        rc_g = [rc_sp[rc_sp.target == t][st].astype(float).values for t in targets]
        est = mean(np.concatenate(hw_g)) - mean(np.concatenate(rc_g)) if targets else np.nan
        rng = np.random.default_rng(args.seed)
        diffs = []
        if targets:
          for row in rng.integers(0, len(targets), size=(args.replicates, len(targets))):
            diffs.append(mean(np.concatenate([hw_g[i] for i in row])) - mean(np.concatenate([rc_g[i] for i in row])))
        diffs = np.array(diffs)
        rows.append({'hardware': name, 'recompilation': rc, 'reference': which, 'primary': primary,
                     'statistic': st, 'n_targets': len(targets), 'estimate': est,
                     'ci_low': float(np.percentile(diffs, 2.5)) if len(diffs) else np.nan,
                     'ci_high': float(np.percentile(diffs, 97.5)) if len(diffs) else np.nan})
  return pd.DataFrame(rows)


def q3(ok, args):
  r1 = ok[(ok.repetition == 'r1') & ok.condition.map(default_precision)]
  conds = sorted(r1.condition.unique(), key=rank)
  seeds = sorted(r1.seed.unique())
  rows, answer = [], None
  for n in range(1, len(seeds) + 1):
    use = seeds[:n]
    succ = {}
    for c in conds:
      for t, g in r1[(r1.condition == c) & (r1.seed.isin(use))].groupby('target'):
        if sorted(g.seed.unique()) != use:
          continue
        best = g.sort_values(['ranking_score'], ascending=False, kind='mergesort').iloc[0]
        succ[(c, t)] = float(best.m1_success == True)  # noqa: E712
    if not succ:
      continue
    for c in conds:
      own = [v for (c2, t), v in succ.items() if c2 == c]
      if own:
        rows.append({'n_seeds': n, 'condition': c, 'statistic': 'success_rate', 'n_targets': len(own),
                     'estimate': mean(own), 'ci_low': np.nan, 'ci_high': np.nan})
    within, compared = True, False
    for a, b in itertools.combinations(conds, 2):
      targets = sorted(t for c, t in succ if c == a and (b, t) in succ)  # each pair on its common targets
      if not targets:
        continue
      compared = True
      diff = np.array([succ[(a, t)] - succ[(b, t)] for t in targets])
      lo, hi = boot_ci([[x] for x in diff], mean, np.random.default_rng(args.seed), args.replicates)
      within &= (lo >= -Q3_BAND) and (hi <= Q3_BAND)
      rows.append({'n_seeds': n, 'condition': f'{a}-{b}', 'statistic': 'success_difference',
                   'n_targets': len(targets), 'estimate': float(diff.mean()), 'ci_low': lo, 'ci_high': hi})
    if compared:
      rows.append({'n_seeds': n, 'condition': 'all conditions', 'statistic': 'spread_max_minus_min',
                   'n_targets': np.nan,
                   'estimate': max(r['estimate'] for r in rows if r['n_seeds'] == n and r['statistic'] == 'success_rate')
                   - min(r['estimate'] for r in rows if r['n_seeds'] == n and r['statistic'] == 'success_rate'),
                   'ci_low': np.nan, 'ci_high': np.nan})
      if within and answer is None:
        answer = n
  return pd.DataFrame(rows), answer


def checks(d, ok, sample_pairs, args):
  rows = []
  for c, g in d.groupby('condition'):
    runs = g.drop_duplicates(['session', 'target'])
    failed = runs[runs.exit_code != 0]
    rows.append({'check': '1 exit codes', 'condition': c, 'value': f'{len(runs) - len(failed)}/{len(runs)} processes exit 0',
                 'detail': '; '.join(f'{r.target} {r.repetition} seed {r.seed}: {r.failure}' for r in failed.itertuples())[:300]})
  for c in sorted(ok.condition.unique()):
    w = sample_pairs[sample_pairs.contrast == f'W({c})']
    if len(w):
      rows.append({'check': '2 warm rerun bit-identical to r1', 'condition': c,
                   'value': f'{int(w.bit_identical.sum())}/{len(w)} samples', 'detail': 'pass' if w.bit_identical.all() else 'FAIL'})
  v6 = [c for c in ok.condition.unique() if arm(c) == 'V6']
  for c in v6:
    r1 = ok[(ok.condition == c) & (ok.repetition == 'r1')]
    best = r1.sort_values('ranking_score', ascending=False, kind='mergesort').groupby('target').head(1)
    rate = float((best.m1_success == True).mean()) if len(best) else np.nan  # noqa: E712
    rows.append({'check': '3 success-rate sanity (overall top-1, r1)', 'condition': c,
                 'value': f'{rate:.0%} of {len(best)} targets',
                 'detail': 'pass' if rate >= SANITY_THRESHOLD else 'BELOW 50%: stop and investigate the inputs'})
    per_seed = r1.sort_values('ranking_score', ascending=False, kind='mergesort').groupby(['target', 'seed']).head(1)
    rate = float((per_seed.m1_success == True).mean()) if len(per_seed) else np.nan  # noqa: E712
    rows.append({'check': '3 success-rate sanity (top-1 per seed, r1; secondary)', 'condition': c,
                 'value': f'{rate:.0%} of {len(per_seed)} (target, seed)',
                 'detail': 'pass' if rate >= SANITY_THRESHOLD else 'BELOW 50%'})
  return pd.DataFrame(rows)


def fmt(x, pct=False):
  if x is None or (isinstance(x, float) and np.isnan(x)):
    return '-'
  return f'{100 * x:.1f}%' if pct else f'{x:.3g}'


def report(summary, hwrc, q3rows, q3answer, chk, excluded, args, out):
  lines = ['# Contrasts report', '',
           f'Source: `{pathlib.Path(args.samples).name}`; filters: labels={args.labels or "all"}, '
           f'include={args.include or "-"}, exclude={args.exclude or "-"}; condition={args.condition}; '
           f'bootstrap: {args.replicates} replicates over targets, seed {args.seed}. '
           'Definitions: analysis/contrasts.py and notes/analysis_plan.md sections 4, 5 and 11 '
           '(PR and HWh: section 11, secondary).', '',
           '## Pilot checks (section 6)', '', '| check | condition | value | detail |', '|---|---|---|---|']
  lines += [f'| {r.check} | {r.condition} | {r.value} | {r.detail} |' for r in chk.itertuples()]
  lines += ['', '## Flip rates', '', '| contrast | primary | statistic | n pairs | targets | estimate | 95% CI |',
            '|---|---|---|---|---|---|---|']
  for r in summary[summary.statistic.isin(FLIP_STATS + ['bit_identical_fraction'])].itertuples():
    lines.append(f'| {r.contrast} | {"yes" if r.primary else "no"} | {r.statistic} | {r.n} | {r.n_targets} | '
                 f'{fmt(r.estimate, True)} | {fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |'
                 + (f' ({r.note})' if r.note else ''))
  lines += ['', '## M4 and M5 (median and 90th percentile)', '',
            '| contrast | statistic | n | estimate | 95% CI |', '|---|---|---|---|---|']
  for r in summary[summary.statistic.str.startswith(('m4_', 'm5_'))].itertuples():
    lines.append(f'| {r.contrast} | {r.statistic} | {r.n} | {fmt(r.estimate)} | '
                 f'{fmt(r.ci_low)} to {fmt(r.ci_high)} |' + (f' {r.note}' if r.note else ''))
  lines += ['', '## Hardware minus recompilation', '']
  if len(hwrc):
    lines += ['| hardware | recompilation | statistic | estimate | 95% CI |', '|---|---|---|---|---|']
    lines += [f'| {r.hardware} | {r.recompilation} ({r.reference}) | {r.statistic} | {fmt(r.estimate, True)} | '
              f'{fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |' for r in hwrc.itertuples()]
  else:
    lines.append('No hardware contrast has a recompilation contrast (r2) to subtract.')
  lines += ['', '## Q3: seeds needed', '']
  if len(q3rows):
    lines += ['| seeds | condition | statistic | estimate | 95% CI |', '|---|---|---|---|---|']
    lines += [f'| {r.n_seeds} | {r.condition} | {r.statistic} | {fmt(r.estimate, True)} | '
              f'{fmt(r.ci_low, True)} to {fmt(r.ci_high, True)} |' for r in q3rows.itertuples()]
    lines += ['', f'Answer: {f"{q3answer} seed(s)" if q3answer else "not reached"} '
                  f'(every pairwise difference within +-{100 * Q3_BAND:.0f} percentage points).']
  else:
    lines.append('No condition has r1 runs for the same targets and seeds.')
  lines += ['', '## Pairs excluded (section 8)', '', '| contrast | used | only in A | only in B |', '|---|---|---|---|']
  lines += [f'| {r.contrast} | {r.pairs_used} | {r.excluded_only_a} | {r.excluded_only_b} |' for r in excluded.itertuples()]
  (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--samples', required=True)
  ap.add_argument('--out')
  ap.add_argument('--labels')
  ap.add_argument('--include')
  ap.add_argument('--exclude')
  ap.add_argument('--condition', choices=('platform', 'platform_libtpu'), default='platform')
  ap.add_argument('--replicates', type=int, default=10000)
  ap.add_argument('--seed', type=int, default=2026)
  args = ap.parse_args()
  out = pathlib.Path(args.out or pathlib.Path(args.samples).parent / 'contrasts')
  out.mkdir(parents=True, exist_ok=True)
  d, ok = load(args)
  seed_pairs, sample_pairs, contrasts, excluded = build_pairs(ok)
  if seed_pairs.empty:
    raise SystemExit('No contrast has pairs with these filters.')
  summary = summarize(seed_pairs, sample_pairs, contrasts, args)
  hwrc = hw_minus_rc(seed_pairs, contrasts, args)
  q3rows, q3answer = q3(ok, args)
  chk = checks(d, ok, sample_pairs, args)
  summary.to_csv(out / 'contrasts.csv', index=False)
  hwrc.to_csv(out / 'hw_minus_rc.csv', index=False)
  seed_pairs.to_csv(out / 'pairs_seed.csv', index=False)
  sample_pairs.to_csv(out / 'pairs_sample.csv', index=False)
  q3rows.to_csv(out / 'q3.csv', index=False)
  chk.to_csv(out / 'checks.csv', index=False)
  excluded.to_csv(out / 'excluded.csv', index=False)
  report(summary, hwrc, q3rows, q3answer, chk, excluded, args, out)
  print(f'>> {len(contrasts)} contrasts, {len(seed_pairs)} (target, seed) pairs, {len(sample_pairs)} sample pairs; '
        f'report: {out / "report.md"}', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
