"""Generate a read-only, matched-scenario report from a benchmark checkpoint.

Usage (from server):
    python create_report.py
    python create_report.py --checkpoint PATH
Requires numpy, pandas and matplotlib (already used by the benchmark).
"""
import argparse
from contextlib import closing
from datetime import datetime
import html
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALGORITHMS = ['Standard JPS', 'Optimized Fast Weighted SA-JPS']
SHORT = ['Standard JPS', 'Optimized FWSAJPS']
METRICS = ['Time ms', 'Path length', 'Turns', 'Total turning angle deg',
           'Expanded nodes', 'Optimality gap percent', 'Waypoints',
           'Generated successors', 'Queue pushes', 'Jump calls']
LABELS = dict(zip(METRICS, ['Runtime (ms)', 'Path length', 'Turns', 'Total turning angle (deg)',
                          'Expanded nodes', 'Gap to dataset reference (%)', 'Waypoints',
                          'Generated successors', 'Queue pushes', 'Jump calls']))
RANGES = ['R1: 0 <= L < 100', 'R2: 100 <= L < 250', 'R3: 250 <= L < 500',
          'R4: 500 <= L < 1000', 'R5: L >= 1000']
COLORS = ['#27659c', '#db7029']


def read_snapshot(path):
    """The first SELECT establishes one snapshot; never create or modify the source."""
    fields = ['Map width', 'Map height', 'Reference path length', 'Execution workers', 'Success', 'Start x', 'Start y', 'Goal x', 'Goal y',
              'Turn penalty weight', 'Turn count weight', 'Heuristic weight'] + METRICS
    rows = []
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        signature = db.execute('SELECT signature FROM metadata').fetchone()
        snapshot_time = datetime.now().astimezone().isoformat()
        for name, scenario, algorithm, payload in db.execute(
                'SELECT map_name, scenario, algorithm, metrics FROM searches WHERE algorithm IN (?, ?)', ALGORITHMS):
            record = json.loads(payload)
            row = {field: record.get(field, np.nan) for field in fields}
            row.update(map=name, scenario=scenario, algorithm=algorithm)
            rows.append(row)
        db.rollback()
    frame = pd.DataFrame(rows, columns=['map', 'scenario', 'algorithm'] + fields)
    if frame.empty:
        raise ValueError('No saved results for the two comparison algorithms.')
    if frame['Success'].isna().any() or not frame['Success'].isin([True, False]).all():
        raise ValueError('Missing or invalid success flags in checkpoint.')
    frame['Success'] = frame['Success'].astype(bool)
    for column in METRICS + ['Map width', 'Map height', 'Reference path length']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['Execution workers'] = frame['Execution workers'].map(
        lambda value: str(int(value)) if pd.notna(value) and float(value) >= 1 else 'unknown')
    return frame, {'snapshot_time': snapshot_time, 'checkpoint': str(path.resolve()),
                   'checkpoint_signature': json.loads(signature[0]) if signature else None}


def match_pairs(frame):
    a, b = [frame[frame.algorithm == name] for name in ALGORITHMS]
    pairs = a.merge(b, on=['map', 'scenario'], suffixes=('_jps', '_opt'), validate='one_to_one')
    initial_count = len(pairs)
    valid = np.ones(len(pairs), dtype=bool)
    for field in ['Reference path length', 'Map width', 'Map height']:
        x, y = pairs[field + '_jps'], pairs[field + '_opt']
        valid &= np.isfinite(x) & np.isfinite(y) & (x == y)
    valid &= (pairs['Reference path length_jps'] >= 0)
    valid &= (pairs['Map width_jps'] > 0) & (pairs['Map height_jps'] > 0)
    invalid = int((~valid).sum())
    pairs = pairs[valid].copy()
    pairs['Map size'] = pairs['Map width_jps'].astype(int).astype(str) + ' x ' + pairs['Map height_jps'].astype(int).astype(str)
    pairs['Range'] = pd.cut(pairs['Reference path length_jps'],
                            bins=[0, 100, 250, 500, 1000, np.inf], labels=RANGES, right=False)
    pairs['Both successful'] = pairs['Success_jps'] & pairs['Success_opt']
    return pairs, {'matched_before_filters': initial_count, 'invalid_pair_metadata_excluded': invalid,
                   'included_pairs': len(pairs),
                   'both_successful_pairs': int(pairs['Both successful'].sum())}


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def quantiles(values):
    values = finite(values)
    return (len(values), *np.quantile(values, [.5, .25, .75])) if len(values) else (0, np.nan, np.nan, np.nan)


def group_sets(pairs):
    yield 'Overall', 'All', pairs
    for size in sorted(pairs['Map size'].unique(), key=lambda x: tuple(map(int, x.split(' x ')))):
        yield 'Map size', size, pairs[pairs['Map size'] == size]
    for label in RANGES:
        yield 'Reference length', label, pairs[pairs['Range'] == label]


def build_tables(frame, pairs):
    coverage = []
    for name in ALGORITHMS:
        saved = frame[frame.algorithm == name]
        coverage.append({'Algorithm': name, 'Saved runs': len(saved),
                         'Successful saved runs': int(saved.Success.sum()),
                         'Saved success rate (%)': 100 * saved.Success.mean()})
    counts, summaries, changes = [], [], []
    for grouping, group, part in group_sets(pairs):
        count = {'Grouping': grouping, 'Group': group, 'Matched scenarios': len(part),
                 'Maps represented': part['map'].nunique(), 'Both successful': int(part['Both successful'].sum())}
        for suffix, name in zip(['jps', 'opt'], SHORT):
            count[name + ' successes'] = int(part['Success_' + suffix].sum())
            count[name + ' success rate (%)'] = 100 * part['Success_' + suffix].mean()
        counts.append(count)
        successful = part[part['Both successful']]
        for metric in METRICS:
            x = successful[metric + '_jps'].to_numpy(dtype=float)
            y = successful[metric + '_opt'].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            meds = []
            for name, values in zip(ALGORITHMS, [x, y]):
                n, med, q1, q3 = quantiles(values)
                meds.append(med)
                summaries.append({'Grouping': grouping, 'Group': group, 'Algorithm': name,
                                  'Metric': LABELS[metric], 'N': n, 'Median': med, 'Q1': q1, 'Q3': q3})
            # Restrict relative changes to physical nonnegative metrics, not signed reference gaps.
            if metric == 'Optimality gap percent':
                continue
            positive = x > 0
            relative = 100 * (y[positive] - x[positive]) / x[positive]
            n, med, q1, q3 = quantiles(relative)
            changes.append({'Grouping': grouping, 'Group': group, 'Metric': LABELS[metric],
                            'Metric pairs': len(x), 'Positive baseline pairs': n,
                            'Zero or negative baseline pairs excluded': int((~positive).sum()),
                            'Difference between medians (%)': 100 * (meds[1] / meds[0] - 1) if meds[0] > 0 else np.nan,
                            'Per-scenario change median (%)': med, 'Per-scenario change Q1 (%)': q1,
                            'Per-scenario change Q3 (%)': q3})
    return {'saved_coverage': pd.DataFrame(coverage), 'matched_counts': pd.DataFrame(counts),
            'summary_numeric': pd.DataFrame(summaries), 'relative_changes': pd.DataFrame(changes)}


def holm_adjust(pvalues):
    """Holm step-down correction, preserving missing (untestable) values."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return adjusted


def difference_statistics(differences, primary=False):
    from scipy.stats import binomtest, rankdata, wilcoxon
    import warnings
    d = np.round(finite(differences), 9)
    nz = d[d != 0]
    positive, negative = int((d > 0).sum()), int((d < 0).sum())
    result = {'N units': len(d), 'Nonzero units': len(nz), 'Tied units': int((d == 0).sum()),
              'Optimized lower units': negative, 'Optimized higher units': positive,
              'Median paired difference': float(np.median(d)) if len(d) else np.nan,
              'Paired sign effect': (positive-negative)/len(nz) if len(nz) else (0.0 if len(d) else np.nan),
              'Rank-biserial effect': np.nan, 'p-value': np.nan,
              'Test': 'Exact two-sided sign test' if primary else 'Two-sided Wilcoxon (normal approximation)',
              'Status': 'No eligible data'}
    if not len(d): return result
    if not len(nz):
        result.update({'Rank-biserial effect': 0.0, 'p-value': 1.0, 'Status': 'All differences zero'})
        return result
    ranks = rankdata(abs(nz), method='average')
    result['Rank-biserial effect'] = float(np.sum(ranks * np.sign(nz)) / ranks.sum())
    if len(nz) < 2:
        result['Status'] = 'Fewer than two nonzero units; no inference'
        return result
    if primary:
        result['p-value'] = float(binomtest(positive, len(nz), .5, alternative='two-sided').pvalue)
    else:
        # mode is supported by the project's scipy>=1.7 requirement.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result['p-value'] = float(wilcoxon(nz, zero_method='wilcox', correction=True,
                                             alternative='two-sided', mode='approx').pvalue)
    result['Status'] = 'OK' if primary or len(nz) >= 10 else 'Small sample; normal approximation unreliable'
    return result


def statistical_comparisons(pairs):
    rows = []
    for grouping, group, part in group_sets(pairs):
        part = part[part['Both successful']]
        for metric in METRICS:
            x, y = part[metric+'_jps'], part[metric+'_opt']
            valid = np.isfinite(x) & np.isfinite(y)
            differences = np.round((y[valid]-x[valid]).to_numpy(), 9)
            per_map = pd.DataFrame({'map': part.loc[valid, 'map'].to_numpy(), 'd': differences}).groupby('map')['d'].median()
            for unit, values, primary in [('Map median difference (primary)', per_map.to_numpy(), True),
                                          ('Scenario difference (exploratory)', differences, False)]:
                rows.append({'Grouping': grouping, 'Group': group, 'Metric': LABELS[metric],
                             'Unit': unit, 'Matched metric scenarios': int(valid.sum()),
                             'Maps': len(per_map), **difference_statistics(values, primary)})
    table = pd.DataFrame(rows)
    table['Holm-adjusted p-value'] = np.nan
    # One family per analysis unit, across ALL reported metrics and groups.
    for unit, indices in table.groupby('Unit').groups.items():
        table.loc[indices, 'Holm-adjusted p-value'] = holm_adjust(table.loc[indices, 'p-value'])
    table['Significant (Holm < 0.05)'] = table['Holm-adjusted p-value'].map(
        lambda p: 'Not tested' if not np.isfinite(p) else ('Yes' if p < .05 else 'No'))
    return table


def save_figure(fig, output, name):
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'{name}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def make_figures(pairs, output):
    data = pairs[pairs['Both successful']]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    sizes = sorted(pairs['Map size'].unique(), key=lambda x: tuple(map(int, x.split(' x '))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, metric in zip(axes, ['Time ms', 'Turns', 'Path length']):
        for index, size in enumerate(sizes):
            part = data[data['Map size'] == size]
            for j, suffix in enumerate(['jps', 'opt']):
                x, y = part[metric + '_jps'], part[metric + '_opt']
                valid = np.isfinite(x) & np.isfinite(y)
                if metric == 'Time ms':
                    valid &= (x > 0) & (y > 0)
                values = part.loc[valid, metric + '_' + suffix].to_numpy()
                if len(values):
                    boxes = ax.boxplot([values], positions=[index + (j - .5) * .32], widths=.27,
                                       patch_artist=True, showfliers=False, manage_ticks=False,
                                       medianprops={'color': 'black'})
                    boxes['boxes'][0].set_facecolor(COLORS[j])
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_ylabel(LABELS[metric]); ax.grid(axis='y', alpha=.2)
        if metric == 'Time ms' and (data['Time ms_jps'] > 0).any(): ax.set_yscale('log')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, SHORT)], loc='upper center', bbox_to_anchor=(.5, 1.05), ncol=2)
    fig.suptitle('Matched successful scenarios by map size', y=1.08)
    save_figure(fig, output, 'figure_1_map_size')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, metric in zip(axes[:2], ['Time ms', 'Turns']):
        for j, suffix in enumerate(['jps', 'opt']):
            for i, label in enumerate(RANGES):
                part = data[data.Range == label]
                valid = np.isfinite(part[metric + '_jps']) & np.isfinite(part[metric + '_opt'])
                _, med, q1, q3 = quantiles(part.loc[valid, metric + '_' + suffix])
                if np.isfinite(med):
                    ax.errorbar(i + (j - .5) * .15, med, yerr=[[med-q1], [q3-med]],
                                fmt='o', capsize=4, color=COLORS[j], label=SHORT[j] if i == 0 else None)
        ax.set_ylabel(LABELS[metric])
    for i, label in enumerate(RANGES):
        part = data[data.Range == label]
        x, y = part['Path length_jps'], part['Path length_opt']
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
        _, med, q1, q3 = quantiles(100*(y[valid]-x[valid])/x[valid])
        if np.isfinite(med): axes[2].errorbar(i, med, yerr=[[med-q1], [q3-med]], fmt='o', color=COLORS[1], capsize=4)
    axes[2].axhline(0, color='grey', linewidth=.8)
    axes[2].set_ylabel('Per-scenario path-length change (%)')
    for ax in axes:
        ax.set_xticks(range(5), [f'R{i+1}' for i in range(5)])
        ax.set_xlabel('Dataset reference-length range'); ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle('Reference-length groups: median and Q1-Q3 intervals')
    save_figure(fig, output, 'figure_2_length_ranges')

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x, y = data['Path length_jps'], data['Path length_opt']
    turns_a, turns_b = data['Turns_jps'], data['Turns_opt']
    valid = (x > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(turns_a) & np.isfinite(turns_b)
    if valid.any():
        chart = ax.hexbin(100*(y[valid]-x[valid])/x[valid], turns_a[valid]-turns_b[valid],
                          gridsize=55, mincnt=1, bins='log', cmap='viridis')
        fig.colorbar(chart, ax=ax, label='Matched scenarios per hexagon (log scale)')
    else:
        ax.text(.5, .5, 'No eligible successful pairs', transform=ax.transAxes, ha='center')
    ax.axhline(0, color='#c34848', linewidth=1); ax.axvline(0, color='#c34848', linewidth=1)
    ax.set_xlabel('Path-length change relative to Standard JPS (%)')
    ax.set_ylabel('Turns saved: Standard JPS minus optimized')
    ax.set_title(f'Path-length / smoothness trade-off | {int(valid.sum()):,} matched scenarios')
    save_figure(fig, output, 'figure_3_tradeoff')
    return {'tradeoff_eligible_pairs': int(valid.sum()),
            'tradeoff_excluded_pairs': int(len(data)-valid.sum()),
            'runtime_boxplot_nonpositive_pairs_excluded': int(((data['Time ms_jps'] <= 0) | (data['Time ms_opt'] <= 0)).sum())}


def write_report(tables, manifest, output):
    for name, table in tables.items(): table.to_csv(output / (name + '.csv'), index=False)
    sections = []
    primary = [LABELS[m] for m in METRICS[:6]]
    for grouping, filename in [('Overall', 'table_1_overall'), ('Map size', 'table_2_map_size'),
                               ('Reference length', 'table_3_length_ranges')]:
        subset = tables['summary_numeric'].query('Grouping == @grouping').copy()
        subset['Median [Q1, Q3]'] = subset.apply(lambda r: f'{r.Median:,.2f} [{r.Q1:,.2f}, {r.Q3:,.2f}]' if r.N else 'N/A', axis=1)
        wide = subset.pivot(index=['Group', 'Metric', 'N'], columns='Algorithm', values='Median [Q1, Q3]').reset_index()
        wide.to_csv(output / (filename + '.csv'), index=False)
        display = wide[wide.Metric.isin(primary)]
        sections.append(f'<h2>{html.escape(grouping)}</h2>' + display.to_html(index=False, border=0))
    captions = [
        'Figure 1. Matched successful scenarios; box = Q1-Q3, line = median, whiskers extend to observations within 1.5 IQR. Outlier markers are hidden but observations remain in all calculations. Runtime uses a log axis and excludes nonpositive times.',
        'Figure 2. Median and Q1-Q3 intervals (not confidence intervals). R1: [0,100); R2: [100,250); R3: [250,500); R4: [500,1000); R5: [1000,infinity). The third panel summarizes per-scenario percentage changes, not differences between medians.',
        'Figure 3. All eligible matched successful scenarios; positive horizontal values mean longer optimized paths; positive vertical values mean fewer turns. Zero-length baselines are excluded; all other eligible values are shown.']
    figures = ''.join(f'<figure><img src="{name}.png"><figcaption>{caption}</figcaption></figure>' for name, caption in zip(
        ['figure_1_map_size', 'figure_2_length_ranges', 'figure_3_tradeoff'], captions))
    notes = '''<p>Each comparison uses the same map/scenario pairs. Success counts include all included pairs;
performance summaries and plots use pairs where both algorithms succeeded. For each metric, both values must be finite.
N is the metric-specific number of pairs. Quartiles use linear interpolation. Results are pooled over scenarios, not equally weighted by map.</p>
<p>All percentage changes in relative_changes.csv are optimized relative to Standard JPS: 100*(optimized-JPS)/JPS.
Negative means a reduction. The file separately reports the percentage difference between group medians and
median [Q1,Q3] of per-scenario changes. Nonpositive baselines are excluded from per-scenario percentages and counted explicitly.</p>
<p>All saved matched scenarios are pooled for comparison.</p>
<p>Reference-length groups pool map sizes and do not isolate path length as a causal factor.
Negative reference gaps are retained; reference compatibility and movement rules must be checked before claiming optimality.
Statistical results are conditional on the saved sample and the assumptions below; no optimality guarantees are inferred. The report reflects saved progress, not necessarily the complete dataset.</p>'''
    report = '<!doctype html><html><head><meta charset="utf-8"><title>JPS benchmark report</title><style>body{font-family:Arial,sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#243444}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #dce3e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8}img{width:100%}figure{margin:32px 0}figcaption{font-size:14px;color:#526475}p{line-height:1.5}</style></head><body>'
    report += '<h1>Standard JPS versus Optimized FWSAJPS</h1><p>Snapshot: ' + html.escape(manifest['snapshot_time']) + '</p>'
    report += f"<p><b>{manifest['included_pairs']:,} included matched scenarios; {manifest['both_successful_pairs']:,} successful for both.</b></p>"
    report += '<p>Source: ' + html.escape(manifest['checkpoint']) + '</p>'
    report += '<p>Invalid pair metadata excluded: ' + str(manifest['invalid_pair_metadata_excluded']) + '.</p>'
    report += '<h2>Saved results</h2>' + tables['saved_coverage'].to_html(index=False, border=0)
    report += '<h2>Included scenario counts and success rates</h2>' + tables['matched_counts'].to_html(index=False, border=0)
    changes = tables['relative_changes']
    changes = changes[changes.Metric.isin(primary)]
    report += ''.join(sections) + '<h2>Relative changes: optimized versus Standard JPS</h2><p>Negative values mean a reduction. The two percentage measures below are different calculations.</p>' + changes.to_html(index=False, border=0, float_format=lambda value: f'{value:,.2f}')
    stats = tables['statistical_comparisons']
    report += '<h2>Statistical comparison</h2><p>Primary analysis: each map contributes its median of scenario-paired differences (optimized minus Standard JPS). An exact two-sided sign test checks whether positive and negative nonzero map differences are equally likely. It does not test equality of pooled scenario medians. Maps are equally weighted.</p>'
    report += '<p>Primary effect size: paired sign effect = (positive - negative) / nonzero units. Rank-biserial effect is also provided: (positive rank sum - negative rank sum) / total rank sum, using average ranks of absolute nonzero differences. Both range from -1 to +1; negative means lower optimized values. For reference gap, lower does not necessarily mean better. Zero differences are excluded from these denominators; tie counts are reported. All-zero differences yield effect 0 and p=1 by convention.</p>'
    report += '<p>Holm adjustment covers all metrics and groups separately for each analysis unit; use adjusted p-values for significance at 0.05. Statistical significance is not practical importance. Primary inference assumes independent maps; related city variants or resolutions may violate that assumption. Scenario-level Wilcoxon results are exploratory because scenarios within a map are dependent and signed-rank inference assumes symmetric differences. Small-sample normal approximations are flagged.</p>'
    report += '<p>Only jointly successful scenarios with finite values are used. Differences are rounded to 9 decimal places before ranking to suppress floating-point artifacts. Fewer than two nonzero units receive no inferential p-value. CSV values of zero p may reflect numerical underflow; HTML displays these as below machine precision.</p>'
    primary_stats = stats[(stats.Grouping == 'Overall') & (stats.Unit == 'Map median difference (primary)')]
    columns = ['Metric', 'N units', 'Nonzero units', 'Median paired difference', 'Paired sign effect', 'Rank-biserial effect', 'p-value', 'Holm-adjusted p-value', 'Significant (Holm < 0.05)', 'Status']
    def format_p(value):
        return 'Not tested' if not np.isfinite(value) else ('< machine precision' if value == 0 else f'{value:.3e}')
    formatters = {key: format_p for key in ['p-value', 'Holm-adjusted p-value']}
    report += primary_stats[columns].to_html(index=False, border=0, formatters=formatters, float_format=lambda v: f'{v:.4g}')
    report += '<details><summary>All metrics, groups and exploratory scenario-level tests</summary>' + stats.to_html(index=False, border=0, formatters=formatters, float_format=lambda v: f'{v:.4g}') + '</details>'
    report += '<p>Methods: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html">SciPy exact binomial test</a>; <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html">SciPy Wilcoxon test</a>.</p>'
    report += figures + '<h2>Presentation figures</h2><p>' + html.escape(manifest.get('path_selection_rule', '')) + '</p>'
    report += pd.DataFrame(manifest.get('path_examples', [])).to_html(index=False, border=0)
    for figure in manifest.get('presentation_figures', []):
        report += '<figure><a href="' + html.escape(figure['name']) + '.pdf"><img src="' + html.escape(figure['name']) + '.png"></a><figcaption>' + html.escape(figure['caption']) + '</figcaption></figure>'
    report += '<h2>Methods and interpretation</h2>' + notes
    report += '<p>CSV tables contain full-precision numeric quartiles, relative changes, and supplementary search-effort metrics. Figures are also supplied as PDF.</p></body></html>'
    (output / 'report.html').write_text(report, encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def main(argv=None):
    default = Path(__file__).resolve().parent.parent / 'results_new_ablation/street/street_all/cache/benchmark_f29e734530e46b1d_street_checkpoint.sqlite3'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=default)
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are not overwritten.')
    parser.add_argument('--maps-dir', type=Path, help='Map directory for reconstructing path illustrations.')
    parser.add_argument('--runtime-tie-percent', type=float, default=1.0, help='Descriptive runtime tie tolerance relative to JPS (default 1%%).')
    parser.add_argument('--export-pairs', action='store_true', help='Also save included scenario-level data as CSV.')
    args = parser.parse_args(argv)
    if not np.isfinite(args.runtime_tie_percent) or args.runtime_tie_percent < 0: parser.error('--runtime-tie-percent must be finite and nonnegative')
    if not args.checkpoint.is_file(): parser.error(f'Checkpoint not found: {args.checkpoint}')
    print('Reading a consistent checkpoint snapshot (read only)...', flush=True)
    frame, manifest = read_snapshot(args.checkpoint)
    pairs, details = match_pairs(frame)
    if pairs.empty: parser.error('No eligible matched scenarios for this selection.')
    manifest.update(details)
    manifest['unmatched_saved_runs'] = {name: int((frame.algorithm == name).sum()) - details['matched_before_filters'] for name in ALGORITHMS}
    output = args.output or args.checkpoint.resolve().parent.parent / 'reports' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'Generating report for {len(pairs):,} matched scenarios...', flush=True)
    tables = build_tables(frame, pairs)
    tables['statistical_comparisons'] = statistical_comparisons(pairs)
    manifest['statistics'] = {'difference': 'optimized minus Standard JPS, rounded to 9 decimals',
                              'primary': 'exact two-sided sign test of within-map median paired differences',
                              'exploratory': 'scenario-level Wilcoxon normal approximation with continuity correction',
                              'correction': 'Holm across all metrics and groups, separately per analysis unit', 'alpha': 0.05}
    manifest.update(make_figures(pairs, output))
    from presentation_plots import presentation_figures
    import sys
    maps_dir = args.maps_dir or Path(__file__).resolve().parent.parent / 'datasets' / manifest['checkpoint_signature'].get('dataset_name', 'street') / 'maps'
    print('Creating presentation figures and reconstructing selected paths...', flush=True)
    manifest.update(presentation_figures(pairs, tables, output, maps_dir, Path(__file__).with_name('Astar_JPS_SA_JPS_config_profiles_ready.py'), manifest, args.runtime_tie_percent/100, sys.modules[__name__]))
    if args.export_pairs: pairs.to_csv(output / 'matched_scenarios.csv', index=False)
    write_report(tables, manifest, output)
    print(f'Report: {(output / "report.html").resolve()}', flush=True)
    return output


if __name__ == '__main__':
    main()
