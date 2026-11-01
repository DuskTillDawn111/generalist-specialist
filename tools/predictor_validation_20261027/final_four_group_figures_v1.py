"""Plot an existing campaign summary without running forecasts or experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def generate(summary_path, output):
    source = receipt(summary_path)
    summary = json.loads(Path(summary_path).read_text())
    final = summary['stages']['final_four_groups']
    cells = {(c['mode'], c['route']): c for c in final['cells']}
    modes = ('shared', 'admission2')
    routes = ('balanced', 'skewed', 'spill_low', 'spill_high')
    colors = ('#377eb8', '#ff7f00')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new figure directory')
    fig = plt.figure(figsize=(14, 10), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    utility = fig.add_subplot(grid[0, 0])
    contrast = fig.add_subplot(grid[0, 1])
    heat = fig.add_subplot(grid[1, :])
    x = np.arange(4)
    width = .34
    for index, mode in enumerate(modes):
        values = [cells[(mode, route)].get('U') for route in routes]
        positions = x + (index - .5) * width
        utility.bar(positions, [np.nan if v is None else v for v in values],
                    width, color=colors[index], label=mode)
        for position, value in zip(positions, values):
            if value is None:
                utility.annotate('UNKNOWN', (position, 0), rotation=90,
                                 fontsize=7, ha='center')
    utility.set(xticks=x, xticklabels=routes, ylabel='U (higher is better)',
                title='Eight fixed configurations: actual six-term utility')
    utility.tick_params(axis='x', labelrotation=15)
    utility.legend()
    comparisons = (final.get('descriptive_comparison') or {}).get('modes', {})
    for index, mode in enumerate(modes):
        value = comparisons.get(mode, {}).get('locked_allow_minus_strong_no_spill')
        contrast.bar(index, np.nan if value is None else value, color=colors[index])
        contrast.annotate('UNKNOWN' if value is None else f'{value:+.6f}',
                          (index, 0 if value is None else value),
                          xytext=(0, 6 if value is None or value >= 0 else -14),
                          textcoords='offset points', ha='center', fontsize=10)
    contrast.axhline(0, color='black', lw=.8)
    contrast.set(xticks=range(2), xticklabels=modes, ylabel='Delta U',
                 title='Locked spill_high minus best observed balanced/skewed')
    keys = [(m, r) for m in modes for r in routes]
    costs = np.full((6, 8), np.nan)
    for j, key in enumerate(keys):
        terms = cells[key].get('terms') or {}
        for i, term in enumerate('QTLPME'):
            value = terms.get(term)
            if value is not None:
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Original normalized costs must be finite and nonnegative')
                costs[i, j] = value
    chart = heat.imshow(np.ma.masked_invalid(np.log10(1 + costs)), aspect='auto',
                        cmap='YlOrRd', vmin=0)
    for i in range(6):
        for j in range(8):
            value = costs[i, j]
            heat.text(j, i, 'UNKNOWN' if np.isnan(value) else f'{value:.3g}',
                      ha='center', va='center', fontsize=8)
    heat.set(xticks=range(8), xticklabels=[m+'\n'+r for m, r in keys],
             yticks=range(6), yticklabels=list('QTLPME'),
             title='Original six normalized costs (lower is better); labels show actual values')
    fig.colorbar(chart, ax=heat, label='Color: log10(1 + normalized cost)', shrink=.8)
    for axis in (utility, contrast):
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Fixed final block | R=1 descriptive evidence\n'
                 'No run-level confidence interval, deployment certification, or physical optimum claim')
    if receipt(summary_path) != source:
        plt.close(fig)
        raise ValueError('Summary changed while plotting')
    output.mkdir(parents=True, exist_ok=False)
    try:
        for suffix in ('png', 'pdf'):
            fig.savefig(output/('four-group-comparison.'+suffix), dpi=180)
    finally:
        plt.close(fig)
    if receipt(summary_path) != source:
        raise ValueError('Summary changed while saving figures')
    result = dict(source=source, script=receipt(__file__),
                  figures=[receipt(output/('four-group-comparison.'+s)) for s in ('png', 'pdf')],
                  descriptive_only=True, GPU_started=False, model_fitted=False,
                  statistical_significance_claim=False, final_gain_eligible=False)
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    generate(args.summary, args.output)
