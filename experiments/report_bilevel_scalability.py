"""Offline tables and figures from the sealed 27 searches; no oracle calls."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bilevel_scalability_benchmark import read, receipt, save


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    p = args.input.resolve()
    summary, plan = read(p/'summary.json'), read(p/'plan.json')
    assert all(receipt(r['path']) == r for r in plan['sources'])
    rows, raw = summary['rows'], {}
    for r in rows:
        raw[r['size'], r['method'], r['seed']] = read(p/f"{r['size']}-{r['method']}-{r['seed']}.json")
    grouped = []
    for n, method in dict.fromkeys((r['size'], r['method']) for r in rows):
        group = [r for r in rows if r['size']==n and r['method']==method]
        metrics = {k: stats([r[k] for r in group]) for k in
                   ('U', 'gain_over_initialization', 'regret_upper', 'certified_evaluations', 'pool_size')}
        metrics.update({k: stats([r['budget'][k] for r in group]) for k in
                        ('full_prediction_calls', 'milp_solve_calls', 'elapsed_seconds')})
        grouped.append(dict(size=n, method=method, metrics=metrics,
            off_grid_winners=sum(r['non_K4'] for r in group), mixed_winners=sum(r['mixed'] for r in group),
            EI_proposals=[r['bo_ei_proposals'] for r in group],
            known_better_neighbors=sum(any(v['U'] is not None and v['U']>r['U']+1e-8 for v in r['neighbor_checks']) for r in group)))
    paired = []
    for method in plan['protocol']['methods_at_seven'][1:]:
        cases = []
        for seed in plan['protocol']['seeds']:
            full = next(r for r in rows if (r['size'],r['method'],r['seed'])==(7,'BO-MADS',seed))
            ablation = next(r for r in rows if (r['size'],r['method'],r['seed'])==(7,method,seed))
            cases.append(dict(seed=seed, full_minus_variant_U=full['U']-ablation['U'],
                prediction_saving_fraction=1-full['budget']['full_prediction_calls']/ablation['budget']['full_prediction_calls'],
                milp_saving_fraction=1-full['budget']['milp_solve_calls']/ablation['budget']['milp_solve_calls']))
        paired.append(dict(variant=method, cases=cases))
    diagnostics = []
    for key, result in raw.items():
        counts = Counter(r['source'] for r in result['events'])
        diagnostics.append(dict(size=key[0], method=key[1], seed=key[2], sources=counts,
            activation_capacity_drops=result['activation_spaces']['capacity_drops'],
            discovered_activation_spaces=result['activation_spaces']['discovered'],
            initial_prediction_calls=result['initial_signature'][-1]['work']['full_prediction_calls'],
            final_surrogate=result['surrogate']))
    save(p/'analysis.json', dict(groups=grouped, paired=paired, scheduler_diagnostics=diagnostics,
        scope='Single manufactured scenario family, three seeds; descriptive, no significance claim',
        conclusions=dict(bounded_scaling_to_14=True, general_scalability_proven=False,
            non_grid_improvement_at_three=True, non_grid_improvement_at_seven=False,
            combo_synergy_demonstrated=False, activation_search_helped_all_three_seeds=True,
            pool_and_progressive_saved_calls_but_not_uniformly_utility=True,
            no_continuous_local_optimum_claim=True, real_seven_model_validation=False)))
    columns = ['size','method','seed','U','gain_over_initialization','regret_upper',
               'full_prediction_calls','milp_solve_calls','elapsed_seconds','mixed','non_K4','bo_ei_proposals','mads_polls']
    with (p/'results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] if k in r else r['budget'][k] for k in columns})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    scale = [g for g in grouped if g['method']=='BO-MADS']
    sizes = [g['size'] for g in scale]
    for metric, label in [('full_prediction_calls', 'Complete predictions'), ('milp_solve_calls', 'MILP calls')]:
        med = np.array([g['metrics'][metric]['median'] for g in scale])
        lo = np.array([g['metrics'][metric]['minimum'] for g in scale])
        hi = np.array([g['metrics'][metric]['maximum'] for g in scale])
        axes[0,0].errorbar(sizes, med, yerr=[med-lo, hi-med], marker='o', capsize=4, label=label)
    axes[0,0].set(title='Bounded scaling: median and seed range', xlabel='Synthetic models', ylabel='Calls', xticks=sizes)
    axes[0,0].legend()
    med = np.array([g['metrics']['regret_upper']['median'] for g in scale])
    lo = np.array([g['metrics']['regret_upper']['minimum'] for g in scale])
    hi = np.array([g['metrics']['regret_upper']['maximum'] for g in scale])
    axes[0,1].errorbar(sizes, med, yerr=[med-lo, hi-med], marker='o', capsize=4)
    axes[0,1].set(title='Remaining utility gap grows with model pool', xlabel='Synthetic models',
                  ylabel='Reference upper U - returned U', xticks=sizes)
    colors = dict(zip(plan['protocol']['methods_at_seven'], plt.get_cmap('tab10').colors))
    for method in ('BO-MADS','BO-only','MADS-only','Grid-K4'):
        curves = []
        for seed in plan['protocol']['seeds']:
            history = raw[7,method,seed]['convergence']
            curves.append([max((h['best_feasible_utility'] for h in history
                if h['full_prediction_calls'] <= call and h['best_feasible_utility'] is not None), default=np.nan)
                for call in range(18,65)])
        axes[1,0].step(range(18,65), np.median(curves,axis=0), where='post', label=method, color=colors[method])
    axes[1,0].axhline(read(p/'reference-7.json')['lower_U'], color='black', linestyle=':', label='Continuous reference')
    axes[1,0].set(title='7 models: attained U within call allowance', xlabel='Prediction allowance (stopped runs retain incumbent)', ylabel='U (higher is better)')
    axes[1,0].legend(fontsize=8)
    seven = [g for g in grouped if g['size']==7]
    for i,g in enumerate(seven):
        m = g['metrics']; val = m['U']['median']
        axes[1,1].errorbar(i, val, yerr=[[val-m['U']['minimum']], [m['U']['maximum']-val]],
            fmt='o', capsize=4, color=colors[g['method']])
    axes[1,1].set(title='7 models: component trade-offs (median / range)',
        ylabel='Returned U (higher is better)', xticks=range(len(seven)), xticklabels=[g['method'] for g in seven])
    axes[1,1].tick_params(axis='x', rotation=30)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('27 bounded CPU searches | manufactured queue cost | not real GPU/FIFO evidence', fontsize=13)
    fig.savefig(p/'overview.png', dpi=160)
    fig.savefig(p/'overview.pdf')
    plt.close(fig)
    save(p/'analysis-receipt.json', dict(source=receipt(__file__), inputs=[receipt(p/name) for name in
        ('plan.json','summary.json','completion.json')], outputs=[receipt(p/name) for name in
        ('analysis.json','results.csv','overview.png','overview.pdf')], new_searches=0, oracle_calls=0))
    print(json.dumps(dict(groups=len(grouped), searches=len(rows), output=str(p)),ensure_ascii=False))


if __name__ == '__main__':
    main()
