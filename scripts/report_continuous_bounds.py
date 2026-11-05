"""Summarize matched inner-search ablations; export standard PNG/CSV figures."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def run(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((source/'summary.json').read_text())
    records=summary['records']
    cases=list(dict.fromkeys(r['case'] for r in records))
    variants=['enumeration_de','control','bounds','warm','combined','combined_de']
    rows=[]
    for case in cases:
        base={r['seed']:r for r in records if r['case']==case and r['variant']=='enumeration_de'}
        for variant in variants:
            selected=[r for r in records if r['case']==case and r['variant']==variant]
            rows.append(dict(case=case,variant=variant,n=len(selected),
                predictions=statistics.median(r['predictions'] for r in selected),
                seconds=statistics.median(r['seconds'] for r in selected),
                bound_seconds=statistics.median(r['bound_seconds'] for r in selected),
                prediction_reduction=statistics.median(1-r['predictions']/base[r['seed']]['predictions'] for r in selected),
                paired_speedup=statistics.median(base[r['seed']]['seconds']/r['seconds'] for r in selected),
                utility=selected[0]['utility'],all_certified=all(r['all_certified'] for r in selected)))
    with (output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(rows=rows,all_matched=summary['all_matched'],
        runs=len(records),route_evaluations=sum(r['outer'] for r in records),total_wall_seconds=summary['total_wall_seconds']),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#7a8491','#bac1c9','#218c74','#e8ab42','#397fba','#8059a2']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    xx=np.arange(len(cases))
    for k,variant in enumerate(variants):
        selected=[next(r for r in rows if r['case']==c and r['variant']==variant) for c in cases]
        for ax,key in zip(axes,['predictions','seconds']):
            ax.bar(xx+(k-2.5)*.13,[r[key] for r in selected],width=.125,color=colors[k],label=variant)
    for ax,title in zip(axes,['Full prediction bundles (median)','Optimization wall time, seconds (median)']):
        ax.set_xticks(xx,cases,rotation=16)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[1].legend(fontsize=8)
    fig.suptitle('Synthetic CPU fixture: identical fixed-route utilities; 64 routes per run')
    fig.savefig(output/'inner-ablation.png',dpi=180)
    plt.close(fig)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.source,args.output)
