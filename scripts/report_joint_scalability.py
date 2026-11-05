"""Render Experiment A tables and standalone plots from frozen raw runs."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def csv_file(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(args):
    src,out=Path(args.source),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    runs=[json.loads(p.read_text()) for p in sorted(src.glob('*-r*.json'))]
    groups={}
    for r in runs: groups.setdefault((r['case'],r['algorithm']),[]).append(r)
    rows=[];rounds=[];evaluations=[]
    for (case,algorithm),rs in groups.items():
        d=json.loads((src/f'{case}-domain.json').read_text())
        med=lambda f:statistics.median(f(r) for r in rs)
        row=dict(case=case,algorithm=algorithm,K=rs[0]['K'],repeats=len(rs),
            total_seconds=med(lambda r:r['seconds']),setup_seconds=med(lambda r:r['setup_seconds']),
            raw_deployments=d['raw_deployment_combinations'],memory_survivors=d['memory_bound_survivors'],
            cover_survivors=d['memory_and_task_cover_survivors'],models=d['models'],tasks=d['tasks'],
            route_dimension=d['routing_dimension'],splitting_tasks=d['splitting_tasks'],
            raw_columns=d['raw_task_count_columns'],
            actual_columns=med(lambda r:r['metrics'].get('task_columns',0)),
            columns_complete=all(r['timing']['column_generation_complete'] for r in rs) if algorithm=='joint' else None,
            variables=max((s['variables'] for r in rs for s in r['timing']['rounds']),default=0),
            constraints=max((s['constraints'] for r in rs for s in r['timing']['rounds']),default=0),
            columns_seconds=med(lambda r:r['timing']['columns_seconds']),
            highs_seconds=med(lambda r:r['timing']['highs_seconds']),
            master_seconds=med(lambda r:r['timing']['master_seconds']),
            prediction_seconds=med(lambda r:r['timing']['prediction_seconds']),
            predictions=med(lambda r:r['metrics'].get('full_prediction_calls',0)),
            milp_calls=med(lambda r:r['metrics'].get('milp_calls',r['metrics'].get('milp_solve_calls',0))),
            utility=rs[0]['utility'],same_utility=all(r['utility']==rs[0]['utility'] for r in rs),
            all_certified=all(r['certified'] for r in rs),statuses='|'.join(sorted({r['status'] for r in rs})))
        rows.append(row)
        for r in rs:
            for i,s in enumerate(r['timing']['rounds']):
                rounds.append(dict(case=case,algorithm=algorithm,repeat=r['repeat'],round=i,**s))
            for i,s in enumerate(r['timing']['evaluations']):
                evaluations.append(dict(case=case,algorithm=algorithm,repeat=r['repeat'],evaluation=i,
                    seconds=r['setup_seconds']+s['seconds'],utility=s['utility'],reason=s['reason']))
    csv_file(out/'scalability.csv',rows);csv_file(out/'rounds.csv',rounds);csv_file(out/'evaluations.csv',evaluations)
    for case in {r['case'] for r in runs}:
        rs=[r for r in runs if r['case']==case]
        assert len({r['input_sha256'] for r in rs})==1
        certified=[r for r in rs if r['certified']]
        if certified:
            assert max(r['utility'] for r in certified)-min(r['utility'] for r in certified)<1e-10
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for prefix in ('p2-n1','p3-n1','p5-n1','p7-n1','p5-n2','p7-n2'):
        rr=sorted([r for r in rows if r['algorithm']=='joint' and r['case'].startswith(prefix+'-K') and 'trace' not in r['case']],key=lambda r:r['K'])
        for ax,key in zip(axs,('total_seconds','actual_columns','predictions')):
            ax.plot([r['K'] for r in rr],[r[key] for r in rr],marker='o',label=prefix)
            bad=[r for r in rr if not r['all_certified']]
            ax.scatter([r['K'] for r in bad],[r[key] for r in bad],marker='x',s=100,color='black',zorder=5)
    for ax,label in zip(axs,('Total wall seconds (budget 8s)','Generated task/count columns','Full prediction calls')):
        ax.set(xlabel='K: denominator union 1..K',ylabel=label);ax.set_yscale('symlog',linthresh=1);ax.grid(alpha=.2)
    axs[0].axhline(8,color='gray',linestyle=':',linewidth=1)
    axs[2].annotate('K16: build stopped; no prediction',xy=(16,0),xytext=(5,.2),fontsize=8,
                    arrowprops=dict(arrowstyle='->'))
    fig.suptitle('X = incomplete domain proof; generated columns can be partial',fontsize=10)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'K-scaling.png',dpi=180);plt.close(fig)
    selected=('p2-n1-K16','p5-n1-K16','p7-n1-K8','p7-n1-K12','p7-n1-K16','p7-n2-K8','p7-n99-K2')
    rr=[next((r for r in rows if r['case']==c and r['algorithm']=='joint'),None) for c in selected]
    rr=[r for r in rr if r]
    fig,ax=plt.subplots(figsize=(11,4));bottom=[0.]*len(rr)
    for label,fun in [('setup',lambda r:r['setup_seconds']),('columns',lambda r:r['columns_seconds']),
                      ('HiGHS',lambda r:r['highs_seconds']),('predictor',lambda r:r['prediction_seconds']),
                      ('other',lambda r:max(0,r['total_seconds']-r['setup_seconds']-r['columns_seconds']-r['highs_seconds']-r['prediction_seconds']))]:
        vals=[fun(r) for r in rr];ax.bar(range(len(rr)),vals,bottom=bottom,label=label)
        bottom=[a+b for a,b in zip(bottom,vals)]
    ax.set_xticks(range(len(rr)),[r['case'] for r in rr],rotation=20);ax.set_ylabel('Wall seconds');ax.legend();fig.tight_layout();fig.savefig(out/'bottlenecks.png',dpi=180);plt.close(fig)
    (out/'verification.json').write_text(json.dumps(dict(runs=len(runs),same_problem_within_case=True,
        same_certified_utility=True,source_unchanged=json.loads((src/'summary.json').read_text())['source_unchanged']),indent=2))
    for r in rows:
        if r['algorithm']=='joint':
            print(r['case'],round(r['total_seconds'],3),'cols',r['actual_columns'],'pred',r['predictions'],
                  'highs',round(r['highs_seconds'],3),'cert',r['all_certified'],r['statuses'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
