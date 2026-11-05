"""Export fair-comparison, phase-accounting and per-MILP-round diagnostic tables."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def csv_file(path,rows):
    if not rows:return
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((source/'summary.json').read_text())
    records=summary['records'];domains=summary['domains']
    winners=[]
    for record in records:
        raw=json.loads((source/record['raw_file']).read_text())
        winner=raw['winner']
        if winner is None:
            continue
        for task,shares in raw['winner_routing_shares'].items():
            counts=raw['fifo']['counts'][task]
            for model,n in winner['deployment'].items():
                for replica in range(n):
                    worker=f'{model}#{replica}'
                    winners.append(dict(case=record['case'],algorithm=record['algorithm'],repeat=record['repeat'],
                        utility=raw['utility'],task=task,model=model,replicas=n,worker=worker,
                        model_route_share=shares.get(model,0),model_route_qps=winner['routing'][task].get(model,0),
                        actual_fifo_requests=counts.get(worker,0),total_task_requests=sum(counts.values()),
                        trace_sha256=raw['fifo']['trace_sha256'],assignment_sha256=raw['fifo']['assignment_sha256']))
    csv_file(output/'winner-routing-fifo.csv',winners)
    grouped=[]
    for domain in domains:
        for algorithm in ('dfs','joint_milp','de_milp'):
            selected=[r for r in records if r['case']==domain['case'] and r['algorithm']==algorithm]
            values=[r['utility'] for r in selected if r['utility'] is not None]
            grouped.append(dict(case=domain['case'],algorithm=algorithm,
                tasks=domain['tasks'],models=domain['models'],deployment_states=domain['deployment_states'],
                routing_dimension=domain['routing_dimension'],grid_identities=domain['grid_complete_identities'],
                seconds=statistics.median(r['seconds'] for r in selected),
                min_seconds=min(r['seconds'] for r in selected),max_seconds=max(r['seconds'] for r in selected),
                predictions=statistics.median(r['predictions'] for r in selected),
                milp_calls=statistics.median(r['milp_calls'] for r in selected),
                utility=min(values) if values else None,max_utility=max(values) if values else None,
                certified=sum(r['certified'] for r in selected),repeats=len(selected),
                statuses=';'.join(r['status'] for r in selected)))
    csv_file(output/'end-to-end.csv',grouped)
    timings=[];rounds=[];fixed=[]
    phases=['outer_optimizer','milp_build','highs_solver','feedback','predictor','other_python']
    for path in sorted(source.glob('*-profile.json')):
        row=json.loads(path.read_text());timing=row['timing'];seconds=timing['seconds']
        phase={k:seconds.get(k,0) for k in phases}
        identity=seconds.get('identity_validation',0)
        phase['other_python']+=identity
        total=timing['wall_seconds']
        timings.append(dict(file=path.name,case=row['case'],algorithm=row['algorithm'],route_id=row.get('route_id'),
            wall_seconds=total,**{k+'_seconds':v for k,v in phase.items()},
            **{k+'_percent':100*v/total for k,v in phase.items()},
            identity_validation_seconds=identity,identity_validation_percent=100*identity/total,
            accounted_seconds=timing['accounted_seconds'],accounting_error_seconds=timing['accounting_error_seconds']))
        for r in timing['rounds']:
            rounds.append(dict(file=path.name,case=row['case'],algorithm=row['algorithm'],
                route_id=row.get('route_id',r['route']),iteration=r['iteration'],
                solved='solver_status' in r,solver_status=r.get('solver_status'),
                rows=r.get('rows'),columns=r.get('columns'),nonzeros=r.get('nonzeros'),
                wall_seconds=r['wall_seconds'],highs_seconds=r['seconds'].get('highs_solver',0),
                build_seconds=r['seconds'].get('milp_build',0),feedback_seconds=r['seconds'].get('feedback',0),
                prediction_seconds=r['seconds'].get('predictor',0),prediction_bundles=r['prediction_bundles'],
                identity_exclusions_added=r['identity_exclusions_added'],
                safely_pruned_deployments=r['safely_pruned_deployments']))
        if row['algorithm'].startswith('fixed_'):
            evaluation=row['raw']['evaluations'][0]
            scored=[tuple(r.get('selected_variables',())) for r in timing['rounds'] if r['prediction_bundles']]
            fixed.append(dict(case=row['case'],route_id=row['route_id'],algorithm=row['algorithm'],
                seconds=row['seconds'],utility=row['utility'],certified=row['certified'],
                domain_size=evaluation['deployment_domain_size'],predictions=row['predictions'],milp_calls=row['milp_calls'],
                safely_pruned=evaluation['safely_pruned_deployments'],
                duplicate_scored_milp_witnesses=len(scored)-len(set(scored))))
    csv_file(output/'timing.csv',timings);csv_file(output/'milp-rounds.csv',rounds);csv_file(output/'fixed-routing.csv',fixed)
    (output/'summary.json').write_text(json.dumps(dict(end_to_end=grouped,timings=timings,fixed_routing=fixed,
        total_wall_seconds=summary['total_wall_seconds']),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'dfs':'#218c74','joint_milp':'#397fba','de_milp':'#d38c26'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(len(domains))
    for i,algorithm in enumerate(colors):
        selected=[r for r in grouped if r['algorithm']==algorithm]
        for ax,key in zip(axes,['seconds','predictions']):
            ax.bar(x+(i-1)*.25,[r[key] for r in selected],width=.24,label=algorithm,color=colors[algorithm])
    for ax,title in zip(axes,['Wall time (two-run median), seconds','Full prediction bundles (two-run median)']):
        ax.set_xticks(x,[d['case'] for d in domains],rotation=24,ha='right',fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].legend();fig.suptitle('Validated real-data replay; four fixed tasks; budget 8 s / 16 routes')
    fig.savefig(output/'scaling.png',dpi=170);plt.close(fig)
    chosen=[r for r in timings if r['case']=='pool5' and (r['algorithm'] in ('dfs','joint_milp','de_milp')
             or (r['algorithm']=='fixed_milp' and r['route_id']==0))]
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    y=np.arange(len(chosen));left=np.zeros(len(y))
    for phase in phases:
        values=np.array([r[phase+'_percent'] for r in chosen]);ax.barh(y,values,left=left,label=phase);left+=values
    ax.set_yticks(y,[r['algorithm'] for r in chosen]);ax.set_xlim(0,100);ax.set_xlabel('Exclusive elapsed-time share (%)')
    ax.set_title('Separate instrumented diagnostics; primary benchmark timings are uninstrumented')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8)
    fig.savefig(output/'time-breakdown.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.source,a.output)
