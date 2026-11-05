"""Aggregate Experiment B; preserve seed variation and censored target times."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def write(path,rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(args):
    src,out=Path(args.source),Path(args.output);out.mkdir(parents=True,exist_ok=False)
    runs=[json.loads(p.read_text()) for p in sorted(src.glob('*-seed*.json'))]
    if args.supplement:
        supplemental=[json.loads(p.read_text()) for p in sorted(Path(args.supplement).glob('*-seed*.json'))]
        replaced={r['case'] for r in supplemental}
        runs=[r for r in runs if r['case'] not in replaced]+supplemental
    groups={}
    for r in runs: groups.setdefault((r['case'],r['algorithm']),[]).append(r)
    targets={case:max((r['utility'] for r in rs if r['algorithm']=='joint' and r['utility'] is not None),default=None)
             for case in {r['case'] for r in runs} for rs in [[r for r in runs if r['case']==case]]}
    seeds=[];rows=[];evals=[]
    for r in runs:
        target=targets[r['case']]
        offset=r['convergence_offset_seconds']
        if r['algorithm']=='joint':
            # Include optimizer construction between session setup and solve.
            # The remaining error is only wrapper entry/exit bookkeeping.
            offset=max(offset,r['seconds']-r['raw']['raw']['elapsed_seconds'])
        r['report_convergence_offset_seconds']=offset
        times=[e['seconds']+offset for e in r['convergence']
               if target is not None and e['utility'] is not None and e['utility']>=target-1e-12]
        hit=min(times) if times else None
        audit=r.get('routing_audit',{})
        history=r['raw'].get('histories',[])
        seeds.append(dict(case=r['case'],algorithm=r['algorithm'],seed=r['seed'],
            seconds=r['seconds'],predictions=r['predictions'],milp_calls=r['milp_calls'],routes=r['route_evaluations'],
            utility=r['utility'],target_utility=target,time_to_target=hit,target_reached=hit is not None,
            initial_utility=(r['raw'].get('initial_winner') or {}).get('utility'),
            improvement=r['raw'].get('utility_improvement'),
            final_changed_requests=audit.get('final_changed_requests'),
            best_off_grid_utility=audit.get('best_off_grid_utility'),
            off_grid_evaluations=sum(e['outside_old_grid'] for e in audit.get('rows',[])),
            changed_mapping_evaluations=sum(e['changed_requests']>0 for e in audit.get('rows',[])),
            same_mapping_off_grid=sum(e['outside_old_grid'] and e['changed_requests']==0 for e in audit.get('rows',[])),
            stages=len(history),neighbor_stages=sum(h['stage'].startswith('neighbor') for h in history),
            fixed_predictions=sum(h['predictions'] for h in history if h['stage']=='fixed'),
            neighbor_predictions=sum(h['predictions'] for h in history if h['stage'].startswith('neighbor')),
            initial_milp_calls=r['raw'].get('initial_counts',{}).get('milp'),
            unique_route_deployment_pairs=len({json.dumps([h['deployment'],h['routing_qps']],sort_keys=True)
                for stage in history for h in stage['history']}) if history else None,
            status=r['status']))
        for e in audit.get('rows',[]):
            evals.append(dict(case=r['case'],algorithm=r['algorithm'],seed=r['seed'],stage=e['stage'],
                seconds=e['seconds'],utility=e['utility'],state=e['constraint_state'],
                changed_requests=e['changed_requests'],off_grid=e['outside_old_grid'],prediction_calls=e['prediction_calls']))
    for (case,algorithm),rs in groups.items():
        ss=[s for s in seeds if s['case']==case and s['algorithm']==algorithm]
        median=lambda key:statistics.median(s[key] for s in ss if s[key] is not None)
        hits=[s['time_to_target'] for s in ss if s['time_to_target'] is not None]
        improvements=[s['improvement'] for s in ss if s['improvement'] is not None]
        rows.append(dict(case=case,algorithm=algorithm,seeds=len(rs),seconds=median('seconds'),
            seconds_min=min(s['seconds'] for s in ss),seconds_max=max(s['seconds'] for s in ss),
            predictions=median('predictions'),milp_calls=median('milp_calls'),routes=median('routes'),
            utility_min=min((s['utility'] for s in ss if s['utility'] is not None),default=None),
            utility_max=max((s['utility'] for s in ss if s['utility'] is not None),default=None),
            target_hit_seeds=len(hits),time_to_target_median=statistics.median(hits) if hits else None,
            improvement_max=max(improvements) if improvements else None,
            fixed_predictions=median('fixed_predictions'),neighbor_predictions=median('neighbor_predictions')))
    for case in targets:
        rr=[r for r in runs if r['case']==case]
        assert len({r['input_sha256'] for r in rr})==1
    for r in runs:
        if r['algorithm'].startswith('refine_'):
            raw=r['raw']
            if raw['initial_winner']:
                assert r['utility']>=raw['initial_winner']['utility']
            assert all(h['milp_calls']==0 for h in raw['histories'])
            assert raw['budget']['milp_solve_calls']==raw['initial_counts']['milp']
    write(out/'comparison.csv',rows);write(out/'seeds.csv',seeds);write(out/'routing-evaluations.csv',evals)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=sorted(targets)
    algos=['joint','de_milp','refine_local','refine_de','refine_de_neighbors']
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for i,a in enumerate(algos):
        rr=[next(r for r in rows if r['case']==c and r['algorithm']==a) for c in cases]
        x=[j+(i-2)*.15 for j in range(len(cases))]
        for ax,key in zip(axs,('seconds','predictions','milp_calls')):
            ax.bar(x,[r[key] for r in rr],width=.15,label=a)
    for ax,label in zip(axs,('Total wall seconds','Full prediction calls','MILP calls')):
        ax.set_xticks(range(len(cases)),cases,rotation=50,ha='right');ax.set_ylabel(label)
    axs[1].set_yscale('symlog',linthresh=1);axs[2].set_yscale('symlog',linthresh=1)
    axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    case='p5-n1-K2'
    for a in algos:
        rr=next((r for r in runs if r['case']==case and r['algorithm']==a and r['seed']==11),None)
        if rr:
            best=None;ts=[];us=[]
            for e in rr['convergence']:
                if e['utility'] is not None:
                    best=e['utility'] if best is None else max(best,e['utility'])
                    ts.append(e['seconds']+rr['report_convergence_offset_seconds']);us.append(best)
            for ax in axes:
                ax.step(ts,us,where='post',label=a)
    axes[1].set_ylim(-.42,-.29)
    for ax in axes:
        ax.set(xlabel='Elapsed seconds including initial Joint MILP',ylabel='Best feasible U')
        ax.grid(alpha=.2)
    axes[0].set_title(case+' optimizer seed 11');axes[1].set_title('Zoom near old grid optimum')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
    (out/'verification.json').write_text(json.dumps(dict(runs=len(runs),same_input_per_case=True,
        initial_incumbent_never_degrades=True,no_deployment_milp_during_refinement=True,
        target_times_censored_when_not_reached=True),indent=2))
    for r in rows:
        print(r['case'],r['algorithm'],round(r['seconds'],3),'pred',r['predictions'],'milp',r['milp_calls'],
              'U',r['utility_min'],r['utility_max'],'hit',r['target_hit_seeds'],'delta',r['improvement_max'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--supplement')
    main(p.parse_args())
