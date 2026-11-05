"""Candidate ranking, budgets, exact FIFO mappings and anytime comparisons."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))


def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def load_runs(sources):
    runs=[]
    for source in sources:
        source=Path(source)
        for r in json.loads((source/'summary.json').read_text())['rows']:
            name=f"{r['case']}-{r['algorithm']}-s{r['seed']}.json"
            row=json.loads((source/name).read_text());row['source']=str(source/name)
            # Reporting needs event counts and exact mapping hashes, while the
            # full predictions remain in the source files. Do not retain many
            # redundant nested prediction dictionaries across dozens of runs.
            row.pop('routing_audit',None)
            raw=row['raw'];raw.pop('joint_solves',None);raw.pop('history',None)
            for event in raw.get('events',[]):event.pop('prediction',None)
            for c in raw.get('candidates',[]):c.pop('grid_prediction',None)
            for stage in raw.get('stages',[]):
                for h in stage['history']:
                    if h.get('prediction'):
                        for trace in h['prediction']['traces']:trace.pop('prediction',None)
            runs.append(row)
    return runs


def analyze(runs):
    candidates=[];stages=[];seeds=[];reversals=[]
    for run in runs:
        raw=run['raw'];pool=raw.get('candidates',[]);refined=raw.get('stages',[])
        selected=raw.get('selected_indices',[0] if pool else [])
        selected_pool=[c for c in pool if c['generation_index'] in selected]
        gridrank={c['generation_index']:i+1 for i,c in enumerate(sorted(selected_pool,key=lambda c:-c['grid_utility']))}
        afterrank={s['generation_index']:i+1 for i,s in enumerate(sorted(refined,key=lambda s:-s['continuous_utility']))}
        for c in pool:
            stage=next((s for s in refined if s['generation_index']==c['generation_index']),None)
            candidates.append(dict(case=run['case'],algorithm=run['algorithm'],seed=run['seed'],
                generation_index=c['generation_index'],deployment=c['deployment'],grid_routing=c['grid_routing'],
                grid_U=c['grid_utility'],selected=c['generation_index'] in selected,
                refined=stage is not None,grid_rank_selected=gridrank.get(c['generation_index']),
                continuous_U=stage['continuous_utility'] if stage else None,
                improvement=stage['improvement'] if stage else None,continuous_rank=afterrank.get(c['generation_index']),
                generation_seconds=c['seconds'],generation_predictions=c['counts']['predictions'],
                generation_milp=c['counts']['milp'],certified=c['restricted_grid_certified'],
                generation_status=c['generation_status']))
        for a in refined:
            for b in refined:
                if a['grid_utility']>b['grid_utility']+1e-12 and a['continuous_utility']<b['continuous_utility']-1e-12:
                    reversals.append(dict(case=run['case'],algorithm=run['algorithm'],seed=run['seed'],
                        original_higher=a['generation_index'],original_lower=b['generation_index'],
                        higher_grid_U=a['grid_utility'],lower_grid_U=b['grid_utility'],
                        higher_continuous_U=a['continuous_utility'],lower_continuous_U=b['continuous_utility'],
                        beats_original_first=b['continuous_utility']>pool[0]['grid_utility']+1e-12))
        for s in refined:
            best=s['grid_utility'];gains=0;calls=0;first_gain=None;maps=[];routes=[];values=[]
            for h in s['history']:
                calls+=h['full_prediction_calls']
                if h['feasible']:
                    v=h['best_feasible_utility']
                    if v>best+1e-12:
                        best=v;gains+=1
                        if first_gain is None:first_gain=calls
                    maps.append(h['prediction']['traces'][0]['assignment_sha256']);values.append(v)
                    routes.append(json.dumps(h['routing_qps'],sort_keys=True))
            stages.append(dict(case=run['case'],algorithm=run['algorithm'],seed=run['seed'],
                index=s['generation_index'],deployment=s['deployment'],grid_U=s['grid_utility'],continuous_U=s['continuous_utility'],
                improvement=s['improvement'],seconds=s['seconds'],predictions=s['counts']['predictions'],
                milp=s['counts']['milp'],routes=s['counts']['outer'],cache_hits=s['counts']['cache_hits'],
                improving_evaluations=gains,first_gain_prediction_calls=first_gain,
                predictions_per_improving_evaluation=s['counts']['predictions']/gains if gains else None,
                distinct_full_assignments=len(set(maps)),distinct_exact_routes=len(set(routes)),
                adjacent_evaluation_same_mapping=sum(a==b for a,b in zip(maps,maps[1:])),
                adjacent_same_mapping_U_changed=sum(a==b and abs(x-y)>1e-12 for a,b,x,y in zip(maps,maps[1:],values,values[1:])),
                mapping_comparison_scope='consecutive DE evaluations, not adjacent sorted scalar ratios',
                status=s['status'],winner_routing=s['winner']['routing']))
        initial=pool[0]['grid_utility'] if pool else None
        seeds.append(dict(case=run['case'],algorithm=run['algorithm'],seed=run['seed'],k=run['k'],strategy=run['strategy'],
            seconds=run['seconds'],U=run['utility'],predictions=run['predictions'],milp=run['milp_calls'],routes=run['route_evaluations'],
            generated_candidates=len(pool),selected_candidates=len(selected),refined_candidates=len(refined),
            generation_seconds=raw.get('generation_seconds'),initial_U=initial,
            improvement_over_first=run['utility']-initial if run['utility'] is not None and initial is not None else None,
            max_candidate_gain=max((s['improvement'] for s in refined),default=None),
            winner_deployment=(raw.get('winner') or {}).get('deployment'),
            winner_routing=(raw.get('winner') or {}).get('routing'),status=run['status'],source=run['source']))
    targets={}
    for r in runs:
        if r['algorithm']=='joint' and r['utility'] is not None:
            targets[r['case']]=max(targets.get(r['case'],-float('inf')),r['utility'])
    for r,s in zip(runs,seeds):
        target=targets.get(r['case']);es=r['raw']['events']
        hits=[e for e in es if target is not None and e['utility'] is not None and e['utility']>=target-1e-12]
        s['target_U']=target;s['time_to_target']=min((e['seconds'] for e in hits),default=None)
        s['predictions_to_target']=min((e['predictions'] for e in hits),default=None)
    comparison=[]
    for case,algorithm in sorted({(r['case'],r['algorithm']) for r in runs}):
        rr=[r for r in seeds if r['case']==case and r['algorithm']==algorithm]
        median=lambda key:statistics.median([r[key] for r in rr if r[key] is not None]) if any(r[key] is not None for r in rr) else None
        comparison.append(dict(case=case,algorithm=algorithm,seeds=len(rr),seconds=median('seconds'),
            seconds_min=min(r['seconds'] for r in rr),seconds_max=max(r['seconds'] for r in rr),
            predictions=median('predictions'),milp=median('milp'),routes=median('routes'),
            U_min=min((r['U'] for r in rr if r['U'] is not None),default=None),U_max=max((r['U'] for r in rr if r['U'] is not None),default=None),
            generated=median('generated_candidates'),refined=median('refined_candidates'),
            target_hit_seeds=sum(r['time_to_target'] is not None for r in rr),time_to_target=median('time_to_target'),
            predictions_to_target=median('predictions_to_target'),max_gain=median('max_candidate_gain')))
    for case in {r['case'] for r in runs}:
        assert len({r['config_sha256'] for r in runs if r['case']==case})==1
    return dict(candidates=candidates,stages=stages,seeds=seeds,reversals=reversals,comparison=comparison,
        total_runs=len(runs),total_predictions=sum(r['predictions'] for r in runs),
        stages_with_positive_gain=sum(s['improvement']>1e-12 for s in stages),
        runs_beating_initial=sum((r['improvement_over_first'] or 0)>1e-12 for r in seeds),
        strict_rank_reversals=len(reversals),reversals_beating_original_first=sum(r['beats_original_first'] for r in reversals))


def plots(out,runs,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    preferred='p5-n1-K1-trace42'
    case=preferred if any(r['case']==preferred for r in runs) else runs[0]['case']
    algorithms=['joint','de_milp','utility-k1','utility-k3','utility-k5','diversity-k5','utility-k8']
    selected=[next((r for r in runs if r['case']==case and r['algorithm']==a and r['seed']==11),None) for a in algorithms]
    selected=[r for r in selected if r]
    for axis,filename,label in [('seconds','anytime-time.png','Wall seconds including setup/generation'),
                               ('predictions','anytime-predictions.png','Full prediction calls')]:
        fig,axs=plt.subplots(1,2,figsize=(13,4.5))
        for r in selected:
            xs=[];ys=[];best=None
            for e in r['raw']['events']:
                if e['utility'] is not None:
                    best=e['utility'] if best is None else max(best,e['utility']);xs.append(e[axis]);ys.append(best)
            if xs:
                xs.append(r['seconds'] if axis=='seconds' else r['predictions']);ys.append(best)
                for ax in axs:ax.step(xs,ys,where='post',label=r['algorithm'])
        axs[1].set_ylim(-.36,-.30)
        for ax in axs:ax.set(xlabel=label,ylabel='Best valid U',title=case+' seed11');ax.grid(alpha=.2)
        axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/filename,dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for i,r in enumerate([r for r in selected if r['algorithm'] in {'utility-k5','diversity-k5'}]):
        ss=[s for s in data['stages'] if s['case']==case and s['algorithm']==r['algorithm'] and s['seed']==11]
        for s in ss:
            axs[0].plot([0,1],[s['grid_U'],s['continuous_U']],'.-',label=f"{r['algorithm']} D{s['index']}")
        axs[1].bar([s['index']+(i-.5)*.3 for s in ss],[s['improvement'] for s in ss],width=.3,label=r['algorithm'])
    axs[0].set(xticks=[0,1],xticklabels=['Grid','Continuous best known'],ylabel='Candidate U',title='Candidate ranking changes')
    axs[1].set(xlabel='Generation index (0 = original Joint)',ylabel='Continuous U minus grid U')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'candidate-gains.png',dpi=170);plt.close(fig)
    rankcases=[c for c in [case,'p5-n1-K1-trace43'] if any(r['case']==c for r in runs)]
    rankcases=list(dict.fromkeys(rankcases))
    fig,axs=plt.subplots(1,len(rankcases),figsize=(8*len(rankcases),5),squeeze=False)
    for ax,rankcase in zip(axs.flat,rankcases):
        for algorithm in ('utility-k5','diversity-k5'):
            cs=[c for c in data['candidates'] if c['case']==rankcase and c['algorithm']==algorithm and c['seed']==11 and c['refined']]
            for c in cs:ax.plot([0,1],[c['grid_rank_selected'],c['continuous_rank']],'.-',
                               label=f"{algorithm} D{c['generation_index']}")
        ax.set(xticks=[0,1],xticklabels=['Grid rank in selected set','Refined rank'],ylabel='Rank (1 = highest U)',title=rankcase)
        ax.invert_yaxis();ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'candidate-ranks.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    groups=[(c,a.split('-k')[0]) for c,a in {(r['case'],r['algorithm']) for r in data['comparison']} if '-k' in a]
    for c,strategy in sorted(set(groups)):
        rr=sorted([r for r in data['comparison'] if r['case']==c and r['algorithm'].startswith(strategy+'-k')],key=lambda r:int(r['algorithm'].split('-k')[1]))
        xs=[int(r['algorithm'].split('-k')[1]) for r in rr]
        for ax,key in zip(axs,['U_max','seconds','predictions']):ax.plot(xs,[r[key] for r in rr],'.-',label=c+' '+strategy)
    for ax,label in zip(axs,['Best observed valid U','Median total wall seconds','Median full predictions']):
        ax.set(xlabel='Requested candidates k',ylabel=label,xticks=[1,3,5,8]);ax.grid(alpha=.2)
    axs[1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'candidate-count.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    runs=load_runs(args.sources);data=analyze(runs)
    for k in ('candidates','stages','seeds','reversals','comparison'):write(out/f'{k}.csv',data[k])
    save(out/'analysis.json',data);plots(out,runs,data)
