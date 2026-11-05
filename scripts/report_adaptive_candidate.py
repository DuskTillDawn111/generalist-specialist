"""Raw candidate allocations, thresholds, restart controls and paired budgets."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.analyze_candidate_potential import load_compact
from scripts.report_multi_candidate import write,save


def collect(sources):
    runs=[];stages=[];decisions=[];segments=[]
    for source in sources:
        source=Path(source)
        for meta in json.loads((source/'summary.json').read_text())['rows']:
            row=load_compact(source/meta['file']);raw=row['raw']
            ident={k:row[k] for k in ('axis','case','method','k','seed')}
            candidates=raw.get('candidates',[])
            initial=candidates[0]['grid_utility'] if candidates else None
            events=[{k:e[k] for k in ('stage','seconds','predictions','outer','utility','deployment','routing','cached')}
                    for e in raw['events']]
            hits=[e for e in events if initial is not None and e['utility'] is not None and e['utility']>=initial-1e-12]
            higher=[e for e in events if initial is not None and e['utility'] is not None and e['utility']>=initial+.001]
            summary=dict(ident,seconds=row['seconds'],U=row['utility'],predictions=row['predictions'],
                milp=row['milp'],routes=row['routes'],cache_hits=raw['budget']['cache_hits'],
                generated=len(candidates),safe_pruned=sum(s.get('safe_pruned',False) for s in raw.get('stages',[])),
                initial_U=initial,improvement=row['utility']-initial if initial is not None else None,
                time_to_joint=min((e['seconds'] for e in hits),default=None),
                predictions_to_joint=min((e['predictions'] for e in hits),default=None),
                higher_system_threshold=initial+.001 if initial is not None else None,
                time_to_higher=min((e['seconds'] for e in higher),default=None),
                predictions_to_higher=min((e['predictions'] for e in higher),default=None),
                generation_seconds=raw.get('generation_seconds'),status=row['status'],source=str(source/meta['file']),events=events)
            for outer_prefix in (64,128):
                prefix=[e for e in events if e['outer']<=outer_prefix and e['utility'] is not None]
                summary[f'U_at_outer_{outer_prefix}']=max((e['utility'] for e in prefix),default=None)
                summary[f'predictions_at_outer_{outer_prefix}']=max((e['predictions'] for e in prefix),default=None)
                summary[f'reached_outer_{outer_prefix}']=row['routes']>=outer_prefix
            summary['predictions_on_zero_gain_candidates']=sum(s['counts']['predictions'] for s in raw.get('stages',[]) if s['improvement']<=1e-12)
            runs.append(summary)
            c=json.loads((source/f"{row['case']}-config.json").read_text())
            byidx={s['generation_index']:s for s in raw.get('stages',[])}
            final_order=sorted(candidates,key=lambda x:-byidx.get(x['generation_index'],{}).get('continuous_utility',x['grid_utility']))
            grid_order=sorted(candidates,key=lambda x:-x['grid_utility'])
            for candidate in candidates:
                i=candidate['generation_index'];s=byidx.get(i)
                if s is None:continue
                calls=0;first=None;gains=0;best=candidate['grid_utility']
                for h in s['history']:
                    calls+=h['full_prediction_calls'];value=h['best_feasible_utility']
                    if value is not None and value>best+1e-12:
                        gains+=1;best=value
                        if first is None:first=calls
                hits=[e for e in events if e['deployment']==candidate['deployment'] and e['utility'] is not None
                      and e['utility']>=candidate['grid_utility']+.01]
                terms=(s.get('bound') or {}).get('trace_floors',[])
                floors={name:sum(item['weight']*item['terms'][name] for item in terms)/sum(item['weight'] for item in terms) for name in 'QTLPME'} if terms else {}
                layers={}
                if floors:
                    weighted={name:c['weights'][name]*value for name,value in floors.items()}
                    layers=dict(floors=floors,UB_memory_only=-weighted['M']+1e-8,
                        UB_memory_energy=-weighted['M']-weighted['E']+1e-8,
                        UB_without_quality=-sum(value for name,value in weighted.items() if name!='Q')+1e-8)
                stages.append(dict(ident,**layers,index=i,deployment=candidate['deployment'],
                    grid_U=candidate['grid_utility'],refined_U=s['continuous_utility'],gain=s['improvement'],
                    grid_rank=next(j+1 for j,v in enumerate(grid_order) if v is candidate),
                    refined_rank=next(j+1 for j,v in enumerate(final_order) if v is candidate),
                    gap_to_incumbent=initial-candidate['grid_utility'],
                    dimension=s.get('dimension',sum(sum(t in c['catalog']['models'][m]['supported_tasks'] for m in candidate['deployment'])-1 for t in c['workload'])),
                    memory_mib=candidate['grid_prediction']['safe_memory_mib'],
                    bound=(s.get('bound') or {}).get('upper_bound'),safe_pruned=s.get('safe_pruned',False),
                    state=s.get('state',s.get('status')),budget_paused=s.get('state')=='ACTIVE' and raw['status'].endswith('_LIMIT'),predictions=s['counts']['predictions'],
                    routes=s['counts']['outer'],completed_routes=len(s['history']),cache_hits=s['counts']['cache_hits'],
                    seconds=s['seconds'],restart_rounds=s.get('restart_rounds',0),
                    screen_predictions=(s.get('screen_counts') or {}).get('predictions'),
                    screen_best=s.get('screen_best'),infeasible_evaluations=sum(h['constraint_state']=='INFEASIBLE' for h in s['history']),unknown_evaluations=sum(h['constraint_state']=='UNKNOWN' for h in s['history']),screen_routes=(s.get('screen_counts') or {}).get('outer'),first_gain_predictions=first,improving_evaluations=gains,
                    predictions_per_gain=s['counts']['predictions']/gains if gains else None,
                    time_to_local_gain_01=min((e['seconds'] for e in hits),default=None),
                    calls_to_local_gain_01=min((e['predictions'] for e in hits),default=None)))
            decisions.extend(dict(ident,**d) for d in raw.get('decisions',[]))
            segments.extend(dict(ident,**s) for s in raw.get('segments',[]))
            del row,raw,c
            gc.collect()
    comparison=[]
    for axis,case,method,k in sorted({tuple(r[n] for n in ('axis','case','method','k')) for r in runs}):
        rr=[r for r in runs if (r['axis'],r['case'],r['method'],r['k'])==(axis,case,method,k)]
        med=lambda key:statistics.median(r[key] for r in rr if r[key] is not None) if any(r[key] is not None for r in rr) else None
        comparison.append(dict(axis=axis,case=case,method=method,k=k,seeds=len(rr),
            seconds=med('seconds'),seconds_min=min(r['seconds'] for r in rr),seconds_max=max(r['seconds'] for r in rr),
            predictions=med('predictions'),milp=med('milp'),routes=med('routes'),cache_hits=med('cache_hits'),
            safe_pruned=med('safe_pruned'),U_min=min(r['U'] for r in rr if r['U'] is not None),
            U_max=max(r['U'] for r in rr if r['U'] is not None),
            time_to_joint=med('time_to_joint'),predictions_to_joint=med('predictions_to_joint'),
            higher_threshold_hits=sum(r['time_to_higher'] is not None for r in rr)))
    return dict(runs=runs,stages=stages,decisions=decisions,segments=segments,comparison=comparison,
        total_runs=len(runs),total_predictions=sum(r['predictions'] for r in runs),
        improved_runs=sum((r['improvement'] or 0)>1e-12 for r in runs))


def plots(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    methods=['joint','top1','uniform','safe_uniform','uniform_restart','priority','adaptive','safe_adaptive']
    case='p5-n1-K1-trace43'
    fig,axs=plt.subplots(3,3,figsize=(17,13))
    for i,axis in enumerate(('predictions','wall','outer')):
        rr=[next((r for r in data['comparison'] if r['case']==case and r['axis']==axis and r['method']==m
                 and r['k']==(1 if m in {'joint','top1'} else 5)),None) for m in methods]
        rr=[r for r in rr if r]
        for ax,key in zip(axs[i],('U_max','predictions','seconds')):
            ax.bar(range(len(rr)),[r[key] for r in rr]);ax.set_xticks(range(len(rr)),[r['method'] for r in rr],rotation=40,ha='right',fontsize=8)
            ax.set_title(axis+' budget: '+key);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'budget-comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for m in methods:
        run=next((r for r in data['runs'] if r['case']==case and r['axis']=='predictions' and r['seed']==11
            and r['method']==m and r['k']==(1 if m in {'joint','top1'} else 5)),None)
        if not run:continue
        events=[e for e in run['events'] if e['utility'] is not None]
        if not events:continue
        ys=np.maximum.accumulate([e['utility'] for e in events]);
        for ax,key in zip(axs,('seconds','predictions')):ax.step([e[key] for e in events],ys,where='post',label=m)
    for ax,key in zip(axs,('Wall seconds including setup/generation','Full predictions')):
        ax.set(xlabel=key,ylabel='Best valid U',title=case+' seed11');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'anytime.png',dpi=170);plt.close(fig)
    selected=['uniform','uniform_restart','priority','adaptive','safe_uniform','safe_adaptive']
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for j,m in enumerate(selected):
        ss=[s for s in data['stages'] if s['axis']=='predictions' and s['case']==case and s['method']==m and s['k']==5]
        ids=sorted({s['index'] for s in ss})
        vals=[statistics.median(s['predictions'] for s in ss if s['index']==i) for i in ids]
        gains=[statistics.median(s['gain'] for s in ss if s['index']==i) for i in ids]
        axs[0].bar(np.asarray(ids)+(j-2.5)*.13,vals,width=.13,label=m)
        axs[1].bar(np.asarray(ids)+(j-2.5)*.13,gains,width=.13,label=m)
    axs[0].set(xlabel='Generation index',ylabel='Median refinement full predictions')
    axs[1].set(xlabel='Generation index',ylabel='Median candidate gain')
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'candidate-allocation.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for m in ('uniform','safe_uniform','adaptive','safe_adaptive'):
        rr=sorted([r for r in data['comparison'] if r['axis']=='predictions' and r['case']==case and r['method']==m],key=lambda r:r['k'])
        for ax,key in zip(axs,('predictions','U_max')):ax.plot([r['k'] for r in rr],[r[key] for r in rr],'.-',label=m)
    for ax,key in zip(axs,('Median full predictions','Best observed U')):ax.set(xlabel='k',ylabel=key);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'candidate-count.png',dpi=170);plt.close(fig)
    ss=[s for s in data['stages'] if s['axis']=='predictions' and s['case']==case and s['method']=='safe_adaptive' and s['k']==5 and s['seed']==11]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot([s['index'] for s in ss],[s['grid_U'] for s in ss],'.-',label='Grid U')
    ax.plot([s['index'] for s in ss],[s.get('UB_memory_energy') for s in ss],'.-',label='Memory + energy UB')
    ax.plot([s['index'] for s in ss],[s.get('UB_without_quality') for s in ss],'.-',label='Plus service, without quality UB')
    ax.plot([s['index'] for s in ss],[s['bound'] for s in ss],'.-',label='Valid continuous-domain UB')
    if ss:ax.axhline(ss[0]['grid_U'],color='black',linestyle='--',label='Original Joint incumbent')
    ax.set(xlabel='Generation index',ylabel='Utility / upper bound',title='Optimistic bounds include zero-flow resident costs');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'safe-bounds.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=collect(args.sources)
    for key in ('runs','stages','decisions','segments','comparison'):
        rows=[{k:v for k,v in r.items() if k!='events'} for r in data[key]]
        write(out/f'{key}.csv',rows)
    save(out/'analysis.json',data);plots(out,data)
    print('Reported',data['total_runs'],'runs',data['total_predictions'],'full predictions')
