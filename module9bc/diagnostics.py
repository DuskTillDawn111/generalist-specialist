"""Post-replay evidence limits and runtime overhead; no solver tuning."""
import time,itertools
import numpy as np
from scipy.stats import spearmanr
from deployment_core import Workload,Hardware
from deployment_core.routing.e2e_router import solve_e2e_routing
from deployment_core.search.e2e_candidate_ranker import E2ECandidateRanker
from .common import *

def main():
    c,h,roles,cal=inputs();grid=cells();replay=read(OUT/'routing_replay.json')['rows'];hist=[r for r in replay if r['origin']=='historical_runtime']
    summary={'all_replay_states':len(replay),'historical_states':len(hist),'historical_mean_L1_rps':float(np.mean([r['routing_L1_rps'] for r in hist])),
             'historical_mean_L1_over_Lambda':float(np.mean([r['routing_L1_over_Lambda'] for r in hist])),
             'historical_changed_states':sum(r['routing_L1_rps']>1e-7 for r in hist),
             'all_replay_mean_L1_rps':float(np.mean([r['routing_L1_rps'] for r in replay])),
             'historical_task_shift':{k:{'mean_fraction':float(np.mean([r['task_shift_fraction'].get(k,0.) for r in hist])),
                 'maximum_fraction':max(r['task_shift_fraction'].get(k,0.) for r in hist)} for k in grid['C01']['lambda_rps']},
             'multi_G':[]}
    for r in hist:
        if r['N_G']<2:continue
        before=flat(r['R0_routing']);after=flat(r['R1_routing'])
        summary['multi_G'].append({'cell':r['cell'],'id':r['id'],'generalists':{m:{'R0_total_rps':sum(v for (k,name),v in before.items() if name==m),'R1_total_rps':sum(v for (k,name),v in after.items() if name==m),'R0_D':r['R0_D'][m],'R1_D':r['R1_D'][m]} for m,n in r['deployment'].items() if n and roles[m]=='generalist'}})
    save('routing_distance_and_task_shift.json',summary)
    candidates=read(OUT/'candidate_replay.json')['rows'];iterations=read(OUT/'candidate_replay.json')['iterations'];metrics=read(OUT/'candidate_ranking_metrics.json')
    # Report objective ties honestly: nonsteady candidates of the same tier
    # are tied under frozen runtime accept/rollback, regardless of their means.
    old_rows=[];new_rows=[];ties=[]
    for iteration in iterations:
        rs=[r for r in candidates if r['cell']==iteration['cell'] and r['iteration']==iteration['iteration'] and r.get('stability')]
        if len(rs)<2:continue
        tier=max(TIER[r['stability']] for r in rs)
        eligible=[r for r in rs if TIER[r['stability']]==tier]
        if tier==2:
            key=lambda r:(r['mean_e2e_ms'],r['weighted_p95_ms'],sum(cal.memory_table_mib[m][n] for m,n in r['deployment'].items()))
            bestkey=min(key(r) for r in eligible);best=[r for r in eligible if key(r)==bestkey]
        else:best=eligible
        # Known-set class diversification uses the per-row class score. Old
        # known-set winners are independently recomputed from frozen objects.
        from types import SimpleNamespace
        from method_v2.selection import diversified_order
        objects=[];lookup={}
        for r in rs:
            original=next(x for x in read(HISTORY/iteration['cell']/'static_v2/search_complete.json')['iterations'][iteration['iteration']-1]['candidates'] if x['mutation']['deployment']==r['deployment'])
            obj=SimpleNamespace(mutation=SimpleNamespace(**original['mutation']),delta_z=r['delta_z'],memory_mib=original['memory_mib']);objects.append(obj);lookup[id(obj)]=r
        current=read(HISTORY/iteration['cell']/'static_v2/search_complete.json')['iterations'][iteration['iteration']-1]['deployment']
        w=Workload(grid[iteration['cell']]['lambda_rps']);old=[lookup[id(o)] for o in diversified_order(objects,current,c,w,sum(cal.memory_table_mib[m][n] for m,n in current.items()))]
        globalnew=sorted(rs,key=lambda r:(-r['delta_J_seconds'],next(row['R1_z'] for row in replay if row['cell']==r['cell'] and row['deployment']==r['deployment']),sum(cal.memory_table_mib[m][n] for m,n in r['deployment'].items()),tuple(sorted(r['deployment'].items()))))
        p=[r for r in globalnew if r['class']=='P'];replacement=[r for r in globalnew if r['class']=='R'];first=p[0] if p else replacement[0]
        second=next((r for r in replacement if r is not first),None) or next((r for r in globalnew if r is not first),None)
        new=[first]+([second] if second else [])+[r for r in globalnew if r is not first and r is not second]
        ties.append({'cell':iteration['cell'],'iteration':iteration['iteration'],'known_candidates':len(rs),'objective_best_ties':len(best),
                     'old_Top1_hit':any(r is old[0] for r in best),'new_Top1_hit':any(r is new[0] for r in best),
                     'old_Top2_hit':any(any(r is a for a in old[:2]) for r in best),'new_Top2_hit':any(any(r is a for a in new[:2]) for r in best)})
    metrics['original_single_best_tiebreak_rates']={'old':metrics['old'].copy(),'new':metrics['new'].copy()}
    for tag in ['old','new']:
        for k in [1,2]:metrics[tag][f'Top{k}_hit_rate']=sum(r[f'{tag}_Top{k}_hit'] for r in ties)/len(ties)
    metrics['Topk_objective_ties']='all best candidates at same nonsteady tier count as ties; steady: mean, P95, memory; no arbitrary nonsteady memory tiebreak'
    metrics['Topk_audit']=ties
    metrics['known_label_count_distribution']={str(n):sum(r['known_candidates']==n for r in ties) for n in sorted({r['known_candidates'] for r in ties})}
    metrics['Top2_ceiling_warning']='Most historical iterations have exactly two labelled candidates; a Top2 hit=1 is largely a measured-subset ceiling, not evidence about unmeasured candidates.'
    # Cluster bootstrap descriptive only, preserving all records from a cell.
    labelled=[r for r in candidates if r.get('mean_improvement_ms') is not None]
    present=sorted({r['cell'] for r in labelled});rng=np.random.default_rng(9_023_101);diff=[]
    bycell={cell:[r for r in labelled if r['cell']==cell] for cell in present}
    for _ in range(10000):
        sampled=[r for cell in rng.choice(present,len(present),replace=True) for r in bycell[cell]]
        y=[r['mean_improvement_ms'] for r in sampled]
        oldcorr=spearmanr([r['delta_z'] for r in sampled],y).statistic;newcorr=spearmanr([r['delta_J_seconds'] for r in sampled],y).statistic
        if np.isfinite(oldcorr) and np.isfinite(newcorr):diff.append(float(newcorr-oldcorr))
    metrics['cell_cluster_bootstrap_Spearman_gain']={'resamples':10000,'seed':9_023_101,'cells':len(present),'CI95':np.quantile(diff,[.025,.975]).tolist(),'selection_bias_removed':False,'does_not_change_gate_rule':True}
    save('candidate_ranking_metrics.json',metrics)
    # Distinguish the exact missing oracle from generated directional ADDs.
    import csv,json
    with (OUT/'oracle_reachability_audit.csv').open() as stream:
        oracle_rows=list(csv.DictReader(stream))
    for row in oracle_rows:
        row['exact_oracle_diagnosis']='GENERATED' if row['exact_generated']=='True' else 'CANDIDATE_GENERATION_LIMIT'
        directional=json.loads(row['second_G_ranks'])
        row['directional_ADD_diagnosis']='GENERATED_OUTSIDE_TOP_B' if directional and all(r['old']>2 and r['new']>2 for r in directional) else 'GENERATED_WITH_TOP_B_ACCESS' if directional else 'CANDIDATE_GENERATION_LIMIT'
        row['diagnosis']=row['exact_oracle_diagnosis']
    csvsave('oracle_reachability_audit.csv',oracle_rows)
    # Uncached solve times, warmed imports/BLAS; round-robin through fixed states.
    def timings(samples):return {'n':len(samples),'median_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),'maximum_ms':float(max(samples))}
    duration=[]
    for index in range(100):
        r=hist[index%len(hist)];w=Workload(grid[r['cell']]['lambda_rps']);start=time.perf_counter();fit=solve_e2e_routing(r['deployment'],w,c,cal);duration.append(1000*(time.perf_counter()-start));assert fit['converged']
    neighborhood=[]
    for index in range(20):
        path=sorted(HISTORY.glob('C*/static_v2/search_complete.json'))[index%12];cell=path.parts[-3];trace=read(path)
        if not trace['iterations']:continue
        tr=trace['iterations'][0];w=Workload(grid[cell]['lambda_rps']);ranker=E2ECandidateRanker(c,cal)
        start=time.perf_counter();count=0
        for row in tr['candidates']:
            if row['canonical'] and row['rank']:
                fit=ranker.evaluate(row['mutation']['deployment'],w);assert fit['converged'];count+=1
        ranker.evaluate(tr['deployment'],w)
        neighborhood.append({'cell':cell,'feasible_candidates':count,'wall_ms':1000*(time.perf_counter()-start)})
    save('runtime_overhead.json',{'deployment_J_solve':timings(duration),'candidate_J_star_evaluation':timings(duration),'whole_frozen_neighborhood_J_evaluations':timings([r['wall_ms'] for r in neighborhood]),'neighborhood_samples':neighborhood,'cache':'per exact deployment/workload; benchmark uncached','GPU_profiling_calls':0,'note':'neighborhood timing includes current/candidate J solves; excludes unchanged generation and canonical feasibility LPs; full generation+LP+ranking timed at closeout'})
    verify(read(OUT/'experiment_freeze.json')['protected_references'])
    print(json.dumps({'routing_summary':summary,'overhead':read(OUT/'runtime_overhead.json'),'candidate_cluster_CI':metrics['cell_cluster_bootstrap_Spearman_gain']},default=str)[:1800],flush=True)
if __name__=='__main__':main()
