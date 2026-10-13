"""Separate J ranking, noise-aware acceptance and U without changing algorithms."""
from .common import *
from .cpu import formal_v3,identity
from deployment_search.runtime import RuntimeEvaluation
from deployment_search.real_objective import NoiseAwareRuntimeObjective

def runtime(p,rates):
    total=sum(rates.values());per=p['per_task']
    return RuntimeEvaluation('steady',sum(rates[t]*per[t]['mean_ms'] for t in rates)/total,
        sum(rates[t]*per[t]['p95_ms'] for t in rates)/total,p['predicted_within_window_QPS'],1.)

def main():
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);refs=read(fm['utility_reference']['path'])['latency_reference_ms'];weights=read(fm['utility_definition']['path'])['weights'];rows=[];bench=[]
    for pool,pr in sp['pools'].items():
        d=read(pr['path'])
        for load,L in sp['loads'].items():
            rates=L['rates'];oracle=read(OUT/'cpu'/f'{pool}_{load}_oracle.json');formal=read(OUT/'cpu'/f'{pool}_{load}_formal_V3.json');u=oracle['U_oracle'];j=oracle['J_rho08_oracle'];w=formal['decision'];p=w['prediction']
            # Original noise-aware pairwise comparator is not a scalar total order.
            # Audit whether any finite candidate would improve the chosen runtime.
            from .cpu import evaluate,catalog
            cp=catalog(d);improvers=[];rule=NoiseAwareRuntimeObjective(1.,.05)
            for rr in csv.DictReader((OUT/'cpu'/f'{pool}_{load}_all_candidates.csv').open()):
                if rr['V3_rho08_eligible']!='True':continue
                x=json.loads(rr['deployment']);route=json.loads(rr['routing']);pred,_,_=evaluate(d,x,route,rates,sp['budget_mib'],refs,weights,cp)
                good,reason=rule.compare(runtime(pred,rates),runtime(p,rates),pred['safe_memory_mib'],p['safe_memory_mib'])
                if good:improvers.append(dict(candidate=rr['candidate_id'],reason=reason,predicted_U=pred['U']))
            row=dict(pool=pool,load=load,formal_objective_is_U=False,formal_objective_is_scalar_J=False,J_is_router_and_ranking_proxy=True,
                oracle_U=u['predicted_U'],formal_U=p['U'],grid_J_oracle_U=j['predicted_U'],
                formal_U_regret=u['predicted_U']-p['U'],grid_J_oracle_U_objective_regret=u['predicted_U']-j['predicted_U'],
                formal_J_regret_vs_grid_J_oracle=w['J']-j['predicted_J_seconds'],
                formal_noise_aware_improving_finite_candidates=improvers,formal_noise_aware_undominated=not improvers,
                J_gap_interpretation='Cannot call this formal search error under scalar J: formal acceptance is original noise-aware latency comparator. Exact J-grid optimum reported independently.',
                finite_native_route=True)
            rows.append(row)
            original_seconds=formal['seconds'];seconds=[original_seconds]
            for rep in [2,3]:
                fit=formal_v3(d,rates,sp['budget_mib'],refs,weights);assert fit['status']=='COMPLETED_CPU_FORECAST_BACKEND'
                assert identity(fit['decision']['deployment'],fit['decision']['routing'])==identity(w['deployment'],w['routing'])
                save(f'cpu/{pool}_{load}_formal_V3_repeat{rep}.json',fit);seconds.append(fit['seconds'])
            for rep,sec in enumerate(seconds,1):
                bench.append(dict(pool=pool,load=load,repetition=rep,algorithm='FORMAL_V3_CPU_FORECAST_BACKEND',seconds=sec,
                    oracle_seconds=oracle['seconds'],speedup=oracle['seconds']/sec,
                    evaluations=formal['objective_calls'],J_route_solves=formal['J_route_solves'],
                    visited_deployment_count=formal['result']['visited_deployment_count'],
                    total_replica_vectors=oracle['total_deployment_vectors']))
    csvsave('objective_error_decomposition.csv',rows);save('objective_error_decomposition.json',rows);csvsave('formal_optimizer_runtime_benchmark.csv',bench)
    print('OBJECTIVE_AND_FORMAL_RUNTIME_AUDIT_DONE',flush=True)
if __name__=='__main__':main()
