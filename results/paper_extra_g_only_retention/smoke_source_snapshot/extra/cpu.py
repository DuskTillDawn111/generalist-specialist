"""Bounded supplement; imports frozen algorithms, never changes their code."""
from .isolation import install
install('supervisor')
from .common import *
import sys
sys.path[:0] = [str(ROOT/'results/module34_candidate_pool'), str(ROOT/'results/module35_utility_routing'), str(ROOT)]
from m34.cpu_core import catalog, evaluate, candidate_routes, optimizer
from m34.cpu import static
from u_aligned_optimizer import UAlignedOptimizer


def active(d, x, r):
    resident = [m for m in x if d['models'][m]['role'] == 'specialist']
    used = {m: sum(row.get(m, 0.) for row in r.values()) for m in resident}
    return sorted(m for m, q in used.items() if q > 1e-9), sorted(m for m, q in used.items() if q <= 1e-9)


def choose(rows):
    legal = [v for v in rows if v['eligible']]
    return min(legal, key=lambda v: (-v['predicted_U'], v['safe_memory_mib'], identity(v['deployment'], v['routing']))) if legal else None


def pack(d, x, r, rates, p):
    used, idle = active(d, x, r)
    return dict(deployment=x, routing=r, route_fractions={t:{m:q/rates[t] for m,q in row.items()} for t,row in r.items()},
        structure=structure(d,x), active_specialists=used, idle_specialists=idle,
        specialist_admitted=bool(used), eligible=p['eligible'], predicted_U=p['U'],
        safe_memory_mib=p.get('safe_memory_mib'), max_rho=p.get('max_rho'),
        prediction_reason=p.get('reason'), prediction=p)


def main():
    proto=read(OUT/'experiment_protocol.json')
    assert ref(OUT/'experiment_protocol.json') == read(OUT/'protocol_freeze_receipt.json')['protocol']
    assert (OUT/'history_sha256_before.json').exists()
    smoke=read(OUT/'admission_smoke_result.json')
    assert smoke['campaign_finished'], 'No CPU timings during GPU startup/IO'
    budget=proto['budget_mib'];weights=read(proto['utility_definition']['path'])['weights']
    refs=read(proto['utility_reference']['path'])['latency_reference_ms'];K=proto['route_grid_denominator']
    oracles=[];decisions=[];breakdown=[];admission=[];candidates_all=[];runtime=[];audit=[]
    for scenario,load in proto['development_scenarios'].items():
        rates=load['rates'];baseU=None;previousU=None
        assert all(q>0 for q in rates.values())
        for pool,poolref in proto['pools'].items():
            d=read(poolref['path']);assert ref(poolref['path'])==poolref
            cp=catalog(d);models=sorted(d['models']);rows=[];vectors=[];start=time.perf_counter()
            for bits in itertools.product([0,1],repeat=len(models)):
                x={m:1 for m,b in zip(models,bits) if b}
                mem=d['safe_factor']*sum(d['memory_table_mib'][m]['1'] for m in x)
                coverage={t for m in x for t in d['models'][m]['supported_tasks']}
                reasons=[]
                if not set(rates)<=coverage:reasons.append('COVERAGE')
                if mem>budget+1e-6:reasons.append('MEMORY')
                vectors.append(dict(deployment=x,safe_memory_mib=mem,reasons=reasons))
                if reasons:continue
                for rr in candidate_routes(d,x,rates,K):
                    p,j,err=evaluate(d,x,rr,rates,budget,refs,weights,cp)
                    rows.append(dict(pool=pool,scenario=scenario,**pack(d,x,rr,rates,p),errors=err))
            elapsed=time.perf_counter()-start;best=choose(rows)
            pure=choose([v for v in rows if v['structure']=='G_ONLY'])
            mixed=choose([v for v in rows if v['structure']=='MIXED' and v['active_specialists']])
            if pool=='P0':baseU=best['predicted_U'] if best else None
            if best:
                assert pure and abs(baseU-pure['predicted_U'])<1e-9
                assert previousU is None or best['predicted_U']>=previousU-1e-9
                previousU=best['predicted_U']
            selected=dict(scenario=scenario,pool=pool,rates=rates,**(best or dict(eligible=False,predicted_U=None,structure=None,deployment=None,routing=None,route_fractions=None,active_specialists=[],idle_specialists=[],specialist_admitted=None,status='NO_ELIGIBLE_CPU_CANDIDATE')),
                regret=0. if best else None,seconds=elapsed,evaluated_routes=len(rows),prediction_calls=4*len(rows),
                resident_vectors=len(vectors),feasible_routes=sum(v['eligible'] for v in rows),
                P0_best_U=baseU,delta_U_admission=best['predicted_U']-baseU if best and baseU is not None else None,
                best_G_only_U=pure['predicted_U'] if pure else None,best_active_mixed_U=mixed['predicted_U'] if mixed else None,
                best_active_mixed_deployment=mixed['deployment'] if mixed else None,
                best_active_mixed_route_fractions=mixed['route_fractions'] if mixed else None,
                G_only_retained=bool(best and pool!='P0' and best['structure']=='G_ONLY' and abs(best['predicted_U']-baseU)<1e-9),
                oracle_monotone=True if best else None,
                rejected_prediction_reasons=sorted({v['prediction_reason'] for v in rows if v['prediction_reason']}))
            oracles.append(selected);candidates_all.extend(rows)
            save(f'cpu/{scenario}_{pool}_oracle.json',dict(selected=selected,best_G_only=pure,best_active_Mixed=mixed,vectors=vectors))
            csvsave(f'cpu/{scenario}_{pool}_candidates.csv',rows)
            print('ORACLE',scenario,pool,'routes',len(rows),'structure',selected['structure'],'U',selected['predicted_U'],'admission_gain',selected['delta_U_admission'],flush=True)

            def record(name,plan,seconds,calls,metrics=None):
                if plan is None:
                    row=dict(pool=pool,scenario=scenario,algorithm=name,status='NO_ELIGIBLE_WINNER',eligible=False,predicted_U=None,regret=None,seconds=seconds,prediction_calls=calls,oracle_U_match=None,oracle_no_plan_match=best is None)
                else:
                    x,r=plan;p,j,err=evaluate(d,x,r,rates,budget,refs,weights,cp)
                    grid=all(abs(q/rates[t]*K-round(q/rates[t]*K))<1e-7 for t,v in r.items() for q in v.values())
                    member=any(identity(x,r)==identity(z['deployment'],z['routing']) for z in rows if z['eligible'])
                    row=dict(pool=pool,scenario=scenario,algorithm=name,**pack(d,x,r,rates,p),
                        status='FEASIBLE_SAME_DOMAIN' if p['eligible'] and grid and member else 'NATIVE_OUTSIDE_COMMON_DOMAIN' if not grid else 'CONSTRAINT_REJECTED',
                        same_domain=bool(grid and member),regret=best['predicted_U']-p['U'] if best and p['eligible'] and grid and member else None,
                        seconds=seconds,prediction_calls=calls,search_metrics=metrics,errors=err,
                        oracle_model_replica_match=x==best['deployment'] if best else None,oracle_plan_match=bool(best and member and identity(x,r)==identity(best['deployment'],best['routing'])),
                        oracle_U_match=bool(p['eligible'] and abs(p['U']-best['predicted_U'])<1e-9) if best else None,oracle_no_plan_match=best is None and not p['eligible'])
                decisions.append(row);runtime.append({k:row.get(k) for k in ['pool','scenario','algorithm','seconds','prediction_calls','search_metrics','regret','same_domain']})
                return row

            record('CPU_U_ORACLE',(best['deployment'],best['routing']) if best else None,elapsed,4*len(rows),dict(evaluated_routes=len(rows),vectors=len(vectors)))
            alg=UAlignedOptimizer(d,rates,budget,refs,weights)
            fit=alg.solve(replica_options={m:[1] for m in models},K=K)
            save(f'cpu/{scenario}_{pool}_MODULE35_FROZEN.json',fit)
            winner=fit['winner']
            record('MODULE35_FROZEN',(winner['deployment'],winner['routing']) if winner else None,fit['seconds'],fit['metrics']['full_prediction_calls'],fit['metrics'])
            for label,native in [('20A_U_GRID_CONTROL',False),('20A_NATIVE_CONTINUOUS_PRESERVED',True)]:
                fit=optimizer(d,rates,budget,refs,weights,K,native=native)
                save(f'cpu/{scenario}_{pool}_{label}.json',fit);winner=fit['winner']
                record(label,(winner['deployment'],winner['routing']) if winner else None,fit['seconds'],fit['objective_calls'],dict(MILP_calls=fit['MILP_calls'],unique_evaluated=fit['unique_evaluated']))
            for label,minimum in [('FASTEST_PATH_STATIC',False),('MIN_MEMORY_COVER_FASTEST',True)]:
                start=time.perf_counter();plan=static(d,rates,minimum,budget)
                record(label,plan,time.perf_counter()-start,4 if plan else 0)

            for arm,v in [('BEST_G_ONLY',pure),('BEST_ACTIVE_MIXED',mixed),('POOL_ORACLE',best)]:
                if v is None:continue
                pred=v['prediction']
                breakdown.append(dict(pool=pool,scenario=scenario,arm=arm,source='CPU_PREDICTION',deployment=v['deployment'],active_specialists=v['active_specialists'],
                    **{k:pred.get(k) for k in ['U','T_norm','L_norm','P_norm','M_norm','Q_contribution','T_contribution','L_contribution','P_contribution','M_contribution']}))
            if mixed:
                a,b=mixed['prediction'],pure['prediction']
                breakdown.append(dict(pool=pool,scenario=scenario,arm='BEST_MIXED_MINUS_G_ONLY',source='CPU_PREDICTION',
                    **{k:a[k]-b[k] for k in ['U','T_norm','L_norm','P_norm','M_norm','Q_contribution','T_contribution','L_contribution','P_contribution','M_contribution']}))

            for m in models:
                if d['models'][m]['role']!='specialist':continue
                positive=choose([v for v in rows if m in v['active_specialists']]);idle=choose([v for v in rows if m in v['idle_specialists']])
                without=choose([v for v in rows if m not in v['deployment']])
                admission.append(dict(pool=pool,scenario=scenario,specialist=m,
                    best_nonzero_U=positive['predicted_U'] if positive else None,
                    best_nonzero_deployment=positive['deployment'] if positive else None,
                    best_nonzero_route_fractions=positive['route_fractions'] if positive else None,
                    best_nonzero_usage={t:positive['route_fractions'][t].get(m,0.) for t in rates} if positive else None,
                    best_idle_U=idle['predicted_U'] if idle else None,best_without_U=without['predicted_U'] if without else None,
                    delta_nonzero_vs_best_without=positive['predicted_U']-without['predicted_U'] if positive and without else None,
                    admitted_by_oracle=m in best['active_specialists'] if best else None,nonzero_candidate_eligible=positive is not None))
                audit.append(dict(pool=pool,scenario=scenario,model=m,audit_type='LEGAL_NONZERO_MIXED_COMPETITOR',
                    capability_admitted=all(t in d['models'][m]['supported_tasks'] for t,row in positive['routing'].items() if row.get(m,0)>0) if positive else False,
                    positive_route_exists=positive is not None,complete_six_task_cover=positive is not None,
                    safe_memory_mib=positive['safe_memory_mib'] if positive else None,budget_mib=budget,
                    resource_legal=positive is not None and positive['safe_memory_mib']<=budget+1e-6,
                    max_rho=positive['max_rho'] if positive else None,nominal_stable=positive is not None,
                    joint_gpu_smoke_pass=smoke.get('passed',False),smoke_scope='Co-residency and serial all-path startup probe; not sustained-load stability or a paired U measurement',
                    passive_exclusion=False if positive else None,eligible_for_optimizer=positive is not None))
    csvsave('cpu_oracle_admission_results.csv',oracles)
    csvsave('optimizer_admission_decisions.csv',decisions)
    csvsave('admission_utility_breakdown.csv',breakdown)
    csvsave('specialist_nonzero_use_audit.csv',admission)
    csvsave('solver_runtime_comparison.csv',runtime)
    csvsave('admission_competitor_feasibility.csv',audit)
    save('cpu_results.json',dict(oracles=oracles,decisions=decisions,nonzero_admission=admission,breakdown=breakdown))
    retention=[v for v in oracles if v['G_only_retained']]
    matches=[v for v in decisions if v['algorithm']=='MODULE35_FROZEN']
    save('CPU_stage_completed.json',dict(cells=len(oracles),oracle_monotone=all(v['oracle_monotone'] for v in oracles),
        retention_cases=[dict(pool=v['pool'],scenario=v['scenario']) for v in retention],
        module35_matches=sum(v['oracle_U_match'] is True for v in matches),module35_total=len(matches),
        module35_feasible_cells=sum(v['oracle_U_match'] is not None for v in matches),
        module35_null_agreements=sum(v.get('oracle_no_plan_match',False) for v in matches),
        max_module35_regret=max((v['regret'] for v in matches if v['regret'] is not None),default=None),GPU_trigger_passed=bool(retention) and smoke.get('passed',False),
        no_algorithm_changes=True,additional_load_search=False,protocol=ref(OUT/'experiment_protocol.json')))


if __name__=='__main__':main()
