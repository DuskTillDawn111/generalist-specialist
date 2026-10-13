import itertools
import numpy as np
from dataclasses import replace
from scipy.optimize import linprog
from deployment_core import Workload,Hardware
from deployment_search.routing import solve_pressure_relaxation
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from method_v2 import MethodV2,frozen_config
from module9bc.frozen_v3 import create as v3_create,snapshot as v3_snapshot
from module8f2_resource_static.planning import rc_sg
from .common import *
from .envelope import solve_envelope

def flat(r):return {tuple(k.split('::')):v for k,v in r.items()}
def serial(r):return {'::'.join(k):v for k,v in r.items()}

def fixed_capacity(x,background,target,catalog,roles,cal,only,ceiling):
    paths=[(k,m) for k in sorted(background) for m in sorted(catalog) if x[m] and k in catalog[m].supported_tasks and not(only and k==target and roles[m]=='generalist')]
    eq=[[float(t==k) for t,m in paths] for k in sorted(background) if k!=target]
    values=[background[k] for k in sorted(background) if k!=target]
    ub=[[catalog[m].service_time_ms[k]/1000 if model==m else 0 for k,model in paths] for m in sorted(catalog)]
    cap=[ceiling*cal.capacity_factors[m][x[m]] for m in sorted(catalog)]
    result=linprog([-float(k==target) for k,m in paths],A_eq=eq,b_eq=values,A_ub=ub,b_ub=cap,bounds=(0,None),method='highs')
    if result.status==2:return None
    assert result.success,result.message
    return -float(result.fun)

def main():
    assert not (OUT/'gpu_manifest.json').exists(),'GPU plan already frozen'
    c,h,roles,cal=inputs();primary=read(OUT/'primary_hotspot_selection.json');points=read(OUT/'hotspot_points.json')['points'];context=primary['context'];target=primary['target'];B=context['memory_budget_mib']
    assert primary['replica_audit']['status']=='PASS'
    rows=read(OUT/'capacity_envelope.json')['rows'];checks={(r['context'],r['target']):r['status'] for r in read(OUT/'replica_capacity_audit.json')['contexts']}
    for r in rows:
        status=checks[(r['context'],r['target'])];r.update(replica_audit_status=status,envelope_certified=status=='PASS',certified_capacity_rps=r['capacity_rps'] if status=='PASS' else None)
    save('capacity_envelope.json',{'rows':rows});csvsave('capacity_envelope.csv',rows)
    # Independent enumeration + LP corroborates the primary MILP optimum and
    # rules out unexamined evidence-supported Specialist identities/counts.
    models=sorted(c);deployments=[];maxima={('S_ONLY',1.):None,('ALL',1.):None,('ALL',.8):None}
    for counts in itertools.product(*(range(cal.evidence_bounds[m]+1) for m in models)):
        x=dict(zip(models,counts));memory=factor()*sum(cal.memory_table_mib[m][n] for m,n in x.items())
        if memory>B+1e-6:continue
        if any(not any(x[m] and k in c[m].supported_tasks for m in models) for k,v in context['background_lambda'].items() if v>0):continue
        deployments.append(x)
        for mode,ceiling in maxima:
            value=fixed_capacity(x,context['background_lambda'],target,c,roles,cal,mode=='S_ONLY',ceiling)
            if value is not None:maxima[(mode,ceiling)]=max(maxima[(mode,ceiling)] or 0,value)
    assert abs(maxima[('S_ONLY',1.)]-primary['C_Sonly'])<1e-7
    assert abs(maxima[('ALL',1.)]-primary['C_ALL'])<1e-7
    for mode in ['S_ONLY','ALL']:
        repeat=solve_envelope(c,roles,cal,B,context['background_lambda'],target,mode=='S_ONLY',factor())
        witness=primary['specialist_only_witness' if mode=='S_ONLY' else 'all_model_witness']
        assert repeat['deployment']==witness['deployment'] and abs(repeat['capacity_rps']-witness['capacity_rps'])<1e-7
    save('cpu_validation.json',{'status':'PASS','known_count_space':math.prod(cal.evidence_bounds[m]+1 for m in c),'primary_memory_and_coverage_feasible_deployments':len(deployments),
                              'independent_enumeration_LP_optima':{f'{m}_rho{rho}':v for (m,rho),v in maxima.items()},'repeated_MILP_deployment_and_capacity_match':True,
                              'all_envelope_MILP_gap_zero':all(all(s['mip_gap'] is None or s['mip_gap']<1e-7 for s in r['solver_stages']) for r in rows),
                              'method_rho_max_unchanged':.8,'new_GPU_runs':0})
    planning=[];routes={};newstates=[];mechanism=[]
    for point in points:
        name=point['point'];w=Workload(point['lambda_rps']);diagnostic=solve_pressure_relaxation(primary['all_model_witness']['deployment'],w,c,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
        all_feasible=diagnostic is not None and diagnostic.max_pressure<=1+1e-8
        s_feasible=point['target_rate']<=primary['C_Sonly']+1e-8
        exhausted=primary['replica_audit']['status']=='PASS' and point['target_rate']>primary['C_Sonly'] and not s_feasible and all_feasible
        mechanism.append({'point':name,'target_rate':point['target_rate'],'S_only_planning_feasible_rho1':s_feasible,'ALL_planning_feasible_rho1':all_feasible,
                          'best_ALL_max_rho':diagnostic.max_pressure if diagnostic else None,'SPECIALIST_EXHAUSTED':exhausted,'claim_scope':'certified frozen empirical planning envelope; real runtime extension must be measured separately',
                          'frozen_algorithm_global_target_ceiling_rps':maxima[('ALL',.8)]})
        if name=='H4':continue
        rc=rc_sg(point['lambda_rps'],B,c,roles,cal,factor(),models)
        save(f'planning/{name}_RC_SG.json',rc)
        for method in ['RC_SG','v2','V3_MINIMAL']:
            if method=='RC_SG':
                data=rc;x=rc['deployment'];routing=rc.get('canonical_routing_rps');status=rc['status'];trace=None
            else:
                captured=[]
                def checkpoint(result):captured.append(result)
                engine=MethodV2(trace_callback=checkpoint) if method=='v2' else v3_create(None,c,cal,trace_callback=checkpoint)
                try:result=engine.search(w,c,Hardware(context['planning_budget_input_mib']),replace(frozen_config(),dry_run=True),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                except RuntimeError as exc:
                    if method!='V3_MINIMAL' or 'J solve failed' not in str(exc):raise
                    from deployment_core.routing.e2e_router import solve_e2e_routing
                    failed=captured[-1];x=failed.deployment
                    fit=solve_e2e_routing(x,w,c,cal)
                    assert not fit['converged'] and fit['status']=='NUMERICAL_SOLVER_FAILURE',fit
                    save(f'planning/{name}_{method}.json',{'status':'MODULE9_J_NUMERICAL_FAILURE','exception':repr(exc),'frozen_source_not_modified':True,'partial_search_trace':failed.to_dict(),'frozen_J_solver_diagnostic':fit})
                    planning.append({'point':name,'method':method,'target_rate':point['target_rate'],'status':'MODULE9_J_NUMERICAL_FAILURE','planning_feasible':False,
                                     'deployment':x,'routing':None,'state_id':None,'canonical_capacity_feasible':True,'optimality_gap_seconds':fit['optimality_gap_seconds'],
                                     'solver_constraints_valid':fit['constraints_valid'],'ALL_rho1_feasible':all_feasible,'S_only_rho1_feasible':s_feasible,'SPECIALIST_EXHAUSTED':exhausted})
                    continue
                data=result.to_dict() if method=='v2' else v3_snapshot(engine,result,w)
                save(f'planning/{name}_{method}.json',data)
                x=result.deployment;status=result.status;trace=result.to_dict();routing=serial(result.canonical.routing) if result.canonical else None
                if x and result.canonical and method=='V3_MINIMAL':
                    fit=engine.serving_routing_adapter.ranker.evaluate(x,w);assert fit['converged'],fit;routing=serial(fit['routing'])
                if result.canonical:assert all(not t['selected_top_b'] for t in trace['iterations']),'Unexpected bounded search needs a pre-frozen expanded GPU plan'
            ready=x is not None and routing is not None
            if ready:
                metrics=evaluate_surrogate(x,flat(routing),w,c,cal);tg=sum(v for p,v in routing.items() if p.startswith(target+'::') and roles[p.split('::')[1]]=='generalist');ts=point['target_rate']-tg
                safe=factor()*sum(cal.memory_table_mib[m][x.get(m,0)] for m in c)
                state_id=f'{name}_'+('J' if method=='V3_MINIMAL' else 'canonical')
                if not any(s['state_id']==state_id for s in newstates):newstates.append({'state_id':state_id,'point':name,'deployment':x,'routing':routing,'lambda_rps':point['lambda_rps'],'methods':[method]})
                else:next(s for s in newstates if s['state_id']==state_id)['methods'].append(method)
                routes[name,method]=routing
            else:metrics={};tg=ts=safe=None;state_id=None
            planning.append({'point':name,'method':method,'target_rate':point['target_rate'],'status':status,'planning_feasible':ready,'deployment':x,'routing':routing,
                             'state_id':state_id,'N_G':sum(x.get(m,0) for m in c if roles[m]=='generalist') if x else None,'N_S':sum(x.get(m,0) for m in c if roles[m]=='specialist') if x else None,
                             'specialist_replicas':{m:x.get(m,0) for m in c if roles[m]=='specialist' and x.get(m,0)} if x else None,
                             'target_S_rps':ts,'target_G_rps':tg,'f_S':ts/point['target_rate'] if ts is not None else None,'f_G':tg/point['target_rate'] if tg is not None else None,
                             'rho':metrics.get('rho'),'max_rho':metrics.get('z'),'J_seconds':metrics.get('J_seconds'),'safe_memory_mib':safe,
                             'ALL_rho1_feasible':all_feasible,'S_only_rho1_feasible':s_feasible,'SPECIALIST_EXHAUSTED':exhausted})
    csvsave('planning_results.csv',planning);save('planning_results.json',{'rows':planning});save('mechanism_planning_checks.json',{'points':mechanism})
    protocol=read(ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json')
    manifest={'frozen':True,'at_utc':now(),'primary_selection':ref(OUT/'primary_hotspot_selection.json'),'hotspot_points':ref(OUT/'hotspot_points.json'),
              'methods':['RC_SG','v2','V3_MINIMAL'],'states':newstates,'settings':protocol['measurement_settings'],'seeds':protocol['search_seeds'],
              'max_timed_runs':3*len(newstates),'method_logical_evaluation_budget':6,'repetitions_per_final_state':3,
              'exact_state_reuse':'RC-SG and v2 share exact canonical deployment/routing/config/seeds; observations are shared, never counted as independent method evidence.',
              'routing_order':'per point repetitions 1/3 canonical then J; repetition2 J then canonical',
              'infeasible_points':'zero GPU runs; keep frozen rates and frozen rho_max=.8. Do not substitute a rho=1 oracle state as a method result.',
              'context_source':ref(ROOT/'results/module8f2_resource_static/contexts/C01.json'),'execution_environment_overrides':protocol['execution_environment_overrides'],
              'new_GPU_before_this_freeze':0,'no_extra_confirmations':True}
    save('gpu_manifest.json',manifest)
    save('benchmark_status.json',{'status':'GPU_PLAN_FROZEN','at_utc':now()})
    print(json.dumps({'method_capacity_ceiling':maxima[('ALL',.8)],'feasible_method_points':[(r['point'],r['method']) for r in planning if r['planning_feasible']],
                      'new_exact_states':len(newstates),'max_timed_runs':manifest['max_timed_runs']}),flush=True)
if __name__=='__main__':main()
