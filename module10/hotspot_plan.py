"""Freeze the authorized operational 10B states before any new GPU label."""
from dataclasses import replace
import math
from decimal import Decimal, localcontext
from deployment_core import Workload, Hardware
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from deployment_core.routing.e2e_router import solve_e2e_routing
from deployment_search.runtime_cache import digest
from method_v2 import MethodV2, frozen_config
from module9bc.frozen_v3 import create, snapshot
from module8f2_resource_static.planning import rc_sg
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator
from .common import ROOT, OUT, inputs, read, save, ref, verify, factor, csvsave, now
from .operational import AUDIT, ARCHIVE, check_archive
from .planning import serial, flat

CERTIFICATE_TOLERANCE_SECONDS=1e-8

def operational_certificate(x,routing,lam,c,cal):
    evaluation=evaluate_surrogate(x,flat(routing),Workload(lam),c,cal)
    with localcontext() as ctx:
        ctx.prec=60
        D={m:sum((Decimal(str(v))*Decimal(str(c[m].service_time_ms[k]))/1000
             for (k,n),v in flat(routing).items() if n==m and v!=0),Decimal(0)) for m,n in x.items() if n}
        residual={m:D[m]-Decimal('.8')*Decimal(str(cal.capacity_factors[m][x[m]])) for m in D}
        # Small positive residuals must be both within the fixed 10 ns envelope
        # and attributable to float representation, rather than real overload.
        roundoff={m:Decimal(str(32*math.ulp(max(1.,float(D[m]))))) for m in D}
        passed=all(v<=0 or v<=min(Decimal('1e-8'),roundoff[m]) for m,v in residual.items())
    return {'status':'PASS' if passed else 'OPERATIONAL_ROUTE_INFEASIBLE',
        'passed':passed,'rho_max':.8,'certificate_tolerance_seconds':CERTIFICATE_TOLERANCE_SECONDS,
        'per_worker_rho':evaluation['rho'],'D':evaluation['D'],
        'capacity_residual_seconds_decimal':{m:str(v) for m,v in residual.items()},
        'float_roundoff_bound_seconds':{m:str(v) for m,v in roundoff.items()},
        'evaluation':evaluation,'true_violating_workers':[m for m,v in residual.items() if v>min(Decimal('1e-8'),roundoff[m])],
        'scientific_threshold_or_rho_max_changed':False}

def main():
    assert not (OUT/'gpu_manifest.json').exists(),'Campaign already frozen'
    assert read(AUDIT/'completion_verification.json')['status']=='PASS'
    verify(read(AUDIT/'input_freeze.json')['protected_references']);check_archive()
    c,h,roles,cal=inputs();primary=read(AUDIT/'primary_hotspot_selection.json');con=primary['context']
    assert con['id']=='B1_I1' and primary['target']=='asr'
    points=read(AUDIT/'hotspot_points.json')['points'][:4]
    protocol=read(ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json')
    for name,data in [('primary_hotspot_selection.json',primary),('hotspot_points.json',read(AUDIT/'hotspot_points.json'))]:
        save(name,data)
    save('input_freeze.json',{'at_utc':now(),'authorized_scope':'Module10B operational hotspot campaign',
        'protected_references':read(AUDIT/'input_freeze.json')['protected_references'],
        'operational_audit_manifest':ref(AUDIT/'audit_output_manifest.json'),
        'rho_max':.8,'certificate_tolerance_seconds':CERTIFICATE_TOLERANCE_SECONDS,
        'settings':protocol['measurement_settings'],'seeds':protocol['search_seeds'],
        'old_sweep_not_reused_as_benchmark':True,'old_archive_original_files':read(ARCHIVE/'archive_manifest.json')['original_files']})
    # identity() performs only CPU schema/memory validation. This cache directory
    # is fresh and never populated with labels from the invalid sweep.
    auditor=ReplicaRuntimeEvaluator(OUT/'gpu',c,ROOT/'results/module8f2_resource_static/contexts/C01.json',
        protocol['measurement_settings'],seeds=protocol['search_seeds'],max_unique=24)
    rows=[];states=[];aliases=[]
    for point in points:
        w=Workload(point['lambda_rps']);B=con['memory_budget_mib']
        rc=rc_sg(point['lambda_rps'],B,c,roles,cal,factor(),sorted(c))
        save(f'planning/{point["point"]}_RC_SG.json',rc)
        for method in ['RC_SG','v2','V3_MINIMAL']:
            trace=None;fit=None;routing=None
            if method=='RC_SG':
                x=rc['deployment'];routing=rc.get('canonical_routing_rps');status=rc['status']
            else:
                captured=[];engine=MethodV2(trace_callback=lambda r:captured.append(r)) if method=='v2' else create(None,c,cal,trace_callback=lambda r:captured.append(r))
                try:
                    result=engine.search(w,c,Hardware(con['planning_budget_input_mib']),replace(frozen_config(),dry_run=True),
                        roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                    trace=result.to_dict();x=result.deployment;status=result.status
                    assert all(not t['selected_top_b'] for t in trace['iterations']), 'Unexpected runtime search candidates: freeze a full budget before proceeding'
                    routing=serial(result.canonical.routing) if result.canonical else None
                    if method=='V3_MINIMAL' and routing is not None:
                        fit=engine.e2e_ranker.evaluate(x,w)
                        routing=serial(fit['routing']) if fit['converged'] else None
                        trace=snapshot(engine,result,w)
                except RuntimeError as exc:
                    if method!='V3_MINIMAL' or 'J solve failed' not in str(exc):raise
                    x=captured[-1].deployment;fit=solve_e2e_routing(x,w,c,cal);status='MODULE9_J_NUMERICAL_FAILURE'
                    trace={'exception':repr(exc),'partial_search_trace':captured[-1].to_dict()}
                save(f'planning/{point["point"]}_{method}.json',{'status':status,'trace':trace,'J_solver':fit})
            certificate=operational_certificate(x,routing,point['lambda_rps'],c,cal) if routing is not None else None
            ready=certificate is not None and certificate['passed'];state_id=None
            if certificate is not None and not certificate['passed']:status='OPERATIONAL_ROUTE_INFEASIBLE'
            safe=factor()*math.fsum(cal.memory_table_mib[m][x.get(m,0)] for m in c) if x else None
            assert safe is None or safe<=B+1e-8
            tg=math.fsum(v for p,v in (routing or {}).items() if p.startswith('asr::') and roles[p.split('::')[1]]=='generalist') if routing is not None else None
            identity=auditor.identity(x,w,flat(routing)) if ready else None
            if ready:
                key=digest(identity)
                existing=next((s for s in states if s['exact_identity_sha256']==key),None)
                if existing:
                    assert existing['identity']==identity;existing['methods'].append(method);state_id=existing['state_id']
                    aliases.append({'point':point['point'],'methods':list(existing['methods']),'exact_identity_sha256':key,
                        'exact_config_and_context_equal':True,'float_routes_not_rounded_for_reuse':True})
                else:
                    state_id=f'{point["point"]}_'+('J' if method=='V3_MINIMAL' else 'canonical')
                    states.append({'state_id':state_id,'point':point['point'],'methods':[method],
                        'deployment':x,'routing':routing,'lambda_rps':point['lambda_rps'],'target_rate':point['target_rate'],
                        'operational_certificate':certificate,'identity':identity,'exact_identity_sha256':key})
            row={'point':point['point'],'method':method,'target_rate':point['target_rate'],'status':status,
                'planning_feasible':ready,'state_id':state_id,'deployment':x,'routing':routing,
                'operational_certificate':certificate,'J_solver_gap_seconds':fit.get('optimality_gap_seconds') if fit else None,
                'S_only_operational_feasible':point['target_rate']<=primary['C_S_op'],
                'SPECIALIST_EXHAUSTED_OPERATIONAL':point['target_rate']>primary['C_S_op'] and point['target_rate']<primary['C_A_op'],
                'f_G':tg/point['target_rate'] if tg is not None else None,'f_S':1-tg/point['target_rate'] if tg is not None else None,
                'target_G_rps':tg,'target_S_rps':point['target_rate']-tg if tg is not None else None,
                'rho':certificate['per_worker_rho'] if certificate else None,
                'max_rho':certificate['evaluation']['z'] if certificate else None,
                'J_seconds':certificate['evaluation']['J_seconds'] if certificate else None,
                'safe_memory_mib':safe,'N_G':sum(x.get(m,0) for m in c if roles[m]=='generalist') if x else None,
                'N_S':sum(x.get(m,0) for m in c if roles[m]=='specialist') if x else None,
                'replicas':x,'specialist_replicas':{m:n for m,n in (x or {}).items() if n and roles[m]=='specialist'},
                'failed_route_retained_as_diagnostic_not_executed':routing is not None and not ready}
            rows.append(row)
            print(__import__('json').dumps({k:row[k] for k in ['point','method','status','planning_feasible','f_S','f_G','max_rho']}),flush=True)
    save('planning_results.json',{'rows':rows});csvsave('planning_results.csv',rows)
    save('state_identity_audit.json',{'aliases':aliases,'unique_exact_states':len(states),
        'old_invalid_sweep_reused':False,'label_aliases_not_independent_observations':True,
        'identity_fields':list(states[0]['identity']),'new_GPU_before_identity_audit':0})
    assert len(states)==7 and sum(r['planning_feasible'] for r in rows)==11
    manifest={'frozen':True,'at_utc':now(),'operational_primary_source':ref(AUDIT/'primary_hotspot_selection.json'),
        'operational_hotspot_points_source':ref(AUDIT/'hotspot_points.json'),'planning_results':ref(OUT/'planning_results.json'),
        'state_identity_audit':ref(OUT/'state_identity_audit.json'),'states':states,'max_timed_runs':3*len(states),
        'repetitions_per_final_state':3,'seeds':protocol['search_seeds'],'settings':protocol['measurement_settings'],
        'context_source':ref(ROOT/'results/module8f2_resource_static/contexts/C01.json'),
        'execution_environment_overrides':protocol['execution_environment_overrides'],
        'routing_gate':'rho<=.8; fixed 10ns numerical tolerance; genuine overload is not executed',
        'invalid_execution_policy':'Preserve failure and stop; zero outcome retries or substitute routes',
        'search_budget_per_method_cell':6,'actual_search_candidates_all_empty':True,
        'launch_order':'H0 canonical then J; H1 J then canonical; H2 canonical then J; H3 canonical',
        'methods':['RC_SG','v2','V3_MINIMAL'],'old_archive_unchanged':True,'new_GPU_results_seen':0}
    states.sort(key=lambda s:(int(s['point'][1:]), (s['state_id'].endswith('_J')) ^ (s['point']=='H1')))
    save('gpu_manifest.json',manifest)
    save('benchmark_status.json',{'status':'OPERATIONAL_GPU_PLAN_FROZEN','at_utc':now(),'max_timed_runs':manifest['max_timed_runs']})
    print('Frozen physical run budget:',manifest['max_timed_runs'],flush=True)

if __name__=='__main__':main()
