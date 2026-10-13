"""Serial bounded GPU validation, frozen configs, no retries or runtime edits."""
import os,time,fcntl
import numpy as np
from deployment_core import Workload,Hardware
from deployment_search.result import jsonable
from integer_replica_pilot.evaluator import runner,summarize,aggregate,ReplicaRuntimeEvaluator,PilotExecutionBlocked
from joint_deploy.measurement.schema import MeasurementConfig
from method_v2 import frozen_config
from .common import *
from .integration import create_search

def preflight():
    verify(read(OUT/'experiment_freeze.json')['protected_references'])
    assert read(OUT/'analytic_and_integration_validation.json')['status']=='PASS'
    from replica_scaling.telemetry import snapshot
    gpu=snapshot();expected=read(HISTORY/'campaign_manifest.json')['GPU']
    for key in ['uuid','name','driver','total_mib']:assert gpu[key]==expected[key],('GPU identity',key)
    assert not gpu['processes'] and gpu['used_mib']<=128,'GPU not isolated'
    assert not any(os.environ.get(k) for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
    import subprocess
    mps=subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True)
    assert not mps.stdout.strip(),'MPS server present'
    sources=[ref(p) for directory in [ROOT/'module9bc',ROOT/'deployment_core/routing',ROOT/'deployment_core/search'] for p in directory.glob('*.py')]
    p=OUT/'execution_freeze.json'
    if p.exists():verify(read(p)['sources'])
    else:save('execution_freeze.json',{'at_utc':now(),'before_first_new_GPU':True,'sources':sources,'GPU':gpu,'MPS_present':False,'max_routing_runs':read(OUT/'routing_gpu_manifest.json')['max_new_runs'],'max_ranking_runs':read(OUT/'ranking_gpu_manifest.json')['max_new_runs'],'failure_policy':'zero retries; invalid execution preserves raw and stops; valid unstable retained'})

def guarded_runner(config,raw_path,log_path,config_path,context_path):
    verify(read(OUT/'execution_freeze.json')['sources'])
    attempt={'raw_path':str(raw_path),'config':config,'context':ref(context_path),'started_at_utc':now(),'status':'STARTED'}
    path=OUT/'gpu_attempts.json';data=read(path) if path.exists() else {'runs':[]};data['runs'].append(attempt);save('gpu_attempts.json',data)
    try:
        outcome=runner(config,raw_path,log_path,config_path,context_path)
        attempt.update(outcome=outcome,status='RETURNED',finished_at_utc=now())
        if raw_path.exists():attempt['raw_sha256']=file_hash(raw_path)
        save('gpu_attempts.json',data);return outcome
    except Exception as exc:
        attempt.update(status='EXCEPTION',exception=repr(exc),finished_at_utc=now());save('gpu_attempts.json',data);raise

def routing_runs():
    catalog,h,roles,cal=inputs();grid=cells();manifest=read(OUT/'routing_gpu_manifest.json');results=[]
    for index,state in enumerate(manifest['states'],1):
        cell=state['cell'];w=Workload(grid[cell]['lambda_rps']);context=HISTORY/'contexts'/f'{cell}.json'
        reps={tag:[] for tag in ['R0','R1']}
        for repetition,seed in enumerate(manifest['seeds'],1):
            for tag in (['R1','R0'] if repetition==2 else ['R0','R1']):
                folder=OUT/'routing_gpu'/f'state_{index}_{cell}'/tag;folder.mkdir(parents=True,exist_ok=True)
                prefix=folder/f'rep_{repetition}';raw=prefix.with_suffix('.json');summary=prefix.with_suffix('.summary.json')
                config=MeasurementConfig(state['deployment'],grid[cell]['lambda_rps'],
                    {k:{m:v for (t,m),v in flat(state[tag+'_routing']).items() if t==k} for k in grid[cell]['lambda_rps']},random_seed=seed,**manifest['settings']).as_dict()
                if summary.exists():
                    rep=read(summary);assert rep['raw_sha256']==file_hash(raw);assert read(prefix.with_suffix('.config.json'))==config
                else:
                    assert not raw.exists() and not prefix.with_suffix('.log').exists(),'Prior incomplete attempt; no retry'
                    save_json(prefix.with_suffix('.config.json'),config)
                    print(json.dumps({'event':'routing_start','state':index,'cell':cell,'routing':tag,'rep':repetition}),flush=True)
                    outcome=guarded_runner(config,raw,prefix.with_suffix('.log'),prefix.with_suffix('.config.json'),context)
                    if not raw.exists():raise PilotExecutionBlocked('No raw; no retry')
                    record=read(raw)
                    assert all(record.get(k)==v for k,v in config.items() if k!='arrival_process'),'Raw identity mismatch'
                    rep=summarize(record,w,catalog,outcome,raw);rep.update(repetition=repetition,random_seed=seed)
                    save_json(summary,rep)
                if not rep['valid_runtime_repetition']:raise PilotExecutionBlocked(rep['run_status'])
                reps[tag].append(rep)
                results.append({'state_index':index,'state_id':state['id'],'cell':cell,'selection_slot':state['selection_slot'],'routing':tag,'repetition':repetition,'seed':seed,
                                'deployment':state['deployment'],'predicted_z':state[tag+'_z'],'J_seconds':state[tag+'_J_seconds'],
                                **labels(rep),'per_task_mean':rep['per_task_mean'],'per_task_p95':rep['per_task_p95'],'raw_path':str(raw)})
                csvsave('routing_gpu_results.csv',results);save('routing_gpu_results.json',{'rows':results})
                print(json.dumps({'event':'routing_complete','cell':cell,'routing':tag,'rep':repetition,**labels(rep)}),flush=True)
        for tag in reps:
            aggregate_path=OUT/'routing_gpu'/f'state_{index}_{cell}'/tag/'aggregate.json'
            save_json(aggregate_path,aggregate(reps[tag],state['id']+'_'+tag,aggregate_path,state['deployment']))
    routing_gate(results,manifest)

def routing_gate(rows,manifest):
    comparisons=[]
    for index,state in enumerate(manifest['states'],1):
        a=sorted([r for r in rows if r['state_index']==index and r['routing']=='R0'],key=lambda r:r['repetition']);b=sorted([r for r in rows if r['state_index']==index and r['routing']=='R1'],key=lambda r:r['repetition'])
        assert len(a)==len(b)==3
        ta=min((r['stability'] for r in a),key=lambda t:TIER[t]);tb=min((r['stability'] for r in b),key=lambda t:TIER[t])
        matched=[(r,s) for r,s in zip(a,b) if r['stability']==s['stability']=='steady']
        differences=[r['mean_e2e_ms']-s['mean_e2e_ms'] for r,s in matched]
        comparisons.append({'state_index':index,'cell':state['cell'],'selection_slot':state['selection_slot'],'R0_tier':ta,'R1_tier':tb,'tier_change':TIER[tb]-TIER[ta],
                            'R0_mean_ms':float(np.mean([r['mean_e2e_ms'] for r in a])),'R1_mean_ms':float(np.mean([r['mean_e2e_ms'] for r in b])),
                            'matched_steady_repetitions':len(matched),'paired_steady_improvement_ms':float(np.mean(differences)) if differences else None,
                            'all_three_steady_improve':len(differences)==3 and all(d>0 for d in differences),
                            'paired_steady_differences_ms':differences,
                            'per_task_mean_change_ms':{k:float(np.mean([s['per_task_mean'][k]-r['per_task_mean'][k] for r,s in zip(a,b)])) for k in a[0]['per_task_mean']}})
    degradations=sum(r['tier_change']<0 for r in comparisons);improvements=sum(r['tier_change']>0 for r in comparisons)
    gains=[r['paired_steady_improvement_ms'] for r in comparisons if r['paired_steady_improvement_ms'] is not None]
    worse=sum(g<0 for g in gains);better=sum(g>0 for g in gains)
    frequent_instability=sum(r['R0_tier']=='steady' and r['R1_tier']!='steady' for r in comparisons)>=2
    gate='ROUTING_NOT_SUPPORTED' if degradations>improvements or worse>better or frequent_instability else 'ROUTING_SUPPORTED' if degradations==0 and gains and sum(gains)>=0 and any(r['all_three_steady_improve'] for r in comparisons) else 'ROUTING_PARTIAL'
    save('routing_gate.json',{'gate':gate,'comparisons':comparisons,'stability_degradations':degradations,'stability_improvements':improvements,'matched_steady_states_better':better,'matched_steady_states_worse':worse,'equal_state_mean_steady_improvement_ms':float(np.mean(gains)) if gains else None,'rule':read(OUT/'experiment_freeze.json')['routing_gate_rule'],'magnitude_threshold':None,'new_GPU_runs':len(rows),'statistical_limit':'three independent repeats per state; finite development states; no general-runtime queueing accuracy claim'})
    print(json.dumps({'event':'routing_gate','gate':gate}),flush=True)

def ranking_runs():
    catalog,h,roles,cal=inputs();grid=cells();manifest=read(OUT/'ranking_gpu_manifest.json');protocol=read(PROTOCOL);factor=read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor'];results=[]
    for cell in manifest['selected_cells']:
        state=grid[cell];w=Workload(state['lambda_rps'])
        for arm in ['V2_delta_z','V3_delta_J']:
            folder=OUT/'ranking_gpu'/cell/arm;folder.mkdir(parents=True,exist_ok=True)
            complete=folder/'search_complete.json'
            if complete.exists():snapshot=read(complete)
            else:
                evaluator=ReplicaRuntimeEvaluator(folder/'cache',catalog,HISTORY/'contexts'/f'{cell}.json',protocol['measurement_settings'],seeds=protocol['search_seeds'],max_unique=6,run=guarded_runner)
                def checkpoint(result):save_json(folder/'search_trace.json',result.to_dict())
                engine=create_search(evaluator,catalog,cal,factor,ranking='delta_z' if arm=='V2_delta_z' else 'delta_J',trace_callback=checkpoint)
                result=engine.search(w,catalog,Hardware(state['planning_budget_input_mib']),frozen_config(),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                snapshot=result.to_dict();save_json(complete,snapshot)
            assert snapshot['runtime_evaluation_count']<=6 and snapshot['search_iterations']<=3
            assert snapshot['seed']['deployment']=={m:state['preflight']['static_v2']['deployment'][m] for m in catalog}
            row={'cell':cell,'arm':arm,'deployment':snapshot['deployment'],'runtime_evaluations':snapshot['runtime_evaluation_count'],'iterations':snapshot['search_iterations'],'status':snapshot['status'],**labels(snapshot['runtime'])}
            results.append(row);csvsave('ranking_gpu_results.csv',results);save('ranking_gpu_results.json',{'rows':results})
            print(json.dumps({'event':'ranking_search_complete',**row}),flush=True)
    if not results:
        csvsave('ranking_gpu_results.csv',[],['cell','arm','status']);save('ranking_gpu_results.json',{'rows':[],'status':'NO_MATERIAL_RANKING_DIFFERENCE'})
    ranking_gate(results)

def ranking_gate(rows):
    metrics=read(OUT/'candidate_ranking_metrics.json');comparisons=[]
    from deployment_search.real_objective import NoiseAwareRuntimeObjective
    from types import SimpleNamespace
    c,h,roles,cal=inputs();objective=NoiseAwareRuntimeObjective(1.,.05)
    for cell in read(OUT/'ranking_gpu_manifest.json')['selected_cells']:
        a=read(OUT/'ranking_gpu'/cell/'V2_delta_z'/'search_complete.json');b=read(OUT/'ranking_gpu'/cell/'V3_delta_J'/'search_complete.json')
        am=sum(cal.memory_table_mib[m][n] for m,n in a['deployment'].items());bm=sum(cal.memory_table_mib[m][n] for m,n in b['deployment'].items())
        better,why=objective.compare(SimpleNamespace(**b['runtime']),SimpleNamespace(**a['runtime']),bm,am)
        worse,reverse=objective.compare(SimpleNamespace(**a['runtime']),SimpleNamespace(**b['runtime']),am,bm)
        comparisons.append({'cell':cell,'V3_better':better,'V3_worse':worse,'reason':why,'reverse_reason':reverse,'same_final_deployment':a['deployment']==b['deployment'],'mean_improvement_ms':a['runtime']['weighted_mean_e2e']-b['runtime']['weighted_mean_e2e'],'V2_budget_used':a['runtime_evaluation_count'],'V3_budget_used':b['runtime_evaluation_count']})
    old=metrics['metrics']['delta_z']['mean_improvement_ms']['Spearman'];new=metrics['metrics']['delta_J_seconds']['mean_improvement_ms']['Spearman']
    topgain=metrics['new']['Top2_hit_rate']-metrics['old']['Top2_hit_rate']
    withinold=metrics['within_iteration_ordering']['delta_z']['accuracy'];withinnew=metrics['within_iteration_ordering']['delta_J_seconds']['accuracy']
    systematic_sanity_worse=bool(comparisons) and all(r['V3_worse'] for r in comparisons)
    consistently_worse=new<old and topgain<=0 and withinnew<withinold
    conflict=withinnew<withinold or sum(r['V3_worse'] for r in comparisons)>0
    gate='RANKING_NOT_SUPPORTED' if systematic_sanity_worse or consistently_worse else 'RANKING_SUPPORTED' if (new>old or topgain>0) and not conflict else 'RANKING_PARTIAL'
    save('ranking_gate.json',{'gate':gate,'comparisons':comparisons,'pooled_Spearman_gain':new-old,'Top2_hit_gain':topgain,'within_iteration_ordering_gain':withinnew-withinold,'rule':read(OUT/'experiment_freeze.json')['ranking_gate_rule'],'unknown_candidate_labels_imputed':False,'evidence_limit':'measured-only replay is selection-biased; two bounded searches per arm, no extra confirmations; mixed ordering evidence remains PARTIAL','magnitude_threshold':None})
    print(json.dumps({'event':'ranking_gate','gate':gate}),flush=True)

def main():
    lock=(OUT/'gpu_execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight();save('campaign_status.json',{'status':'RUNNING','phase':'routing','at_utc':now()})
    try:
        routing_runs();save('campaign_status.json',{'status':'RUNNING','phase':'ranking','at_utc':now()});ranking_runs()
        verify(read(OUT/'experiment_freeze.json')['protected_references']);save('campaign_status.json',{'status':'GPU_VALIDATION_COMPLETE','at_utc':now()})
    except Exception as exc:
        save('campaign_status.json',{'status':'STOPPED_INVALID_EXECUTION','error':repr(exc),'at_utc':now(),'retries':0});raise
if __name__=='__main__':main()
