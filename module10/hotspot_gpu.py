"""Execute the frozen operational states through the unchanged GPU harness."""
import os,fcntl,subprocess
from dataclasses import asdict
from deployment_core import Workload,Hardware
from method_v2 import MethodV2,frozen_config
from module9bc.frozen_v3 import create,snapshot as v3_snapshot
from integer_replica_pilot.evaluator import runner,ReplicaRuntimeEvaluator,summarize
from deployment_search.runtime_cache import digest
from replica_scaling.telemetry import snapshot
from .common import ROOT,OUT,inputs,read,save,csvsave,ref,verify,now
from .operational import AUDIT,check_archive
from .hotspot_plan import operational_certificate
from .planning import flat

def guarded_runner(config,raw_path,log_path,config_path,context_path):
    execution=read(OUT/'execution_freeze.json');verify(execution['sources'])
    manifest=read(OUT/'gpu_manifest.json');c,h,roles,cal=inputs()
    routing={f'{k}::{m}':v for k,row in config['routing_rps'].items() for m,v in row.items()}
    certificate=operational_certificate(config['deployment'],routing,config['workload_rps'],c,cal)
    assert certificate['passed'],certificate
    state=next(s for s in manifest['states'] if s['identity']['deployment']==config['deployment'] and
        s['identity']['workload']==config['workload_rps'] and s['identity']['routing']==config['routing_rps'])
    assert config['random_seed'] in manifest['seeds'] and ref(context_path)==manifest['context_source']
    assert not snapshot()['processes'],'GPU is occupied before launch'
    ledger=read(OUT/'gpu_attempts.json') if (OUT/'gpu_attempts.json').exists() else {'runs':[]}
    assert len(ledger['runs'])<manifest['max_timed_runs']
    assert not any(r['state_id']==state['state_id'] and r['seed']==config['random_seed'] for r in ledger['runs']), 'No duplicate attempt'
    entry={'started_at_utc':now(),'state_id':state['state_id'],'methods':state['methods'],
        'seed':config['random_seed'],'raw_path':str(raw_path),'config':config,'context':ref(context_path),
        'operational_certificate':certificate,'status':'STARTED'}
    ledger['runs'].append(entry);save('gpu_attempts.json',ledger)
    outcome=runner(config,raw_path,log_path,config_path,context_path)
    entry.update(status='RETURNED',outcome=outcome,finished_at_utc=now(),raw_ref=ref(raw_path) if raw_path.exists() else None)
    save('gpu_attempts.json',ledger)
    if raw_path.exists():
        raw=read(raw_path)
        entry['summary']=summarize(raw,Workload(config['workload_rps']),c,outcome,raw_path)
        save('gpu_attempts.json',ledger)
    print(__import__('json').dumps({'event':'guarded_return','state':state['state_id'],'seed':config['random_seed'],
        'elapsed_seconds':outcome['elapsed_seconds'],'valid':entry.get('summary',{}).get('valid_runtime_repetition'),
        'stability':entry.get('summary',{}).get('stability_state_v2'),'mean_ms':entry.get('summary',{}).get('weighted_mean_e2e')}),flush=True)
    return outcome

def main():
    assert read(OUT/'benchmark_status.json')['status']=='OPERATIONAL_GPU_PLAN_FROZEN', 'No campaign relaunch'
    lock=(OUT/'gpu_execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'gpu_manifest.json');verify(read(OUT/'input_freeze.json')['protected_references']);check_archive()
    verify([manifest[k] for k in ['operational_primary_source','operational_hotspot_points_source','planning_results','state_identity_audit','context_source']])
    gpu=snapshot();original=read(ROOT/'results/module8f2_resource_static/campaign_manifest.json')['GPU']
    assert all(gpu[k]==original[k] for k in ['uuid','name','driver','total_mib'])
    assert not gpu['processes'] and gpu['used_mib']<=128
    assert not subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True).stdout.strip()
    assert not any(os.environ.get(k) for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
    for k,v in manifest['execution_environment_overrides'].items():os.environ[k]=v
    sources=[ref(p) for p in (ROOT/'module10').glob('*.py')]+[ref(OUT/'gpu_manifest.json')]
    save('execution_freeze.json',{'at_utc':now(),'sources':sources,'GPU':gpu,'no_MPS':True,
        'no_runtime_worker_edits':True,'max_timed_runs':manifest['max_timed_runs'],'settings':manifest['settings'],
        'context_source':manifest['context_source'],'zero_outcome_retries':True})
    save('benchmark_status.json',{'status':'OPERATIONAL_GPU_RUNNING','at_utc':now(),'max_timed_runs':manifest['max_timed_runs']})
    c,h,roles,cal=inputs();con=read(OUT/'primary_hotspot_selection.json')['context'];results=[];aggregates=[]
    evaluator=ReplicaRuntimeEvaluator(OUT/'gpu',c,manifest['context_source']['path'],manifest['settings'],
        seeds=manifest['seeds'],max_unique=len(manifest['states']),run=guarded_runner)
    try:
        for state in manifest['states']:
            w=Workload(state['lambda_rps'])
            assert evaluator.identity(state['deployment'],w,flat(state['routing']))==state['identity']
            actual=evaluator.evaluate(state['deployment'],w,flat(state['routing']))
            for method in state['methods']:
                if method=='RC_SG':continue
                engine=MethodV2(evaluator) if method=='v2' else create(evaluator,c,cal)
                result=engine.search(w,c,Hardware(con['planning_budget_input_mib']),frozen_config(),
                    roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
                assert result.deployment==state['deployment'] and result.runtime_evaluation_count==1
                assert all(not t['selected_top_b'] for t in result.iterations)
                save(f'gpu/{state["state_id"]}_{method}_search_complete.json',result.to_dict() if method=='v2' else v3_snapshot(engine,result,w))
            save(f'gpu/{state["state_id"]}_aggregate.json',asdict(actual))
            for index,rep in enumerate(actual.repetitions,1):
                results.append({'state_id':state['state_id'],'point':state['point'],'methods_sharing_exact_label':state['methods'],
                    'repetition':index,'seed':rep['random_seed'],'deployment':state['deployment'],'routing':state['routing'],
                    'stability':rep['stability_state_v2'],'stability_reason':rep['stability_reason_v2'],
                    'mean_e2e_ms':rep['weighted_mean_e2e'],'weighted_p95_ms':rep['weighted_p95_e2e'],
                    'per_task_mean':rep['per_task_mean'],'per_task_p95':rep['per_task_p95'],
                    'completion_ratio':rep['completion_ratio'],'throughput':rep['throughput'],
                    'queue_metrics':rep['queue_metrics'],'memory_observations':rep['memory_observations'],
                    'raw_path':rep['raw_result_path'],'raw_sha256':rep['raw_sha256'],
                    'operational_certificate':state['operational_certificate']})
            aggregates.append({'state_id':state['state_id'],'methods':state['methods'],'runtime':asdict(actual)})
            save('gpu_results.json',{'rows':results,'independent_physical_measurements':len(results),
                'exact_state_aliases_are_not_extra_observations':True});csvsave('gpu_results.csv',results)
            save('gpu_aggregate.json',{'states':aggregates})
            save('benchmark_status.json',{'status':'OPERATIONAL_GPU_RUNNING','at_utc':now(),
                'completed_states':len(aggregates),'physical_runs_completed':len(results),'max_timed_runs':manifest['max_timed_runs']})
        assert len(results)==manifest['max_timed_runs'] and not snapshot()['processes']
        verify(read(OUT/'execution_freeze.json')['sources'])
        save('benchmark_status.json',{'status':'OPERATIONAL_GPU_VALIDATION_COMPLETE','at_utc':now(),'physical_runs':len(results)})
    except Exception as exc:
        save('benchmark_status.json',{'status':'STOPPED_INVALID_EXECUTION','exception':repr(exc),'at_utc':now(),'retry':False});raise

if __name__=='__main__':main()
