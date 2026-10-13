"""Run only method states admitted by the frozen CPU plan, with exact reuse."""
import os,fcntl
from dataclasses import asdict
from deployment_core import Workload,Hardware
from method_v2 import MethodV2,frozen_config
from integer_replica_pilot.evaluator import runner,ReplicaRuntimeEvaluator
from .common import *
from .planning import flat

def guarded_runner(config,raw_path,log_path,config_path,context_path):
    verify(read(OUT/'execution_freeze.json')['sources'])
    ledger=read(OUT/'gpu_attempts.json') if (OUT/'gpu_attempts.json').exists() else {'runs':[]}
    max_runs=read(OUT/'gpu_manifest.json')['max_timed_runs']
    assert len(ledger['runs'])<max_runs,'Frozen physical budget exhausted'
    entry={'started_at_utc':now(),'raw_path':str(raw_path),'config':config,'context':ref(context_path),'status':'STARTED'}
    ledger['runs'].append(entry);save('gpu_attempts.json',ledger)
    outcome=runner(config,raw_path,log_path,config_path,context_path)
    entry.update(status='RETURNED',outcome=outcome,finished_at_utc=now(),raw_ref=ref(raw_path) if raw_path.exists() else None)
    save('gpu_attempts.json',ledger);return outcome

def main():
    assert read(OUT/'benchmark_status.json')['status']=='GPU_PLAN_FROZEN','Do not relaunch a completed or failed attempt'
    lock=(OUT/'gpu_execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'gpu_manifest.json');assert manifest['max_timed_runs']==3,'This frozen design has one exact runnable method state'
    assert read(OUT/'cpu_validation.json')['status']=='PASS'
    verify(read(OUT/'input_freeze.json')['protected_references'])
    from replica_scaling.telemetry import snapshot
    import subprocess
    gpu=snapshot();original=read(ROOT/'results/module8f2_resource_static/campaign_manifest.json')['GPU']
    assert all(gpu[k]==original[k] for k in ['uuid','name','driver','total_mib'])
    assert not gpu['processes'] and gpu['used_mib']<=128
    assert not subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True).stdout.strip()
    assert not any(os.environ.get(k) for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
    for k,v in manifest['execution_environment_overrides'].items():os.environ[k]=v
    sources=[ref(p) for p in (ROOT/'module10').glob('*.py')]
    save('execution_freeze.json',{'at_utc':now(),'sources':sources,'GPU':gpu,'no_MPS':True,'no_runtime_worker_edits':True,
                                'max_timed_runs':3,'settings':manifest['settings'],'context_source':manifest['context_source'],
                                'policy':'zero retries; invalid or resource failure preserved separately; valid instability retained'})
    save('benchmark_status.json',{'status':'GPU_RUNNING','at_utc':now()})
    c,h,roles,cal=inputs();context=read(OUT/'primary_hotspot_selection.json')['context'];results=[]
    try:
        for state in manifest['states']:
            w=Workload(state['lambda_rps']);evaluator=ReplicaRuntimeEvaluator(OUT/'gpu'/state['state_id'],c,manifest['context_source']['path'],manifest['settings'],seeds=manifest['seeds'],max_unique=6,run=guarded_runner)
            actual=evaluator.evaluate(state['deployment'],w,flat(state['routing']))
            # Reuse the exact baseline identity to execute the unchanged v2
            # accept/rollback stack without charging duplicate physical labels.
            engine=MethodV2(evaluator)
            result=engine.search(w,c,Hardware(context['planning_budget_input_mib']),frozen_config(),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
            assert result.deployment==state['deployment'] and result.runtime_evaluation_count==1
            assert all(not t['selected_top_b'] for t in result.iterations)
            save(f'gpu/{state["state_id"]}/v2_search_complete.json',result.to_dict())
            for index,rep in enumerate(actual.repetitions,1):
                results.append({'state_id':state['state_id'],'point':state['point'],'methods_sharing_exact_label':state['methods'],'repetition':index,'seed':rep['random_seed'],
                                'deployment':state['deployment'],'routing':state['routing'],'stability':rep['stability_state_v2'],'mean_e2e_ms':rep['weighted_mean_e2e'],
                                'weighted_p95_ms':rep['weighted_p95_e2e'],'per_task_mean':rep['per_task_mean'],'per_task_p95':rep['per_task_p95'],
                                'memory_observations':rep['memory_observations'],'raw_path':rep['raw_result_path'],'raw_sha256':rep['raw_sha256']})
            save('gpu_results.json',{'rows':results,'independent_physical_measurements':len(results),'exact_state_aliases_are_not_extra_observations':True});csvsave('gpu_results.csv',results)
            save('gpu_aggregate.json',{'state_id':state['state_id'],'methods':state['methods'],'runtime':asdict(actual)})
        assert not snapshot()['processes'],'Owned GPU workers still active'
        verify(read(OUT/'execution_freeze.json')['sources']);save('benchmark_status.json',{'status':'GPU_VALIDATION_COMPLETE','at_utc':now(),'physical_runs':len(results)})
    except Exception as exc:
        save('benchmark_status.json',{'status':'STOPPED_INVALID_EXECUTION','exception':repr(exc),'at_utc':now(),'retry':False});raise
if __name__=='__main__':main()
