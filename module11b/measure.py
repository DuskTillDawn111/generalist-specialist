"""Characterization harness calling the unchanged DeploymentRuntime/worker scripts.

Count n is represented by n independent family#i IDs with x=1 transport entries,
as in Module7 replica measurements. No serving transport is patched or replaced.
"""
import argparse,os,time,math
from dataclasses import replace
from types import SimpleNamespace
from collections import Counter
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.load_generator import SamplePool
from joint_deploy.measurement.run_measurement import new_record
from joint_deploy.measurement.metrics import annotate_records,summarize,percentile
from replica_scaling.runtime import execute,scalar_metrics
from replica_scaling.telemetry import snapshot,release
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from .telemetry import CostMonitor
from .common import *

def transport(slot,plan):
    mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
    route={k:{i:v/slot['deployment'][m] for m,v in row.items() for i,family in mapping.items() if family==m} for k,row in slot['routing'].items()}
    config=MeasurementConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=route,
         random_seed=slot['seed'],warmup_seconds=plan['warmup_seconds'],measurement_seconds=slot['measurement_seconds'],
         drain_timeout_seconds=plan['drain_timeout_seconds'],startup_timeout_seconds=plan['startup_timeout_seconds'],
         shutdown_timeout_seconds=plan['shutdown_timeout_seconds'],max_pending_per_model=10000,routing_source='module11B_characterization_equal_family_replica_round_robin')
    config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
    return mapping,config,{i:plan['registry'][m] for i,m in mapping.items()}

def prime(runtime,config,plan,mapping,slot):
    records=[];pool=SamplePool(plan['sample_pool']['path'],list({k for m in mapping.values() for k in plan['models'][m]['supported_tasks']}),slot['seed'])
    for i,m in sorted(mapping.items()):
        # Sequential per-path calls exclude queueing from the per-task probes.
        for k in plan['models'][m]['supported_tasks']:
            for j in range(3):
                item=pool.next(k);r=new_record('path_probe',len(records),k,i,item,time.perf_counter_ns());records.append(r)
                runtime.enqueue(r,item['sample']);runtime.drain([r],config.drain_timeout_seconds)
                if r['status']!='completed':raise RuntimeError(f'Low-load path probe failed: {i}/{k}')
    annotate_records(records);return records

def phase(runtime,config,plan,mapping,label,duration,guard,monitor):
    records=[];monitor.phase=label
    s0=max(plan['models'][m]['service_time_ms'][k] for k,row in config.routing_rps.items() for i,v in row.items() if v for m in [mapping[i]])
    window=execute(runtime,config,plan['sample_pool']['path'],label,duration,records,s0,guard,monitor)
    annotate_records(records)
    per_task,system,models=summarize(records,config,window)
    temp={'request_records':records,'measurement_window':window,'system_metrics':system,
          'status':'completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures',
          'errors':[],'drain_timeout_seconds':config.drain_timeout_seconds,
          'worker_identities':{i:{'pid':w.process.pid} for i,w in runtime.workers.items()}}
    metrics=scalar_metrics(temp,replace(config,measurement_seconds=window['duration_seconds']))
    aliases={i:SimpleNamespace(service_time_ms=plan['models'][m]['service_time_ms']) for i,m in mapping.items()}
    ev=reconstruct(temp,aliases,DEFAULT_THRESHOLDS)
    return {'window':window,'request_records':records,'per_task_metrics':per_task,'system_metrics':system,
            'metrics':metrics,'stability':classify(ev,DEFAULT_THRESHOLDS),'stability_evidence':ev}

def run(slot,output):
    plan=read(OUT/'measurement_plan.json');freeze=read(OUT/'input_freeze.json');fast_verify(freeze['protected_references'])
    verify([freeze[k] for k in ['measurement_plan','resident_set_manifest','samples']])
    assert not output.exists(),'Raw observation already exists; no outcome retries'
    for k,v in plan['environment_overrides'].items():os.environ[k]=v
    assert not any(os.environ.get(k) for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128
    assert baseline['uuid']==plan['GPU']['uuid']
    mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry)
    monitor=CostMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib'])
    result={'slot':slot,'transport_config':config.as_dict(),'replica_to_family':mapping,'baseline_GPU':baseline,
        'source_freeze':ref(OUT/'input_freeze.json'),'measurement_plan':ref(OUT/'measurement_plan.json'),
        'status':'STARTING','errors':[],'phases':{},'worker_identities':{},'GPU_new_run_scope':'Module11B_characterization'}
    monitor.start();begin=time.perf_counter_ns()
    try:
        runtime.start();ready=time.perf_counter_ns();monitor.check()
        result['startup_seconds']=(ready-begin)/1e9
        result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()}
        assert len({r['pid'] for r in result['worker_identities'].values()})==len(mapping)
        if slot['kind']=='fixed_cost':result['cold_idle']=monitor.idle_window('cold_idle',plan['idle_observation_seconds'])
        result['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
        # Warm at 0.45*g_previous capacity for profiling, and at the frozen low
        # characterization rates for resident sets. No high-load warmup backlog.
        warm=replace(config,workload_rps=slot['warm_lambda'],routing_rps={k:{i:v/slot['deployment'][m] for m,v in row.items() for i,family in mapping.items() if family==m} for k,row in slot['warm_routing'].items()})
        result['warmup']=phase(runtime,warm,plan,mapping,'warmup',config.warmup_seconds,None,monitor)
        if any(r['status']!='completed' for r in result['warmup']['request_records']):raise RuntimeError('Lower-load warmup failed')
        result['postwarm_idle']=monitor.idle_window('postwarm_idle',plan['idle_observation_seconds'] if slot['kind']=='fixed_cost' else 2.)
        if slot['kind']=='replica_profile':
            result['phases']['capacity']=phase(runtime,config,plan,mapping,'measurement',slot['measurement_seconds'],plan['protective_stop'],monitor)
            # High-pressure instability alone cannot declare an unavailable count.
            low=replace(config,workload_rps=slot['low_lambda'],routing_rps={k:{i:v/slot['deployment'][m] for m,v in row.items() for i,family in mapping.items() if family==m} for k,row in slot['low_routing'].items()})
            result['phases']['low_load']=phase(runtime,low,plan,mapping,'low_load',slot['measurement_seconds'],None,monitor)
        else:
            modes=['G_ONLY','G_PLUS_LOW_S'] if slot['deployment'] and any(plan['models'][m]['role']=='specialist' for m in slot['deployment']) else ['G_ONLY']
            if slot['seed']==plan['seeds'][1]:modes.reverse()
            for mode in modes:
                rates=slot['lambda_rps'] if mode=='G_ONLY' else slot['active_lambda']
                routing=slot['routing'] if mode=='G_ONLY' else slot['active_routing']
                phaseconfig=replace(config,workload_rps=rates,routing_rps={k:{i:v/slot['deployment'][m] for m,v in row.items() for i,family in mapping.items() if family==m} for k,row in routing.items()})
                result['phases'][mode]=phase(runtime,phaseconfig,plan,mapping,mode,slot['measurement_seconds'],None,monitor)
        monitor.check();result['status']='COMPLETED'
    except (Exception,KeyboardInterrupt) as exc:
        result['status']='FAILED';result['errors'].append(repr(exc))
    finally:
        try:result['shutdown']=runtime.close()
        except Exception as exc:result['errors'].append('shutdown:'+repr(exc))
        monitor.close();result['errors']+=runtime.errors+monitor.errors
        result['telemetry']={'samples':monitor.samples,'errors':monitor.errors,'interval_seconds':monitor.interval,
            'context_count_observation':'NVML compute PID count; exact CUDA context count unavailable',
            'startup_READY_observation_ns':monitor.first_ready_seen,'startup_PID_observation_ns':monitor.first_pid_seen,
            'peak_scope':'sampled NVML high-water; no allocator/worker instrumentation'}
        result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and
             all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result.get('shutdown',{}).values()))
        result['elapsed_seconds']=(time.perf_counter_ns()-begin)/1e9
        save_json(output,result)
    print(json.dumps({'run_id':slot['run_id'],'status':result['status'],'valid':result['valid_session'],
         'release':result['gpu_release']['verified'],'seconds':result['elapsed_seconds'],'errors':result['errors']}),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slot',required=True);args=parser.parse_args()
    slot=read(args.slot);return 0 if run(slot,OUT/'raw_runs'/f"{slot['run_id']}.json")['gpu_release']['verified'] else 1

if __name__=='__main__':raise SystemExit(main())
