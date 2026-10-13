"""Replica-aware measurement harness preserving Module 3C timing and task inputs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import threading
import time
from dataclasses import replace

from joint_deploy.measurement.load_generator import SamplePool, periodic_arrivals, phases_for
from joint_deploy.measurement.run_measurement import DEFAULT_POOL, new_record
from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.metrics import annotate_records, summarize, percentile
from replica_scaling.telemetry import Monitor, snapshot, release, owned_sample
from deployment_search.runtime_cache import save_json, file_hash, digest
from .adapter import ReplicaAwareDeploymentRuntime, FamilyReplicaRouter, RuntimeResourceInfeasible


class SafetyMonitor(Monitor):
    def __init__(self,runtime,interval,capacity):
        super().__init__(runtime,interval)
        self.capacity=capacity
        self.resource_violation=threading.Event()
        self.foreign_interference=threading.Event()

    def collect(self):
        while not self.stop.is_set():
            try:
                pids={w.process.pid for w in list(self.runtime.workers.values()) if w.process is not None}
                value=owned_sample(snapshot(),pids);value['phase']=self.phase
                self.samples.append(value)
                if value['owned_used_mib']>self.capacity or value['used_mib']>self.capacity:self.resource_violation.set()
                if value['foreign_pids']:self.foreign_interference.set()
            except Exception as exc:self.errors.append(str(exc))
            self.stop.wait(self.interval)

    def check(self):
        if self.resource_violation.is_set():raise RuntimeResourceInfeasible('Actual VRAM exceeds frozen GPU budget')
        if self.foreign_interference.is_set():raise RuntimeError('external_gpu_process_interference')


def execute_window(runtime,config,pool_path,phase,duration,records,monitor):
    # Same input shuffle/cycle, periodic phases and family Router as Module 3C.
    pool=SamplePool(pool_path,[k for k,v in config.workload_rps.items() if v>0],config.random_seed)
    router=FamilyReplicaRouter(config.routing_rps,config.deployment)
    start,unix=time.perf_counter_ns(),time.time_ns();end=start+round(duration*1e9)
    window={'start_time_ns':start,'start_unix_ns':unix,'end_time_ns':end,'duration_seconds':duration,
        'cohort':'requests scheduled in [start,end)',
        'completion_count_definition':'successful cohort completions up to the drain deadline'}
    def wait(deadline):
        while True:
            monitor.check()
            remaining=(deadline-time.perf_counter_ns())/1e9
            if remaining<=0:return
            time.sleep(min(remaining,.1))
    for index,(offset,task) in enumerate(periodic_arrivals(config.workload_rps,duration,config.random_seed)):
        item=pool.next(task);replica,family=router.choose(task)
        row=new_record(phase,index,task,replica,item,start+round(offset*1e9))
        row.update(model_family=family,replica_id=replica)
        records.append(row);wait(row['scheduled_time_ns']);runtime.enqueue(row,item['sample'])
    wait(end);window['arrival_loop_end_time_ns']=time.perf_counter_ns()
    monitor.phase='drain' if phase=='measurement' else 'warmup_drain'
    runtime.drain(records,config.drain_timeout_seconds)
    monitor.check()
    window['drain_end_time_ns']=time.perf_counter_ns()
    window['drain_seconds']=(window['drain_end_time_ns']-end)/1e9
    return window


def replica_metrics(rows,config,window,mapping):
    duration=window['duration_seconds'];end=window['end_time_ns']
    def metrics(subset):
        done=[r for r in subset if r['status']=='completed']
        means={}
        for key in ('worker_latency_ms','worker_queue_wait_ms','e2e_latency_ms'):
            values=[r[key] for r in done if r.get(key) is not None]
            means[key]={'mean':sum(values)/len(values) if values else None,'p95':percentile(values,95),'observations':len(values)}
        return {'request_count':len(subset),'completed_requests':len(done),
            'per_task_request_count':dict(Counter(r['task'] for r in subset)),
            'completed_throughput_rps':sum(r['completion_time_ns']<end for r in done)/duration,
            'cohort_completion_throughput_rps':len(done)/duration,
            'completion_ratio':len(done)/len(subset) if subset else None,**means}
    per={i:metrics([r for r in rows if r['model']==i]) for i in mapping}
    family={m:dict(metrics([r for r in rows if r['model_family']==m]),replica_count=n)
        for m,n in config.deployment.items() if n}
    return {'replica_count_by_model':config.deployment,'per_replica_request_count':{i:v['request_count'] for i,v in per.items()},
        'per_replica_throughput':{i:v['completed_throughput_rps'] for i,v in per.items()},
        'per_replica_worker_latency':{i:v['worker_latency_ms'] for i,v in per.items()},
        'per_replica_queue_wait':{i:v['worker_queue_wait_ms'] for i,v in per.items()},
        'per_replica_metrics':per,'aggregate_family_metrics':family}


def run(config,context,output,registry=None,monitor_factory=SafetyMonitor,release_fn=release):
    if registry is None:
        registry=context['registry']
    capabilities={m:row['supported_tasks'] for m,row in context['models'].items()}
    runtime=ReplicaAwareDeploymentRuntime(config,registry,capabilities,context['replica_bounds'],context['memory_table_mib'],context['gpu_capacity_mib'])
    baseline=snapshot()
    if baseline['processes'] or baseline['used_mib']>128 or baseline['uuid']!=context['gpu_uuid']:
        raise RuntimeError('GPU isolation/identity gate failed; no model worker launched')
    monitor=monitor_factory(runtime,context['telemetry_interval_seconds'],context['gpu_capacity_mib'])
    result=dict(schema_version=2,**config.as_dict(),replica_count_by_model=dict(config.deployment),
        transport_deployment=runtime.config.deployment,transport_routing_rps=runtime.config.routing_rps,
        replica_to_family=runtime.replica_to_family,planning_memory_mib=runtime.planning_memory_mib,
        source=context['source_hashes'],runtime_context_sha256=context['context_sha256'],baseline_gpu=baseline,
        arrival_process_details={'name':config.arrival_process,'random_seed':config.random_seed,
            'initial_phase_seconds':phases_for(config.workload_rps,config.random_seed)},
        routing_policy='Frozen family smooth weighted round-robin, followed by deterministic round-robin across that family replicas; counters reset each phase.',
        clock='Existing host perf_counter_ns request clocks and frozen worker wall-clock handler/queue timers; no cross-clock subtraction.',
        status='starting',execution_status='not_started',runtime_resource_status='feasible',errors=[],
        request_records=[],priming_request_records=[],warmup_request_records=[],worker_identities={},
        warmup={},measurement_window=None,models={},per_task_metrics={},system_metrics=None)
    monitor.start()
    try:
        print('Starting replica workers...',flush=True);runtime.start();monitor.check()
        result['worker_identities']={i:{'worker_id':i,'family':m,'replica_index':int(i.rsplit('#',1)[1]),
            'pid':runtime.workers[i].process.pid,'ready_payload':runtime.ready[i],
            'cuda_process_identity':'NVML compute-process PID; independent model process / CUDA residency',
            'model_source':context['models'][m]['source']} for i,m in runtime.replica_to_family.items()}
        if len(set(v['pid'] for v in result['worker_identities'].values()))!=sum(config.deployment.values()):raise RuntimeError('Duplicate worker PID')
        print(json.dumps({'event':'workers_ready','worker_pids':{i:v['pid'] for i,v in result['worker_identities'].items()},'planning_idle_mib':runtime.planning_memory_mib}),flush=True)
        result['idle_after_load']=monitor.idle('idle_after_load');monitor.check()
        if config.warmup_seconds>0:
            pool=SamplePool(context['input_pool'],[k for k,v in config.workload_rps.items() if v>0],config.random_seed)
            for task,route in sorted(config.routing_rps.items()):
                for family,rate in sorted(route.items()):
                    if rate<=0:continue
                    for i,m in runtime.replica_to_family.items():
                        if m!=family:continue
                        item=pool.next(task);row=new_record('priming',len(result['priming_request_records']),task,i,item,time.perf_counter_ns())
                        row.update(model_family=family,replica_id=i);result['priming_request_records'].append(row);runtime.enqueue(row,item['sample'])
            runtime.drain(result['priming_request_records'],config.drain_timeout_seconds)
            if any(r['status']!='completed' for r in result['priming_request_records']):raise RuntimeError('Service-path priming failed')
            print('Priming complete; starting periodic warmup.',flush=True)
            monitor.phase='warmup';result['warmup']=execute_window(runtime,config,context['input_pool'],'warmup',config.warmup_seconds,result['warmup_request_records'],monitor)
            if any(r['status']!='completed' for r in result['warmup_request_records']):raise RuntimeError('Warmup failed; measurement not started')
        result['idle_before_workload']=monitor.idle('idle_before_workload');monitor.check()
        monitor.phase='measurement';print('Starting measurement window.',flush=True)
        result['measurement_window']=execute_window(runtime,config,context['input_pool'],'measurement',config.measurement_seconds,result['request_records'],monitor)
        result['execution_status']='completed'
        result['status']='completed' if all(r['status']=='completed' for r in result['request_records']) else 'completed_with_failures'
    except KeyboardInterrupt:
        result.update(status='failed',execution_status='external_interruption');result['errors'].append('Observed KeyboardInterrupt')
    except Exception as exc:
        text=f'{type(exc).__name__}: {exc}';result['errors'].append(text);result['status']='failed'
        if isinstance(exc,RuntimeResourceInfeasible) or 'out of memory' in text.lower():
            result.update(execution_status='RUNTIME_RESOURCE_INFEASIBLE',runtime_resource_status='RUNTIME_RESOURCE_INFEASIBLE')
        else:result['execution_status']='model_or_harness_failure'
    finally:
        monitor.phase='shutdown'
        try:result['shutdown']=runtime.close()
        except Exception as exc:result['errors'].append(f'shutdown: {exc}');result['status']='failed'
        monitor.close();result['errors'].extend(runtime.errors)
        for key in ('request_records','priming_request_records','warmup_request_records'):
            for row in result[key]:
                if row['status'] in ('planned','queued','submitted'):row.update(status='incomplete',error='run terminated before response')
            annotate_records(result[key])
        if runtime.errors or any(s['exit_code']!=0 or s['forced'] or not s['bye_received'] for s in result.get('shutdown',{}).values()):result['status']='failed'
        if any('out of memory' in str(r.get('error','')).lower() for kind in ('request_records','priming_request_records','warmup_request_records') for r in result[kind]):
            result.update(runtime_resource_status='RUNTIME_RESOURCE_INFEASIBLE',execution_status='RUNTIME_RESOURCE_INFEASIBLE',status='failed')
        if result['measurement_window']:
            per_task,system,models=summarize(result['request_records'],runtime.config,result['measurement_window'])
            result.update(per_task_metrics=per_task,system_metrics=system,models=models)
            result.update(replica_metrics(result['request_records'],config,result['measurement_window'],runtime.replica_to_family))
            family_records=[dict(r,model=r['model_family']) for r in result['request_records']]
            result['family_per_task_metrics']=summarize(family_records,config,result['measurement_window'])[0]
        samples=monitor.samples
        result['telemetry']={'samples':samples,'errors':monitor.errors,'interval_seconds':context['telemetry_interval_seconds'],
            'peak_note':'Sampled NVML high water over load/warmup/measurement/drain; not exact allocation peak.'}
        workload=[s for s in samples if s['phase'] in ('measurement','drain')]
        result['memory_observations']={'planning_idle_mib':runtime.planning_memory_mib,
            'actual_idle_mib':math.fsum(s['owned_used_mib'] for s in result.get('idle_before_workload',[]))/len(result['idle_before_workload']) if result.get('idle_before_workload') else None,
            'peak_owned_mib':max((s['owned_used_mib'] for s in samples),default=None),
            'peak_workload_mib':max((s['owned_used_mib'] for s in workload),default=None),
            'peak_device_mib':max((s['used_mib'] for s in samples),default=None),
            'budget_mib':context['gpu_capacity_mib']}
        result['gpu_release']=release_fn(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        result['warmup'].update(status_counts=dict(Counter(r['status'] for r in result['warmup_request_records'])),configured_seconds=config.warmup_seconds,priming_excluded_from_measurement=True)
        result['valid_execution']=bool(result['execution_status']=='completed' and result['measurement_window'] and
            result['measurement_window']['duration_seconds']==config.measurement_seconds and result['gpu_release']['verified'] and
            not monitor.errors and not monitor.foreign_interference.is_set())
        save_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();context=json.loads(Path(args.context).read_text())
    expected=Path(args.context+'.sha256').read_text().strip()
    if file_hash(args.context)!=expected or digest({k:v for k,v in context.items() if k!='context_sha256'})!=context['context_sha256']:
        raise RuntimeError('Frozen runtime context changed')
    result=run(MeasurementConfig(**json.loads(Path(args.config).read_text())),context,args.output)
    print(json.dumps({'output':args.output,'status':result['status'],'execution_status':result['execution_status']}),flush=True)
    return 0 if result['valid_execution'] else 1


if __name__=='__main__':raise SystemExit(main())
