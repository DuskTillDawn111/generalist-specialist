"""Service-capacity experiment only. No periodic E2E/stability interpretation."""
import argparse,os,time
from .measure import transport,prime,phase
from .common import *
from .telemetry import CostMonitor
from dataclasses import replace
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.load_generator import SamplePool
from joint_deploy.measurement.run_measurement import new_record
from replica_scaling.telemetry import snapshot,release

def window(runtime,slot,plan,mapping,label,duration,monitor):
    pool=SamplePool(plan['sample_pool']['path'],[slot['primary_task']],slot['seed'])
    records=[];active={};refill=[];monitor.phase=label
    start=time.perf_counter_ns();end=start+round(duration*1e9)
    while time.perf_counter_ns()<end:
        now=time.perf_counter_ns()
        with runtime.lock:
            for i in mapping:
                previous=active.get(i)
                if previous and previous['status'] in ('queued','submitted'):continue
                if previous:
                    if previous['status']!='completed':raise RuntimeError('closed-loop worker request failed')
                    refill.append(now-previous['completion_time_ns'])
                item=pool.next(slot['primary_task'])
                record=new_record(label,len(records),slot['primary_task'],i,item,now)
                records.append(record);active[i]=record;runtime.enqueue(record,item['sample'])
        # Host observer only; transport, worker and model never patched.
        time.sleep(.0001)
        if any(w.process.poll() is not None for w in runtime.workers.values()):raise RuntimeError('closed-loop worker exit')
        if any(time.perf_counter_ns()-r['scheduled_time_ns']>20_000_000_000 for r in active.values() if r['status'] in ('queued','submitted')):
            raise RuntimeError('closed-loop outstanding age exceeds frozen protective guard')
    observed_end=time.perf_counter_ns();monitor.phase=label+'_drain';runtime.drain(records,plan['drain_timeout_seconds']);monitor.check()
    completed=[r for r in records if r['status']=='completed' and start<=r['completion_time_ns']<observed_end]
    elapsed=(observed_end-start)/1e9
    return {'window':{'start_time_ns':start,'end_time_ns':observed_end,'planned_end_time_ns':end,'duration_seconds':elapsed,
        'planned_duration_seconds':duration,'stop':None,'completion_definition':'successful responses received in fixed timed window; drain excluded'},
        'request_records':records,'completed_requests':len(completed),'throughput_rps':len(completed)/elapsed,
        'per_replica_throughput_rps':{i:sum(r['model']==i for r in completed)/elapsed for i in mapping},
        'max_outstanding_per_replica':1,'refill_delay_ns':stat(refill),
        'all_requests_completed_after_drain':all(r['status']=='completed' for r in records),
        'metric_scope':'service-capacity through unchanged serial worker transport; never periodic E2E/stability'}

def run(slot,output):
    plan=read(OUT/'measurement_plan.json');supp=read(OUT/'closed_loop_protocol.json');freeze=read(OUT/'input_freeze.json')
    fast_verify(freeze['protected_references']);verify([freeze[k] for k in ['measurement_plan','resident_set_manifest','samples']]);verify(supp['semantics_source_refs'])
    assert not output.exists()
    for k,v in plan['environment_overrides'].items():os.environ[k]=v
    assert not any(os.environ.get(k) for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'])
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128 and baseline['uuid']==plan['GPU']['uuid']
    mapping,config,registry=transport(slot,plan);runtime=DeploymentRuntime(config,registry)
    monitor=CostMonitor(runtime,plan['telemetry_interval_seconds'],plan['GPU']['total_mib']);monitor.start();begin=time.perf_counter_ns()
    result={'slot':slot,'closed_loop_protocol':ref(OUT/'closed_loop_protocol.json'),'closed_loop_source_revision':ref(OUT/'closed_loop_source_revision_v2.json'),'transport_config':config.as_dict(),
        'replica_to_family':mapping,'baseline_GPU':baseline,'status':'STARTING','errors':[],'phases':{},'worker_identities':{}}
    try:
        runtime.start();result['startup_seconds']=(time.perf_counter_ns()-begin)/1e9
        result['worker_identities']={i:{'family':m,'pid':runtime.workers[i].process.pid,'READY':runtime.ready[i]} for i,m in mapping.items()}
        result['path_probe_records']=prime(runtime,config,plan,mapping,slot)
        result['closed_loop_warmup']=window(runtime,slot,plan,mapping,'closed_loop_warmup',supp['warmup_seconds'],monitor)
        result['postwarm_idle']=monitor.idle_window('postwarm_idle',2.)
        result['phases']['closed_loop_capacity']=window(runtime,slot,plan,mapping,'closed_loop_capacity',supp['measurement_seconds'],monitor)
        low=replace(config,workload_rps=slot['low_lambda'],routing_rps={k:{i:v/slot['deployment'][m] for m,v in row.items() for i,family in mapping.items() if family==m} for k,row in slot['low_routing'].items()})
        result['phases']['low_load']=phase(runtime,low,plan,mapping,'low_load',supp['low_load_seconds'],None,monitor)
        if not result['phases']['closed_loop_capacity']['all_requests_completed_after_drain']:raise RuntimeError('closed-loop drain failed')
        monitor.check();result['status']='COMPLETED'
    except (Exception,KeyboardInterrupt) as exc:result['status']='FAILED';result['errors'].append(repr(exc))
    finally:
        try:result['shutdown']=runtime.close()
        except Exception as exc:result['errors'].append('shutdown:'+repr(exc))
        monitor.close();result['errors']+=runtime.errors+monitor.errors
        result['telemetry']={'samples':monitor.samples,'errors':monitor.errors,'interval_seconds':monitor.interval}
        result['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        result['valid_session']=bool(result['status']=='COMPLETED' and result['gpu_release']['verified'] and not result['errors'] and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in result.get('shutdown',{}).values()))
        result['elapsed_seconds']=(time.perf_counter_ns()-begin)/1e9;save_json(output,result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slot',required=True);a=p.parse_args();slot=read(a.slot)
    raise SystemExit(0 if run(slot,OUT/'raw_runs'/f"{slot['run_id']}.json")['gpu_release']['verified'] else 1)
