"""Exactly one frozen 600s dynamic plumbing validation; no controller campaign."""
import sys
sys.dont_write_bytecode=True
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import collections
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/root/autodl-tmp/generalist-specialist')
sys.path.insert(0,str(ROOT))
from instrumentation import OUT,Journal,LoggedRuntime,save,ref,router_state,guards
from dynamic_router import DynamicRouter
from module22.gpu import PoissonConfig
from module11b.measure import prime
from module16.gpu import ProcessMonitor
from module26.gpu import thermal
from module21.trajectory import choose
from joint_deploy.measurement.run_measurement import new_record,wait_until
from joint_deploy.measurement.metrics import annotate_records,summarize
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS
from replica_scaling.telemetry import snapshot,release

def protected_check():
    frozen=json.loads((OUT/'input_freeze.json').read_text())
    changes=[r['path'] for r in frozen['inputs'] if not Path(r['path']).is_file()
             or ref(r['path'])['sha256']!=r['sha256'] or Path(r['path']).stat().st_size!=r['bytes']]
    current=sorted(str(p) for p in (ROOT/'results/module28_dynamic_split').rglob('*') if p.is_file())
    assert not changes and current==frozen['module28_files'],(changes,'Module28 file set')
    return dict(unchanged=True,checked=len(frozen['inputs']),changes=changes,module28_file_set_unchanged=True)

class Monitor(ProcessMonitor):
    def __init__(self,runtime,budget):
        super().__init__(runtime,.5,budget);self.active_phase=None;self.last_progress=0.
    def sample(self):
        v=super().sample();v.update(thermal())
        a=self.active_phase
        if a and time.monotonic()-self.last_progress>20:
            self.last_progress=time.monotonic();rs=a['request_records']
            done=[r for r in rs if r['status']=='completed']
            sent=[r for r in rs if r.get('actual_submit_time_ns')]
            g=[r for r in sent if r['task']=='lid' and r['model']=='qwen#0']
            progress=dict(phase=self.phase,elapsed_seconds=(time.perf_counter_ns()-a['planned_start_time_ns'])/1e9,
                          planned=len(rs),sent=len(sent),successful=len(done),unfinished=len(sent)-len(done),
                          target_G_sent=len(g),target_G_successful=sum(r['status']=='completed' for r in g),
                          temperature_gpu_c=v['temperature_gpu_c'],owned_memory_mib=v['owned_used_mib'])
            save('live_progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
        return v

def cool(f,monitor=None):
    policy=f['thermal_protocol'];resident=monitor is not None
    limit=policy['resident_cool_timeout_seconds'] if resident else policy['max_wait_seconds']
    maxc=policy['resident_premeasurement_temperature_max_c'] if resident else policy['prelaunch_temperature_max_c']
    needed=policy['resident_consecutive_samples'] if resident else policy['consecutive_idle_samples_required']
    begin=time.monotonic();samples=[];consecutive=0
    while time.monotonic()-begin<limit:
        v=monitor.sample() if resident else snapshot()
        if not resident:v.update(thermal())
        samples.append(v)
        if not resident and (v['processes'] or v['used_mib']>128):raise RuntimeError('GPU_NOT_ISOLATED')
        if monitor:monitor.check()
        consecutive=consecutive+1 if v['temperature_gpu_c']<=maxc else 0
        if consecutive>=needed:return dict(passed=True,samples=samples,elapsed_seconds=time.monotonic()-begin)
        time.sleep(2.)
    raise RuntimeError('THERMAL_LAUNCH_FAILURE')

def phase(runtime,config,slot,f,mapping,label,monitor):
    trajectory=json.loads(Path(slot['trajectories'][label]['path']).read_text())
    assert ref(slot['trajectories'][label]['path'])['sha256']==slot['trajectories'][label]['sha256']
    duration=trajectory['duration'];events=trajectory['events']
    records=[]
    with runtime.lock:
        runtime.observation(time.perf_counter_ns(),True)
        runtime.router=DynamicRouter(copy.deepcopy(f['parameters']))
        start=time.perf_counter_ns();unix=time.time_ns();end=start+round(duration*1e9)
        runtime.phase=label;runtime.window_origin_ns=start
        runtime.journal.emit('PHASE_START',phase=label,time_ns=start,start_unix_ns=unix,
                             end_time_ns=end,duration_seconds=duration,router_state=router_state(runtime.router))
    monitor.phase=label
    monitor.active_phase=dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,
                              planned_duration_seconds=duration,request_records=records)
    for i,e in enumerate(events):
        # Default S for target; nominal routing rates never select target's actual route.
        worker=f['parameters']['S_worker'] if e['task']==f['target'] else choose(e,slot['routing'],slot['deployment'])
        r=new_record(label,i,e['task'],worker,e['item'],start+round(e['offset_seconds']*1e9))
        r.update(frozen_offset_seconds=e['offset_seconds'],routing_uniform=e['routing_uniform'],replica_uniform=e['replica_uniform'])
        records.append(r)
    for r,e in zip(records,events):
        wait_until(r['scheduled_time_ns']);monitor.check()
        assert not runtime.errors,runtime.errors
        overhead_start=time.perf_counter_ns()
        with runtime.lock:
            now=time.perf_counter_ns() # actual atomic observation/decision point, after lock acquisition
            target=r['task']==f['target']
            observation=runtime.observation(now,target)
            decision=None;guard=None;router_end=observation['router_before_decide']
            if target:
                admitted=all(f['target'] in runtime.ready[w]['tasks'] for w in [f['parameters']['G_worker'],f['parameters']['S_worker']])
                capacity_ok=not monitor.resource_failure and not runtime.errors and all(runtime.workers[w].process.poll() is None for w in [f['parameters']['G_worker'],f['parameters']['S_worker']])
                decide_start=time.perf_counter_ns()
                worker,decision=runtime.router.decide(r['request_id'],now/1e9,observation['pending'],admitted,capacity_ok)
                decide_us=(time.perf_counter_ns()-decide_start)/1000
                r['model']=worker;router_end=router_state(runtime.router)
                guard=guards(f['parameters'],observation['router_before_decide'],decision,now/1e9,admitted,capacity_ok)
                r['routing_decision']=dict(**decision,decision_time_ns=now,
                                          router_decide_only_us=decide_us,
                                          scheduled_time_ns=r['scheduled_time_ns'],
                                          latest_observed_completion_time_ns=runtime.latest_received_ns,
                                          router_observation_and_decision_us=(time.perf_counter_ns()-overhead_start)/1000)
            runtime.journal.emit('ARRIVAL_DECISION',phase=label,time_ns=now,
                scheduled_time_ns=r['scheduled_time_ns'],request_id=r['request_id'],task=r['task'],
                target_router_invoked=target,selected_worker=r['model'],
                **observation,decision=decision,guards=guard,router_after_decide=router_end,
                worker_process_alive={w:runtime.workers[w].process.poll() is None for w in runtime.workers},
                original_admission_capacity_semantics='READY supports task; no resource failure/errors; G/S process alive',
                native_handler_state_used=False)
            runtime.enqueue(r,e['item']['sample'])
            r['instrumented_arrival_total_us']=(time.perf_counter_ns()-overhead_start)/1000
    wait_until(end);monitor.check();arrival_end=time.perf_counter_ns()
    monitor.phase=label+'_drain'
    runtime.journal.emit('WINDOW_END',phase=label,time_ns=end,actual_observation_time_ns=arrival_end)
    runtime.drain(records,60.);drain=time.perf_counter_ns()
    with runtime.lock:
        final_observation=runtime.observation(drain,True)
        runtime.journal.emit('PHASE_DRAIN_END',phase=label,time_ns=drain,**final_observation,
                             router_state=router_state(runtime.router))
    annotate_records(records)
    window=dict(start_time_ns=start,start_unix_ns=unix,end_time_ns=end,planned_end_time_ns=end,
                duration_seconds=duration,planned_duration_seconds=duration,arrival_loop_end_time_ns=arrival_end,
                drain_end_time_ns=drain,drain_seconds=(drain-end)/1e9,stop=None,
                cohort='Frozen requests scheduled in [start,end); drain completions reported separately')
    per_task,system,models=summarize(records,config,window)
    temp=dict(request_records=records,measurement_window=window,errors=runtime.errors,
              status='completed' if all(r['status']=='completed' for r in records) else 'completed_with_failures',
              drain_timeout_seconds=60.,worker_identities={i:dict(pid=w.process.pid) for i,w in runtime.workers.items()})
    aliases={i:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for i,m in mapping.items()}
    ev=reconstruct(temp,aliases,DEFAULT_THRESHOLDS)
    monitor.active_phase=None
    return dict(window=window,request_records=records,per_task_metrics=per_task,system_metrics=system,
                worker_metrics=models,stability_evidence=ev,stability=classify(ev,DEFAULT_THRESHOLDS),
                prediction_errors=runtime.router.prediction_errors)

def sanitized(value):
    if isinstance(value,list):return [sanitized(v) for v in value]
    if not isinstance(value,dict):return value
    out={}
    allowed={'worker','request_id','task_type','pool_id','arrival_index','submit_ns','start_ns','end_ns',
             'worker_latency_ms','queue_wait_ms','correct','wer','cer','exact_match','valid_closed_set_output',
             'model_identity','model_id','pid'}
    for k,v in value.items():
        if k in ['worker_response','late_response_after_drain']:
            out[k]={kk:vv for kk,vv in v.items() if kk in allowed}
        elif k in ['sample','input','prediction','answer','transcript','label','traceback']:
            continue
        else:out[k]=sanitized(v)
    return out

def main():
    assert not (OUT/'gpu_attempt_started.json').exists(),'One attempt only; never rerun'
    protocol=json.loads((OUT/'controlled_manifest.json').read_text())
    for item in protocol['implementation']:assert ref(item['path'])['sha256']==item['sha256']
    assert json.loads((OUT/'cpu_preflight.json').read_text())['passed']
    protected_check()
    f=json.loads((ROOT/'results/module28_dynamic_split/frozen_candidate_manifest.json').read_text())
    slot=protocol['slot'];plan=f['plan']
    for k,v in plan['environment_overrides'].items():os.environ[k]=v
    pre=cool(f);save('thermal_prelaunch.json',pre);baseline=pre['samples'][-1]
    assert all(baseline[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
    save('gpu_attempt_started.json',dict(run_id=slot['run_id'],at_unix_ns=time.time_ns(),attempt=1,maximum_attempts=1))
    mapping={m+'#0':m for m in slot['deployment']}
    routes={t:{m+'#0':q for m,q in row.items()} for t,row in slot['routing'].items()}
    config=PoissonConfig(deployment={w:1 for w in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,
        arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=30.,measurement_seconds=600.,
        startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,
        routing_source='Module29 stage2 one frozen dynamic functional validation; not formal comparison')
    config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
    journal=Journal('decision_event_journal.jsonl')
    runtime=LoggedRuntime(config,{w:plan['registry'][m] for w,m in mapping.items()},journal)
    monitor=Monitor(runtime,f['budget_mib']);monitor.start();begin=time.perf_counter()
    raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,replica_to_family=mapping,
             transport_config=config.as_dict(),worker_identities={},legal_U=None,delta_U=None,
             scope='FUNCTIONAL_VALIDATION_ONLY_NO_LEGAL_UTILITY_OR_CAUSAL_GAIN_CLAIM')
    try:
        print('WORKER_START',slot['run_id'],flush=True);runtime.start();monitor.check()
        raw['worker_identities']={w:dict(family=m,pid=runtime.workers[w].process.pid,READY=runtime.ready[w]) for w,m in mapping.items()}
        assert len({v['pid'] for v in raw['worker_identities'].values()})==5
        raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
        print('WARMUP_START 30s',flush=True)
        raw['warmup']=phase(runtime,config,slot,f,mapping,'warmup',monitor)
        assert all(r['status']=='completed' for r in raw['warmup']['request_records'])
        monitor.phase='resident_cooling';raw['resident_thermal_control']=cool(f,monitor)
        raw['postwarm_idle']=monitor.idle_window('postwarm_idle',2.);monitor.check()
        assert f['safe_factor']*sum(v['owned_used_mib'] for v in raw['postwarm_idle'])/len(raw['postwarm_idle'])<=f['budget_mib']
        print('MEASUREMENT_START 600s',flush=True)
        raw['phases']['measurement']=phase(runtime,config,slot,f,mapping,'measurement',monitor)
        raw['postmeasurement_idle']=monitor.idle_window('postmeasurement_idle',2.);monitor.check()
        raw['status']='COMPLETED'
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='FAILED';raw['errors'].append(repr(exc))
        if monitor.active_phase:
            raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
    finally:
        try:raw['shutdown']=runtime.close()
        except Exception as exc:raw['errors'].append('shutdown:'+repr(exc))
        if raw.get('aborted_phase'):
            for r in raw['aborted_phase']['request_records']:
                if r['status'] in ['planned','queued','submitted']:r.update(status='incomplete',error='Safety abort; full planned cohort retained')
            annotate_records(raw['aborted_phase']['request_records'])
        monitor.close();raw['errors']+=runtime.errors+monitor.errors
        raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,interval_seconds=.5)
        raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        try:journal.close()
        except Exception as exc:raw['errors'].append('journal:'+repr(exc))
        raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors']
            and len(raw.get('shutdown',{}))==5 and all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw['shutdown'].values()))
        raw['elapsed_seconds']=time.perf_counter()-begin
        save('gpu_validation_raw.json',sanitized(raw))
        save('input_integrity_after.json',protected_check())
        print('RUN_END',json.dumps(dict(status=raw['status'],valid_session=raw['valid_session'],release=raw['gpu_release']['verified'],errors=raw['errors'])),flush=True)
    return 0 if raw['valid_session'] else 1

if __name__=='__main__':raise SystemExit(main())
