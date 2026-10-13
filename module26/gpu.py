"""Six preregistered native GPU runs; no outcome retry, no prediction fitting."""
from .common import *
import subprocess,threading
from module22.gpu import phase,PoissonConfig
from module11b.measure import prime
from module16.gpu import ProcessMonitor,memory_values
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.metrics import annotate_records
from replica_scaling.telemetry import snapshot,release

def thermal():
    fields=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,power.draw,power.limit',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    return dict(temperature_gpu_c=float(fields[0]),sm_clock_mhz=float(fields[1]),memory_clock_mhz=float(fields[2]),
        pstate=fields[3].strip(),power_draw_w=float(fields[4]),power_limit_w=float(fields[5]))

def cool(slot,manifest):
    begin=time.monotonic();samples=[];consecutive=0;policy=manifest['thermal_protocol']
    while time.monotonic()-begin<=policy['max_wait_seconds']:
        value=snapshot();value.update(thermal());samples.append(value)
        if value['processes'] or value['used_mib']>128:raise RuntimeError('PRELAUNCH_FOREIGN_GPU_PROCESS_OR_MEMORY')
        consecutive=consecutive+1 if value['temperature_gpu_c']<=policy['prelaunch_temperature_max_c'] else 0
        if consecutive>=policy['consecutive_idle_samples_required']:
            save('thermal_prelaunch/'+slot['run_id']+'.json',dict(passed=True,samples=samples));return value
        time.sleep(policy['between_samples_seconds'])
    save('thermal_prelaunch/'+slot['run_id']+'.json',dict(passed=False,samples=samples))
    raise RuntimeError('PRELAUNCH_THERMAL_STATE_NOT_COMPARABLE')

class ThermalMonitor(ProcessMonitor):
    def __init__(self,runtime,slot):
        super().__init__(runtime,.5,BUDGET);self.slot=slot;self.last_progress=0.
    def sample(self):
        value=super().sample();value.update(thermal())
        active=getattr(self,'active_phase',None)
        if active and time.monotonic()-self.last_progress>=30:
            self.last_progress=time.monotonic();records=active['request_records'];offered=[r for r in records if r.get('actual_submit_time_ns')]
            done=[r for r in records if r['status']=='completed'];now_ns=time.perf_counter_ns()
            progress=dict(run_id=self.slot['run_id'],phase=self.phase,
                elapsed_seconds=(now_ns-active['planned_start_time_ns'])/1e9,
                planned_seconds=active['planned_duration_seconds'],planned=len(records),submitted=len(offered),completed=len(done),
                outstanding=len(offered)-len(done),temperature_gpu_c=value['temperature_gpu_c'],used_mib=value['owned_used_mib'])
            save('live_progress.json',progress)
        return value

def safe_idle(raw,key,monitor,manifest):
    raw[key]=monitor.idle_window(key,2.);monitor.check()
    actual=manifest['safe_factor']*float(np.mean([r['owned_used_mib'] for r in raw[key]]))
    if actual>BUDGET+1e-6:raise RuntimeError(f'SAFE_MEMORY_INFEASIBLE: {key}={actual} > {BUDGET}')

def run(slot,manifest):
    path=OUT/'raw_runs'/(slot['run_id']+'.json');assert not path.exists(),'No outcome retry/overwrite'
    baseline=cool(slot,manifest);plan=manifest['plan']
    assert baseline['uuid']==plan['GPU']['uuid'] and baseline['name']==plan['GPU']['name'] and baseline['driver']==plan['GPU']['driver']
    for k,v in plan['environment_overrides'].items():os.environ[k]=v
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'),'Controller must retain historical physical GPU visibility'
    mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
    routes={t:{i:q/slot['deployment'][m] for m,q in row.items() for i,f in mapping.items() if f==m} for t,row in slot['routing'].items()}
    config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,arrival_process='poisson',
        random_seed=slot['seed'],warmup_seconds=10.,measurement_seconds=slot['measurement_seconds'],
        startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,routing_source='Original24A_native_Poisson_independent_route_replica_uniforms')
    config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
    runtime=DeploymentRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=ThermalMonitor(runtime,slot)
    raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,worker_identities={},replica_to_family=mapping,
        transport_config=config.as_dict(),frozen_manifest=ref(OUT/'frozen_replication_manifest.json'))
    begin=time.perf_counter();monitor.start()
    try:
        runtime.start();monitor.check()
        raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
        assert len({v['pid'] for v in raw['worker_identities'].values()})==5
        raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
        print('WARMUP_START',slot['run_id'],flush=True);raw['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor)
        assert all(r['status']=='completed' for r in raw['warmup']['request_records']),'Warmup incomplete'
        safe_idle(raw,'postwarm_idle',monitor,manifest)
        print('MEASUREMENT_START',slot['run_id'],'2400 seconds',flush=True)
        raw['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor)
        safe_idle(raw,'postmeasurement_idle',monitor,manifest);raw['status']='COMPLETED'
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='FAILED';raw['errors'].append(repr(exc))
        if getattr(monitor,'active_phase',None):raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
    finally:
        try:raw['shutdown']=runtime.close()
        except Exception as exc:raw['errors'].append('shutdown: '+repr(exc))
        if raw.get('aborted_phase'):
            for r in raw['aborted_phase']['request_records']:
                if r['status'] in ('planned','queued','submitted'):r.update(status='incomplete',error='Safety abort; entire planned cohort retained')
            annotate_records(raw['aborted_phase']['request_records'])
        monitor.close();raw['errors']+=runtime.errors+monitor.errors
        raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,interval_seconds=.5,
            peak_scope='0.5s sampled NVML owned-process high-water; instantaneous allocator peaks unavailable')
        raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and
            all(v['exit_code']==0 and not v['forced'] and v['bye_received'] for v in raw.get('shutdown',{}).values()))
        raw['elapsed_seconds']=time.perf_counter()-begin;save('raw_runs/'+path.name,raw)
    return raw

def main():
    manifest=read(OUT/'frozen_replication_manifest.json');assert read(OUT/'positive_probe_audit.json')['passed']
    assert not (OUT/'replication_campaign.json').exists(),'Completed campaign cannot restart'
    verify(manifest['implementation']+[manifest['catalog'],manifest['utility_definition'],manifest['utility_reference'],manifest['plan']['sample_pool']])
    observations=[];stop=None;executed=[]
    for slot in manifest['slots']:
        print('REPLICATION_START',slot['run_id'],now(),flush=True)
        try:raw=run(slot,manifest)
        except Exception as exc:
            save('prelaunch_failures/'+slot['run_id']+'.json',dict(slot=slot,error=repr(exc),no_measurement=True))
            stop=dict(run_id=slot['run_id'],reason='PRELAUNCH_FAILED',error=repr(exc));break
        executed.append(slot['run_id']);path=OUT/'raw_runs'/(slot['run_id']+'.json');ob=observe(raw,path)
        observations.append(ob);save('replication_observations.json',observations)
        mem=memory_values(raw,manifest['safe_factor'])
        print('REPLICATION_END',slot['run_id'],'eligible',ob['eligible'],'unstable',ob['hard_unstable'],'U',ob['U'],'errors',raw['errors'],now(),flush=True)
        if not ob['eligible'] or not mem['passed'] or not raw['gpu_release']['verified']:
            stop=dict(run_id=slot['run_id'],reason='UNSTABLE_OR_ILLEGAL_OR_RUNTIME_FAILURE',eligible=ob['eligible'],
                hard_unstable=ob['hard_unstable'],memory=mem,errors=raw['errors']);break
        pair=[o for o in observations if o['repetition']==slot['repetition']]
        if len(pair)==2:
            a=next(o for o in pair if 'proposed' in o['run_id']);b=next(o for o in pair if 'control' in o['run_id'])
            print('PAIRED_DELTA',slot['repetition'],a['U']-b['U'],'retain regardless of sign',flush=True)
    save('replication_campaign.json',dict(at_utc=now(),executed=executed,stop=stop,planned=6,
        no_outcome_retry=True,no_negative_outcome_early_stop=True,
        remaining=[s['run_id'] for s in manifest['slots'] if s['run_id'] not in executed],new_predictor=False,new_pool_search=False))
    print('STAGE_B_CAMPAIGN_COMPLETE',len(executed),'stop',stop,flush=True)

if __name__=='__main__':main()
