"""Fresh frozen HIGH validation; native worker/transport/arrival semantics."""
from .common import *
import os,argparse
from collections import Counter
from module22.gpu import PoissonConfig,phase
from module11b.measure import prime
from module16.gpu import ProcessMonitor,memory_values
from joint_deploy.measurement.deployment_runtime import DeploymentRuntime
from joint_deploy.measurement.load_generator import SamplePool
from joint_deploy.measurement.run_measurement import new_record
from joint_deploy.measurement.metrics import annotate_records
from module21.trajectory import arrivals
from replica_scaling.telemetry import snapshot,release
from module24.calibration import extract

def setup():
    manifest=read(OUT/'frozen_high_manifest.json');assert manifest['CPU_passed']
    if 'plan' in manifest:return manifest
    verify(read(OUT/'read_only_input_inventory.json'))
    gpu=snapshot();save('hardware_before_preflight.json',gpu)
    assert not gpu['processes'] and gpu['used_mib']<=128 and gpu['total_mib']>=BUDGET,'GPU isolation/physical capacity failed'
    plan=copy.deepcopy(read(HISTORY/'frozen_experiment_manifest.json')['plan'])
    plan['GPU']=gpu;plan['warmup_seconds']=10.;plan['measurement_seconds']=120.
    launcher=OUT/'launchers/whisper_large_v3.py';launcher.parent.mkdir(parents=True,exist_ok=True)
    launcher.write_bytes((HISTORY/'launchers/whisper_large_v3.py').read_bytes())
    plan['registry'][S]['script']=str(launcher)
    slots=[]
    for rep,seed in enumerate([24200,24201,24202,24203]):
        trajectories={}
        for label,duration in [('warmup',10.),('measurement',120.)]:
            pool=SamplePool(plan['sample_pool']['path'],list(manifest['rates']),seed)
            events=arrivals(manifest['rates'],duration,seed,label)
            for event in events:event['item']=pool.next(event['task'])
            name=f'trajectories/rep{rep}_{label}.json'
            save(name,dict(rates=manifest['rates'],duration=duration,seed=seed,label=label,events=events))
            trajectories[label]=ref(OUT/name)
        arms=['preflight'] if rep==0 else manifest['protocol']['arm_order'][rep-1]
        for arm in arms:
            routing=manifest['comparator_routing'] if arm=='control' else manifest['routing']
            pred=manifest['control_prediction'] if arm=='control' else manifest['prediction']
            slots.append(dict(run_id=f'high_{arm}_rep{rep}',stage='HIGH',arm=arm,repetition=rep,seed=seed,
                deployment=manifest['deployment'],routing=routing,lambda_rps=manifest['rates'],
                safe_memory_budget_mib=BUDGET,measurement_seconds=120.,trajectories=trajectories,
                CPU_predicted_U=pred['U'],CPU_rho=pred['rho'],CPU_nominal_feasible=pred['eligible'],
                optimizer_selected_configuration=arm!='control',diagnostic_only=rep==0,
                capacity_diagnostic_only=rep==0,arrival_process='independent_per_task_Poisson',
                training_role='SAFETY_PREFLIGHT_ONLY' if rep==0 else 'PROSPECTIVE_INDEPENDENT_VALIDATION_NEVER_TRAIN'))
    sources=[ROOT/'module24b/gpu.py',ROOT/'module24b/freeze.py',ROOT/'module24b/common.py',ROOT/'module22/gpu.py',
        ROOT/'module21/trajectory.py',ROOT/'module11b/measure.py',ROOT/'module16/gpu.py',ROOT/'module11b/telemetry.py',
        ROOT/'module13b/worker.py',ROOT/'joint_deploy/measurement/deployment_runtime.py',ROOT/'joint_deploy/worker_protocol.py',
        ROOT/'performance_model/local_calibration/stability_v2.py']
    sources += [Path(v['script']) for v in plan['registry'].values()]
    sources += [ROOT/'joint_deploy/measurement'/name for name in ['worker_process.py','load_generator.py','schema.py','metrics.py','run_measurement.py'] if (ROOT/'joint_deploy/measurement'/name).exists()]
    manifest.update(status='FROZEN_BEFORE_GPU',plan=plan,slots=slots,implementation=[ref(p) for p in sources],
        input_pool=ref(plan['sample_pool']['path']),
        raw_evidence='All planned arrivals, failures, queues, service samples and owned-process telemetry retained',
        formal_sessions_including_startup_attempts_max=6,no_development_trajectory_reuse=True,
        original_Module24A_gate=ref(HISTORY/'module24a_gate.json'))
    save('frozen_high_manifest.json',manifest)
    return manifest

def safe_idle(raw,key,monitor,manifest):
    raw[key]=monitor.idle_window(key,2.);monitor.check()
    actual=manifest['safe_factor']*float(np.mean([r['owned_used_mib'] for r in raw[key]]))
    if actual>BUDGET+1e-6:raise RuntimeError(f'SAFE_MEMORY_INFEASIBLE: {key}={actual:.6f} MiB > frozen {BUDGET:.6f}')

def capacity_window(runtime,plan,slot,mapping,monitor,label,active,task,duration,output):
    """One outstanding request per active serial worker, drain excluded."""
    pool=SamplePool(plan['sample_pool']['path'],[task],slot['seed'])
    records=[];pending={};refills=[];monitor.phase=label
    start=time.perf_counter_ns();end=start+round(duration*1e9)
    output.update(request_records=records,active_workers=active,task=task,
        window=dict(start_time_ns=start,planned_end_time_ns=end,planned_duration_seconds=duration))
    monitor.active_phase=dict(label=label,planned_start_time_ns=start,planned_end_time_ns=end,
        planned_duration_seconds=duration,request_records=records)
    while time.perf_counter_ns()<end:
        now=time.perf_counter_ns()
        with runtime.lock:
            for worker in active:
                previous=pending.get(worker)
                if previous and previous['status'] in ('queued','submitted'):continue
                if previous:
                    assert previous['status']=='completed','Capacity request failed'
                    refills.append(now-previous['completion_time_ns'])
                item=pool.next(task);r=new_record(label,len(records),task,worker,item,now)
                records.append(r);pending[worker]=r;runtime.enqueue(r,item['sample'])
        time.sleep(.0001);monitor.check();assert not runtime.errors,runtime.errors
        if any(w.process.poll() is not None for w in runtime.workers.values()):raise RuntimeError('Capacity worker exited')
        if any(now-r['scheduled_time_ns']>20e9 for r in pending.values() if r['status'] in ('queued','submitted')):
            raise RuntimeError('Capacity request exceeded frozen 20s guard')
    observed_end=time.perf_counter_ns();monitor.phase=label+'_drain';runtime.drain(records,60.);annotate_records(records);monitor.check()
    dd=[r for r in records if r['status']=='completed' and r['completion_time_ns']<observed_end]
    elapsed=(observed_end-start)/1e9
    output.update(completed_in_window=len(dd),throughput_rps=len(dd)/elapsed,
        per_replica_throughput_rps={worker:sum(r['model']==worker for r in dd)/elapsed for worker in active},
        service_distribution=summarize_samples([r['worker_latency_ms'] for r in dd]),
        queue_distribution=summarize_samples([r['worker_queue_wait_ms'] for r in dd]),
        refill_delay_ms=summarize_samples(np.array(refills)/1e6),all_completed_after_drain=all(r['status']=='completed' for r in records),
        capacity_scope='All eight selected replicas resident; other models idle. Full-HIGH GPU competition is measured separately; no extrapolated joint g2.',
        max_outstanding_per_active_worker=1,drain_excluded_from_throughput=True)
    output['window'].update(end_time_ns=observed_end,duration_seconds=elapsed,drain_end_time_ns=time.perf_counter_ns())
    monitor.active_phase=None
    assert output['all_completed_after_drain'],'Capacity diagnostic did not drain'

def run(slot,manifest):
    target=OUT/'raw_runs'/(slot['run_id']+'.json');assert not target.exists(),'No run retry or overwrite'
    verify(manifest['implementation']+[manifest['input_pool'],manifest['predictor'],manifest['original_Module24A_gate']])
    for k,v in manifest['plan']['environment_overrides'].items():os.environ[k]=v
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'),'Controller must retain original GPU visibility'
    baseline=snapshot();assert not baseline['processes'] and baseline['used_mib']<=128 and baseline['uuid']==manifest['plan']['GPU']['uuid']
    plan=manifest['plan'];mapping={f'{m}#{i}':m for m,n in slot['deployment'].items() for i in range(n)}
    routes={t:{i:q/slot['deployment'][m] for m,q in rr.items() for i,mm in mapping.items() if m==mm} for t,rr in slot['routing'].items()}
    config=PoissonConfig(deployment={i:1 for i in mapping},workload_rps=slot['lambda_rps'],routing_rps=routes,
        arrival_process='poisson',random_seed=slot['seed'],warmup_seconds=10.,measurement_seconds=120.,
        startup_timeout_seconds=300.,drain_timeout_seconds=60.,shutdown_timeout_seconds=15.,
        routing_source='module24b_frozen_Poisson_independent_route_replica_uniforms')
    config.validate({i:plan['models'][m]['supported_tasks'] for i,m in mapping.items()})
    runtime=DeploymentRuntime(config,{i:plan['registry'][m] for i,m in mapping.items()});monitor=ProcessMonitor(runtime,.5,BUDGET)
    raw=dict(slot=slot,status='STARTING',errors=[],phases={},baseline_GPU=baseline,worker_identities={},
        replica_to_family=mapping,transport_config=config.as_dict(),frozen_manifest=ref(OUT/'frozen_high_manifest.json'))
    start=time.perf_counter();monitor.start()
    try:
        runtime.start();monitor.check()
        raw['worker_identities']={i:dict(family=m,pid=runtime.workers[i].process.pid,READY=runtime.ready[i]) for i,m in mapping.items()}
        assert len({w['pid'] for w in raw['worker_identities'].values()})==8
        raw['path_probe_records']=prime(runtime,config,plan,mapping,slot);monitor.check()
        if slot['arm']=='preflight':
            safe_idle(raw,'postprime_idle',monitor,manifest)
            raw['capacity_windows']={}
            for family,task in [(TEXT,'text'),(IMAGE,'image'),(OCR,'ocr')]:
                active=[f'{family}#0',f'{family}#1']
                conditions=[('one_active',active[:1]),('two_active',active)]
                if family==IMAGE:conditions.reverse()
                for mode,workers in conditions:
                    label=f'capacity_{family}_{mode}';raw['capacity_windows'][label]={}
                    print('CAPACITY_START',family,mode,flush=True)
                    capacity_window(runtime,plan,slot,mapping,monitor,label,workers,task,10.,raw['capacity_windows'][label])
                    print('CAPACITY_END',family,mode,raw['capacity_windows'][label]['throughput_rps'],flush=True)
            safe_idle(raw,'postcapacity_idle',monitor,manifest)
        print('WARMUP_START',slot['run_id'],flush=True)
        raw['warmup']=phase(runtime,config,slot,plan,mapping,'warmup',monitor)
        assert all(r['status']=='completed' for r in raw['warmup']['request_records']),'Warmup incomplete'
        safe_idle(raw,'postwarm_idle',monitor,manifest)
        print('MEASUREMENT_START',slot['run_id'],flush=True)
        raw['phases']['measurement']=phase(runtime,config,slot,plan,mapping,'measurement',monitor)
        safe_idle(raw,'postmeasurement_idle',monitor,manifest)
        raw['status']='COMPLETED'
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='FAILED';raw['errors'].append(repr(exc))
        if getattr(monitor,'active_phase',None):
            raw['aborted_phase']=monitor.active_phase;raw['aborted_phase']['abort_time_ns']=time.perf_counter_ns()
    finally:
        try:raw['shutdown']=runtime.close()
        except Exception as exc:raw['errors'].append('shutdown: '+repr(exc))
        if raw.get('aborted_phase'):
            for r in raw['aborted_phase']['request_records']:
                if r['status'] in ('planned','queued','submitted'):r.update(status='incomplete',error='Safety abort; full planned cohort retained')
            annotate_records(raw['aborted_phase']['request_records'])
        monitor.close();raw['errors']+=runtime.errors+monitor.errors
        raw['telemetry']=dict(samples=monitor.samples,errors=monitor.errors,interval_seconds=.5,
            peak_scope='0.5s sampled NVML owned-process peak; allocator instantaneous peak unavailable')
        raw['gpu_release']=release(baseline,[w.process.pid for w in runtime.workers.values() if w.process])
        raw['valid_session']=bool(raw['status']=='COMPLETED' and raw['gpu_release']['verified'] and not raw['errors'] and
            all(w['exit_code']==0 and not w['forced'] and w['bye_received'] for w in raw.get('shutdown',{}).values()))
        raw['elapsed_seconds']=time.perf_counter()-start;save('raw_runs/'+target.name,raw)
    return raw

def observation(raw):
    d,refs,weights,model=model_inputs();path=OUT/'raw_runs'/(raw['slot']['run_id']+'.json')
    return extract(raw,24,path,refs,weights,d=d,split='SAFETY_PREFLIGHT_ONLY' if raw['slot']['arm']=='preflight' else 'PROSPECTIVE_INDEPENDENT_VALIDATION')

def preflight():
    assert not (OUT/'replica_colocation_preflight.json').exists(),'Completed preflight cannot be retried'
    manifest=setup();slot=next(s for s in manifest['slots'] if s['arm']=='preflight')
    print('PREFLIGHT_START',slot['run_id'],flush=True);raw=run(slot,manifest);obs=observation(raw)
    memory=memory_values(raw,manifest['safe_factor']);scalings=[]
    for family,task in [(TEXT,'text'),(IMAGE,'image'),(OCR,'ocr')]:
        one=raw.get('capacity_windows',{}).get(f'capacity_{family}_one_active',{})
        two=raw.get('capacity_windows',{}).get(f'capacity_{family}_two_active',{})
        mu1=one.get('throughput_rps');mu2=two.get('throughput_rps')
        scalings.append(dict(model=family,task=task,predicted_g2=manifest['legal_replica_counts'][family]['2'],
            measured_g2=mu2/mu1 if mu1 and mu2 else None,one_active_capacity_rps=mu1,two_active_capacity_rps=mu2,
            per_replica_capacity_rps=two.get('per_replica_throughput_rps'),
            one_active_service=one.get('service_distribution'),two_active_service=two.get('service_distribution'),
            fixed_HIGH_demand_rps=manifest['routing'][task][family],
            standalone_colocation_capacity_exceeds_HIGH_demand=mu2>manifest['routing'][task][family] if mu2 else None,
            full_HIGH_interference_supported_by='Separate 120s frozen workload completion, worker busy/queues; not these idle-background saturated capacities',
            raw_one=one,raw_two=two))
    ph=raw.get('phases',{}).get('measurement',{});full=bool(ph and all(r['status']=='completed' for r in ph['request_records']))
    passed=bool(raw['valid_session'] and memory['passed'] and full and not obs['hard_unstable'] and
        all(s['standalone_colocation_capacity_exceeds_HIGH_demand'] for s in scalings) and raw['gpu_release']['verified'])
    value=dict(passed=passed,gate_if_failed='HIGH_REPLICA_CAPACITY_UNSUPPORTED',raw=ref(OUT/'raw_runs'/(slot['run_id']+'.json')),
        independent_utility_evidence=False,scope='Safety and current HIGH capacity only; never a ranking/utility validation sample',
        selected_eight_independent_worker_PIDs=raw['worker_identities'],memory=memory,replica_scalings=scalings,
        full_HIGH_observation=obs,errors=raw['errors'],valid_session=raw['valid_session'],gpu_release=raw['gpu_release'],
        prediction_updated=False,route_or_residency_changed=False,
        capacity_joint_g2_uncertainty='10s single/two-active tests isolate replica contention with all models resident. Joint HIGH capacity supported only at fixed offered rate; no steady-state saturated joint g2 or confidence interval claimed.')
    save('replica_colocation_preflight.json',value)
    print('PREFLIGHT_END','PASS',passed,'memory',memory,'unstable',obs['hard_unstable'],'errors',raw['errors'],flush=True)

def formal():
    manifest=read(OUT/'frozen_high_manifest.json');assert manifest['CPU_passed']
    assert read(OUT/'replica_colocation_preflight.json')['passed'],'Stage B must pass before formal GPU'
    assert not (OUT/'formal_campaign.json').exists(),'Completed campaign cannot be retried'
    observations=[];stop=None
    for slot in [s for s in manifest['slots'] if s['arm']!='preflight']:
        print('FORMAL_START',slot['run_id'],flush=True);raw=run(slot,manifest);obs=observation(raw)
        observations.append(obs);save('formal_observations.json',observations)
        print('FORMAL_END',slot['run_id'],'eligible',obs['eligible'],'unstable',obs['hard_unstable'],'U',obs['U'],'errors',raw['errors'],flush=True)
        memory=memory_values(raw,manifest['safe_factor'])
        if raw['errors'] or not memory['passed'] or not raw['gpu_release']['verified']:
            stop=dict(run_id=slot['run_id'],reason='GPU_RUNTIME_OR_MEMORY_FAILURE',errors=raw['errors']);break
        pair=[o for o in observations if o['repetition']==slot['repetition']]
        if len(pair)==2:
            a=next(o for o in pair if 'proposed' in o['run_id']);b=next(o for o in pair if 'control' in o['run_id'])
            if a['U'] is None or b['U'] is None:
                stop=dict(repetition=slot['repetition'],reason='UNSTABLE_OR_INCOMPARABLE_PAIR',proposed_U=a['U'],control_U=b['U']);break
            delta=a['U']-b['U'];print('PAIRED_DELTA',slot['repetition'],delta,flush=True)
            if delta<0:
                stop=dict(repetition=slot['repetition'],reason='NEGATIVE_UTILITY_GAIN',delta_U=delta);break
    save('formal_campaign.json',dict(completed_runs=[o['run_id'] for o in observations],stop=stop,
        planned_maximum=6,executed=len(observations),configuration_unchanged=True,predictor_never_refit=True,
        Module24C_executed=False,remaining_runs=[s['run_id'] for s in manifest['slots'] if s['arm']!='preflight' and s['run_id'] not in {o['run_id'] for o in observations}]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','formal']);args=parser.parse_args()
    preflight() if args.stage=='preflight' else formal()
