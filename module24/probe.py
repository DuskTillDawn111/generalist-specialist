"""One short matched HIGH pair for development only, never Module24B.

Uses the existing native Poisson harness/workers without editing historical
modules. Residencies and all non-ASR routes match. No outcome retry.
"""
from .common import *

def main():
    assert (OUT/'original_optimizer_audit.json').exists()
    assert (OUT/'offline_validation.json').exists(),'Offline validation must finish before any GPU probe'
    assert not (OUT/'probe_campaign.json').exists(),'No rerun of completed probes'
    from replica_scaling.telemetry import snapshot
    try:gpu=snapshot()
    except Exception as exc:
        save('probe_campaign.json',dict(status='GPU_UNAVAILABLE',attempts=0,reason=repr(exc)))
        csvsave('high_colocation_probe.csv',[dict(status='NOT_RUN_GPU_UNAVAILABLE',actual_U=None,reason=repr(exc))],['run_id','arm','actual_U'])
        return
    save('probe_hardware_preflight.json',gpu)
    if gpu['processes'] or gpu['used_mib']>128 or gpu['total_mib']<BUDGET:
        reason='Current GPU isolation or available physical memory does not support diagnostic launch'
        save('probe_campaign.json',dict(status='PRELAUNCH_UNAVAILABLE',attempts=0,reason=reason,hardware=gpu))
        csvsave('high_colocation_probe.csv',[dict(status='NOT_RUN_PRELAUNCH',actual_U=None,reason=reason)],['run_id','arm','actual_U'])
        return
    from module21.trajectory import arrivals
    from joint_deploy.measurement.load_generator import SamplePool
    from module24.calibration import extract
    from module24.replay import replay
    import module22.gpu as harness
    d,refs,weights=inputs();plan=copy.deepcopy(read(ROOT/'results/module22_mid_high_utility/frozen_experiment_manifest.json')['plan'])
    plan['GPU']=gpu;plan['models']=d['models'];plan['warmup_seconds']=10.;plan['measurement_seconds']=120.
    plan['environment_overrides'].update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    # Pure launcher wrapper: same Module13B checkpoint/dtype/adapter/handler.
    launcher=OUT/'launchers/whisper_large_v3.py';launcher.parent.mkdir(parents=True,exist_ok=True)
    argv=[str(ROOT/'module13b/worker.py'),'--model-id','openai/whisper-large-v3','--task','asr',
          '--directory',str(ROOT/'models/module13b/openai--whisper-large-v3')]
    # The historical profiling worker advertises model/task metadata, but the
    # deployment transport requires an explicit tasks list. Identical to the
    # existing Module14D launcher: change READY metadata only, not inference.
    launcher.write_text('import ast, sys\nfrom pathlib import Path\nsys.argv='+repr(argv)+
        '\np=Path(sys.argv[0]);tree=ast.parse(p.read_text())\n'+
        'for node in ast.walk(tree):\n'+
        ' if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=="run_worker":\n'+
        '  node.value.args[2]=ast.parse("dict(metadata,tasks=[\'asr\'])",mode="eval").body\n'+
        'ast.fix_missing_locations(tree)\nexec(compile(tree,str(p),"exec"),{"__file__":str(p),"__name__":"__main__"})\n')
    plan['registry'][S]=dict(python=str(ROOT/'envs/module13b_transformers/bin/python'),script=str(launcher))
    plan['registry']={m:plan['registry'][m] for m in MODELS}
    x={m:1 for m in MODELS};rates=RATES['HIGH'];seed=24101;trajectories={}
    for label,duration in [('warmup',10.),('measurement',120.)]:
        pool=SamplePool(plan['sample_pool']['path'],list(rates),seed)
        events=arrivals(rates,duration,seed,label)
        for event in events:event['item']=pool.next(event['task'])
        name=f'trajectories/high_rep1_{label}.json';save(name,dict(rates=rates,duration=duration,seed=seed,label=label,events=events))
        trajectories[label]=ref(OUT/name)
    slots=[]
    for arm,f in [('S-only',0.),('GS18',.18)]:
        rr=route(d,x,rates,[0.,0.,f]);p,ps=replay(d,x,rr,rates,BUDGET,refs,weights)
        slots.append(dict(run_id='high_'+arm.lower().replace('-','_')+'_diagnostic_rep1',stage='HIGH',arm=arm,
            repetition=1,seed=seed,deployment=x,routing=rr,lambda_rps=rates,safe_memory_budget_mib=BUDGET,
            measurement_seconds=120.,trajectories=trajectories,CPU_predicted_U=p['U'],CPU_rho=p['rho'],
            CPU_nominal_feasible=p['eligible'],capacity_diagnostic_only=True,diagnostic_only=True,
            optimizer_selected_configuration=False,arrival_process='independent_per_task_Poisson'))
    sources=[ROOT/'module22/gpu.py',ROOT/'module21/trajectory.py',ROOT/'module13b/worker.py']
    sources+=[Path(v['script']) for v in plan['registry'].values()]
    retained=[]
    failed_path=OUT/'raw_runs/high_s_only_startup_failure_attempt1.json'
    if failed_path.exists():
        failed=read(failed_path)
        assert not failed['phases'] and not failed.get('path_probe_records'),'No measured outcome retry'
        retained=[failed]
        save('probe_startup_repair_amendment.json',dict(reason='READY omitted tasks; profiling adapter needs only explicit ASR metadata for DeploymentRuntime',
            no_request_measurement_before_repair=True,retained_failure=ref(failed_path),
            retained_original_manifest=ref(OUT/'frozen_experiment_manifest.startup_v1.json'),
            inference_QPS_routes_budget_weights_unchanged=True,maximum_S_attempts_including_startup=2,
            artifact_save_scope_repaired='Bind harness.save to Module24 save; move only new diagnostic failure into Module24 output',
            no_existing_historical_file_changed=True))
        slots[0]['run_id']='high_s_only_diagnostic_attempt2'
    manifest=dict(status='FROZEN_BEFORE_GPU',plan=plan,slots=slots,inputs=[ref(p) for p in sources],
        utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),
        utility_references=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),
        catalog=ref(OUT/'selected_pool_catalog.json'),offline_validation=ref(OUT/'offline_validation.json'),
        purpose='Diagnostic service/queue mismatch; fixed S-only vs exploratory 18% ASR G/S; neither declares optimizer choice or formal utility success',
        max_sessions_per_route=2,user_max_sessions_per_route=2,planned_measurement_sessions_per_route=1,measurement_seconds=120.,warmup_seconds=10.,drain_seconds=60.,
        stop_rule='No retries. Isolation/runtime/memory/release failure stops campaign; hard queue instability retains full cohort and no comparable U. No further repetitions.',
        training_role='DEVELOPMENT_ONLY; excluded from all final independent validation',Module24B=False)
    save('frozen_experiment_manifest.json',manifest)
    harness.OUT=OUT;harness.BUDGET=BUDGET;harness.save=save
    results=list(retained);observations=[];csvrows=[]
    if retained:
        ob=extract(retained[0],24,failed_path,refs,weights,d=d,split='DEVELOPMENT');observations.append(ob)
        csvrows.append(dict(run_id=retained[0]['slot']['run_id'],arm='S-only',kind='STARTUP_FAILURE',
            status=retained[0]['status'],actual_U=None,diagnostic_only=True,errors=retained[0]['errors'],raw_path=str(failed_path),
            split='DEVELOPMENT_ONLY_NOT_INDEPENDENT',no_measurement=True))
    for slot in slots:
        print('PROBE_START',slot['run_id'],flush=True)
        raw=harness.run(slot,plan);results.append(raw)
        pth=OUT/'raw_runs'/(slot['run_id']+'.json')
        ob=extract(raw,24,pth,refs,weights,d=d,split='DEVELOPMENT');observations.append(ob)
        old=ob.get('prediction_A',{});ph=raw['phases'].get('measurement',{})
        csvrows.append(dict(run_id=slot['run_id'],arm=slot['arm'],kind='SYSTEM',diagnostic_only=True,
            optimizer_selected=False,split='DEVELOPMENT_ONLY_NOT_INDEPENDENT',status=raw['status'],
            actual_U=ob['U'],old_predicted_U=old.get('U'),
            actual_minus_old_U=ob['U']-old['U'] if ob['U'] is not None and old.get('U') is not None else None,
            eligible=ob['eligible'],hard_unstable=ob['hard_unstable'],censored=ob['censored'],
            system_mean_ms=ob.get('pooled_mean_ms') if not ob['censored'] and not ob['hard_unstable'] else None,
            system_p95_ms=ob.get('pooled_p95_ms') if not ob['censored'] and not ob['hard_unstable'] else None,
            completed_subset_mean_DIAGNOSTIC_ms=ob.get('pooled_mean_ms'),completed_subset_p95_DIAGNOSTIC_ms=ob.get('pooled_p95_ms'),
            offered=ob.get('offered'),completed=ob.get('completed'),safe_memory_mib=ob.get('safe_memory_mib'),
            sampled_peak_mib=ob.get('sampled_peak_mib'),hardware_uuid=gpu['uuid'],
            matched_trajectory_sha256=trajectories['measurement']['sha256'],raw_path=str(pth),raw_sha256=ref(pth)['sha256']))
        rs=ph.get('request_records',[]);w=ph.get('window',{})
        for worker,ident in raw.get('worker_identities',{}).items():
            md=[r for r in rs if r['model']==worker];done=[r for r in md if r['status']=='completed'];m=raw['replica_to_family'][worker]
            start=w.get('start_unix_ns',0);end=start+int(120e9)
            busy=sum(max(0,min(r['worker_response']['end_ns'],end)-max(r['worker_response']['start_ns'],start)) for r in done)/120e9 if w else None
            wr=next((v for v in ob.get('workers',[]) if v['worker']==worker),{})
            for t in (sorted({r['task'] for r in md}) or ['IDLE']):
                dd=[r for r in done if r['task']==t];svc=[r['worker_latency_ms'] for r in dd];qs=[r['worker_queue_wait_ms'] for r in dd]
                csvrows.append(dict(run_id=slot['run_id'],arm=slot['arm'],kind='WORKER_TASK',worker=worker,model=m,task=t,
                    diagnostic_only=True,split='DEVELOPMENT_ONLY_NOT_INDEPENDENT',state=wr.get('state','IDLE'),
                    offered=sum(r['task']==t for r in md),completed=len(dd),service_distribution=summarize_samples(svc),
                    queue_distribution=summarize_samples(qs),busy_fraction_LOWER_BOUND_if_censored=busy,
                    end_backlog=wr.get('end_backlog'),peak_backlog=wr.get('peak_backlog'),
                    final_backlog=wr.get('final_backlog'),tail_slope_rps=wr.get('tail_slope_rps'),censored=ob['censored']))
        # All resident ASR path probes are retained even when a route is idle.
        for m in [G,S]:
            dd=[r for r in raw.get('path_probe_records',[]) if r['task']=='asr' and raw['replica_to_family'][r['model']]==m and r['status']=='completed']
            csvrows.append(dict(run_id=slot['run_id'],arm=slot['arm'],kind='RESIDENT_LOW_LOAD_PATH_PROBE',model=m,task='asr',
                diagnostic_only=True,service_distribution=summarize_samples([r['worker_latency_ms'] for r in dd]),
                count=len(dd),scope='Sequential co-resident probe, not concurrent HIGH service distribution'))
        csvsave('high_colocation_probe.csv',csvrows,['run_id','arm','kind','model','task','actual_U'])
        save('probe_observations.json',observations)
        print('PROBE_END',slot['run_id'],'eligible',ob['eligible'],'unstable',ob['hard_unstable'],'U',ob['U'],flush=True)
        if raw['errors'] or not raw.get('gpu_release',{}).get('verified'):
            break
    save('probe_campaign.json',dict(status='COMPLETED_DIAGNOSTICS_WITH_RETAINED_STARTUP_FAILURE' if len(results)==len(slots)+len(retained) else 'STOPPED_RETAINED_FAILURE',
        attempts=len(results),planned_measurements=2,maximum_per_route=2,retained_startup_failures=len(retained),
        per_route_attempts={arm:sum(r['slot']['arm']==arm for r in results) for arm in ['S-only','GS18']},
        observed_normal_utility_sessions=sum(r['eligible'] for r in observations),
        actual_hard_unstable=sum(r['hard_unstable'] for r in observations),formal_Module24B_runs=0,
        final_independent_validation=False,model_and_workload_unchanged=True))

if __name__=='__main__':main()
