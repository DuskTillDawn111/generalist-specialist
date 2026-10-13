"""CPU checks and preregistration before the single GPU session."""
import sys
sys.dont_write_bytecode=True
import copy
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');sys.path.insert(0,str(ROOT))
from instrumentation import OUT,LoggedRuntime,router_state,restore_router,guards,ref,save
from dynamic_router import DynamicRouter

class MemoryJournal:
    def __init__(self):self.events=[]
    def emit(self,kind,**values):self.events.append(dict(kind=kind,**values))

def main():
    assert not (OUT/'controlled_manifest.json').exists()
    f=json.loads((ROOT/'results/module28_dynamic_split/frozen_candidate_manifest.json').read_text())
    p=f['parameters'];tests=[]
    assert ref(OUT/'dynamic_router.py')['sha256']==ref(ROOT/'results/module28_dynamic_split/dynamic_router.py')['sha256']
    tests.append('Router source byte identical; parameters copied without tuning')
    # Exercise real collector with one already received RESULT; no worker/CUDA imports started.
    journal=MemoryJournal();rt=LoggedRuntime(None,{},journal)
    record=dict(request_id='test:1',phase='test',model=p['S_worker'],task='lid',status='submitted',
                enqueued_time_ns=time.perf_counter_ns()-1000000,actual_submit_time_ns=time.perf_counter_ns()-900000)
    rt.pending['test:1']=record;rt.visible['test:1']=record;rt.router=DynamicRouter(p)
    payload=dict(request_id='test:1',task_type='lid',worker_latency_ms=70.,queue_wait_ms=3.)
    rt.events.put((p['S_worker'],'RESULT',payload));rt.collect_stop.set();rt._collect()
    assert record['status']=='completed' and record['collector_commit_time_ns']>=record['completion_time_ns']
    with rt.lock:
        now=time.perf_counter_ns();obs=rt.observation(now,True)
    assert len(obs['harvested_completions'])==1 and obs['pending']=={}
    assert obs['router_before_decide']['service']==[dict(worker=p['S_worker'],task='lid',ewma_ms=70.)]
    assert journal.events[0]['kind']=='COLLECTOR_COMMIT'
    tests.append('Native collector AST unchanged except appended commit observer; RESULT commit and harvest tested')
    pending={p['S_worker']:[dict(task='lid',request_id=f'test:{i}') for i in range(13)]}
    r=DynamicRouter(p);before=router_state(r);route,d=r.decide('test:G',100.,pending,True,True)
    assert route==p['G_worker'] and d['reason']=='G_LOWER_SYSTEM_MARGINAL_COST'
    replay=restore_router(DynamicRouter,p,before)
    assert replay.decide('test:G',100.,pending,True,True)==(route,d)
    assert router_state(replay)==router_state(r)
    gs=guards(p,before,d,100.,True,True)
    assert all(v=='PASS' for v in gs.values())
    tests.append('State serialization restores exact decision and guards for supported first-step condition')
    # Confirm guard bookkeeping does not mistake favorable costs for an allowed route.
    blocked=restore_router(DynamicRouter,p,before)
    busy=copy.deepcopy(pending);busy[p['G_worker']]=[dict(task='visual_qa',request_id='test:VQA')]
    route,d=blocked.decide('test:blocked',100.,busy,True,True)
    assert route==p['S_worker'] and d['reason']=='G_BACKGROUND_OR_OTHER_WORK_PENDING'
    tests.append('Observed G background work blocks route; not replaced by assumed-idle condition')
    before=router_state(r)
    r.observe_completion(dict(request_id='test:G',model=p['G_worker'],task='lid',worker_latency_ms=1000.),101.)
    assert r.fallback_until==131. and not r.mode and r.next_g_time==101.+p['cooldown_seconds']
    replay=restore_router(DynamicRouter,p,before)
    replay.observe_completion(dict(request_id='test:G',model=p['G_worker'],task='lid',worker_latency_ms=1000.),101.)
    assert router_state(replay)==router_state(r)
    tests.append('Prediction error, EWMA, cooldown and fallback remain original and replayable')
    slot=copy.deepcopy(next(s for s in f['exploration_slots'] if s['arm']=='dynamic' and s['arrival_profile']=='steady'))
    slot.update(run_id='M29_stage2_one_dynamic_validation',stage='MODULE29_FUNCTIONAL_VALIDATION')
    counts={}
    for label,source in slot['trajectories'].items():
        assert ref(source['path'])['sha256']==source['sha256']
        trajectory=json.loads(Path(source['path']).read_text())
        assert all(0<=e['offset_seconds']<trajectory['duration'] for e in trajectory['events'])
        assert all(a['offset_seconds']<=b['offset_seconds'] for a,b in zip(trajectory['events'],trajectory['events'][1:]))
        counts[label]={t:sum(e['task']==t for e in trajectory['events']) for t in f['rates']}
        assert all(v>0 for v in counts[label].values())
    assert slot['measurement_seconds']==600. and slot['seed']==28101
    assert f['parameters']==json.loads((ROOT/'results/module28_dynamic_split/router_parameters.json').read_text())
    for k in ['quality_policy','utility_definition','utility_reference']:
        assert ref(f[k]['path'])['sha256']==f[k]['sha256']
    for item in f['implementation']:
        assert ref(item['path'])['sha256']==item['sha256'],item['path']
    for worker in f['plan']['registry'].values():
        assert Path(worker['python']).exists() and Path(worker['script']).exists()
    tests.append('Original worker/quality/utility/source hashes, fixed trajectories and positive six-task cohorts verified')
    from run_controlled import protected_check
    integrity=protected_check()
    save('cpu_preflight.json',dict(passed=True,checks=tests,trajectory_counts=counts,
                                 input_integrity=integrity,GPU_inferences_started=0))
    manifest=dict(status='FROZEN_BEFORE_SINGLE_GPU_VALIDATION',frozen_at_unix_ns=time.time_ns(),
        authorization='User requested Module29 stage2 complete logging and ONE controlled real GPU split validation',
        slot=slot,parameters=copy.deepcopy(p),router_source=ref(OUT/'dynamic_router.py'),
        original_router_source=ref(ROOT/'results/module28_dynamic_split/dynamic_router.py'),
        original_frozen_configuration=ref(ROOT/'results/module28_dynamic_split/frozen_candidate_manifest.json'),
        implementation=[ref(OUT/name) for name in ['dynamic_router.py','instrumentation.py','run_controlled.py','prepare_validation.py','analyze_validation.py']],
        models=f['models'],registry=f['plan']['registry'],budget_mib=f['budget_mib'],safe_factor=f['safe_factor'],
        thermal_protocol=f['thermal_protocol'],stability_thresholds_source=ref(ROOT/'performance_model/local_calibration/stability_v2.py'),
        admission_source=f['catalog'],quality_policy=f['quality_policy'],
        original_gate_preserved='DYNAMIC_EVIDENCE_GAP',original_S_only_stability_preserved='unstable',
        maximum_GPU_sessions=1,maximum_formal_seconds=600.,warmup_seconds=30.,drain_timeout_seconds=60.,
        path_probe='Original sequential per-path prime, three requests/path; excluded from routing proof and formal cohort',
        experiment_scope='FUNCTIONAL_EXECUTION_AND_LOGGING_VALIDATION; not comparison or independent replication',
        trajectory_selection='Reuse exact original C_steady frozen cohort from offline audit; no new seed/load/grid selection',
        route_logic='Original decide, estimate, observe_completion bytes and parameters unchanged',
        observation_extension='Native collector AST preserved with appended locked commit observer; atomic host snapshot and decide; asynchronous sidecar disk writes',
        observation_time='Actual perf_counter_ns sampled after obtaining runtime lock; separately log planned scheduled time',
        no_future_policy_inputs=True,native_START_END_policy_inputs=False,
        logging_acceptance={'complete_target_pre_harvest_pre_decide_post_decide_state':True,
            'all_decisions_replay_exactly':True,'no_post_observation_completion_in_decision':True,
            'all_arrivals_and_collector_commits_logged':True},
        execution_acceptance={'formal_LID_G_actual_success_at_least':1,'formal_LID_S_actual_success_at_least':1,
            'successful_G_target_native_service_start_end_and_queue_records':True,
            'successful_S_service_and_queue_records':True,'retrospective_S_G_overlap_analysis_available':True,
            'clean_workers_and_GPU_release':True,'memory_constraints_unchanged':True},
        caution='Instrumentation success and causal interference/utility benefit are separate; within-run overlap strata are descriptive and confounded',
        legal_U=None,delta_U=None,optimizer_modified=False,next_experiment_automatic=False,
        stop='One session then report. No retries/threshold tuning/load change to force diversion. Runtime error/OOM/memory/isolation failure aborts; instability retained unchanged.')
    save('controlled_manifest.json',manifest)
    print(json.dumps(dict(passed=True,output_directory=str(OUT),checks=tests,counts=counts)))

if __name__=='__main__':main()
