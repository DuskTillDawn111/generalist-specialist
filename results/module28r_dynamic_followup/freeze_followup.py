"""One-shot pre-GPU workload, routing, order, and safety freeze."""
from audit_followup import *
import copy, time
from replica_scaling.telemetry import snapshot

def main():
    assert not (OUT/'frozen_workloads.json').exists()
    audit=json.loads((OUT/'stage_a_audit.json').read_text())
    original=json.loads((OLD/'frozen_candidate_manifest.json').read_text())
    f=copy.deepcopy(original)
    gpu=snapshot()
    assert not gpu['processes'] and gpu['used_mib']<=128
    assert gpu['uuid']==original['GPU']['uuid'], 'No cross-GPU development substitution'
    f['GPU']=gpu;f['plan']['GPU']=gpu
    # Immutable router implementation, parameters, pool, replicas and admissions.
    (OUT/'dynamic_router.py').write_bytes((OLD/'dynamic_router.py').read_bytes())
    (OUT/'selected_pool_catalog.json').write_bytes(Path(original['catalog']['path']).read_bytes())
    f['catalog']=ref(OUT/'selected_pool_catalog.json')
    f['independent_slots']=[];f['maximum_validation_attempts']=0;f['maximum_total_attempts']=6
    f['maximum_formal_gpu_seconds']=3600;f['maximum_exploration_attempts']=6
    f['at_unix_ns']=time.time_ns();f['module']='28R'
    rates_a=copy.deepcopy(original['rates']);rates_b=copy.deepcopy(rates_a);rates_b['lid']=audit['lower_load_rule']['selected_lid_qps']
    s=original['parameters']['S_worker'].rsplit('#',1)[0];g='qwen'
    # A: exact original HIGH trajectory. B: preserve same background events,
    # input items and uniforms; deterministic target time dilation 11.5/10.5,
    # truncate at fixed window end. Thus B is a lower-rate Poisson trajectory,
    # paired without new seed selection or input changes.
    trajectories={}
    for scene,rates in [('HIGH',rates_a),('LOW',rates_b)]:
        trajectories[scene]={}
        for label in ['warmup','measurement']:
            base=original['exploration_slots'][0]['trajectories'][label]
            tr=json.loads(Path(base['path']).read_text())
            if scene=='LOW':
                for e in tr['events']:
                    if e['task']=='lid':e['offset_seconds']*=rates_a['lid']/rates_b['lid']
                tr['events']=[e for e in tr['events'] if e['offset_seconds']<tr['duration']]
                tr['events'].sort(key=lambda e:(e['offset_seconds'],e['task']))
            tr.update(rates=rates,source_original=base,transformation='IDENTITY' if scene=='HIGH' else 'LID-only time dilation 11.5/10.5; fixed end truncation; other5 identical')
            fn=f'trajectories/{scene}_{label}.json';save(fn,tr)
            trajectories[scene][label]=ref(OUT/fn)
    # Balanced interleaving fixed before observing any 28R GPU result.
    order=[('HIGH','s_only'),('LOW','dynamic'),('HIGH','fixed10'),('LOW','s_only'),('HIGH','dynamic'),('LOW','fixed10')]
    slots=[]
    for i,(scene,arm) in enumerate(order,1):
        rates=rates_a if scene=='HIGH' else rates_b
        slot=copy.deepcopy(original['exploration_slots'][0])
        rr=copy.deepcopy(slot['routing']);fraction=.1 if arm=='fixed10' else .05 if arm=='dynamic' else 0
        rr['lid']={s:rates['lid']*(1-fraction)}
        if fraction:rr['lid'][g]=rates['lid']*fraction
        slot.update(run_id=f'R{i}_{scene}_{arm}',stage='C',arm=arm,arrival_profile=scene,point_id=scene,
            scenario=scene,lambda_rps=rates,routing=rr,trajectories=trajectories[scene],G_fraction=fraction,run_order=i)
        slots.append(slot)
    f['exploration_slots']=slots
    f['safety']=dict(max_host_outstanding_total=256,max_host_outstanding_per_worker=192,
        max_owned_gpu_mib=f['budget_mib'],max_total_gpu_mib=f['budget_mib'],min_host_available_ram_gib=4,
        max_gpu_temperature_c=85,max_scheduled_dispatch_lag_seconds=5,max_formal_overrun_seconds=10,
        max_worker_no_completion_seconds=30,watchdog_check_seconds=.5,drain_timeout_seconds=60,
        campaign_policy='Continue on original finite-window unstable classifier if complete/error-free/fully drained/memory legal. Stop entire campaign on safety, infrastructure, malformed/invalid output or unclean release. No retry or replacement.')
    f['failure_policy']=f['safety']['campaign_policy']
    f['selection']=dict(delta_over=.01,all_three_legal_stable=True,both_dynamic_paths_nonzero=True,
        min_task_count=72,thermal_match_c=5.,independent_validation='NOT_AUTHORIZED_THIS_ROUND')
    f['G_only']=dict(executed=False,legal_stable_not_demonstrated=True,
        HIGH_native_mean_demand_lower_estimate=11.5*383.52/1000,
        LOW_native_mean_demand_lower_estimate=10.5*383.52/1000,
        reason='Module27 observed G LID mean ~383.52ms implies target alone ~4.41/4.03 handler-s/s, plus VQA. Capacity screen rejects same single-G residency. This is a historical screen, not a measured G-only outcome; no superiority over all legal routes.')
    f['implementation']=[ref(OUT/name) for name in ['audit_followup.py','freeze_followup.py','run_followup.py','report_followup.py','dynamic_router.py']]
    f['reused_implementation']=original['implementation']
    workloads=dict(status='FROZEN_BEFORE_GPU',at_unix_ns=f['at_unix_ns'],module='28R',model_pool=original['models'],
        deployment=slots[0]['deployment'],budget_mib=f['budget_mib'],safe_factor=f['safe_factor'],catalog=f['catalog'],
        quality_policy=f['quality_policy'],utility_definition=f['utility_definition'],utility_reference=f['utility_reference'],
        weights=f['weights'],refs=f['refs'],router_source=ref(OUT/'dynamic_router.py'),router_parameters=f['parameters'],
        fixed_G_fraction=.1,fixed_fraction_source=original['fixed_comparator'],low_selection_rule=audit['lower_load_rule'],
        scenarios={scene:dict(rates=rates,trajectories=trajectories[scene],seed=28101,measurement_seconds=600,warmup_seconds=30,
            drain_timeout_seconds=60,all_tasks_positive=all(q>0 for q in rates.values())) for scene,rates in [('HIGH',rates_a),('LOW',rates_b)]},
        order=order,safety=f['safety'],original_stability_thresholds=DEFAULT_THRESHOLDS,original_manifest=ref(OLD/'frozen_candidate_manifest.json'),
        G_only=f['G_only'],thermal_protocol=f['thermal_protocol'],no_QPS_or_parameter_changes_after_freeze=True)
    save('frozen_workloads.json',workloads);save('frozen_candidate_manifest.json',f)
    save('gpu_run_manifest.json',dict(status='FROZEN_BEFORE_GPU',at_unix_ns=f['at_unix_ns'],GPU=gpu,
        slots=slots,maximum_attempts=6,measurement_seconds_each=600,total_formal_max_seconds=3600,
        workloads=ref(OUT/'frozen_workloads.json'),protocol=ref(OUT/'frozen_candidate_manifest.json'),
        no_outcome_retry=True,no_automatic_next_module=True,no_optimizer_modification=True))
    print('FROZEN',order,'rates',rates_a,rates_b)

if __name__=='__main__':main()
