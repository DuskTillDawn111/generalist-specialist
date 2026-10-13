"""Freeze six independent replication runs before any new GPU request."""
from .common import *
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool

def main():
    audit=read(OUT/'positive_probe_audit.json');assert audit['eligible_for_independent_replication']
    assert not (OUT/'frozen_replication_manifest.json').exists(),'No protocol replacement'
    d,refs,weights=inputs();original=read(AROOT/'frozen_experiment_manifest.json')
    plan=copy.deepcopy(original['plan']);plan['measurement_seconds']=2400.
    # Identical native code, interpreter, model, inference options and thread settings.
    launcher=OUT/'launchers/whisper_large_v3.py';launcher.parent.mkdir(parents=True,exist_ok=True)
    launcher.write_bytes(Path(plan['registry'][S]['script']).read_bytes());plan['registry'][S]['script']=str(launcher)
    oldslots={s['arm']:s for s in original['slots']};orders=[['control','proposed'],['proposed','control'],['control','proposed']]
    slots=[];support=[]
    for rep,seed in enumerate([26101,26102,26103],1):
        trajectories={}
        for label,duration in [('warmup',10.),('measurement',2400.)]:
            pool=SamplePool(plan['sample_pool']['path'],list(RATES['HIGH']),seed)
            events=arrivals(RATES['HIGH'],duration,seed,label)
            for event in events:event['item']=pool.next(event['task'])
            name=f'trajectories/rep{rep}_{label}.json';save(name,dict(rates=RATES['HIGH'],duration=duration,seed=seed,label=label,events=events))
            trajectories[label]=ref(OUT/name)
            if label=='measurement':support.append(dict(repetition=rep,seed=seed,planned_task_counts=dict(Counter(e['task'] for e in events)),
                planned_G_ASR=sum(e['task']=='asr' and e['routing_uniform']>=.82 for e in events)))
        for arm in orders[rep-1]:
            source=oldslots['GS18' if arm=='proposed' else 'S-only'];slot=copy.deepcopy(source)
            slot.update(run_id=f'high_{arm}_rep{rep}',arm=arm,repetition=rep,seed=seed,measurement_seconds=2400.,trajectories=trajectories,
                diagnostic_only=False,capacity_diagnostic_only=False,optimizer_selected_configuration=False,
                training_role='NEW_INDEPENDENT_REPLICATION_NEVER_TRAIN',historical_source_slot=source['run_id'])
            slots.append(slot)
    # Request counts are reported, never seed-selected to improve observed outcomes.
    assert all(r['planned_task_counts']['asr']>=1000 for r in support)
    manifest=dict(status='FROZEN_BEFORE_NEW_GPU',at_utc=now(),Stage_A=ref(OUT/'positive_probe_audit.json'),plan=plan,slots=slots,
        original_manifest=ref(AROOT/'frozen_experiment_manifest.json'),catalog=ref(AROOT/'selected_pool_catalog.json'),
        utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),
        utility_reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),weights=weights,refs_ms=refs,
        budget_mib=BUDGET,safe_factor=d['safe_factor'],frozen_pool=list(oldslots['GS18']['deployment']),
        identical_checkpoint_replica_QPS_route_quality_utility_reference=True,ASR_G_fraction=.18,
        measurement_seconds=2400.,original_measurement_seconds=120.,warmup_seconds=10.,drain_seconds=60.,
        window_extension_user_authorized=True,window_extension_reason='Sparse LID .03 QPS requires long window; target ASR tails must not depend on few observations',
        arrival_protocol='Original module21 independent per-task Poisson; matched inputs/uniforms within pair, new seeds26101-26103',
        arm_order=orders,balanced_order_not_outcome_selected=True,trajectory_support=support,
        thermal_protocol=dict(prelaunch_temperature_max_c=38.,consecutive_idle_samples_required=3,max_wait_seconds=300.,
            between_samples_seconds=2.,same_fresh_worker_start_prime_warmup_and_postwarm_idle=True,
            no_clock_or_power_limit_changes=True,temperature_clocks_power_sampled_every=.5,
            postwarm_pair_temperature_difference_max_c=5.,
            thermal_limit_failure_policy='retain all evidence; stop launch or invalidate comparability, never outcome retry'),
        replication_gate=dict(minimum_pairs=3,all_delta_U_over=.01,all_pairs_legal_stable_comparable=True,
            approximate_two_sided_student_t95_lower_over_zero_required=True,
            confidence_limitations='n=3 normal-effects t interval is assumption-dependent; sign-permutation min one-sided p=.125, not definitive population proof',
            per_task_P95_min_requests=59,why59='At least59 iid samples needed for max order statistic to bound P95 with97.5 percent one-sided confidence; intervals remain conditional under time dependence',
            no_outcome_retry=True,no_negative_gain_early_stop=True),
        stop_rule='Stop unsafe/runtime/error/incomplete/actual-unstable/memory/isolation/release failure. Complete all three pairs despite negative delta; no QPS/routes/count/model tuning or extra trials.',
        Stage_C=dict(permitted_only_if_replicated=True,scope='Fixed five-model pool only; original measured counts/free routes and memory budget; no new predictor',
            baseline='Unmodified Module20 Generator and utility scorer',new_scorer='Frozen Module25 predictor without fit/parameter changes',
            route_sensitivity_grid=[.16,.17,.18,.19,.20],CPU_only_sensitivity_not_empirical_route_validation=True),
        implementation=[ref(p) for p in [ROOT/'module26/common.py',ROOT/'module26/audit.py',ROOT/'module26/freeze.py',ROOT/'module26/gpu.py',
            ROOT/'module22/gpu.py',ROOT/'module21/trajectory.py',ROOT/'module11b/measure.py',ROOT/'module16/gpu.py',ROOT/'module11b/telemetry.py',
            ROOT/'joint_deploy/measurement/deployment_runtime.py',ROOT/'joint_deploy/measurement/metrics.py',ROOT/'performance_model/local_calibration/stability_v2.py',
            *[Path(v['script']) for v in plan['registry'].values()]]],
        historical_development_pair_used_as_independent=False,new_predictor_developed=False,new_pool_search=False,Module27_automatic=False)
    save('frozen_replication_manifest.json',manifest)
    print('FROZEN',len(slots),'runs',manifest['measurement_seconds'],'seconds_each','support',support,flush=True)

if __name__=='__main__':main()

