"""Re-run every legal placement, preserving the Module24A fitted predictor."""
from .common import *
from module24.replay import replay
import module24.audit as audit

class HighRates(dict):
    # The existing optimizer uses LOW only to obtain the set of covered tasks.
    # All stages cover exactly the same tasks; iteration performs HIGH alone.
    def __missing__(self,key):
        if key=='LOW':return self['HIGH']
        raise KeyError(key)

def main():
    assert not (OUT/'high_cpu_recheck.json').exists(),'Completed CPU audit cannot be overwritten'
    save('historical_metadata_before.json',historical_metadata())
    inventory=[]
    # Read every Module24A output, including retained raw failures and trajectories.
    for path in sorted(HISTORY.rglob('*')):
        if not path.is_file():continue
        if path.suffix=='.json':read(path)
        elif path.suffix=='.csv':list(csv.DictReader(path.open()))
        else:path.read_bytes()
        inventory.append(ref(path))
    inventory.extend(ref(p) for p in [ROOT/'module20/optimizer.py',ROOT/'module24/audit.py',ROOT/'module24/replay.py',
        ROOT/'module24/calibration.py',ROOT/'results/module20_utility_optimizer/utility_definition.json',
        ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'])
    save('read_only_input_inventory.json',inventory)
    d,refs,weights,model=model_inputs();old=read(HISTORY/'reoptimized_three_phase.json')['stages']['HIGH']
    rates=old['rates'];assert rates==RATES['HIGH']
    audit.RATES=HighRates(HIGH=rates);audit.save=save
    result=audit.search(d,refs,weights,model,label='high_cpu_search')
    st=result['stages']['HIGH'];winner=st['winner'];original=old['new_winner']
    p,paths=replay(d,original['deployment'],original['routing'],rates,BUDGET,refs,weights,model)
    pure=st['route_class_best']['S-only'];gonly=st['route_class_best']['G-only']
    pcontrol_rr=copy.deepcopy(original['routing']);pcontrol_rr['asr']={S:rates['asr']}
    control,cpaths=replay(d,original['deployment'],pcontrol_rr,rates,BUDGET,refs,weights,model)
    f=original['ASR_G_fraction'];gp=paths.get(identity([G,'asr']),{});sp=paths.get(identity([S,'asr']),{})
    meaningful=bool(.01<f<.99 and gp.get('service',{}).get('count',0)>=3 and sp.get('service',{}).get('count',0)>=3)
    gap=p['U']-pure['prediction']['U'] if pure and p['U'] is not None else None
    tolerance=1e-8
    passed=bool(winner and st['choice']=='G/S' and meaningful and p['eligible'] and gap is not None and gap>tolerance
        and abs(winner['prediction']['U']-p['U'])<=tolerance and abs(p['U']-original['prediction']['U'])<=tolerance)
    evidence=[]
    for m,n in original['deployment'].items():
        for t in d['models'][m]['supported_tasks']:
            parameter=model.parameter(original['deployment'],m,n,t)
            evidence.append(dict(model=m,replicas=n,task=t,calibration_parameter=parameter,
                selected_colocation_empirically_supported=parameter['scope']=='EXACT_COMBINATION',
                historical_replica_capacity_g=d['capacity_factors'][m][str(n)],
                historical_replica_profile=d.get('model_evidence',{}).get(m),
                selected_colocation_dual_replica_capacity_measured=False if n==2 else None))
    result.update(passed=passed,gate_if_failed='HIGH_CPU_NOT_REPRODUCED',original_frozen_candidate=original,
        original_candidate_recomputed_prediction=p,original_candidate_paths=paths,
        original_candidate_numerical_reproduction_tolerance_U=tolerance,meaningful_ASR_split=meaningful,
        meaningful_split_check=dict(fraction=f,periodic_G_requests=gp.get('service',{}).get('count'),
            periodic_S_requests=sp.get('service',{}).get('count'),gap_vs_best_S_only=gap),
        same_residency_ASR_S_only=dict(routing=pcontrol_rr,prediction=control,paths=cpaths),
        evidence=evidence,historical_Module24A_gate_unchanged=ref(HISTORY/'module24a_gate.json'),
        user_authorized_HIGH_only_independent_experiment=True,LOW_MID_pattern_is_not_failure=True,
        no_refit_no_QPS_weight_pool_or_replica_constraint_changes=True)
    save('high_cpu_recheck.json',result)
    frozen=dict(status='CPU_FROZEN' if passed else 'CPU_NOT_REPRODUCED',CPU_passed=passed,
        pool=MODELS,stage='HIGH',target_task='asr',rates=rates,offered_total_qps=sum(rates.values()),
        deployment=original['deployment'],routing=original['routing'],ASR_G_fraction=f,
        comparator_routing=pcontrol_rr,quality_admission={m:d['models'][m] for m in MODELS},
        budget_mib=BUDGET,safe_factor=d['safe_factor'],predicted_safe_memory_mib=p['safe_memory_mib'],
        legal_replica_counts=d['capacity_factors'],memory_tables=d['memory_table_mib'],
        weights=weights,refs_ms=refs,prediction=p,control_prediction=control,
        predicted_delta_U_same_residency=p['U']-control['U'] if p['U'] is not None and control['U'] is not None else None,
        predictor=ref(HISTORY/'residual_model_final_development.json'),cpu_recheck=ref(OUT/'high_cpu_recheck.json'),
        historical_input_inventory=ref(OUT/'read_only_input_inventory.json'),
        dual_replica_evidence_before_preflight=evidence,
        independent_ranking_samples_before_this_experiment=0,
        protocol=dict(arrival='independent_per_task_Poisson; independent frozen route/replica uniforms',
            warmup_seconds=10.,measurement_seconds=120.,drain_seconds=60.,post_idle_seconds=2.,
            formal_seeds=[24201,24202,24203],preflight_seed=24200,
            arm_order=[['proposed','control'],['control','proposed'],['proposed','control']],
            max_formal_GPU_runs=6,no_outcome_retry=True,no_seed_screening=True,
            fixed_window_same_as_Module24A=True,
            sparse_LID_warning='Expected 3.6 requests/120s; per-run small-n tails and missing task coverage must be reported; no seed/window extension',
            preflight_capacity='Unchanged Module11B closed-loop one outstanding request/worker: each duplicated family 1 active vs 2 active, all eight workers resident; then frozen full HIGH Poisson workload',
            capacity_seconds_per_condition=10.,capacity_metric_scope='Saturated service diagnostic with other models idle, not workload QPS adjustment or utility evidence; full HIGH competition checked separately',
            success='Both arms legal/stable/memory-safe, actual nonzero G/S, all three paired delta_U>0.01 and two-sided paired t95 CI lower>0 for clear validation; otherwise inconclusive',
            stop='CPU reproduction failure, replica preflight hard failure, GPU/runtime/memory failure, or comparable negative/nonpositive gain; retain all raw records. Unstable U=null.'),
        stage_C_permitted_only_after_stage_B=True,Module24C_auto_execution=False)
    save('frozen_high_manifest.json',frozen)
    print('HIGH_CPU_PASS',passed,'fraction',f,'U',p['U'],'gap_to_S',gap,flush=True)

if __name__=='__main__':main()
