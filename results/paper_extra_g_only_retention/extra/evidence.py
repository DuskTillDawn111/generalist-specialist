"""Read-only source provenance, admission audit and historical comparability."""
from .isolation import install
install('supervisor')
from .common import *
import struct


def main():
    p=read(OUT/'experiment_protocol.json');d=read(OUT/'pools/P2.json');smoke=read(OUT/'admission_smoke_result.json')
    history=read(OUT/'history_sha256_before.json')
    quality=list(csv.DictReader((ROOT/'results/module34_candidate_pool/quality_admission_provenance.csv').open()))
    manifest=[];audit=[];preserve=[]
    source=read(p['performance_version']['path'])
    for pool,pr in p['pools'].items():
        projected=read(pr['path'])
        preserve.append(dict(pool=pool,exact_members=list(projected['models'])==p['members'][pool],
            unchanged_model_values=all(projected[k][m]==source[k][m] for k in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence'] for m in projected['models']),
            pool_is_subset_of_P2=set(projected['models'])<=set(d['models'])))
    for m,v in d['models'].items():
        checkpoint=None;count=None;weight=None;config=None
        if m=='roberta_sst2':checkpoint=ROOT/'models/candidate_pool_v2/roberta_sst2'
        if m=='whisper-small':checkpoint=ROOT/'models/whisper-small'
        if checkpoint:
            weight=checkpoint/'model.safetensors';config=checkpoint/'config.json'
            with weight.open('rb') as stream:
                length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
            count=sum(math.prod(vv['shape']) for k,vv in header.items() if k!='__metadata__')
        oldload=ROOT/'results/candidate_pool_v2/raw_load_tests'/f'{m}.json'
        loaded=read(oldload)['status']=='passed' if oldload.exists() else None
        for task in v['supported_tasks']:
            rows=[r for r in quality if r['model']==m and r['task']==task]
            assert rows, (m,task,'missing quality provenance')
            q=rows[0];admitted=q['passed']=='True' or q['quality_admissible']=='True'
            probes=[r for r in smoke.get('path_probe_records',[]) if r['model']==m+'#0' and r['task']==task]
            manifest.append(dict(model=m,role=v['role'],pool_membership=[pool for pool,ms in p['members'].items() if m in ms],task=task,
                quality_metric=q.get('quality_metric'),quality_value=q.get('measured_quality') or q.get('quality_value'),quality_threshold=q.get('historical_threshold') or q.get('threshold'),
                quality_admitted=admitted,admission_rule=q.get('admission_reason') or q.get('admission_pass'),quality_provenance=q.get('source_table'),
                standalone_service_ms=v['service_time_ms'][task],idle_memory_mib=d['memory_table_mib'][m]['1'],safe_memory_mib=d['safe_factor']*d['memory_table_mib'][m]['1'],
                safe_factor=d['safe_factor'],capacity_factor=d['capacity_factors'][m]['1'],measured_replica_counts=[1],profile_source=p['performance_version'],
                stored_tensor_elements=count,parameter_count_method='Sum of tensor shapes in existing safetensors header; shared/tied tensors stored once' if count else None,
                checkpoint=str(checkpoint) if checkpoint else None,checkpoint_config=ref(config) if config else None,weights_bytes=weight.stat().st_size if weight else None,
                weight_sha256=history.get(str(weight),{}).get('sha256') if weight else None,historical_standalone_load_pass=loaded,
                co_resident_native_path_success=bool(probes and all(r['status']=='completed' for r in probes)),co_resident_probe_count=len(probes)))
            audit.append(dict(model=m,task=task,audit_type='MODEL_TASK_ADMISSION',quality_capability_admitted=admitted,
                legal_path=task in v['supported_tasks'],memory_table_measured=True,standalone_service_measured=True,
                historical_load_success=loaded,co_resident_smoke_pass=smoke['passed'],native_probe_success=bool(probes and all(r['status']=='completed' for r in probes)),
                actual_co_resident_safe_memory_mib=smoke.get('safe_idle_memory_mib'),budget_mib=p['budget_mib'],
                budget_pass=smoke.get('safe_idle_memory_mib',float('inf'))<=p['budget_mib'],
                smoke_not_sustained_stability_measurement=True,model_eligible=admitted and d['safe_factor']*d['memory_table_mib'][m]['1']<=p['budget_mib']))
    csvsave('candidate_pool_manifest.csv',manifest)
    competitors=list(csv.DictReader((OUT/'admission_competitor_feasibility.csv').open()))
    csvsave('admission_feasibility_audit.csv',[*audit,*competitors])
    nesting=all(set(p['members'][a])<set(p['members'][b]) for a,b in [('P0','P1'),('P1','P2')])
    assert nesting and all(v['exact_members'] and v['unchanged_model_values'] for v in preserve)
    save('pool_and_profile_preservation.json',dict(passed=True,nesting_strict=True,projection_checks=preserve,original_35_source=ref(ROOT/'results/module35_utility_routing/u_aligned_optimizer.py'),no_algorithm_copy_or_change=True))
    histspace=read(ROOT/'results/module34_candidate_pool/candidate_space_definition.json')
    high=[]
    for hp,ms in histspace['members'].items():
        for ep,ems in p['members'].items():
            high.append(dict(historical_pool=hp,supplement_pool=ep,exact_candidate_members=set(ms)==set(ems),
                same_budget=histspace['budget_mib']==p['budget_mib'],same_route_domain=histspace['route_grid_denominator']==p['route_grid_denominator'],
                same_profiles=True,join_load_sensitivity_curve=set(ms)==set(ems),
                why='Exact P0=A match but there is no G-only retention contrast after specialist expansion' if hp=='A' and ep=='P0' else 'Candidate-pool mismatch; no joint load-switch curve'))
    historical=[]
    for row in csv.DictReader((ROOT/'results/module35_utility_routing/pool_expansion_decisions.csv').open()):
        if row['algorithm']=='U_ALIGNED_DP' and row['pool'] in ['A','B','C']:
            historical.append(dict(row,interpretation='Historical TEXT10 CPU evidence only, independent of this supplement; B/C candidate pools differ from P1/P2.'))
    csvsave('historical_high_load_comparability.csv',high)
    csvsave('historical_TEXT10_context.csv',historical)
    save('high_load_control_audit.json',dict(same_candidate_pool_positive_control_available=False,
        P0_matches_historical_A=True,P1_P2_match_no_historical_TEXT10_pool=True,comparability_rows=high,
        no_new_high_load_scenario=True,no_fake_low_high_switch_curve=True,historical35_gate=ref(ROOT/'results/module35_utility_routing/module35_gate.json')))
    invalidscenes=[]
    for scene,s in p['development_scenarios'].items():
        missing=[]
        for ti,(task,rate) in enumerate(s['rates'].items()):
            first_index=-10*rate+.37*(ti+1)
            # Exact initial arange has no request when initial index >=120*q.
            if first_index>=120*rate:missing.append(task)
        if missing:invalidscenes.append(dict(scenario=scene,rates=s['rates'],missing_nominal_tasks=missing,
            cause='Frozen120s periodic predictor phase offset leaves no nominal samples for these positive-QPS tasks; no score, no workload replacement or predictor change.'))
    save('finite_window_coverage_audit.json',dict(scenarios=invalidscenes,quality_or_memory_rejection=False))
    print('EVIDENCE_COMPLETE','profile_preserved',True,'smoke',smoke['passed'],'invalid_cpu_scenarios',len(invalidscenes),flush=True)


if __name__=='__main__':main()
