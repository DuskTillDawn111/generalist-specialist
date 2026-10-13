"""Final CPU-only evidence validation for Module 8D-3."""
import csv
import json
import math
from pathlib import Path
from module8d3_diagnosis import OUT, ROOT, D2, C, B, CAT, NAMES, BOUNDS, POLICY, read, ref, file_hash, save_json, validate_raw, verify_inputs, digest
from candidate_pool_audit.provenance import environment_snapshot
from replica_scaling.telemetry import snapshot


def run():
    verify_inputs()
    protected=read(OUT/'diagnostic_analysis_protocol.json')['protected_8D2_artifacts']
    assert file_hash(protected['path'])==protected['sha256']
    old=read(protected['path'])['artifacts']
    for r in old.values(): assert file_hash(r['path'])==r['sha256'],r['path']
    for name in ['diagnostic_analysis_protocol.json','crossover_plan.json','crossover_stage2_gate_policy.json','crossover_plan_stage2.json']:
        p=OUT/name;assert file_hash(p)==(OUT/(name+'.sha256')).read_text().strip(),name
    for name in ['crossover_plan.json','crossover_stage2_gate_policy.json','crossover_plan_stage2.json']:
        for k in ['implementation','existing_plan_unchanged','stage1_evidence','gate_policy']:
            r=read(OUT/name).get(k)
            if r: assert file_hash(r['path'])==r['sha256'],r['path']
    plans=[read(OUT/'crossover_plan.json'),read(OUT/'crossover_plan_stage2.json')]
    results=[read(OUT/'crossover_results.json'),read(OUT/'crossover_stage2_results.json')]
    admitted={(r['task'],r['model']) for r in read(B/'quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    assert len(admitted)==21
    proof=[];raw_paths=[]
    for plan,res in zip(plans,results):
        assert res['status']=='COMPLETE';assert res['new_timed_runs']==plan['maximum_new_timed_runs']
        assert len(res['new_states'])==len(plan['states'])
        expected={(s['workload'],s['variant']):s for s in plan['states']}
        for state in res['new_states']:
            p=expected[state['workload'],state['variant']];ident=state['identity']
            assert ident==p['identity'];assert not state['search']
            rt=state['runtime'];assert rt['valid_repetition_count']==3 and rt['execution_status']=='completed'
            assert rt['stability_state_v2']=='steady'
            for m,x in ident['deployment'].items():assert 0<=x<=BOUNDS[m]
            assert POLICY.deployment_memory(ident['deployment'])<=49140
            for t,route in ident['routing'].items():
                assert math.isclose(sum(route.values()),ident['workload'][t],rel_tol=1e-9,abs_tol=1e-10)
                assert all((t,m) in admitted and ident['deployment'][m]>0 for m,v in route.items() if v>0)
            assert len(rt['raw_result_paths'])==3
            for path,seed in zip(rt['raw_result_paths'],plan['seeds']):
                proof.append(validate_raw(path,ident,seed,state['workload']));raw_paths.append(path)
    assert len(raw_paths)==len(set(raw_paths))==9
    physical=list((OUT/'crossover_raw').glob('**/rep_[123].json'))+list((OUT/'crossover_raw_D').glob('**/rep_[123].json'))
    assert {str(p) for p in physical}==set(raw_paths)
    allres=read(OUT/'all_crossover_results.json');assert allres['new_timed_runs']==9 and len(allres['new_states'])==3
    required=['deployment_distance.csv','reachability_trace.csv','reasoning_vqa_case_study.md','lp_runtime_alignment.csv','lp_runtime_alignment_summary.json','random_path_analysis.json','routing_crossover_audit.json','crossover_plan.json','crossover_results.json','greedy_strength_analysis.csv','resource_constraint_audit.csv','search_budget_audit.json','failure_classification.json','method_revision_recommendations.md','module8d3_summary.md']
    for name in required:assert (OUT/name).is_file() and (OUT/name).stat().st_size>0,name
    def rows(name):
        with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
    assert len(rows('deployment_distance.csv'))==12
    align=rows('lp_runtime_alignment.csv');a=read(OUT/'lp_runtime_alignment_summary.json')
    assert len(align)==a['evaluated_event_count']==69 and a['exact_unique_transition_count']==30 and a['all_events_retained']
    assert {r['workload'] for r in align}==set(NAMES)
    assert {r['method'] for r in align}=={'Proposed','Random'}
    for n in ['demand_balanced','vision_heavy','text_ocr_heavy']:assert a['deduplicated_sensitivity']['per_workload'][n]['status']=='INSUFFICIENT'
    assert len(rows('greedy_strength_analysis.csv'))==36 and len(rows('resource_constraint_audit.csv'))==24
    assert all(r['reachable']=='True' for r in rows('deployment_distance.csv'))
    budget=read(OUT/'search_budget_audit.json')['workloads'];assert len(budget)==6
    assert all(r['termination_reason']=='NO_RUNTIME_IMPROVEMENT' and r['actual_evaluated_states']<6 for r in budget)
    failures=read(OUT/'failure_classification.json');assert len(failures['workloads'])==6 and failures['DOMINANT_CAUSE']=='MULTIPLE'
    assert next(r for r in failures['workloads'] if r['workload']=='no_vqa_phase')['primary_failure_cause']=='NONE'
    reused=read(OUT/'routing_crossover_reuse_manifest.json')['states'];assert len(reused)==17
    assert all(s['all_12_conditions_equal'] and all(p['decision']=='REUSE_ALLOWED' for p in s['proofs']) for s in reused)
    assert environment_snapshot()==read(C/'environment_manifest.json')
    natives=read(D2/'source_preflight.json')['native_hashes']
    for i,(path,sha) in enumerate(natives.items(),1):
        assert file_hash(path)==sha,path
        if i%40==0:print(json.dumps({'event':'native_source_verified','files':i,'total':len(natives)}),flush=True)
    gpu=snapshot();oldgpu=read(C/'experiment_freeze.json')['gpu']
    assert all(gpu[k]==oldgpu[k] for k in ['uuid','driver','total_mib'])
    assert not gpu['processes'] and gpu['used_mib']<=128
    save_json(OUT/'crossover_measurement_validation.json',{'status':'PASS','new_raw_proofs':proof,'historical_exact_states':17,'new_fixed_states':3,'new_valid_timed_repetitions':9,'automatic_retries':0,'new_search':False})
    result={'status':'PASS','protected_8D2_artifacts':len(old),'all_frozen_inputs_sources_algorithms_unchanged':True,'native_source_files_rehashed':len(natives),'quality_admitted_paths':21,'all_new_states_steady':True,'new_fixed_states':3,'new_timed_runs':9,'alignment_events_retained':69,'unique_alignment_edges':30,'historical_exact_crossover_states':17,'required_outputs_present':required,'GPU_idle':gpu,'no_ablation_dynamic_second_hardware':True,'validator_source':ref(Path(__file__).resolve())}
    save_json(OUT/'final_validation.json',result)
    artifacts={str(p.relative_to(ROOT)):ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','artifact_manifest.json.sha256']}
    for p in [ROOT/'module8d3_diagnosis.py',ROOT/'module8d3_reporting.py',ROOT/'module8d3_stage2.py',Path(__file__).resolve()]:artifacts[str(p.relative_to(ROOT))]=ref(p)
    save_json(OUT/'artifact_manifest.json',{'module':'8D-3','status':'COMPLETE','artifacts':artifacts,'protected_inputs_unchanged':True,'method_revision_implemented':False,'stopped_after_module':'8D-3'})
    (OUT/'artifact_manifest.json.sha256').write_text(file_hash(OUT/'artifact_manifest.json')+'\n')
    print(json.dumps({'event':'final_validation_complete','status':'PASS','new_runs':9,'artifacts':len(artifacts)}),flush=True)

if __name__=='__main__':run()
