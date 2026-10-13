from pathlib import Path
from module8d_real_baselines import campaign as original
from module8d_real_baselines.evidence import EvidenceBank
from module8d_real_baselines.common import C,F,OUT,ROOT,read,ref,verify_inputs,save_json
from deployment_search.runtime_cache import file_hash


def reserved_seed_paths(workload):
    return {str(C/workload/'confirmation'/f'seed_rep{i}.json') for i in (1,2)}


class ConfirmationProtectedEvidenceBank(EvidenceBank):
    def confirmation(self,directory,owner,workload,identity,role,index,banned):
        # Exact 8C seed donors are reserved for the two logical seed labels, even
        # when the second seed label comes after both final labels. This affects
        # evidence selection only; no measurement, routing or objective changes.
        protected=reserved_seed_paths(workload) if role=='final' else set()
        rep=super().confirmation(directory,owner,workload,identity,role,index,set(banned)|protected)
        if role=='final':assert rep['raw_result_path'] not in reserved_seed_paths(workload)
        return rep


def prepare_bridge():
    verify_inputs()
    refs={str(p.relative_to(ROOT)):ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    doc={'reason':'Reserve both original seed confirmation raw paths before interleaved final1/final2 can select evidence',
        'source_requirement':ref(F/'runtime_reuse_execution_contract.json'),
        'original_campaign':ref(OUT/'campaign_manifest.json'),'original_campaign_implementation_unchanged':True,
        'compatibility_sources':refs,'baseline_definitions_changed':False,'baseline_freeze_changed':False,
        'analysis_policy_changed':False,'search_or_candidate_prioritization_changed':False,
        'measurement_harness_or_protocol_changed':False,'confirmations_still_seed_2_final_2':True,
        'Phase_A_outcomes_used_for_rule':False,'fix_basis':'Static inspection of frozen confirmation ordering and raw-donor disjointness contract',
        'timed_runs_interrupted_repeated_or_discarded':0,
        'guard_before_first_B4_job':not any(OUT.glob('b4_seed_*/ */search_complete.json'.replace(' ','')))}
    target=OUT/'confirmation_donor_compatibility_freeze.json'
    if target.exists():assert read(target)==doc
    else:save_json(target,doc)
    for row in refs.values():assert file_hash(row['path'])==row['sha256']
    bank=ConfirmationProtectedEvidenceBank()
    # Pure CPU identity test: final2 of an unchanged deployment selects final-role
    # original raw, while both original seed donors remain available for seed1/2.
    name='demand_balanced';identity=read(C/name/'search_cache/runtime_cache_manifest.json')
    trace=read(C/name/'search_complete.json');identity=identity['configurations'][trace['seed_runtime']['cache_key']]['identity']
    from joint_deploy.measurement.schema import MeasurementConfig
    settings=read(F/'baseline_definitions.json')['measurement_protocol']['settings']
    config=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],random_seed=8202,**settings).as_dict()
    candidates=[r for r in bank.confirmations if r['config']==config and r['path'] not in reserved_seed_paths(name)]
    candidates.sort(key=lambda r:(r['origin_campaign']!='8C',r['role']!='seed',r['path']))
    assert candidates[0]['path']==str(C/name/'confirmation/final_rep2.json')
    save_json(OUT/'confirmation_donor_CPU_validation.json',{'status':'PASS','GPU_runs':0,
        'seed_1_and_2_reserved_before_final_selection':True,'unchanged_final2_donor':candidates[0]['path'],
        'same_original_search_and_runtime_evaluator':True,'original_frozen_sources_hash_verified':True})


def main():
    prepare_bridge()
    original.EvidenceBank=ConfirmationProtectedEvidenceBank
    return original.main()


if __name__=='__main__':main()
