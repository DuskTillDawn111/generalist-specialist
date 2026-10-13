"""Freeze development status and v1 before implementing revision; CPU only."""
from pathlib import Path
import shutil
from module8d3_diagnosis import ROOT, C, F, D2, B, P, NAMES, read, ref, file_hash, save_json, verify_inputs
OUT=ROOT/'results/module8e1_method_revision'

def freeze():
    verify_inputs();OUT.mkdir(parents=True,exist_ok=True)
    d3=ROOT/'results/module8d3_failure_diagnosis/artifact_manifest.json'
    assert file_hash(d3)==Path(str(d3)+'.sha256').read_text().strip()
    for r in read(d3)['artifacts'].values():assert file_hash(r['path'])==r['sha256']
    source_dirs=('deployment_core','deployment_search','integer_replica_pilot','module8c_final_real_search','replica_capacity_calibration')
    sources={str(p.relative_to(ROOT)):ref(p) for folder in source_dirs for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts}
    archive={}
    for name,r in sources.items():
        target=OUT/'method_v1_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert file_hash(target)==r['sha256']
        else:shutil.copyfile(r['path'],target)
        archive[name]=ref(target)
    record={'method':'method_v1','status':'FROZEN','entrypoint':'method_v1.create() delegates to unchanged deployment_search.IntegerGuidedDeploymentSearch','sources':sources,'archived_sources':archive,'config':read(C/NAMES[0]/'protocol.json')['search_config'],'candidate_generation':'Original generate_replica_neighborhood / binary generate_neighborhood','ranking':'Original rank_candidates: exact delta_z then deterministic model order, predictor=None','routing':'Original canonical min-max pressure LP','runtime_objective':read(C/NAMES[0]/'protocol.json')['objective'],'search_budget':{'top_b':2,'max_iterations':3,'max_unique_runtime_states':6,'patience':1,'search_repetitions':3,'seed_confirmation_repetitions':2,'final_confirmation_repetitions':2},'trace_sources':{str(p.relative_to(ROOT)):ref(p) for parent in (C,D2) for p in sorted(parent.glob('**/search_complete.json'))},'protected_inputs':{'8C':ref(C/'experiment_freeze.json'),'8D1':ref(F/'baseline_freeze_hashes.json'),'8D2':ref(D2/'artifact_manifest.json'),'8D3':ref(d3)},'v1_original_files_overwritten':False,'wrapper':ref(ROOT/'method_v1/__init__.py')}
    path=OUT/'method_v1_freeze.json'
    if path.exists():assert read(path)==record
    else:save_json(path,record);Path(str(path)+'.sha256').write_text(file_hash(path)+'\n')
    policy={'workload_status':'DEVELOPMENT_WORKLOADS','workloads':NAMES,'source':ref(B/'final_benchmark_gate.json'),'lambda_vectors':{n:read(C/n/'protocol.json')['workload']['arrival_rates_rps'] for n in NAMES},'unbiased_final_validation_claim_allowed':False,'reason':'Method revision selected using Module 8C/8D/8D3 evidence','no_new_GPU_runs':True,'heldout_required_before_effectiveness_claim':True}
    path=OUT/'development_workload_designation.json'
    if path.exists():assert read(path)==policy
    else:save_json(path,policy);Path(str(path)+'.sha256').write_text(file_hash(path)+'\n')
    print({'v1_frozen':True,'source_files':len(sources),'trace_files':len(record['trace_sources']),'GPU_runs':0},flush=True)
if __name__=='__main__':freeze()
