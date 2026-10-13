import copy,json
from pathlib import Path
from deployment_core.models import ModelProfile,Hardware
from deployment_core.replica_calibration import load_replica_calibration,ReplicaCalibration
from deployment_search.runtime_cache import file_hash,save_json,digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module8c_final_real_search'
B=ROOT/'results/module8b_quality_pool_finalization'
P=ROOT/'results/candidate_pool_v2_1'
def read(p):return json.loads(Path(p).read_text())
def ref(p):return {'path':str(Path(p).resolve()),'sha256':file_hash(p)}
def inputs():
    pool=read(P/'candidate_pool_v2_1.json');service=read(P/'service_demand_v2_1.json')
    admission=read(B/'quality_admission_matrix.json')
    allowed={(r['task'],r['model']) for r in admission['rows'] if r['quality_admissible']}
    assert len(allowed)==21 and service['completeness']['available_paths']==service['completeness']['required_paths']==27
    raw={m:ModelProfile(m,frozenset(v['capabilities_verified']),v['measured_idle_vram_mib'],v['service_time_ms'],
         {'completed_catalog':ref(P/'service_demand_v2_1.json')}) for m,v in service['models'].items()}
    catalog={}
    for m,v in raw.items():
        tasks=frozenset(k for k in v.supported_tasks if (k,m) in allowed)
        if tasks:catalog[m]=ModelProfile(m,tasks,v.memory_mib,{k:v.service_time_ms[k] for k in tasks},v.sources)
    base=load_replica_calibration(raw)
    cal=ReplicaCalibration({m:base.capacity_factors[m] for m in catalog},
          {m:base.memory_table_mib[m] for m in catalog},
          {m:base.evidence_bounds[m] for m in catalog},
          {m:base.model_evidence[m] for m in catalog},base.source_hashes).validate(catalog)
    roles={m['id']:m['role'] for m in pool['models'] if m['id'] in pool['formal_model_ids']}
    assert set(raw)==set(roles) and set(raw)-set(catalog)=={'paddleocr_vl'}
    expected={m:2 if m in ('qwen','whisper-small','resnet50') else 1 for m in roles}
    assert base.evidence_bounds==expected
    return catalog,Hardware(49140,'Frozen RTX 4090'),{m:roles[m] for m in catalog},cal

def verify_freeze():
    freeze=read(OUT/'experiment_freeze.json')
    for group in ('protected_files','execution_files','implementation_files'):
        for r in freeze[group].values():
            if file_hash(r['path'])!=r['sha256']:raise RuntimeError('Frozen input/code changed: '+r['path'])
    return freeze
