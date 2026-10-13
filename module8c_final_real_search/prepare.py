import copy,os
from dataclasses import asdict
from .common import *
from candidate_pool_audit.provenance import environment_snapshot
from module8b_quality_pool_finalization.common import registry
from deployment_core import Workload
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_search import IntegerGuidedDeploymentSearch,SearchConfig
from replica_scaling.telemetry import snapshot

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'experiment_freeze.json').exists():return verify_freeze()
    gate=read(B/'final_benchmark_gate.json');assert gate['FINAL_BENCHMARK_MATRIX_READY']
    for path,r in read(B/'artifact_manifest.json')['artifacts'].items():
        assert file_hash(path)==r['sha256'],path
    matrix=read(B/'final_benchmark_workload_matrix.json')
    assert matrix['frozen'] and not matrix['lambda_vectors_modified'] and len(matrix['proposals'])==6
    assert file_hash(B/'final_benchmark_workload_matrix.json')==(B/'final_benchmark_workload_matrix.sha256').read_text().strip()
    catalog,hardware,roles,cal=inputs()
    env=environment_snapshot()
    assert env==read(ROOT/'results/candidate_pool_v2/environment_manifest.json')['environments']
    save_json(OUT/'environment_manifest.json',env)
    native=read(ROOT/'results/integer_replica_search_real_pilot/before_hashes.json')['native_models_and_adapters']
    for i,(path,r) in enumerate(native.items(),1):
        assert file_hash(ROOT/path)==r['sha256'],path
        if i%50==0:print(json.dumps({'event':'native_source_verified','checked':i,'total':len(native)}),flush=True)
    bounds,provenance=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal).bounds(hardware)
    gpu=snapshot()
    assert not gpu['processes'] and gpu['used_mib']<=128 and gpu['total_mib']==49140
    oldprotocol=read(ROOT/'results/integer_replica_search_real_pilot/pilot_protocol.json')
    config=SearchConfig(**oldprotocol['search_config'])
    assert config.top_b==2 and config.runtime_evaluation_budget==6 and config.max_iterations==3
    sources=copy.deepcopy(read(ROOT/'results/integer_replica_search_real_pilot/model_source_hashes.json'))
    launchers={m:{k:str(v) for k,v in row.items()} for m,row in registry().items() if m in catalog}
    for m in ('phi4mm','minicpm'):
        sources[m]['adapter']=ref(launchers[m]['script'])
        sources[m]['vqa_extension']=ref(ROOT/'candidate_pool_v2_1_adapters/vqa_extension.py')
    inputrefs={k:ref(p) for k,p in {
        'candidate_pool':P/'candidate_pool_v2_1.json','quality_admission':B/'quality_admission_matrix.json',
        'quality_policy':B/'quality_policy.json','service_demand':P/'service_demand_v2_1.json',
        'workloads':B/'final_benchmark_workload_matrix.json',
        'capacity':ROOT/'results/replica_capacity_calibration/replica_capacity_factors.json',
        'replica_memory':ROOT/'results/replica_capacity_calibration/replica_memory_table.json',
        'bounds':ROOT/'results/replica_capacity_calibration/evidence_replica_bounds.json',
        'pool':ROOT/'benchmark/task_pools/mixed_formal_1200_6task.jsonl',
        'prompts':ROOT/'benchmark/prompts/task_prompts.json'}.items()}
    failed=ROOT/'results/module8c_final_real_search_startup_failure_001'
    recovery={'failure_directory':str(failed),'failed_premeasurement_physical_attempts':1,
       'valid_runtime_states':0,'measured_requests':0,
       'reason':'Replica alias paddleocr#0 bypasses the literal-name historical Worker.start OCR LD_LIBRARY_PATH branch',
       'fix':'Reuse existing Module 8A CUDA library path in inherited execution environment; no package/weight/handler/search modification',
       'runtime_environment_overrides':{'LD_LIBRARY_PATH':os.environ['LD_LIBRARY_PATH']},
       'original_failed_freeze':ref(failed/'experiment_freeze.json'),
       'original_failed_log':ref(failed/'campaign.log'),
       'new_valid_search_budget_unchanged':True,'further_automatic_retries':False}
    save_json(OUT/'startup_recovery.json',recovery)
    inputrefs['execution_environment']=ref(OUT/'startup_recovery.json')
    context={'models':{m:{'supported_tasks':sorted(v.supported_tasks),'service_time_ms':dict(v.service_time_ms),
                    'role':roles[m],'source':sources[m]} for m,v in catalog.items()},
        'registry':launchers,'gpu_uuid':gpu['uuid'],'gpu_capacity_mib':49140,'replica_bounds':bounds,
        'bound_provenance':provenance,'memory_table_mib':{m:{str(n):v for n,v in table.items()} for m,table in cal.memory_table_mib.items()},
        'input_pool':inputrefs['pool']['path'],'telemetry_interval_seconds':.5,'source_hashes':inputrefs,
        'fixed_zero_models':['paddleocr_vl'],'formal_model_ids':read(P/'candidate_pool_v2_1.json')['formal_model_ids']}
    context['context_sha256']=digest(context)
    save_json(OUT/'model_source_hashes.json',sources)
    execution={}
    for proposal in matrix['proposals']:
        name=proposal['name'];directory=OUT/name;directory.mkdir(exist_ok=True)
        protocol={'module':'8C','workload':{'name':name,'arrival_rates_rps':proposal['lambda_rps']},
          'search_config':asdict(config),'measurement_settings':oldprotocol['measurement_settings'],
          'search_seeds':oldprotocol['search_seeds'],'confirmation_seeds':oldprotocol['confirmation_seeds'],
          'confirmation_order':oldprotocol['confirmation_order'],'confirmation_influences_search':False,
          'predictor':None,'capacity_mode':'empirical','memory_mode':'empirical',
          'max_valid_unique_runtime_states':6,'search_repetitions_per_state':3,'search_physical_attempt_cap':18,
          'confirmation_repetitions_per_state':2,'confirmation_physical_attempt_cap':4,
          'objective':oldprotocol['objective'],'DQA':oldprotocol['DQA'],'failure_policy':oldprotocol['failure_policy'],
          'quality_admissible_path_count':21,'fixed_zero_model':'paddleocr_vl','lambda_modified':False}
        save_json(directory/'protocol.json',protocol);save_json(directory/'runtime_context.json',context)
        (directory/'runtime_context.json.sha256').write_text(file_hash(directory/'runtime_context.json')+'\n')
        plan=IntegerGuidedDeploymentSearch().search(Workload(proposal['lambda_rps']),catalog,hardware,
              SearchConfig(**dict(asdict(config),dry_run=True)),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        assert plan.seed.feasible
        expected=next(w for w in read(B/'planning_details.json')['workloads'] if w['workload']==name)
        assert plan.seed.deployment==expected['seed']['deployment']
        save_json(directory/'planning_preflight.json',plan.to_dict())
        for file in ('protocol.json','runtime_context.json','runtime_context.json.sha256','planning_preflight.json'):
            execution[name+'/'+file]=ref(directory/file)
    protected={}
    directories=('results/module8b_quality_pool_finalization','results/candidate_pool_v2_1','results/candidate_pool_v2',
        'results/candidate_pool_v2_service_demand','results/quality_workload_audit','results/replica_capacity_calibration',
        'results/integer_replica_search_multiworkload','results/integer_replica_search_real_pilot',
        'deployment_core','deployment_search','integer_replica_pilot','integer_replica_multiworkload',
        'joint_deploy','candidate_pool_audit','candidate_pool_service_demand','replica_scaling',
        'benchmark/scripts','benchmark/task_pools','benchmark/prompts','benchmark/manifests','benchmark/profiles',
        'candidate_pool_v2_1_adapters','module8b_quality_pool_finalization',
        'results/module8c_final_real_search_startup_failure_001')
    for folder in directories:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('.campaign.lock','.runtime_cache.lock','.pilot.lock'):
                protected[str(p.relative_to(ROOT))]=ref(p)
    implementation={str(p.relative_to(ROOT)):ref(p) for p in (ROOT/'module8c_final_real_search').glob('*.py')}
    execution['startup_recovery.json']=ref(OUT/'startup_recovery.json')
    freeze={'module':'8C','frozen_before_first_gpu_run':True,'execution_order':[p['name'] for p in matrix['proposals']],
       'inputs':inputrefs,'search_config':asdict(config),'gpu':gpu,'protected_files':protected,
       'execution_files':execution,'implementation_files':implementation,'max_total_physical_attempts':132,
       'replica_bounds_full':{m:2 if m in ('qwen','whisper-small','resnet50') else 1 for m in read(P/'candidate_pool_v2_1.json')['formal_model_ids']},
       'quality_admissible_path_count':21,'source_model_weight_manifest_verified':True,
       'search_algorithms_changed':False,'predictor':None,'enumeration_allowed':False,
       'baseline':False,'ablation':False,'dynamic':False,'second_hardware':False,
       'no_new_smoke_or_standalone_measurements':True,'failed_premeasurement_attempts_preserved':1,
       'adapter_validation_reused':ref(ROOT/'results/integer_replica_search_real_pilot/replica_adapter_validation.json')}
    save_json(OUT/'experiment_freeze.json',freeze)
    print(json.dumps({'event':'module8c_frozen','workloads':freeze['execution_order'],'max_physical_attempts':132}),flush=True)
    return verify_freeze()
