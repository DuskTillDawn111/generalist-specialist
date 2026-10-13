"""Freeze measured inputs, unmodified W2 rates, adapter code and strict budgets."""
import json
from pathlib import Path
from candidate_pool_service_demand.dry_run import load_completed_catalog
from candidate_pool_audit.provenance import environment_snapshot
from candidate_pool_audit.registry import registry
from deployment_core.replica_calibration import load_replica_calibration, ReplicaPlanningPolicy
from deployment_core.workload import Workload
from deployment_core.candidate_pool_v2 import mandatory_presence
from deployment_search.runtime_cache import file_hash, digest, save_json
from deployment_search.real_pilot import SETTINGS, SEEDS
from deployment_search import SearchConfig
from replica_scaling.telemetry import snapshot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/integer_replica_search_real_pilot'
CONFIG=SearchConfig(top_b=2,max_iterations=3,runtime_evaluation_budget=6,no_improvement_patience=1,rho_max=.8,mean_tolerance_ms=1.)
CONFIRM_SEEDS=(8101,8202)


def ref(path):return {'path':str(Path(path).resolve()),'sha256':file_hash(path)}
def read(path):return json.loads(Path(path).read_text())


def protected():
    files=set()
    for directory in ('candidate_pool_v2','candidate_pool_v2_service_demand','replica_scaling_real','replica_capacity_calibration','integer_replica_search_foundation','deployment_search_multiworkload'):
        files.update(p for p in (ROOT/'results'/directory).rglob('*') if p.is_file() and p.name!='.runtime_cache.lock')
    for directory in ('deployment_core','deployment_search','joint_deploy','candidate_pool_audit','replica_scaling','performance_model'):
        files.update((ROOT/directory).rglob('*.py'))
    files.update([ROOT/'benchmark/task_pools/mixed_formal_1200_6task.jsonl',ROOT/'benchmark/prompts/task_prompts.json'])
    return sorted(files)


def verify_freeze():
    plan=read(OUT/'pilot_freeze.json')
    for refdata in plan['frozen_files'].values():
        if file_hash(refdata['path'])!=refdata['sha256']:raise RuntimeError('Frozen pilot file changed: '+refdata['path'])
    for refdata in plan['implementation_hashes'].values():
        if file_hash(refdata['path'])!=refdata['sha256']:raise RuntimeError('Pilot code changed after freeze: '+refdata['path'])
    return plan


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pilot_freeze.json').exists():return verify_freeze()
    catalog,hardware,roles,records,data=load_completed_catalog();cal=load_replica_calibration(catalog)
    env=environment_snapshot();frozen_env=read(ROOT/'results/candidate_pool_v2/environment_manifest.json')['environments']
    if env!=frozen_env:raise RuntimeError('Frozen candidate environment changed')
    save_json(OUT/'environment_manifest.json',{'environments':env,'matches_frozen_v2':True})
    expected=read(ROOT/'results/candidate_pool_v2/source_hashes.json')['files']
    # Recompute all frozen weight/config/tokenizer/native-adapter hashes once before any launch.
    native={k:v for k,v in expected.items() if k.startswith('models/') or k.startswith('joint_deploy/') or
        k.startswith('candidate_pool_audit/worker_') or k=='candidate_pool_audit/new_worker.py'}
    for i,(name,value) in enumerate(native.items(),1):
        if file_hash(ROOT/name)!=value['sha256']:raise RuntimeError('Frozen native model/adapter changed: '+name)
        if i%50==0:print(json.dumps({'event':'source_hash_check','checked':i,'total':len(native)}),flush=True)
    models={m['id']:m for m in read(ROOT/'results/candidate_pool_v2/candidate_models.json')['models'] if m['id'] in catalog}
    launchers={m:{k:str(v) for k,v in row.items()} for m,row in registry().items() if m in catalog}
    source_models={}
    for m,row in launchers.items():
        # Model manifests use the frozen v2 model source and all authenticated weight files;
        # directory mapping from original inventory avoids guessing source filenames.
        from candidate_pool_audit.inventory import SOURCES
        directory=SOURCES[m][1]
        model_files={name:value for name,value in native.items() if directory and name.startswith('models/'+directory+'/')}
        source_models[m]={'source':models[m]['source'],'source_revision':models[m]['source_revision'],
            'model_manifest_sha256':digest(model_files),'model_files':model_files,
            'adapter':ref(row['script']),'environment_hash':env[row['python']]['package_manifest_sha256'],
            'executable_hash':env[row['python']]['executable_sha256'],'precision':models[m]['precision']}
    save_json(OUT/'model_source_hashes.json',source_models)
    bounds,provenance=ReplicaPlanningPolicy(catalog,'empirical',calibration=cal).bounds(hardware)
    gpu=snapshot()
    if gpu['processes'] or gpu['used_mib']>128 or gpu['total_mib']!=hardware.gpu_memory_mib:raise RuntimeError('GPU identity/isolation/capacity mismatch')
    workload_path=ROOT/'results/deployment_search_multiworkload/benchmark_workloads.json'
    if file_hash(workload_path)!=read(ROOT/'results/deployment_search_multiworkload/freeze_manifest.json')['files']['benchmark_workloads.json']:
        raise RuntimeError('Module 3C frozen workload hash mismatch')
    workload=next(w for w in read(workload_path)['workloads'] if w['name']=='W2_generalist_heavy')
    calibration_dir=ROOT/'results/replica_capacity_calibration'
    calibration_manifest=read(calibration_dir/'artifact_manifest.json')
    for name in ('replica_capacity_factors.json','replica_memory_table.json','evidence_replica_bounds.json'):
        if file_hash(calibration_dir/name)!=calibration_manifest['artifacts'][name]['sha256']:raise RuntimeError('Frozen 7C-2 table changed')
    if {m:{str(n):v for n,v in table.items()} for m,table in cal.memory_table_mib.items()}!={m:row['M'] for m,row in read(calibration_dir/'replica_memory_table.json')['models'].items()}:raise RuntimeError('Memory policy differs from frozen 7C-2 table')
    inputs={name:ref(ROOT/path) for name,path in {
        'candidate_models':'results/candidate_pool_v2/candidate_models.json',
        'capability':'results/candidate_pool_v2/compatibility_matrix.json',
        'idle_memory':'results/candidate_pool_v2/measured_vram.json',
        'Stage1':'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json',
        'capacity_factors':'results/replica_capacity_calibration/replica_capacity_factors.json',
        'replica_memory_table':'results/replica_capacity_calibration/replica_memory_table.json',
        'evidence_bounds':'results/replica_capacity_calibration/evidence_replica_bounds.json',
        'workload':'results/deployment_search_multiworkload/benchmark_workloads.json',
        'pool':'benchmark/task_pools/mixed_formal_1200_6task.jsonl',
        'prompts':'benchmark/prompts/task_prompts.json'}.items()}
    context={'models':{m:{'supported_tasks':sorted(p.supported_tasks),'service_time_ms':dict(p.service_time_ms),'role':roles[m],'source':source_models[m]} for m,p in catalog.items()},
        'registry':launchers,'gpu_uuid':gpu['uuid'],'gpu_capacity_mib':hardware.gpu_memory_mib,
        'replica_bounds':bounds,'bound_provenance':provenance,
        'memory_table_mib':{m:{str(n):v for n,v in table.items()} for m,table in cal.memory_table_mib.items()},
        'input_pool':inputs['pool']['path'],'telemetry_interval_seconds':.5,'source_hashes':inputs}
    context['context_sha256']=digest(context)
    save_json(OUT/'runtime_context.json',context)
    (OUT/'runtime_context.json.sha256').write_text(file_hash(OUT/'runtime_context.json')+'\n')
    save_json(OUT/'before_hashes.json',{'frozen_files':{str(p.relative_to(ROOT)):ref(p) for p in protected()},'native_models_and_adapters':native})
    settings=dict(SETTINGS,routing_source='module7C3_empirical_canonical_family_then_replica_round_robin')
    smoke=[{'model_id':m,'replica_count':2,'task':task,'offered_load_rps':1.,'measurement_seconds':2.,'warmup_seconds':0.,'random_seed':7101} for m,task in [('qwen','asr'),('whisper-small','asr'),('resnet50','image')]]
    save_json(OUT/'pilot_protocol.json',{'workload':workload,'search_config':CONFIG,'capacity_mode':'empirical','predictor':None,
        'measurement_settings':settings,'search_seeds':SEEDS,'max_valid_unique_runtime_evaluations':6,
        'max_reserved_unique_states':6,'repetitions_per_state':3,'search_physical_attempt_cap':18,
        'confirmation_repetitions_per_state':2,'confirmation_seeds':CONFIRM_SEEDS,'confirmation_physical_attempt_cap':4,
        'confirmation_order':[['seed',1],['final',1],['final',2],['seed',2]],'confirmation_influences_search':False,
        'adapter_smoke_plan':smoke,'smoke_physical_attempt_cap':3,'total_physical_attempt_cap':25,
        'objective':'Unchanged Module 3 NoiseAwareRuntimeObjective(1ms, 5%); stability first; same-steady mean then weighted task P95 then memory; same-nonsteady retains incumbent.',
        'DQA':'Same completed-timed measurement / stability_v2 semantics; completed unstable remains valid. External/uncertain/model/harness interruptions retained without serving label; no auto retry.',
        'memory_idle_diagnostic_tolerance_relative':.05,
        'failure_policy':'Prelaunch calibrated memory/evidence gate; postlaunch resource infeasible preserved and pilot stops without silent replacement or budget growth.',
        'mandatory_presence':mandatory_presence(records,Workload(workload['arrival_rates_rps'])),
        'interpretation':'ASR-only Qwen/Whisper scaling transferred to aggregate family workload is a planning surrogate; mixed-task/heterogeneous interference tested by this pilot, not assumed.'})
    files=('pilot_protocol.json','runtime_context.json','runtime_context.json.sha256','environment_manifest.json','model_source_hashes.json','before_hashes.json')
    implementation={str(p.relative_to(ROOT)):ref(p) for d in ('integer_replica_pilot','deployment_core','deployment_search','joint_deploy/measurement') for p in sorted((ROOT/d).glob('*.py'))}
    plan={'module':'7C-3','frozen_before_first_GPU_run':True,'gpu':gpu,'frozen_files':{name:ref(OUT/name) for name in files},
        'implementation_hashes':implementation,'input_hashes':inputs,'max_iterations':3,'runtime_unique_budget':6,'repetitions':3,
        'smoke_attempts':3,'confirmation_attempts':4,'capacity_mode':'empirical','replica_bounds':bounds}
    save_json(OUT/'pilot_freeze.json',plan)
    print(json.dumps({'event':'pilot_frozen','sha256':file_hash(OUT/'pilot_freeze.json'),'GPU_runs':0,'W2':workload['arrival_rates_rps']}),flush=True)
    return plan


if __name__=='__main__':prepare()
