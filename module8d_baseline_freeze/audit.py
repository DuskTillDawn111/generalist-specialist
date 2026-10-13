"""CPU-only exact identity audit. This module never constructs a RuntimeEvaluator."""
import copy
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from candidate_pool_audit.provenance import environment_snapshot
from deployment_search.runtime_cache import digest, file_hash
from .definitions import ROOT, B, C, OUT, TASKS, read, ref, freeze_file, csvwrite, verify_freeze

CONDITIONS = (
    'exact_workload_lambda', 'exact_replica_vector', 'exact_routing_semantics_and_fractions',
    'same_model_revisions', 'same_environments', 'same_quality_admission_path_set',
    'same_arrival_generation_protocol', 'same_warmup', 'same_measurement_window',
    'same_runtime_metric_definition', 'same_task_input_source_subset', 'same_precision_runtime_backend',
)


def verify_sources():
    frozen = verify_freeze()
    manifest = read(C/'artifact_manifest.json')
    for record in manifest['artifacts'].values():
        assert file_hash(record['path']) == record['sha256'], record['path']
    print('Module 8C artifacts and protected source files unchanged', flush=True)
    expected = read(ROOT/'results/integer_replica_search_real_pilot/before_hashes.json')['native_models_and_adapters']
    native = {}
    for i, (name, record) in enumerate(expected.items(), 1):
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        actual = file_hash(path)
        assert actual == record['sha256'], name
        native[name] = {'path':str(path), 'expected_sha256':record['sha256'], 'actual_sha256':actual}
        if i % 40 == 0:
            print(f'Native files verified {i}/{len(expected)}', flush=True)
    environments = environment_snapshot()
    assert environments == read(C/'environment_manifest.json'), 'Runtime environment drift'
    # Read-only driver/hardware identity query. No model or CUDA worker is started.
    gpu_values = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total',
                                         '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    gpu = dict(zip(('uuid','name','driver','total_mib'),gpu_values))
    old_gpu = frozen['gpu']
    assert gpu['uuid']==old_gpu['uuid'] and gpu['driver']==old_gpu['driver']
    assert float(gpu['total_mib'])==old_gpu['total_mib']
    for record in read(C/'zero_rate_transport_recovery.json')['compatibility_sources'].values():
        assert file_hash(record['path'])==record['sha256']
    sources = read(C/'model_source_hashes.json')
    # Verify the actual files named in every model manifest, including extensions.
    all_model_files = {}
    for model, source in sources.items():
        for name, record in source['model_files'].items():
            if name not in all_model_files:
                actual = native.get(name,{}).get('actual_sha256') or file_hash(ROOT/name)
                assert actual==record['sha256'], name
                all_model_files[name]={'path':str(ROOT/name),'sha256':actual}
        for key in ('adapter','vqa_extension'):
            if key in source:
                record=source[key]
                assert file_hash(record['path'])==record['sha256']
    result={'status':'PASS','new_GPU_experiments':0,'Module8C_artifact_count':len(manifest['artifacts']),
        'protected_file_counts':{key:len(frozen[key]) for key in ('protected_files','execution_files','implementation_files')},
        'native_files':native,'model_manifest_files':all_model_files,'current_environments':environments,
        'read_only_gpu_identity':gpu,'same_native_weights_adapters_precision_backends':True,
        'same_runtime_environment':True,'compatibility_sources_unchanged':True,
        'sources':{'8C_artifacts':ref(C/'artifact_manifest.json'),'8C_freeze':ref(C/'experiment_freeze.json'),
                   'model_manifest':ref(C/'model_source_hashes.json'),'environment':ref(C/'environment_manifest.json')}}
    freeze_file('source_identity_validation.json',result)
    return sources, frozen, manifest


def scan_raws(manifest):
    """Include legacy real runs; exclude configs, aggregates and standalone samples explicitly."""
    records=[];counts=Counter(); scanned=0
    expected_hash={r['path']:r['sha256'] for r in manifest['artifacts'].values()}
    for directory in (ROOT/'results', ROOT/'benchmark/results'):
        for path in sorted(directory.rglob('*.json')):
            if OUT in path.parents:continue
            scanned+=1
            with path.open() as stream:prefix=stream.read(8192)
            if not all(f'"{key}"' in prefix for key in ('workload_rps','routing_rps','measurement_seconds')):
                counts['non_timed_multitask_artifact']+=1;continue
            raw=read(path)
            if not isinstance(raw,dict) or not all(k in raw for k in ('workload_rps','routing_rps','request_records','measurement_window')):
                counts['configuration_or_aggregate_not_raw_measurement']+=1;continue
            family=str(path.relative_to(directory).parts[0])
            counts[family]+=1
            actual=file_hash(path)
            protected_hash=expected_hash.get(str(path))
            if protected_hash is not None:assert protected_hash==actual
            process_path=path.with_suffix('.process.json')
            process=read(process_path) if process_path.exists() else None
            config_path=path.with_suffix('.config.json')
            config=read(config_path) if config_path.exists() else None
            summary_path=path.with_suffix('.summary.json')
            summary=read(summary_path) if summary_path.exists() else None
            # Search repetitions in 8C have summary evidence in the cache manifest.
            context_path=None
            if process and '--context' in process.get('command',[]):
                command=process['command']; context_path=Path(command[command.index('--context')+1])
            current_8c = C in path.parents and len(path.relative_to(C).parts)>1 and path.relative_to(C).parts[0] in {
                p['name'] for p in read(B/'final_benchmark_workload_matrix.json')['proposals']}
            phase='confirmation' if 'confirmation' in path.parts else 'search'
            role=('seed' if path.name.startswith('seed_rep') else 'final' if path.name.startswith('final_rep') else 'search')
            valid=bool(raw.get('valid_execution') and raw.get('measurement_window'))
            data={k:raw.get(k) for k in ('deployment','workload_rps','routing_rps','warmup_seconds','measurement_seconds',
                'arrival_process','random_seed','routing_source','startup_timeout_seconds','drain_timeout_seconds',
                'shutdown_timeout_seconds','max_pending_per_model','source','runtime_context_sha256','baseline_gpu',
                'arrival_process_details','routing_policy','clock','execution_status','measurement_window',
                'worker_identities','zero_rate_transport_compatibility')}
            records.append({'original_run_path':str(path),'raw_sha256':actual,'original_family':family,
                'Module8C_current_campaign':current_8c,'phase':phase,'confirmation_role':role,
                'valid_timed_execution':valid,'raw':data,'process':process,'config':config,'summary':summary,
                'context_path':str(context_path) if context_path else None,
                'config_reference':ref(config_path) if config_path.exists() else None,
                'process_reference':ref(process_path) if process_path.exists() else None})
    print(f'Scanned {scanned} JSON artifacts; found {len(records)} multitask raw measurements',flush=True)
    return records,{'directories':['results','benchmark/results'],'JSON_files_scanned':scanned,
        'classification_counts':dict(counts),'raw_measurements':len(records),
        'standalone_quality_or_service_demand_records':'Not deployment measurements; excluded',
        'startup_or_pretransport_failures':'Not valid timed measurements; cannot serve as reused states'}


def make_checker(definitions, sources, frozen):
    contexts={name:read(record['path']) for name,record in definitions['measurement_protocol']['runtime_context_by_workload'].items()}
    settings=definitions['measurement_protocol']['settings']
    routing_policy='Frozen family smooth weighted round-robin, followed by deterministic round-robin across that family replicas; counters reset each phase.'
    context_valid={}
    for name,ctx in contexts.items():
        content=dict(ctx);content.pop('context_sha256')
        assert digest(content)==ctx['context_sha256']
        for r in ctx['source_hashes'].values():assert file_hash(r['path'])==r['sha256']
        for model,row in ctx['models'].items():assert row['source']==sources[model]
        context_valid[name]=True
    def check(state,record):
        raw=record['raw'];ctx=contexts[state['workload']]
        # Formal PadOCR-VL x=0 is restored explicitly; no unknown positive model is dropped.
        observed_x={m:raw['deployment'].get(m,0) for m in state['deployment']}
        known_models=set(state['deployment'])
        vector_equal=observed_x==state['deployment'] and all(v==0 for m,v in raw['deployment'].items() if m not in known_models)
        observed_routes={t:{m:raw['routing_rps'].get(t,{}).get(m,0.) for m in state['deployment']} for t in TASKS}
        unknown_route=any(value!=0 for task,row in raw['routing_rps'].items() for model,value in row.items()
                          if task not in TASKS or model not in known_models)
        src=raw.get('source') or {};worker_sources={v.get('family',key.split('#')[0]):v.get('model_source')
                              for key,v in (raw.get('worker_identities') or {}).items()}
        deployed={m for m,n in raw['deployment'].items() if n}
        revisions=bool(deployed) and set(worker_sources)==deployed and all(worker_sources[m]==sources.get(m) for m in deployed)
        proc=record.get('process') or {};command=proc.get('command',[])
        harness=definitions['measurement_protocol']['harness_by_workload'][state['workload']]
        harness_same=len(command)>2 and command[1:3]==['-m',harness]
        context_same=raw.get('runtime_context_sha256')==ctx['context_sha256']
        context_path_same=record.get('context_path')==definitions['measurement_protocol']['runtime_context_by_workload'][state['workload']]['path']
        gpu=raw.get('baseline_gpu') or {}
        gpu_same=all(gpu.get(k)==frozen['gpu'][k] for k in ('uuid','driver','name','total_mib'))
        execution_env=src.get('execution_environment')==ctx['source_hashes']['execution_environment']
        quality=all(src.get(k)==ctx['source_hashes'][k] for k in ('quality_admission','quality_policy','candidate_pool'))
        arrival=raw.get('arrival_process')==settings['arrival_process'] and (raw.get('arrival_process_details') or {}).get('name')==settings['arrival_process']
        window=raw.get('measurement_window') or {}
        measurement=raw.get('measurement_seconds')==settings['measurement_seconds'] and window.get('duration_seconds')==settings['measurement_seconds']
        measurement=measurement and window.get('cohort')=='requests scheduled in [start,end)' and window.get('completion_count_definition')=='successful cohort completions up to the drain deadline'
        drain=all(raw.get(k)==settings[k] for k in ('startup_timeout_seconds','drain_timeout_seconds','shutdown_timeout_seconds','max_pending_per_model'))
        inputs_equal=all(src.get(k)==ctx['source_hashes'][k] for k in ('pool','prompts'))
        precision=revisions and context_same and harness_same and drain
        if harness=='module8c_runtime_compat.runtime':
            compat=raw.get('zero_rate_transport_compatibility') or {}
            precision=precision and compat.get('source_sha256')==file_hash(ROOT/'module8c_runtime_compat/runtime.py') and compat.get('original_family_workload_preserved') is True
        checks={
            CONDITIONS[0]:raw['workload_rps']==state['lambda_rps'],
            CONDITIONS[1]:vector_equal,
            CONDITIONS[2]:observed_routes==state['routing_rps'] and not unknown_route and raw.get('routing_policy')==routing_policy and raw.get('routing_source')==settings['routing_source'],
            CONDITIONS[3]:revisions,
            CONDITIONS[4]:context_valid[state['workload']] and context_same and context_path_same and gpu_same and execution_env and harness_same,
            CONDITIONS[5]:quality,
            CONDITIONS[6]:arrival,
            CONDITIONS[7]:raw.get('warmup_seconds')==settings['warmup_seconds'],
            CONDITIONS[8]:bool(measurement and drain),
            CONDITIONS[9]:bool(record['Module8C_current_campaign'] and context_same and src.get('service_demand')==ctx['source_hashes']['service_demand'] and drain),
            CONDITIONS[10]:inputs_equal,
            CONDITIONS[11]:bool(precision),
        }
        config=record.get('config')
        config_equal=config is not None and all(raw.get(k)==v for k,v in config.items())
        checks={k:bool(v) for k,v in checks.items()}
        passed=all(checks.values()) and record['valid_timed_execution'] and config_equal
        return {'original_run_path':record['original_run_path'],'original_raw_sha256':record['raw_sha256'],
                'target_state_sha256':state['measurement_state_sha256'],'target_baseline':state['baseline'],
                'target_workload':state['workload'],'target_random_seed':state['random_seed'],
                'identity_equality_checks':checks,'valid_timed_execution':record['valid_timed_execution'],
                'raw_equals_original_frozen_config':config_equal,
                'source_config':record['config_reference'],'source_process':record['process_reference'],
                'target_context':definitions['measurement_protocol']['runtime_context_by_workload'][state['workload']],
                'original_context':ref(record['context_path']) if record['context_path'] and Path(record['context_path']).exists() else None,
                'phase':record['phase'],'measurement_random_seed':raw['random_seed'],'original_confirmation_role':record['confirmation_role'],
                'decision':'REUSE_ALLOWED' if passed else 'REUSE_FORBIDDEN',
                'failed_or_unproven_conditions':[k for k,v in checks.items() if not v]}
    return check


def audit(definitions,states):
    sources,frozen,manifest=verify_sources()
    records,coverage=scan_raws(manifest)
    check=make_checker(definitions,sources,frozen)
    comparisons=[];donors={};by_lambda={}
    for r in records:by_lambda.setdefault(digest(r['raw']['workload_rps']),[]).append(r)
    for state in states:
        key=(state['baseline'],state['workload'],state['random_seed'],state['measurement_state_sha256'])
        possible=[]
        for record in by_lambda.get(digest(state['lambda_rps']),[]):
            comparison=check(state,record);comparisons.append(comparison)
            if comparison['decision']=='REUSE_ALLOWED':possible.append(record)
        donors[key]=possible
    # Legacy workloads differ. Record the twelve checks against every frozen workload
    # instead of silently treating same deployment names as sufficient evidence.
    legacy=[]
    base_states=[s for s in states if s['baseline']=='B0']
    for record in records:
        if digest(record['raw']['workload_rps']) not in {digest(s['lambda_rps']) for s in base_states}:
            legacy.extend(check(s,record) for s in base_states)
    comparisons.extend(legacy)
    reuse=[];missing=[];state_groups=[]
    def key(s):return(s['baseline'],s['workload'],s['random_seed'],s['measurement_state_sha256'])
    def select(s,phase,seeds):
        candidates=[r for r in donors[key(s)] if r['phase']==phase]
        candidates.sort(key=lambda r:(not r['Module8C_current_campaign'],r['confirmation_role']!='seed',r['original_run_path']))
        chosen=[]
        for i,seed in enumerate(seeds,1):
            options=[r for r in candidates if r['raw']['random_seed']==seed]
            if options:
                r=options[0]
                chosen.append(r)
                comparison=check(s,r)
                reuse.append({'baseline':s['baseline'],'workload':s['workload'],'random_seed':s['random_seed'],
                    'state_sha256':s['measurement_state_sha256'],'phase':phase,'repetition':i,
                    'measurement_random_seed':seed,'original_run_path':r['original_run_path'],
                    'original_raw_sha256':r['raw_sha256'],'decision':'REUSE_ALLOWED',
                    'all_12_identity_conditions_equal':True,'equality_checks_sha256':digest(comparison),
                    'confirmation_original_role':r['confirmation_role']})
            else:
                missing.append({'baseline':s['baseline'],'workload':s['workload'],'random_seed':s['random_seed'],
                    'state_sha256':s['measurement_state_sha256'],'phase':phase,'repetition':i,
                    'measurement_random_seed':seed,'decision':'NEW_MEASUREMENT_REQUIRED',
                    'reason':'No original run passes all 12 exact identity checks and this phase/measurement-seed contract'})
        return chosen
    for s in states:
        search=select(s,'search',definitions['measurement_protocol']['measurement_seeds'])
        confirmation=[]
        if s['baseline']!='B4_INITIAL_CANDIDATE':
            confirmation=select(s,'confirmation',definitions['measurement_protocol']['fixed_baseline_independent_confirmation_seeds'])
        state_groups.append({'baseline':s['baseline'],'workload':s['workload'],'random_seed':s['random_seed'],
            'state_sha256':s['measurement_state_sha256'],'search_reused_runs':len(search),
            'confirmation_reused_runs':len(confirmation),
            'search_state_reusable':len(search)==3,
            'confirmation_state_reusable':len(confirmation)==2 if s['baseline']!='B4_INITIAL_CANDIDATE' else None})
    # Validate the rejection boundary using real raw evidence, without generating measurements.
    s=base_states[0];r=donors[key(s)][0]
    assert check(s,r)['decision']=='REUSE_ALLOWED'
    changed=copy.deepcopy(s);changed['lambda_rps']['text']=math.nextafter(changed['lambda_rps']['text'],math.inf)
    assert not check(changed,r)['identity_equality_checks'][CONDITIONS[0]]
    changed=copy.deepcopy(s);changed['routing_rps']['text']['distilbert']=math.nextafter(changed['routing_rps']['text']['distilbert'],math.inf)
    assert not check(changed,r)['identity_equality_checks'][CONDITIONS[2]]
    missing_quality=copy.deepcopy(r);missing_quality['raw']['source'].pop('quality_admission')
    assert check(s,missing_quality)['decision']=='REUSE_FORBIDDEN'
    inventory=[]
    target_lambda_hashes={digest(s['lambda_rps']) for s in base_states}
    for r in records:
        inventory.append({k:r[k] for k in ('original_run_path','raw_sha256','original_family','Module8C_current_campaign','phase','confirmation_role','valid_timed_execution')}
            | {'lambda_sha256':digest(r['raw']['workload_rps']),
               'lambda_matches_any_final_workload':digest(r['raw']['workload_rps']) in target_lambda_hashes,
               'deployment':r['raw']['deployment'],'source_input_hashes':r['raw']['source']})
    result={'module':'8D-1','required_identity_conditions':list(CONDITIONS),
        'policy':'Any false/unproven condition, invalid execution, missing exact config, or wrong phase/seed forbids reuse',
        'scan_coverage':coverage,'raw_inventory':inventory,'proposed_reuse_comparisons':comparisons,
        'measurement_selection':reuse,'missing_known_measurements':missing,'state_groups':state_groups,
        'unique_existing_raw_runs_selected':len({r['original_run_path'] for r in reuse}),
        'selected_measurement_references':len(reuse),
        'deduplication':'Same source raw referenced by several baselines/RNG seeds is one measurement, not independent replication',
        'B4_final_states':'Unknown before runtime; audit exact identity and require two final confirmation runs disjoint from that run seed confirmations',
        'rejection_boundary_tests':{'one_ULP_lambda_difference':'REUSE_FORBIDDEN','one_ULP_route_difference':'REUSE_FORBIDDEN','missing_quality_gate_hash':'REUSE_FORBIDDEN'},
        'no_GPU_experiments':True}
    freeze_file('measurement_reuse_audit.json',result)
    csvwrite('reuse_manifest.csv',reuse)
    csvwrite('missing_measurements.csv',missing)
    freeze_file('measurement_reuse_state_summary.json',{'states':state_groups})
    return result
