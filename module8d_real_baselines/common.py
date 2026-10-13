import copy, json, math, os, subprocess
from collections import Counter
from pathlib import Path
from module8c_final_real_search.common import ROOT, B, P, OUT as C, inputs, read, ref, verify_freeze
from deployment_search.runtime_cache import file_hash, digest, save_json
from candidate_pool_audit.provenance import environment_snapshot

F=ROOT/'results/module8d_baseline_freeze'
OUT=ROOT/'results/module8d_real_baselines'
TASKS=('text','image','ocr','asr','visual_qa','lid')


def verify_inputs():
    # Never call the 8D-1 generator. A mismatched hash is a hard stop.
    f=read(F/'baseline_freeze_hashes.json')
    assert file_hash(F/'baseline_freeze_hashes.json')==(F/'baseline_freeze_hashes.json.sha256').read_text().strip()
    for key in ('artifacts','implementation_files','inputs'):
        for name,row in f[key].items():assert file_hash(row['path'])==row['sha256'],(key,name)
    assert read(F/'baseline_fairness_audit.json')['BASELINE_FREEZE_PASS']
    verify_freeze()
    for row in read(C/'artifact_manifest.json')['artifacts'].values():
        assert file_hash(row['path'])==row['sha256'],row['path']
    for row in read(C/'zero_rate_transport_recovery.json')['compatibility_sources'].values():
        assert file_hash(row['path'])==row['sha256']
    manifest=OUT/'campaign_manifest.json'
    if manifest.exists():
        for row in read(manifest)['implementation_files'].values():
            assert file_hash(row['path'])==row['sha256'],row['path']
    return f


def prepare():
    verify_inputs()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'campaign_manifest.json').exists():return read(OUT/'campaign_manifest.json')
    # No package installation, no model loading, no smoke experiment.
    env=environment_snapshot();assert env==read(C/'environment_manifest.json')
    sources=read(C/'model_source_hashes.json');observed={}
    expected=read(F/'source_identity_validation.json')
    files={r['path']:r['actual_sha256'] for r in expected['native_files'].values()}
    files.update({r['path']:r['sha256'] for r in expected['model_manifest_files'].values()})
    for i,(path,sha) in enumerate(files.items(),1):
        actual=file_hash(path);assert actual==sha,path
        observed[path]=actual
        if i%40==0:print(json.dumps({'event':'source_verified','files':i,'total':len(files)}),flush=True)
    from replica_scaling.telemetry import snapshot
    gpu=snapshot();old=read(C/'experiment_freeze.json')['gpu']
    assert gpu['uuid']==old['uuid'] and gpu['driver']==old['driver'] and gpu['total_mib']==old['total_mib']
    assert not gpu['processes'] and gpu['used_mib']<=128,'GPU not idle: no competing experiment permitted'
    definitions=read(F/'baseline_definitions.json')
    names=list(definitions['measurement_protocol']['runtime_context_by_workload'])
    analysis={
        'frozen_before_first_baseline_measurement':True,
        'primary_latency_analysis':'Search-protocol aggregate: three equal-weight repetitions, unchanged metric semantics',
        'independent_validation':'Two equal-weight final confirmations; report separately, never feed back into decisions',
        'objective':'Existing NoiseAwareRuntimeObjective(mean_tolerance_ms=1,min_relative_improvement=.05)',
        'win_tie_loss':{'win':'Proposed strictly improves Random under existing objective, reverse comparison does not',
            'loss':'Random strictly improves Proposed under existing objective, reverse comparison does not',
            'tie':'Neither direction strictly improves under existing objective; not a claim of identical latencies',
            'incomparable':'Missing valid evidence/resource failure; never invent a stability or latency'},
        'memory_tie_break':'Frozen empirical deployment memory, exactly as search; actual sampled memory reported separately',
        'claim_labels':{'INSUFFICIENT':'Any missing required workload/RNG final search or confirmation evidence',
            'SUPPORTED':'Proposed wins every required comparison under the frozen objective, both search and confirmation',
            'PARTIAL':'At least one search win also confirmed, but not every required comparison wins in both phases',
            'NOT_SUPPORTED':'No search win confirmed by independent confirmation',
            'LP_efficiency_exception':'For E only, if no confirmed objective win, PARTIAL is allowed for objective ties plus fewer logical evaluations in at least two distinct workloads, provided no search/confirmation loss in those comparisons. Report as efficiency evidence, not latency dominance'},
        'comparison_sets':{'A':'B0 six workloads','B':'B1 six workloads','C':'B2 six workloads',
            'D':'B3 six workloads','E':'B4 all 18 frozen workload/RNG conditions'},
        'random_reporting':'All 1103/2207/47; equal-weight mean, sample std ddof=1, median and range; no best-of selection',
        'repetition_reporting':'Per condition n=3 search, n=2 confirmation, sample std ddof=1; no invalid/unstable result deletion',
        'paired_analysis':'No paired inferential claims; shared donors are not independent new samples',
        'relative_improvement':'100*(baseline-Proposed)/baseline, signed; undefined for absent/zero denominator',
        'quality_reporting':'Same frozen routing-fraction weighted audit metric; report-only, not objective',
        'retry_policy':'Existing 8D-1/evaluator policy has zero automatic retries; genuine unstable is valid, resource failure retained, external interruption recorded and stops the affected run',
        'no_significance_or_global_optimum_claim':True}
    save_json(OUT/'analysis_policy_freeze.json',analysis)
    (OUT/'analysis_policy_freeze.json.sha256').write_text(file_hash(OUT/'analysis_policy_freeze.json')+'\n')
    save_json(OUT/'source_preflight.json',{'status':'PASS','environments':env,'native_hashes':observed,
        'gpu':gpu,'same_model_revisions_precision_runtime_backend':True,'new_GPU_measurements':0})
    manifest={'module':'8D-2','frozen_before_first_GPU_measurement':True,
        'baseline_freeze':ref(F/'baseline_freeze_hashes.json'),
        'baseline_definitions':ref(F/'baseline_definitions.json'),'reuse_audit':ref(F/'measurement_reuse_audit.json'),
        'reuse_contract':ref(F/'runtime_reuse_execution_contract.json'),
        'analysis_policy':ref(OUT/'analysis_policy_freeze.json'),'source_preflight':ref(OUT/'source_preflight.json'),
        'phase_A':[{'baseline':b,'workload':n} for b in ('B0','B1','B2','B3') for n in names],
        'phase_B':[{'baseline':'B4','random_seed':s,'workload':n} for n in names for s in definitions['B4']['random_seeds']],
        'random_seeds':definitions['B4']['random_seeds'],'runtime_config':definitions['B4']['search_config'],
        'measurement_settings':definitions['measurement_protocol']['settings'],
        'same_original_frozen_context_paths':definitions['measurement_protocol']['runtime_context_by_workload'],
        'implementation_files':{str(p.relative_to(ROOT)):ref(p) for p in sorted((ROOT/'module8d_real_baselines').glob('*.py'))},
        'physical_run_cap':read(F/'gpu_campaign_budget.json')['campaign']['conservative_new_physical_run_ceiling'],
        'cross_job_exact_cache_reuse_required':True,'Phase_A_results_modify_Phase_B':False,
        'Proposed_rerun':False,'baseline_definitions_modified':False,'core_search_or_runtime_algorithm_modified':False,
        'new_predictor':False,'new_routing_optimizer':False,
        'ablation':False,'dynamic':False,'second_hardware':False,'stop_after_module':'8D-2'}
    save_json(OUT/'campaign_manifest.json',manifest)
    (OUT/'campaign_manifest.json.sha256').write_text(file_hash(OUT/'campaign_manifest.json')+'\n')
    return manifest


def validate_raw(path,identity,seed,workload,valid_required=True):
    """Twelve independent equalities plus exact config and run validity evidence."""
    path=Path(path);raw=read(path);ctx=read(C/workload/'runtime_context.json')
    protocol=read(F/'baseline_definitions.json')['measurement_protocol'];settings=protocol['settings']
    config_path=path.with_suffix('.config.json');process_path=path.with_suffix('.process.json')
    config=read(config_path) if config_path.exists() else {}
    process=read(process_path) if process_path.exists() else {};command=process.get('command',[])
    sources=read(C/'model_source_hashes.json');workers=raw.get('worker_identities',{})
    active={m:n for m,n in identity['deployment'].items() if n}
    source_eq=bool(workers) and Counter(v['family'] for v in workers.values())==Counter(active)
    source_eq=source_eq and all(v.get('model_source')==sources[v['family']] for v in workers.values())
    ctx_eq=raw.get('runtime_context_sha256')==ctx['context_sha256']
    harness=protocol['harness_by_workload'][workload]
    harness_eq=command[1:3]==['-m',harness] and '--context' in command and command[command.index('--context')+1]==str(C/workload/'runtime_context.json')
    src=raw.get('source',{});window=raw.get('measurement_window') or {};gpu=raw.get('baseline_gpu',{})
    oldgpu=read(C/'experiment_freeze.json')['gpu']
    current_context_ref=ref(C/workload/'runtime_context.json')
    routing_policy='Frozen family smooth weighted round-robin, followed by deterministic round-robin across that family replicas; counters reset each phase.'
    checks={
        'exact_workload_lambda':raw.get('workload_rps')==identity['workload'],
        'exact_replica_vector':raw.get('deployment')==identity['deployment'],
        'exact_routing_semantics_and_fractions':raw.get('routing_rps')==identity['routing'] and raw.get('routing_policy')==routing_policy and raw.get('routing_source')==settings['routing_source'],
        'same_model_revisions':bool(source_eq),
        'same_environments':ctx_eq and harness_eq and all(gpu.get(k)==oldgpu[k] for k in ('uuid','driver','name','total_mib')) and src.get('execution_environment')==ctx['source_hashes']['execution_environment'],
        'same_quality_admission_path_set':all(src.get(k)==ctx['source_hashes'][k] for k in ('candidate_pool','quality_policy','quality_admission')),
        'same_arrival_generation_protocol':raw.get('arrival_process')==settings['arrival_process'] and raw.get('arrival_process_details',{}).get('name')==settings['arrival_process'],
        'same_warmup':raw.get('warmup_seconds')==settings['warmup_seconds'],
        'same_measurement_window':raw.get('measurement_seconds')==settings['measurement_seconds'] and window.get('duration_seconds')==settings['measurement_seconds'] and window.get('cohort')=='requests scheduled in [start,end)' and window.get('completion_count_definition')=='successful cohort completions up to the drain deadline',
        'same_runtime_metric_definition':ctx_eq and src.get('service_demand')==ctx['source_hashes']['service_demand'] and identity['runtime_context_sha256']==current_context_ref['sha256'],
        'same_task_input_source_subset':all(src.get(k)==ctx['source_hashes'][k] for k in ('pool','prompts')),
        'same_precision_runtime_backend':bool(source_eq and ctx_eq and harness_eq),
    }
    timing=all(raw.get(k)==v for k,v in settings.items())
    if workload=='no_vqa_phase':
        compat=raw.get('zero_rate_transport_compatibility',{})
        checks['same_precision_runtime_backend'] &= compat.get('source_sha256')==file_hash(ROOT/'module8c_runtime_compat/runtime.py') and compat.get('original_family_workload_preserved') is True
    expected_config=dict(deployment=identity['deployment'],workload_rps=identity['workload'],routing_rps=identity['routing'],random_seed=seed,**settings)
    config_equal=config==expected_config and all(raw.get(k)==v for k,v in config.items())
    valid=bool(raw.get('valid_execution') and window)
    admitted={(r['task'],r['model']) for r in read(B/'quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    if valid:
        for field in ('request_records','warmup_request_records','priming_request_records'):
            assert all((r['task'],r['model_family']) in admitted for r in raw.get(field,[])),field
    result={'original_run':ref(path),'target_identity_sha256':digest(identity),'measurement_seed':seed,
        'equality_checks':checks,'same_timing_settings':timing,'exact_config_and_seed':config_equal,
        'valid_timed_execution':valid,'decision':'REUSE_ALLOWED' if all(checks.values()) and timing and config_equal and valid else 'REUSE_FORBIDDEN',
        'metric_environment_evidence':ref(OUT/'campaign_manifest.json')}
    if valid_required:assert result['decision']=='REUSE_ALLOWED',result
    return result


def full_vector(x):
    return {m:x.get(m,0) for m in sorted(read(P/'candidate_pool_v2_1.json')['formal_model_ids'])}


def effective_quality(identity):
    admitted={(r['task'],r['model']):r for r in read(B/'quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    result={}
    for task in TASKS:
        rate=identity['workload'][task];parts=[]
        for m,value in identity['routing'][task].items():
            if value>0:
                r=admitted[task,m]
                parts.append({'model':m,'fraction':value/rate,'metric':r['quality_metric'],
                    'quality':r['measured_quality'],'secondary_cer':r['secondary_cer'],'source':r['source']})
        result[task]={'active':rate>0,'metric':parts[0]['metric'] if parts else None,
            'routing_fraction_weighted_audit_metric':math.fsum(p['fraction']*p['quality'] for p in parts) if parts else None,
            'secondary_CER':math.fsum(p['fraction']*p['secondary_cer'] for p in parts) if parts and all(p['secondary_cer'] is not None for p in parts) else None,
            'components':parts,'quality_admission_compliant':True,'report_only_not_runtime_objective':True}
    return result
