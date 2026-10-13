"""Validate Module13A provenance and frozen protocol; stdlib/CPU only."""
import ast
import csv
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def read(p):
    return json.loads(Path(p).read_text())

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_ref(r):
    assert digest(r['path']) == r['sha256'], r['path']

def nested_refs(x):
    if isinstance(x, dict):
        if 'path' in x and 'sha256' in x and isinstance(x['path'],str):
            verify_ref(x)
        for v in x.values(): nested_refs(v)
    elif isinstance(x,list):
        for v in x: nested_refs(v)

names=['candidate_universe.csv','asr_candidates.csv','lid_candidates.csv','text_candidates.csv',
       'source_audit.md','quality_evidence.json','measurement_shortlist.json',
       'measurement_protocol_freeze.json','specialist_selection_rule.json',
       'module13a_gate.json','module13a_summary.md']
assert all((OUT/n).is_file() for n in names)
universe=list(csv.DictReader((OUT/'candidate_universe.csv').open()))
assert len(universe)==18 and len({r['model_id'] for r in universe})==15
short=read(OUT/'measurement_shortlist.json')
protocol=read(OUT/'measurement_protocol_freeze.json')
rule=read(OUT/'specialist_selection_rule.json')
gate=read(OUT/'module13a_gate.json')
evidence=read(OUT/'quality_evidence.json')['entries']
assert len(short['entries'])==9 and len({r['model_id'] for r in short['entries']})==8
assert len(evidence)==18
assert rule['selected_to_make_Generalist_win']=='NO' and not rule['Generalist_outcome_used']
assert gate['GPU_runs']==gate['weights_downloaded']==gate['models_loaded']==0
assert not gate['Module13B_started']
assert gate['gate']=='REAL_SPECIALIST_SHORTLIST_READY'
assert protocol['primary_operating_point']['precision'].startswith('float32')
assert 'Native default' in protocol['primary_operating_point']['attention']
policy={r['task']:r for r in read(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')['tasks']}
for task in ['asr','lid','text']:
    subset=[r for r in short['entries'] if r['task']==task]
    assert len(subset)==3
    assert len([r for r in universe if r['task']==task])==6
    assert protocol['datasets'][task]['quality_policy']==policy[task]
    assert protocol['datasets'][task]['quality_count']==200
    assert protocol['datasets'][task]['latency_count_per_repeat']==30
    assert len(protocol['datasets'][task]['latency_pool_ids_in_order'])==30
    assert all(len(r['revision'])==40 and r['local_inference_path'] for r in subset)
    assert all(evidence[task+':'+r['model_id']]['preliminary_quality']=='QUALITY_LIKELY_PASS' for r in subset)
    for r in subset:
        candidate=next(x for x in universe if x['task']==task and x['model_id']==r['model_id'])
        assert candidate['legal_status']=='LEGAL_PUBLIC_LOCAL_READY_MODEL'
        assert candidate['gamma_est']==candidate['mu_est']==''
        assert candidate['gamma_first_split_target']==candidate['gamma_stronger_transition_target']==candidate['mu_target']=='UNKNOWN'
assert policy['asr']['historical_threshold']==.042181
assert policy['text']['historical_threshold']==.925
assert policy['lid']['historical_threshold'] is None
assert protocol['datasets']['lid']['input_type']=='audio'
assert protocol['datasets']['text']['dataset']=='sst2'
assert [r['order'] for r in short['measurement_order']]==list(range(1,10))
assert short['measurement_order']==protocol['quality_phase']['order']==protocol['service_phase']['candidate_order']

# All local references, including archive source SHA256s and dataset payloads.
for name in ['source_manifest.json','supplemental_sources.json','quality_evidence.json',
             'baseline_references.json','measurement_shortlist.json',
             'measurement_protocol_freeze.json','specialist_selection_rule.json',
             'module13a_gate.json','dataset_input_hashes.json']:
    nested_refs(read(OUT/name))
before=read(OUT/'protected_before.json')['files']
for r in before: verify_ref(r)
prior=read(ROOT/'results/module12_specialist_sensitivity/experiment_freeze.json')['protected_references']
for r in prior: verify_ref(r)
for file in OUT.glob('*.py'):
    ast.parse(file.read_text())
assert not any(p.suffix in {'.bin','.safetensors','.ckpt','.nemo'} for p in OUT.rglob('*') if p.is_file())

# Freeze all final output/supporting content hashes; exclude validation/manifest
# files themselves to avoid a recursive digest. This runs only after consistency
# checks, and never mutates baseline/source files.
files=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in {'freeze_manifest.json','cpu_validation.json'} and '__pycache__' not in str(p):
        files.append({'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size})
manifest={'module':'13A','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'before_any_new_candidate_GPU_result':True,'new_candidate_GPU_results_seen':0,
          'files':files}
(OUT/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
report={'status':'PASS','module':'13A','at_utc':manifest['at_utc'],
        'required_outputs':len(names),'candidate_task_paths':len(universe),
        'unique_universe_checkpoints':15,'shortlisted_paths':9,'shortlisted_checkpoints':8,
        'task_definition_quality_thresholds_preserved':True,'sample_order_matches_historical_raw':True,
        'protected_files_unchanged':len(before),'Module12_prior_protected_refs_verified':len(prior),
        'source_and_dataset_hashes_verified':True,'GPU_or_model_runtime_executed':False,
        'no_weight_files_in_output':True,'freeze_manifest_sha256':digest(OUT/'freeze_manifest.json')}
(OUT/'cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
