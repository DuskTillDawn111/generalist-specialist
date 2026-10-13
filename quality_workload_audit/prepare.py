import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from candidate_pool_audit.registry import registry
OUT=ROOT/'results/quality_workload_audit'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):
 p=OUT/name
 text=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
 if p.exists() and p.read_text()!=text:raise RuntimeError('Refuse frozen overwrite '+name)
 p.write_text(text)

def main():
 pool=json.loads((ROOT/'results/candidate_pool_v2/candidate_models.json').read_text())
 catalog=json.loads((ROOT/'results/candidate_pool_v2_service_demand/updated_service_demand_catalog.json').read_text())
 definitions=json.loads((ROOT/'results/candidate_pool_v2/task_definitions.json').read_text())
 protected={}
 for directory in ['deployment_core','deployment_search','integer_replica_pilot','integer_replica_multiworkload','replica_capacity_calibration','candidate_pool_audit','candidate_pool_service_demand','joint_deploy','benchmark/scripts','benchmark/manifests','benchmark/prompts','benchmark/profiles','results/candidate_pool_v2','results/candidate_pool_v2_service_demand','results/integer_replica_search_multiworkload','results/replica_capacity_calibration','results/deployment_search_multiworkload']:
  for p in (ROOT/directory).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl','.csv','.md','.sha256'):
    protected[str(p.relative_to(ROOT))]=digest(p)
 write('protected_input_hashes.json',protected)
 tasks={}
 for task,definition in definitions.items():
  p=Path(definition['evaluation_manifest'])
  assert digest(p)==definition['manifest_sha256']
  rows=[json.loads(l) for l in p.read_text().splitlines() if l]
  assert len(rows)==200 and len({s['id'] for s in rows})==200
  tasks[task]=dict(definition,sample_ids=[s['id'] for s in rows],scope='existing frozen 200-sample evaluation subset; not full dataset benchmark',
                   input_payload_sha256=digest(p),quality_metrics=['cer','exact_match'] if task=='ocr' else ['wer'] if task=='asr' else ['top1_accuracy'] if task=='image' else ['accuracy'],
                   sample_asset_hashes={s['id']:digest(Path(s['input'])) for s in rows if task!='text'})
 models={}
 configs=registry()
 for model in pool['models']:
  name=model['id']
  if name not in catalog['models']:continue
  script=configs[name]['script']
  verified=list(model['capabilities_verified'])
  assert set(verified)==set(catalog['models'][name]['capabilities_verified'])
  extra=['visual_qa'] if name in ('phi4mm','minicpm') else []
  models[name]={'source_revision':model['source_revision'],'local_directory':pool['metadata_notes'][name]['local_directory'],
                'precision':model['precision'],'verified_tasks':verified,'tasks':verified+extra,
                'revalidation_only_tasks':extra,'python':str(configs[name]['python']),
                'adapter':str(script),'adapter_sha256':digest(script),
                'decoding_contract':'Unchanged current frozen serving handler for verified paths; ScienceQA revalidation max_new_tokens=1, do_sample=False, same prompt and strict normalization',
                'serving_token_caps':{'text':16,'image':16,'ocr':32,'asr':256,'visual_qa':1,'lid':16} if model['role']=='generalist' else 'exact current specialist adapter; see hashed source'}
 history=[]
 for p in sorted((ROOT/'benchmark/results/formal').glob('*_summary.json')):
  d=json.loads(p.read_text())
  history.append({'path':str(p),'sha256':digest(p),'num_samples':d.get('num_samples'),
                  'generation':d.get('generation'),'decision':'NOT_REUSED',
                  'reason':'Historical summary/predictions do not bind all required model revision, current adapter/preprocessing, manifest and decoding hashes to this result. Re-evaluate current frozen serving path on identical 200 IDs.'})
 scenario=ROOT/'benchmark/profiles/placement_scenario.json'
 manifest={'module':'8A','frozen_before_quality_results':True,'scope':'measurement + audit, no deployment search',
           'tasks':tasks,'models':models,'verified_path_count':sum(len(m['verified_tasks']) for m in models.values()),
           'historical_reuse_audit':history,'quality_threshold_status':'UNSPECIFIED_FOR_CURRENT_SIX_TASK_V2_OPTIMIZER',
           'legacy_quality_presets':{'path':str(scenario),'sha256':digest(scenario),'presets':json.loads(scenario.read_text())['quality_presets'],
                                     'scope':'Historical four-task placement presets; no evidence of a selected formal v2 six-task optimizer threshold; visual_qa/lid unspecified'},
           'quality_thresholds_created':False,'f1_status':'Existing task definitions and active scripts use accuracy, not F1; no metric replacement',
           'model_provenance':{'candidate_models_sha256':digest(ROOT/'results/candidate_pool_v2/candidate_models.json'),
                               'frozen_download_integrity_sha256':digest(ROOT/'results/candidate_pool_v2/download_integrity.json'),
                               'note':'Local frozen model directories and revisions retained; per-file current model hashes stored separately'},
           'evaluation_method':'one isolated worker per model; existing handlers unchanged; every sample scored including invalid/empty output; corpus WER/CER; no timing from this run substitutes for s0'}
 assert manifest['verified_path_count']==25
 write('quality_evaluation_manifest.json',manifest)
 print('Frozen quality manifest: 25 verified paths + 2 VQA revalidations, 200 IDs per task')
if __name__=='__main__':main()
