import ast,copy,hashlib,json
from .common import *

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);POOL_OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'before_hashes.json').exists():raise RuntimeError('Preparation already frozen')
 protected={}
 for folder in ('deployment_core','deployment_search','integer_replica_pilot','integer_replica_multiworkload','replica_capacity_calibration','candidate_pool_audit','candidate_pool_service_demand','quality_workload_audit','joint_deploy','benchmark/scripts','benchmark/evaluation','benchmark/manifests','benchmark/task_pools','benchmark/prompts','benchmark/profiles','results/candidate_pool_v2','results/candidate_pool_v2_service_demand','results/quality_workload_audit','results/replica_capacity_calibration','results/integer_replica_search_multiworkload','results/integer_replica_search_real_pilot'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:protected[str(p.relative_to(ROOT))]=ref(p)
 write(OUT/'before_hashes.json',{'files':protected,'new_model_types_allowed':0},freeze=True)
 before=read(ROOT/'results/candidate_pool_v2/candidate_models.json')
 pool=copy.deepcopy(before)
 verified=read(QUALITY/'generalist_vqa_revalidation.json')['results']
 diffs=[]
 for model in pool['models']:
  name=model['id']
  if name not in ('phi4mm','minicpm'):continue
  e=verified[name]
  assert e['status']=='VERIFIED' and all(e[k] for k in ['same_semantics','same_prompt','same_metric','same_answer_normalization'])
  assert e['model_revision']==model['source_revision']
  assert sha(e['raw_evidence'])==e['raw_sha256']
  model['capabilities_verified'].append('visual_qa')
  model['capabilities_pending'].remove('visual_qa')
  model['task_compatibility']['visual_qa']='verified'
  diffs.append({'model':name,'task':'visual_qa','old_status':'pending','new_status':'verified','evidence':ref(QUALITY/'generalist_vqa_revalidation.json'),'quality_accuracy':e['quality']['accuracy']})
 pool['pool_version']='2.1';pool['frozen']=True
 formal=read(ROOT/'results/candidate_pool_v2/pool_resource_summary.json')['formal_model_ids']
 pool['formal_model_ids']=formal
 pool['base_pool_source']=ref(ROOT/'results/candidate_pool_v2/candidate_models.json')
 pool['versioned_adapters']={name:{'launcher':ref(ROOT/f'candidate_pool_v2_1_adapters/worker_{name}.py'),
                                  'extension':ref(ROOT/'candidate_pool_v2_1_adapters/vqa_extension.py'),
                                  'original_handler':ref(registry()[name]['script']) if name not in ('phi4mm','minicpm') else ref(ROOT/f"joint_deploy/{'phi_worker.py' if name=='phi4mm' else 'minicpm_worker.py'}")}
                            for name in ('phi4mm','minicpm')}
 write(POOL_OUT/'candidate_pool_v2_1.json',pool,freeze=True);write(OUT/'candidate_pool_v2_1.json',pool,freeze=True)
 # Verify executable inference function body is identical to Module 8A's body,
 # except version metadata; original task handlers use the same original source.
 old=ast.parse((ROOT/'quality_workload_audit/vqa_worker.py').read_text())
 new=ast.parse((ROOT/'candidate_pool_v2_1_adapters/vqa_extension.py').read_text())
 old_handler=next(n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name=='audited_handler')
 new_handler=next(n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name=='versioned_handler')
 old_handler.name=new_handler.name='semantic_handler'
 # Compare actual inference/prompt/normalization statements, excluding return
 # metadata dict's version marker. Other scoring fields stay byte-for-byte.
 old_handler.body[-1].value.keys.pop();old_handler.body[-1].value.values.pop()
 new_handler.body[-1].value.keys.pop();new_handler.body[-1].value.values.pop()
 assert ast.dump(old_handler)==ast.dump(new_handler)
 diff={'base_version':'2','new_version':'2.1','capability_updates':diffs,'vqa_capable_generalists_old':['qwen'],
       'vqa_capable_generalists_new':['qwen','phi4mm','minicpm'],'vqa_capable_count_old':1,'vqa_capable_count_new':3,
       'old_v2_overwritten':False,'new_model_types':0,'unchanged_fields':['model IDs','revisions','weights','precision','roles','idle VRAM','all other capabilities'],
       'semantic_adapter_equivalence':'PASS: AST-equivalent Module 8A revalidated inference/prompt/scoring function; only version metadata changed',
       'coverage_requirement':'For lambda_visual_qa>0, sum presence of admitted VQA-capable models >=1; qwen>=1 no longer follows from capability coverage',
       'quality_requirement':'VQA-capable does not imply numeric quality compliance; consult frozen quality policy/admission matrix'}
 write(OUT/'candidate_pool_v2_to_v2_1_diff.json',diff,freeze=True)
 catalog=read(SERVICE_OLD)
 assert sum(len(v['service_time_ms']) for v in catalog['models'].values())==25
 targets=[json.loads(l) for l in TASK_POOL.read_text().splitlines() if l and json.loads(l)['task_type']=='visual_qa'][:30]
 assert len(targets)==30
 # Ensure the profiling inputs have exactly the same samples as the frozen
 # evaluation manifest, not a convenient substitute dataset.
 evalrows={s['id']:s for s in [json.loads(l) for l in (ROOT/'benchmark/manifests/visual_qa.jsonl').read_text().splitlines()]}
 assert all(row['sample']==evalrows[row['sample']['id']] for row in targets)
 pending=[{'model_id':name,'task':'visual_qa','capability_status':'verified','existing_value':None,
           'repetitions':3,'requests_per_repetition':30,'warmups_per_repetition':1,
           'target_pool_ids':[r['pool_id'] for r in targets],
           'input_subset_sha256':hashlib.sha256(json.dumps(targets,sort_keys=True).encode()).hexdigest(),
           'adapter':pool['versioned_adapters'][name],'model_revision':verified[name]['model_revision']}
          for name in ('phi4mm','minicpm') if 'visual_qa' not in catalog['models'][name]['service_time_ms']]
 assert len(pending)==2
 write(POOL_OUT/'pending_service_demand_paths.json',{'paths':pending,'pending_path_count':len(pending),'protocol':'active_single_profile_v1'},freeze=True)
 write(POOL_OUT/'standalone_protocol.json',{'protocol':'active_single_profile_v1','implementation':ref(ROOT/'joint_deploy/run_active_single_profile.py'),
       'reuse_profile_wrapper':ref(ROOT/'candidate_pool_service_demand/profile.py'),'sample_pool':ref(TASK_POOL),
       'target_ids':[r['pool_id'] for r in targets],'requests_per_repetition':30,'independent_repetitions':3,
       'warmup':'first target once, request_id=-1, excluded','aggregation':'sample-count-weighted arithmetic mean of worker_latency_ms',
       'unit':'milliseconds','quantity':'(handler end_ns - start_ns)/1e6; unchanged worker_protocol timer',
       'decoding':{'max_new_tokens':1,'do_sample':False,'normalization':'strip + uppercase only'},
       'existing_25_paths_remeasured':0,'existing_25_values_modified':0},freeze=True)
 print('Frozen v2.1 capability pool; prepared 2 paths × 3 repetitions × 30 samples')
if __name__=='__main__':prepare()
