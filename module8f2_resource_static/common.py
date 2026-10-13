import copy,csv,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
from deployment_search.runtime_cache import file_hash,digest,save_json
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'results/module8f1a_memory_breakpoints';F=ROOT/'results/module8f1b_intensity_grid_freeze';OUT=ROOT/'results/module8f2_resource_static';PROTOCOL=ROOT/'results/module8e2_heldout_freeze/method_comparison_freeze.json'
def read(p):return json.loads(Path(p).read_text())
def ref(p):return {'path':str(Path(p).resolve()),'sha256':file_hash(p)}
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,d):save_json(OUT/n,d)
def csvwrite(n,rows,columns=None):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns or list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def inputs():
 from module8c_final_real_search.common import inputs as original
 return original()
def verify_frozen():
 checks=[]
 for directory in (A,F):
  for r in read(directory/'artifact_hashes.json').values():
   actual=file_hash(r['path']);checks.append(dict(r,actual=actual,match=actual==r['sha256']))
 for r in read(F/'resource_campaign_gate.json')['frozen_source_hashes'].values():
  actual=file_hash(r['path']);checks.append(dict(r,actual=actual,match=actual==r['sha256']))
 protocol=read(PROTOCOL)
 for r in protocol['frozen_inputs'].values():
  actual=file_hash(r['path']);checks.append(dict(r,actual=actual,match=actual==r['sha256']))
 for r in read(ROOT/'results/module8e1_method_revision/method_v2_source_hashes.json')['sources'].values():
  actual=file_hash(r['path']);checks.append(dict(r,actual=actual,match=actual==r['sha256']))
 verification={'status':'PASS' if all(c['match'] for c in checks) else 'FAIL','checked_at_utc':utc(),'checks':checks,'new_GPU_runs':0}
 write('hash_verification.json',verification)
 assert verification['status']=='PASS','Frozen hash mismatch; STOP without regeneration'
 assert read(F/'resource_campaign_gate.json')['RESOURCE_STATIC_GPU_CAMPAIGN_READY']
 assert read(F/'validation_results.json')['status']=='PASS'
 cells=read(F/'campaign_cells.json')['cells'];assert len(cells)==12
 levels=read(F/'frozen_intensity_levels.json');factor=read(A/'memory_safety_factor.json')['safety_factor']
 for c in cells:
  assert hashlib.sha256(json.dumps(c['lambda_rps'],sort_keys=True).encode()).hexdigest()==c['workload_hash']
  level=next(l for l in levels['levels'] if l['id']==c['intensity_level']);record=level['mixtures'][c['mixture']]
  assert c['alpha']==record['alpha'] and c['beta']==level['beta'] and c['lambda_rps']==record['lambda_rps']
  assert c['planning_budget_input_mib']==c['memory_budget_mib']/factor
 return checks

def prepare():
 OUT.mkdir(exist_ok=True);checks=verify_frozen();protocol=read(PROTOCOL)
 from candidate_pool_audit.provenance import environment_snapshot
 from replica_scaling.telemetry import snapshot
 sources=read(protocol['frozen_inputs']['model_sources']['path']);assets=[]
 for model,record in sources.items():
  for name,r in record['model_files'].items():
   path=ROOT/name;actual=file_hash(path);assets.append({'model':model,'path':str(path),'expected':r['sha256'],'actual':actual,'match':actual==r['sha256']});assert actual==r['sha256'],str(path)
  for key in ['adapter','vqa_extension']:
   if key in record:assert file_hash(record[key]['path'])==record[key]['sha256']
  print(json.dumps({'event':'model_assets_verified','model':model}),flush=True)
 native=read(ROOT/'results/integer_replica_search_real_pilot/before_hashes.json')['native_models_and_adapters']
 for name,r in native.items():assert file_hash(ROOT/name)==r['sha256'],name
 env=environment_snapshot();assert env==read(protocol['frozen_inputs']['environment']['path']),'Runtime environment mismatch; STOP'
 gpu=snapshot();old=read(ROOT/'results/module8c_final_real_search/experiment_freeze.json')['gpu']
 verification=read(OUT/'hash_verification.json')
 verification.update(frozen_hash_status=verification['status'],model_assets=assets,native_manifest_files_verified=len(native),live_environment_matches=True,expected_GPU=old,live_GPU=gpu,GPU_identity_matches={k:gpu[k]==old[k] for k in ['uuid','name','driver','total_mib']},verification_complete_at_utc=utc())
 authorization=read(OUT/'hardware_identity_exception.json') if (OUT/'hardware_identity_exception.json').exists() else None
 if not verification['GPU_identity_matches']['uuid'] and authorization:
  assert authorization['authorized_uuid']==gpu['uuid'] and authorization['UUID_only_exception']
  verification['user_authorized_GPU_UUID_exception']=ref(OUT/'hardware_identity_exception.json')
 if not all(verification['GPU_identity_matches'].values()):verification['status']='PASS_WITH_USER_AUTHORIZED_UUID_EXCEPTION' if authorization and all(verification['GPU_identity_matches'][k] for k in ['name','driver','total_mib']) else 'BLOCKED_GPU_IDENTITY_MISMATCH'
 write('hash_verification.json',verification)
 for k in ['name','driver','total_mib']:assert gpu[k]==old[k],'GPU identity mismatch '+k
 assert gpu['uuid']==old['uuid'] or authorization,'GPU UUID exception requires explicit user authorization'
 assert not gpu['processes'] and gpu['used_mib']<=128,'GPU isolation failed'
 verification=read(OUT/'hash_verification.json');verification.update(model_assets=assets,native_manifest_files_verified=len(native),live_environment_matches=True,live_GPU=gpu,verification_complete_at_utc=utc());write('hash_verification.json',verification)
 # Exact resource-limit semantics are part of context identity. Physical carrier
 # remains 49140 MiB; gpu_capacity_mib is the active per-cell peak limit B.
 template=read(protocol['runtime_context_template']['path']);contexts={}
 for cell in read(F/'campaign_cells.json')['cells']:
  ctx=copy.deepcopy(template);ctx['gpu_uuid']=gpu['uuid'];ctx['gpu_capacity_mib']=cell['memory_budget_mib'];ctx['physical_gpu_capacity_mib']=49140;ctx['resource_budget_semantics']={'B_mib':cell['memory_budget_mib'],'planning_budget_mib':cell['planning_budget_input_mib'],'epsilon_mem':read(A/'memory_safety_factor.json')['epsilon_mem'],'safe_memory_required':True,'actual_peak_limit_mib':cell['memory_budget_mib']}
  if authorization:ctx['source_hashes']['user_authorized_UUID_exception']=ref(OUT/'hardware_identity_exception.json')
  ctx['source_hashes']['workloads']=ref(F/'campaign_cells.json');ctx['source_hashes']['resource_grid']=ref(F/'final_memory_budgets.json');ctx['source_hashes']['memory_safety']=ref(A/'memory_safety_factor.json');ctx['context_sha256']=digest({k:v for k,v in ctx.items() if k!='context_sha256'})
  p=OUT/'contexts'/f'{cell["cell_id"]}.json';p.parent.mkdir(exist_ok=True)
  if p.exists():assert read(p)==ctx,'Existing runtime context differs; no regeneration'
  else:save_json(p,ctx)
  Path(str(p)+'.sha256').write_text(file_hash(p)+'\n');contexts[cell['cell_id']]=ref(p)
 # Full audit of the exact 54 potential repetition savings marked in 8F-1B.
 frozen_cells={c['cell_id']:c for c in read(F/'campaign_cells.json')['cells']};settings=protocol['measurement_settings'];rows=[]
 for group in read(F/'reuse_opportunities.json')['known_execution_signature_groups']:
  donor=group['members'][0]
  for member in group['members'][1:]:
   dc=frozen_cells[donor['cell_id']];tc=frozen_cells[member['cell_id']];dp=dc['preflight'][donor['method']];tp=tc['preflight'][member['method']]
   conditions={'exact_workload_lambda':dc['lambda_rps']==tc['lambda_rps'],'exact_memory_budget_semantics':dc['memory_budget_mib']==tc['memory_budget_mib'],'exact_deployment_vector':dp['deployment']==tp['deployment'],'exact_routing_identity':dp['canonical_routing_rps']==tp['canonical_routing_rps'],'exact_model_revisions':True,'exact_runtime_environment':True,'exact_input_subset':read(Path(contexts[dc['cell_id']]['path']))['input_pool']==read(Path(contexts[tc['cell_id']]['path']))['input_pool'],'exact_arrival_protocol':True,'exact_warmup':True,'exact_measurement_duration':True,'exact_metric_definition':True,'exact_resource_limit_context':contexts[dc['cell_id']]['sha256']==contexts[tc['cell_id']]['sha256']}
   for seed in protocol['search_seeds']:
    rows.append({'signature_sha256':group['signature_sha256'],'donor_cell':dc['cell_id'],'donor_method':donor['method'],'target_cell':tc['cell_id'],'target_method':member['method'],'random_seed':seed,'status':'REUSE_ELIGIBLE_FUTURE_DONOR' if all(conditions.values()) else 'REUSE_FORBIDDEN','failed_conditions':';'.join(k for k,v in conditions.items() if not v),**conditions,'physical_measurement_exists':False,'metric_values_inspected':False})
 assert len(rows)==54
 csvwrite('reuse_manifest.csv',rows);write('reuse_audit.json',{'completed_before_first_GPU':True,'potential_repetition_reuses_audited':54,'eligible_future_donor_references':sum(r['status']=='REUSE_ELIGIBLE_FUTURE_DONOR' for r in rows),'forbidden':sum(r['status']=='REUSE_FORBIDDEN' for r in rows),'historical_reuse':0,'all_conditions_required':True,'resource_budget_is_identity_component':True,'only_same_cell_search_sharing_eligible':True,'phase_contract':'search may share between independently committed v2/RC-SG states; no search/confirmation sharing, no seed/final confirmation sharing; fixed confirmations separate','same_exact_input_subset_rule':'Same frozen sample pool, active tasks, repetition seed and unchanged SamplePool/arrival protocol; actual raw identity and hashes validated before any future reuse','GPU_runs_before_audit':0})
 manifest={'module':'8F-2','prepared_at_utc':utc(),'frozen_inputs':{'8F1A':ref(A/'artifact_hashes.json'),'8F1B':ref(F/'artifact_hashes.json'),'grid':ref(F/'campaign_cells.json'),'hypotheses':ref(F/'success_hypotheses.json'),'RC_SG_spec':ref(F/'resource_constrained_sg_spec.json'),'v2_spec':ref(ROOT/'results/module8e1_method_revision/method_v2_spec.json')},'execution_sources':{str(p.relative_to(ROOT)):ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'contexts':contexts,'GPU':gpu,'environment':env,'reuse_audit':ref(OUT/'reuse_audit.json'),'resource_limit_binding':'Only execution context capacity changes to frozen B; physical GPU identity remains fixed; model memory table, evidence bounds, handlers, transport, routes and timing unchanged. Admission requires M_safe<=B; existing SafetyMonitor enforces actual device peak<=B.','methods':['static_v2','RC_SG','Generalist_centric'],'logical_cap':84,'physical_state_cap':84,'physical_attempt_cap':324,'global_caps':{'logical':96,'physical':96,'timed':384},'retries':0,'structural_classification':'Frozen 8F1B predicates: Generalist-only NG>0 NS=0; Hybrid NG>0 NS>0; Specialist-heavy NS>NG (overlaps Hybrid). Display precedence Specialist-heavy, Hybrid, Generalist-only->Generalist-heavy; preserve all predicate flags and do not invent a new threshold.','hypotheses_modified':False,'runtime_protocol':protocol['measurement_settings'],'search_seeds':protocol['search_seeds'],'confirmation_seeds':protocol['confirmation_seeds']}
 if (OUT/'campaign_manifest.json').exists():raise RuntimeError('Campaign already prepared; do not overwrite/reschedule')
 manifest['user_authorized_GPU_UUID_exception']=ref(OUT/'hardware_identity_exception.json') if authorization else None
 manifest['historical_runtime_reuse_on_original_UUID']='FORBIDDEN; user exception permits new campaign on current UUID, not historical cross-device reuse'
 write('campaign_manifest.json',manifest);write('campaign_status.json',{'status':'PREPARED','GPU_runs':0});print(json.dumps({'event':'PREPARED','reuse_audit':read(OUT/'reuse_audit.json')}),flush=True)
if __name__=='__main__':prepare()
