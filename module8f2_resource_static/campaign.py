"""Resource campaign adapter around frozen v2 and frozen real serving harness."""
import fcntl,json,os,traceback
from pathlib import Path
from deployment_core import Workload,Hardware
from deployment_search.runtime import RuntimeEvaluator
from deployment_search.runtime_cache import save_json,file_hash,digest
from deployment_search.result import jsonable
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator,PilotExecutionBlocked,summarize,aggregate
from joint_deploy.measurement.schema import MeasurementConfig
from method_v2 import MethodV2,frozen_config
from .common import ROOT,A,F,OUT,PROTOCOL,read,write,ref,inputs,utc
from .bank import MeasurementBank

def verify_execution():
 manifest=read(OUT/'execution_freeze.json')
 for r in manifest['protected_references']:
  if file_hash(r['path'])!=r['sha256']:raise PilotExecutionBlocked('Frozen hash mismatch: '+r['path'])
 for r in manifest['execution_sources'].values():
  if file_hash(r['path'])!=r['sha256']:raise PilotExecutionBlocked('Execution adapter hash mismatch: '+r['path'])

def active(bank,cell,method,phase,role,index):
 bank.active={'method':method,'workload':cell['cell_id'],'workload_sha256':cell['workload_hash'],'logical_slot':index,'phase':phase,'role':role,'protocol_sha256':file_hash(PROTOCOL),'sharing_policy_sha256':file_hash(F/'gpu_budget_estimate.json')}
 bank.schedule_log.append(dict(bank.active,status='STARTED',at_utc=utc()))
 write('campaign_status.json',{'status':'RUNNING','cell_id':cell['cell_id'],'method':method,'phase':phase,'role':role,'accounting':bank.accounting(),'updated_at_utc':utc()});bank.write()

def eval_search(bank,cell,method,evaluator,x,w,r,index,role):
 active(bank,cell,method,'search',role,index)
 identity=evaluator.identity(x,w,r)
 record={'method':method,'workload':cell['cell_id'],'phase':'search','role':role,'logical_slot':index,'deployment':identity['deployment'],'routing_hash':digest(identity['routing']),'identity_sha256':digest(identity),'state_identity':identity,'budget_charge':1,'status':'selected'}
 bank.evaluations.append(record);assert bank.accounting()['logical_search_evaluations']<=84;bank.write()
 runtime=evaluator.evaluate(x,w,r);record.update(status='completed',raw_result_paths=runtime.raw_result_paths,stability=runtime.stability_state_v2);bank.schedule_log[-1]['status']='COMPLETED';bank.write();return runtime

class SearchAdapter(RuntimeEvaluator):
 def __init__(self,bank,cell,evaluator,catalog):self.bank,self.cell,self.evaluator,self.catalog,self.count=bank,cell,evaluator,catalog,0
 def evaluate(self,deployment,workload,routing):
  index=self.count;self.count+=1;assert self.count<=6
  if index==0:
   frozen=self.cell['preflight']['static_v2'];assert deployment=={m:frozen['deployment'][m] for m in self.catalog};assert jsonable(routing)==frozen['canonical_routing_rps']
   save_json(OUT/self.cell['cell_id']/'seed_reproducibility.json',{'deployment':deployment,'routing':jsonable(routing),'matches_frozen_exactly':True,'verified_before_first_cell_GPU_run':True})
  else:
   trace=read(OUT/self.cell['cell_id']/'static_v2'/'search_trace.json');tr=trace['iterations'][-1];candidate=next(c for c in tr['candidates'] if c['selected_top_b'] and c['mutation']['deployment']==deployment)
   assert (OUT/self.cell['cell_id']/'static_v2'/f'round_{tr["iteration"]+1}_commitment.json').exists()
   save_json(OUT/self.cell['cell_id']/'static_v2'/f'selected_state_{index}.json',{'iteration':tr['iteration']+1,'rank':candidate['rank'],'deployment':deployment,'routing':jsonable(routing),'selected_before_measurement':True})
  return eval_search(self.bank,self.cell,'static_v2',self.evaluator,deployment,workload,routing,index,'cold_seed' if index==0 else 'candidate')

def checkpoint(cell,result):
 snap=result.to_dict();directory=OUT/cell['cell_id']/'static_v2';directory.mkdir(parents=True,exist_ok=True)
 save_json(directory/'search_trace.json',snap)
 factor=read(A/'memory_safety_factor.json')['safety_factor'];audit=[]
 for tr in snap['iterations']:
  save_json(directory/f'iteration_{tr["iteration"]+1}.json',tr)
  if tr['selected_top_b']:
   p=directory/f'round_{tr["iteration"]+1}_commitment.json'
   if not p.exists():save_json(p,{'iteration':tr['iteration']+1,'ranking':tr['ranking'],'selected_top_b':tr['selected_top_b'],'candidates':tr['candidates'],'at_utc':utc(),'before_first_round_measurement':True})
  for c in tr['candidates']:
   safe=factor*c['memory_mib'];audit.append({'iteration':tr['iteration']+1,'deployment':c['mutation']['deployment'],'M_plan_mib':c['memory_mib'],'M_safe_mib':safe,'budget_mib':cell['memory_budget_mib'],'resource_status':'RESOURCE_BUDGET_INFEASIBLE' if safe>cell['memory_budget_mib'] else 'MEMORY_FEASIBLE','engine_status':c['status'],'reason':c['reason']})
 save_json(directory/'candidate_resource_audit.json',{'rows':audit,'method_trace_unmodified':True})

def confirm(bank,cell,method,evaluator,x,w,r,role,repetition,seed):
 active(bank,cell,method,'confirmation',method+'_'+role,repetition)
 identity=evaluator.identity(x,w,r);config=MeasurementConfig(identity['deployment'],identity['workload'],identity['routing'],random_seed=seed,**read(PROTOCOL)['measurement_settings']).as_dict()
 directory=OUT/cell['cell_id']/method/'confirmation';directory.mkdir(parents=True,exist_ok=True);prefix=directory/f'{role}_rep_{repetition}'
 raw_path=Path(str(prefix)+'.json');assert not raw_path.exists(),'Existing confirmation; no retry'
 config_path=Path(str(prefix)+'.config.json');save_json(config_path,config)
 outcome=bank.raw_run(config,raw_path,Path(str(prefix)+'.log'),config_path,OUT/'contexts'/f'{cell["cell_id"]}.json')
 rep=summarize(read(raw_path),w,bank.catalog,outcome,raw_path);rep.update(repetition=repetition,random_seed=seed,post_search_confirmation=True)
 save_json(Path(str(prefix)+'.summary.json'),rep)
 if not rep['valid_runtime_repetition']:raise PilotExecutionBlocked('Invalid confirmation; zero retries')
 bank.schedule_log[-1]['status']='COMPLETED';bank.write();return rep

def run_cell(cell,bank,catalog,roles,cal):
 directory=OUT/cell['cell_id'];directory.mkdir(exist_ok=True)
 w=Workload(cell['lambda_rps']);bank.catalog,bank.workload=catalog,w
 protocol=read(PROTOCOL);ctx=OUT/'contexts'/f'{cell["cell_id"]}.json'
 evaluators={m:ReplicaRuntimeEvaluator(directory/m/'search_cache',catalog,ctx,protocol['measurement_settings'],seeds=protocol['search_seeds'],max_unique=6 if m=='static_v2' else 1,run=bank.raw_run) for m in ['static_v2','RC_SG']}
 xrc={m:cell['preflight']['RC_SG']['deployment'][m] for m in catalog};rrc={tuple(k.split('::')):v for k,v in cell['preflight']['RC_SG']['canonical_routing_rps'].items()}
 # RC-SG's construction is committed before v2 receives any runtime value.
 save_json(directory/'RC_SG'/'pre_runtime_commitment.json',{'frozen_preflight':cell['preflight']['RC_SG'],'spec':ref(F/'resource_constrained_sg_spec.json'),'verified_exactly_in':ref(OUT/'planning_execution_verification.json'),'at_utc':utc(),'no_runtime_input':True})
 adapter=SearchAdapter(bank,cell,evaluators['static_v2'],catalog)
 engine=MethodV2(adapter,predictor=None,trace_callback=lambda r:checkpoint(cell,r))
 result=engine.search(w,catalog,Hardware(cell['planning_budget_input_mib'],'Frozen memory budget / safety factor'),frozen_config(),roles=roles,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
 checkpoint(cell,result);snap=result.to_dict();assert result.runtime is not None
 save_json(directory/'static_v2'/'search_complete.json',snap)
 save_json(directory/'post_search_freeze.json',{'v2_final':snap['deployment'],'v2_routing':snap['canonical']['routing'],'RC_SG_final':xrc,'confirmation_influences_search':False,'at_utc':utc()})
 rc_runtime=eval_search(bank,cell,'RC_SG',evaluators['RC_SG'],xrc,w,rrc,0,'fixed')
 save_json(directory/'RC_SG'/'search_result.json',{'deployment':xrc,'routing':jsonable(rrc),'runtime':rc_runtime})
 confirmations={'static_v2':{'seed':[],'final':[]},'RC_SG':{'fixed':[]}}
 seed_x=snap['seed']['deployment'];seed_r={tuple(k.split('::')):v for k,v in cell['preflight']['static_v2']['canonical_routing_rps'].items()};final_x=snap['deployment'];final_r={tuple(k.split('::')):v for k,v in snap['canonical']['routing'].items()}
 for role,repetition in protocol['confirmation_order']:
  x,r=(seed_x,seed_r) if role=='seed' else (final_x,final_r)
  rep=confirm(bank,cell,'static_v2',evaluators['static_v2'],x,w,r,role,repetition,protocol['confirmation_seeds'][repetition-1]);confirmations['static_v2'][role].append(rep);save_json(directory/'confirmation_progress.json',confirmations)
 for repetition,seed in enumerate(protocol['confirmation_seeds'],1):
  rep=confirm(bank,cell,'RC_SG',evaluators['RC_SG'],xrc,w,rrc,'fixed',repetition,seed);confirmations['RC_SG']['fixed'].append(rep);save_json(directory/'confirmation_progress.json',confirmations)
 for method in ['static_v2','RC_SG']:
  aggs={}
  for role,reps in confirmations[method].items():
   assert len(reps)==2 and len({str(Path(r['raw_result_path']).resolve()) for r in reps})==2
   counts=xrc if method=='RC_SG' else seed_x if role=='seed' else final_x
   path=directory/method/'confirmation'/f'{role}_aggregate.json';aggs[role]=aggregate(sorted(reps,key=lambda r:r['repetition']),'independent_confirmation',path,counts);save_json(path,aggs[role])
  if method=='static_v2':assert not ({str(Path(r['raw_result_path']).resolve()) for r in confirmations[method]['seed']} & {str(Path(r['raw_result_path']).resolve()) for r in confirmations[method]['final']})
  record={'method':method,'cell_id':cell['cell_id'],'status':'COMPLETE','deployment':final_x if method=='static_v2' else xrc,'routing':jsonable(final_r if method=='static_v2' else rrc),'confirmations':aggs,'search_runtime':result.runtime if method=='static_v2' else rc_runtime,'logical_search_evaluations':result.runtime_evaluation_count if method=='static_v2' else 1,'budget_mib':cell['memory_budget_mib'],'mixture':cell['mixture'],'intensity':cell['intensity_level']}
  save_json(directory/method/'result.json',record)
 save_json(directory/'status.json',{'status':'COMPLETE','completed_at_utc':utc()});print(json.dumps({'event':'cell_complete','cell_id':cell['cell_id'],'accounting':bank.accounting()}),flush=True)
 from .report import report
 report()

def freeze_execution():
 assert read(OUT/'campaign_status.json')['status']=='PREPARED';assert read(OUT/'planning_execution_verification.json')['status']=='PASS'
 verification=read(OUT/'hash_verification.json');assert all(r['match'] for r in verification['checks'])
 protected=[{'path':r['path'],'sha256':r['sha256']} for r in verification['checks']]
 protected += [ref(OUT/n) for n in ['campaign_manifest.json','reuse_audit.json','reuse_manifest.csv','hardware_identity_exception.json','planning_execution_verification.json']]
 protected += list(read(OUT/'campaign_manifest.json')['contexts'].values())
 for key in ['runtime_evaluator','runtime_harness','runtime_objective','runtime_objective_base','catalog_loader','runtime_context_template']:protected.append(read(PROTOCOL)[key])
 write('execution_freeze.json',{'frozen_before_first_GPU':True,'at_utc':utc(),'cell_order':[c['cell_id'] for c in read(F/'campaign_cells.json')['cells']],'per_cell_order':'v2 search; RC-SG independently committed fixed search; frozen v2 seed1/final1/final2/seed2 confirmations; RC-SG fixed1/fixed2 confirmations','execution_sources':{str(p.relative_to(ROOT)):ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'protected_references':protected,'GPU_UUID_exception':ref(OUT/'hardware_identity_exception.json'),'no_history_measurement_reuse':True,'retries':0})

def main():
 for key,value in read(PROTOCOL)['execution_environment_overrides'].items():os.environ[key]=value
 with (OUT/'.campaign.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze_execution();verify_execution()
  catalog,_,roles,cal=inputs();bank=MeasurementBank();bank.write()
  try:
   with forbid_enumeration() as guards:
    for cell in read(F/'campaign_cells.json')['cells']:run_cell(cell,bank,catalog,roles,cal)
    assert sum(g.call_count for g in guards)==0
   verify_execution();assert bank.accounting()['within_budget'];write('campaign_status.json',{'status':'COMPLETE','accounting':bank.accounting(),'completed_at_utc':utc(),'enumeration_calls':0})
  except BaseException as exc:
   write('campaign_status.json',{'status':'INCOMPLETE','error':repr(exc),'traceback':traceback.format_exc(),'accounting':bank.accounting(),'no_automatic_retry_or_budget_growth':True,'at_utc':utc()});bank.write()
   from .report import report
   report();raise
  from .report import report
  report()
if __name__=='__main__':main()
