"""Only requested checkpoints; preserve historical files and all planning inputs."""
import copy
from .common import *
from replica_scaling.telemetry import snapshot

def main():
 assert not (OUT/'replica_measurement_manifest.json').exists()
 historical=read(OLD/'protected_before.json')['files']+read(OLD/'module14c_artifact_manifest.json')['files']+[ref(OLD/'module14c_artifact_manifest.json')]
 refs=list({r['path']:r for r in historical}.values());verify(refs)
 protected=[ref(r['path']) for r in refs]
 save('protected_before.json',{'at_utc':now(),'files':protected,'verified':True})
 h=read(OLD/'regime_h.json');dec=read(OLD/'replica_profiling_required.json')['decisions']
 required=[d for d in dec if d['requires_new_measurement']]
 assert len(required)==6 and sum(len(d['tasks']) for d in required)==7
 # Complete recursive reference integrity, including old replica raw repetitions.
 reusable=[]
 def walk(v):
  if isinstance(v,dict):
   if 'path' in v and 'sha256' in v and Path(v['path']).is_file():reusable.append(v)
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 for m in ['whisper_turbo','roberta_sst2','convnext_tiny']:
  walk(h['model_evidence'][m])
 walk(h['source_hashes']);verify(reusable)
 save('existing_replica_integrity.json',{'at_utc':now(),'passed':True,'files':list({r['path']:r for r in reusable}.values()),'reused_models':['whisper_turbo','roberta_sst2','convnext_tiny'],'no_remeasure':True})
 baseplan=read(ROOT/'results/module11_replica_and_cost/measurement_plan.json')
 protocol=read(ROOT/'results/module11_replica_and_cost/closed_loop_protocol.json')
 verify(protocol['semantics_source_refs'])
 plan=copy.deepcopy(baseplan);plan.update(at_utc=now(),GPU=snapshot(),models=h['models'],registry={})
 plan['environment_overrides'].update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 paths=[]
 for d in required:
  m=d['planning_id'];model=d['model_id'];script=OUT/'launchers'/('worker_'+str(len(plan['registry']))+'.py')
  if model.startswith('baudm/'):
   python=ROOT/'envs/module14b_str/bin/python'
   code=f"import runpy,sys\nsys.argv=[{str(ROOT/'module14b/worker.py')!r},'--model-id',{model!r}]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n"
  else:
   python=ROOT/'envs/module13b_transformers/bin/python'
   argv=[str(ROOT/'module13b/worker.py'),'--model-id',model,'--task',d['tasks'][0],'--directory',str(ROOT/'models/module13b'/model.replace('/','--'))]
   if model=='openai/whisper-medium':
    # Dispatch the identical two native handler branches through one checkpoint.
    # No inference/preprocessing/timer/concurrency modification.
    argv[4]='asr'
    code="import ast,sys\nfrom pathlib import Path\n"
    code+=f"sys.argv={argv!r}\np=Path(sys.argv[0]);tree=ast.parse(p.read_text())\n"
    code+="""class Dispatch(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name!='handler':return node
  for child in ast.walk(node):
   if isinstance(child,ast.Attribute) and isinstance(child.value,ast.Name) and child.value.id=='args' and child.attr=='task':
    child.value.id='request_task_namespace'
  node.body.insert(0,ast.parse("request_task_namespace=type('Task',(),{'task':request['task_type']})()").body[0])
  return node
for node in tree.body:
 if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':
  node.value.args[2]=ast.parse("dict(metadata,tasks=['asr','lid'])",mode='eval').body
namespace={'__file__':str(p),'__name__':'__main__'}
tree=Dispatch().visit(tree);ast.fix_missing_locations(tree)
exec(compile(tree,str(p),'exec'),namespace)
"""
   else:code=f"import runpy,sys\nsys.argv={argv!r}\nrunpy.run_path(sys.argv[0],run_name='__main__')\n"
  script.write_text(code);plan['registry'][m]={'python':str(python),'script':str(script)}
  for task in d['tasks']:paths.append({'planning_id':m,'model_id':model,'task_path':task,'revision':d['revision'],'g1':1.0,'M1_idle_mib':h['memory_table_mib'][m]['1'],'s0_ms':h['models'][m]['service_time_ms'][task]})
 assert not plan['GPU']['processes'] and plan['GPU']['used_mib']<=128
 slots=[]
 for path_index,p in enumerate(paths):
  for rep,seed in enumerate(protocol['seeds'],1):
   for x in ([1,2] if rep%2 else [2,1]):
    m=p['planning_id'];task=p['task_path'];low=.45*1000/p['s0_ms'];rid=f'path{path_index}_{task}_x{x}_rep{rep}'
    slot={'run_id':rid,'kind':'closed_loop_profile','model':m,'primary_task':task,'task_path':task,'required_x':2,'repetition':rep,'seed':seed,'deployment':{m:x},'measurement_seconds':30.,'lambda_rps':{task:low},'routing':{task:{m:low}},'low_lambda':{task:low},'low_routing':{task:{m:low}}}
    save('slots/'+rid+'.json',slot);slots.append(slot)
 save('measurement_plan.json',plan);save('closed_loop_protocol.json',protocol)
 save('resident_set_manifest.json',{'states':slots,'scope':'only requested models; each count is independently launched workers'})
 save('closed_loop_source_revision_v2.json',{'source':ref(ROOT/'module11b/closed_loop.py'),'reused_window_transport_monitor':True})
 implementation=[ref(p) for p in sorted((ROOT/'module14d').glob('*.py'))]+[ref(p) for p in sorted((OUT/'launchers').glob('*.py'))]
 save('input_freeze.json',{'protected_references':protected,'measurement_plan':ref(OUT/'measurement_plan.json'),'resident_set_manifest':ref(OUT/'resident_set_manifest.json'),'samples':plan['sample_pool'],'implementation':implementation})
 save('replica_measurement_manifest.json',{'at_utc':now(),'status':'FROZEN_BEFORE_GPU','requested_paths':paths,'new_x2_states':21,'paired_x1_references':21,'frozen_g1_M1_reused':True,'x1_reference_reason':'11B paired empirical throughput ratios require fresh x1 capacity denominators; historical g1/M1 are never overwritten','existing_replica_evidence':ref(OUT/'existing_replica_integrity.json'),'protocol':ref(OUT/'closed_loop_protocol.json'),'input_freeze':ref(OUT/'input_freeze.json'),'slots':slots,'no_x_ge3':True,'family_capacity_rule':'For shared Whisper-medium, retain both empirical path g2; frozen family-level scalar calibration uses minimum measured path gain as conservative aggregate transfer. This does not certify an unmeasured mixed-task capacity.','memory_rule':'M2=mean postwarm idle for each path; shared checkpoint family M2=max empirical path M2; preserve original M1','CPU_contexts':read(OLD/'screen_protocol_freeze.json')['context_count'],'immutable_E_and_H_pool':True})
 print('FROZEN',len(paths),'paths',len(slots),'sessions',len(protected),'protected files',flush=True)
if __name__=='__main__':main()
