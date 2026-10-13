"""Freeze optimizer-selected cross-task plans before any GPU observations."""
import copy
from .common import *
from replica_scaling.telemetry import snapshot

def main():
 assert not (OUT/'frozen_task_manifests.json').exists()
 f=read(OUT/'cpu_protocol_freeze.json');verify(f['implementation']+[f['full_pool'],f['controlled_pool'],f['pool_definition']]);verify(read(OUT/'protected_before.json')['files'])
 c=read(OUT/'cpu_search_results.json');data=read(f['full_pool']['path']);base=read(ROOT/'results/module16_fullpool_gpu/frozen_gpu_manifest.json');plan=copy.deepcopy(base['plan']);plan['models']=data['models'];plan['seeds']=f['GPU_protocol']['seeds'];plan['warmup_seconds']=10.;plan['measurement_seconds']=120.;plan['capacity_scope']='Module18 frozen serving validation only; no new capacity profiling or capacity-table updates.'
 slots=[];tasks=[]
 for task in f['task_priority']:
  if task not in c['selected']:
   tasks.append({'task':task,'CPU_status':'NOT_SUPPORTED','slots':[]});continue
  w=c['selected'][task];points=[]
  for p in w['points']:
   fit=p['fit']['winner'];assert p['check']['passed'];routing={k:{path.split('::')[1]:v for path,v in fit['routing'].items() if path.split('::')[0]==k and v>0} for k in p['rates']}
   assert all(m in plan['registry'] for m in fit['deployment'])
   for rep,seed in enumerate(f['GPU_protocol']['seeds'],1):
    slots.append({'run_id':f'{task}_{p["stage"].lower()}_rep{rep}','target_task':task,'pool':w['pool'],'stage':p['stage'],'repetition':rep,'seed':seed,'deployment':fit['deployment'],'lambda_rps':p['rates'],'routing':routing,'measurement_seconds':120.,'safe_memory_budget_mib':f['budget_mib'],'CPU_safe_memory_mib':fit['safe_memory_mib'],'CPU_max_rho':fit['max_rho'],'CPU_rho':fit['rho'],'CPU_J_seconds':fit['J_seconds'],'CPU_phase_check':p['check'],'CPU_strict_J_certificate_gap_seconds':fit['J_gap_seconds']})
   points.append({'stage':p['stage'],'all_task_QPS':p['rates'],'model_identities_and_replicas':fit['deployment'],'routing_QPS':routing,'routing_fractions':{k:{m:v/p['rates'][k] for m,v in row.items()} for k,row in routing.items()},'safe_memory_mib':fit['safe_memory_mib'],'rho':fit['rho'],'predicted_phase_and_margin':p['check'],'complete_pool_competition':p['fit'],'original_V3_independent_CPU_baseline':p['original_V3_CPU_baseline']})
  tasks.append({'task':task,'CPU_status':'THEORY_WITNESS','pool':w['pool'],'point_source':w['point_source'],'generator':w.get('generator'),'points':points,'slots':[s['run_id'] for s in slots if s['target_task']==task]})
 implementation=[ref(p) for p in sorted((ROOT/'module18').glob('*.py'))]
 reused=base['reused_implementation']+base['implementation']+[ref(ROOT/p) for p in ['module15/solver.py','module11b/convex_audit.py','replica_scaling/runtime.py','replica_scaling/telemetry.py','joint_deploy/measurement/metrics.py','joint_deploy/measurement/load_generator.py']]
 reused=list({r['path']:r for r in reused}.values());verify(reused)
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128 and gpu['uuid']==plan['GPU']['uuid']
 save('frozen_task_manifests.json',{'status':'FROZEN_BEFORE_GPU','at_utc':now(),'CPU':ref(OUT/'cpu_search_results.json'),'CPU_protocol':ref(OUT/'cpu_protocol_freeze.json'),'catalog':f['full_pool'],'controlled_pool':f['controlled_pool'],'pool_admission':f['pool_definition'],'quality_audit':ref(OUT/'quality_and_empirical_path_audit.json'),'sample_pool':plan['sample_pool'],'plan':plan,'tasks':tasks,'slots':slots,'GPU_protocol':f['GPU_protocol'],'maximum_sessions':len(slots),'not_full_pool_witness_when_controlled':True,'no_new_models_no_profiling_no_point_movement':True,'memory_rule':base['memory_rule'],'utilization_semantics':base['utilization_semantics'],'stability_thresholds_source':ref(ROOT/'performance_model/local_calibration/stability_v2.py'),'implementation':implementation,'reused_implementation':reused})
 print('FROZEN',len(slots),'slots',[(t['task'],t.get('pool')) for t in tasks],flush=True)
if __name__=='__main__':main()
