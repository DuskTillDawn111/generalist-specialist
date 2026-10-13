"""Outcome-independent distinct-control selection and counterbalanced schedule."""
import copy
from .common import *
from replica_scaling.telemetry import snapshot

def equivalent(a,b):
 if a['catalog']['sha256']!=b['catalog']['sha256'] or a['rates']!=b['rates'] or a['budget_mib']!=b['budget_mib'] or a['deployment']!=b['deployment'] or a['seeds']!=b['seeds']:return False
 return max(abs(a['flat_routing'].get(k,0)-b['flat_routing'].get(k,0)) for k in set(a['flat_routing'])|set(b['flat_routing']))<=1e-9

def main():
 assert not (OUT/'frozen_benchmark_manifest.json').exists()
 inp=read(OUT/'input_protocol_freeze.json');verify(inp['implementation']);cpu=read(OUT/'cpu_results.json');configs=[];logical={};selection=set();comparisons=[];historical_only=[]
 def register(case,label,fit,legal):
  w=fit['winner'];body={'catalog':case['catalog'],'rates':case['rates'],'budget_mib':case['budget_mib'],'deployment':w['deployment'],'flat_routing':w['routing'],'seeds':case['seed_sequence']};match=next((c for c in configs if equivalent(body,c)),None)
  if not match:
   identity=case['case_id']+'_'+{'PROPOSED':'proposed','BASELINE_A_G_ONLY':'g_only','BASELINE_B_S_ORIENTED':'s_oriented','BASELINE_C_ORIGINAL_V3':'original_v3','FIXED_S_ONLY':'fixed_s_only'}[label];match={**body,'config_id':identity,'owner_case':case['case_id'],'target_task':case['target_task'],'stage':case['stage'],'aliases':[],'operationally_legal':legal,'CPU_safe_memory_mib':w['safe_memory_mib'],'CPU_rho':w['rho'],'CPU_max_rho':w['max_rho'],'CPU_J_seconds':w['J_seconds'],'routing':{k:{p.split('::')[1]:q for p,q in w['routing'].items() if p.split('::')[0]==k and q>0} for k in case['rates']}};configs.append(match)
  assert match['operationally_legal']==legal
  match['aliases'].append({'case_id':case['case_id'],'label':label,'independent_CPU_solve_seconds':fit['solve_seconds'],'maximum_route_roundoff_difference_QPS':max(abs(match['flat_routing'].get(k,0)-body['flat_routing'].get(k,0)) for k in set(match['flat_routing'])|set(body['flat_routing']))});logical[case['case_id'],label]=match['config_id'];return match['config_id']
 for r in cpu['results']:
  c=r['case'];P=register(c,'PROPOSED',r['strategies']['PROPOSED']['fit'],True)
  for name,st in r['strategies'].items():
   if name=='PROPOSED':continue
   if st['audit']['passed']:
    B=register(c,name,st['fit'],True);comparisons.append({'case_id':c['case_id'],'kind':'DEPLOYMENT_STRATEGY','control_label':name,'proposed_id':P,'control_id':B,'configuration_equivalent':P==B,'control_operationally_legal':True})
    if B!=P:selection.update([P,B])
   else:comparisons.append({'case_id':c['case_id'],'kind':'DEPLOYMENT_STRATEGY','control_label':name,'proposed_id':P,'control_id':None,'CPU_control_status':st['fit']['status'],'control_audit':st['audit'],'original_V3_certificate':st['fit'].get('original_operational_certificate'),'no_GPU_reason':'No legal independently solved control, mathematical infeasibility distinguished from returned-plan gate violation or numerical failure.'})
  if c['stage']=='HIGH':
   ab=r['routing_ablation_S_only'];B=register(c,'FIXED_S_ONLY',ab,ab['operationally_legal']);selection.update([P,B]);comparisons.append({'case_id':c['case_id'],'kind':'FIXED_RESIDENCY_ROUTING_ABLATION','control_label':'FIXED_S_ONLY','proposed_id':P,'control_id':B,'configuration_equivalent':False,'control_operationally_legal':ab['operationally_legal'],'control_theoretical_status':ab['status'],'same_residency':True,'non_target_routes_unchanged':True,'diagnostic_only':not ab['operationally_legal']})
 case_by_id={r['case']['case_id']:r['case'] for r in cpu['results']};blocks=[];scheduled=set()
 for r in cpu['results']:
  case=r['case'];key='common18_low' if case['source_module']==18 and case['stage']=='LOW' else case['case_id']
  groupcases=[z['case'] for z in cpu['results'] if z['case']['source_module']==18 and z['case']['stage']=='LOW'] if key=='common18_low' else [case]
  ids=[c['config_id'] for c in configs if c['config_id'] in selection and any(a['case_id'] in [g['case_id'] for g in groupcases] for a in c['aliases']) and c['config_id'] not in scheduled]
  if not ids:continue
  blocks.append({'block_id':key,'case_ids':[c['case_id'] for c in groupcases],'config_ids':ids});scheduled.update(ids)
 assert scheduled==selection
 slots=[]
 for block in blocks:
  ids=block['config_ids']
  for rep in [1,2,3]:
   order=ids if rep==1 else list(reversed(ids)) if rep==2 else ids[1:]+ids[:1]
   for ident in order:
    c=next(c for c in configs if c['config_id']==ident);case=case_by_id[c['owner_case']];slots.append({'run_id':ident+'_rep'+str(rep),'config_id':ident,'block_id':block['block_id'],'stage':c['stage'],'target_task':c['target_task'],'repetition':rep,'seed':c['seeds'][rep-1],'deployment':c['deployment'],'lambda_rps':c['rates'],'routing':c['routing'],'measurement_seconds':120.,'safe_memory_budget_mib':c['budget_mib'],'CPU_safe_memory_mib':c['CPU_safe_memory_mib'],'CPU_rho':c['CPU_rho'],'CPU_max_rho':c['CPU_max_rho'],'CPU_J_seconds':c['CPU_J_seconds'],'theoretically_legal':c['operationally_legal'],'configuration_role':'LEGAL_STRATEGY' if c['operationally_legal'] else 'NON_BASELINE_DIAGNOSTIC','owner_case':case['case_id']})
 for c in configs:
  c['GPU_selection']='FRESH3REPS' if c['config_id'] in selection else 'HISTORICAL3REPS_DESCRIPTIVE_ONLY';c['historical_source_case']=case_by_id[c['owner_case']] if c['config_id'] not in selection else None
  if c['config_id'] not in selection:historical_only.append(c['config_id'])
 implementations=[ref(p) for p in sorted((ROOT/'module19').glob('*.py'))];reused=read(ROOT/'results/module18_cross_task/frozen_task_manifests.json')['reused_implementation']+read(ROOT/'results/module15_three_phase/gpu_validation_freeze.json')['implementation']+[ref(ROOT/p) for p in ['module11b/measure.py','module11b/telemetry.py','module16/gpu.py','replica_scaling/runtime.py','replica_scaling/telemetry.py','joint_deploy/measurement/load_generator.py','joint_deploy/measurement/metrics.py']];reused=list({r['path']:r for r in reused}.values());verify(reused)
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
 for case in case_by_id.values():assert gpu['uuid']==case['historical_plan']['GPU']['uuid'] and gpu['total_mib']==case['historical_plan']['GPU']['total_mib']
 save('frozen_benchmark_manifest.json',{'status':'FROZEN_BEFORE_GPU','at_utc':now(),'input_protocol':ref(OUT/'input_protocol_freeze.json'),'CPU':ref(OUT/'cpu_results.json'),'cases':list(case_by_id.values()),'configurations':configs,'comparisons':comparisons,'blocks':blocks,'slots':slots,'maximum_new_sessions':len(slots),'new_configurations':len(selection),'historical_descriptive_only_configs':historical_only,'same_seed_counterbalancing':'rep1 listed order, rep2 reverse, rep3 rotate one;3 fresh independent processes perselectedphysicalconfiguration, exact same seeds asitsownsourceModule','deduplication':'Exact residency/workload/catalog/budget/seeds; routefloat differences<=1e-9QPS (<1.2e-7expectedrequests/120s). Equivalence is reported as samephysicalconfig, not statisticalindependent performancewin. COMMON18LOW shared onlyamongidenticalsix-taskcases; ASR module15 nevermerged.','historical_LID_worker_warning_preserved':ref(ROOT/'results/module18_cross_task/speech_colocation_review.json'),'diagnostic_configs_not_legal_baselines':[c['config_id'] for c in configs if not c['operationally_legal']],'stopping':'Stop remaining reps of aconfig after any systemORindividualworker unstable/sustainedbacklog, invalidsession, memoryexceeded. Keep recoverable observations, stillcollect3unlesshardfailure. Continue otherfrozencontrols onlyifGPUreleased. Never retry ormove point.','implementation':implementations,'reused_implementation':reused,'all_history_protected':ref(OUT/'protected_before.json'),'no_new_models_capacity_profiling_GlobalPenalty_V3Changes_or_Module20':True});print('FROZEN',len(selection),'freshconfigs',len(slots),'maximumruns','hist-only',historical_only,flush=True)
if __name__=='__main__':main()
