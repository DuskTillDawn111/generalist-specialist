"""Full-pool revalidation and immutable pre-GPU experiment freeze."""
import copy
from pathlib import Path
from .common import *
from .solver import CertifiedResourceSolver
from module15.baseline import run as original_v3
from replica_scaling.telemetry import snapshot
BUDGET=27980.1894
BACKGROUND={'text':1.1664,'image':.3729,'ocr':.5501,'lid':12.,'visual_qa':.3603}
def main():
 assert not (OUT/'frozen_gpu_manifest.json').exists()
 history=read(ROOT/'results/module15_three_phase/protected_before.json')['files'];a=read(ROOT/'results/module15_three_phase/artifact_manifest.json');history+=a['artifacts']+a['implementation']+[ref(ROOT/'results/module15_three_phase/artifact_manifest.json')]
 history=list({r['path']:r for r in history}.values());verify(history);save('protected_before.json',{'files':history,'at_utc':now()})
 source=ROOT/'results/module14f_memory_ocr_x3/heavy_empirical_x3_reconciled.json';data=read(source);s=CertifiedResourceSolver(data)
 assert {m for m in s.c if s.roles[m]=='generalist'}=={'qwen','phi4mm','minicpm'}
 points=[];ok=True
 for stage,asr in [('LOW',.12),('MID',1.),('HIGH',3.8)]:
  rates={**BACKGROUND,'asr':asr};fit=s.solve(rates,BUDGET,competition=3);w=fit['winner'];checks={};condition=False
  if w:
   route=w['routing'];G=sum(v for p,v in route.items() if p.startswith('asr::') and s.roles[p.split('::')[1]]=='generalist');S=asr-G
   deployed_asr_S=[m for m in w['deployment'] if s.roles[m]=='specialist' and 'asr' in s.c[m].supported_tasks]
   checks={'quality_and_task_path_admission':all(k in s.c[m].supported_tasks for p,v in route.items() if v>0 for k,m in [p.split('::')]),'six_task_coverage':all(any(p.startswith(k+'::') and v>0 for p,v in route.items()) for k in rates),'memory_feasible':w['safe_memory_mib']<=BUDGET,'measured_replica_counts_only':all(str(n) in data['memory_table_mib'][m] and str(n) in data['capacity_factors'][m] for m,n in w['deployment'].items()),'routing_conservation':all(abs(sum(v for p,v in route.items() if p.startswith(k+'::'))-q)<=1e-10 for k,q in rates.items()),'rho_le_0_8':w['J_operational_certificate']['passed'],'strict_J_optimality':w['J_accepted'] and w['J_gap_seconds']<=1e-8,'memory_global_lower_bound':abs(w['safe_memory_mib']-fit['minimum_memory_lower_bound_mib'])<=1e-6,'all_minimum_resource_competitors_evaluated':all(z['J_accepted'] for z in fit['tied_minimum_states'])}
   condition=(not deployed_asr_S and G>0 and S<1e-10) if stage=='LOW' else (bool(deployed_asr_S) and S/asr>=.8) if stage=='MID' else (bool(deployed_asr_S) and G>0 and S>0)
   fit['status']='MINIMUM_MEASURED_SAFE_MEMORY_THEN_STRICT_SAME_J_CERTIFIED';fit['routing_policy']='Independent resource policy using unchanged J with original tolerance; original Module9/V3 numerical failures preserved, not replaced or relabeled.'
  ok=ok and bool(w) and all(checks.values()) and condition
  points.append({'stage':stage,'rates':rates,'full_pool_result':fit,'checks':checks,'three_phase_condition':condition,'original_V3_CPU_baseline':original_v3(rates,BUDGET,s)})
  print('CPU',stage,checks,condition,flush=True)
 result={'at_utc':now(),'source_catalog':ref(source),'budget_mib':BUDGET,'background':BACKGROUND,'Generalists':{m:data['models'][m] for m in s.c if s.roles[m]=='generalist'},'full_pool_size':len(s.c),'all_admitted_measured_paths_compete':True,'points':points,'three_phase_reproduced':ok,'model_identities_forced':False,'original_V3_unchanged':True,'numerical_note':'HIGH original Module9 fit rejects strict convex gaps 2.29e-7 / 3.88e-7 s. Independent same-J CPU active-face Newton plus original 1e-8 global LP certificate resolves without changing source, tolerance, objective or constraints. Original V3 is separately recorded.','historical_six_task_witness_file_found':False,'comparison_reference':'User-provided six-task points and HIGH ~5.16% prediction; no historical CPU route asserted as an authenticated artifact.'};save('cpu_witness_revalidation.json',result)
 if not ok:
  save('frozen_gpu_manifest.json',{'status':'NOT_FROZEN_CPU_REVALIDATION_FAILED','CPU':ref(OUT/'cpu_witness_revalidation.json'),'slots':[]});return
 base=read(ROOT/'results/module11_replica_and_cost/measurement_plan.json');new=read(ROOT/'results/module14_heavy_replica/measurement_plan.json');plan=copy.deepcopy(base);plan['models']=data['models'];plan['registry']={**base['registry'],**new['registry']};plan['warmup_seconds']=10.
 # Launch the already measured XLNet native handler. Only READY task metadata is
 # added to match serving protocol; weights, tokenizer, inference/timers unchanged.
 model='textattack/xlnet-base-cased-SST-2';m=next(m for m in s.c if m.startswith(model+'@'))
 launch=OUT/'launchers/xlnet.py';launch.parent.mkdir(parents=True,exist_ok=True)
 argv=[str(ROOT/'module13b/worker.py'),'--model-id',model,'--task','text','--directory',str(ROOT/'models/module13b'/model.replace('/','--'))]
 launch.write_text("import ast,sys\nfrom pathlib import Path\n"+f"sys.argv={argv!r}\np=Path(sys.argv[0]);tree=ast.parse(p.read_text())\n"+"for node in tree.body:\n if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='run_worker':\n  node.value.args[2]=ast.parse(\"dict(metadata,tasks=['text'])\",mode='eval').body\nast.fix_missing_locations(tree)\nexec(compile(tree,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})\n")
 plan['registry'][m]={'python':str(ROOT/'envs/module13b_transformers/bin/python'),'script':str(launch)}
 slots=[]
 for p in points:
  w=p['full_pool_result']['winner'];routing={k:{path.split('::')[1]:v for path,v in w['routing'].items() if path.split('::')[0]==k and v>0} for k in p['rates']}
  for rep,seed in enumerate([16101,16202,16303],1):
   assert all(m in plan['registry'] for m in w['deployment'])
   slots.append({'run_id':p['stage'].lower()+'_rep'+str(rep),'stage':p['stage'],'repetition':rep,'seed':seed,'deployment':w['deployment'],'lambda_rps':p['rates'],'routing':routing,'measurement_seconds':120.,'safe_memory_budget_mib':BUDGET,'CPU_safe_memory_mib':w['safe_memory_mib'],'CPU_max_rho':w['max_rho'],'target_task':'asr'})
 implementation=[ref(p) for p in sorted((ROOT/'module16').glob('*.py'))]+[ref(launch)]
 reused=[ref(ROOT/p) for p in ['module11b/measure.py','module11b/telemetry.py','module13b/worker.py','module15/baseline.py','deployment_core/routing/e2e_router.py','deployment_core/routing/e2e_surrogate.py','joint_deploy/measurement/deployment_runtime.py','performance_model/local_calibration/stability_v2.py']]
 reused += [ref(v['script']) for k,v in plan['registry'].items() if any(k in t['deployment'] for t in slots)]
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128 and gpu['uuid']==plan['GPU']['uuid']
 save('frozen_gpu_manifest.json',{'status':'FROZEN_BEFORE_GPU','at_utc':now(),'CPU':ref(OUT/'cpu_witness_revalidation.json'),'catalog':ref(source),'plan':plan,'slots':slots,'maximum_sessions':9,'warmup_seconds':10,'measurement_seconds':120,'seeds':[16101,16202,16303],'required_tasks':list(BACKGROUND)+['asr'],'stop_on_first_unstable':True,'stop_on_runtime_failure_or_memory_exceedance':True,'no_outcome_retries':True,'no_models_QPS_budget_routes_changed_after_freeze':True,'no_latency_SLO_defined_or_claimed':True,'stability':'Unmodified stability_v2 thresholds; recoverable is not steady.','memory_rule':'factor*owned postwarm idle mean AND factor*postmeasurement idle mean <= B; sampled owned GPU peak <= B separately, never substitute peak*factor for idle cost.','utilization_semantics':'Per-model handler busy wall fraction plus calibrated observed offered-rho; per-process GPU SM utilization only if NVML supports it, else explicitly unavailable. Device GPU utilization sampled separately.','implementation':implementation,'reused_implementation':reused})
 print('FROZEN',len(slots),'sessions',flush=True)
if __name__=='__main__':main()
