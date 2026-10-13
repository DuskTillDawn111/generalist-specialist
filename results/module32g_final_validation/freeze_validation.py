"""Single preregistration after CPU regression, before any formal GPU start."""
from m32g.isolation import install
install('supervisor')
from m32g.common import *
import subprocess,ast
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from replica_scaling.telemetry import snapshot
from m32g.environment import thermal
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS

def main():
 assert not (OUT/'frozen_validation_manifest.json').exists(),'No refreeze'
 prior=ROOT/'results/module32f_cuda_recovery';old=read(prior/'frozen_validation_manifest_v2.json')
 assert read(prior/'module32f_gate.json')['gate']=='GPU_PREFLIGHT_RECOVERED'
 for r in old['sources']+old['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for n in ['router_optimized.py','vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']:
  assert ref(OUT/n)['sha256']==ref(prior/n)['sha256']
 assert read(OUT/'history_write_protection_tests.json')['passed']
 assert read(OUT/'CPU_phase_transition_acceptance.json')['passed']
 assert read(OUT/'isolated_cache_CPU_acceptance.json')['passed']
 assert read(OUT/'cpu_regression/CPU_runtime_acceptance.json')['passed']
 assert read(OUT/'cpu_regression/routing_equivalence_report.json')['passed']
 assert read(OUT/'history_hashes/before_regression_vs_before_campaign.json')['passed']
 env={}
 for py,expected in old['environment'].items():
  result=subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True)
  actual=json.loads(result);assert actual==expected,py;env[py]=actual
 save('initial_environment_identity.json',env)
 gpu=snapshot();gpu.update(thermal());assert not gpu['processes'] and gpu['used_mib']<=128
 for k in ['uuid','name','driver','total_mib']:assert gpu[k]==old['GPU'][k]
 for p,v in old['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 plan=copy.deepcopy(old['plan']);launchers=[]
 for i,(m,entry) in enumerate(plan['registry'].items()):
  native=old['registry_original'][m];p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
  p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker_v2.py'))+','+repr(native['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker_v2.py'))+",run_name='__main__')\n")
  entry['script']=str(p);launchers.append(dict(model=m,launcher=ref(p),native_backend=ref(native['script']),python=native['python']))
 seeds=[320701,320702,320703];order=['P1_S','P1_D','P2_D','P2_S','P3_S','P3_D'];slots=[];traces=[]
 used=set()
 for name in ['module32b_vqa_aware_routing','module32c_dynamic_replication','module32d_router_repair','module32e_infrastructure_repair','module32f_cuda_recovery']:
  for p in (ROOT/'results'/name).glob('*manifest*.json'):
   f=read(p)
   if isinstance(f,dict):used.update(s.get('seed') for s in f.get('slots',[]))
 assert not used.intersection(seeds)
 rates=old['slots'][0]['lambda_rps'];meta=read(OUT/'audio_metadata_cache.json')
 for pair,seed in enumerate(seeds,1):
  ts={}
  for label,seconds in [('conditioning',300.),('warmup',30.),('measurement',2400.)]:
   name=f'Module32G_independent_pair_{pair}_{label}';ev=arrivals(rates,seconds,seed,name)
   pool=SamplePool(plan['sample_pool']['path'],list(rates),seed)
   for e in ev:
    e['item']=pool.next(e['task'])
    if e['task']=='asr':assert e['item']['sample']['input'] in meta
   fn=f'trajectories/P{pair}_{label}_{seed}.json';save(fn,dict(duration=seconds,seed=seed,label=name,events=ev,independent_replication=True));ts[label]=ref(OUT/fn);traces.append(ts[label])
  for suffix,arm in [('S','s_only'),('D','dynamic')]:
   s=copy.deepcopy(next(s for s in old['slots'] if s['arm']==arm));s.update(run_id=f'P{pair}_{suffix}',stage='Module32G_FINAL_INDEPENDENT_VALIDATION',pair=pair,seed=seed,repetition=pair,measurement_seconds=2400.,trajectories=ts,frozen_offered_counts=dict(collections.Counter(e['task'] for e in ev)))
   s.pop('actual_planned_G_asr',None);s.pop('actual_planned_G_fraction',None);slots.append(s)
 f=copy.deepcopy(old);f.update(module='Module32G Final Independent GPU Utility Validation',frozen_unix_ns=time.time_ns(),plan=plan,GPU=gpu,environment=env,slots=slots,formal_order=order,formal_seconds=2400.,maximum_formal_runs=6,confined_launchers=launchers,independent_replication=True,optimizer_modified=False,next_module_automatic=False,acceptance=dict(router_P95_max_ms=1.,all_control_max_strict_lt_ms=10.,all_requests_complete=True,original_stability=True,original_quality=True,original_budget=True,historical_hashes_match=True))
 for k in ['no_U_calculation','no_formal_runs','no_long_window','maximum_preflight_arms','preflight_order','preflight_seconds','engineering_slots','engineering_order','initial_manifest']:f.pop(k,None)
 f['sources']=[ref(p) for p in sorted((OUT/'m32g').glob('*.py'))]+[ref(p) for p in sorted((OUT/'launchers').glob('*.py'))]+[ref(OUT/n) for n in ['restricted_worker_v2.py','router_optimized.py','vqa_aware_dynamic_router.py','campaign.py','freeze_validation.py','finalize_validation.py']]+[ref(ROOT/'module20/optimizer.py'),ref(ROOT/'module26/common.py'),ref(ROOT/'performance_model/local_calibration/stability_v2.py')]+[ref(x['script']) for x in old['registry_original'].values()]
 for row in f['sources']:
  if row['path'].endswith('.py'):ast.parse(Path(row['path']).read_text())
 f['frozen_inputs']=[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json','validation_protocol.md','historical_input_review.json','initial_environment_identity.json','cpu_regression/CPU_runtime_acceptance.json','cpu_regression/routing_equivalence_report.json']]+traces+[ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),old['quality_policy'],ref(prior/'module32f_gate.json')]
 f['stability_thresholds']=DEFAULT_THRESHOLDS
 f['failure_policy']=dict(technical='STOP_NO_RETRY',environment='STOP_NO_REFREEZE',negative='Complete all remaining predefined runs; no tuning',retry_count=0)
 f['trace_independence']=dict(seeds_chosen_once=seeds,prior_manifest_seeds=sorted(x for x in used if x is not None),same_original_input_pool=True,new_datasets=False,new_arrival_and_input_order_streams=True,no_screening=True,no_seed_replacement=True)
 f['positive_gate']=dict(all_three_pairs_legal_stable=True,real_G_and_S_each_dynamic=True,each_delta_U_gt=.01,pair_count=3,CI95_lower_gt_zero=True,t_critical=4.302652729911275,df=2)
 f['replication_statistical_policy']=dict(primary_unit='independent matched pair',n=3,confidence=.95,t_df=2,t_critical=4.302652729911275,no_parameter_adaptation=True)
 f['source_Module32F']=ref(prior/'frozen_validation_manifest_v2.json')
 f['history_source']=ref(OUT/'history_hashes/before_campaign.json')
 f['preflight_U_excluded']=True;f['historical_U_excluded']=True
 save('frozen_validation_manifest.json',f);save('frozen_gpu_manifest.json',f)
 print('FROZEN',order,seeds,flush=True)
if __name__=='__main__':main()
