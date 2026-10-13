"""One preregistration, before any GPU experiment; seeds never screened/replaced."""
import sys,subprocess,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32d.common import *
from replica_scaling.telemetry import snapshot
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from m32d.environment import thermal

def main():
 assert read(OUT/'routing_equivalence_report.json')['passed']
 assert read(OUT/'CPU_runtime_acceptance.json')['passed']
 assert not (OUT/'frozen_validation_manifest.json').exists(),'No configuration refreeze after GPU'
 old=read(ROOT/'results/module32c_dynamic_replication/experiment_manifest.json');f=copy.deepcopy(old);gpu=snapshot();gpu.update(thermal())
 assert not gpu['processes'] and gpu['used_mib']<=128
 for k in ['uuid','name','driver','total_mib']:assert gpu[k]==old['GPU'][k]
 for r in old['sources']+old['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for p,v in old['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
 env={}
 for py,prior in old['environment'].items():
  v=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));assert v==prior,py;env[py]=v
 save('initial_environment_identity.json',env)
 for n in ['vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']:assert ref(OUT/n)['sha256']==ref(OLD/n)['sha256']
 base=next(s for s in old['slots'] if s['arm']=='s_only');dyn=next(s for s in old['slots'] if s['arm']=='dynamic');rates=base['lambda_rps'];slots=[];traces=[]
 seeds=[320401,320402,320403];order=['P1_S','P1_D','P2_D','P2_S','P3_S','P3_D']
 def make(pair,seed,seconds,prefix):
  ts={}
  for label,duration in [('conditioning',300.),('warmup',30.),('measurement',seconds)]:
   name=f'{prefix}_{label}';ev=arrivals(rates,duration,seed,name);pool=SamplePool(old['plan']['sample_pool']['path'],list(rates),seed)
   for e in ev:
    e['item']=pool.next(e['task'])
    if e['task']=='asr':assert e['item']['sample']['input'] in read(OUT/'audio_metadata_cache.json')
   n=f'trajectories/{name}_{seed}.json';save(n,dict(duration=duration,seed=seed,label=name,events=ev));ts[label]=ref(OUT/n);traces.append(ts[label])
  result=[]
  for suffix,template in [('S',base),('D',dyn)]:
   s=copy.deepcopy(template);s.update(run_id=f'{prefix}_{suffix}',pair=pair,seed=seed,repetition=pair,stage='Module32D_ENGINEERING_PREFLIGHT' if pair==0 else 'Module32D_INDEPENDENT_VALIDATION',measurement_seconds=seconds,trajectories=ts,frozen_offered_counts=dict(collections.Counter(e['task'] for e in ev)));result.append(s)
  return result
 preflight=make(0,320490,180.,'E0')
 for pair,seed in enumerate(seeds,1):slots.extend(make(pair,seed,2400.,f'P{pair}'))
 # Formal events are never fed to the CPU benchmark or router development.
 for p in (OUT/'m32d').glob('*.py'):ast.parse(p.read_text())
 f.update(module='Module32D Dynamic Router Runtime Repair and Controlled Validation',GPU=gpu,environment=env,slots=slots,engineering_slots=preflight,engineering_order=['E0_S','E0_D'],formal_order=order,frozen_unix_ns=time.time_ns(),maximum_formal_runs=6,formal_seconds=2400.,optimizer_modified=False,next_module_automatic=False,independent_replication=True)
 f['sources']=old['sources']+[ref(p) for p in sorted((OUT/'m32d').glob('*.py'))]+[ref(OUT/n) for n in ['router_optimized.py','vqa_aware_dynamic_router.py','cpu_validation.py','freeze_validation.py']]
 f['frozen_inputs']=old['frozen_inputs']+traces+[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json','routing_equivalence_report.json','CPU_runtime_acceptance.json','gpu_environment_control.md']]
 f['runtime_repair']=dict(strategy_version='original32B_policy_unchanged_runtime_index_v1',same_state_equivalence_exact=True,wrapper_timing='original full lock acquisition+state projection+decide+enqueue',overhead_p95_max_ms=1.,state_update='O(1) OrderedDict insert/delete under existing RLock; active projection O(A), unchanged original decide O(AlogA), A bounded by original max_pending300 safety; O(1) relative to cumulative history N',archive_kept_for_audit_and_late_results=True)
 f['environment_control']=dict(conditioning_seconds=300.,conditioning_route='S-only in both arms, same pair trace and original six-task load; actual useful inference only',original_warmup_seconds=30.,original_thermal_protocol=old['thermal_protocol'],reload_all_models_each_arm=True,no_clocks_power_fan_changes=True,sensor_interval_seconds=.5,journal_interval_seconds=10,postwarm_pair_temperature_difference_max_c=5.,measurement_mean_temperature_difference_max_c=5.,measurement_mean_sm_clock_relative_difference_max=.15,power_limit_must_equal_frozen=True,any_hw_or_sw_thermal_slowdown_invalid=True,fan_and_clock_events_logged=True)
 f['failure_policy']=dict(technical='STOP_NO_RETRY',environment='STOP_NO_REFREEZE',negative='Keep all frozen pairs regardless sign when engineering valid; never tune',CPU_benchmark_error='no GPU until repaired and development retested before freeze')
 f['trace_independence']=dict(seeds_chosen_once=seeds,engineering_seed=320490,prior_seeds=[320232,320301,320302,320303],no_replacement=True,no_formal_screening=True,same_frozen_input_pool=True,new_arrival_and_input_order_streams=True)
 f['positive_gate']=dict(all_three_pairs_legal_stable=True,real_G_and_S_each_dynamic=True,each_delta_U_gt=.01,pair_count=3,CI95_lower_gt_zero=True,t_critical=4.302652729911275,df=2)
 save('frozen_validation_manifest.json',f);save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
 save('frozen_routing_manifest.json',dict(router=ref(OUT/'vqa_aware_dynamic_router.py'),runtime=ref(OUT/'router_optimized.py'),parameters=ref(OUT/'frozen_routing_parameters.json'),duration=2400.,metadata=ref(OUT/'audio_metadata_cache.json'),semantic_version='Module32B original',no_threshold_changes=True))
 print('FROZEN',order,seeds,flush=True)
if __name__=='__main__':main()
