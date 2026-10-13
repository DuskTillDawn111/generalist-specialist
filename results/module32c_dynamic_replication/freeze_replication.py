import sys, subprocess, ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
from replica_scaling.telemetry import snapshot
from module21.trajectory import arrivals
from joint_deploy.measurement.load_generator import SamplePool
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS
assert read(OUT/'stage_A_gate.json')['passed'];assert not (OUT/'experiment_manifest.json').exists()
old=read(OLD/'experiment_manifest.json');f=copy.deepcopy(old);gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
for k in ['uuid','name','total_mib','driver']:assert gpu[k]==old['GPU'][k]
for r in old['sources']+old['frozen_inputs']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
for n in ['vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']:assert ref(OUT/n)['sha256']==ref(OLD/n)['sha256']
for p,v in old['checkpoint_metadata'].items():assert [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v,p
env={}
for py,prior in old['environment'].items():
 p=subprocess.run([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True,capture_output=True,check=True)
 v=json.loads(p.stdout);assert v==prior,py;env[py]=v
save('initial_environment_identity.json',env)
base=next(s for s in old['slots'] if s['arm']=='s_only');dyn=next(s for s in old['slots'] if s['arm']=='dynamic');rates=base['lambda_rps'];slots=[];traces=[];metadata=read(OUT/'audio_metadata_cache.json')
# Chosen once before drawing trajectories; no counts/opportunity/U screening.
seeds=[320301,320302,320303];order=['P1_S','P1_D','P2_D','P2_S','P3_S','P3_D']
for pair,seed in enumerate(seeds,1):
 ts={}
 for label,seconds in [('warmup',30.),('measurement',2400.)]:
  name=f'independent_pair_{pair}_{label}';ev=arrivals(rates,seconds,seed,name);pool=SamplePool(old['plan']['sample_pool']['path'],list(rates),seed)
  for e in ev:
   e['item']=pool.next(e['task'])
   if e['task']=='asr':assert e['item']['sample']['input'] in metadata, 'Uncalibrated inputs must not modify frozen router metadata'
  fn=f'trajectories/{name}_{seed}.json';save(fn,dict(duration=seconds,seed=seed,label=name,events=ev,independent_replication=True));ts[label]=ref(OUT/fn);traces.append(ts[label])
 for suffix,template in [('S',base),('D',dyn)]:
  s=copy.deepcopy(template);s.update(run_id=f'P{pair}_{suffix}',pair=pair,seed=seed,repetition=pair,stage='Module32C_INDEPENDENT_REPLICATION',measurement_seconds=2400.,trajectories=ts,frozen_offered_counts=dict(collections.Counter(e['task'] for e in read(ts['measurement']['path'])['events'])))
  slots.append(s)
allprior=[read(OLD/'experiment_manifest.json')['slots'][0]['trajectories'][k]['sha256'] for k in ['warmup','measurement']]
assert all(t['sha256'] not in allprior for t in traces);assert len({t['sha256'] for t in traces})==6
for p in (OUT/'m32c').glob('*.py'):ast.parse(p.read_text())
f.update(module='Module32C Utility Attribution and Independent Replication',slots=slots,formal_order=order,formal_seconds=2400.,maximum_formal_runs=6,formal_load_frozen_unix_ns=time.time_ns(),frozen_unix_ns=time.time_ns(),GPU=gpu,environment=env,independent_replication=True,optimizer_modified=False,next_module_automatic=False,Stage_D=False)
f['sources']=old['sources']+[ref(p) for p in sorted((OUT/'m32c').glob('*.py'))]+[ref(OUT/'vqa_aware_dynamic_router.py'),ref(OUT/'freeze_replication.py')]
f['frozen_inputs']=old['frozen_inputs']+traces+[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json','stage_A_gate.json','historical_utility_recalculation.json']]
f['positive_gate']=dict(all_three_pairs_legal_stable=True,real_G_and_S_in_each_dynamic=True,each_delta_U_gt=.01,independent_pair_count=3,mean_delta_95_percent_CI_lower_gt=0,CI_method='Two-sided Student t, df=2, t critical=4.302652729911275; three pair-level contrasts, not requests as independent replicas',if_all_positive_but_CI_crosses_zero='POSITIVE_TREND_EVIDENCE_INSUFFICIENT')
f['history_baseline_policy']='Module32B is CPU attribution only; no historical U used as strict comparator. Three new independently seeded arrival/input-order trajectories; frozen sample pool reused.'
f['trace_independence']=dict(seeds_chosen_once=seeds,no_replacement=True,no_screening_or_development=True,same_input_pool=True,new_datasets=False,prior_development_seed=320232)
f['protocol_adaptation']=dict(original_router_and_parameters_byte_identical=True,original_runtime_phase_byte_identical=True,observation_duration_only_generalization='600s denominators and fixed window assertion/backlog range replaced with frozen duration. Original components, arithmetic, quality, stability thresholds and transport unchanged.',paired_order=order,no_extra_GPU_smoke=True,seed_selection='Three consecutive predeclared Module32C seeds; order counterbalanced S/D, D/S, S/D before outcomes')
f['replication_statistical_policy']=dict(primary_unit='independent matched pair',n=3,confidence=.95,t_df=2,t_critical=4.302652729911275,within_trace_bootstrap_not_success_gate=True,no_parameter_adaptation=True)
save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
save('frozen_routing_manifest.json',dict(status='FROZEN_BEFORE_GPU',at_unix_ns=f['frozen_unix_ns'],source_Module32B=ref(OLD/'frozen_routing_manifest.json'),router=ref(OUT/'vqa_aware_dynamic_router.py'),parameters=ref(OUT/'frozen_routing_parameters.json'),metadata=ref(OUT/'audio_metadata_cache.json'),pool=base['deployment'],rates=rates,quality_policy=f['quality_policy'],budget_mib=base['safe_memory_budget_mib'],utility=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),utility_reference=f['utility_reference'],order=order,seeds=seeds,trajectories=traces,measurement_seconds=2400.,warmup_seconds=30.,drain_seconds=60.,max_GPU_runs=6,positive_gate=f['positive_gate'],no_adjustments=True,backpressure_pending_max=300,backpressure_oldest_seconds=120,optimizer_modified=False))
print('FROZEN',order,seeds,'OFFERS',[s['frozen_offered_counts'] for s in slots[::2]],flush=True)
