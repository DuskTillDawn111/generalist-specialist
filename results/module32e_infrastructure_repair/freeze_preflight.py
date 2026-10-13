"""One engineering preflight, no search/tuning/formal utility calculation."""
import sys,subprocess,ast
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
from replica_scaling.telemetry import snapshot
from m32e.environment import thermal
def main():
 assert not (OUT/'frozen_validation_manifest.json').exists(),'Freeze once before GPU'
 assert read(OUT/'history_write_protection_tests.json')['passed']
 assert read(OUT/'control_probe_attribution.json')['passed']
 assert read(OUT/'CPU_phase_transition_acceptance.json')['passed']
 assert read(OUT/'cpu_regression/routing_equivalence_report.json')['passed']
 assert read(OUT/'cpu_regression/CPU_runtime_acceptance.json')['passed']
 assert read(OUT/'history_hashes/before_CPU_vs_after_CPU.json')['passed']
 assert read(OUT/'isolated_cache_CPU_acceptance.json')['passed']
 d=read(ROOT/'results/module32d_router_repair/frozen_validation_manifest.json')
 b=read(ROOT/'results/module32b_vqa_aware_routing/experiment_manifest.json')
 slot=copy.deepcopy(next(s for s in b['slots'] if s['arm']=='dynamic'))
 slot.update(run_id='E0_DYNAMIC',stage='Module32E_ENGINEERING_ONLY',measurement_seconds=600.,pair=None)
 # Keep the already saved 32B development input stream; not an independent
 # scientific replicate and not selected using new routing outcomes.
 slot['trajectories']['conditioning']=d['engineering_slots'][0]['trajectories']['conditioning']
 slot['frozen_offered_counts']=dict(collections.Counter(e['task'] for e in read(slot['trajectories']['measurement']['path'])['events']))
 assert slot['lambda_rps']==d['engineering_slots'][1]['lambda_rps']
 assert slot['deployment']==d['engineering_slots'][1]['deployment']
 for n in ['router_optimized.py','vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']:
  assert ref(OUT/n)['sha256']==ref(ROOT/'results/module32d_router_repair'/n)['sha256']
 gpu=snapshot();gpu.update(thermal());assert not gpu['processes'] and gpu['used_mib']<=128
 for k in ['uuid','name','driver','total_mib']:assert gpu[k]==d['GPU'][k]
 env={}
 for py,previous in d['environment'].items():
  v=json.loads(subprocess.check_output([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True));assert v==previous,py;env[py]=v
 save('environment_identity.json',env)
 # ABI1 metadata holes are denied by seccomp, including chmod. Transformers
 # shutil.copy normally chmods a new dynamic-code cache file. Preseed identical
 # local source bytes into the isolated cache so its equality test skips copy;
 # no inference code/settings or historical cache metadata are changed.
 cache=OUT/'work/cache/hf/modules/transformers_modules/phi4-multimodal';cache.mkdir(parents=True,exist_ok=True)
 cache_refs=[]
 for source in sorted((ROOT/'models/phi4-multimodal').glob('*.py')):
  target=cache/source.name;target.write_bytes(source.read_bytes());assert ref(target)['sha256']==ref(source)['sha256']
  cache_refs.append(dict(source=ref(source),isolated_copy=ref(target)))
 save('isolated_dynamic_code_cache.json',dict(copies=cache_refs,reason='Preseed byte-identical native local Phi dynamic module cache; avoid cache copy metadata syscalls under strict read-only confinement'))
 plan=copy.deepcopy(d['plan']);registry_original=copy.deepcopy(plan['registry']);launchers=[]
 for i,(name,v) in enumerate(plan['registry'].items()):
  target=v['script'];p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
  code="import sys,runpy\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(OUT))+ ")\nsys.argv=["+repr(str(OUT/'restricted_worker.py'))+","+repr(target)+"]+sys.argv[1:]\nrunpy.run_path("+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n"
  p.write_text(code);v['script']=str(p);launchers.append(dict(model=name,launcher=ref(p),original_backend=ref(target),python=v['python']))
 # Import/parse the full executable infrastructure before it is frozen.
 from m32e.harness import run
 codepaths=sorted((OUT/'m32e').glob('*.py'))+sorted((OUT/'launchers').glob('*.py'))+sorted(OUT.glob('*.py'))
 for p in codepaths:ast.parse(p.read_text())
 f=dict(module='Module32E Experimental Infrastructure Repair and GPU Preflight',frozen_unix_ns=time.time_ns(),
  plan=plan,GPU=gpu,slot=slot,safe_factor=d['safe_factor'],thermal_protocol=d['thermal_protocol'],quality_policy=d['quality_policy'],
  environment_control=d['environment_control'],environment=env,registry_original=registry_original,confined_launchers=launchers,
  original_configuration=ref(ROOT/'results/module32d_router_repair/frozen_validation_manifest.json'),
  inherited_sources=d['sources'],inherited_frozen_inputs=d['frozen_inputs'],checkpoints=d['checkpoints'],checkpoint_metadata=d['checkpoint_metadata'],request_input_files=d['request_input_files'],
  sources=[ref(p) for p in codepaths],frozen_inputs=[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json','gpu_environment_control.md','isolated_dynamic_code_cache.json']]+[v['isolated_copy'] for v in cache_refs]+list(slot['trajectories'].values()),
  acceptance=dict(router_p95_max_ms=1.,control_max_strict_lt_ms=10.,all_offered_requests_retained=True,no_unstable_acceptance=True,quality_and_budget_original=True,history_hashes_must_match=True),
  scope=dict(maximum_GPU_runs=1,measurement_seconds=600.,conditioning_seconds=300.,warmup_seconds=30.,drain_seconds=60.,formal_runs=0,U_computed=False,model_pool_search=False,policy_changed=False,optimizer_changed=False),
  failure_policy='Immediate stop on any control check/publication >=10ms; preserve all probes/entire planned cohort; no retry/no threshold change',
  no_future_information=True,original_policy_SHA=ref(OUT/'vqa_aware_dynamic_router.py'),original_runtime_SHA=ref(OUT/'router_optimized.py'),
  output_protection='Per-process inherited Landlock + seccomp + Python audit; explicit Module32E output namespace; all historical scientific input read-only; /dev and /dev/shm IPC exception',
  CPU_evidence=[ref(OUT/n) for n in ['history_write_protection_tests.json','control_probe_attribution.json','CPU_phase_transition_acceptance.json','cpu_regression/routing_equivalence_report.json','cpu_regression/CPU_runtime_acceptance.json','history_hashes/before_CPU_vs_after_CPU.json']])
 assert not VIOLATIONS
 save('frozen_validation_manifest.json',f);save('frozen_gpu_manifest.json',f)
 print('FROZEN_PREFLIGHT_ONLY',slot['run_id'],slot['frozen_offered_counts'],flush=True)
if __name__=='__main__':main()
