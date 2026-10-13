"""Freeze engineering-only initialization and two bounded preflight arms."""
from m32f.isolation import install,VIOLATIONS
install()
from m32f.common import *
from replica_scaling.telemetry import snapshot
from m32f.environment import thermal
d=read(ROOT/'results/module32d_router_repair/frozen_validation_manifest.json')
e=read(ROOT/'results/module32e_infrastructure_repair/frozen_validation_manifest.json')
plan=copy.deepcopy(d['plan']);launchers=[]
for i,(model,entry) in enumerate(plan['registry'].items()):
 original=e['registry_original'][model];assert entry==original
 p=OUT/'launchers'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
 p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker.py'))+','+repr(original['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker.py'))+",run_name='__main__')\n")
 entry['script']=str(p);launchers.append(dict(model=model,launcher=ref(p),native_backend=ref(original['script']),python=original['python']))
gpu=snapshot();gpu.update(thermal());assert not gpu['processes']
for k in ['uuid','name','driver','total_mib']:assert gpu[k]==e['GPU'][k]
slots=[]
# Reuse 32D's pre-existing 180s engineering cohort. Both arms match all inputs,
# arrival times and the original 180s router budget horizon. No trace screening.
for arm,rid in [('s_only','E0_S'),('dynamic','E0_D')]:
 slot=copy.deepcopy(next(s for s in d['engineering_slots'] if s['arm']==arm))
 slot.update(run_id=rid,stage='Module32F_SHORT_ENGINEERING_ONLY',pair=None)
 slots.append(slot)
order=['baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt','roberta_sst2','convnext_tiny','openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1','phi4mm','whisper-small']
for n in ['router_optimized.py','vqa_aware_dynamic_router.py','frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json']:
 assert ref(OUT/n)['sha256']==ref(ROOT/'results/module32e_infrastructure_repair'/n)['sha256']
f=dict(module='Module32F CUDA Recovery',frozen_unix_ns=time.time_ns(),plan=plan,GPU=gpu,slots=slots,safe_factor=d['safe_factor'],thermal_protocol=d['thermal_protocol'],quality_policy=d['quality_policy'],environment_control=d['environment_control'],
 environment=e['environment'],registry_original=e['registry_original'],confined_launchers=launchers,checkpoint_metadata=d['checkpoint_metadata'],checkpoints=d['checkpoints'],
 initialization_order=order,preflight_order=['E0_S','E0_D'],preflight_seconds=180.,conditioning_seconds=300.,warmup_seconds=30.,drain_seconds=60.,maximum_preflight_arms=2,
 no_policy_change=True,no_model_or_precision_change=True,no_U_calculation=True,no_formal_runs=True,no_long_window=True,
 acceptance=dict(router_P95_max_ms=1.,all_control_max_strict_lt_ms=10.,all_requests_complete=True,original_stability=True,original_quality=True,original_budget=True,historical_hashes_match=True),
 repair=dict(cause='Landlock write-existing denial of CUDA thread-name /proc/self/task/<tid>/comm',
  worker_rule='Own /proc/self/task WRITE_FILE only; no procfs creation or metadata rights',
  supervisor_rule='Inherited /proc WRITE_FILE envelope for fresh child PIDs; workers narrow before backend import; Python permits only current process task/comm',
  native_start='Unchanged subprocess.Popen then exec, sequential READY; no CUDA in controller'),
 failure_policy='Retain all results; stop on CUDA startup, control >=10ms or invalid preflight; no routing threshold/load/trace/measurement-duration changes after freeze',
 traces={key:value for s in slots for key,value in s['trajectories'].items()},historical_sources=d['sources'],
 frozen_inputs=[ref(OUT/n) for n in ['frozen_routing_parameters.json','audio_metadata_cache.json','selected_pool_catalog.json','frozen_utility_context.json','gpu_preflight_protocol.md']]+list(slots[0]['trajectories'].values()),
 sources=[ref(p) for p in sorted((OUT/'m32f').glob('*.py'))]+[ref(p) for p in sorted((OUT/'launchers').glob('*.py'))]+[ref(OUT/n) for n in ['restricted_worker.py','router_optimized.py','vqa_aware_dynamic_router.py','single_model_regression.py','co_residency_regression.py','gpu_preflight.py','prepare_regression.py']],
 same_native_weights_source=e['original_configuration'])
save('frozen_validation_manifest.json',f);save('frozen_gpu_manifest.json',f)
assert not VIOLATIONS
print('ENGINEERING_FROZEN',order,['E0_S','E0_D'],180,flush=True)
