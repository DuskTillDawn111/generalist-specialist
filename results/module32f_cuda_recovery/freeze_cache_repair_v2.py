"""Pre-preflight infrastructure amendment; retain v1 and failed raw records."""
from m32f.isolation import install
install()
from m32f.common import *
import ast
assert not (OUT/'frozen_validation_manifest_v2.json').exists()
assert not list((OUT/'run_attempts').glob('*.json')),'No preflight outcomes can tune repair'
assert read(OUT/'triton_cache_CPU_audit.json')['passed'] and read(OUT/'triton_cache_CPU_repaired.json')['passed']
f=read(OUT/'frozen_validation_manifest.json');f['initial_manifest']=ref(OUT/'frozen_validation_manifest.json');f['infrastructure_revision']=2
f['repair']['triton_cache']='SameDirectoryCacheManager, native cache key/bytes/get/group/compiler flags unchanged; same-dir temp atomic os.replace for Landlock ABI1 EXDEV compatibility'
f['repair']['native_inference_settings_unchanged']=True
for i,(model,entry) in enumerate(f['plan']['registry'].items()):
 original=f['registry_original'][model];p=OUT/'launchers_v2'/f'worker_{i}.py';p.parent.mkdir(exist_ok=True)
 p.write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(OUT))+')\nsys.argv=['+repr(str(OUT/'restricted_worker_v2.py'))+','+repr(original['script'])+']+sys.argv[1:]\nrunpy.run_path('+repr(str(OUT/'restricted_worker_v2.py'))+",run_name='__main__')\n")
 entry['script']=str(p)
sources=sorted((OUT/'m32f').glob('*.py'))+sorted((OUT/'launchers_v2').glob('*.py'))+[OUT/n for n in ['restricted_worker_v2.py','router_optimized.py','vqa_aware_dynamic_router.py','single_model_regression_v2.py','co_residency_regression_v2.py','gpu_preflight_v2.py','run_initialization_regression_v2.py','freeze_cache_repair_v2.py']]
for p in sources:ast.parse(p.read_text())
f['sources']=[ref(p) for p in sources];f['frozen_unix_ns']=time.time_ns()
f['failure_policy']='Version2 engineering: stop on initialization/control/preflight failure, retain results; no strategy/config/trace/threshold/window edits. Version1 startup diagnosis/failed path probes retained separately; no preflight was executed in version1.'
save('frozen_validation_manifest_v2.json',f)
save('infrastructure_amendment_v2.json',dict(before_preflight=True,preflight_attempts=0,policy_or_model_changes=False,
 original_configuration=ref(OUT/'frozen_validation_manifest.json'),new_configuration=ref(OUT/'frozen_validation_manifest_v2.json'),
 reason='PARSeq READY successful after CUDA procfs repair; first native torch bmm triggers Triton cold cache rename EXDEV under ABI1. CPU source inspection and filesystem/concurrent tests support same-directory atomic cache publication.',
 initial_negative_preserved=ref(OUT/'single_model_runs/0.json'),diagnostic_full_worker_error=ref(OUT/'single_model_diagnostic/0.json'),
 cache_tests=[ref(OUT/'triton_cache_CPU_audit.json'),ref(OUT/'triton_cache_CPU_repaired.json')],
 measurement_parameters_unchanged=True,optimizer_changed=False,historical_sources_or_Gates_changed=False))
print('V2_CACHE_INFRASTRUCTURE_FROZEN_NO_INFERENCE_CHANGE',flush=True)
