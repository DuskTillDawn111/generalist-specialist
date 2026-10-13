"""Sequential dependency gates; stop on first failure, retain every attempt."""
import subprocess,sys,traceback
from m32f.isolation import install
install('supervisor')
from m32f.common import *
f=read(OUT/'frozen_validation_manifest.json');py=sys.executable;attempts=[];failure=None
assert read(OUT/'cpu_regression/CPU_runtime_acceptance.json')['passed']
assert read(OUT/'cpu_regression/routing_equivalence_report.json')['passed']
assert read(OUT/'history_write_protection_tests.json')['passed'] and read(OUT/'proc_scope_protection.json')['passed']
assert read(OUT/'history_hashes/before_diagnosis_vs_before_application.json')['passed']
try:
 for i,model in enumerate(f['initialization_order']):
  print('SINGLE_MODEL_START',i,model,flush=True);attempts.append(dict(stage='single',index=i,model=model))
  save('initialization_campaign_status.json',dict(attempts=attempts,status='RUNNING',stage='single',index=i))
  subprocess.run([py,'-u',str(OUT/'single_model_regression.py'),str(i)],check=True,timeout=600)
 print('SIX_CO_RESIDENCY_START',flush=True);attempts.append(dict(stage='co_residency'))
 subprocess.run([py,'-u',str(OUT/'co_residency_regression.py')],check=True,timeout=900)
except BaseException as exc:failure=dict(error=repr(exc),traceback=traceback.format_exc())
save('initialization_campaign_status.json',dict(attempts=attempts,status='PASSED' if failure is None else 'FAILED',failure=failure,no_retries=True))
print('INITIALIZATION_CAMPAIGN_FINISH',failure,flush=True)
if failure:raise SystemExit(1)
