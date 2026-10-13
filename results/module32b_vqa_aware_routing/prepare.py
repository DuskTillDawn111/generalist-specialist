import sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p))
from m32b.common import *
assert not (OUT/'protected_before.json').exists()
b=read(OLD/'protected_before.json')
for f in OLD.rglob('*'):
 if f.is_file():b[str(f)]=[f.stat().st_size,f.stat().st_mtime_ns]
save('protected_before.json',b)
save('historical_module32_file_hashes.json',[ref(f) for f in sorted(OLD.rglob('*')) if f.is_file()])
save('original_frozen_context.json',dict(at_unix_ns=time.time_ns(),Module32_manifest=ref(OLD/'experiment_manifest.json'),Module32_gate=ref(OLD/'module32_gate.json'),Module20A_utility=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),Module32_final_audit=ref(OLD/'final_data_acceptance.json'),no_GPU_started=True))
print('PROTECTED',len(b))
