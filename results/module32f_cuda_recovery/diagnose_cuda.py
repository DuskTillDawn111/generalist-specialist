"""Sequential fresh exec diagnostics; no models/policy changes or utility."""
import sys,subprocess,traceback
from m32f.common import *
from replica_scaling.telemetry import snapshot
f=read(ROOT/'results/module32e_infrastructure_repair/frozen_validation_manifest.json')
env=os.environ.copy();env.update(f['plan']['environment_overrides']);env['PYTHONDONTWRITEBYTECODE']='1'
py=f['registry_original']['baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt']['python']
rows=[]
for layer in ['audit','full','landlock','seccomp','audit']:
 i=len(rows);before=snapshot();assert not before['processes'] and before['used_mib']<=128
 if (OUT/f'diagnosis_runs/run_{i}.json').exists():
  prior=read(OUT/f'diagnosis_runs/run_{i}.json');assert prior['layer']==layer;output=prior['stdout']
 else:
  result=subprocess.run([py,str(OUT/'minimal_cuda.py'),layer],env=env,cwd=str(ROOT),text=True,capture_output=True,timeout=90)
  save(f'diagnosis_runs/run_{i}.json',dict(layer=layer,command=[py,str(OUT/'minimal_cuda.py'),layer],exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,before=before,after=snapshot()));output=result.stdout
 line=output.strip().splitlines()[-1];parsed=json.loads(line);rows.append(parsed)
 print('CUDA_LAYER',layer,parsed['passed'],parsed.get('error'),flush=True)
 if i==0 and not parsed['passed']:
  save('CUDA_environment_gate.json',dict(gate='CUDA_ENVIRONMENT_UNAVAILABLE',results=rows,application_experiments_stopped=True));raise SystemExit(1)
save('CUDA_layer_comparison.json',dict(results=rows,baseline_usable=rows[0]['passed'] and rows[-1]['passed'],
 classification='Fresh independent exec baseline and isolation components; do not infer PARSeq-specific error from shared CUDA init failure'))
csvsave('CUDA_layer_comparison.csv',rows)
