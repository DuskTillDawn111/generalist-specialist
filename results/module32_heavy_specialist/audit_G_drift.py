import sys,subprocess
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
import numpy as np
oldf=read(ROOT/'results/module31a_qwen_whisper_legal_control/experiment_manifest.json')
paths=['joint_deploy/qwen_worker.py','benchmark/scripts/eval_qwen_librispeech.py','benchmark/scripts/eval_qwen_voxlingua.py','benchmark/scripts/eval_qwen_scienceqa.py']
sources=[]
for name in paths:
 p=ROOT/name;prior=next(r for r in oldf['sources'] if r['path']==str(p));now=ref(p);sources.append(dict(path=str(p),old_hash=prior['sha256'],current_hash=now['sha256'],unchanged=prior['sha256']==now['sha256']))
py=oldf['plan']['registry']['qwen']['python']
probe=subprocess.run([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True,capture_output=True,check=True)
env=json.loads(probe.stdout);prior=oldf['environment'][py];changes={k:[prior['packages'].get(k),v] for k,v in env['packages'].items() if prior['packages'].get(k)!=v}
old=[]
for a in ['A1','A2']:
 d=read(ROOT/'results/module31a_qwen_whisper_legal_control/raw_runs'/(a+'.json'));old +=[r for r in d['phases']['measurement']['request_records'] if r['task']=='asr' and r['model']=='qwen#0' and r['status']=='completed']
new=read(OUT/'raw_runs/B_G.json');rr=[r for r in new['phases']['measurement']['request_records'] if r['task']=='asr' and r['model']=='qwen#0' and r['status']=='completed'];paired=[]
for sid in sorted(set(r['sample_id'] for r in old)&set(r['sample_id'] for r in rr)):
 a=[r for r in old if r['sample_id']==sid];b=[r for r in rr if r['sample_id']==sid]
 paired.append(dict(sample_id=sid,old_n=len(a),new_n=len(b),old_service_ms=float(np.mean([x['worker_latency_ms'] for x in a])),new_service_ms=float(np.mean([x['worker_latency_ms'] for x in b])),prediction_changed=any(x['worker_response']['prediction']!=a[0]['worker_response']['prediction'] for x in b)))
w=new['phases']['measurement']['window'];ss=[s for s in new['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
save('G_service_drift_audit.json',dict(source_hashes=sources,package_version_changes=changes,current_environment=env,matched_inputs=paired,mean_matched_old_ms=float(np.mean([x['old_service_ms'] for x in paired])),mean_matched_new_ms=float(np.mean([x['new_service_ms'] for x in paired])),failed_probe_censored=True,U=None,telemetry={k:dict(mean=float(np.mean([s[k] for s in ss])),min=min(s[k] for s in ss),max=max(s[k] for s in ss)) for k in ['sm_clock_mhz','memory_clock_mhz','temperature_gpu_c','worker_CPU_core_percent','gpu_util_percent']},scope='Same sample ID, different co-resident S and load; descriptive, not isolated causal attribution'))
print('G_DRIFT_AUDIT',sources,changes,len(paired))
