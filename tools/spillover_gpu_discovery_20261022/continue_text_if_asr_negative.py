"""Execute the predeclared text fallback only if the ASR1.9 development pair fails.

The decision uses development seed51 only. A positive ASR result keeps the GPU
free for that direction instead. No validation measurements are read here.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
OLD=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
progress=BASE/'development-small-spillover-v2/runs-asr1p9/queue-progress.json'
while True:
    if progress.exists():
        q=json.loads(progress.read_text())
        if q['status']=='CALIBRATION_COMPLETE':
            break
        if q['status']!='RUNNING':
            raise RuntimeError('ASR development stopped: '+q['status'])
    time.sleep(20)
expected={'asr1p9_specialist_share_1p0','asr1p9_specialist_share_0p875'}
rows={j['labels'][0]:j for j in q['jobs']}
if set(rows)!=expected or not all(j.get('eligible') and j.get('U') is not None for j in rows.values()):
    raise ValueError('Incomplete/ineligible ASR development pair')
delta=rows['asr1p9_specialist_share_0p875']['U']-rows['asr1p9_specialist_share_1p0']['U']
out=BASE/'text-backup-development-v1'
decision=dict(schema='gsdeploy.text_backup_scheduling_decision.v1',
    source=dict(path=str(progress),sha256=hashlib.sha256(progress.read_bytes()).hexdigest()),
    measured_ASR1p9_development_delta_U=delta,asr_positive=delta>0,
    run_text_fallback=delta<=0,validation_results_used=False,confirmation_results_used=False,
    scheduling_source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
with (out/'scheduling-decision.json').open('x') as stream:
    stream.write(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision),flush=True)
if delta>0:
    sys.exit(0)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
cmd=[sys.executable,'-u',str(OLD/'scripts/run_static_gpu_queue.py'),
    '--packet',str(out/'experiment-packet.json'),'--runner',str(OLD/'scripts/prepare_static_experiments.py'),
    '--output',str(out/'runs'),'--execute']
with (out/'queue.log').open('x') as stream:
    subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
print(json.dumps(json.loads((out/'runs/queue-progress.json').read_text())),flush=True)
