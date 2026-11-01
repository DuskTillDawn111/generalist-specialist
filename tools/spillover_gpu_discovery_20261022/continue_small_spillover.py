"""Queue predeclared development measurements after existing validation GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
OLD=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
progress=BASE/'transfer-validation-asr18-v1/runs/queue-progress.json'
while True:
    state=json.loads(progress.read_text())
    if state['status']=='CALIBRATION_COMPLETE':
        break
    if state['status']!='RUNNING':
        raise RuntimeError('Prior GPU collection stopped: '+state['status'])
    time.sleep(20)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
out=BASE/'development-small-spillover-v2'
for rate in ('1p8','1p9'):
    packet=out/('experiment-packet-asr'+rate+'.json')
    cmd=[sys.executable,'-u',str(OLD/'scripts/run_static_gpu_queue.py'),
         '--packet',str(packet),'--runner',str(OLD/'scripts/prepare_static_experiments.py'),
         '--output',str(out/('runs-asr'+rate)),'--execute']
    with (out/('queue-asr'+rate+'.log')).open('x') as stream:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/('runs-asr'+rate)/'queue-progress.json').read_text())
    print(json.dumps(result),flush=True)
    if result['status']!='CALIBRATION_COMPLETE':
        raise RuntimeError('Development collection stopped')
