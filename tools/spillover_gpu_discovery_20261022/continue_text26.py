"""Collect the newly declared text2.6 development packet after ASR repetitions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
previous=BASE/'development-asr19-replication-v1/remaining-queue-progress.json'
while True:
    state=json.loads(previous.read_text())
    if state['status']=='ALL_PREDECLARED_ATTEMPTS_COLLECTED':
        break
    if state['status']!='RUNNING':
        raise RuntimeError('ASR collection stopped before all declared attempts')
    time.sleep(20)
out=BASE/'text26-development-v1'
packet=out/'experiment-packet.json'
decision=dict(schema='gsdeploy.text26_development_scheduling.v1',
    preceding_ASR_progress=dict(path=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest()),
    packet=dict(path=str(packet),sha256=hashlib.sha256(packet.read_bytes()).hexdigest()),
    purpose='Independent alternate-target development after preserving all ASR attempts; no optimum or final confirmation claim',
    low_load_text_packet_executed=False,validation_measurements_used_for_fitting=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (out/'scheduling-decision.json').open('x') as stream:
    stream.write(json.dumps(decision,indent=2)+'\n')
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
runtime=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
with (out/'queue.log').open('x') as stream:
    subprocess.run([sys.executable,'-u',str(runtime/'scripts/run_static_gpu_queue.py'),
        '--packet',str(packet),'--runner',str(runtime/'scripts/prepare_static_experiments.py'),
        '--output',str(out/'runs'),'--execute'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
print(json.dumps(json.loads((out/'runs/queue-progress.json').read_text())),flush=True)
