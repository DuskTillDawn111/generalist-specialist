"""Use development repeats to decide whether to execute predeclared text fallback.

A failed baseline is retained, not replaced by a score or treated as evidence of
both-feasible utility improvement. This driver reads no holdout measurements.
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
rep=BASE/'development-asr19-replication-v1'
progress=rep/'remaining-queue-progress.json'
while True:
    q=json.loads(progress.read_text())
    if q['status']=='ALL_PREDECLARED_ATTEMPTS_COLLECTED':
        break
    if q['status']!='RUNNING':
        raise RuntimeError('Previous development collection stopped: '+q['status'])
    time.sleep(20)
packet=json.loads((rep/'experiment-packet.json').read_text())
conditions=[]
for j in packet['jobs']:
    p=rep/'runs'/(j['labels'][0]+'-attempt1')/'run.json'
    r=json.loads(p.read_text())
    if r['experiment_job_id']!=j['id']:
        raise ValueError('Development identity mismatch')
    conditions.append(dict(label=j['labels'][0],seed=j['seed'],status=r['status'],
        eligible=r['evaluation']['utility']['eligible'],U=r['evaluation']['utility']['U'],
        path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
all_valid=all(c['eligible'] and c['U'] is not None for c in conditions)
out=BASE/'text-backup-development-v1'
decision=dict(schema='gsdeploy.text_backup_replication_decision.v1',
    prior_remaining_progress_sha256=hashlib.sha256(progress.read_bytes()).hexdigest(),
    development_conditions=conditions,all_declared_ASR_repeat_conditions_valid=all_valid,
    rationale='ASR direction is not established when any declared repeat remains invalid; preserve failures and explore the predeclared alternate target.',
    run_text_fallback=not all_valid,validation_results_used=False,confirmation_results_used=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (out/'replication-scheduling-decision.json').open('x') as stream:
    stream.write(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision),flush=True)
if all_valid:
    sys.exit(0)
runtime=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
with (out/'queue.log').open('x') as stream:
    subprocess.run([sys.executable,'-u',str(runtime/'scripts/run_static_gpu_queue.py'),
        '--packet',str(out/'experiment-packet.json'),'--runner',str(runtime/'scripts/prepare_static_experiments.py'),
        '--output',str(out/'runs'),'--execute'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
print(json.dumps(json.loads((out/'runs/queue-progress.json').read_text())),flush=True)
