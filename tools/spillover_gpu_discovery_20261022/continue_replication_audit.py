"""Audit all declared ASR1.9 development repetitions after serial GPU collection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
progress=BASE/'development-asr19-replication-v1/runs/queue-progress.json'
while True:
    state=json.loads(progress.read_text())
    if state['status']=='CALIBRATION_COMPLETE':
        break
    if state['status']!='RUNNING':
        raise RuntimeError('Development repeat collection stopped: '+state['status'])
    time.sleep(20)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([sys.executable,str(ROOT/'tools/spillover_discovery_20261022/audit_asr19_replication.py'),
    '--output',str(BASE/'asr19-replication-development-audit-v1')],cwd=ROOT,env=env,check=True)
