"""Audit declared ASR repeats including failures; never impute failed utility."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
progress=BASE/'development-asr19-replication-v1/remaining-queue-progress.json'
while True:
    state=json.loads(progress.read_text())
    if state['status']=='ALL_PREDECLARED_ATTEMPTS_COLLECTED':
        break
    if state['status']!='RUNNING':
        raise RuntimeError('Development collection stopped: '+state['status'])
    time.sleep(20)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([sys.executable,str(ROOT/'tools/spillover_discovery_20261022/audit_asr19_replication_v2.py'),
    '--output',str(BASE/'asr19-replication-development-audit-v2')],cwd=ROOT,env=env,check=True)
