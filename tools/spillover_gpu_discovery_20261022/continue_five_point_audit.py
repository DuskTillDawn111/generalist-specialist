"""Run the strict CPU audit only after all predeclared development jobs finish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
progress=BASE/'development-small-spillover-v2/runs-asr1p9/queue-progress.json'
while True:
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['status']=='CALIBRATION_COMPLETE':
            break
        if state['status']!='RUNNING':
            raise RuntimeError('Predeclared development stopped: '+state['status'])
    time.sleep(20)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
subprocess.run([sys.executable,str(ROOT/'tools/spillover_discovery_20261022/audit_five_development_points.py'),
    '--output',str(BASE/'five-development-point-audit-v1')],cwd=ROOT,env=env,check=True)
