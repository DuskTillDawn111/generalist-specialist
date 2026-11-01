"""Serial progression from discovery pair to untouched new-load holdouts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/generalist-specialist-project')
BASE=ROOT/'artifacts/static-spillover-study-20261022'
OLD=ROOT/'artifacts/static-study-gpu-20261022'
while True:
    q=json.loads((BASE/'development-asr18-v1/runs/queue-progress.json').read_text())
    if q['status']=='CALIBRATION_COMPLETE':break
    if q['status']!='RUNNING':raise RuntimeError('Development stopped; review before validation')
    time.sleep(20)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env.pop('CUDA_VISIBLE_DEVICES',None)
cmd=[sys.executable,'-u',str(OLD/'execution-source-v5/scripts/run_static_gpu_queue.py'),
     '--packet',str(BASE/'transfer-validation-asr18-v1/experiment-packet.json'),
     '--runner',str(OLD/'execution-source-v5/scripts/prepare_static_experiments.py'),
     '--output',str(BASE/'transfer-validation-asr18-v1/runs'),'--execute']
with (BASE/'transfer-validation-asr18-v1/queue.log').open('x') as stream:
    subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
sys.path.insert(0,str(BASE/'predictor-source-v1'))
from gsdeploy.config import SystemConfig
from gsdeploy.replay_transfer import validate_transfer,activate_transfer,receipt
directory=BASE/'transfer-validation-asr18-v1'
c=SystemConfig(**json.loads((directory/'unvalidated-config.json').read_text()))
protocol=json.loads((directory/'protocol.json').read_text())
packet=json.loads((directory/'experiment-packet.json').read_text())
byid={json.loads(p.read_text())['experiment_job_id']:p for p in (directory/'runs').rglob('run.json')}
sources=[dict(id=j['id'],path=str(byid[j['id']].resolve()),sha256=receipt(byid[j['id']])['sha256'],
              packet_path=str((directory/'experiment-packet.json').resolve()),
              packet_sha256=receipt(directory/'experiment-packet.json')['sha256']) for j in packet['jobs']]
report=validate_transfer(c,protocol,sources)
(directory/'validation-report.json').write_text(json.dumps(report,indent=2)+'\n')
activated=activate_transfer(c,directory/'protocol.json',directory/'validation-report.json')
(directory/'activation-result-config.json').write_text(json.dumps(activated,indent=2)+'\n')
print(json.dumps({'status':report['status'],'report':str(directory/'validation-report.json'),
                  'conditions':[{'seed':x['seed'],'errors':x['errors'],'passed':x['passed']} for x in report['conditions']]}),flush=True)
