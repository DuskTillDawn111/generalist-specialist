"""Collect the remaining predeclared development jobs after a failed first job.

No retries, no imputation, no modification of packet/runtime/raw. Failed
experiments remain failures and do not stop unrelated predeclared conditions.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022/development-asr19-replication-v1'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.static_study import validate_packet

def receipt(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def publish(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

def main():
    packet_path=BASE/'experiment-packet.json'
    packet=json.loads(packet_path.read_text())
    if not validate_packet(packet)['valid']:
        raise ValueError('Frozen packet validation failed')
    old=BASE/'runs/queue-progress.json'
    failed=json.loads(old.read_text())
    if failed['status']!='STOPPED_FOR_REVIEW' or len(failed['jobs'])!=1:
        raise ValueError('Expected exactly one preserved first failed attempt')
    failed_raw=ROOT/failed['jobs'][0]['output']
    if failed_raw.is_absolute():
        pass
    raw=json.loads(failed_raw.read_text())
    if raw['status']!='failed' or raw['evaluation']['utility']['U'] is not None:
        raise ValueError('Expected failed first attempt with missing U')
    labels=['asr19_n3_specialist_share_1p0_seed51',
            'asr19_n2_specialist_share_0p875_seed52',
            'asr19_n2_specialist_share_1p0_seed53',
            'asr19_n2_specialist_share_0p875_seed53']
    bylabel={j['labels'][0]:j for j in packet['jobs']}
    if set(labels)|set(failed['jobs'][0]['labels'])!=set(bylabel):
        raise ValueError('Remaining roster changed')
    progress=BASE/'remaining-queue-progress.json'
    if progress.exists():
        raise FileExistsError(progress)
    state=dict(schema='gsdeploy.development_remaining_queue.v1',status='RUNNING',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        packet=receipt(packet_path),source=receipt(Path(__file__)),prior_failure=receipt(failed_raw),
        execution_order=labels,jobs=[],policy='All attempts preserved. No retries or imputed U; continue independent predeclared conditions after run-level invalidity.')
    publish(progress,state)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    for label in labels:
        job=bylabel[label]
        output=BASE/'runs'/(label+'-attempt1')
        log=BASE/'runs'/(label+'-attempt1.log')
        if output.exists() or log.exists():
            raise FileExistsError('Refuse duplicate attempts: '+label)
        if not validate_packet(packet)['valid']:
            raise ValueError('Packet bindings changed')
        state['current']=dict(label=label,job_id=job['id'],output=str(output))
        publish(progress,state)
        cmd=[sys.executable,'-u',str(SOURCE/'scripts/prepare_static_experiments.py'),'run',
             '--packet',str(packet_path),'--job',job['id'],'--output',str(output),'--execute']
        with log.open('x') as stream:
            done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        path=output/'run.json'
        if not path.exists():
            state.update(status='STOPPED_INFRASTRUCTURE_ERROR')
            publish(progress,state)
            raise RuntimeError('No raw result: '+label)
        result=json.loads(path.read_text())
        if result['experiment_job_id']!=job['id'] or result['transport_config']['random_seed']!=job['seed']:
            raise ValueError('Raw identity differs')
        utility=result['evaluation']['utility']
        row=dict(label=label,job_id=job['id'],raw=receipt(path),exit_code=done.returncode,
            status=result['status'],eligible=utility['eligible'],U=utility['U'],errors=result['errors'])
        state['jobs'].append(row)
        print(json.dumps(row),flush=True)
        publish(progress,state)
    state.pop('current',None)
    state.update(status='ALL_PREDECLARED_ATTEMPTS_COLLECTED',all_attempts_eligible=all(j['eligible'] for j in state['jobs']) and False)
    publish(progress,state)

if __name__=='__main__':
    main()
