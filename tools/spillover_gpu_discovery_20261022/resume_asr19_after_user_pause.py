"""Resume only conditions interrupted or not started when the user paused.

The selected attempt is declared before collection; run scores never select it.
Previously failed seed52 attempts cannot be replaced by this driver.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
BASE=STUDY/'development-asr19-replication-v1'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.static_study import validate_packet

def receipt(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def save_new(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n')

def publish(path,value):
    tmp=path.with_suffix('.resume.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)

def main():
    progress=BASE/'remaining-queue-progress.json'
    paused=json.loads(progress.read_text())
    packet_path=BASE/'experiment-packet.json'
    packet=json.loads(packet_path.read_text())
    if paused['status']!='PAUSED_BY_USER' or not validate_packet(packet)['valid']:
        raise ValueError('Expected paused queue with unchanged packet')
    interrupted_label='asr19_n2_specialist_share_1p0_seed53'
    old=BASE/'runs'/(interrupted_label+'-attempt1')/'run.json'
    raw=json.loads(old.read_text())
    if (not any('KeyboardInterrupt' in e for e in raw['errors'])
            or not raw['measurement_window'].get('aborted') or raw['status']!='failed'):
        raise ValueError('Only an explicitly interrupted window may be recollected')
    jobs={j['labels'][0]:j for j in packet['jobs']}
    plan=[(interrupted_label,2),('asr19_n2_specialist_share_0p875_seed53',1)]
    for label,attempt in plan:
        output=BASE/'runs'/(label+'-attempt'+str(attempt))
        if output.exists() or output.with_suffix('.log').exists():
            raise FileExistsError(output)
    launch=BASE/'resume-after-user-pause-v1'
    launch.mkdir(exist_ok=False)
    save_new(launch/'paused-progress.json',paused)
    declaration=dict(schema='gsdeploy.user_pause_resume_declaration.v1',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        packet=receipt(packet_path),user_pause_receipt=receipt(STUDY/'user-pause-20261022.json'),
        interrupted_run=receipt(old),source=receipt(Path(__file__)),
        selected_attempts=[dict(label=label,job_id=jobs[label]['id'],attempt=attempt,
            output=str(BASE/'runs'/(label+'-attempt'+str(attempt)))) for label,attempt in plan],
        selection_policy='Declared before seeing resumed results; only user-interrupted condition recollected; seed52 failures remain selected.',
        stability_rules_changed=False)
    save_new(launch/'declaration.json',declaration)
    state={**paused,'status':'RUNNING','resume_declaration':receipt(launch/'declaration.json')}
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    for label,attempt in plan:
        job=jobs[label]
        output=BASE/'runs'/(label+'-attempt'+str(attempt))
        log=BASE/'runs'/(label+'-attempt'+str(attempt)+'.log')
        if not validate_packet(packet)['valid']:
            raise ValueError('Packet bindings changed')
        state['current']=dict(label=label,job_id=job['id'],attempt=attempt,output=str(output))
        publish(progress,state)
        cmd=[sys.executable,'-u',str(SOURCE/'scripts/prepare_static_experiments.py'),'run',
            '--packet',str(packet_path),'--job',job['id'],'--output',str(output),'--execute']
        with log.open('x') as stream:
            done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        path=output/'run.json'
        if not path.exists():
            state['status']='STOPPED_INFRASTRUCTURE_ERROR'
            publish(progress,state)
            raise RuntimeError('Missing raw result: '+label)
        result=json.loads(path.read_text())
        if result['experiment_job_id']!=job['id'] or result['transport_config']['random_seed']!=53:
            raise ValueError('Execution identity changed')
        utility=result['evaluation']['utility']
        row=dict(label=label,job_id=job['id'],raw=receipt(path),exit_code=done.returncode,
            status=result['status'],eligible=utility['eligible'],U=utility['U'],errors=result['errors'],attempt=attempt)
        state['jobs'].append(row)
        if label==interrupted_label:
            save_new(BASE/'attempt-selection.json',dict(schema='gsdeploy.user_pause_attempt_selection.v1',
                target_job_id=job['id'],selected_run=receipt(path),previous_run=receipt(old),
                user_pause_receipt=receipt(STUDY/'user-pause-20261022.json'),
                reason='user_requested_pause_retry',declaration=receipt(launch/'declaration.json'),
                selected_before_observation=True,selection_depends_on_U=False))
        publish(progress,state)
        print(json.dumps(row),flush=True)
    state.pop('current',None)
    state['status']='ALL_PREDECLARED_ATTEMPTS_COLLECTED'
    state['all_attempts_eligible']=False
    state['all_selected_attempts_eligible']=False
    publish(progress,state)
    save_new(launch/'completion.json',state)

if __name__=='__main__':
    main()
