"""Append-only serial native batch. Stop on infrastructure errors, never retry."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_calibration_20261023.protocol import receipt

GPU='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'

def publish(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def gpu_idle():
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],text=True).strip()
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if uuid!=GPU or pids:raise RuntimeError('Serial GPU identity/idle boundary failed: '+uuid+' '+pids)
    return dict(uuid=uuid,pids=[],unix_ns=time.time_ns())

def execute(packet_path,output,package):
    protocol=importlib.import_module(package+'.protocol');native=importlib.import_module(package+'.native_runner')
    packet_path=Path(packet_path).resolve();packet=json.loads(packet_path.read_text())
    if not protocol.validate_packet(packet)['valid']:raise ValueError('Invalid packet')
    output=Path(output).resolve();output.mkdir(exist_ok=False,parents=True)
    state=dict(status='RUNNING',phase=packet.get('phase','calibration'),PID=os.getpid(),jobs=[],packet=receipt(packet_path),
        started_unix_ns=time.time_ns(),no_retries=True,package=package)
    def pulse():state['heartbeat_unix_ns']=time.time_ns();publish(output/'progress.json',state)
    child=None;raws=[];pulse()
    def stop(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        for job in packet['jobs']:
            if receipt(packet_path)!=state['packet'] or not protocol.validate_packet(packet)['valid']:raise ValueError('Bound inputs changed')
            boundary=gpu_idle();target=output/job['label'];log=output/(job['label']+'.log')
            state['current']=dict(label=job['label'],output=str(target),started_unix_ns=time.time_ns(),gpu_boundary=boundary)
            env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            cmd=[sys.executable,'-u',str(Path(native.__file__)), '--packet',str(packet_path),'--job',job['id'],'--output',str(target),'--execute']
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['current']['pid']=child.pid;pulse()
                while child.poll() is None:time.sleep(5);pulse()
            path=target/'run.json'
            if not path.exists():raise RuntimeError('Missing terminal: '+str(log))
            raw=json.loads(path.read_text());u=raw.get('evaluation',{}).get('utility',{})
            state['jobs'].append(dict(label=job['label'],job_id=job['id'],raw=receipt(path),exit_code=child.returncode,
                status=raw['status'],eligible=u.get('eligible'),U=u.get('U'),errors=raw.get('errors')));pulse();raws.append(path)
            if raw['experiment_job_id']!=job['id']:raise RuntimeError('Terminal identity mismatch')
            if raw['status'] not in {'completed','completed_with_failures'} or child.returncode:raise RuntimeError('Native error '+str(raw.get('errors')))
            if not raw.get('memory_evidence',{}).get('resource_valid') or not raw.get('energy_evidence',{}).get('coverage_valid'):raise RuntimeError('Invalid resource evidence')
            if not raw.get('evaluation',{}).get('dynamic_audit',{}).get('verified'):raise RuntimeError('Causal replay failed')
            gpu_idle()
        state.pop('current',None)
        analysis=importlib.import_module(package+'.analysis').analyze(raws,packet['seeds'],output/'analysis')
        state.update(status='COMPLETE',phase='COMPLETE',analysis=receipt(output/'analysis/report.json'),
            all_eligible=analysis['full_declared_table_complete'])
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());return 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        state['finished_unix_ns']=time.time_ns();pulse()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--output');p.add_argument('--execute',action='store_true')
    p.add_argument('--package',default='tools.queue_refined_calibration_20261023');a=p.parse_args()
    if not a.execute:print(importlib.import_module(a.package+'.protocol').validate_packet(json.loads(Path(a.packet).read_text())))
    else:
        if not a.output:p.error('--output required')
        raise SystemExit(execute(a.packet,a.output,a.package))
