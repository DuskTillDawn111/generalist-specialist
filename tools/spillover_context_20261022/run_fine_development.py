"""Serial, no-retry collection for predeclared development pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.static_study import validate_packet


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publish(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def gpu_idle(device,expected_uuid):
    identity=subprocess.run(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=5)
    if identity.stdout.strip()!=expected_uuid:raise RuntimeError('GPU UUID differs from declared hardware')
    result=subprocess.run(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=5)
    if result.stdout.strip():raise RuntimeError('GPU occupied; do not overlap experiments')
    return dict(device=device,uuid=identity.stdout.strip(),compute_pids=[],unix_ns=time.time_ns())


def resources_valid(raw,expected_uuid):
    if raw.get('preflight',{}).get('gpu',{}).get('uuid')!=expected_uuid:
        raise RuntimeError('Terminal raw hardware identity differs')
    memory=raw.get('memory_evidence',{});energy=raw.get('energy_evidence',{})
    if memory.get('measured') is not True or memory.get('resource_valid') is not True:
        raise RuntimeError('Memory resource evidence failed; preserve raw and stop')
    if energy.get('coverage_valid') is not True:
        raise RuntimeError('Energy coverage failed; preserve raw and stop')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    packet=json.loads(args.packet.read_text());validation=validate_packet(packet)
    if not validation['valid']:raise ValueError(str(validation))
    if not args.execute:
        print(json.dumps(validation));return 0
    if args.output is None:parser.error('--output required for execution')
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(schema='gsdeploy.fine_route_development_queue.v1',status='RUNNING',
        packet=receipt(args.packet),source=receipt(__file__),jobs=[],policy='No retries. Retain ineligible observations and complete declared paired conditions. Stop infrastructure/resource errors.')
    progress=args.output/'queue-progress.json';publish(progress,state)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    process=None
    devices={w['env'].get('CUDA_VISIBLE_DEVICES','0') for w in packet['workers'].values()}
    if len(devices)!=1:raise ValueError('One declared worker GPU required')
    device=next(iter(devices));expected_uuid=packet['config']['catalog']['measurement_context']['gpu_uuid']
    def interrupted(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for job in packet['jobs']:
            assert receipt(args.packet)==state['packet'] and receipt(__file__)==state['source']
            if not validate_packet(packet)['valid']:raise ValueError('Packet binding changed')
            boundary=gpu_idle(device,expected_uuid);label=job['labels'][0];output=args.output/(label+'-attempt1');log=args.output/(label+'-attempt1.log')
            if output.exists() or log.exists():raise FileExistsError(label)
            state['current']=dict(job_id=job['id'],label=label,output=str(output.resolve()),started_unix_ns=time.time_ns(),gpu_boundary=boundary)
            publish(progress,state)
            command=[sys.executable,'-u',str(SOURCE/'scripts/prepare_static_experiments.py'),'run',
                '--packet',str(args.packet.resolve()),'--job',job['id'],'--output',str(output.resolve()),'--execute']
            with log.open('x') as stream:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['current']['pid']=process.pid;publish(progress,state)
                while process.poll() is None:
                    state['heartbeat_unix_ns']=time.time_ns();publish(progress,state);time.sleep(5)
            rawpath=output/'run.json'
            if not rawpath.exists():raise RuntimeError('Missing raw terminal; inspect '+str(log))
            raw=json.loads(rawpath.read_text());assert raw['experiment_job_id']==job['id']
            utility=raw['evaluation']['utility']
            row=dict(job_id=job['id'],label=label,raw=receipt(rawpath),exit_code=process.returncode,
                status=raw['status'],eligible=utility['eligible'],U=utility['U'],errors=raw['errors'])
            state['jobs'].append(row);publish(progress,state);print(json.dumps(row),flush=True)
            if raw['status'] not in {'completed','completed_with_failures'}:
                raise RuntimeError('Run-level failure requires diagnosis; independent conditions remain declared')
            resources_valid(raw,expected_uuid)
            gpu_idle(device,expected_uuid)
        state['status']='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED'
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',error=f'{type(exc).__name__}: {exc}')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        state.pop('current',None);publish(progress,state)
    print(json.dumps(state),flush=True)
    return 0 if state['status']=='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED' else 1


if __name__=='__main__':raise SystemExit(main())
