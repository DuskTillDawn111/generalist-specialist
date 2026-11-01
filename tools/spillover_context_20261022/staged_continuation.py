"""Prepare or execute one finite, serial continuation of declared experiments.

Preparation performs CPU file checks only. Execution waits for all four fine
terminals, audits them, then runs the already declared strong-control packet.
No retries, live raw inspection, utility-dependent scheduling, or packet edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/static-spillover-study-20261022'
TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
TERMINAL='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED'


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publish(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def queue_gate(queue,packet,packet_receipt,expected_count,runner_receipt=None):
    if len(packet['jobs'])!=expected_count:raise ValueError('Unexpected declared job count')
    if queue['packet']!=packet_receipt:raise ValueError('Queue packet binding changed')
    if runner_receipt is not None and queue.get('source')!=runner_receipt:raise ValueError('Queue runner binding changed')
    expected={j['id'] for j in packet['jobs']};rows=queue['jobs'];ids=[j['job_id'] for j in rows]
    if len(set(ids))!=len(ids) or not set(ids)<=expected:raise ValueError('Unexpected/duplicate terminal identity')
    if queue['status']=='RUNNING':return False
    if queue['status']!=TERMINAL:raise RuntimeError('Queue stopped: '+str(queue['status']))
    if len(rows)!=expected_count or set(ids)!=expected:raise ValueError('Terminal queue missing declared jobs')
    for row in rows:
        if row['exit_code']!=0 or row['status'] not in {'completed','completed_with_failures'}:
            raise RuntimeError('Infrastructure/run-level terminal failure')
        path=Path(row['raw']['path'])
        if path.name!='run.json' or receipt(path)!=row['raw']:raise ValueError('Terminal receipt changed')
    return True


def audit_gate(report,expected_ids,status=None):
    if status is not None and report.get('status')!=status:raise RuntimeError('Terminal audit incomplete')
    jobs=report.get('jobs',report.get('observations',[]))
    if {j['job_id'] for j in jobs}!=set(expected_ids) or len(jobs)!=len(expected_ids):
        raise RuntimeError('Audit did not cover exact declared jobs')
    for job in jobs:
        if job.get('audit_status')!='VERIFIED_TERMINAL_EVIDENCE':raise RuntimeError('Unverified terminal')
        if job.get('raw_status') not in {'completed','completed_with_failures'}:raise RuntimeError('Terminal run failed')
        if not all(job.get(key) is True for key in ['memory_pass','energy_pass','shutdown_pass']):
            raise RuntimeError('Terminal resource or shutdown audit failed')
    # Eligibility and U are deliberately absent from the scheduling gate.


def freeze_valid(request):
    for expected in request['frozen_files']:
        if receipt(expected['path'])!=expected:raise ValueError('Frozen file changed: '+expected['path'])
    if str(Path(sys.executable).resolve())!=request['python']['path'] or receipt(sys.executable)!=request['python']:
        raise ValueError('Python interpreter changed')


def prepare(output,fine_packet,strong_packet,fine_queue):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite prepared continuation')
    packets=[json.loads(Path(p).read_text()) for p in [fine_packet,strong_packet]]
    if [len(p['jobs']) for p in packets]!=[4,2]:raise ValueError('Expected fine4 and strong2 declarations')
    if [(j['seed'],bool(j['plan']['routing']['text'].get('generalist_2'))) for j in packets[0]['jobs']]!=[(107,False),(107,True),(108,True),(108,False)]:
        raise ValueError('Unexpected fine declarations')
    if [j['seed'] for j in packets[1]['jobs']]!=[107,108]:raise ValueError('Unexpected strong seeds')
    for job in packets[1]['jobs']:
        if job['plan']['deployment']!=dict(text=2,image=1,ocr=1,asr=1):raise ValueError('Unexpected strong structure')
    validations=[]
    for packet_path in [fine_packet,strong_packet]:
        checked=subprocess.run([sys.executable,str(TOOLS/'run_fine_development.py'),'--packet',str(Path(packet_path).resolve())],
            cwd=ROOT,text=True,capture_output=True,check=True,timeout=60)
        validation=json.loads(checked.stdout)
        if validation.get('valid') is not True:raise ValueError('Frozen packet validation failed')
        validations.append(validation)
    files={Path(__file__).resolve(),Path(fine_packet).resolve(),Path(strong_packet).resolve()}
    files.update(TOOLS/name for name in ['analyze_fine_development.py','run_fine_development.py','compare_development.py','plot_development.py'])
    # Conservative source closure includes every Python file in the three tool
    # packages imported by terminal audit, plus every frozen execution module.
    for directory in [ROOT/'tools/spillover_background_20261022',ROOT/'tools/spillover_capacity_20261022',SOURCE]:
        files.update(p.resolve() for p in directory.rglob('*.py'))
    for packet in packets:
        for bound in packet['bindings']:
            path=Path(bound['path']).resolve()
            if receipt(path)!=bound:raise ValueError('Packet dependency changed: '+str(path))
            files.add(path)
    frozen=[receipt(p) for p in sorted(files)];python=receipt(sys.executable)
    execute=output/'execution';audit=execute/'fine-terminal-audit';strong=execute/'strong-controls-gpu'
    compare=execute/'all-six-comparison';plots=execute/'figures'
    command=lambda tool,*args:[python['path'],'-u',str(TOOLS/tool),*map(str,args)]
    request=dict(schema='gsdeploy.staged_continuation.request.v1',status='PREPARED_NOT_EXECUTED',
        fine_packet=receipt(fine_packet),strong_packet=receipt(strong_packet),fine_queue=str(Path(fine_queue).resolve()),runner=receipt(TOOLS/'run_fine_development.py'),
        frozen_files=frozen,python=python,root=str(ROOT),execution=str(execute),packet_validations=validations,
        wait_seconds=5,maximum_seconds=7200,
        outputs=dict(fine_audit=str(audit),strong=str(strong),comparison=str(compare),plots=str(plots)),
        stages=[dict(name='fine_audit',command=command('analyze_fine_development.py','--packet',Path(fine_packet).resolve(),'--queue',Path(fine_queue).resolve(),'--output',audit)),
            dict(name='strong_gpu',command=command('run_fine_development.py','--packet',Path(strong_packet).resolve(),'--output',strong,'--execute')),
            dict(name='all_compare',command=command('compare_development.py','--registration',Path(fine_packet).resolve(),Path(fine_queue).resolve(),
                '--registration',Path(strong_packet).resolve(),strong/'queue-progress.json','--output',compare)),
            dict(name='plot',command=command('plot_development.py','--comparison',compare/'report.json','--output',plots))],
        policy=dict(no_retries=True,no_live_raw=True,no_utility_dependent_scheduling=True,
            failure='STOP_WITH_EVIDENCE',deadline='7200s from execute start, including wait and all stages',
            source_closure='Frozen packets/bindings, all Python source in execution-source-v5 and background/capacity tool directories, four invoked context tools and supervisor; Python executable. External installed package contents are not rehashed.'))
    freeze_valid(request)
    output.mkdir(parents=True,exist_ok=False)
    publish(output/'request.json',request)
    snapshots=output/'source-snapshot';snapshots.mkdir()
    for path in sorted(files):
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            target=snapshots/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    return receipt(output/'request.json')


class Supervisor:
    def __init__(self,request):
        self.request=request;self.output=Path(request['execution']);self.process=None
        self.deadline=time.monotonic()+request['maximum_seconds'];self.state=dict(
            schema='gsdeploy.staged_continuation.progress.v1',status='STARTING',stages=[],started_unix_ns=time.time_ns())

    def checkpoint(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Finite continuation deadline reached')
        self.state['heartbeat_unix_ns']=time.time_ns();publish(self.output/'progress.json',self.state)

    def stop_child(self):
        if self.process is not None and self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=10)

    def run_stage(self,stage):
        freeze_valid(self.request);self.checkpoint();self.state['status']='RUNNING_'+stage['name'].upper()
        event=dict(name=stage['name'],command=stage['command'],started_unix_ns=time.time_ns());self.state['stages'].append(event)
        with (self.output/(stage['name']+'.log')).open('x') as stream:
            self.process=subprocess.Popen(stage['command'],cwd=self.request['root'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            event['pid']=self.process.pid
            while self.process.poll() is None:self.checkpoint();time.sleep(self.request['wait_seconds'])
        event.update(exit_code=self.process.returncode,ended_unix_ns=time.time_ns());self.checkpoint()
        if self.process.returncode!=0:raise RuntimeError('Stage failed: '+stage['name'])
        freeze_valid(self.request)

    def wait_fine(self):
        packet=json.loads(Path(self.request['fine_packet']['path']).read_text())
        self.state['status']='WAITING_FINE_TERMINALS'
        while True:
            self.checkpoint();queue_path=Path(self.request['fine_queue']);content=queue_path.read_bytes();queue=json.loads(content)
            if queue_gate(queue,packet,self.request['fine_packet'],4,self.request['runner']):
                freeze_valid(self.request)
                (self.output/'fine-terminal-queue-snapshot.json').write_bytes(content);return packet
            time.sleep(self.request['wait_seconds'])

    def work(self):
        packet=self.wait_fine();self.run_stage(self.request['stages'][0])
        fine_report=Path(self.request['outputs']['fine_audit'])/'report.json'
        audit_gate(json.loads(fine_report.read_text()),[j['id'] for j in packet['jobs']],'ALL_FOUR_TERMINALS_AUDITED')
        self.state['fine_audit']=receipt(fine_report);self.checkpoint()
        self.run_stage(self.request['stages'][1])
        strong=json.loads(Path(self.request['strong_packet']['path']).read_text())
        queue_path=Path(self.request['outputs']['strong'])/'queue-progress.json';queue=json.loads(queue_path.read_text())
        if not queue_gate(queue,strong,self.request['strong_packet'],2,self.request['runner']):raise RuntimeError('Strong queue still running after exit')
        self.run_stage(self.request['stages'][2])
        comparison=Path(self.request['outputs']['comparison'])/'report.json'
        audit_gate(json.loads(comparison.read_text()),[j['id'] for p in [packet,strong] for j in p['jobs']])
        self.run_stage(self.request['stages'][3]);self.state.update(status='COMPLETE',comparison=receipt(comparison))

    def execute(self):
        self.output.mkdir(parents=True,exist_ok=False)
        previous={sig:signal.getsignal(sig) for sig in [signal.SIGTERM,signal.SIGINT]}
        def interrupted(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
        for sig in previous:signal.signal(sig,interrupted)
        try:
            freeze_valid(self.request)
            for path in self.request['outputs'].values():
                if Path(path).exists():raise FileExistsError('Stage output already exists: '+path)
            self.work()
        except (Exception,KeyboardInterrupt) as exc:
            self.state.update(status='STOPPED_NO_RETRY',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            try:self.stop_child()
            finally:
                self.state['ended_unix_ns']=time.time_ns();publish(self.output/'progress.json',self.state)
                for sig,handler in previous.items():signal.signal(sig,handler)
        return 0 if self.state['status']=='COMPLETE' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--fine-packet',type=Path,default=BASE/'text185-fine-development-v1/experiment-packet.json')
    prep.add_argument('--strong-packet',type=Path,default=BASE/'text185-strong-controls-v1/experiment-packet.json')
    prep.add_argument('--fine-queue',type=Path,default=BASE/'text185-fine-development-v1/gpu-runs-v1/queue-progress.json')
    run=sub.add_parser('execute');run.add_argument('--request',type=Path,required=True);run.add_argument('--request-sha256',required=True)
    args=p.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args.output,args.fine_packet,args.strong_packet,args.fine_queue)));return 0
    if receipt(args.request)['sha256']!=args.request_sha256:raise ValueError('Request SHA mismatch')
    return Supervisor(json.loads(args.request.read_text())).execute()


if __name__=='__main__':raise SystemExit(main())
