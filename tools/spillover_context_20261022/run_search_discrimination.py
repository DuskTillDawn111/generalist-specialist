"""Serial native top-two collection followed by terminal-only CPU auditing.

No retry or adaptive conditions. Default validates the prepared packet on CPU.
Execution is explicit; every phase has a persistent atomic progress receipt.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
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
HERE=Path(__file__).resolve().parent


def receipt(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def publish(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');args=p.parse_args()
    packet=args.packet.resolve();env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    base=[sys.executable,'-u',str(HERE/'run_fine_development.py'),'--packet',str(packet)]
    if not args.execute:
        return subprocess.run(base,cwd=ROOT,env=env,check=False).returncode
    if args.output is None:p.error('--output required for execution')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(schema='gsdeploy.search_discrimination_execution.v1',status='RUNNING',phase='GPU_COLLECTION',
        started_utc=datetime.now(timezone.utc).isoformat(),packet=receipt(packet),
        sources=[receipt(v) for v in [__file__,HERE/'run_fine_development.py',HERE/'compare_development.py',
            HERE/'analyze_fine_development.py',HERE/'assess_gain_uncertainty.py']],
        collection=None,audit=None,uncertainty=None,no_retry=True,PID=os.getpid())
    progress=out/'progress.json';publish(progress,state);process=None
    def execute(command,log):
        nonlocal process
        for r in [state['packet'],*state['sources']]:
            if receipt(r['path'])!=r:raise ValueError('Pinned execution source changed: '+r['path'])
        with log.open('x') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['current_process']=dict(pid=process.pid,command=command,log=str(log));publish(progress,state)
            while process.poll() is None:
                state['heartbeat_unix_ns']=time.time_ns();publish(progress,state);time.sleep(5)
        state.pop('current_process',None);publish(progress,state);return process.returncode
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        queue=out/'gpu-runs'/'queue-progress.json'
        code=execute(base+['--output',str(out/'gpu-runs'),'--execute'],out/'gpu-collection.log')
        state['collection']=dict(exit_code=code,queue=receipt(queue) if queue.exists() else None)
        if code!=0:
            state['status']='COLLECTION_STOPPED_FOR_DIAGNOSIS_NO_RETRY'
            return 1
        registered=json.loads(queue.read_text());expected={j['id'] for j in json.loads(packet.read_text())['jobs']}
        if registered['status']!='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED' or {j['job_id'] for j in registered['jobs']}!=expected:
            raise ValueError('Queue returned without all declared terminal receipts')
        state['phase']='TERMINAL_CPU_AUDIT';publish(progress,state)
        code=execute([sys.executable,'-u',str(HERE/'compare_development.py'),'--registration',str(packet),str(queue),
            '--output',str(out/'terminal-audit')],out/'terminal-audit.log')
        report=out/'terminal-audit'/'report.json'
        state['audit']=dict(exit_code=code,report=receipt(report) if report.exists() else None)
        if code!=0:raise RuntimeError('Terminal CPU audit process failed; raw results preserved')
        audit=json.loads(report.read_text())
        if len(audit['observations'])!=len(expected) or {r['job_id'] for r in audit['observations']}!=expected:
            raise RuntimeError('Terminal audit did not cover all declared job identities')
        if any(r['audit_status']!='VERIFIED_TERMINAL_EVIDENCE' for r in audit['observations']):
            raise RuntimeError('Terminal audit contains unresolved evidence; inspect report')
        if any(not r['utility']['eligible'] for r in audit['observations']):
            state['status']='COMPLETE_AUDITED_INELIGIBLE_CONDITIONS_RETAINED'
            return 0
        state['phase']='WHOLE_PAIR_UNCERTAINTY';publish(progress,state)
        code=execute([sys.executable,'-u',str(HERE/'assess_gain_uncertainty.py'),'--audit',str(report),
            '--output',str(out/'whole-pair-uncertainty')],out/'whole-pair-uncertainty.log')
        report2=out/'whole-pair-uncertainty'/'report.json'
        state['uncertainty']=dict(exit_code=code,report=receipt(report2) if report2.exists() else None)
        if code!=0:raise RuntimeError('Whole-pair uncertainty summary failed; terminal audit preserved')
        state.update(status='COMPLETE_COLLECTION_AUDIT_AND_PAIR_SUMMARY',phase='COMPLETE')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='STOPPED_FOR_DIAGNOSIS_NO_RETRY',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        return 1
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        state['updated_utc']=datetime.now(timezone.utc).isoformat();state.pop('current_process',None);publish(progress,state)


if __name__=='__main__':raise SystemExit(main())
