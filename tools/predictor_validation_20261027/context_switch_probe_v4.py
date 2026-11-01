"""Exclusive small two-process CUDA context-switch capability probe.

No driver, MPS, permission, or resource configuration is changed. Parent uses
one Nsight session and two separate CUDA-owning processes. Output proves only
the observed trace coverage, never same-SM concurrent execution.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time


def receipt(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=output/'probe-frozen.py';shutil.copyfile(__file__,source)
    result=dict(schema='gsdeploy.context_switch_capability_probe.v4',source=receipt(source),
        nsys=shutil.which('nsys'),python=sys.executable,worker_count=2,active_seconds=0.5,
        matrix_size=1024,owned_gpu_bytes='three float32 matrices per process (~12 MiB), plus CUDA/runtime workspaces',
        environment={name:'1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        expected_scope='two independent CUDA contexts; single session CUDA+NVTX+GPU context-switch records',
        free_disk_reserve_bytes=12*2**30,process_timeout_seconds=180,
        permission_changes=False,MPS_configuration_changed=False,no_retries=True,GPU_started=False)
    if result['nsys'] is None:raise ValueError('nsys required')
    save(output/'plan.json',result);return result


def emit(value):print(json.dumps(value),flush=True)


def worker(index):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    a=torch.ones((1024,1024),device='cuda');b=torch.ones_like(a);c=torch.empty_like(a)
    torch.mm(a,b,out=c);torch.cuda.synchronize()
    emit(dict(event='READY',pid=os.getpid(),index=index,device=torch.cuda.get_device_name(0),clock='host_perf_counter_ns'))
    if sys.stdin.readline().strip()!='GO':raise ValueError('Explicit GO required')
    begin=time.perf_counter_ns();count=0
    torch.cuda.nvtx.range_push('GS_CONTEXT_PROBE:'+str(index))
    while time.perf_counter_ns()-begin<500_000_000:
        for _ in range(8):torch.mm(a,b,out=c);count+=1
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop();end=time.perf_counter_ns()
    emit(dict(event='DONE',pid=os.getpid(),index=index,start_perf_ns=begin,end_perf_ns=end,
              matrix_multiplies=count,observed_scalar=float(c[0,0].item()),clock='host_perf_counter_ns'))


def cohort():
    workers=[];events=[]
    try:
        for index in (0,1):
            child=subprocess.Popen([sys.executable,__file__,'worker','--index',str(index)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            workers.append(child)
        for child in workers:
            event=json.loads(child.stdout.readline())
            if event.get('event')!='READY' or event.get('pid')!=child.pid:raise ValueError('Owned child READY required')
            events.append(event)
        for child in workers:child.stdin.write('GO\n');child.stdin.flush()
        for child in workers:
            event=json.loads(child.stdout.readline());events.append(event)
            if event.get('event')!='DONE' or event.get('pid')!=child.pid or event.get('observed_scalar')!=1024.:
                raise ValueError('Owned child actual CUDA work required')
            if child.wait(timeout=30)!=0:raise ValueError('CUDA child failed')
        emit(dict(event='COHORT_COMPLETED',children=events))
    finally:
        for child in workers:
            if child.poll() is None:child.kill();child.wait()


def execute(run):
    run=Path(run).resolve();plan=json.loads((run/'plan.json').read_text())
    if receipt(plan['source']['path'])!=plan['source']:raise ValueError('Frozen probe changed')
    if (run/'execution.json').exists() or list(run.glob('trace*')):raise FileExistsError('Never retry actual probe')
    if shutil.disk_usage(run).free<plan['free_disk_reserve_bytes']:raise ValueError('Disk reserve insufficient')
    command=[plan['nsys'],'profile','--trace=cuda,nvtx','--gpuctxsw=true','--sample=none',
        '--cpuctxsw=none','--cuda-memory-usage=false','--force-overwrite=false',
        '--output='+str(run/'trace'),plan['python'],plan['source']['path'],'cohort']
    lock=run.parents[1]/'.gpu-experiment.lock'
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        start=time.time_ns();status='UNKNOWN';code=None
        try:
            with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
                process=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True,
                    env={**os.environ,**plan['environment']})
                try:code=process.wait(timeout=plan['process_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    status='TIMEOUT_NO_RETRY'
                else:status='COLLECTED' if code==0 else 'FAILED_NO_RETRY'
        finally:
            result=dict(schema='gsdeploy.context_switch_probe_execution.v4',status=status,return_code=code,
                command=command,collection_started_unix_ns=start,collection_ended_unix_ns=time.time_ns(),
                plan=receipt(run/'plan.json'),source=plan['source'],GPU_started=True,
                same_sm_concurrency_proven=False,permission_changes=False)
            save(run/'execution.json',result);fcntl.flock(handle,fcntl.LOCK_UN)
    return result


def review(run):
    run=Path(run).resolve();execution=json.loads((run/'execution.json').read_text())
    output=run/'review';output.mkdir(exist_ok=False)
    issues=[];tables={};context_rows=[];kernels=[]
    if execution['status']=='COLLECTED':
        sql=output/'trace.sqlite'
        result=subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=false','--output='+str(sql),str(run/'trace.nsys-rep')],capture_output=True,text=True)
        (output/'export-stdout.log').write_text(result.stdout);(output/'export-stderr.log').write_text(result.stderr)
        if result.returncode:issues.append('SQLITE_EXPORT_FAILED')
        else:
            connection=sqlite3.connect('file:'+str(sql)+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
            for row in connection.execute('select name from sqlite_master where type="table"'):
                name=row['name']
                if any(word in name.upper() for word in ('CONTEXT','SWITCH','SCHED')):
                    tables[name]=dict(columns=[r['name'] for r in connection.execute('pragma table_info("'+name+'")')],
                        rows=[dict(r) for r in connection.execute('select * from "'+name+'"')])
            if 'TARGET_INFO_CUDA_CONTEXT_INFO' in tables:context_rows=tables['TARGET_INFO_CUDA_CONTEXT_INFO']['rows']
            if connection.execute("select count(*) from sqlite_master where name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone()[0]:
                kernels=[dict(r) for r in connection.execute('select globalPid,contextId,count(*) count from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid,contextId')]
            connection.close()
    else:issues.append('GPU_CONTEXT_SWITCH_CAPTURE_DID_NOT_COMPLETE')
    switch_tables=[name for name,doc in tables.items() if ('SWITCH' in name.upper() or 'SCHED' in name.upper()) and doc['rows']]
    if not switch_tables:issues.append('NO_GPU_CONTEXT_SWITCH_RECORDS_VERIFIED')
    result=dict(schema='gsdeploy.context_switch_probe_review.v4',status='RECORDS_REQUIRE_INTERPRETATION' if not issues else 'UNKNOWN',
        issues=issues,execution=receipt(run/'execution.json'),context_tables=tables,kernel_groups=kernels,
        candidate_switch_tables=switch_tables,distinct_context_processes=len({r['processId'] for r in context_rows}),
        evaluator=receipt(__file__),same_sm_concurrency_proven=False,
        interpretation='Context-switch records require engine/device/PID attribution; overlapping kernel lifetimes alone do not prove same-SM simultaneous execution.')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('execute');p.add_argument('--run',required=True);p.add_argument('--execute',action='store_true',required=True)
    p=sub.add_parser('review');p.add_argument('--run',required=True)
    p=sub.add_parser('worker');p.add_argument('--index',type=int,required=True)
    sub.add_parser('cohort');args=parser.parse_args()
    if args.op=='worker':worker(args.index)
    elif args.op=='cohort':cohort()
    else:
        result=prepare(args.output) if args.op=='prepare' else execute(args.run) if args.op=='execute' else review(args.run)
        print(json.dumps({k:v for k,v in result.items() if k not in ('context_tables',)},indent=2))


if __name__=='__main__':main()
