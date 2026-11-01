"""CPU validation by default; future execution requires exactly one context ID.

Native standalone discovery only: no score, service fit, route-domain exclusion,
formal validate_plan, retry, or automatic next context.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

spec=importlib.util.spec_from_file_location('_seven_baseline_check',Path(__file__).with_name('check.py'))
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
PROTOCOL=checker.PROTOCOL
TERMINAL={'completed','error','incomplete'}


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def record(template,phase):
    return {**template,'phase':phase,'scheduled_time_ns':None,'actual_submit_time_ns':None,
        'completion_time_ns':None,'worker_latency_ms':None,'worker_queue_wait_ms':None,'status':'planned'}


def validate_response(row,now,request_timeout_seconds):
    if row['status'] in TERMINAL:
        if row['status']!='completed':raise RuntimeError('Native request failure: '+row['request_id'])
        if row['completion_time_ns'] is None:raise RuntimeError('Missing completion timestamp')
        if row['completion_time_ns']-row['scheduled_time_ns']>=request_timeout_seconds*1e9:
            raise TimeoutError('Request timeout: '+row['request_id'])
        return True
    if now-row['scheduled_time_ns']>=request_timeout_seconds*1e9:raise TimeoutError('Request timeout: '+row['request_id'])
    return False


def finite_with_support(runtime,phase,samples,records,monitor,*,progress=lambda:None,metadata=None):
    """Global pool x3 primary; only finished aliases emit same-task support.

    Primary records must all exist before release. Every alias has at most one
    outstanding request. No support admission after the last primary response;
    already admitted support must drain inside the same phase hard wall.
    """
    start=time.perf_counter_ns();meta={} if metadata is None else metadata
    cohorts=phase['primary_requests'];support=phase['support_requests'];aliases=list(cohorts)
    finite={a:[r for r in records if r['model']==a and r.get('role')=='primary'] for a in aliases}
    if any([{k:r[k] for k in t} for r,t in zip(finite[a],cohorts[a])]!=cohorts[a] or len(finite[a])!=len(cohorts[a]) for a in aliases):
        raise ValueError('Primary preregistration differs')
    indexes={a:0 for a in aliases};completed={a:0 for a in aliases};support_indexes={a:0 for a in aliases};pending={};stopping=False
    meta.update(start_time_ns=start,start_unix_ns=time.time_ns(),max_outstanding_per_alias=1,
        primary_completion_time_ns=None,support_admission_stop_time_ns=None,alias_completion_time_ns={},
        admission_policy='Own complete primary cohort first, then same-task support until global primary completion')
    try:
        while True:
            now=time.perf_counter_ns();monitor.check()
            if now-start>=phase['measurement_timeout_seconds']*1e9:raise TimeoutError('Measurement wall timeout including support drain')
            with runtime.lock:
                for alias,row in list(pending.items()):
                    if validate_response(row,now,300):
                        del pending[alias]
                        if row['role']=='primary':
                            completed[alias]+=1
                            if completed[alias]==len(cohorts[alias]):meta['alias_completion_time_ns'][alias]=row['completion_time_ns']
                if not stopping and all(completed[a]==len(cohorts[a]) for a in aliases):
                    stopping=True;meta['primary_completion_time_ns']=max(meta['alias_completion_time_ns'].values())
                    meta['support_admission_stop_time_ns']=time.perf_counter_ns()
                if not stopping:
                    for alias in aliases:
                        if alias in pending:continue
                        if indexes[alias]<len(finite[alias]):
                            row=finite[alias][indexes[alias]];indexes[alias]+=1
                        else:
                            si=support_indexes[alias]
                            if si>=len(support[alias]):raise RuntimeError('Support cap exhausted before all primary complete: '+alias)
                            row=record(support[alias][si],'measurement');records.append(row);support_indexes[alias]+=1
                        row['scheduled_time_ns']=start if indexes[alias]+support_indexes[alias]==1 else time.perf_counter_ns()
                        runtime.enqueue(row,samples[row['pool_id']]['sample']);pending[alias]=row
            progress()
            if time.perf_counter_ns()-start>=phase['measurement_timeout_seconds']*1e9:raise TimeoutError('Measurement wall timeout including snapshots')
            if stopping and not pending:break
            time.sleep(.002)
    finally:
        meta.update(end_time_ns=time.perf_counter_ns(),primary_released_by_alias=indexes,
            primary_completed_by_alias=completed,support_released_by_alias=support_indexes)
        meta['duration_seconds']=(meta['end_time_ns']-start)/1e9
    return meta


def warmup(runtime,phase,samples,records,monitor,*,progress=lambda:None,metadata=None):
    start=time.perf_counter_ns();meta={} if metadata is None else metadata
    indexes={a:0 for a in phase['warmup_requests']};pending={}
    meta.update(start_time_ns=start,start_unix_ns=time.time_ns(),max_outstanding_per_alias=1)
    try:
        while True:
            now=time.perf_counter_ns();monitor.check()
            if now-start>=phase['warmup_timeout_seconds']*1e9:raise TimeoutError('Warmup wall timeout')
            with runtime.lock:
                for alias,row in list(pending.items()):
                    if validate_response(row,now,300):del pending[alias]
                admitting=now-start<phase['warmup_seconds']*1e9
                if admitting:
                    for alias,templates in phase['warmup_requests'].items():
                        if alias in pending:continue
                        i=indexes[alias];row=record({**templates[i%len(templates)],'request_id':f'warmup:{phase["task"]}:{alias}:{i}',
                            'arrival_index':i,'warmup_sequence':i},'warmup')
                        row['scheduled_time_ns']=start if i==0 else time.perf_counter_ns()
                        records.append(row);runtime.enqueue(row,samples[row['pool_id']]['sample']);pending[alias]=row;indexes[alias]+=1
            progress()
            if time.perf_counter_ns()-start>=phase['warmup_timeout_seconds']*1e9:raise TimeoutError('Warmup wall timeout including snapshots')
            if not admitting and not pending:break
            time.sleep(.002)
    finally:
        meta.update(end_time_ns=time.perf_counter_ns(),released_by_alias=indexes)
        meta['duration_seconds']=(meta['end_time_ns']-start)/1e9
    return meta


class SafeOwnedMemory:
    """Process-owned sampling before load, through READY and final failure.

    Startup PIDs can legitimately be absent before their CUDA allocation; this
    absence is recorded. After READY any missing owned PID invalidates evidence.
    Safety-factor, foreign-PID and query-error failures latch across samples.
    """
    def __init__(self,runtime,packet):
        self.runtime=runtime;self.packet=packet;self.samples=[];self.errors=[];self.ready=False
        self.stop=threading.Event();self.thread=None;self.deadline=None

    def sample(self):
        begin=time.perf_counter_ns()
        try:
            result=subprocess.run(['nvidia-smi','-i',self.packet['device'],'--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True,timeout=2)
            memory={int(p.strip()):float(v.strip()) for p,v in (line.split(',') for line in result.stdout.splitlines() if line.strip())}
            if any(not math.isfinite(v) or v<0 for v in memory.values()):raise ValueError('Invalid process memory observation')
            # Resolve ownership after the query to cover a worker launched while
            # nvidia-smi was running. Do not iterate a mutable dictionary view.
            owned={w.process.pid for w in list(self.runtime.workers.values()) if w.process}
            missing=sorted(owned-memory.keys());foreign=sorted(memory.keys()-owned);used=sum(v for p,v in memory.items() if p in owned)
            row=dict(time_ns=(begin+time.perf_counter_ns())//2,unix_ns=time.time_ns(),owned_pids=sorted(owned),
                owned_used_mib=used,safe_used_mib=used*self.packet['safe_factor'],foreign_pids=foreign,
                missing_owned_pids=missing,all_workers_ready=self.ready,query_elapsed_seconds=(time.perf_counter_ns()-begin)/1e9)
            self.samples.append(row)
            if foreign:self.errors.append('Foreign GPU process observed')
            if row['safe_used_mib']>self.packet['gpu_memory_mib']:self.errors.append('Original safe_factor times owned memory exceeds budget')
            if self.ready and missing:self.errors.append('Owned process memory missing after READY')
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))

    def start(self):
        self.sample()
        def loop():
            while not self.stop.wait(.5):self.sample()
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()

    def check(self):
        if self.deadline is not None and time.perf_counter_ns()>=self.deadline:raise TimeoutError('Context hard wall')
        if self.runtime.errors:raise RuntimeError('Native transport error: '+str(self.runtime.errors))
        if self.errors:raise RuntimeError('Memory evidence invalid: '+self.errors[0])

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():self.errors.append('Memory sampling thread did not stop')
        self.sample()


def gpu_idle(packet):
    uuid=subprocess.run(['nvidia-smi','-i',packet['device'],'--query-gpu=uuid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=5).stdout.strip()
    if uuid!=packet['gpu_uuid']:raise RuntimeError('GPU UUID differs from immutable packet')
    apps=subprocess.run(['nvidia-smi','-i',packet['device'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],
        text=True,capture_output=True,check=True,timeout=5).stdout.strip()
    if apps:raise RuntimeError('GPU compute processes present; never overlap experiments')
    return dict(uuid=uuid,compute_pids=[],unix_ns=time.time_ns())


def frozen_api(packet):
    source=Path(packet['frozen_source']);sys.path.insert(0,str(source))
    names=['gsdeploy.config','gsdeploy.runtime.runner','gsdeploy.evidence','gsdeploy.energy','gsdeploy.metrics']
    modules=[importlib.import_module(n) for n in names]
    if any(not Path(m.__file__).resolve().is_relative_to(source) for m in modules):raise ValueError('Fresh process with frozen imports required')
    return modules


def cpu_launcher_checks(packet):
    """Explicit CPU-only child checks. This function never uses --execute."""
    workers=json.loads(Path(packet['workers']).read_text());results={}
    for family in ('generalist_1','generalist_2','generalist_3'):
        entry=workers[family];command=['--check' if x=='--execute' else x for x in entry['command']]
        if '--check' not in command or '--execute' in command:raise ValueError('CPU launcher mode missing')
        command+=['--pool',packet['samples']]
        env={**os.environ,**entry.get('env',{}),**{k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
        checked=subprocess.run(command,cwd=entry.get('cwd'),env=env,text=True,capture_output=True,timeout=300)
        result=dict(command=command,exit_code=checked.returncode,stdout=checked.stdout,stderr=checked.stderr)
        if checked.returncode:raise RuntimeError('CPU launcher check failed: '+family+' '+checked.stderr)
        parsed=json.loads(checked.stdout.strip().splitlines()[-1])
        if parsed.get('gpu_started') is not False or parsed.get('status')!='CPU_READY_GPU_UNVERIFIED' or parsed.get('inputs_checked')!=160:
            raise ValueError('CPU launcher did not attest full pool CPU-only check')
        results[family]=result
    return dict(GPU_started=False,checks=results,current_device_verified=False)


def execute_context(packet,job,output,packet_sha):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    phases=[dict(task=p['task'],status='NOT_STARTED_UNKNOWN',request_records=[record(r,'measurement') for rows in p['primary_requests'].values() for r in rows],
        warmup_request_records=[],warmup_window={},measurement_window={}) for p in job['phases']]
    raw=dict(schema='gsdeploy.missing_state_baseline.raw.v1',protocol=PROTOCOL,packet_sha256=packet_sha,job=job,
        status='starting',errors=[],phases=phases,priming_request_records=[record(r,'priming') for r in job['priming_requests']],
        worker_ready={},shutdown={},replica_to_family=job['aliases'],U=None,formal_stability=None,rho=None,
        evaluation={'utility':{'U':None,'eligible':False,'reason':'STANDALONE_BASELINE_DISCOVERY_NOT_FORMAL_UTILITY'}},
        formal_feasibility='UNKNOWN',domain_pruning=False,exact_failure_scope=job['exact_failure_scope'],
        snapshot_policy='At most one live snapshot every5s plus phase changes; overhead included in hard walls')
    runtime=monitor=power=None;api=None;startup=None;startup_end=None;system=None;last_snapshot=0;last_status=None
    def checkpoint(force=False):
        nonlocal last_snapshot,last_status
        now=time.monotonic()
        if not force and raw['status']==last_status and now-last_snapshot<5:return
        raw['last_checkpoint_unix_ns']=time.time_ns()
        if runtime:
            with runtime.lock:encoded=json.dumps(raw,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        else:encoded=json.dumps(raw,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        tmp=output/'live.json.tmp';tmp.write_text(encoded);tmp.replace(output/'live.json')
        atomic_json(output/'progress.json',dict(status=raw['status'],unix_ns=time.time_ns(),
            phases=[dict(task=p['task'],status=p['status'],counts=dict(Counter(r['status'] for r in p['request_records']))) for p in phases]))
        last_snapshot=now;last_status=raw['status']
    checkpoint()
    active=None
    try:
        checker.validate_packet(packet)
        api=frozen_api(packet);cfg,runner,evidence,energy,metrics=api
        system=cfg.SystemConfig.load(packet['system_config']);workers=json.loads(Path(packet['workers']).read_text())
        # Native identity only; UNKNOWN service is never replaced to pass the
        # production validate_plan or its rho/stability/resource gates.
        coverage={'deployment':job['resident_deployment'],'routing':{t:{job['model']:1} for t in job['full_resident_supported_tasks']}}
        raw['native_identity']=evidence.verify_quality_evidence(system,coverage,workers,packet['admission'])
        if not raw['native_identity'].get('verified'):raise RuntimeError('Native identity coverage failed')
        aliases=job['aliases'];task=job['phases'][0]['task']
        config=runner.MeasurementConfig(deployment={a:1 for a in aliases},workload_rps={task:1},
            routing_rps={task:{a:1/len(aliases) for a in aliases}},warmup_seconds=30,measurement_seconds=600,
            startup_timeout_seconds=300,drain_timeout_seconds=300,shutdown_timeout_seconds=10,max_pending_per_model=1,
            random_seed=0,routing_source=PROTOCOL)
        config.validate({a:system.catalog['models'][f]['supported_tasks'] for a,f in aliases.items()})
        registry=runner.registry_for(system,aliases,'native',workers)
        if any(e['env']['CUDA_VISIBLE_DEVICES']!=packet['device'] for e in registry.values()):raise ValueError('Native launcher device differs')
        raw.update(config=system.as_dict(),worker_registry=registry,transport_config=config.as_dict(),
            transport_config_scope='Only transport metadata; actual workload is exact closed-loop priming/warmup/primary/support rosters')
        raw['gpu_before_worker_start']=gpu_idle(packet)
        runtime=runner.DeploymentRuntime(config,registry);monitor=SafeOwnedMemory(runtime,packet)
        monitor.deadline=time.perf_counter_ns()+round(job['timeouts']['context_hard_wall_seconds']*1e9)
        power=runner.PowerMonitor(device=packet['device']);power.sample();startup=time.perf_counter_ns();power.start();monitor.start();monitor.check()
        # Reuse frozen native startup while checking safe memory and context wall
        # at each startup event wait (0.1s), including partially loaded workers.
        original_get=runtime.events.get;loading=True
        def guarded_get(*args,**kwargs):
            if loading:monitor.check()
            result=original_get(*args,**kwargs)
            if loading:monitor.check()
            return result
        runtime.events.get=guarded_get
        raw['status']='loading';checkpoint();runtime.start();loading=False
        startup_end=time.perf_counter_ns();raw['worker_ready']=dict(runtime.ready)
        for alias,family in aliases.items():
            entry=workers[family];spec_path=Path(entry.get('spec') or entry['command'][entry['command'].index('--spec')+1])
            spec_data=json.loads(spec_path.read_text());ready=runtime.ready[alias]
            if (ready.get('worker')!=spec_data['candidate']['slot'] or ready.get('model')!=spec_data['candidate']['model']
                or ready.get('spec_sha256')!=checker.sha(spec_path) or set(ready.get('tasks',[]))!=set(job['full_resident_supported_tasks'])):
                raise RuntimeError('Native READY identity mismatch: '+alias)
        monitor.ready=True;monitor.sample();monitor.check()
        samples={r['pool_id']:r for r in (json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip())}
        raw['status']='priming';checkpoint();priming_start=time.perf_counter_ns()
        for row in raw['priming_request_records']:
            monitor.check()
            if time.perf_counter_ns()-priming_start>=job['timeouts']['priming_hard_wall_seconds']*1e9:raise TimeoutError('Full priming hard wall')
            row['scheduled_time_ns']=time.perf_counter_ns();runtime.enqueue(row,samples[row['pool_id']]['sample'])
            while True:
                monitor.check();now=time.perf_counter_ns()
                if now-priming_start>=job['timeouts']['priming_hard_wall_seconds']*1e9:raise TimeoutError('Full priming hard wall')
                with runtime.lock:done=validate_response(row,now,300)
                checkpoint()
                if done:break
                time.sleep(.005)
        raw['full_capability_priming_completed_time_ns']=time.perf_counter_ns()
        for phase,phase_raw in zip(job['phases'],phases):
            active=phase_raw;phase_raw['status']='warmup';raw['status']='warmup:'+phase['task'];checkpoint()
            warmup(runtime,phase,samples,phase_raw['warmup_request_records'],monitor,progress=checkpoint,metadata=phase_raw['warmup_window'])
            monitor.sample();monitor.check();phase_raw['status']='measurement';raw['status']='measurement:'+phase['task'];checkpoint()
            finite_with_support(runtime,phase,samples,phase_raw['request_records'],monitor,progress=checkpoint,metadata=phase_raw['measurement_window'])
            monitor.sample();monitor.check();phase_raw['status']='completed';active=None;checkpoint(force=True)
        raw['status']='completed'
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
        if active:active['status']='failed'
    finally:
        raw['status_before_shutdown']=raw['status']
        if runtime:loading=False
        if monitor:
            try:monitor.close();monitor.check()
            except Exception as exc:raw['status']='failed';raw['errors'].append('Final memory: '+str(exc))
            peak=max((s['owned_used_mib'] for s in monitor.samples),default=None)
            raw['memory_evidence']=dict(samples=monitor.samples,errors=monitor.errors,peak_owned_mib=peak,
                safe_memory_mib=peak*packet['safe_factor'] if peak is not None else None,safe_factor=packet['safe_factor'],
                budget_mib=packet['gpu_memory_mib'],all_workers_reached_ready=monitor.ready,
                resource_valid=bool(monitor.samples) and not monitor.errors and monitor.ready,
                scope='Observed process-owned sampled peak only; not guaranteed continuous peak or physical lower bound')
        if power:
            try:power.close()
            except Exception as exc:raw['status']='failed';raw['errors'].append('Power close: '+str(exc))
            intervals={}
            if api:
                def integral(begin,end):return api[3].integrate_power(power.samples,begin,end) if begin is not None and end is not None else None
                intervals['startup_to_ready_joules']=integral(startup,startup_end)
                for p in phases:
                    intervals[p['task']]={key:integral(p[key].get('start_time_ns'),p[key].get('end_time_ns')) for key in ('warmup_window','measurement_window')}
            raw['energy_evidence']=dict(scope='whole_gpu_board',samples=power.samples,errors=power.errors,intervals=intervals,
                measured=not power.failed,U=None)
            if power.failed or (raw['status']=='completed' and (intervals.get('startup_to_ready_joules') is None or
                any(v is None for p in phases for v in intervals[p['task']].values()))):
                raw['status']='failed';raw['errors'].append('Power sample/interval coverage invalid')
        if runtime:
            try:raw['shutdown']=runtime.close()
            except Exception as exc:raw['status']='failed';raw['errors'].append('Shutdown: '+str(exc))
            raw['errors'].extend(runtime.errors)
            raw['worker_ready']=dict(runtime.ready)
            clean=(set(raw['shutdown'])==set(job['aliases']) and all(not r.get('forced') and r.get('exit_code')==0 and r.get('bye_received') for r in raw['shutdown'].values()))
            raw['shutdown_clean']=clean
            if not clean or runtime.errors:raw['status']='failed';raw['errors'].append('Unclean native shutdown/transport')
            try:raw['gpu_after_shutdown']=gpu_idle(packet)
            except Exception as exc:raw['status']='failed';raw['errors'].append('Post-shutdown GPU check: '+str(exc))
        for rows in [raw['priming_request_records'],*[p[k] for p in phases for k in ('warmup_request_records','request_records')]]:
            for row in rows:
                if row['status'] not in TERMINAL:row.update(status='incomplete',
                    release_state='not_released' if row['scheduled_time_ns'] is None else 'released_without_response',
                    error='Context ended; unreleased times remain unknown')
            if api:api[4].annotate_records(rows)
        raw['finished_unix_ns']=time.time_ns();checkpoint(force=True)
        with (output/'run.json').open('x') as stream:json.dump(raw,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return raw


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--context-id');parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--launcher-check',action='store_true')
    args=parser.parse_args(argv);packet=json.loads(args.packet.read_text());checked=checker.validate_packet(packet)
    if args.launcher_check:
        result=cpu_launcher_checks(packet)
        if args.output:
            with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps(result));return 0
    if not args.execute:print(json.dumps(checked));return 0
    if not args.context_id or not args.output:parser.error('--execute requires one --context-id and fresh --output')
    jobs=[j for j in packet['jobs'] if j['context_id']==args.context_id]
    if len(jobs)!=1:parser.error('Unknown context ID; no default queue')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    def interrupted(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    raw=execute_context(packet,jobs[0],args.output,checker.sha(args.packet))
    print(json.dumps(dict(context_id=args.context_id,status=raw['status'],run=str(args.output/'run.json'),U=None)))
    return 0 if raw['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
