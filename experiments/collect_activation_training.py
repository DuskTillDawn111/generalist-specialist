"""Matched full-input native solo/co-run captures with explicit continuous spans."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.runtime.runner import record
from gsdeploy.trace import make_trace
from tools.predictor_validation_20261027.native_collect import BoardMonitor,descendant_or_self

POOL=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def emit(**value):print(json.dumps(value,ensure_ascii=False),flush=True)


def prepare(output,reference,*,image=True,ocr_replicas=1,mode='solo',seed=6110501,capacity_task='ocr'):
    output.mkdir(parents=True,exist_ok=False)
    workers=json.loads((reference/'worker-registry.json').read_text())
    aliases={'generalist_2#0':'generalist_2'}
    if image:aliases['image#0']='image'
    aliases.update({f'ocr#{i}':'ocr' for i in range(ocr_replicas)})
    registry={alias:copy.deepcopy(workers[family]) for alias,family in aliases.items()}
    sources={Path(__file__),ROOT/'tools/predictor_validation_20261027/native_collect.py',
        ROOT/'tools/predictor_validation_20261027/complete_observed_worker.py',
        ROOT/'tools/predictor_validation_20261027/observed_worker.py',ROOT/'gsdeploy/resources.py'}
    sources.update((ROOT/'gsdeploy/runtime').glob('*.py'))
    for entry in registry.values():
        sources.add(Path(entry['spec']))
        cmd=entry['command'];sources.update(Path(cmd[cmd.index('--source-root')+1]).rglob('*.py'))
        spec=json.loads(Path(entry['spec']).read_text())
        for key in ('adapter_source',):
            if spec.get(key):sources.add(Path(spec[key]))
        if spec['resources'].get('inference_source'):sources.add(Path(spec['resources']['inference_source']))
    receipts=[]
    for i,source in enumerate(sorted(sources)):
        dest=output/'source-snapshot'/str(i)/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
        content=source.read_bytes();dest.write_bytes(content)
        receipts.append(dict(path=str(source),sha256=hashlib.sha256(content).hexdigest(),snapshot=str(dest)))
    context=dict(schema='gsdeploy.native_execution_context.v1',resident_aliases=aliases,
        hardware=dict(policy='same_model',name='NVIDIA GeForce RTX 4090'),
        worker_bindings=registry,source_content=[{k:r[k] for k in ('path','sha256')} for r in receipts],
        input_pool_sha256=hashlib.sha256(POOL.read_bytes()).hexdigest(),
        native_scope='continuous mixed host/transfers/CUDA calls; not pure kernel stages',
        threads=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    identity=digest(context)
    inputs=[json.loads(line) for line in POOL.read_text().splitlines()]
    by_task={t:[r for r in inputs if r['task_type']==t] for t in ('text','image','ocr','asr')}
    jobs=[]
    if mode=='solo':
        for alias,family in aliases.items():
            for task in (by_task if family=='generalist_2' else (family,)):
                for item in by_task[task]:
                    jobs.append(dict(id=f'solo:{alias}:{item["pool_id"]}',requests=[
                        dict(alias=alias,task=task,pool_id=item['pool_id'],offset_ms=0)]))
    elif mode=='overlap':
        for task,items in by_task.items():
            for i,item in enumerate(items):
                requests=[dict(alias='generalist_2#0',task=task,pool_id=item['pool_id'],offset_ms=0)]
                for alias,family in aliases.items():
                    if family=='generalist_2':continue
                    other=by_task[family][i%len(by_task[family])]
                    requests.append(dict(alias=alias,task=family,pool_id=other['pool_id'],offset_ms=0))
                jobs.append(dict(id=f'corun:{task}:{i}',requests=requests))
        # Sustained activity identifies overlap exposure beyond short launch coincidences.
        for task,items in by_task.items():
            for q in (.25,.5,.75):
                item=items[round(q*(len(items)-1))]
                requests=[dict(alias='generalist_2#0',task=task,pool_id=item['pool_id'],offset_ms=0)]
                for alias,family in aliases.items():
                    if family=='generalist_2':continue
                    count=16 if family=='image' else 6
                    for i in range(count):
                        other=by_task[family][i%len(by_task[family])]
                        requests.append(dict(alias=alias,task=family,pool_id=other['pool_id'],offset_ms=i*75))
                jobs.append(dict(id=f'sustained:{task}:{q}',requests=requests))
    elif mode=='capacity':
        if capacity_task+'#0' not in aliases:raise ValueError('Capacity task requires its specialist resident')
        source=reference/('isolated-'+capacity_task)/'run.json'
        measured=json.loads(source.read_text())
        if measured['status']!='completed':raise ValueError('Complete isolated capacity development source required')
        mean_ms=sum(r['worker_response']['worker_latency_ms'] for r in measured['request_records'])/len(measured['request_records'])
        mu=1000/mean_ms
        capacity_source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            mean_handler_ms=mean_ms,isolated_service_rate=mu,
            meaning='Nominal solo load probe; dynamic capacity remains to be measured')
        counters=dict.fromkeys(by_task,0)
        for level,rho in enumerate((.5,.8,1.05)):
            rates=dict.fromkeys(by_task,.0748360842705697);rates[capacity_task]=rho*mu
            requests=[]
            for offset,task in make_trace(rates,20.,seed+level,'poisson'):
                n=counters[task];counters[task]+=1;item=by_task[task][n%len(by_task[task])]
                candidates=[a for a,f in aliases.items() if f==task]
                alias=candidates[n%len(candidates)] if candidates else 'generalist_2#0'
                requests.append(dict(alias=alias,task=task,pool_id=item['pool_id'],offset_ms=offset*1000))
            jobs.append(dict(id=f'capacity:{capacity_task}:{rho}',requests=requests,duration_ms=20000,
                rates=rates,nominal_solo_rho=rho,pilot_not_formal_rho_qualification=True))
    else:raise ValueError('Unsupported capture mode')
    if mode!='capacity':random.Random(seed).shuffle(jobs)
    plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode=mode,
        split='development_capacity' if mode=='capacity' else 'training',
        execution_context_sha256=identity,run_id=output.name,seed=seed,aliases=aliases,
        jobs=jobs,budget_seconds=1800,priming='largest_three_per_supported_task',
        confirmation_seeds_reserved=[6110491,6110492],pure_gpu_stage_claim=False,
        independent_run_requirement='at least two separately launched rounds per fit context')
    if mode=='capacity':plan['capacity_source']=capacity_source
    save(output/'plan.json',plan);save(output/'context.json',context);save(output/'sources.json',receipts)
    emit(event='prepared',output=str(output),jobs=len(jobs),context=identity)


def collect(output):
    if (output/'raw.json').exists():raise FileExistsError(output/'raw.json')
    plan=json.loads((output/'plan.json').read_text());context=json.loads((output/'context.json').read_text())
    if digest(context)!=plan['execution_context_sha256']:raise ValueError('Execution context changed')
    for source in json.loads((output/'sources.json').read_text()):
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Executable source changed: '+source['path'])
    aliases=plan['aliases'];registry=context['worker_bindings']
    config=MeasurementConfig(deployment=dict.fromkeys(aliases,1),workload_rps={'text':1.},
        routing_rps={'text':{'generalist_2#0':1.}},startup_timeout_seconds=300,shutdown_timeout_seconds=15,
        routing_source='controlled_full_pool_training_not_utility')
    runtime=DeploymentRuntime(config,registry);monitor=None
    raw=dict(schema='gsdeploy.native_activity_capture.v1',kind='measured',backend='native',
        split=plan['split'],run_id=plan['run_id'],execution_context_sha256=plan['execution_context_sha256'],
        admission_enabled=False,status='starting',request_records=[],priming_request_records=[],batches=[],
        errors=[],U=None,pure_gpu_stage_claim=False,resident_aliases=aliases)
    pool={r['pool_id']:r for r in (json.loads(l) for l in POOL.read_text().splitlines())}
    lock=(ROOT/'artifacts/.gpu-experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def timeout(signum,frame):raise TimeoutError('Declared native training budget exhausted')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,plan['budget_seconds']-20)
    try:
        monitor=BoardMonitor(runtime);monitor.sample();monitor.check();monitor.start()
        raw['startup_begin_time_ns']=time.perf_counter_ns();runtime.start()
        for alias,worker in runtime.workers.items():
            if not descendant_or_self(runtime.ready[alias]['pid'],worker.process.pid):
                raise RuntimeError('READY is not an owned process')
        raw['ready']=copy.deepcopy(runtime.ready)
        def drain(rows):
            runtime.drain(rows,120);deadline=time.monotonic()+3
            while any('previous_handler_next_request_ready' not in row for row in rows) and time.monotonic()<deadline:
                time.sleep(.005)
            if any(row['status']!='completed' or 'previous_handler_next_request_ready' not in row for row in rows):
                raise RuntimeError('Failed training request or missing READY')
            monitor.check()
        for alias,family in aliases.items():
            tasks=('text','image','ocr','asr') if family=='generalist_2' else (family,)
            for task in tasks:
                items=[r for r in pool.values() if r['task_type']==task]
                def size(r):
                    s=r['sample'];return len(s['context'])+len(s['question']) if task=='text' else Path(s['input']).stat().st_size
                for item in sorted(items,key=lambda r:(size(r),r['pool_id']),reverse=True)[:3]:
                    row=record('priming',len(raw['priming_request_records']),task,alias,item,time.perf_counter_ns())
                    raw['priming_request_records'].append(row);runtime.enqueue(row,item['sample']);drain([row])
        raw['observation_start_ns']=time.perf_counter_ns();time.sleep(.5)
        for i,job in enumerate(plan['jobs']):
            start=time.perf_counter_ns();rows=[]
            for request in sorted(job['requests'],key=lambda r:r['offset_ms']):
                target=start+round(request['offset_ms']*1e6)
                while time.perf_counter_ns()<target:
                    monitor.check();time.sleep(min(.005,max(0,(target-time.perf_counter_ns())/1e9)))
                item=pool[request['pool_id']]
                row=record('training',len(raw['request_records']),request['task'],request['alias'],item,target)
                row['profile_job_id']=job['id'];raw['request_records'].append(row);rows.append(row)
                runtime.enqueue(row,item['sample'])
            target_end=start+round(job.get('duration_ms',0)*1e6)
            while time.perf_counter_ns()<target_end:
                monitor.check();time.sleep(.01)
            drain(rows);raw['batches'].append(dict(id=job['id'],start_ns=start,
                                                 arrival_end_ns=target_end if job.get('duration_ms') else None,
                                                 end_ns=time.perf_counter_ns(),
                                                 request_ids=[r['request_id'] for r in rows]))
            if i%8==0 or i+1==len(plan['jobs']):
                save(output/'progress.json',dict(completed_jobs=i+1,total_jobs=len(plan['jobs']),
                    requests=len(raw['request_records']),elapsed_seconds=(time.perf_counter_ns()-raw['startup_begin_time_ns'])/1e9))
                save(output/'progress-raw.json',raw)
                emit(event='progress',jobs=i+1,total=len(plan['jobs']),requests=len(raw['request_records']))
        time.sleep(.5);raw['observation_end_ns']=time.perf_counter_ns();monitor.sample();monitor.check()
        raw['status']='completed'
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc));emit(event='failed',error=raw['errors'][-1])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        runtime.close();raw['shutdown']=runtime.shutdown_report;raw['runtime_errors']=runtime.errors
        if monitor:
            monitor.close();raw['board_samples']=monitor.samples;raw['telemetry_errors']=monitor.errors
        if runtime.errors or any(v['forced'] or v['exit_code']!=0 or not v['bye_received'] for v in raw['shutdown'].values()):
            raw['status']='failed';raw['errors'].append('Runtime or shutdown failure')
        save(output/'raw.json',raw);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    emit(status=raw['status'],output=str(output/'raw.json'),requests=len(raw['request_records']))
    return raw['status']=='completed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('op',choices=('prepare','collect'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--no-image',action='store_true');p.add_argument('--ocr-replicas',type=int,choices=(0,1,2),default=1)
    p.add_argument('--mode',choices=('solo','overlap','capacity'),default='solo');p.add_argument('--seed',type=int,default=6110501)
    p.add_argument('--capacity-task',choices=('ocr','image'),default='ocr')
    args=p.parse_args();output=args.output.resolve()
    if args.op=='prepare':
        if args.reference is None:p.error('--reference is required')
        prepare(output,args.reference.resolve(),image=not args.no_image,ocr_replicas=args.ocr_replicas,
                mode=args.mode,seed=args.seed,capacity_task=args.capacity_task)
    else:raise SystemExit(0 if collect(output) else 1)
