"""Native reference campaign with durable progress; no optimization or fake U."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_gpu_pilot import rebind_checkpoint, receipt
from gsdeploy.config import SystemConfig
from gsdeploy.runtime import runner
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from gsdeploy.runtime.telemetry import MemoryMonitor,PowerMonitor
from gsdeploy.runtime.preflight import preflight
from gsdeploy.runtime.workload import SamplePool,largest_inputs
from gsdeploy.energy import energy_key
from gsdeploy import normalization as norm
from gsdeploy.trace import trace_digest

BASE=ROOT/'artifacts/static-study-v6-20261021/inputs'
HISTORICAL=ROOT/'artifacts/static-study-gpu-20261022'


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def emit(**value):print(json.dumps(value,ensure_ascii=False),flush=True)


def prepare(base):
    base.mkdir(parents=True,exist_ok=False)
    binding=HISTORICAL/'calibration/specialists_shared_board-attempt1/native-binding'
    g2binding=HISTORICAL/'v2-holdout-runs/v2_holdout_A_g2_image075_ocr100-attempt1/native-binding'
    workers=json.loads((binding/'worker-registry.json').read_text())
    manifest=json.loads((binding/'quality-evidence.json').read_text())
    g2manifest=json.loads((g2binding/'quality-evidence.json').read_text())
    if manifest['quality_sha256']!=g2manifest['quality_sha256']:
        raise ValueError('Historical quality identities differ')
    manifest['models']['generalist_2']=g2manifest['models']['generalist_2']
    selected={};verification=[]
    worker_script=ROOT/'tools/predictor_validation_20261027/complete_observed_worker.py'
    sources={Path(__file__),ROOT/'experiments/collect_activation_gpu_pilot.py',worker_script,
        worker_script.with_name('observed_worker.py')}
    sources.update((ROOT/'gsdeploy').rglob('*.py'))
    for family in ('text','image','ocr','asr','generalist_2'):
        entry=copy.deepcopy(workers[family]);original=Path(entry['spec'])
        spec=json.loads(original.read_text());destination=base/(family+'-worker.json')
        verification.extend(rebind_checkpoint(spec,destination))
        spec=json.loads(destination.read_text())
        entry['spec']=str(destination);entry['spec_sha256']=receipt(destination)['sha256']
        command=entry['command'];command[1]=str(worker_script)
        command[command.index('--spec')+1]=str(destination)
        entry['env'].update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        spec['resources']['request_output']=str(base/'worker-output'/family)
        save(destination,spec);entry['spec_sha256']=receipt(destination)['sha256']
        selected[family]=entry
        sources.update(Path(command[command.index('--source-root')+1]).rglob('*.py'))
        sources.update((original,destination))
        if spec.get('adapter_source'):sources.add(Path(spec['adapter_source']))
        if spec['resources'].get('inference_source'):sources.add(Path(spec['resources']['inference_source']))
    save(base/'worker-registry.json',selected)
    save(base/'quality-evidence.json',manifest)
    save(base/'checkpoint-verification.json',verification)
    snapshots=[]
    for i,source in enumerate(sorted(sources)):
        destination=base/'source-snapshot'/str(i)/source.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
        snapshots.append(dict(source=str(source),snapshot=receipt(destination)))
    save(base/'sources.json',snapshots)
    plan=dict(schema='gsdeploy.activation_reference_campaign.v1',
        gpu_compatibility=dict(policy='same_model',name='NVIDIA GeForce RTX 4090'),
        config=str(ROOT/'configs/system_reference_v2_20261103.json'),
        input_pool=receipt(BASE/'runtime-inputs.jsonl'),
        episode_seeds=[6110411,6110412],isolated_seeds=dict(text=6110421,image=6110422,ocr=6110423,asr=6110424),
        confirmation_seeds=[6110491,6110492],episode_seconds=1800,warmup_seconds=10,
        priming_policy='largest_three_per_task',drain_seconds=300,
        reference_phase_wall_budget_seconds=7200,selection_role='normalization_development',
        isolated_policy='serial_full_task_input_pool_no_outstanding_predecessor',
        prior_three_run_pilot_limit='completed; user explicitly authorized sustained goal work')
    save(base/'plan.json',plan)
    emit(event='prepared',output=str(base))


def reporting_runtime(output):
    class ReportingRuntime(DeploymentRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.report_stop=threading.Event();self.report_begin=time.monotonic()
            def report():
                while not self.report_stop.wait(15):
                    with self.lock:
                        rows=copy.deepcopy(list(self.pending.values()))
                    status=dict(elapsed_seconds=time.monotonic()-self.report_begin,
                        counts={phase:dict(Counter(r['status'] for r in rows if r['phase']==phase))
                                for phase in ('priming','warmup','measurement')},
                        ready_aliases=sorted(self.ready))
                    save(output/'progress-records.json',rows)
                    save(output/'progress.json',status)
                    emit(event='progress',**status)
            self.report_thread=threading.Thread(target=report,daemon=True)
            self.report_thread.start()
        def close(self):
            result=super().close();self.report_stop.set();self.report_thread.join(timeout=3)
            return result
    return ReportingRuntime


def episode(base,index):
    plan=json.loads((base/'plan.json').read_text());output=base/f'episode-{index}'
    output.mkdir(exist_ok=False)
    for source in json.loads((base/'sources.json').read_text()):
        if receipt(source['source'])['sha256']!=source['snapshot']['sha256']:
            raise ValueError('Campaign executable source changed: '+source['source'])
    config=SystemConfig.load(plan['config'])
    winner=dict(deployment=dict.fromkeys(('text','image','ocr','asr'),1),
        routing={t:{t:q} for t,q in config.workload.items()})
    workers=json.loads((base/'worker-registry.json').read_text())
    save(output/'plan.json',winner)
    lock=(ROOT/'artifacts/.gpu-experiment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=runner.DeploymentRuntime;runner.DeploymentRuntime=reporting_runtime(output)
    try:
        raw=runner.run(config,winner,BASE/'runtime-inputs.jsonl',backend='native',workers=workers,
            duration=plan['episode_seconds'],warmup=plan['warmup_seconds'],
            seed=plan['episode_seeds'][index-1],arrival_process=config.arrival_process,
            startup_timeout=300,drain_timeout=plan['drain_seconds'],
            admission=base/'quality-evidence.json',priming_policy=plan['priming_policy'])
        raw['measurement_role']='normalization_development'
        save(output/'run.json',raw)
        emit(event='episode_finished',index=index,status=raw['status'],errors=raw['errors'],
            requests=len(raw['request_records']),energy_j=raw['energy_evidence'].get('episode_joules'),
            evaluation=raw.get('evaluation'))
        return raw['status']=='completed'
    finally:
        runner.DeploymentRuntime=previous
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def isolated(base,task):
    """Serial whole-pool latency reference; no open-loop stability or U claim."""
    plan=json.loads((base/'plan.json').read_text());output=base/('isolated-'+task)
    output.mkdir(exist_ok=False)
    full=SystemConfig.load(plan['config']);data=full.as_dict()
    data['workload']={task:full.workload[task]};data['latency_reference_ms']={task:None}
    data['horizon_seconds']=1
    # Only the read-only single-model admission check uses the isolated proxy.
    # Neither this proxy nor serial throughput becomes an optimization profile.
    data['catalog'].update(performance_mode='independent',memory_mode='standalone_proxy')
    normalization=data['catalog'][norm.KEY]
    normalization.update(workload_rps=data['workload'],horizon_seconds=1,
                         tasks={task:normalization['tasks'][task]})
    view=SystemConfig(**data)
    workers=json.loads((base/'worker-registry.json').read_text())
    winner=dict(deployment={task:1},routing={task:{task:full.workload[task]}})
    checks=preflight(view,winner,BASE/'runtime-inputs.jsonl',workers,
                     base/'quality-evidence.json',require_gpu=True)
    save(output/'preflight.json',checks)
    if not checks['ready']:raise ValueError(str(checks['issues']))
    if checks['gpu']['name']!=plan['gpu_compatibility']['name']:
        raise ValueError('Unexpected GPU model')
    inputs=[json.loads(line) for line in (BASE/'runtime-inputs.jsonl').read_text().splitlines()]
    inputs=[row for row in inputs if row['task_type']==task]
    declared=dict(schema='gsdeploy.isolated_reference_plan.v1',task=task,
        input_pool=receipt(BASE/'runtime-inputs.jsonl'),pool_ids=[r['pool_id'] for r in inputs])
    save(output/'input-plan.json',declared)
    alias=task+'#0'
    config=MeasurementConfig(deployment={alias:1},workload_rps=view.workload,
        routing_rps={task:{alias:full.workload[task]}},warmup_seconds=0,
        startup_timeout_seconds=300,drain_timeout_seconds=90,shutdown_timeout_seconds=15,
        random_seed=plan['isolated_seeds'][task])
    registry=runner.registry_for(view,{alias:task},'native',workers)
    runtime=reporting_runtime(output)(config,registry)
    monitor=MemoryMonitor(runtime,full.gpu_memory_mib,enabled=True,device=checks['device'])
    power=PowerMonitor(enabled=True,device=checks['device'])
    raw=dict(schema_version=1,backend='native',status='starting',errors=[],
        config=view.as_dict(),transport_config=config.as_dict(),plan=winner,
        replica_to_family={alias:task},worker_registry=registry,preflight=checks,
        admission_evidence=checks['admission_evidence'],isolated_reference_plan=declared,
        measurement_role='normalization_development',measurement_window=None,
        request_records=[],priming_request_records=[],warmup_request_records=[],
        priming_policy=plan['priming_policy'],U=None,
        collection_scope='isolated serial e2e; empirical arrival mean is not open-loop capacity')
    lock=(ROOT/'artifacts/.gpu-experiment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=after=None
    try:
        power.sample();monitor.allow_uninitialized=True;monitor.sample()
        startup=time.perf_counter_ns();power.start();monitor.start();runtime.start()
        monitor.allow_uninitialized=False;monitor.sample();monitor.check()
        def deliver(row,item):
            runtime.enqueue(row,item['sample']);runtime.drain([row],90)
            deadline=time.monotonic()+3
            while 'previous_handler_next_request_ready' not in row and time.monotonic()<deadline:
                time.sleep(.005)
            if row['status']!='completed' or 'previous_handler_next_request_ready' not in row:
                raise RuntimeError('Isolated request failed or missing READY: '+str(row))
            monitor.check()
        pool=SamplePool(BASE/'runtime-inputs.jsonl',[task],config.random_seed)
        for item in largest_inputs(pool,task):
            row=runner.record('priming',len(raw['priming_request_records']),task,alias,item,time.perf_counter_ns())
            raw['priming_request_records'].append(row);deliver(row,item)
        before=monitor.sample();start=time.perf_counter_ns()
        raw['measurement_window']=dict(start_time_ns=start,start_unix_ns=time.time_ns())
        for i,item in enumerate(inputs):
            row=runner.record('measurement',i,task,alias,item,time.perf_counter_ns())
            raw['request_records'].append(row);deliver(row,item)
            emit(event='isolated_request',task=task,completed=i+1,total=len(inputs))
        end=time.perf_counter_ns();after=monitor.sample();monitor.check()
        duration=(end-start)/1e9
        raw['measurement_window'].update(end_time_ns=end,drain_end_time_ns=end,duration_seconds=duration,
            offered_requests=len(inputs),trace_sha256=trace_digest([
                ((r['scheduled_time_ns']-start)/1e9,task) for r in raw['request_records']]))
        rate=len(inputs)/duration
        raw['transport_config'].update(arrival_process='isolated_serial',measurement_seconds=duration,
            workload_rps={task:rate},routing_rps={task:{alias:rate}})
        raw['plan']['routing']={task:{task:rate}}
        raw['status']='completed'
    except BaseException as exc:
        raw['status']='failed';raw['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        monitor.close();power.close();raw['shutdown']=runtime.close();raw['errors'].extend(runtime.errors)
        samples=monitor.samples
        measured=before is not None and after is not None and not monitor.failed
        peak=max((s['owned_used_mib'] for s in samples),default=0)
        raw['memory_evidence']=dict(measured=measured,resource_valid=measured and peak<=full.gpu_memory_mib
            and not any(s['foreign_pids'] for s in samples),safe_memory_mib=peak*full.catalog['safe_factor'],
            scope='owned process sampled peak',samples=samples,before=before,after=after,
            episode_coverage_valid=measured)
        w=raw['measurement_window']
        if w and raw['status']=='completed':
            rate=raw['transport_config']['workload_rps']
            raw['energy_evidence']=norm.measured_episode(dict(kind='measured',scope='whole_gpu_board',
                startup_begin_time_ns=startup,samples=power.samples,sampling_errors=power.errors,
                profile_key=energy_key(raw['plan']['deployment'],raw['plan']['routing'],rate,
                    w['duration_seconds'],config.random_seed,'isolated_serial',catalog=full.catalog)),w)
        if any(s['forced'] or s['exit_code']!=0 or not s['bye_received'] for s in raw['shutdown'].values()):
            raw['status']='failed';raw['errors'].append('Worker shutdown failed')
        save(output/'run.json',raw)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    emit(event='isolated_finished',task=task,status=raw['status'],errors=raw['errors'])
    return raw['status']=='completed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('op',choices=('prepare','episode','isolated'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--index',type=int,choices=(1,2),default=1)
    p.add_argument('--task',choices=('text','image','ocr','asr'))
    args=p.parse_args();base=args.output.resolve()
    if args.op=='prepare':prepare(base)
    elif args.op=='episode':raise SystemExit(0 if episode(base,args.index) else 1)
    else:
        if not args.task:p.error('--task is required for isolated')
        raise SystemExit(0 if isolated(base,args.task) else 1)
