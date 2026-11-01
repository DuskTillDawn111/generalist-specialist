"""Background full-input joint diagnostics; never score as formal utility.

The experiment packet is immutable. Every condition starts fresh resident workers,
and any failed condition pauses the serial queue without retrying any attempt.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PROTOCOL = 'background_joint_primary_support_v1'
TERMINAL = {'completed', 'error', 'incomplete'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_digest(job):
    return hashlib.sha256(canonical({k: v for k, v in job.items() if k != 'job_identity_sha256'})).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def validate_packet(packet):
    """Only the two bounded, predeclared background diagnostics may execute."""
    if packet.get('protocol')!=PROTOCOL or packet.get('diagnostic_only') is not True or packet.get('U') is not None:
        raise ValueError('Unknown/non-diagnostic protocol')
    if packet.get('priming_policy')!='largest_three_all_resident_supported_tasks' or len(packet.get('priming_requests',[]))!=21:
        raise ValueError('Full resident lazy-component priming required')
    if packet.get('resident_deployment')!={'text':2,'generalist_2':1,'ocr':1}:raise ValueError('Unexpected resident deployment')
    aliases={'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'}
    if packet.get('aliases')!=aliases:raise ValueError('Unexpected aliases')
    for key in ('frozen_source','system_config','samples','workers','admission'):
        if not Path(packet[key]).is_absolute():raise ValueError('Absolute identity path required: '+key)
    if not packet['bindings'] or hashlib.sha256(canonical(packet['bindings'])).hexdigest()!=packet['bindings_sha256']:
        raise ValueError('Binding map digest mismatch')
    for filename,expected in packet['bindings'].items():
        if digest(filename)!=expected:raise ValueError('Binding changed: '+filename)
    for key in ('system_config','samples','workers','admission'):
        if str(Path(packet[key])) not in packet['bindings']:raise ValueError('Unbound identity file: '+key)
    source_files=list((Path(packet['frozen_source'])/'gsdeploy').rglob('*.py'))
    if not source_files or any(str(p) not in packet['bindings'] for p in source_files):raise ValueError('Frozen Python source not fully bound')
    if str(Path(__file__).resolve()) not in packet['bindings']:raise ValueError('Runner source unbound')
    limits=dict(warmup_seconds=30,measurement_timeout_seconds=600,request_timeout_seconds=300,
                startup_timeout_seconds=300,shutdown_timeout_seconds=10,support_limit_per_alias=1024,generalist_support_every_primary=8)
    if any(packet.get(k)!=v for k,v in limits.items()):raise ValueError('Unexpected diagnostic limit')
    coverage=dict(minimum_common_handler_fraction=.9,minimum_covered_repetitions_per_input=2,
                  cross_block_per_input_median_ratio_min=.5,cross_block_per_input_median_ratio_max=2.)
    if packet.get('coverage_policy')!=coverage:raise ValueError('Unexpected coverage declaration')
    samples=[json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    pools={t:{r['pool_id'] for r in samples if r['task_type']==t} for t in ('text','image','asr','ocr')}
    if {t:len(v) for t,v in pools.items()}!={'text':32,'image':64,'asr':32,'ocr':32}:raise ValueError('Frozen task pools differ')
    if [(j['block'],j['seed']) for j in packet['jobs']]!=[(1,105),(2,106)]:raise ValueError('Exact two fresh blocks required')
    if len({j['job_id'] for j in packet['jobs']})!=2:raise ValueError('Duplicate job identity')
    for job in packet['jobs']:
        if job_digest(job)!=job['job_identity_sha256']:raise ValueError('Job identity differs')
        if job['active_aliases']!=['text#0','text#1','generalist_2#0','ocr#0']:raise ValueError('Unexpected active aliases')
        primary=job['planned_requests'];support=job['support_requests'];ids=[]
        if set(primary)!=set(aliases) or set(support)!={'text#0','text#1','generalist_2#0'}:raise ValueError('Unexpected roster aliases')
        if primary['text#0'] or primary['text#1']:raise ValueError('Text aliases are support-only')
        for alias,rows in primary.items():
            expected_counts={'image':192,'asr':96} if alias=='generalist_2#0' else ({'ocr':96} if alias=='ocr#0' else {})
            if dict(Counter(r['task'] for r in rows))!=expected_counts:raise ValueError('Background primary counts differ')
            task_indexes=Counter()
            for i,row in enumerate(rows):
                task=row['task'];ti=task_indexes[task];task_indexes[task]+=1
                expected=dict(request_id=f'measurement:{alias}:{i}',model=alias,task=task,role='primary',
                    pool_id=row['pool_id'],sequence=i,arrival_index=i,task_sequence=ti,repetition=ti//len(pools[task]))
                if row!=expected or row['pool_id'] not in pools[task]:raise ValueError('Invalid background template')
                ids.append(row['request_id'])
            for task in expected_counts:
                seq=[r for r in rows if r['task']==task];size=len(pools[task])
                if any({r['pool_id'] for r in seq[i:i+size]}!=pools[task] for i in range(0,len(seq),size)):
                    raise ValueError('Incomplete background input repetition')
        pattern=['image','image','asr'] if job['block']==1 else ['asr','image','image']
        if [r['task'] for r in primary['generalist_2#0']]!=pattern*96:raise ValueError('G2 task tape differs')
        for alias,rows in support.items():
            if len(rows)!=1024:raise ValueError('Support cap roster differs')
            for i,row in enumerate(rows):
                expected=dict(request_id=f'support:{alias}:{i}',model=alias,task='text',role='support',pool_id=row['pool_id'],
                    sequence=len(primary[alias])+i,arrival_index=len(primary[alias])+i,support_sequence=i,repetition=i//32)
                if row!=expected:raise ValueError('Invalid support template')
                ids.append(row['request_id'])
            if any({r['pool_id'] for r in rows[i:i+32]}!=pools['text'] for i in range(0,1024,32)):
                raise ValueError('Unbalanced text support')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate request IDs')
        if set(job['warmup_requests'])!=set(aliases) or any(not v for v in job['warmup_requests'].values()):
            raise ValueError('Missing warmup cohort')
        for alias,rows in job['warmup_requests'].items():
            if any(r['model']!=alias or r['pool_id'] not in pools[r['task']] for r in rows):raise ValueError('Invalid warmup identity')
    return dict(valid=True,jobs=2,protocol=PROTOCOL)


def gpu_idle(packet):
    query = subprocess.run(['nvidia-smi','-i',packet['device'],'--query-gpu=uuid','--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=5)
    if query.stdout.strip() != packet['gpu_uuid']:
        raise RuntimeError('GPU UUID differs from packet')
    apps = subprocess.run(['nvidia-smi','-i',packet['device'],'--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True,capture_output=True,check=True,timeout=5)
    if apps.stdout.strip():
        raise RuntimeError('GPU has compute processes; never overlap runs')
    return {'uuid': query.stdout.strip(), 'compute_pids': [], 'unix_ns':time.time_ns()}


def new_record(template, phase):
    return {**template, 'phase':phase, 'scheduled_time_ns':None, 'actual_submit_time_ns':None,
            'completion_time_ns':None, 'worker_latency_ms':None, 'worker_queue_wait_ms':None, 'status':'planned'}


def closed_loop(runtime, cohorts, samples, records, monitor, *, phase, timeout_seconds,
                request_timeout_seconds, warmup_seconds=None, progress=lambda: None, metadata=None):
    """Central release loop, one outstanding request per alias, common first release.

    No synthesized worker timestamps. Planned but unreleased finite requests remain
    explicitly planned until finalization marks them incomplete.
    """
    start = time.perf_counter_ns()
    meta = {} if metadata is None else metadata
    meta.update(start_time_ns=start,start_unix_ns=time.time_ns(), synchronous_release_time_ns=start,
                max_outstanding_per_alias=1, timeout_seconds=timeout_seconds, alias_completion_time_ns={})
    indexes = {a:0 for a in cohorts}
    pending = {}
    warmup = warmup_seconds is not None
    for alias, templates in cohorts.items():
        if not warmup and not any(r['model']==alias for r in records):
            records.extend(new_record(row, phase) for row in templates)
    finite = {a:[r for r in records if r['model']==a] for a in cohorts} if not warmup else {}
    progress()
    try:
        while True:
            now = time.perf_counter_ns()
            monitor.check()
            if (now-start)/1e9 >= timeout_seconds:
                raise TimeoutError(phase + ' wall timeout')
            with runtime.lock:
                for alias, row in list(pending.items()):
                    if row['status'] in TERMINAL:
                        if row['status'] != 'completed':
                            raise RuntimeError(phase + ' request failure: ' + row['request_id'])
                        if row['completion_time_ns']-row['scheduled_time_ns'] >= request_timeout_seconds*1e9:
                            raise TimeoutError('Request timeout: ' + row['request_id'])
                        del pending[alias]
                        if not warmup and indexes[alias] == len(cohorts[alias]):
                            meta['alias_completion_time_ns'][alias] = row['completion_time_ns']
                    elif (now-row['scheduled_time_ns'])/1e9 >= request_timeout_seconds:
                        raise TimeoutError('Request timeout: ' + row['request_id'])
                admitting = not warmup or (now-start)/1e9 < warmup_seconds
                for alias, templates in cohorts.items():
                    if alias in pending or not admitting:
                        continue
                    index = indexes[alias]
                    if not warmup and index >= len(templates):
                        continue
                    if warmup:
                        template = templates[index % len(templates)]
                        row = new_record({**template,'request_id':f'{phase}:{alias}:{index}', 'arrival_index':index}, phase)
                        records.append(row)
                    else:
                        row = finite[alias][index]
                    row['scheduled_time_ns'] = start if index == 0 else time.perf_counter_ns()
                    runtime.enqueue(row, samples[row['pool_id']]['sample'])
                    pending[alias] = row
                    indexes[alias] += 1
            progress()
            if time.perf_counter_ns()-start >= timeout_seconds*1e9:
                raise TimeoutError(phase + ' wall timeout, including progress snapshots')
            if not pending and ((warmup and not admitting) or (not warmup and all(indexes[a] == len(v) for a,v in cohorts.items()))):
                break
            time.sleep(.002)
    finally:
        meta['end_time_ns'] = time.perf_counter_ns()
        meta['duration_seconds'] = (meta['end_time_ns']-start)/1e9
        meta['released_by_alias'] = indexes
    return meta


def background_loop(runtime, cohorts, support_cohorts, samples, records, monitor, *,
                    timeout_seconds, request_timeout_seconds, support_every=8, progress=lambda:None, metadata=None):
    """One G2 worker serializes its explicit background tape and text support.

    Text specialists maintain bounded support from release. OCR idles resident
    after its primary. Every admitted record, including support drain, is kept.
    """
    start=time.perf_counter_ns();meta={} if metadata is None else metadata
    aliases=list(cohorts)
    meta.update(start_time_ns=start,start_unix_ns=time.time_ns(),synchronous_release_time_ns=start,
        max_outstanding_per_alias=1,timeout_seconds=timeout_seconds,alias_completion_time_ns={},
        primary_completion_time_ns=None,support_admission_stop_time_ns=None,
        primary_completed_by_alias={a:0 for a in aliases},support_released_by_alias={a:0 for a in aliases},
        generalist_inserted_support_at_primary_count=[],admission_policy='G2 support after each8 primary except global completion; OCR no support; specialist support throughout')
    finite={a:[r for r in records if r['model']==a and r.get('role')=='primary'] for a in aliases}
    if any(len(finite[a])!=len(cohorts[a]) for a in aliases):raise ValueError('Primary preregistration incomplete')
    indexes={a:0 for a in aliases};pending={};stopping=False;inserted_counts=set();admissions=Counter()
    def role_to_admit(alias):
        count=indexes[alias];total=len(cohorts[alias])
        if alias.startswith('text#'):return 'support'
        if alias=='ocr#0':return 'primary' if count<total else None
        if count<total:
            if count>0 and count%support_every==0 and count not in inserted_counts:return 'support'
            return 'primary'
        return 'support'
    try:
        while True:
            now=time.perf_counter_ns();monitor.check()
            if now-start>=timeout_seconds*1e9:raise TimeoutError('measurement wall timeout, including support drain')
            with runtime.lock:
                for alias,row in list(pending.items()):
                    if row['status'] in TERMINAL:
                        if row['status']!='completed':raise RuntimeError('measurement request failure: '+row['request_id'])
                        if row['completion_time_ns']-row['scheduled_time_ns']>=request_timeout_seconds*1e9:raise TimeoutError('Request timeout: '+row['request_id'])
                        del pending[alias]
                        if row['role']=='primary':
                            meta['primary_completed_by_alias'][alias]+=1
                            if meta['primary_completed_by_alias'][alias]==len(cohorts[alias]):meta['alias_completion_time_ns'][alias]=row['completion_time_ns']
                    elif now-row['scheduled_time_ns']>=request_timeout_seconds*1e9:raise TimeoutError('Request timeout: '+row['request_id'])
                if not stopping and all(meta['primary_completed_by_alias'][a]==len(cohorts[a]) for a in aliases):
                    stopping=True;meta['primary_completion_time_ns']=max(meta['alias_completion_time_ns'].values())
                    meta['support_admission_stop_time_ns']=time.perf_counter_ns()
                if not stopping:
                    choices={a:role_to_admit(a) for a in aliases if a not in pending}
                    for alias,role in choices.items():
                        if role=='support' and meta['support_released_by_alias'][alias]>=len(support_cohorts.get(alias,[])):
                            raise RuntimeError('Support cap exhausted before all primary complete: '+alias)
                    for alias,role in choices.items():
                        if role is None:continue
                        if role=='primary':row=finite[alias][indexes[alias]];indexes[alias]+=1
                        else:
                            si=meta['support_released_by_alias'][alias];row=new_record(support_cohorts[alias][si],'measurement');records.append(row)
                            meta['support_released_by_alias'][alias]+=1
                            if alias=='generalist_2#0' and indexes[alias]<len(cohorts[alias]):
                                inserted_counts.add(indexes[alias]);meta['generalist_inserted_support_at_primary_count'].append(indexes[alias])
                        row['admission_sequence']=admissions[alias]
                        row['scheduled_time_ns']=start if admissions[alias]==0 else time.perf_counter_ns()
                        runtime.enqueue(row,samples[row['pool_id']]['sample']);pending[alias]=row;admissions[alias]+=1
            progress()
            if time.perf_counter_ns()-start>=timeout_seconds*1e9:raise TimeoutError('measurement wall timeout, including progress snapshots and support drain')
            if stopping and not pending:break
            time.sleep(.002)
    finally:
        meta['end_time_ns']=time.perf_counter_ns();meta['duration_seconds']=(meta['end_time_ns']-start)/1e9
        meta['primary_released_by_alias']=indexes
        meta['released_by_alias']={a:indexes[a]+meta['support_released_by_alias'][a] for a in aliases}
    return meta


class GPUObserver:
    def __init__(self, device):
        self.device=device; self.samples=[]; self.errors=[]; self.stop=threading.Event(); self.thread=None
    def sample(self):
        start=time.perf_counter_ns()
        try:
            result=subprocess.run(['nvidia-smi','-i',self.device,
                '--query-gpu=uuid,pstate,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem,clocks_event_reasons.active',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            processes=subprocess.run(['nvidia-smi','-i',self.device,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            self.samples.append({'time_ns':start,'unix_ns':time.time_ns(),'query_elapsed_seconds':(time.perf_counter_ns()-start)/1e9,'gpu_csv':result.stdout.strip(),'compute_apps_csv':processes.stdout.strip()})
        except Exception as exc:
            self.errors.append(str(exc))
    def start(self):
        self.sample()
        def loop():
            while not self.stop.wait(10): self.sample()
        self.thread=threading.Thread(target=loop,daemon=True); self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=11)
        self.sample()


def begin_power_capture(power):
    # Bracket startup with a real sample; CPU hash/check time is outside the
    # measured GPU startup interval. integrate_power rejects extrapolation.
    power.sample()
    startup=time.perf_counter_ns()
    power.start()
    return startup


def frozen_api(packet):
    source=Path(packet['frozen_source'])
    sys.path.insert(0,str(source))
    names=['gsdeploy.config','gsdeploy.runtime.runner','gsdeploy.runtime.preflight','gsdeploy.runtime.schema','gsdeploy.runtime.workload','gsdeploy.metrics','gsdeploy.energy']
    modules=[importlib.import_module(n) for n in names]
    if any(not Path(m.__file__).resolve().is_relative_to(source) for m in modules):
        raise RuntimeError('Nonfrozen gsdeploy import; start fresh Python process')
    return modules


def stop_memory_with_final_sample(monitor):
    """Bracket even a failed measurement before owned GPU processes exit."""
    try:
        monitor.sample()
        monitor.check()
    finally:
        monitor.close()


def run_job(packet, job, output, packet_sha256):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    raw={'schema':'gsdeploy.background_joint_capacity.raw.v1','protocol':PROTOCOL,
         'packet_sha256':packet_sha256,'job':job,'job_identity_sha256':job_digest(job),
         'replica_to_family':packet['aliases'],'plan':{'deployment':packet['resident_deployment']},
         'status':'starting','errors':[],
         'request_records':[new_record(r,'measurement') for rows in job.get('planned_requests',{}).values() for r in rows],
         'warmup_request_records':[],
         'priming_request_records':[],'measurement_window':None,'shutdown':{},
         'evaluation':{'utility':{'U':None,'eligible':False,'reason':'CAPACITY_DIAGNOSTIC_NOT_FORMAL_UTILITY'}},
         'formal_stability':None,'rho':None,
         'telemetry_overhead_note':'Read-only GPU telemetry every10s plus frozen memory0.5s/power0.25s monitors; query time recorded; overhead not subtracted.'}
    runtime=monitor=power=observer=None
    api=None; startup=None
    raw['attempt_begin_time_ns']=time.perf_counter_ns()
    raw['snapshot_policy']='Progress at most1Hz; full live snapshots at most every5s; phase/terminal snapshots forced. Serialization overhead retained.'
    last_progress=last_live=0;last_phase=None;snapshot_seconds=0.
    def checkpoint(force=False):
        nonlocal last_progress,last_live,last_phase,snapshot_seconds
        now=time.monotonic();phase_changed=raw['status']!=last_phase
        forced=force or phase_changed
        if not forced and now-last_progress<1: return
        snapshot_start=time.perf_counter_ns()
        raw['last_checkpoint_unix_ns']=time.time_ns()
        if forced or now-last_live>=5:
            # Copy/serialize atomically relative to worker updates, then release the
            # runtime lock before disk I/O. Dispatch loop calls this outside lock.
            if runtime:
                with runtime.lock: encoded=json.dumps(raw,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
            else: encoded=json.dumps(raw,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
            tmp=output/'live.json.tmp';tmp.write_text(encoded);tmp.replace(output/'live.json')
            last_live=now
        atomic_json(output/'progress.json',{'status':raw['status'],'unix_ns':time.time_ns(),
            'counts':{key:dict(Counter(r['status'] for r in raw[key])) for key in ('priming_request_records','warmup_request_records','request_records')}})
        snapshot_seconds+=(time.perf_counter_ns()-snapshot_start)/1e9
        raw['snapshot_elapsed_seconds']=snapshot_seconds
        last_progress=now;last_phase=raw['status']
    checkpoint()
    try:
        validate_packet(packet)
        raw['gpu_boundary_preflight']=gpu_idle(packet)
        api=frozen_api(packet)
        cfgmod,runner,pre,schema,workload,metrics,energy=api
        system=cfgmod.SystemConfig.load(packet['system_config'])
        workers=json.loads(Path(packet['workers']).read_text())
        admission=packet['admission']
        aliases=packet['aliases']
        routes={t:({'text':rate/2,'generalist_2':rate/2} if t=='text' else {'ocr' if t=='ocr' else 'generalist_2':rate}) for t,rate in system.workload.items()}
        plan=packet.get('preflight_plan',{'deployment':packet['resident_deployment'],'routing':routes})
        raw['preflight']=pre.preflight(system,plan,packet['samples'],workers,admission,require_gpu=True)
        if not raw['preflight']['ready']:
            raise RuntimeError('Frozen native identity preflight failed: '+ '; '.join(raw['preflight']['issues']))
        registry=runner.registry_for(system,aliases,'native',workers)
        for entry in registry.values():
            if entry['env']['CUDA_VISIBLE_DEVICES'] != packet['device']:
                raise ValueError('Launcher device differs from packet')
        config=schema.MeasurementConfig(deployment={a:1 for a in aliases},workload_rps={'text':1,'image':1,'asr':1,'ocr':1},
            routing_rps={'text':{'text#0':.25,'text#1':.25,'generalist_2#0':.5},'image':{'generalist_2#0':1},'asr':{'generalist_2#0':1},'ocr':{'ocr#0':1}},
            warmup_seconds=packet['warmup_seconds'],measurement_seconds=packet['measurement_timeout_seconds'],
            startup_timeout_seconds=packet['startup_timeout_seconds'],drain_timeout_seconds=packet['request_timeout_seconds'],
            shutdown_timeout_seconds=packet['shutdown_timeout_seconds'],max_pending_per_model=1,random_seed=job['seed'],routing_source=PROTOCOL)
        config.validate({a:system.catalog['models'][f]['supported_tasks'] for a,f in aliases.items()})
        raw.update(config=system.as_dict(),transport_config=config.as_dict(),worker_registry=registry,preflight_plan=plan)
        raw['cpu_launcher_checks']={}
        for family in packet['resident_deployment']:
            entry=workers[family]
            command=['--check' if arg=='--execute' else arg for arg in entry['command']]
            if '--check' not in command: raise ValueError('No CPU-check mode in worker launcher')
            command += ['--pool',packet['samples']]
            checked=subprocess.run(command,cwd=entry.get('cwd'),env={**os.environ,**entry.get('env',{})},
                text=True,capture_output=True,timeout=300)
            raw['cpu_launcher_checks'][family]={'command':command,'exit_code':checked.returncode,
                'stdout':checked.stdout,'stderr':checked.stderr}
            checkpoint()
            if checked.returncode: raise RuntimeError('CPU launcher check failed: '+family)
        # Include G2/OCR priming path identity, which the low-rate routing does not use.
        identity=importlib.import_module('gsdeploy.evidence')
        coverage_plan={'deployment':packet['resident_deployment'],'routing':{
            'text':{'text':1,'generalist_2':1},'image':{'generalist_2':1},
            'asr':{'generalist_2':1},'ocr':{'generalist_2':1,'ocr':1}}}
        raw['priming_path_identity']=identity.verify_quality_evidence(system,coverage_plan,workers,admission)
        if not raw['priming_path_identity'].get('verified'): raise RuntimeError('Full priming path identity missing')
        raw['priming_path_identity_scope']='Coverage plan only; not actual rates/routing'
        raw['gpu_before_worker_start']=gpu_idle(packet)
        runtime=runner.DeploymentRuntime(config,registry)
        monitor=runner.MemoryMonitor(runtime,system.gpu_memory_mib,device=packet['device'])
        power=runner.PowerMonitor(device=packet['device']); observer=GPUObserver(packet['device'])
        startup=begin_power_capture(power); observer.start()
        raw['status']='loading';checkpoint();runtime.start()
        raw['worker_ready']=runtime.ready
        for alias,family in aliases.items():
            entry=workers[family]
            spec_path=Path(entry.get('spec') or entry['command'][entry['command'].index('--spec')+1])
            spec=json.loads(spec_path.read_text()); ready=runtime.ready[alias]
            if (ready.get('worker')!=spec['candidate']['slot'] or ready.get('model')!=spec['candidate']['model']
                or ready.get('spec_sha256')!=digest(spec_path)
                or set(ready.get('tasks',[]))!=set(system.catalog['models'][family]['supported_tasks'])):
                raise RuntimeError('READY identity mismatch: '+alias)
        monitor.sample();monitor.start();monitor.check()
        pool=workload.SamplePool(packet['samples'],['text','image','ocr','asr'],job['seed'])
        samples={r['pool_id']:r for rows in pool.samples.values() for r in rows}
        raw['status']='priming';checkpoint()
        # Serial priming still obeys one outstanding per alias. All resident supported tasks,
        # including idle G2 image/OCR/ASR and idle OCR, are loaded before timed warmup.
        expected_priming=[]
        for alias in sorted(aliases):
            for task in system.catalog['models'][aliases[alias]]['supported_tasks']:
                for item in workload.largest_inputs(pool,task,3):
                    expected_priming.append({'model':alias,'task':task,'pool_id':item['pool_id']})
        if packet['priming_requests']!=expected_priming or len(expected_priming)!=21:
            raise ValueError('Bound full-resident priming order differs')
        for item_id in packet['priming_requests']:
            item=samples[item_id['pool_id']];index=len(raw['priming_request_records'])
            row=runner.record('priming',index,item_id['task'],item_id['model'],item,time.perf_counter_ns())
            raw['priming_request_records'].append(row);runtime.enqueue(row,item['sample']);checkpoint()
            runtime.drain([row],packet['request_timeout_seconds']);monitor.check();checkpoint()
            if row['status']!='completed': raise RuntimeError('Resident priming failed: '+row['request_id'])
            if row['completion_time_ns']-row['scheduled_time_ns'] >= packet['request_timeout_seconds']*1e9:
                raise TimeoutError('Resident priming request timeout: '+row['request_id'])
        cohorts=job['planned_requests']
        raw['status']='warmup';raw['warmup_window']={};checkpoint()
        closed_loop(runtime,job['warmup_requests'],samples,raw['warmup_request_records'],monitor,phase='warmup',
            timeout_seconds=packet['warmup_seconds']+packet['request_timeout_seconds'],request_timeout_seconds=packet['request_timeout_seconds'],
            warmup_seconds=packet['warmup_seconds'],progress=checkpoint,metadata=raw['warmup_window'])
        monitor.sample();monitor.check()
        raw['status']='measurement';raw['measurement_window']={};checkpoint()
        background_loop(runtime,cohorts,job['support_requests'],samples,raw['request_records'],monitor,
            timeout_seconds=packet['measurement_timeout_seconds'],request_timeout_seconds=packet['request_timeout_seconds'],
            progress=checkpoint,metadata=raw['measurement_window'])
        raw['status']='completed';monitor.sample();monitor.check()
    except (Exception,KeyboardInterrupt) as exc:
        raw['status']='failed';raw['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        # Stop process-memory sampling before workers leave the GPU: absent owned
        # PIDs after clean shutdown are not a measurement sampling failure.
        if monitor:
            try:
                stop_memory_with_final_sample(monitor)
            except Exception as exc:
                raw['status']='failed';raw['errors'].append('final memory sample: '+str(exc))
        if power: power.close()
        if observer: observer.close()
        if runtime:
            raw['status_before_shutdown']=raw['status'];checkpoint()
            try: raw['shutdown']=runtime.close()
            except Exception as exc: raw['errors'].append('shutdown: '+str(exc));raw['status']='failed'
            raw['errors'].extend(runtime.errors)
            if runtime.errors or set(raw['shutdown']) != set(packet['aliases']) or any(r.get('forced') or r.get('exit_code') !=0 or not r.get('bye_received') for r in raw['shutdown'].values()):
                raw['status']='failed';raw['errors'].append('Unclean worker shutdown or transport error')
        for key in ('request_records','warmup_request_records','priming_request_records'):
            for row in raw[key]:
                if row['status'] not in TERMINAL:
                    row.update(status='incomplete', release_state='not_released' if row['scheduled_time_ns'] is None else 'released_without_response', error='diagnostic ended before response; unreleased requests have unknown times')
            if api: api[5].annotate_records(raw[key])
        if monitor:
            samples=monitor.samples;safe=system.catalog['safe_factor']*max((s['owned_used_mib'] for s in samples),default=0)
            valid=bool(samples) and not monitor.failed and safe<=system.gpu_memory_mib and all(not s['foreign_pids'] for s in samples)
            raw['memory_evidence']={'measured':not monitor.failed,'samples':samples,'safe_memory_mib':safe if samples else None,'resource_valid':valid,'budget_mib':system.gpu_memory_mib,'safe_factor':system.catalog['safe_factor']}
            if not valid: raw['status']='failed';raw['errors'].append('Missing/invalid memory evidence or resource budget exceeded')
        if power:
            w=raw['measurement_window'];end=w.get('end_time_ns') if w else None;begin=w.get('start_time_ns') if w else None
            runj=api[6].integrate_power(power.samples,begin,end) if begin and end else None
            startupj=api[6].integrate_power(power.samples,startup,begin) if begin else None
            raw['energy_evidence']={'scope':'whole_gpu_board','samples':power.samples,'sampling_errors':power.errors,'coverage_valid':not power.failed and runj is not None and startupj is not None,'run_joules':runj,'startup_joules':startupj,'startup_begin_time_ns':startup}
            if not raw['energy_evidence']['coverage_valid']: raw['status']='failed';raw['errors'].append('Power coverage invalid')
        if observer: raw.update(telemetry_samples=observer.samples,telemetry_errors=observer.errors)
        raw['finished_unix_ns']=time.time_ns()
        checkpoint(force=True);exclusive_json(output/'run.json',raw)
    return raw


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True);parser.add_argument('--output');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    packetpath=Path(args.packet).resolve();packet=json.loads(packetpath.read_text());checked=validate_packet(packet)
    if not args.execute:
        print(json.dumps(checked));return 0
    if not args.output: parser.error('--output required for execution')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
    def interrupted(signum,frame): raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    destination=Path(args.output).resolve();destination.mkdir(parents=True,exist_ok=False)
    packetsha=digest(packetpath)
    state={'schema':'gsdeploy.background_joint_capacity_queue.v1','status':'RUNNING','packet_sha256':packetsha,'jobs':[]}
    atomic_json(destination/'queue-progress.json',state)
    try:
        gpu_idle(packet)
        for job in packet['jobs']:
            if digest(packetpath)!=packetsha: raise ValueError('Packet bytes changed')
            validate_packet(packet)
            state['current']=job['job_id'];atomic_json(destination/'queue-progress.json',state)
            raw=run_job(packet,job,destination/(job['job_id']+'-attempt1'),packetsha)
            row={'job_id':job['job_id'],'status':raw['status'],'raw':{'path':str(destination/(job['job_id']+'-attempt1')/'run.json'),'sha256':digest(destination/(job['job_id']+'-attempt1')/'run.json')}}
            state['jobs'].append(row)
            if raw['status']!='completed':
                state['status']='PAUSED_ON_FAILED_ATTEMPT_NO_RETRY';break
            gpu_idle(packet)
        else: state['status']='ALL_TWO_BACKGROUND_DIAGNOSTIC_ATTEMPTS_COMPLETE'
    except (Exception,KeyboardInterrupt) as exc:
        state.update(status='PAUSED_INFRASTRUCTURE_ERROR_NO_RETRY',error=f'{type(exc).__name__}: {exc}')
    finally:
        state.pop('current',None);atomic_json(destination/'queue-progress.json',state)
    print(json.dumps(state),flush=True)
    return 0 if state['status']=='ALL_TWO_BACKGROUND_DIAGNOSTIC_ATTEMPTS_COMPLETE' else 1

if __name__=='__main__':
    raise SystemExit(main())
