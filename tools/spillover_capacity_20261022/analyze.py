#!/usr/bin/env python3
"""Independent descriptive closed-loop diagnostics; no fit, U or ranking.

Observed handler overlap is retrospective exposure, never a predictor feature or
an estimate of GPU utilization. Packet/raw integration requires frozen receipts.
"""
from collections import Counter, defaultdict
import math

import numpy as np


def stats(values):
    values=[float(v) for v in values if type(v) in (int,float) and math.isfinite(v)]
    if not values:
        return dict(count=0,mean=None,min=None,p50=None,p95=None,max=None)
    return dict(count=len(values),mean=math.fsum(values)/len(values),min=min(values),
        p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),max=max(values))


def activity_segments(intervals, active_aliases):
    """Union busy intervals per alias into exact half-open activity masks.

    intervals: [{alias,start_ns,end_ns}], all wallclock from worker handlers.
    Counting each alias once prevents overlapping intervals from manufacturing
    concurrency; overlapping handlers on a serial worker are separately flagged.
    """
    aliases=set(active_aliases)
    if not aliases or len(aliases)!=len(active_aliases):raise ValueError('Unique nonempty active aliases required')
    events=defaultdict(Counter);by_alias=defaultdict(list)
    for interval in intervals:
        alias=interval['alias'];start=interval['start_ns'];end=interval['end_ns']
        if alias not in aliases:raise ValueError('Unexpected active alias')
        if type(start) is not int or type(end) is not int or end<=start:
            raise ValueError('Invalid handler interval')
        events[start][alias]+=1;events[end][alias]-=1
        by_alias[alias].append((start,end))
    counts=Counter();segments=[];previous=None
    for stamp,delta in sorted(events.items()):
        if previous is not None and stamp>previous:
            busy=sorted(a for a,count in counts.items() if count>0)
            segments.append(dict(start_ns=previous,end_ns=stamp,aliases=busy,duration_ns=stamp-previous))
        counts.update(delta);previous=stamp
    serial_overlap={}
    for alias,seq in by_alias.items():
        prev=None;bad=[]
        for start,end in sorted(seq):
            if prev is not None and start<prev:bad.append(dict(start_ns=start,previous_end_ns=prev))
            prev=max(end,prev) if prev is not None else end
        serial_overlap[alias]=bad
    return segments,serial_overlap


def interval_exposure(start,end,segments,active_aliases,alias):
    if type(start) is not int or type(end) is not int or end<=start:
        raise ValueError('Invalid request interval')
    aliases=set(active_aliases)
    if alias not in aliases:raise ValueError('Request alias absent from declared active set')
    histogram=Counter();all_active=other_busy=0
    for segment in segments:
        overlap=max(0,min(end,segment['end_ns'])-max(start,segment['start_ns']))
        if not overlap:continue
        busy=set(segment['aliases']);histogram[str(len(busy))]+=overlap
        if aliases<=busy:all_active+=overlap
        if busy-{alias}:other_busy+=overlap
    return dict(handler_duration_ns=end-start,all_active_aliases_busy_ns=all_active,
        all_active_aliases_busy_fraction=all_active/(end-start),
        any_other_alias_busy_ns=other_busy,any_other_alias_busy_fraction=other_busy/(end-start),
        concurrency_duration_ns=dict(histogram),scope='DESCRIPTIVE_OBSERVED_HANDLER_OVERLAP_NOT_GPU_UTILIZATION_OR_PREDICTOR')


def common_activity(intervals, active_aliases, unknown_requests=0):
    segments,overlap=activity_segments(intervals,active_aliases)
    busy=defaultdict(int);concurrency=Counter();all_active=0
    for segment in segments:
        duration=segment['duration_ns'];concurrency[str(len(segment['aliases']))]+=duration
        for alias in segment['aliases']:busy[alias]+=duration
        if set(active_aliases)<=set(segment['aliases']):all_active+=duration
    by_alias=defaultdict(list)
    for row in intervals:by_alias[row['alias']].append(row)
    if all(by_alias.get(alias) for alias in active_aliases):
        start=max(min(i['start_ns'] for i in by_alias[a]) for a in active_aliases)
        end=min(max(i['end_ns'] for i in by_alias[a]) for a in active_aliases)
        common_span=dict(start_ns=start,end_ns=max(start,end),duration_ns=max(0,end-start))
    else:common_span=None
    return dict(segments=segments,serial_overlap_flags=overlap,observed_busy_ns_by_alias={a:busy[a] for a in active_aliases},
        observed_concurrency_duration_ns=dict(concurrency),observed_all_active_aliases_busy_ns=all_active,
        common_observed_lifetime_span=common_span,unknown_requests=unknown_requests,
        observation_complete=unknown_requests==0,
        active_alias_count=len(active_aliases),singleton_all_active_condition_is_trivial=len(active_aliases)==1,
        limitation='Observed completed handler intervals only; UNKNOWN work can add unobserved activity. Common lifetime span does not imply simultaneous activity. Intervals are wallclock and do not measure GPU execution.',
        U=None,ranking_performed=False,fitted_parameters=False)


def digest(value):
    import hashlib,json
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def job_digest(job):
    return digest({k:v for k,v in job.items() if k!='job_identity_sha256'})


def frozen_scorers(source):
    """Import an explicit frozen package without replacing live gsdeploy imports."""
    import importlib,importlib.util,sys
    from pathlib import Path
    source=Path(source).resolve();package='_capacity_frozen_'+digest(str(source))[:12]
    if package not in sys.modules:
        spec=importlib.util.spec_from_file_location(package,source/'gsdeploy/__init__.py',
            submodule_search_locations=[str(source/'gsdeploy')])
        module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
    tasks=importlib.import_module(package+'.tasks');quality=importlib.import_module(package+'.quality')
    return tasks.score_result,quality.sample_totals


def validate_job(job, aliases):
    if job.get('job_identity_sha256')!=job_digest(job):raise ValueError('Job identity digest differs')
    active=job['active_aliases'];planned=job['planned_requests']
    if not active or len(set(active))!=len(active) or not set(active)<=set(aliases) or set(planned)!=set(active):
        raise ValueError('Active/planned alias roster differs')
    request_ids=set();pool_set=None;repetitions=None
    for alias in active:
        sequence=planned[alias]
        counts=Counter(r['pool_id'] for r in sequence)
        if len(sequence)!=96 or len(counts)!=32 or set(counts.values())!={3}:
            raise ValueError('Every active alias requires 32 inputs times 3 repeats')
        if pool_set is not None and set(counts)!=pool_set:raise ValueError('Alias input pools differ')
        pool_set=set(counts)
        repeat_groups=defaultdict(list)
        for index,row in enumerate(sequence):
            if row.get('model')!=alias or row.get('task')!='text':raise ValueError('Planned text/alias differs')
            if row['request_id'] in request_ids:raise ValueError('Duplicate planned request_id')
            request_ids.add(row['request_id']);repeat_groups[str(row['repetition'])].append(row['pool_id'])
            for key in ['sequence','arrival_index']:
                if type(row.get(key)) is not int:raise ValueError('Missing integer plan order')
                if row[key]!=index:raise ValueError('Plan order differs from cohort position')
            if row['repetition']!=index//32:raise ValueError('Plan repetition differs from cohort position')
        if len({r['sequence'] for r in sequence})!=96:raise ValueError('Repeated alias sequence position')
        if len(repeat_groups)!=3 or any(Counter(samples)!=Counter({s:1 for s in counts}) for samples in repeat_groups.values()):
            raise ValueError('Each repetition must cover each input exactly once')
        if repetitions is not None and set(repeat_groups)!=repetitions:raise ValueError('Alias repetition labels differ')
        repetitions=set(repeat_groups)
    return [row for alias in active for row in planned[alias]]


def ledger(record,response):
    keys=['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','actual_submit_unix_ns',
          'submit_write_completed_time_ns','completion_time_ns','completion_unix_ns']
    try:
        if any(type(record.get(k)) is not int for k in keys):raise ValueError('Missing integer host timing')
        scheduled,enqueue,submit,submitunix,write,complete,completeunix=[record[k] for k in keys]
        start,end=response['start_ns'],response['end_ns']
        terms=dict(host_schedule_to_enqueue_ms=(enqueue-scheduled)/1e6,
            host_enqueue_to_submit_ms=(submit-enqueue)/1e6,host_submit_write_ms=(write-submit)/1e6,
            submit_to_worker_start_ms=(start-submitunix)/1e6,worker_handler_ms=(end-start)/1e6,
            worker_end_to_host_receive_ms=(completeunix-end)/1e6,observed_e2e_ms=(complete-scheduled)/1e6)
        terms['clock_span_residual_ms']=terms['observed_e2e_ms']-math.fsum(terms[k] for k in
            ['host_schedule_to_enqueue_ms','host_enqueue_to_submit_ms','submit_to_worker_start_ms','worker_handler_ms','worker_end_to_host_receive_ms'])
        return dict(status='AVAILABLE',terms=terms,negative_time_flags=[k for k in
            ['host_schedule_to_enqueue_ms','host_enqueue_to_submit_ms','submit_to_worker_start_ms','worker_end_to_host_receive_ms'] if terms[k]<-1e-6])
    except (KeyError,TypeError,ValueError) as exc:return dict(status='UNKNOWN',reason=str(exc),terms=None)


def score_native(record,response,references,score_result,sample_totals):
    try:
        if record.get('status')!='completed' or response.get('success') is not True:
            raise ValueError('Completed successful native output required')
        if 'prediction' not in response:raise ValueError('Native prediction missing')
        result=dict(response,task=response.get('task',response.get('task_type')))
        score=score_result(references[record['task']][record['pool_id']],result)
        a,b=sample_totals(record['task'],score)
        return dict(status='AVAILABLE',score=score,error_numerator=a,reference_denominator=b,
            prediction_sha256=digest(response['prediction']),loss=a/b,
            scope='BALANCED_COHORT_NATIVE_OUTPUT_DIAGNOSTIC_NOT_FORMAL_Q_OR_U')
    except (KeyError,TypeError,ValueError) as exc:return dict(status='UNKNOWN',reason=str(exc),loss=None)


def summarize_requests(rows):
    service=[r for r in rows if r['service_status']=='AVAILABLE']
    quality=[r for r in rows if r['quality']['status']=='AVAILABLE']
    numerator=math.fsum(r['quality']['error_numerator'] for r in quality)
    denominator=math.fsum(r['quality']['reference_denominator'] for r in quality)
    return dict(planned_requests=len(rows),recorded_requests=sum(r['recorded'] for r in rows),
        raw_completed_requests=sum(r['original_record'].get('status')=='completed' for r in rows),
        service_available_requests=len(service),service_unknown_requests=len(rows)-len(service),
        quality_available_requests=len(quality),quality_unknown_requests=len(rows)-len(quality),
        service_ms=stats([r['service_ms'] for r in service]),handler_ms=stats([r['handler_ms'] for r in service]),
        generated_tokens=stats([r['worker_response'].get('generated_tokens') for r in service]),
        input_tokens=stats([r['worker_response'].get('input_tokens') for r in service]),
        diagnostic_native_quality_loss=numerator/denominator if denominator else None,
        diagnostic_native_accuracy=1-numerator/denominator if denominator else None,
        quality_cohort_complete=len(quality)==len(rows),
        prediction_variants=len({r['quality']['prediction_sha256'] for r in quality}),
        unit='Repeated observations do not expand the number of independent inputs; no confidence interval or fit.',U=None)


def finite_cohort_timing(rows,window,active_aliases):
    """Host monotonic completion rates, explicitly distinct from stable capacity."""
    window=window or {};start=window.get('start_time_ns');end=window.get('end_time_ns')
    valid=type(start) is int and type(end) is int and end>start
    duration=(end-start)/1e9 if valid else None
    aliases={};finishes=[]
    for alias in active_aliases:
        seq=[r for r in rows if r['model']==alias]
        completed=[r for r in seq if r['service_status']=='AVAILABLE']
        times=[r['original_record'].get('completion_time_ns') for r in completed]
        fully_observed=len(completed)==len(seq) and all(type(t) is int for t in times)
        finish=max(times) if fully_observed and times else None
        elapsed=(finish-start)/1e9 if type(start) is int and finish is not None and finish>start else None
        if finish is not None:finishes.append(finish)
        alias_window_rate=len(completed)/duration if duration else None
        aliases[alias]=dict(completed=len(completed),planned=len(seq),cohort_complete=fully_observed,
            completion_time_ns=finish,release_to_alias_completion_seconds=elapsed,
            finite_cohort_completion_rps=len(completed)/elapsed if elapsed else None,
            completions_per_common_measurement_second=alias_window_rate)
    all_complete=len(finishes)==len(active_aliases)
    return dict(measurement_window=window,observed_duration_seconds=duration,
        service_verified_completions=sum(r['service_status']=='AVAILABLE' for r in rows),
        completions_per_common_measurement_second=sum(r['service_status']=='AVAILABLE' for r in rows)/duration if duration else None,
        aliases=aliases,all_alias_cohorts_complete=all_complete,
        post_first_alias_completion_tail_seconds=(max(finishes)-min(finishes))/1e9 if all_complete else None,
        scope='FINITE_COHORT_CLOSED_LOOP_COMPLETION_RATES_NOT_OPEN_LOOP_STABLE_CAPACITY',U=None)


def repetition_drift(repeats):
    result=[]
    for alias in sorted({r['alias'] for r in repeats}):
        seq=sorted([r for r in repeats if r['alias']==alias],key=lambda r:int(r['repetition']))
        first=seq[0]['service_ms']['p50']
        result.append(dict(alias=alias,repetitions=[dict(repetition=r['repetition'],
            service_ms=r['service_ms'],service_available_requests=r['service_available_requests'],
            diagnostic_native_accuracy=r['diagnostic_native_accuracy'],
            generated_tokens=r['generated_tokens'],
            finite_cohort_timing=r['finite_cohort_timing'],
            median_service_over_repetition0=r['service_ms']['p50']/first if first and r['service_ms']['p50'] else None) for r in seq],
            scope='Balanced32-input descriptive repeated-cohort drift; no independence, stability, fit or ranking claim.'))
    return result


def analyze_job(packet,job,raw,packet_sha256,references,score_result,sample_totals,runtime_checks=None):
    planned=validate_job(job,packet['aliases'])
    if raw.get('status') not in {'completed','completed_with_failures','failed'}:raise ValueError('Raw is not terminal')
    if raw.get('schema')!='gsdeploy.balanced_closed_loop_capacity.raw.v1' or raw.get('protocol')!='balanced_closed_loop_capacity_v1':
        raise ValueError('Wrong diagnostic raw schema/protocol')
    if raw.get('job')!=job or raw.get('job_identity_sha256')!=job['job_identity_sha256'] or raw.get('packet_sha256')!=packet_sha256:
        raise ValueError('Raw packet/job identity differs')
    if raw.get('replica_to_family')!=packet['aliases'] or raw.get('plan',{}).get('deployment')!=packet['resident_deployment']:
        raise ValueError('Raw resident deployment/aliases differ')
    actual=raw.get('request_records')
    if not isinstance(actual,list):raise ValueError('request_records missing')
    runtime_checks=runtime_checks or dict(verified=False,reason='Runtime execution identity not supplied')
    if any(r.get('status')=='completed' for r in actual) and runtime_checks.get('verified') is not True:
        raise ValueError('Completed requests without verified runtime execution identity')
    by_id={}
    expected={r['request_id']:r for r in planned}
    for row in actual:
        rid=row.get('request_id')
        if rid not in expected or rid in by_id:raise ValueError('Unexpected/duplicate raw request_id')
        if any(row.get(k)!=v for k,v in expected[rid].items()):raise ValueError('Immutable planned record differs')
        if row.get('phase')!='measurement':raise ValueError('Nonmeasurement record in measurement cohort')
        by_id[rid]=row
    rows=[];intervals=[]
    for plan in planned:
        rec=by_id.get(plan['request_id'],dict(plan,status='NOT_RECORDED'))
        response=rec.get('worker_response') or {}
        result=dict(**plan,recorded=plan['request_id'] in by_id,original_record=rec,original_record_sha256=digest(rec),
            worker_response=response,service_status='UNKNOWN',service_ms=None,handler_ms=None,ledger=None)
        try:
            if rec.get('status')!='completed':raise ValueError('Request not completed')
            if response.get('request_id')!=rec['request_id'] or response.get('pool_id')!=rec['pool_id'] or response.get('task_type',response.get('task'))!=rec['task']:
                raise ValueError('Native response request/input/task differs')
            if response.get('submit_ns')!=rec.get('actual_submit_unix_ns') or type(response.get('submit_ns')) is not int:
                raise ValueError('Native response submit identity differs')
            start,end=response.get('start_ns'),response.get('end_ns');service=response.get('service_ms')
            if type(start) is not int or type(end) is not int or end<=start:raise ValueError('Invalid handler interval')
            if type(service) not in (int,float) or not math.isfinite(service) or service<=0:raise ValueError('Invalid native service')
            result.update(service_status='AVAILABLE',service_ms=service,handler_ms=(end-start)/1e6,
                ledger=ledger(rec,response),worker_start_unix_ns=start,worker_end_unix_ns=end)
            intervals.append(dict(alias=rec['model'],start_ns=start,end_ns=end))
            result['quality']=score_native(rec,response,references,score_result,sample_totals)
        except (KeyError,TypeError,ValueError) as exc:
            result['service_unknown_reason']=str(exc)
            result['quality']=dict(status='UNKNOWN',reason='Unverified request/response identity or completion',loss=None)
        rows.append(result)
    activity=common_activity(intervals,job['active_aliases'],sum(r['service_status']!='AVAILABLE' for r in rows))
    for row in rows:
        if row['service_status']=='AVAILABLE':
            row['activity']=interval_exposure(row['worker_start_unix_ns'],row['worker_end_unix_ns'],
                activity['segments'],job['active_aliases'],row['model'])
        else:row['activity']=None
    aliases={};inputs=[];repeats=[];blocks=[];closed_loop_flags=[]
    for alias in job['active_aliases']:
        seq=[r for r in rows if r['model']==alias];expected_counts=Counter(r['pool_id'] for r in seq)
        completed_counts=Counter(r['pool_id'] for r in seq if r['service_status']=='AVAILABLE')
        common_counts=Counter(r['pool_id'] for r in seq if r['activity'] and r['activity']['all_active_aliases_busy_ns']>0)
        aliases[alias]=dict(**summarize_requests(seq),family=packet['aliases'][alias],
            planned_per_input_counts=dict(expected_counts),completed_per_input_counts={k:completed_counts[k] for k in expected_counts},
            common_activity_positive_per_input_counts={k:common_counts[k] for k in expected_counts},
            completed_unique_inputs=len(completed_counts),common_activity_positive_unique_inputs=len(common_counts))
        for sample in sorted(expected_counts):
            s=[r for r in seq if r['pool_id']==sample]
            inputs.append(dict(alias=alias,pool_id=sample,**summarize_requests(s),
                repetitions=[dict(repetition=r['repetition'],sequence=r['sequence'],service_status=r['service_status'],
                    service_ms=r['service_ms'],quality=r['quality'],activity=r['activity']) for r in s]))
        for repetition in sorted({str(r['repetition']) for r in seq}):
            cohort=[r for r in seq if str(r['repetition'])==repetition]
            start=cohort[0]['original_record'].get('scheduled_time_ns')
            end=cohort[-1]['original_record'].get('completion_time_ns')
            repeats.append(dict(alias=alias,repetition=repetition,
                finite_cohort_timing=finite_cohort_timing(cohort,dict(start_time_ns=start,end_time_ns=end),[alias]),
                **summarize_requests(cohort)))
        observed=sorted([r for r in seq if r['service_status']=='AVAILABLE'],key=lambda r:r['worker_start_unix_ns'])
        for before,after in zip(sorted(seq,key=lambda r:r['sequence']),sorted(seq,key=lambda r:r['sequence'])[1:]):
            completion=before['original_record'].get('completion_time_ns')
            submit=after['original_record'].get('actual_submit_time_ns')
            if type(completion) is int and type(submit) is int and submit<completion:
                closed_loop_flags.append(dict(alias=alias,request_id=after['request_id'],reason='Submission precedes previous request completion'))
        for k,part in enumerate(np.array_split(np.array(observed,dtype=object),min(4,len(observed)))) if observed else []:
            blocks.append(dict(alias=alias,time_block=k,scope='Chronological within-run descriptive quartile; not independent observations',
                **summarize_requests(list(part))))
    memory=raw.get('memory_evidence') or {};shutdown=raw.get('shutdown') or {}
    clean_shutdown=bool(shutdown) and set(shutdown)==set(packet['aliases']) and all(
        r.get('exit_code')==0 and not r.get('forced') and r.get('bye_received') for r in shutdown.values())
    return dict(schema='gsdeploy.balanced_capacity_analysis.job.v1',job_id=job['job_id'],state=job['state'],block=job['block'],seed=job['seed'],
        identity_verified=True,runtime_execution_identity=runtime_checks,status=raw['status'],scope='DEVELOPMENT_CAPACITY_DIAGNOSTIC_NOT_INDEPENDENT_FORMAL_VALIDATION',
        completion=summarize_requests(rows),aliases=aliases,per_input=inputs,per_repetition=repeats,time_blocks=blocks,
        finite_cohort_timing=finite_cohort_timing(rows,raw.get('measurement_window'),job['active_aliases']),
        repetition_drift=repetition_drift(repeats),
        common_activity=activity,closed_loop_contract_flags=closed_loop_flags,request_diagnostics=rows,errors=raw.get('errors',[]),shutdown=shutdown,
        clean_shutdown=clean_shutdown,resource_evidence={k:v for k,v in memory.items() if k!='samples'},
        resource_energy_verification_scope='Raw runner declarations preserved; analyzer does not independently recompute memory peak or integrate power.',
        gpu_memory_mib=packet['gpu_memory_mib'],energy_evidence=raw.get('energy_evidence'),
        resource_valid=memory.get('resource_valid'),
        diagnostic_U=None,U=None,ranking_performed=False,fitted_parameters=False,
        limitations=['Closed-loop completed throughput is not open-loop stable capacity.',
            'Balanced alias input assignment is a distinct calibration identity, not the formal deterministic routing cohort.',
            'Observed activity is retrospective and must never be used as a future predictor feature.',
            'Two blocks are development repetition, not independent formal validation; repeated inputs are clustered.',
            'Failure/UNKNOWN requests and partial activity censor completed-only distributions.'])


def runtime_identity_checks(packet,raw,config,workers):
    import json,os
    from pathlib import Path
    registry=raw.get('worker_registry') or {};source=str(Path(packet['frozen_source']).resolve())
    worker_checks={};pythonpath={};ready_checks={};launcher_checks={}
    for alias,family in packet['aliases'].items():
        expected=dict(workers[family]);actual=registry.get(alias) or {}
        expected_env=dict(expected.pop('env',{}));actual_env=dict(actual.get('env') or {})
        actual_path=actual_env.pop('PYTHONPATH',None)
        expected_suffix=expected_env.pop('PYTHONPATH',None)
        expected_env.setdefault('CUDA_VISIBLE_DEVICES','0')
        expected.setdefault('cwd',source)
        actual_other={k:v for k,v in actual.items() if k!='env'}
        path_valid=(isinstance(actual_path,str) and actual_path.split(os.pathsep)[0]==source
                    and (expected_suffix is None or actual_path==source+os.pathsep+expected_suffix))
        worker_checks[alias]=(actual_other==expected and actual_env==expected_env and path_valid)
        pythonpath[alias]=actual_path
        entry=workers[family];command=entry['command']
        spec_path=Path(entry.get('spec') or command[command.index('--spec')+1])
        spec=json.loads(spec_path.read_text());ready=(raw.get('worker_ready') or {}).get(alias) or {}
        ready_checks[alias]=(ready.get('worker')==spec['candidate']['slot'] and
            ready.get('model')==spec['candidate']['model'] and ready.get('spec_sha256')==file_receipt(spec_path)['sha256'] and
            set(ready.get('tasks',[]))==set(config['catalog']['models'][family]['supported_tasks']))
        launcher=(raw.get('cpu_launcher_checks') or {}).get(family) or {}
        expected_command=['--check' if arg=='--execute' else arg for arg in command]+['--pool',packet['samples']]
        try:cpu_output=json.loads(launcher.get('stdout',''))
        except (TypeError,ValueError):cpu_output={}
        launcher_checks[family]=(launcher.get('command')==expected_command and launcher.get('exit_code')==0 and
            cpu_output.get('gpu_started') is False)
    preflight=raw.get('preflight') or {}
    job=raw.get('job') or {};active=job.get('active_aliases') or []
    expected_transport=dict(deployment={a:1 for a in packet['aliases']},workload_rps={'text':1},
        routing_rps={'text':{a:1/len(active) for a in active}},warmup_seconds=packet['warmup_seconds'],
        measurement_seconds=packet['measurement_timeout_seconds'],arrival_process='deterministic_periodic',
        random_seed=job.get('seed'),routing_source=packet['protocol'],startup_timeout_seconds=packet.get('startup_timeout_seconds',300),
        drain_timeout_seconds=packet['request_timeout_seconds'],shutdown_timeout_seconds=packet.get('shutdown_timeout_seconds',10),
        max_pending_per_model=1,idle_residency=False)
    priming=raw.get('priming_request_records') or [];warmup=raw.get('warmup_request_records') or []
    admission=json.loads(Path(packet['admission']).read_text());quality=config['catalog']['quality']
    identity=raw.get('priming_path_identity') or {}
    expected_pairs={(family,task) for family in packet['resident_deployment'] for task in config['catalog']['models'][family]['supported_tasks']}
    actual_pairs={(p.get('model'),p.get('task')) for p in identity.get('pairs',[])}
    bound_pairs=True
    for pair in identity.get('pairs',[]):
        family,task=pair.get('model'),pair.get('task')
        if (family,task) not in expected_pairs:bound_pairs=False;continue
        record=quality['paths'][family][task];checkpoint=admission['models'][family]['checkpoint']['sha256']
        recorded=record.get('checkpoint_sha256') or (record.get('checkpoint_identity',{}).get('sha256') if record.get('checkpoint_identity',{}).get('kind')=='checkpoint_files' else None)
        bound_pairs=bound_pairs and (pair.get('measurement')==record and pair.get('runtime_checkpoint_sha256')==checkpoint and
            pair.get('offline_full_checkpoint_equivalence_verified') is (recorded==checkpoint))
    expected_reports=[dict(model=family,**report) for family in packet['resident_deployment'] for report in admission['models'][family]['quality_reports']]
    priming_binding=(identity.get('contract')=='quality_evidence_v1' and identity.get('manifest_sha256')==packet['bindings'][packet['admission']] and
        identity.get('quality_sha256')==digest(quality)==admission.get('quality_sha256') and
        identity.get('checkpoints')=={family:admission['models'][family]['checkpoint']['sha256'] for family in packet['resident_deployment']} and
        actual_pairs==expected_pairs and len(identity.get('pairs',[]))==len(expected_pairs) and bound_pairs and identity.get('reports')==expected_reports)
    priming_roster=[{k:r.get(k) for k in ['model','task','pool_id']} for r in priming]
    def completed_phase(row,phase):
        response=row.get('worker_response') or {}
        return (row.get('phase')==phase and row.get('status')=='completed' and
            response.get('request_id')==row.get('request_id') and
            response.get('pool_id')==row.get('pool_id') and response.get('task_type',response.get('task'))==row.get('task') and
            type(row.get('completion_time_ns')) is int)
    warm_window=raw.get('warmup_window') or {};measurement=raw.get('measurement_window') or {}
    warm_start=warm_window.get('start_time_ns');warm_end=warm_window.get('end_time_ns');measure_start=measurement.get('start_time_ns')
    warm_timing=(all(type(t) is int for t in [warm_start,warm_end,measure_start]) and
        warm_end-warm_start>=packet['warmup_seconds']*1e9 and warm_end<=measure_start)
    warm_drained=(bool(warmup) and all(completed_phase(r,'warmup') and r['completion_time_ns']<=measure_start for r in warmup)) if type(measure_start) is int else False
    checks=dict(config=raw.get('config')==config,worker_registry=all(worker_checks.values()),
        cpu_launchers=all(launcher_checks.values()),
        worker_ready=all(ready_checks.values()),transport_config=raw.get('transport_config')==expected_transport,
        sample_pool=preflight.get('sample_pool_sha256')==packet['bindings'][str(Path(packet['samples']))],
        gpu_context=preflight.get('gpu',{}).get('uuid')==packet['gpu_uuid'],
        gpu_preflight=preflight.get('ready') is True and preflight.get('gpu_checked') is True,
        priming_identity=(raw.get('priming_path_identity') or {}).get('verified') is True,
        priming_identity_bindings=priming_binding,
        priming_roster=priming_roster==packet['priming_requests'],
        priming_completed=len(priming)==21 and all(completed_phase(r,'priming') for r in priming),
        priming_drained_before_warmup=type(warm_start) is int and bool(priming) and all(
            type(r.get('completion_time_ns')) is int and r['completion_time_ns']<=warm_start for r in priming),
        warmup_active_aliases=set(r.get('model') for r in warmup)==set(active),
        warmup_drained=warm_drained,warmup_timing=warm_timing,
        boundary_idle=(raw.get('gpu_boundary_preflight') or {}).get('uuid')==packet['gpu_uuid'] and
            (raw.get('gpu_boundary_preflight') or {}).get('compute_pids')==[],
        before_worker_start_idle=(raw.get('gpu_before_worker_start') or {}).get('uuid')==packet['gpu_uuid'] and
            (raw.get('gpu_before_worker_start') or {}).get('compute_pids')==[])
    return dict(verified=all(checks.values()),checks=checks,worker_checks=worker_checks,ready_checks=ready_checks,
        cpu_launcher_checks=launcher_checks,
        phase_native_success_complete={phase:bool(records) and all((r.get('worker_response') or {}).get('success') is True for r in records)
            for phase,records in [('priming',priming),('warmup',warmup)]},
        observed_pythonpath=pythonpath,note='Inherited PYTHONPATH suffix disclosed; frozen source must be first and all packet worker fields match.')


def file_receipt(path):
    import hashlib
    from pathlib import Path
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inspect_packet(path):
    import json
    from pathlib import Path
    path=Path(path).resolve();packet=json.loads(path.read_text())
    if packet.get('schema_version')!=1 or packet.get('protocol')!='balanced_closed_loop_capacity_v1' or packet.get('diagnostic_only') is not True or packet.get('U') is not None:
        raise ValueError('Expected explicit balanced diagnostic packet with U=null')
    if packet.get('aliases')!={'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'} or packet.get('resident_deployment')!={'text':2,'generalist_2':1,'ocr':1}:
        raise ValueError('Unexpected fixed resident context')
    bindings=packet.get('bindings') or {}
    if packet.get('bindings_sha256')!=digest(bindings):raise ValueError('Packet binding digest differs')
    if not bindings:raise ValueError('Missing frozen bindings')
    for filename,sha256 in bindings.items():
        if not Path(filename).is_absolute() or file_receipt(filename)['sha256']!=sha256:raise ValueError('Frozen file binding differs: '+filename)
    for key in ['system_config','samples','workers','admission']:
        if not Path(packet[key]).is_absolute() or str(Path(packet[key])) not in bindings:raise ValueError('Missing bound path: '+key)
    source=Path(packet['frozen_source'])
    files=list((source/'gsdeploy').rglob('*.py'))
    if not files or any(str(p) not in bindings for p in files):raise ValueError('Frozen Python package not fully bound')
    if packet.get('priming_policy')!='largest_three_all_resident_supported_tasks' or len(packet.get('priming_requests',[]))!=21:
        raise ValueError('Full resident lazy-path priming declaration required')
    jobs=packet['jobs']
    if [(j['block'],j['state'],j['seed']) for j in jobs]!=[(1,'A',101),(1,'B',101),(1,'C',101),(2,'C',102),(2,'B',102),(2,'A',102)]:
        raise ValueError('Unexpected six-job development roster; holdout/final seeds never accepted')
    if len({j['job_id'] for j in jobs})!=6:raise ValueError('Repeated job_id')
    sample_rows=[json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    pool={r['pool_id'] for r in sample_rows if r['task_type']=='text'}
    if len(pool)!=32:raise ValueError('Expected bound32 text input pool')
    active={'A':['text#0','text#1'],'B':['generalist_2#0'],'C':['text#0','text#1','generalist_2#0']}
    block_orders={}
    for job in jobs:
        rows=validate_job(job,packet['aliases'])
        if job['active_aliases']!=active[job['state']]:raise ValueError('Active state differs')
        if set(job['input_pool_ids'])!=pool or any([r['pool_id'] for r in job['planned_requests'][alias]]!=job['input_pool_ids'] for alias in job['active_aliases']):
            raise ValueError('Bound pool or alias input order differs')
        previous=block_orders.setdefault(job['block'],job['input_pool_ids'])
        if previous!=job['input_pool_ids']:raise ValueError('State input order differs within block')
    config=json.loads(Path(packet['system_config']).read_text());workers=json.loads(Path(packet['workers']).read_text())
    if config['gpu_memory_mib']!=packet['gpu_memory_mib'] or config['catalog']['safe_factor']!=packet['safe_factor']:
        raise ValueError('Resource declaration differs from bound config')
    references=config['catalog']['quality']['references']
    if set(references['text'])!=pool:raise ValueError('Frozen quality reference pool differs')
    scorer,total=frozen_scorers(source)
    metadata=dict(schema='gsdeploy.balanced_capacity_analysis.readiness.v1',packet=file_receipt(path),
        bindings=[dict(path=k,sha256=v) for k,v in bindings.items()],tool=file_receipt(__file__),
        jobs=[{k:j[k] for k in ['job_id','state','block','seed','active_aliases','job_identity_sha256']} for j in jobs],
        raw_files_read=0,progress_read=False,gpu_started=False,U=None)
    return packet,config,workers,references,scorer,total,metadata


def registered_paths(packet,paths,packet_sha256):
    import json
    from pathlib import Path
    declared={j['job_id']:j for j in packet['jobs']};registered={}
    for path in paths:
        progress=json.loads(Path(path).read_text())
        if progress.get('schema')!='gsdeploy.capacity_queue.v1':raise ValueError('Wrong queue schema')
        if progress.get('status') not in {'RUNNING','ALL_SIX_DIAGNOSTIC_ATTEMPTS_COMPLETE','PAUSED_ON_FAILED_ATTEMPT_NO_RETRY','PAUSED_INFRASTRUCTURE_ERROR_NO_RETRY'}:
            raise ValueError('Unknown queue status')
        if progress.get('packet_sha256')!=packet_sha256:raise ValueError('Queue packet receipt differs')
        for row in progress.get('jobs',[]):
            job_id=row.get('job_id')
            if job_id not in declared:raise ValueError('Undeclared queue job')
            if job_id in registered and registered[job_id]!=row:raise ValueError('Conflicting registration; no attempt selection')
            registered[job_id]=row
    ready=[];waiting=[]
    for job in packet['jobs']:
        row=registered.get(job['job_id'])
        if not row or row.get('status') not in {'completed','completed_with_failures','failed'}:
            waiting.append(dict(job_id=job['job_id'],status=row.get('status') if row else 'NOT_REGISTERED'));continue
        source=row.get('raw') or {};path=Path(source.get('path',''))
        if path.name!='run.json' or not source.get('sha256') or not path.is_absolute() or not path.is_file():
            waiting.append(dict(job_id=job['job_id'],status='PUBLISHED_FINAL_RECEIPT_MISSING'));continue
        ready.append((job,row,path))
    if ready and not waiting and any(json.loads(Path(path).read_text()).get('status')!='ALL_SIX_DIAGNOSTIC_ATTEMPTS_COMPLETE' for path in paths):
        waiting.append(dict(job_id=None,status='WAITING_ALL_SIX_QUEUE_PUBLICATION'))
    return waiting,ready


def read_final(path,expected_sha256):
    import hashlib,json
    from pathlib import Path
    path=Path(path)
    if path.name!='run.json':raise ValueError('Only exact final run.json accepted')
    before=path.stat();payload=path.read_bytes();after=path.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ['st_ino','st_size','st_mtime_ns','st_ctime_ns']) or not payload.endswith(b'\n'):
        raise ValueError('Final publication changed or incomplete')
    sha256=hashlib.sha256(payload).hexdigest()
    if sha256!=expected_sha256:raise ValueError('Registered final SHA differs')
    return json.loads(payload),dict(path=str(path.resolve()),sha256=sha256,bytes=len(payload))


def pair_job_results(left,right,comparison):
    output=[]
    for alias in sorted(set(left['aliases'])&set(right['aliases'])):
        a={r['pool_id']:r for r in left['per_input'] if r['alias']==alias}
        b={r['pool_id']:r for r in right['per_input'] if r['alias']==alias}
        for pool_id in sorted(set(a)&set(b)):
            ma=a[pool_id]['service_ms']['p50'];mb=b[pool_id]['service_ms']['p50']
            output.append(dict(alias=alias,pool_id=pool_id,left_job=left['job_id'],right_job=right['job_id'],
                comparison=comparison,left_completed=a[pool_id]['service_available_requests'],right_completed=b[pool_id]['service_available_requests'],
                left_median_service_ms=ma,right_median_service_ms=mb,right_over_left=mb/ma if ma and mb else None,
                left_generated_tokens=a[pool_id]['generated_tokens'],right_generated_tokens=b[pool_id]['generated_tokens'],
                left_native_accuracy=a[pool_id]['diagnostic_native_accuracy'],right_native_accuracy=b[pool_id]['diagnostic_native_accuracy'],
                scope='Within-input descriptive comparison; blocks are not pooled or used to fit a predictor.'))
    return output


def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--queue-progress',type=Path,nargs='+')
    p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    packet,config,workers,references,score_result,sample_totals,metadata=inspect_packet(args.packet)
    if args.check_only:print(json.dumps(metadata));return
    if not args.queue_progress or not args.output:p.error('Analysis requires explicit queue progress and new output')
    progresses=[file_receipt(path) for path in args.queue_progress]
    waiting,ready=registered_paths(packet,args.queue_progress,metadata['packet']['sha256'])
    if not ready or waiting and not args.partial:
        print(json.dumps(dict(status='WAITING_TERMINAL_RECEIPTS',waiting=waiting,raw_files_read=0,gpu_started=False,U=None)))
        raise SystemExit(3)
    if args.output.exists():raise FileExistsError('Never overwrite diagnostic evidence')
    reports=[]
    for job,registration,path in ready:
        raw,source=read_final(path,registration['raw']['sha256'])
        if raw.get('status')!=registration['status']:raise ValueError('Raw/receipt terminal status differs')
        runtime_checks=runtime_identity_checks(packet,raw,config,workers)
        result=analyze_job(packet,job,raw,metadata['packet']['sha256'],references,score_result,sample_totals,runtime_checks=runtime_checks)
        result.update(raw=source);reports.append(result)
    comparisons=[]
    by_key={(r['block'],r['state']):r for r in reports}
    for block in [1,2]:
        for single in ['A','B']:
            if (block,single) in by_key and (block,'C') in by_key:
                comparisons.extend(pair_job_results(by_key[(block,single)],by_key[(block,'C')],f'block{block}_{single}_to_C'))
    for state in ['A','B','C']:
        if (1,state) in by_key and (2,state) in by_key:
            comparisons.extend(pair_job_results(by_key[(1,state)],by_key[(2,state)],f'state{state}_block1_to_block2'))
    for item in [metadata['packet'],metadata['tool'],*metadata['bindings'],*progresses,*[r['raw'] for r in reports]]:
        if file_receipt(item['path'])['sha256']!=item['sha256']:raise ValueError('Analysis input/source changed')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data):(args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    overview=[]
    for report in reports:
        filename=report['job_id']+'.json';save(filename,report)
        overview.append({k:v for k,v in report.items() if k not in ['request_diagnostics','per_input','time_blocks','common_activity']})
        overview[-1].update(detail_file=filename,common_activity={k:v for k,v in report['common_activity'].items() if k!='segments'})
    status='INCOMPLETE' if waiting or len(reports)!=6 or any(r['status']!='completed' for r in reports) else 'ALL_SIX_DEVELOPMENT_DIAGNOSTICS_COMPLETE'
    save('report.json',dict(schema='gsdeploy.balanced_capacity_analysis.v1',status=status,jobs=overview,
        common_input_comparisons=comparisons,waiting=waiting,partial_requested=args.partial,
        raw_files_read=len(reports),U=None,diagnostic_U=None,ranking_performed=False,fitted_parameters=False,
        independent_formal_validation=False,gpu_started=False))
    lines=['# 闭环容量开发诊断', '',f'状态：{status}；已登记终态 {len(reports)}/6。', '',
        '固定驻留 text×2、generalist_2×1、ocr×1。每个活动 alias 接收相同的32个 text 输入×3轮。', '',
        '| job | alias | 服务完成/计划 | 原生准确率 | 服务中位数 ms | 有限 cohort 完成 rps |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(value):return 'UNKNOWN' if value is None else f'{value:.4g}'
    for r in reports:
        for alias,a in r['aliases'].items():
            rate=r['finite_cohort_timing']['aliases'][alias]['finite_cohort_completion_rps']
            lines.append(f"| {r['job_id']} | {alias} | {a['service_available_requests']}/{a['planned_requests']} | {fmt(a['diagnostic_native_accuracy'])} | {fmt(a['service_ms']['p50'])} | {fmt(rate)} |")
    lines.extend(['', '服务有效与资源有效分别报告；缺失、失败或身份不符的请求保留 UNKNOWN。质量仅来自原生输出与冻结参考的逐输入评分。', '',
        '有限 cohort 完成速率不代表开放负载稳定容量。共同生命周期不代表实际同时忙；handler 重叠不代表 GPU 利用率，也不得作为预测特征。', '',
        '两个 block 是开发重复，32个输入的重复轮次不增加独立样本数。跨状态与跨 block 的同 alias/同输入比较见 report.json，逐请求与活动区间见各 job 文件。', '',
        'U=null；不拟合、不排序、不选择最优部署，不据此宣称独立正式验证通过。'])
    if status=='INCOMPLETE':lines.extend(['', '六次诊断尚未全部有效完成；仅描述已登记终态，不判断最终重复性。'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    save('protocol.json',dict(metadata=metadata,queue_receipts=progresses,scope=__doc__,raw_files_read=len(reports),progress_read=True,U=None))
    save('manifest.json',dict(inputs=[metadata['packet'],*progresses,*[r['raw'] for r in reports]],
        source_bindings=metadata['bindings'],tool=metadata['tool'],
        files={path.name:file_receipt(path)['sha256'] for path in args.output.iterdir() if path.is_file()}))
    print(json.dumps(dict(status=status,jobs=len(reports),output=str(args.output.resolve()),U=None,gpu_started=False)))


if __name__=='__main__':main()
