#!/usr/bin/env python3
"""Descriptive primary/support diagnostic analysis; never fit capacity or utility.

Primary observations are clustered by the 32 input identities. Support requests
maintain the observed handler context and never increase independent sample size.
"""
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path

_SPEC=importlib.util.spec_from_file_location('_sustained_legacy_analysis',Path(__file__).with_name('analyze.py'))
legacy=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(legacy)


def coverage_gate(rows,active_aliases,state,threshold=.90,min_repeats=2):
    """Exact preregistered per-alias/input gate; never average missing coverage."""
    if state=='B':
        return dict(status='NOT_APPLICABLE_SINGLE_ACTIVE_ALIAS',passed=True,threshold=threshold,
            reason='A singleton all-active condition is trivial and is not evidence of joint activity.',aliases={})
    aliases={};passed=True
    for alias in active_aliases:
        seq=[r for r in rows if r['model']==alias and r['role']=='primary']
        inputs={}
        for pool_id in sorted({r['pool_id'] for r in seq}):
            cohort=[r for r in seq if r['pool_id']==pool_id]
            fractions=[r['activity']['all_active_aliases_busy_fraction'] if r.get('activity') else None for r in cohort]
            eligible=sum(v is not None and v>=threshold for v in fractions)
            positive=sum(v is not None and v>0 for v in fractions)
            good=len(cohort)==3 and eligible>=min_repeats and positive==3
            inputs[pool_id]=dict(primary_count=len(cohort),fraction_at_or_above_threshold=eligible,
                positive_common_activity_count=positive,passes=good,repetitions=[dict(repetition=r['repetition'],
                fraction=r['activity']['all_active_aliases_busy_fraction'] if r.get('activity') else None) for r in cohort])
        alias_pass=len(seq)==96 and len(inputs)==32 and all(r['passes'] for r in inputs.values())
        aliases[alias]=dict(passes=alias_pass,primary_count=len(seq),independent_input_clusters=len(inputs),
            positive_primary_count=sum(r.get('activity') is not None and r['activity']['all_active_aliases_busy_fraction']>0 for r in seq),
            fraction=legacy.stats([r['activity']['all_active_aliases_busy_fraction'] for r in seq if r.get('activity')]),inputs=inputs)
        passed=passed and alias_pass
    return dict(status='SUFFICIENT_OBSERVED_JOINT_COVERAGE' if passed else 'INSUFFICIENT_OBSERVED_JOINT_COVERAGE',
        passed=passed,threshold=threshold,min_repeats_per_input=min_repeats,all_primary_positive_required=True,aliases=aliases,
        scope='Completed observed handler intersection only; not GPU execution, utilization or a predictor feature.')


def repetition_pairs(baseline,joint):
    """Pair same input AND repetition; support is never eligible."""
    a={(r['model'],r['pool_id'],r['repetition']):r for r in baseline['request_diagnostics'] if r['role']=='primary'}
    output=[]
    for b in joint['request_diagnostics']:
        if b['role']!='primary':continue
        key=(b['model'],b['pool_id'],b['repetition']);left=a.get(key)
        if left is None:continue
        x,y=left.get('service_ms'),b.get('service_ms')
        output.append(dict(alias=key[0],pool_id=key[1],repetition=key[2],baseline_job=baseline['job_id'],joint_job=joint['job_id'],
            baseline_service_ms=x,joint_service_ms=y,joint_over_baseline=y/x if x and y else None,
            joint_activity=b.get('activity'),prediction_same=(left['worker_response'].get('prediction')==b['worker_response'].get('prediction'))
                if left['quality']['status']==b['quality']['status']=='AVAILABLE' else None,
            generated_tokens_same=(left['worker_response'].get('generated_tokens')==b['worker_response'].get('generated_tokens'))
                if left['service_status']==b['service_status']=='AVAILABLE' else None))
    return output


def repeatability_gate(first,second,lower=.5,upper=2.):
    """Per input median of covered (>=.90 overlap) primary services across C blocks."""
    a={(r['alias'],r['pool_id']):r for r in first['per_input']}
    b={(r['alias'],r['pool_id']):r for r in second['per_input']}
    output=[];passed=True
    for key in sorted(set(a)|set(b)):
        left,right=a.get(key),b.get(key)
        x=left['covered_primary_service_ms']['p50'] if left else None;y=right['covered_primary_service_ms']['p50'] if right else None
        complete=left is not None and right is not None and left['covered_primary_service_ms']['count']>=2 and right['covered_primary_service_ms']['count']>=2
        ratio=y/x if complete and x and y else None
        good=ratio is not None and lower<=ratio<=upper
        output.append(dict(alias=key[0],pool_id=key[1],first_median_service_ms=x,second_median_service_ms=y,
            second_over_first=ratio,at_least_two_covered_primary_each_block=complete,passes=good))
        passed=passed and good
    expected={(alias,pool_id) for alias in first['active_aliases'] for pool_id in first['input_pool_ids']}
    passed=passed and set(a)==set(b)==expected and len(expected)==96
    return dict(status='WITHIN_PREREGISTERED_REPEATABILITY_RANGE' if passed else 'INSUFFICIENT_CROSS_BLOCK_REPEATABILITY',
        passed=passed,lower_inclusive=lower,upper_inclusive=upper,per_input=output,
        scope='32 input clusters per alias, two development blocks; descriptive reproducibility screen, not causal or independent validation.')


def measurement_timing(rows,window,active_aliases):
    start=window.get('start_time_ns');end=window.get('end_time_ns');primary=window.get('primary_completion_time_ns')
    stop=window.get('support_admission_stop_time_ns');flags=[]
    def duration(a,b):return (b-a)/1e9 if type(a) is int and type(b) is int and b>=a else None
    aliases={}
    for alias in active_aliases:
        seq=[r for r in rows if r['model']==alias]
        p=[r for r in seq if r['role']=='primary'];s=[r for r in seq if r['role']=='support']
        successful=[r for r in p if r['quality']['status']=='AVAILABLE']
        finish=max((r['original_record'].get('completion_time_ns',-1) for r in successful),default=None) if len(successful)==96 else None
        declared=(window.get('alias_completion_time_ns') or {}).get(alias)
        if finish!=declared:flags.append(dict(alias=alias,reason='Declared alias primary completion differs from complete primary records'))
        support_released=sum(type(r['original_record'].get('scheduled_time_ns')) is int for r in s)
        if (window.get('primary_completed_by_alias') or {}).get(alias)!=len(successful):flags.append(dict(alias=alias,reason='Declared primary_completed_by_alias differs'))
        if (window.get('support_released_by_alias') or {}).get(alias)!=support_released:flags.append(dict(alias=alias,reason='Declared support_released_by_alias differs'))
        primary_released=sum(type(r['original_record'].get('scheduled_time_ns')) is int for r in p)
        if (window.get('released_by_alias') or {}).get(alias)!=primary_released+support_released:flags.append(dict(alias=alias,reason='Declared total released_by_alias differs'))
        for r in s:
            scheduled=r['original_record'].get('scheduled_time_ns')
            if type(scheduled) is int and type(stop) is int and scheduled>stop:flags.append(dict(request_id=r['request_id'],reason='Support admitted after stop'))
            if type(scheduled) is int and (type(finish) is not int or scheduled<finish):flags.append(dict(request_id=r['request_id'],reason='Support admitted before own primary cohort completed'))
        aliases[alias]=dict(primary_completion_time_ns=finish,release_to_primary_completion_seconds=duration(start,finish),
            primary_completed=len(successful),support_released=support_released)
    finishes=[r['primary_completion_time_ns'] for r in aliases.values()]
    observed_primary=max(finishes) if finishes and all(type(t) is int for t in finishes) else None
    if primary!=observed_primary:flags.append(dict(reason='Declared global primary completion differs'))
    if type(primary) is int and type(stop) is int and stop<primary:flags.append(dict(reason='Support admission stop precedes final primary completion'))
    if duration(start,end) is None:flags.append(dict(reason='Invalid measurement window'))
    for row in rows:
        rec=row['original_record'];scheduled=rec.get('scheduled_time_ns');completion=rec.get('completion_time_ns')
        if type(scheduled) is int and type(start) is int and scheduled<start:flags.append(dict(request_id=row['request_id'],reason='Admission before measurement start'))
        if type(completion) is int and type(end) is int and completion>end:flags.append(dict(request_id=row['request_id'],reason='Completion after measurement end'))
    if type(primary) is int and duration(primary,end) is None:flags.append(dict(reason='Primary completion after measurement end'))
    return dict(measurement_seconds=duration(start,end),primary_window_seconds=duration(start,primary),
        support_drain_seconds=duration(primary,end),primary_to_admission_stop_seconds=duration(primary,stop),
        aliases=aliases,contract_flags=flags,measurement_window=window,
        scope='Finite closed-loop timing; support and primary have distinct roles. No open-loop sustainable rate or utility.')


def validate_job(job,aliases):
    primary=legacy.validate_job(job,aliases)
    if any(r.get('role')!='primary' or r['request_id']!=f"measurement:{r['model']}:{r['sequence']}" for r in primary):raise ValueError('Primary role or request identity differs')
    support=job.get('support_requests')
    if not isinstance(support,dict) or set(support)!=set(job['active_aliases']):raise ValueError('Missing per-alias planned support roster')
    ids={r['request_id'] for r in primary};pool=set(job['input_pool_ids'])
    for alias in job['active_aliases']:
        seq=support[alias]
        if len(seq)!=1024:raise ValueError('Support must be predeclared at exactly1024 per alias')
        for i,row in enumerate(seq):
            if row.get('role')!='support' or row.get('model')!=alias or row.get('task')!='text' or row.get('pool_id') not in pool:
                raise ValueError('Invalid support role/identity')
            if row.get('support_sequence')!=i or row.get('sequence')!=96+i or row.get('arrival_index')!=96+i or row.get('repetition')!=i//32:
                raise ValueError('Invalid independent support cursor')
            if row['request_id']!=f'support:{alias}:{i}':raise ValueError('Invalid support request identity')
            if row['request_id'] in ids:raise ValueError('Repeated primary/support request_id')
            ids.add(row['request_id'])
        for begin in range(0,1024,32):
            if {r['pool_id'] for r in seq[begin:begin+32]}!=pool:raise ValueError('Support cycle not balanced32 inputs')
    return primary,[r for alias in job['active_aliases'] for r in support[alias]]


def analyze_job(packet,job,raw,packet_sha,references,score_result,sample_totals,runtime_checks):
    primary,support=validate_job(job,packet['aliases'])
    if raw.get('status') not in {'completed','completed_with_failures','failed'}:raise ValueError('Raw not terminal')
    if raw.get('schema')!='gsdeploy.sustained_joint_capacity.raw.v1' or raw.get('protocol')!=packet['protocol']:
        raise ValueError('Wrong sustained raw schema/protocol')
    if raw.get('job')!=job or raw.get('job_identity_sha256')!=job['job_identity_sha256'] or raw.get('packet_sha256')!=packet_sha:
        raise ValueError('Packet/job raw identity differs')
    if raw.get('replica_to_family')!=packet['aliases'] or raw.get('plan',{}).get('deployment')!=packet['resident_deployment']:
        raise ValueError('Resident aliases/deployment differs')
    actual=raw.get('request_records')
    if not isinstance(actual,list):raise ValueError('Missing request records')
    if any(r.get('status')=='completed' for r in actual) and runtime_checks.get('verified') is not True:
        raise ValueError('Completed requests require verified frozen runtime identity')
    expected={r['request_id']:r for r in primary+support};by_id={}
    for rec in actual:
        rid=rec.get('request_id')
        if rid not in expected or rid in by_id:raise ValueError('Unexpected or duplicate measurement record')
        if rec.get('phase')!='measurement' or any(rec.get(k)!=v for k,v in expected[rid].items()):raise ValueError('Immutable measurement template differs')
        by_id[rid]=rec
    for alias in job['active_aliases']:
        released=[r['support_sequence'] for r in actual if r['model']==alias and r['role']=='support']
        if sorted(released)!=list(range(len(released))):raise ValueError('Recorded support is not a contiguous predeclared prefix')
    planned=primary+[r for r in support if r['request_id'] in by_id]
    rows=[];intervals=[]
    for plan in planned:
        rec=by_id.get(plan['request_id'],dict(plan,status='NOT_RECORDED'));response=rec.get('worker_response') or {}
        result=dict(**plan,recorded=plan['request_id'] in by_id,original_record=rec,original_record_sha256=legacy.digest(rec),
            worker_response=response,service_status='UNKNOWN',service_ms=None,handler_ms=None,ledger=None)
        try:
            if rec.get('status')!='completed':raise ValueError('Request not completed')
            if response.get('request_id')!=rec['request_id'] or response.get('pool_id')!=rec['pool_id'] or response.get('task_type',response.get('task'))!=rec['task']:
                raise ValueError('Native response identity differs')
            if response.get('submit_ns')!=rec.get('actual_submit_unix_ns') or type(response.get('submit_ns')) is not int:raise ValueError('Native submit differs')
            start,end=response.get('start_ns'),response.get('end_ns');service=response.get('service_ms')
            if type(start) is not int or type(end) is not int or end<=start:raise ValueError('Invalid handler')
            if type(service) not in (int,float) or not math.isfinite(service) or service<=0:raise ValueError('Invalid native service')
            result.update(service_status='AVAILABLE',service_ms=service,handler_ms=(end-start)/1e6,
                ledger=legacy.ledger(rec,response),worker_start_unix_ns=start,worker_end_unix_ns=end)
            result['quality']=legacy.score_native(rec,response,references,score_result,sample_totals)
            intervals.append(dict(alias=rec['model'],start_ns=start,end_ns=end))
        except (KeyError,TypeError,ValueError) as exc:
            result['service_unknown_reason']=str(exc);result['quality']=dict(status='UNKNOWN',reason=str(exc),loss=None)
        rows.append(result)
    activity=legacy.common_activity(intervals,job['active_aliases'],sum(r['service_status']!='AVAILABLE' for r in rows))
    for r in rows:
        r['activity']=legacy.interval_exposure(r['worker_start_unix_ns'],r['worker_end_unix_ns'],activity['segments'],job['active_aliases'],r['model']) if r['service_status']=='AVAILABLE' else None
    per_input=[];aliases={};repeats=[];contract_flags=[]
    for alias in job['active_aliases']:
        seq=[r for r in rows if r['model']==alias];p=[r for r in seq if r['role']=='primary'];s=[r for r in seq if r['role']=='support']
        aliases[alias]=dict(primary=legacy.summarize_requests(p),support=legacy.summarize_requests(s),predeclared_support_cap=1024,
            support_never_released=1024-len(s),independent_primary_input_clusters=32)
        for pool_id in sorted(set(job['input_pool_ids'])):
            part=[r for r in p if r['pool_id']==pool_id]
            per_input.append(dict(alias=alias,pool_id=pool_id,**legacy.summarize_requests(part),
                covered_primary_service_ms=legacy.stats([r['service_ms'] for r in part if r['activity'] and r['activity']['all_active_aliases_busy_fraction']>=.90])))
        for repetition in range(3):
            part=[r for r in p if r['repetition']==repetition]
            repeats.append(dict(alias=alias,repetition=repetition,**legacy.summarize_requests(part),
                joint_fraction=legacy.stats([r['activity']['all_active_aliases_busy_fraction'] for r in part if r['activity']])))
        submitted=sorted([r for r in seq if type(r['original_record'].get('scheduled_time_ns')) is int],key=lambda r:r['original_record']['scheduled_time_ns'])
        for before,after in zip(submitted,submitted[1:]):
            completion=before['original_record'].get('completion_time_ns');submit=after['original_record'].get('actual_submit_time_ns')
            if type(completion) is not int or type(submit) is not int or submit<completion:
                contract_flags.append(dict(alias=alias,request_id=after['request_id'],reason='One-inflight completion ledger broken/unknown'))
    timing=measurement_timing(rows,raw.get('measurement_window') or {},job['active_aliases'])
    contract_flags.extend(timing['contract_flags'])
    for r in rows:
        if r['service_status']=='AVAILABLE' and (r['ledger']['status']!='AVAILABLE' or r['ledger']['negative_time_flags']):
            contract_flags.append(dict(request_id=r['request_id'],reason='Timestamp ledger unavailable or negative'))
    for alias,overlaps in activity['serial_overlap_flags'].items():
        if overlaps:contract_flags.append(dict(alias=alias,reason='Serial worker handler overlap',overlaps=overlaps))
    shutdown=raw.get('shutdown') or {};clean_shutdown=set(shutdown)==set(packet['aliases']) and all(r.get('exit_code')==0 and not r.get('forced') and r.get('bye_received') for r in shutdown.values())
    memory=raw.get('memory_evidence') or {};energy=raw.get('energy_evidence') or {}
    quality_complete=all(r['quality']['status']=='AVAILABLE' for r in rows)
    checks=dict(raw_completed=raw['status']=='completed',no_recorded_errors=not raw.get('errors'),all_primary_and_released_support_native_scored=quality_complete,
        resource_valid=memory.get('resource_valid') is True,energy_coverage_valid=energy.get('coverage_valid') is True and not energy.get('sampling_errors'),
        clean_shutdown=clean_shutdown,serial_and_window_contract=not contract_flags,runtime_identity=runtime_checks.get('verified') is True,
        priming_and_warmup_native_success=runtime_checks.get('phase_native_success_complete')=={'priming':True,'warmup':True})
    cov=coverage_gate(rows,job['active_aliases'],job['state'])
    status='COMPLETE_DESCRIPTIVE_DIAGNOSTIC' if all(checks.values()) and cov['passed'] else 'INSUFFICIENT_DIAGNOSTIC_EVIDENCE'
    return dict(schema='gsdeploy.sustained_capacity_analysis.job.v1',job_id=job['job_id'],state=job['state'],block=job['block'],seed=job['seed'],
        active_aliases=job['active_aliases'],input_pool_ids=sorted(set(job['input_pool_ids'])),status=status,raw_status=raw['status'],
        runtime_execution_identity=runtime_checks,diagnostic_checks=checks,coverage=cov,aliases=aliases,per_input=per_input,per_repetition=repeats,
        primary=legacy.summarize_requests([r for r in rows if r['role']=='primary']),support=legacy.summarize_requests([r for r in rows if r['role']=='support']),
        measurement_timing=timing,common_activity=activity,contract_flags=contract_flags,request_diagnostics=rows,errors=raw.get('errors',[]),
        resource_evidence={k:v for k,v in memory.items() if k!='samples'},energy_evidence=energy,
        resource_energy_verification_scope='Raw runner declarations preserved. Independent audit is a separate artifact; this analyzer does not independently integrate energy or recompute memory.',
        scope='DEVELOPMENT_PRIMARY32_INPUT_CLUSTER_DIAGNOSTIC_NOT_FORMAL_VALIDATION',U=None,diagnostic_U=None,
        fitted_parameters=False,ranking_performed=False,independent_formal_validation=False,
        limitations=['Primary is32 input clusters times3 repeated observations; support adds no independent inputs.',
            'Handler overlap is descriptive wallclock exposure, not GPU utilization or causal evidence.',
            'Closed-loop timing is not sustainable open-loop capacity; no continuous rate/share model is fitted.',
            'Support is excluded from primary input medians, formal Q, throughput utility and all utility denominators.',
            'Failures and missing primary remain UNKNOWN; completed-only distributions may be censored.'])


def inspect_packet(path):
    path=Path(path).resolve();packet=json.loads(path.read_text())
    if packet.get('schema_version')!=1 or packet.get('protocol')!='sustained_joint_primary_support_v1' or packet.get('diagnostic_only') is not True or packet.get('U') is not None:
        raise ValueError('Expected explicit sustained diagnostic packet with U=null')
    if packet.get('aliases')!={'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'} or packet.get('resident_deployment')!={'text':2,'generalist_2':1,'ocr':1}:
        raise ValueError('Unexpected fixed resident context')
    if packet.get('support_limit_per_alias')!=1024:raise ValueError('Support cap differs')
    policy=dict(minimum_common_handler_fraction=.9,minimum_covered_repetitions_per_input=2,all_primary_positive_overlap=True,
        cross_block_per_input_median_ratio_min=.5,cross_block_per_input_median_ratio_max=2.)
    if packet.get('coverage_policy')!=policy:raise ValueError('Preregistered coverage policy differs')
    for key,value in [('warmup_seconds',30),('measurement_timeout_seconds',600),('request_timeout_seconds',300)]:
        if packet.get(key)!=value:raise ValueError('Frozen timing differs: '+key)
    bindings=packet.get('bindings') or {}
    if not bindings or packet.get('bindings_sha256')!=legacy.digest(bindings):raise ValueError('Missing/different frozen bindings')
    for filename,sha in bindings.items():
        if not Path(filename).is_absolute() or legacy.file_receipt(filename)['sha256']!=sha:raise ValueError('Frozen file binding differs: '+filename)
    for key in ['system_config','samples','workers','admission']:
        if not Path(packet[key]).is_absolute() or str(Path(packet[key])) not in bindings:raise ValueError('Missing bound path: '+key)
    source=Path(packet['frozen_source']);files=list((source/'gsdeploy').rglob('*.py'))
    if not files or any(str(p) not in bindings for p in files):raise ValueError('Frozen Python package not fully bound')
    if packet.get('priming_policy')!='largest_three_all_resident_supported_tasks' or len(packet.get('priming_requests',[]))!=21:
        raise ValueError('Full resident lazy-path priming declaration required')
    jobs=packet['jobs']
    if [(j['block'],j['state'],j['seed']) for j in jobs]!=[(1,'B',103),(1,'C',103),(2,'C',104),(2,'B',104)] or len({j['job_id'] for j in jobs})!=4:
        raise ValueError('Unexpected four-job development roster')
    sample_rows=[json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    pool={r['pool_id'] for r in sample_rows if r['task_type']=='text'}
    if len(pool)!=32:raise ValueError('Expected bound32 text input pool')
    block_orders={}
    for job in jobs:
        validate_job(job,packet['aliases'])
        active={'B':['generalist_2#0'],'C':['text#0','text#1','generalist_2#0']}[job['state']]
        if job['active_aliases']!=active:raise ValueError('Active roster differs')
        if set(job['input_pool_ids'])!=pool or any([r['pool_id'] for r in job['planned_requests'][alias]]!=job['input_pool_ids'] for alias in active):
            raise ValueError('Bound pool/alias input order differs')
        if block_orders.setdefault(job['block'],job['input_pool_ids'])!=job['input_pool_ids']:raise ValueError('Within-block B/C input order differs')
    config=json.loads(Path(packet['system_config']).read_text());workers=json.loads(Path(packet['workers']).read_text())
    if config['gpu_memory_mib']!=packet['gpu_memory_mib'] or config['catalog']['safe_factor']!=packet['safe_factor']:raise ValueError('Resource declarations differ')
    references=config['catalog']['quality']['references']
    if set(references['text'])!=pool:raise ValueError('Frozen quality reference pool differs')
    scorer,total=legacy.frozen_scorers(source)
    metadata=dict(schema='gsdeploy.sustained_capacity_analysis.readiness.v1',packet=legacy.file_receipt(path),
        bindings=[dict(path=k,sha256=v) for k,v in bindings.items()],tool=legacy.file_receipt(__file__),
        shared_analysis_source=legacy.file_receipt(Path(__file__).with_name('analyze.py')),
        jobs=[{k:j[k] for k in ['job_id','state','block','seed','active_aliases','job_identity_sha256']} for j in jobs],
        raw_files_read=0,progress_read=False,gpu_started=False,U=None)
    return packet,config,workers,references,scorer,total,metadata


def registered_paths(packet,paths,packet_sha):
    declared={j['job_id']:j for j in packet['jobs']};registered={};statuses=[]
    for path in paths:
        progress=json.loads(Path(path).read_text());statuses.append(progress.get('status'))
        if progress.get('schema')!='gsdeploy.sustained_capacity_queue.v1':raise ValueError('Wrong sustained queue schema')
        if progress.get('status') not in {'RUNNING','ALL_FOUR_SUSTAINED_DIAGNOSTIC_ATTEMPTS_COMPLETE','PAUSED_ON_FAILED_ATTEMPT_NO_RETRY','PAUSED_INFRASTRUCTURE_ERROR_NO_RETRY'}:
            raise ValueError('Unknown queue status')
        if progress.get('packet_sha256')!=packet_sha:raise ValueError('Queue packet identity differs')
        for row in progress.get('jobs',[]):
            job_id=row.get('job_id')
            if job_id not in declared:raise ValueError('Undeclared queue job')
            if job_id in registered and registered[job_id]!=row:raise ValueError('Conflicting registrations; no attempt selection')
            registered[job_id]=row
    waiting=[];ready=[]
    for job in packet['jobs']:
        row=registered.get(job['job_id'])
        if not row or row.get('status') not in {'completed','completed_with_failures','failed'}:
            waiting.append(dict(job_id=job['job_id'],status=row.get('status') if row else 'NOT_REGISTERED'));continue
        source=row.get('raw') or {};path=Path(source.get('path',''))
        if path.name!='run.json' or not source.get('sha256') or not path.is_absolute() or not path.is_file():
            waiting.append(dict(job_id=job['job_id'],status='PUBLISHED_FINAL_RECEIPT_MISSING'));continue
        ready.append((job,row,path))
    if ready and not waiting and any(s!='ALL_FOUR_SUSTAINED_DIAGNOSTIC_ATTEMPTS_COMPLETE' for s in statuses):
        waiting.append(dict(job_id=None,status='WAITING_ALL_FOUR_QUEUE_PUBLICATION'))
    return waiting,ready


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--queue-progress',type=Path,nargs='+')
    p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--partial',action='store_true')
    args=p.parse_args();packet,config,workers,references,scorer,total,metadata=inspect_packet(args.packet)
    if args.check_only:print(json.dumps(metadata));return
    if not args.queue_progress or not args.output:p.error('Analysis requires explicit queue progress and a new output directory')
    progresses=[legacy.file_receipt(path) for path in args.queue_progress]
    waiting,ready=registered_paths(packet,args.queue_progress,metadata['packet']['sha256'])
    if not ready or waiting and not args.partial:
        print(json.dumps(dict(status='WAITING_TERMINAL_RECEIPTS',waiting=waiting,raw_files_read=0,gpu_started=False,U=None)));raise SystemExit(3)
    if args.output.exists():raise FileExistsError('Never overwrite diagnostic evidence')
    reports=[]
    for job,registration,path in ready:
        raw,source=legacy.read_final(path,registration['raw']['sha256'])
        if raw.get('status')!=registration['status']:raise ValueError('Raw/receipt terminal status differs')
        runtime=legacy.runtime_identity_checks(packet,raw,config,workers)
        report=analyze_job(packet,job,raw,metadata['packet']['sha256'],references,scorer,total,runtime)
        report.update(raw=source);reports.append(report)
    by_key={(r['block'],r['state']):r for r in reports};pairs=[];input_pairs=[]
    for block in [1,2]:
        if (block,'B') in by_key and (block,'C') in by_key:
            pairs.extend(repetition_pairs(by_key[(block,'B')],by_key[(block,'C')]))
            input_pairs.extend(legacy.pair_job_results(by_key[(block,'B')],by_key[(block,'C')],f'block{block}_B_to_C'))
    repeatability=repeatability_gate(by_key[(1,'C')],by_key[(2,'C')]) if (1,'C') in by_key and (2,'C') in by_key else dict(status='INSUFFICIENT_CROSS_BLOCK_REPEATABILITY',passed=False,reason='Both terminal C blocks required')
    complete=not waiting and len(reports)==4
    sufficient=complete and all(r['status']=='COMPLETE_DESCRIPTIVE_DIAGNOSTIC' for r in reports) and repeatability['passed']
    status='COMPLETE_SUSTAINED_DEVELOPMENT_DIAGNOSTICS' if sufficient else 'INSUFFICIENT_DIAGNOSTIC_EVIDENCE'
    inputs=[metadata['packet'],metadata['tool'],metadata['shared_analysis_source'],*metadata['bindings'],*progresses,*[r['raw'] for r in reports]]
    for item in inputs:
        if legacy.file_receipt(item['path'])['sha256']!=item['sha256']:raise ValueError('Analysis input/source changed')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data):(args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    overview=[]
    for report in reports:
        filename=report['job_id']+'.json';save(filename,report)
        small={k:v for k,v in report.items() if k not in ['request_diagnostics','per_input','common_activity','energy_evidence']}
        small.update(detail_file=filename,common_activity={k:v for k,v in report['common_activity'].items() if k!='segments'},
            energy_evidence={k:v for k,v in report['energy_evidence'].items() if k!='samples'})
        overview.append(small)
    save('report.json',dict(schema='gsdeploy.sustained_capacity_analysis.v1',status=status,all_four_terminal=complete,
        jobs=overview,paired_primary_repetitions=pairs,paired_input_primary_all_three_medians=input_pairs,
        repeatability=repeatability,waiting=waiting,partial_requested=args.partial,raw_files_read=len(reports),
        U=None,diagnostic_U=None,fitted_parameters=False,ranking_performed=False,independent_formal_validation=False,gpu_started=False))
    def fmt(v):return 'UNKNOWN' if v is None else f'{v:.4g}'
    lines=['# 持续共同活动开发诊断','',f'状态：{status}；终态 {len(reports)}/4。','',
        '| job | alias | primary有效/计划 | support有效/登记 | primary EM | primary服务中位数ms |','|---|---|---:|---:|---:|---:|']
    for r in reports:
        for alias,a in r['aliases'].items():
            x,y=a['primary'],a['support'];lines.append(f"| {r['job_id']} | {alias} | {x['service_available_requests']}/{x['planned_requests']} | {y['service_available_requests']}/{y['planned_requests']} | {fmt(x['diagnostic_native_accuracy'])} | {fmt(x['service_ms']['p50'])} |")
    lines.extend(['','| job | 测量秒 | primary窗口秒 | support排空秒 | 覆盖门槛 |','|---|---:|---:|---:|---|'])
    for r in reports:
        t=r['measurement_timing'];lines.append(f"| {r['job_id']} | {fmt(t['measurement_seconds'])} | {fmt(t['primary_window_seconds'])} | {fmt(t['support_drain_seconds'])} | {r['coverage']['status']} |")
    lines.extend(['',f"跨block持续共同活动服务可重复性：{repeatability['status']}。",'',
        'C中每alias必须有96条primary全部正共同活动；每个输入3轮至少2轮handler共同重叠比例≥0.90。跨block每输入只用满足≥0.90的primary服务中位数，比例须在[0.5,2.0]。缺失或超界不能平均成通过。','',
        '同输入同轮B/C配对、所有primary输入中位数、covered-primary跨block比值见report.json。逐请求、逐输入覆盖、primary/support全量原生质量与时序见各job JSON。','',
        'primary为32个输入簇×3轮，support不增加独立输入，不进入正式效用分母。support全量计入实测活动区间、资源和整卡能量。','',
        '资源与能量为runner原始声明，本工具不冒充独立审计。有限闭环速率不是开放负载稳定容量；handler交叠不是GPU利用率或因果结论。','',
        'U=null；不拟合容量、不排序、不选择部署，不据此宣称正式验证或分流收益成立。'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    save('protocol.json',dict(metadata=metadata,queue_receipts=progresses,raw_files_read=len(reports),progress_read=True,scope=__doc__,U=None))
    save('manifest.json',dict(inputs=[metadata['packet'],*progresses,*[r['raw'] for r in reports]],source_bindings=metadata['bindings'],
        tool=metadata['tool'],shared_analysis_source=metadata['shared_analysis_source'],
        files={path.name:legacy.file_receipt(path)['sha256'] for path in args.output.iterdir() if path.is_file()}))
    print(json.dumps(dict(status=status,jobs=len(reports),output=str(args.output.resolve()),U=None,gpu_started=False)))


if __name__=='__main__':main()
