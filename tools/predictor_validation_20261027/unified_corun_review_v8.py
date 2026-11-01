"""CPU-only strict review of one unified-session training cohort and its work pair."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3

from .native_collect import receipt,save
from .paired_training_work_review_v4 import actual_work
from .profiling import RESIDENTS
from .smoke_verifier_v3 import check_receipt,load_cpu_adapter,validate_output,validate_request,validate_parsed
from .startup_telemetry_review_v4_1 import validate_startup_telemetry
from .startup_training_collection_review_v4_1 import coverage
from .trace_export import export_sqlite,parse_sqlite,linux_pid
from .unified_corun_collect_v8 import audit,build_registry,priming_items,outer_command,LIGHT,NATURAL,RAW_SCHEMA

REVIEW_SCHEMA='gsdeploy.unified_corun_strict_review.v8'


def trace_metadata(path):
    """Read native same-session metadata; do not guess scheduler/CUPTI joins."""
    connection=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in connection.execute('select name from sqlite_master where type="table"')}
        rows=lambda name:[dict(r) for r in connection.execute('select * from "'+name+'"')] if name in tables else []
        sessions=rows('TARGET_INFO_SESSION_START_TIME');diagnostics=rows('DIAGNOSTIC_EVENT')
        switches=rows('GPU_CONTEXT_SWITCH_EVENTS')
        contexts=rows('TARGET_INFO_CUDA_CONTEXT_INFO');processes=rows('PROCESSES')
        kernel_groups=[]
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
            kernel_groups=[dict(r) for r in connection.execute('select globalPid,contextId,count(*) count,min(start) first_start_ns,max(end) last_end_ns from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid,contextId')]
        issues=[]
        if len(sessions)!=1 or any(type(sessions[0].get(k)) is not int for k in ('utcEpochNs','systemClockNs')):
            issues.append('EXACT_SINGLE_SESSION_CLOCK_REQUIRED')
        if 'DIAGNOSTIC_EVENT' not in tables:issues.append('TRACE_DIAGNOSTICS_MISSING')
        dangerous=re.compile(r'\b(dropped|dropping|lost|overflow|truncated|truncation)\b',re.I)
        if any(r.get('severity',0)>=3 or dangerous.search(r.get('text','')) for r in diagnostics):
            issues.append('TRACE_ERROR_OR_DATA_LOSS_DIAGNOSTIC')
        if not switches:issues.append('GPU_CONTEXT_SWITCH_RECORDS_REQUIRED_FOR_UNIFIED_CAPABILITY')
        return dict(issues=issues,session=sessions,diagnostics=diagnostics,
            switch_event_count=len(switches),switch_pid_values=sorted({linux_pid(r['globalPid']) for r in switches if r.get('globalPid') is not None}),
            switch_context_ids=sorted({r['contextId'] for r in switches}),
            switch_time_extent_ns=[min(r['timestamp'] for r in switches),max(r['timestamp'] for r in switches)] if switches else None,
            switches_by_tag=dict(Counter(str(r['tag']) for r in switches)),
            CUDA_context_metadata=contexts,process_metadata=processes,kernel_groups=kernel_groups,
            owned_scheduler_attribution='UNKNOWN',
            attribution_note='Recorded namespace views are evidence only. No host/container arithmetic PID mapping or CUPTI/driver context-ID equality join is performed.',
            same_sm_concurrency='UNKNOWN',calibrated_cross_clock_error_bound_ns=None,
            timestamp_domain='native single Nsight SQLite relative nanoseconds; do not apply independent median offsets to inter-owner intervals')
    finally:connection.close()


def owner_view(parsed,pid):
    return dict(phases=[p for p in parsed['phases'] if p['pid']==pid],
        clock_mappings={str(pid):parsed.get('clock_mappings',{}).get(str(pid),{})})


def canonical_measured(parsed,raw_receipt):
    """Serialize each measured phase kernel once; full session stays in SQLite."""
    return dict(schema='gsdeploy.unified_measured_phases.v8',sources=parsed['sources']+[raw_receipt],
        phases=parsed['phases'],clock_mappings=parsed['clock_mappings'],
        unmatched_anchor_count=parsed['unmatched_anchor_count'],
        full_session_nvtx_range_count=len(parsed.get('nvtx_ranges',[])),
        full_session_unattributed_kernel_count=len(parsed.get('unattributed_kernels',[])),
        timestamp_domain='nsys_start_ns/nsys_end_ns retain the single common session clock',
        worker_mapping='start_ns/end_ns are descriptive per-owner median maps only, not common-clock reconstruction',
        scope='measured request phases only; full startup/priming/copy/sync/runtime/scheduler evidence remains in source SQLite',
        demand=None,same_sm_concurrency='UNKNOWN')


def collection_issues(plan,context,raw,executed,execution):
    issues=[];traced=plan['domain']==LIGHT
    if (raw.get('schema')!=RAW_SCHEMA or raw.get('status')!='completed' or raw.get('errors')
            or raw.get('runtime_errors') or raw.get('telemetry_errors')):issues.append('COLLECTION_ERRORS_OR_SCHEMA')
    if raw.get('execution_context_sha256')!=plan['execution_context_sha256']:issues.append('RAW_CONTEXT_MISMATCH')
    if (execution.get('schema')!='gsdeploy.unified_corun_execution.v8'
            or execution.get('status')!='COLLECTED' or execution.get('domain')!=plan['domain']
            or execution.get('exception_cleanup')):
        issues.append('CONTROLLER_COLLECTION_NOT_QUALIFIED')
    watchdog=execution.get('watchdog',{})
    if (watchdog.get('status')!='EXITED' or watchdog.get('return_code')!=0
            or watchdog.get('cleanup',{}).get('live_survivors')!=[] or not watchdog.get('samples')):
        issues.append('WATCHDOG_OR_OWNED_CLEANUP_FAILED')
    budget=context['collection_budget']
    if (execution.get('watchdog_interval_seconds')!=budget['watchdog_interval_seconds']
            or execution.get('free_bytes_at_end',0)<budget['free_disk_reserve_bytes']
            or any(s.get('free_bytes',0)<budget['free_disk_reserve_bytes'] for s in watchdog.get('samples',[]))):
        issues.append('DISK_FLOOR_OR_WATCHDOG_IDENTITY_FAILED')
    if (executed.get('trace') is not traced or executed.get('single_session') is not traced
            or executed.get('capture_mode')!='whole_lifetime' or executed.get('child_profilers') is not False
            or executed.get('profile_controls_sent') is not False or raw.get('child_profilers') is not False
            or raw.get('profile_controls_sent') is not False
            or raw.get('performance_scope')!=('single_session_profiler_perturbed' if traced else 'unprofiled_with_phase_markers')
            or raw.get('measurement_domain')!=plan['domain']):issues.append('PROFILER_DOMAIN_OR_CONTROL_MISMATCH')
    expected_commands={a:r['command'] for a,r in build_registry(context).items()}
    if raw.get('child_commands')!=expected_commands:issues.append('ORIGINAL_PLAIN_CHILD_COMMANDS_CHANGED')
    ready=raw.get('ready',{})
    if set(ready)!=set(RESIDENTS) or len({x.get('pid') for x in ready.values()})!=4:
        issues.append('FOUR_DISTINCT_OWNED_READY_PIDS_REQUIRED')
    shutdown=raw.get('shutdown',{})
    if set(shutdown)!=set(RESIDENTS) or any(r.get('exit_code')!=0 or r.get('forced') is not False or r.get('bye_received') is not True for r in shutdown.values()):
        issues.append('CLEAN_FOUR_OWNER_SHUTDOWN_REQUIRED')
    identities=raw.get('live_owned_processes',{})
    if set(identities)!=set(RESIDENTS):issues.append('LIVE_OWNER_IDENTITY_EVIDENCE_REQUIRED')
    for alias,identity in identities.items():
        if (identity.get('error') or identity.get('pid')!=ready.get(alias,{}).get('pid')
                or type(identity.get('start_ticks')) is not int or not identity.get('nspid')
                or identity.get('execution_nonce')!=context['execution_nonce']):issues.append('LIVE_OWNER_IDENTITY_INVALID:'+alias)
    if not raw.get('live_identity_events'):issues.append('STARTUP_PID_VIEWS_MISSING')
    expected=[(a,i['task'],i['pool_id']) for a,i in priming_items(plan)]
    observed=[(r.get('model'),r.get('task'),r.get('pool_id')) for r in raw.get('priming_request_records',[])]
    if observed!=expected or any(r.get('status')!='completed' for r in raw.get('priming_request_records',[])):
        issues.append('EXACT_ORIGINAL_TWELVE_PRIMING_REQUESTS_REQUIRED')
    issues+=coverage(plan,raw,executed);issues+=validate_startup_telemetry(raw)
    return issues


def review(run,output):
    run=Path(run).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    issues=[];owners={};trace=None;raw=None
    try:
        audit(run)
        plan=json.loads((run/'plan.json').read_text());context=json.loads((run/'context.json').read_text())
        raw=json.loads((run/'raw.json').read_text());executed=json.loads((run/'executed-jobs.json').read_text())
        execution=json.loads((run/'execution.json').read_text())
        if not check_receipt(execution.get('execution_start'),run/'execution-start.json'):
            issues.append('EXECUTION_INTENT_RECEIPT_FAILED')
        for field,name in [('plan','plan.json'),('context','context.json'),('executed_jobs','executed-jobs.json')]:
            if not check_receipt(raw.get(field),run/name):issues.append('RAW_INPUT_RECEIPT_FAILED:'+field)
        for field,name in [('plan','plan.json'),('context','context.json'),('raw','raw.json')]:
            if not check_receipt(execution.get(field),run/name):issues.append('EXECUTION_RECEIPT_FAILED:'+field)
        issues+=collection_issues(plan,context,raw,executed,execution)
        traced=plan['domain']==LIGHT
        expected_command=outer_command(run,context,plan['domain'])
        if execution.get('command')!=expected_command:issues.append('OUTER_PROFILER_COMMAND_CHANGED')
        reports=list(run.glob('*.nsys-rep'))
        if (traced and reports!=[run/'trace-unified.nsys-rep']) or (not traced and reports):issues.append('EXACT_ONE_OR_ZERO_PROFILER_REPORT_REQUIRED')
        pool_source=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        pool={r['pool_id']:r for r in map(json.loads,Path(pool_source).read_text().splitlines())}
        anchors=[];owner_rows={}
        for alias,family in RESIDENTS.items():
            binding=context['worker_bindings'][family];spec=json.loads(Path(binding['spec']).read_text());pid=raw['ready'][alias]['pid']
            if raw['ready'][alias].get('spec_sha256')!=receipt(binding['spec'])['sha256']:issues.append('READY_SPEC_IDENTITY_FAILED:'+alias)
            adapter=None if family=='ocr' else load_cpu_adapter(binding,spec)
            rows=[r for r in raw['request_records'] if r['model']==alias];owner_rows[alias]=rows
            for row in raw['priming_request_records']+rows:
                if row['model']==alias:issues+=validate_output(row,pool[row['pool_id']],spec,adapter)
            for row in rows:
                found,paired=validate_request(row,pid);issues+=found;anchors+=paired
                if 'affine_operator_observations' in row['worker_response'].get('semantic_forward_observations',{}):issues.append('AFFINE_HOOKS_OUTSIDE_DECLARED_DOMAIN')
            owners[alias]=dict(pid=pid,request_ids=[r['request_id'] for r in rows],kernel_evidence='UNKNOWN_UNPROFILED_DOMAIN')
        if traced:
            sql=output/'trace-unified.sqlite';export=export_sqlite(run/'trace-unified.nsys-rep',sql)
            metadata=trace_metadata(sql);issues+=metadata['issues']
            stderr=(run/'stderr.log').read_text()
            if re.search(r'\b(dropped|dropping|lost|overflow|truncated|truncation)\b',stderr,re.I):issues.append('PROFILER_STDERR_DATA_LOSS')
            parsed=parse_sqlite(sql,anchors)
            for alias,rows in owner_rows.items():
                if not rows:
                    owners[alias]['kernel_evidence']='NO_MEASURED_REQUESTS_FOR_RESIDENT_OWNER';continue
                found,summary=validate_parsed(owner_view(parsed,owners[alias]['pid']),[a for a in anchors if a['pid']==owners[alias]['pid']],rows,owners[alias]['pid'])
                issues += [s+':'+alias for s in found]
                owners[alias].update(kernel_evidence='VERIFIED_MEASURED_CUPTI_AND_NVTX' if not found else 'FAILED',summary=summary)
            canonical=canonical_measured(parsed,receipt(run/'raw.json'));save(output/'parsed-measured.json',canonical)
            save(output/'session-metadata.json',metadata)
            trace=dict(export=export,sqlite=receipt(sql),parsed=receipt(output/'parsed-measured.json'),metadata=receipt(output/'session-metadata.json'),
                common_session_clock_identified=not metadata['issues'],calibrated_cross_clock_error_bound_ns=None,
                owned_scheduler_attribution='UNKNOWN',same_sm_concurrency='UNKNOWN')
        audit(run)
    except Exception as exc:issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+': '+str(exc))
    evidence={name:receipt(run/name) for name in ('raw.json','plan.json','context.json','execution.json','executed-jobs.json') if (run/name).exists()}
    evaluators=[Path(__file__),Path(__file__).with_name('unified_corun_collect_v8.py'),
        Path(__file__).with_name('startup_training_collection_review_v4_1.py'),Path(__file__).with_name('startup_telemetry_review_v4_1.py'),
        Path(__file__).with_name('smoke_verifier_v3.py'),Path(__file__).with_name('trace_export.py'),Path(__file__).with_name('nvtx_interval_index_v2.py')]
    result=dict(schema=REVIEW_SCHEMA,status='PASS' if not issues else 'FAIL',issues=issues,owners=owners,trace=trace,
        evidence=evidence,evaluator_sources=[receipt(p) for p in evaluators],training_only=True,model_fitted=False,
        validation_started=False,natural_timing_transfer_validated=False,
        owned_scheduler_attribution='UNKNOWN',same_sm_concurrency='UNKNOWN',resource_bottleneck='UNKNOWN',
        capture_run_id='raw-sha256:'+receipt(run/'raw.json')['sha256'] if (run/'raw.json').exists() else None,
        independent_capture_count=1 if not issues else 0,
        interpretation='Qualification covers collection/work/phase and single-session trace evidence; no throughput, same-SM, owned scheduling or deployment utility claim.')
    save(output/'report.json',result);return result


def load_review(path,domain):
    report=json.loads(Path(path).read_text())
    if report.get('schema')!=REVIEW_SCHEMA or report.get('status')!='PASS':raise ValueError('Strict unified review PASS required')
    for r in list(report['evidence'].values())+report['evaluator_sources']:
        if not check_receipt(r):raise ValueError('Changed strict evidence/evaluator')
    if report['trace'] is not None:
        for key in ('sqlite','parsed','metadata'):
            if not check_receipt(report['trace'][key]):raise ValueError('Changed unified trace evidence')
        if not check_receipt(report['trace']['export']['source']):raise ValueError('Changed source Nsight report')
    plan=json.loads(Path(report['evidence']['plan.json']['path']).read_text());raw=json.loads(Path(report['evidence']['raw.json']['path']).read_text())
    if plan['domain']!=domain:raise ValueError('Explicit paired domain required')
    return report,plan,raw


def pair(light_report,natural_report,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    light,lplan,lraw=load_review(light_report,LIGHT);natural,nplan,nraw=load_review(natural_report,NATURAL)
    issues=[];pairs=[]
    if lplan['jobs']!=nplan['jobs']:issues.append('EXACT_PAIRED_JOB_MISMATCH')
    key=lambda r:(r['profile_job_id'],r['model'],r['pool_id'],r['task'])
    if Counter(map(key,lraw['request_records']))!=Counter(map(key,nraw['request_records'])):issues.append('EXACT_PAIRED_REQUEST_MISMATCH')
    if len(set(map(key,lraw['request_records'])))!=len(lraw['request_records']):issues.append('DUPLICATE_LIGHT_TRIAL')
    lookup={key(r):r for r in nraw['request_records']}
    for row in lraw['request_records']:
        other=lookup.get(key(row));same=other is not None and actual_work(row['worker_response'])==actual_work(other['worker_response'])
        if not same:issues.append('ACTUAL_WORK_OR_OUTPUT_MISMATCH:'+row['request_id'])
        pairs.append(dict(key=list(key(row)),actual_work_matched=same,light_request_id=row['request_id'],natural_request_id=other['request_id'] if other else None))
    primer=lambda r:(r['model'],r['task'],r['pool_id'],actual_work(r['worker_response']))
    priming_match=list(map(primer,lraw['priming_request_records']))==list(map(primer,nraw['priming_request_records']))
    if not priming_match:issues.append('ACTUAL_PRIMING_WORK_MISMATCH')
    result=dict(schema='gsdeploy.unified_corun_paired_actual_work.v8',status='PASS' if not issues else 'FAIL',issues=issues,
        pairs=pairs,priming_actual_work_matched=priming_match,light_review=receipt(light_report),natural_review=receipt(natural_report),
        evaluator=receipt(__file__),training_only=True,validation_started=False,duration_scaling_used=False,
        natural_kernel_timing_transfer_validated=False,owned_scheduler_attribution='UNKNOWN',same_sm_concurrency='UNKNOWN')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('review');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('pair');p.add_argument('--light-report',required=True);p.add_argument('--natural-report',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args();result=review(args.run,args.output) if args.op=='review' else pair(args.light_report,args.natural_report,args.output)
    print(json.dumps(dict(status=result['status'],issues=result['issues'],report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
