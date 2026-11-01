"""CPU-only review/export of complete repeated-owner training observations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import whole57_joint_training_v23 as collect
from .native_collect import receipt,save
from .joint_host_scenarios_v13 import light_prefixes
from .session_clock_audit_v2 import audit_one
from .smoke_verifier_v3 import validate_request,validate_parsed
from .startup_telemetry_review_v4_1 import validate_startup_telemetry
from .trace_export import export_sqlite,parse_sqlite
from .unified_corun_review_v8_1 import trace_metadata,owner_view,canonical_measured,classify_stderr_loss


def host_lifecycle_rows(plan,raw):
    issues=[];anchors=[];rows=[];origin=raw.get('phase_start_ns')
    if (type(origin) is not int or raw.get('planned_public_cohort')!=plan['trace']
            or len(raw.get('request_records',[]))!=57):
        return ['FULL57_EXECUTED_COHORT_AND_ORIGIN_REQUIRED'],[],[]
    deadline=raw.get('deadline_ns');offered_end=raw.get('arrival_loop_end_ns');observed_end=raw.get('observation_end_ns')
    if (deadline!=origin+round(collect.RECIPE['window_seconds']*1e9)
            or type(offered_end) is not int or type(observed_end) is not int
            or not deadline<=offered_end<=observed_end):
        issues.append('EXACT_ORIGINAL_WINDOW_OFFER_AND_TAIL_DRAIN_BOUNDARIES_REQUIRED')
    previous_ready={};pids={a:r['pid'] for a,r in raw.get('ready',{}).items()}
    all_rows=raw.get('priming_request_records',[])+raw['request_records']
    for row in all_rows:
        pid=pids.get(row['model']);found,paired=validate_request(row,pid);issues.extend(found);anchors.extend(paired)
        begin=row.get('worker_response',{}).get('start_perf_ns');ready=row.get('previous_handler_next_request_ready',{}).get('perf_ns')
        prior=previous_ready.get(row['model'])
        if type(begin) is int and type(prior) is int and begin<prior:issues.append('OWNER_NEXT_HANDLER_PRECEDES_PREVIOUS_READY:'+row['request_id'])
        previous_ready[row['model']]=ready
    for declared,row in zip(plan['trace'],raw['request_records']):
        rid=declared['request_id'];response=row.get('worker_response',{});ready=row.get('previous_handler_next_request_ready',{})
        if ((row.get('request_id'),row.get('task'),row.get('pool_id'),row.get('model'))
                !=(rid,declared['task'],declared['pool_id'],plan['routes'][rid])
                or row.get('scheduled_time_ns')!=origin+declared['offset_ns']
                or row.get('split')!='training' or row.get('validation_view')!='TRAINING_ONLY_NOT_VALIDATION'):
            issues.append('DECLARED_REQUEST_IDENTITY_OR_ARRIVAL_CHANGED:'+rid)
        fields=dict(scheduled_time_ns=row.get('scheduled_time_ns'),enqueued_time_ns=row.get('enqueued_time_ns'),
            dispatch_begin_ns=row.get('actual_submit_time_ns'),dispatch_write_completed_ns=row.get('submit_write_completed_time_ns'),
            handler_start_ns=response.get('start_perf_ns'),handler_end_ns=response.get('end_perf_ns'),
            parent_receipt_ns=row.get('response_reader_event_enqueue_time_ns'),
            parent_dequeue_ns=row.get('response_collector_dequeued_time_ns'),
            parent_processing_ns=row.get('response_collector_processing_time_ns'),worker_next_ready_ns=ready.get('perf_ns'))
        if any(type(v) is not int for v in fields.values()):issues.append('MISSING_ORIGINAL_HOST_BOUNDARY:'+rid)
        if all(type(fields[k]) is int for k in ('scheduled_time_ns','enqueued_time_ns','dispatch_begin_ns','handler_start_ns')):
            if not fields['scheduled_time_ns']<=fields['enqueued_time_ns']<=fields['dispatch_begin_ns']<=fields['handler_start_ns']:
                issues.append('HOST_DISPATCH_ORDER_INVALID:'+rid)
        if (type(fields['dispatch_begin_ns']) is int and type(fields['dispatch_write_completed_ns']) is int
                and fields['dispatch_write_completed_ns']<fields['dispatch_begin_ns']):
            issues.append('PIPE_WRITE_COMPLETION_PRECEDES_DISPATCH:'+rid)
        semantic=response.get('semantic_forward_observations',{}).get('records',[])
        first=min((s['start_ns'] for s in semantic),default=None)
        rows.append(dict(request_id=rid,alias=row.get('model'),task=declared['task'],pool_id=declared['pool_id'],
            offsets_ns={k:None if v is None else v-origin for k,v in fields.items()},
            first_semantic_start_offset_ns=None if first is None else first-origin,
            handler_to_first_semantic_ns=None if first is None or fields['handler_start_ns'] is None else first-fields['handler_start_ns'],
            first_semantic_is_first_CUDA_release=False,host_queue_wait_removed=False))
    return issues,rows,anchors


def review(run,output):
    run=Path(run).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    issues=[];trace=None;rows=[]
    try:
        collect.audit(run);plan=collect.read(run/'plan.json');context=collect.read(run/'context.json')
        raw=collect.read(run/'raw.json');execution=collect.read(run/'execution.json')
        if (raw.get('status')!='completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors')
                or raw.get('final_source_audit')!='PASS' or execution.get('status')!='COLLECTED_TRAINING_REQUIRES_INDEPENDENT_REVIEW'
                or execution.get('final_source_audit')!='PASS' or execution.get('exception_cleanup') or execution.get('cleanup_error')):
            issues.append('RAW_OR_EXECUTION_NOT_QUALIFIED')
        for doc in (raw,execution):
            for key,name in (('plan','plan.json'),('context','context.json')):
                if doc.get(key)!=receipt(run/name):issues.append('BOUND_INPUT_RECEIPT_CHANGED:'+key)
        if execution.get('raw')!=receipt(run/'raw.json') or execution.get('execution_start')!=receipt(run/'execution-start.json'):
            issues.append('EXECUTION_PARENT_RECEIPT_CHANGED')
        if execution.get('command')!=collect.outer_command(run,context,plan['domain']):issues.append('DECLARED_COMMAND_CHANGED')
        if raw.get('execution_context_sha256')!=plan['execution_context_sha256']:issues.append('RAW_CONTEXT_CHANGED')
        ready=raw.get('ready',{});pids=[r.get('pid') for r in ready.values()]
        if set(ready)!=set(collect.RESIDENTS) or any(type(p) is not int for p in pids) or len(set(pids))!=4:
            issues.append('FOUR_DISTINCT_OWNED_READY_IDENTITIES_REQUIRED')
        for alias,family in collect.RESIDENTS.items():
            if ready.get(alias,{}).get('spec_sha256')!=receipt(context['worker_bindings'][family]['spec'])['sha256']:
                issues.append('READY_SPEC_IDENTITY_CHANGED:'+alias)
        expected=[(a,i['task'],i['pool_id']) for a,i in collect.existing.priming_items(plan)]
        priming=raw.get('priming_request_records',[])
        if ([(r.get('model'),r.get('task'),r.get('pool_id')) for r in priming]!=expected
                or any(r['status']!='completed' for r in priming)):
            issues.append('ORIGINAL_TWELVE_PRIMING_REQUIRED')
        if set(raw.get('shutdown',{}))!=set(collect.RESIDENTS) or any(r.get('forced') or r.get('exit_code')!=0 or r.get('bye_received') is not True for r in raw.get('shutdown',{}).values()):
            issues.append('UNCLEAN_SHUTDOWN')
        watchdog=execution.get('watchdog',{});budget=context['collection_budget']
        if (watchdog.get('status')!='EXITED' or watchdog.get('return_code')!=0
                or watchdog.get('cleanup',{}).get('live_survivors')!=[] or not watchdog.get('samples')
                or any(s.get('free_bytes',0)<budget['free_disk_reserve_bytes'] for s in watchdog.get('samples',[]))):
            issues.append('WATCHDOG_OR_DISK_FLOOR_FAILED')
        issues.extend(validate_startup_telemetry(raw))
        clocks=raw.get('clock_anchors',[])
        if ([r.get('label') for r in clocks]!=['before_startup','after_priming','after_drain','after_shutdown']
                or any(type(r.get(k)) is not int for r in clocks for k in ('perf_before_ns','unix_ns','perf_after_ns'))
                or any(r['perf_before_ns']>r['perf_after_ns'] for r in clocks)):
            issues.append('EXPLICIT_CAUSAL_HOST_CLOCK_ANCHORS_REQUIRED')
        found,rows,anchors=host_lifecycle_rows(plan,raw);issues.extend(found)
        save(out/'host-lifecycle.json',dict(schema='gsdeploy.whole57_joint_training_host_lifecycle_observations.v23',
            source_raw=receipt(run/'raw.json'),source_plan=receipt(run/'plan.json'),rows=rows,
            complete_offered_cohort_preserved=True,training_only=True,origin=raw.get('phase_start_ns'),
            clock_anchors=raw.get('clock_anchors',[]),causal_host_supply_identified=False,
            cross_domain_pairing_identified=False,scenario_probability=None))
        reports=list(run.glob('*.nsys-rep'))
        if plan['domain']==collect.LIGHT:
            if reports!=[run/'trace-whole57.nsys-rep']:raise ValueError('One complete profiler report required')
            sql=out/'trace-whole57.sqlite';export=export_sqlite(reports[0],sql)
            metadata=trace_metadata(sql);issues.extend(metadata['issues'])
            stderr=classify_stderr_loss((run/'stderr.log').read_text())
            if stderr['confirmed_profiler_loss_lines'] or stderr['unattributed_loss_lines']:issues.append('PROFILER_LOSS_OR_UNATTRIBUTED_LOSS_REQUIRES_REVIEW')
            parsed=parse_sqlite(sql,anchors)
            for alias,ready in raw['ready'].items():
                owner_rows=[r for r in raw['request_records'] if r['model']==alias]
                if owner_rows:
                    found,_=validate_parsed(owner_view(parsed,ready['pid']),[a for a in anchors if a['pid']==ready['pid']],
                        [r for r in raw['priming_request_records']+owner_rows if r['model']==alias],ready['pid'])
                    issues.extend(found)
            save(out/'parsed-measured.json',canonical_measured(parsed,receipt(run/'raw.json')))
            clocks=audit_one(sql,out/'parsed-measured.json');save(out/'session-clock.json',clocks)
            prefixes=light_prefixes(dict(trace=dict(sqlite=receipt(sql),parsed=receipt(out/'parsed-measured.json'))),raw)
            if set(prefixes)!={r['request_id'] for r in plan['trace']}:issues.append('FIRST_API_KERNEL_PREFIX_INCOMPLETE')
            save(out/'native-first-api-kernel.json',dict(schema='gsdeploy.whole57_native_first_api_kernel_observations.v23',
                rows=prefixes,source_raw=receipt(run/'raw.json'),source_sqlite=receipt(sql),
                source_host_lifecycle=receipt(out/'host-lifecycle.json'),source_clock_audit=receipt(out/'session-clock.json'),
                timestamp_domain='single_native_Nsight_session_ns',
                host_absolute_clock_point_alignment_claimed=False,calibrated_cross_clock_error_bound_ns=None,
                same_execution_only=True,training_only=True,causal_release_identified=False))
            save(out/'session-metadata.json',dict(**metadata,stderr_loss_classification=stderr))
            trace=dict(export=export,sqlite=receipt(sql),parsed=receipt(out/'parsed-measured.json'),
                session_clock=receipt(out/'session-clock.json'),session_metadata=receipt(out/'session-metadata.json'),
                native_first_api_kernel=receipt(out/'native-first-api-kernel.json'),
                per_owner_median_as_common_clock=False,calibrated_cross_clock_error_bound_ns=None)
        elif reports:issues.append('NATURAL_DOMAIN_HAS_UNDECLARED_PROFILER')
        collect.audit(run)
    except Exception as exc:issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+': '+str(exc))
    result=dict(schema='gsdeploy.whole57_joint_training_review.v23',status='PASS' if not issues else 'FAIL',
        issues=issues,host_lifecycle=receipt(out/'host-lifecycle.json') if (out/'host-lifecycle.json').exists() else None,
        trace=trace,requests=len(rows),training_only=True,model_fitted=False,
        validation_started=False,natural_first_CUDA_observed=False,calibrated_cross_clock_error_bound_ns=None,
        evidence={name:receipt(run/name) for name in ('plan.json','context.json','raw.json','execution-start.json','execution.json') if (run/name).exists()},
        reviewer=receipt(__file__),GPU_started=False)
    save(out/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=review(args.run,args.output);print(json.dumps(result));raise SystemExit(0 if result['status']=='PASS' else 1)
