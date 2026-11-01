"""Complete-cohort CPU audit for the new repeated-owner training schema.

This is not the old 26-job audit. Late sender write records are legal; failed
and unstarted rows remain in the denominator and never become fitting labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .stage_response_features_v1 import checked, receipt, stamp, ordered_events
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .twoFIFO_training_manifest_v1 import MEASUREMENT_NS, DRAIN_NS, OWNERS, digest

SCHEMA = 'gsdeploy.original_twoFIFO_owned_training_review.v3'
RAW_SCHEMA = 'gsdeploy.original_twoFIFO_owned_training_capture.v3'
PLAN_SCHEMA = 'gsdeploy.original_twoFIFO_owned_training_campaign.v3.attempt'


def require(value, message):
    if not value:
        raise ValueError(message)


def attempt_events(raw):
    # Original priming rows do not carry profile_job_id. Annotate a temporary
    # view for this helper; preserve the captured row and raw event ordinals.
    priming=[dict(r,profile_job_id='__priming__') for r in raw['priming_request_records']]
    return ordered_events(priming+raw['request_records'])


def inspect_request_clock(row):
    keys = ('scheduled_time_ns', 'enqueued_time_ns', 'actual_submit_time_ns',
            'submit_write_completed_time_ns', 'observed_worker_start_time_ns',
            'completion_time_ns', 'response_reader_event_enqueue_time_ns',
            'response_collector_dequeued_time_ns', 'response_collector_processing_time_ns')
    require(all(stamp(row.get(k)) for k in keys), 'Complete integer request clocks required')
    response = row['worker_response']
    begin, end = response.get('start_perf_ns'), response.get('end_perf_ns')
    scheduled, enqueue, submit, write, start, complete, reader, dequeue, process = (row[k] for k in keys)
    require(stamp(begin) and stamp(end) and start == begin,
            'START must bind the actual handler')
    require(scheduled <= enqueue <= submit <= begin <= end <= reader <= dequeue <= process
            and complete == reader and submit <= write,
            'Separate sender and RESULT chronology required')
    ready=row.get('previous_handler_next_request_ready')
    if ready is not None:
        for event_key,result in (('received_time_ns',reader),('collector_dequeued_time_ns',dequeue),
                                 ('collector_processing_time_ns',process)):
            require(stamp(ready.get(event_key)) and result<=ready[event_key],
                    'Same stdout RESULT must precede READY in each parent clock')
            require(all(stamp(e.get(event_key)) and e[event_key]<=result for e in row.get('execution_events',[])),
                    'Same stdout START/PHASE must precede RESULT in each parent clock')
    # write is a post-send lock acquisition, not pipe line availability. There
    # is deliberately no ordering between write and START, END or RESULT.
    return dict(request_id=row['request_id'], late_write_after_RESULT=write > reader)


def inspect_cohort(plan, raw):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'training'
            and raw.get('schema') == RAW_SCHEMA and raw.get('split') == 'training',
            'Exact new training plan/raw role required')
    for name in ('execution_id', 'capture_id', 'attempt_id'):
        require(isinstance(plan.get(name), str) and plan[name] and raw.get(name) == plan[name],
                'Plan/raw identity mismatch: ' + name)
    cases = plan['cases']; rows = raw['request_records']; observed_cases = raw['cases']
    require(0 < len(cases) <= 16 and len(observed_cases) == len(cases),
            'Exact bounded complete case list required')
    require(len({c['case_id'] for c in cases}) == len(cases), 'Unique planned case IDs required')
    expected = [(c['case_id'], a) for c in cases for a in c['arrivals']]
    require(len({a['request_id'] for _, a in expected}) == len(expected), 'Unique planned request IDs required')
    require([r.get('request_id') for r in rows] == [a['request_id'] for _, a in expected],
            'Every offered row must be retained in original plan order')
    by_id = {r['request_id']: r for r in rows}
    issues = []; late_writes = 0; prior_end = None; saw_unstarted = False
    for case, observed in zip(cases, observed_cases):
        cid = case['case_id']; arrivals = case['arrivals']
        require(observed.get('case_id') == cid and case['public_arrivals_sha256'] == digest(arrivals)
                and observed.get('public_arrivals_sha256') == case['public_arrivals_sha256'],
                'Exact ordered case and public arrival digest required')
        require(observed.get('request_ids') == [a['request_id'] for a in arrivals],
                'Case must retain all offered identities')
        epoch = observed.get('epoch_ns', observed.get('observation_start_ns'))
        if epoch is None:
            saw_unstarted = True
            for a in arrivals:
                r = by_id[a['request_id']]
                require(r.get('status') == 'not_started' and r.get('scheduled_time_ns') is None,
                        'Unstarted offered rows must be explicit, not fabricated arrivals')
            issues.append('UNSTARTED_CASE:' + cid)
        else:
            require(not saw_unstarted and stamp(epoch) and (prior_end is None or prior_end <= epoch),
                    'Started cases must form an ordered prefix')
            end, deadline = epoch + MEASUREMENT_NS, epoch + MEASUREMENT_NS + DRAIN_NS
            require(observed.get('measurement_end_ns') == end
                    and observed.get('fixed_drain_deadline_ns') == deadline,
                    'Original fixed measurement/drain required')
            final = observed.get('observation_finalized_ns')
            if not stamp(final) or final < deadline:
                issues.append('FIXED_DRAIN_NOT_OBSERVED:' + cid)
            prior_end = final if stamp(final) else deadline
        for a in arrivals:
            r = by_id[a['request_id']]
            require(r.get('profile_job_id') == cid and r.get('split') == 'training'
                    and r.get('model') == a['owner'] and r.get('task') == a['task']
                    and r.get('pool_id') == a['pool_id'] and r.get('arrival_index') == a['arrival_index']
                    and r.get('arrival_ns') == a['arrival_ns'], 'Exact public owner/task/input/arrival required')
            if epoch is not None:
                require(r.get('scheduled_time_ns') == epoch + a['arrival_ns'], 'Exact scheduled arrival required')
                if r.get('status') == 'completed':
                    clock = inspect_request_clock(r); late_writes += clock['late_write_after_RESULT']
                    if (r.get('worker_response', {}).get('success') is not True
                            or r['completion_time_ns'] > deadline):
                        issues.append('FAILED_OR_AFTER_CUTOFF_RESULT:' + r['request_id'])
                else:
                    issues.append('INCOMPLETE_OFFERED_REQUEST:' + r['request_id'])
    priming = raw['priming_request_records']; public_priming = plan['priming']
    require(len(public_priming) == 18 and len(priming)<=18, 'Exactly eighteen planned priming requests required per attempt')
    require([(r['model'], r['task'], r['pool_id']) for r in priming]
            == [(a['owner'], a['task'], a['pool_id']) for a in public_priming[:len(priming)]], 'Exact ordered priming prefix required')
    if len(priming)!=18:issues.append('PRIMING_NOT_COMPLETE')
    all_rows = priming + rows
    require(len({r.get('request_id') for r in all_rows}) == len(all_rows), 'Priming/measured IDs must be disjoint')
    for r in priming:
        if r.get('status') != 'completed' or r.get('worker_response', {}).get('success') is not True:
            issues.append('INCOMPLETE_PRIMING:' + str(r.get('request_id')))
        else:
            inspect_request_clock(r)
    # Source order constrains two independent serial branches. Sender can have
    # several submitted requests while its worker is still in an earlier one.
    owners = defaultdict(list)
    public_order=priming+[by_id[a['request_id']] for c in cases
        for a in sorted(c['arrivals'],key=lambda a:(a['arrival_ns'],a['arrival_index']))]
    for r in public_order:
        if r.get('actual_submit_time_ns') is not None:
            owners[r['model']].append(r)
    for owner, members in owners.items():
        require(owner in OWNERS, 'Unknown resident owner')
        for previous, current in zip(members, members[1:]):
            require(previous['enqueued_time_ns']<=current['enqueued_time_ns'],
                    'Dispatcher must preserve public per-owner arrival and tie order')
            if stamp(previous.get('submit_write_completed_time_ns')):
                require(previous['submit_write_completed_time_ns'] <= current['actual_submit_time_ns'],
                        'Original sender FIFO release order required')
            if stamp(current.get('observed_worker_start_time_ns')):
                ready = previous.get('previous_handler_next_request_ready', {})
                require(stamp(ready.get('perf_ns')) and ready['perf_ns'] <= current['observed_worker_start_time_ns'],
                        'Original worker FIFO must release at prior READY')
                starts=[e for e in current.get('execution_events',[]) if e.get('event') is None]
                require(len(starts)==1 and starts[0].get('perf_ns')==current['observed_worker_start_time_ns'],
                        'Exact next handler START notification required')
                for key in ('received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns'):
                    require(stamp(ready.get(key)) and stamp(starts[0].get(key)) and ready[key]<=starts[0][key],
                            'Same producer READY must precede next START on every parent clock')
    return dict(issues=issues, offered_requests=len(rows), planned_cases=len(cases),
                completed_requests=sum(r.get('status') == 'completed' for r in rows),
                unstarted_requests=sum(r.get('status') == 'not_started' for r in rows),
                priming_requests=len(priming), late_write_after_RESULT_requests=late_writes,
                complete_offered_denominator_preserved=True, global_admission_cap_added=False)


def inspect_samples(raw, collector_pid):
    """New case-window telemetry, including repeated drain and priming_begin."""
    from .device_progress_actual_review_v3 import valid_live_ownership
    from .profile_owned_collect_v3 import MONITORING_SCOPE
    boundary=raw['lifecycle_boundaries'];names=[b['boundary'] for b in boundary];times=[b['time_ns'] for b in boundary]
    expected=['startup_begin','startup_ready','priming_begin','priming_end','measurement_and_drain_end',
              'monitoring_end','shutdown_begin','shutdown_end']
    require(names==expected and all(stamp(t) for t in times) and times==sorted(times),
            'Exact complete new lifecycle boundaries required')
    b=dict(zip(names,times));samples=raw['board_samples'];ready=raw['ready'];pids={r['pid'] for r in ready.values()}
    require(raw.get('monitoring_scope')==MONITORING_SCOPE and raw.get('shutdown_memory_sampled') is False
            and set(ready)==set(OWNERS) and len(pids)==4, 'Exact monitoring scope and four resident PIDs required')
    require(samples and all(stamp(s.get('time_ns')) for s in samples), 'Timestamped monitor samples required')
    stamps=[s['time_ns'] for s in samples]
    require(stamps==sorted(stamps) and stamps[0]<=b['startup_begin']<=b['startup_ready']
            <=b['priming_end']<=raw['cases'][0]['observation_start_ns']
            <raw['observation_end_ns']==b['measurement_and_drain_end']<=stamps[-1]
            <=b['monitoring_end']<b['shutdown_begin']<=b['shutdown_end'],
            'Startup through final live-worker sample before shutdown required')
    require(samples[0]['phase']=='startup' and samples[-1]['phase']=='final_drain'
            and sum(s['phase']=='final_drain' for s in samples)==1, 'Exact final live sample required')
    require(set(raw['ready_ownership_evidence'])==set(OWNERS), 'Every READY requires ownership evidence')
    pins={}
    for owner in OWNERS:
        evidence=raw['ready_ownership_evidence'][owner]
        require(len(evidence)==1 and evidence[0]['alias']==owner and evidence[0]['pid']==ready[owner]['pid']
                and valid_live_ownership(evidence[0],collector_pid,collector_pid), 'Pinned live READY ancestry required')
        pins[owner]=tuple(evidence[0][k] for k in ('launcher_pid','launcher_start_ticks','collector_start_ticks'))
    for s in samples:
        require(s['phase'] in ('startup','priming','measurement','drain','final_drain'), 'Known monitor scope required')
        owned={int(k):v for k,v in s['owned_processes_mib'].items()};processes={int(k):v for k,v in s['processes_mib'].items()}
        evidence=s['ownership_evidence']
        require(owned==processes and stamp(s['query_duration_ns']) and type(s['watts']) in (int,float)
            and math.isfinite(s['watts']) and s['watts']>0 and not s['foreign_pids']
            and not s['unresolved_nvml_pids'] and not s['unresolved_ownership_reasons'], 'Entirely owned finite telemetry required')
        require(len(evidence)==len(owned) and {e['pid'] for e in evidence}==set(owned)
            and all(valid_live_ownership(e,collector_pid,collector_pid) for e in evidence), 'Stable per-poll ancestry required')
        require(len(s['owned_aliases'])==len(set(s['owned_aliases']))
            and set(s['owned_aliases'])=={e['alias'] for e in evidence}, 'Exact sample aliases required')
        require(all(e['alias'] in pins and tuple(e[k] for k in ('launcher_pid','launcher_start_ticks','collector_start_ticks'))==pins[e['alias']]
            for e in evidence), 'Stable original launcher identities required')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in owned.values())
            and type(s['owned_used_mib']) in (int,float) and math.isfinite(s['owned_used_mib'])
            and abs(sum(owned.values())-s['owned_used_mib'])<1e-6, 'Owned sampled peak arithmetic required')
        if s['phase'] in ('measurement','drain','final_drain'):
            require(pids<=set(owned) and set(s['owned_aliases'])==set(OWNERS)
                and all(any(e['alias']==a and e['pid']==r['pid'] for e in evidence) for a,r in ready.items()),
                'All four resident memories required throughout measurement and drain')
    windows=[dict(case_id=c['case_id'],start_ns=c['observation_start_ns'],end_ns=c['measurement_end_ns'],
                  fixed_drain_deadline_ns=c['fixed_drain_deadline_ns']) for c in raw['cases']]
    require(raw.get('measurement_windows')==windows, 'Separate explicit case measurement energy windows required')
    return dict(total_samples=len(samples),owned_sampled_peak_mib=max(s['owned_used_mib'] for s in samples),
                boundaries=b,measurement_windows=windows,continuous_peak_certified=False,
                drain_energy_included=False,startup_occurrences_in_attempt=1)


def inspect_histories(plan, raw):
    started=[c for c in raw['cases'] if c.get('observation_start_ns') is not None]
    histories=raw.get('case_histories',[])
    require([h['case_id'] for h in histories]==[c['case_id'] for c in started],
            'Every started case must retain its original prior history')
    expected=[r['request_id'] for r in raw['priming_request_records']]
    final={r['request_id']:r for r in raw['priming_request_records']+raw['request_records']}
    for case, history in zip(started,histories):
        snapshot=history['prior_history_snapshot'];rows=snapshot['request_records']
        require([r['request_id'] for r in rows]==expected and set(snapshot['pending_request_ids'])==set(expected),
                'Full priming and all prior offered rows required at each case boundary')
        require(stamp(snapshot['snapshot_time_ns']) and snapshot['snapshot_time_ns']<=case['observation_start_ns']
                and snapshot.get('empty_initial_state_assumed') is False
                and snapshot.get('ready')==raw['ready'] and not snapshot['runtime_errors'],
                'Actual pre-case state and stable READY roster required')
        for row in rows:
            end=final[row['request_id']]
            for key in ('model','task','pool_id','scheduled_time_ns','enqueued_time_ns',
                        'actual_submit_time_ns','submit_write_completed_time_ns','observed_worker_start_time_ns',
                        'worker_response','previous_handler_next_request_ready'):
                if row.get(key) is not None:
                    require(row[key]==end.get(key), 'Snapshot known history cannot be rewritten: '+key)
            events=row.get('execution_events',[])
            require(events==end.get('execution_events',[])[:len(events)],'Snapshot must retain actual processed event prefix')
            require(all(e['collector_processing_time_ns']<=snapshot['snapshot_time_ns'] for e in events),
                    'Pre-case history cannot contain future processed events')
        expected.extend(case['request_ids'])
    return dict(started_cases=len(started),full_prior_histories=len(histories),empty_initial_state_assumed=False)


def inspect_checkpoint_activity(plan,raw,ledger):
    """Authenticate the complete copy/write ledger against every fixed case.

    This is a timing-path audit, not proof of zero observation overhead. Tiny
    ledger writes occur outside the protected path by the bound collector code.
    """
    from .twoFIFO_owned_collect_v3 import ACTIVITY_FILE,ACTIVITY_SCHEMA,TIME_DOMAIN
    require(raw.get('no_mutating_checkpoint_during_case') is True
        and raw.get('checkpoint_spans_file')==ACTIVITY_FILE
        and raw.get('checkpoint_spans_schema')==ACTIVITY_SCHEMA
        and raw.get('checkpoint_spans_execution_id')==plan['execution_id']
        and raw.get('checkpoint_time_domain')==TIME_DOMAIN,
        'Explicit v3 protected timing path required')
    require(ledger.get('schema')==ACTIVITY_SCHEMA and ledger.get('execution_id')==plan['execution_id']
        and ledger.get('time_domain')==TIME_DOMAIN and ledger.get('all_spans_complete') is True,
        'Canonical complete source-bound checkpoint ledger required')
    spans=ledger['checkpoint_spans'];windows=raw['measurement_windows']
    require(spans and [s['span_id'] for s in spans]==list(range(len(spans))),
        'Complete ordered checkpoint span identities required')
    expected_fields={'span_id','kind','case_id','phase','stage','time_domain','start_ns','end_ns'}
    for span in spans:
        require(set(span)==expected_fields and span['time_domain']==TIME_DOMAIN
            and stamp(span['start_ns']) and stamp(span['end_ns'])
            and span['start_ns']<=span['end_ns'],'Every recorded operation must have complete exact clocks')
        require(all(span['end_ns']<w['start_ns'] or span['start_ns']>w['fixed_drain_deadline_ns']
                    for w in windows),'No history copy or checkpoint may intersect measurement or fixed drain')
    require([s['start_ns'] for s in spans]==sorted(s['start_ns'] for s in spans),
        'Ledger must retain actual operation issue order')
    wrappers=[s for s in spans if s['kind']=='checkpoint']
    expected=[('before_startup',None),('postpriming',None)]
    expected.extend((stage,w['case_id']) for w in windows for stage in ('precase','postdrain'))
    expected.append(('shutdown',None))
    require([(s['stage'],s['case_id']) for s in wrappers]==expected,
        'Every original startup, priming, case and final checkpoint required exactly once')
    for parent in wrappers:
        children=spans[parent['span_id']+1:parent['span_id']+3]
        require([s['kind'] for s in children]==['checkpoint_locked_snapshot','checkpoint_serialization_and_atomic_save']
            and all(s['stage']==parent['stage'] and s['case_id']==parent['case_id']
                and parent['start_ns']<=s['start_ns']<=s['end_ns']<=parent['end_ns'] for s in children)
            and children[0]['end_ns']<=children[1]['start_ns'],
            'Each checkpoint must retain its full lock and serialization spans')
    histories=[s for s in spans if s['kind']=='precase_history_snapshot']
    require([s['case_id'] for s in histories]==[w['case_id'] for w in windows]
        and len(spans)==3*len(wrappers)+len(histories),
        'No full-history operation may be omitted or substituted')
    require(raw.get('public_case_calendars')==plan['cases'] and raw.get('priming_public_calendar')==plan['priming'],
        'Complete offered and priming calendars must persist before execution')
    preparations=raw.get('case_preparations',[])
    require(len(preparations)==len(windows),'Every case must have a durable preparation')
    for i,(window,history,prepared,case,snapshot) in enumerate(zip(windows,histories,preparations,plan['cases'],raw['case_histories'])):
        pre=next(s for s in wrappers if s['stage']=='precase' and s['case_id']==window['case_id'])
        post=next(s for s in wrappers if s['stage']=='postdrain' and s['case_id']==window['case_id'])
        require(history['start_ns']<=snapshot['prior_history_snapshot']['snapshot_time_ns']<=history['end_ns']
            <=pre['start_ns']<=pre['end_ns']<window['start_ns']
            and post['start_ns']>window['fixed_drain_deadline_ns'],
            'History and durable preparation must precede epoch; persistence follows fixed drain')
        require(set(prepared)=={'case_id','public_arrivals_sha256','public_arrivals',
                'prior_history_index','prepared_before_epoch','durable_checkpoint_return_ns'}
            and {k:v for k,v in prepared.items() if k!='durable_checkpoint_return_ns'}==dict(
                case_id=case['case_id'],public_arrivals_sha256=case['public_arrivals_sha256'],
                public_arrivals=case['arrivals'],prior_history_index=i,prepared_before_epoch=True)
            and stamp(prepared['durable_checkpoint_return_ns'])
            and pre['end_ns']<=prepared['durable_checkpoint_return_ns']<window['start_ns'],
            'Exact complete public arrival preparation required')
    return dict(policy='NO_FULL_COPY_OR_PERSISTENCE_FROM_CASE_EPOCH_THROUGH_FIXED_DRAIN',
        complete_spans=len(spans),checkpoints=len(wrappers),precase_histories=len(histories),
        critical_interval_overlaps=0,zero_observation_overhead_proven=False,
        canonical_independent_sidecar_required=True)


def review(raw_path, output, execution_path=None):
    raw_ref = receipt(raw_path); raw = checked(raw_ref)
    # The role gate happens before reading response labels.
    require(raw.get('schema') == RAW_SCHEMA and raw.get('split') == 'training', 'Training-only raw required')
    plan = checked(raw['plan']); context = checked(raw['context'])
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'training', 'Training-only plan required')
    from .twoFIFO_owned_collect_v3 import audit_plan, SCHEMA as CAMPAIGN_SCHEMA
    require(audit_plan(raw['plan']['path']) == plan and raw['context']==plan['context']
            and raw.get('execution_context_sha256')==plan['execution_context_sha256']
            and raw.get('prepared_pool')==plan['prepared_pool']
            and raw.get('source_bindings')==plan['sources'] and raw.get('source_freeze')==plan['source_freeze'],
            'Semantic plan, context, input and source bindings required')
    executed=checked(raw['executed_cases'])
    require(executed==dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],cases=plan['cases']),
            'Exact all-offered executed case binding required')
    cohort = inspect_cohort(plan, raw); issues = list(cohort['issues']); sources = [raw_ref, raw['plan'], raw['context']]
    histories=inspect_histories(plan,raw)
    from .twoFIFO_owned_collect_v3 import ACTIVITY_FILE
    activity_ref=receipt(Path(raw_path).resolve().parent/ACTIVITY_FILE)
    activity=inspect_checkpoint_activity(plan,raw,checked(activity_ref));sources.append(activity_ref)
    if raw.get('status') != 'completed' or any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors')):
        issues.append('CLEAN_COMPLETE_ATTEMPT_REQUIRED')
    native = raw.get('backend') == 'native' and raw.get('kind') == 'measured'
    if not native:
        issues.append('CPU_TEST_ONLY_NOT_NATIVE_TRAINING')
    # Pure request and original-output checks do not import or launch CUDA.
    protocol = None; requests = []; telemetry = None
    if not issues:
        from .device_progress_smoke_raw_verify_v3 import validate_request, reparse_outputs
        protocol = reparse_outputs(raw); issues.extend(protocol['issues']); sources.extend(protocol['sources'])
        for row in raw['priming_request_records'] + raw['request_records']:
            found, detail = validate_request(row, raw['ready'].get(row['model'], {}), protocol['policies'].get(row['model']))
            issues.extend(found); requests.append(detail)
        shutdown=raw.get('shutdown',{})
        require(set(shutdown)==set(OWNERS) and all(v.get('exit_code')==0 and v.get('forced') is False
                and v.get('bye_received') is True for v in shutdown.values()), 'Clean exit of all four resident workers required')
        # Check availability of all history events as one continuous attempt.
        attempt_events(raw)
        if execution_path is None:
            issues.append('OWNED_SUPERVISOR_EXECUTION_REQUIRED')
        else:
            execution_ref = receipt(execution_path); execution = checked(execution_ref); sources.append(execution_ref)
            require(execution.get('profile_execution')==dict(schema=CAMPAIGN_SCHEMA+'.execution_binding',
                plan=raw['plan'],execution_id=plan['execution_id'],attempt_id=plan['attempt_id'])
                and execution.get('budget')==plan['supervisor_budget'],
                'Supervisor must bind this exact attempt and original budget')
            require(execution.get('status') == 'CHILD_EXITED_REQUIRES_PROFILE_REVIEW'
                    and execution.get('child_reaped') is True and execution.get('live_group_pids') == []
                    and execution.get('experiment_lock_held_through_group_exit') is True
                    and execution.get('all_owned_descendants_exited_and_reaped') is True
                    and execution.get('final_source_audit') == 'PASS'
                    and execution.get('watchdog', {}).get('status') == 'EXITED'
                    and execution['watchdog'].get('return_code') == 0,
                    'Completed owned supervisor with original source/cleanup checks required')
            telemetry = inspect_samples(raw, execution['process_group'])
    sources = strict_references([sources,plan['sources'],raw['executed_cases'],implementation_sources(__file__)])
    require(all(receipt(r['path']) == r for r in sources), 'Evidence changed during review')
    eligible = not issues
    result = dict(schema=SCHEMA, status='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION' if eligible else 'NOT_ELIGIBLE',
        raw=raw_ref, inputs={'raw.json':raw_ref,'plan.json':raw['plan'],'context.json':raw['context']},
        issues=issues, cohort=cohort, output_protocol_review=protocol, requests=requests, telemetry=telemetry,
        histories=histories,checkpoint_activity=activity,checkpoint_activity_source=activity_ref,
        sources=sources, training_feature_extraction_eligible=eligible,
        response_structures_inspected=len(requests),response_structures_not_inspected=cohort['offered_requests']+cohort['priming_requests']-len(requests),
        execution=None if execution_path is None else receipt(execution_path),
        fitted_model_qualified=False, independent_prediction_accuracy_validated=False,
        main_queue_provider_eligible=False, final_gain_eligible=False, GPU_started=False, validation_labels_read=False)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return receipt(output/'report.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execution');a=p.parse_args()
    print(json.dumps(review(a.raw,a.output,a.execution)))
