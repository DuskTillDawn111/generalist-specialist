"""Explicit matching native review adapter; CPU-only, no fit entrypoint.

Numerical cohort/history/telemetry/checkpoint/protocol checks retain frozen v3
semantics. New schema, matching audit and execution manifest binding are explicit;
old module globals and artifacts are never mutated.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from .stage_response_features_v1 import checked, receipt, stamp
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .twoFIFO_training_manifest_v1 import MEASUREMENT_NS, DRAIN_NS, OWNERS, digest
from .twoFIFO_owned_training_review_v3 import (require, inspect_request_clock,
    inspect_histories, inspect_checkpoint_activity, inspect_samples, attempt_events)
from . import twoFIFO_matching_execute_v1 as executor
SCHEMA = 'gsdeploy.twoFIFO_matching_native_review.v1'
RAW_SCHEMA = executor.RAW_SCHEMA
PLAN_SCHEMA = executor.SCHEMA + '.attempt'


def resource_facts(raw):
    """Observed training windows only; no 90+120s resource transfer claim."""
    from .twoFIFO_observed_resources_v2 import observed
    for window in raw['measurement_windows']:
        require(window['end_ns']-window['start_ns'] == MEASUREMENT_NS
                and window['fixed_drain_deadline_ns']-window['end_ns'] == DRAIN_NS,
                'Exact ten-second measurement and sixty-second drain required')
    facts=observed(raw)
    values=[facts['startup']['energy_joules'],
        facts['all_startup_through_final_drain_owned_sampled_peak_mib']]
    for case in facts['cases']:
        values.extend([case['measurement_energy_joules'],
            case['startup_through_this_drain_owned_sampled_peak_mib'],
            case['local_measurement_through_drain_owned_sampled_peak_mib']])
    require(all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in values),
            'Observable startup/every measurement energy and owned sampled memory required')
    return facts


def inspect_cohort(plan, raw):
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'training'
            and raw.get('schema') == RAW_SCHEMA and raw.get('split') == 'training',
            'Exact new training plan/raw role required')
    for name in ('execution_id', 'capture_id', 'attempt_id'):
        require(isinstance(plan.get(name), str) and plan[name] and raw.get(name) == plan[name],
                'Plan/raw identity mismatch: ' + name)
    cases = plan['cases']; rows = raw['request_records']; observed_cases = raw['cases']
    require(0 < len(cases) <= 13 and len(observed_cases) == len(cases),
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
    require(len(public_priming) == 18 and len(priming)==18, 'Exactly eighteen planned priming requests required per attempt')
    require([(r['model'], r['task'], r['pool_id']) for r in priming]
            == [(a['owner'], a['task'], a['pool_id']) for a in public_priming[:len(priming)]], 'Exact ordered priming prefix required')
    require([r.get('request_id') for r in priming] == [r['request_id'] for r in public_priming],
            'Exact predeclared priming identities required')
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



def review(raw_path, output, execution_path=None, *, manifest_path):
    raw_ref = receipt(raw_path); raw = checked(raw_ref)
    # The role gate happens before reading response labels.
    require(raw.get('schema') == RAW_SCHEMA and raw.get('split') == 'training', 'Training-only raw required')
    plan = checked(raw['plan']); context = checked(raw['context'])
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('split') == 'training', 'Training-only plan required')
    manifest_ref = receipt(manifest_path); manifest = executor.audit(manifest_path)
    audit_plan = executor.audit_plan
    expected_executor_sources = implementation_sources(executor.__file__)
    require(raw.get('collector_schema') == RAW_SCHEMA
        and raw.get('matching_executor_sources') == expected_executor_sources
        and all(r in manifest['sources'] for r in expected_executor_sources),
        'Explicit complete matching executor closure must bind raw and execution manifest')
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
    protocol = None; requests = []; telemetry = None; resources = None; priming_semantics = None
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
            item = next((i for i in manifest['attempts'] if i['plan'] == raw['plan']), None)
            require(item is not None and Path(raw_path).resolve() == Path(item['collection'])/'raw.json'
                and Path(execution_path).resolve() == Path(item['collection'])/'execution.json',
                'Exact canonical matching raw/execution evidence required')
            executor.completed_execution(execution, manifest_ref, item)
            sources.append(receipt(item['claim']))
            telemetry = inspect_samples(raw, execution['process_group'])
            resources = resource_facts(raw)
            from .natural_macro_resources_candidate_v1 import priming_identity
            planned=executor.planned_priming_semantics(plan,context)
            actual_identity=priming_identity(raw,context)
            require(actual_identity==planned['sha256'],
                    'Reviewed actual priming semantics differ from exact planned semantics')
            priming_semantics=dict(planned=planned,actual_native_identity=actual_identity,
                provenance_request_ids=[r['request_id'] for r in raw['priming_request_records']],
                actual_native_history_qualified=True)
    sources.append(manifest_ref)
    require(receipt(manifest_path) == manifest_ref and executor.audit(manifest_path) == manifest, 'Manifest changed during review')
    sources = strict_references([sources,plan['sources'],raw['executed_cases'],implementation_sources(__file__)])
    require(all(receipt(r['path']) == r for r in sources), 'Evidence changed during review')
    eligible = not issues
    result = dict(schema=SCHEMA, matching_manifest=manifest_ref, status='PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION' if eligible else 'NOT_ELIGIBLE',
        raw=raw_ref, inputs={'raw.json':raw_ref,'plan.json':raw['plan'],'context.json':raw['context']},
        issues=issues, cohort=cohort, output_protocol_review=protocol, requests=requests,
        telemetry=telemetry, observed_resources=resources,priming_semantics=priming_semantics,
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    p.add_argument('--execution', required=True); p.add_argument('--manifest', required=True)
    a = p.parse_args()
    print(json.dumps(review(a.raw, a.output, a.execution, manifest_path=a.manifest)))
