"""New-schema whole-attempt observed sender/worker FIFO diagnostics.

Busy submit-to-START includes earlier worker occupancy. It is never a worker
pre-START demand estimate, an idle donor, or a learned transport distribution.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path

from . import twoFIFO_composable_macros_v2 as macros_v2
from . import worker_stdin_fifo_contract_v1 as fifo
from . import worker_stdin_fifo_partial_identification_v1 as partial
from .twoFIFO_owned_training_review_v2 import inspect_cohort, require
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_busy_transport_observed.v1'


def counts(rows):
    return dict(offered=len(rows),
        predecessor_known=sum(r['previous_request_id'] is not None for r in rows),
        predecessor_unknown=sum(r['previous_request_id'] is None for r in rows),
        busy_at_submit=sum(r['previous_handler_busy_at_submit'] is True for r in rows),
        not_busy_at_submit=sum(r['previous_handler_busy_at_submit'] is False for r in rows),
        sender_predecessor_write_record_pending_at_enqueue=sum(
            r['sender_predecessor_write_record_pending_at_enqueue'] is True for r in rows),
        positive_enqueue_to_submit=sum(r['enqueue_to_submit_ns'] > 0 for r in rows),
        START_before_write_record=sum(r['worker_START_precedes_parent_write_record'] for r in rows),
        write_record_after_RESULT=sum(r['write_record_after_RESULT_reader'] for r in rows),
        START_before_previous_READY_processing=sum(
            r['START_precedes_previous_READY_processing'] is True for r in rows))


def summarize(raw, plan, *, capture_sha256):
    """Pure CPU helper; validates cohort clocks, does not authenticate native data."""
    cohort = inspect_cohort(plan, raw)
    require(not cohort['issues'], 'Complete offered cohort required; no selected-success transport donors')
    by_id = {r['request_id']: r for r in raw['request_records']}
    priming = raw['priming_request_records']
    public = priming + [by_id[a['request_id']] for case in plan['cases']
        for a in sorted(case['arrivals'], key=lambda a: (a['arrival_ns'], a['arrival_index']))]
    prime_ids = {r['request_id'] for r in priming}
    prior = {}; ordinals = defaultdict(int); records = []; previous_cases = []
    case_history = []
    for case in plan['cases']:
        preceding = priming + [r for r in raw['request_records'] if r['profile_job_id'] in previous_cases]
        history = next((h for h in raw.get('case_histories', []) if h['case_id'] == case['case_id']), None)
        snapshot = None if history is None else history['prior_history_snapshot']
        case_history.append(dict(case_id=case['case_id'],
            prior_public_request_ids=[r['request_id'] for r in preceding],
            prior_case_ids=list(previous_cases),
            captured_snapshot_present=snapshot is not None,
            captured_snapshot_time_ns=None if snapshot is None else snapshot['snapshot_time_ns'],
            captured_snapshot_request_ids=None if snapshot is None else [r['request_id'] for r in snapshot['request_records']],
            empty_initial_state_assumed=False))
        previous_cases.append(case['case_id'])
    for row in public:
        owner = row['model']; previous = prior.get(owner)
        ordinal = ordinals[owner]; ordinals[owner] += 1
        enqueue = row['enqueued_time_ns']; submit = row['actual_submit_time_ns']
        write = row['submit_write_completed_time_ns']; start = row['observed_worker_start_time_ns']
        ready = None if previous is None else previous['previous_handler_next_request_ready']
        ready_at = None if ready is None else ready['perf_ns']
        busy = None if ready is None else submit < ready_at
        pending = None if previous is None else enqueue < previous['submit_write_completed_time_ns']
        if row['request_id'] not in prime_ids:
            bounds = None; endpoint_pairs = None
            # The old mathematical API accepts a prior READY; we explicitly
            # restrict this use to READY<=submit. This is an idle-HANDLER
            # condition, not proof that the worker is blocked in stdin read.
            if busy is False:
                bounds = fifo.idle_donor_decomposition_bounds(submit, write, start, ready_at)
                u = min(write-submit, start-submit)
                endpoint_pairs = [partial.paired_idle_components(start-submit, write-submit, x)
                                  for x in dict.fromkeys((0, u))]
            records.append(dict(capture_sha256=capture_sha256,
                request_id=row['request_id'], owner=owner, case_id=row['profile_job_id'],
                public_owner_ordinal_including_priming=ordinal,
                previous_request_id=None if previous is None else previous['request_id'],
                previous_case_id=None if previous is None else previous.get('profile_job_id', '__priming__'),
                predecessor_state='UNKNOWN_INITIAL_OWNER_HISTORY' if ready is None else
                    ('PREVIOUS_HANDLER_BUSY_AT_SUBMIT' if busy else 'PREVIOUS_HANDLER_READY_BY_SUBMIT'),
                scheduled_ns=row['scheduled_time_ns'], enqueue_ns=enqueue, submit_ns=submit,
                write_record_ns=write, worker_START_ns=start,
                previous_READY_occurrence_ns=ready_at,
                previous_READY_processing_ns=None if ready is None else ready['collector_processing_time_ns'],
                previous_handler_busy_at_enqueue=None if ready is None else enqueue < ready_at,
                previous_handler_busy_at_submit=busy,
                sender_predecessor_write_record_pending_at_enqueue=pending,
                scheduled_to_enqueue_ns=enqueue-row['scheduled_time_ns'],
                enqueue_to_submit_ns=submit-enqueue, enqueue_to_write_record_ns=write-enqueue,
                submit_to_write_record_ns=write-submit, observed_submit_to_START_ns=start-submit,
                previous_READY_to_START_ns=None if ready is None else start-ready_at,
                prior_READY_to_submit_signed_ns=None if ready is None else submit-ready_at,
                sender_turn_after_observed_release_ns=None if previous is None else
                    submit-max(enqueue, previous['submit_write_completed_time_ns']),
                worker_START_precedes_parent_write_record=start < write,
                write_record_after_RESULT_reader=write > row['completion_time_ns'],
                START_precedes_previous_READY_processing=None if ready is None else start < ready['collector_processing_time_ns'],
                same_producer_READY_precedes_START_parent_axes=None if ready is None else
                    all(ready[k] <= next(e for e in row['execution_events'] if e.get('event') is None)[k]
                        for k in ('received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns')),
                idle_handler_component_bounds=bounds, idle_handler_joint_endpoint_pairs=endpoint_pairs,
                idle_bounds_condition=None if bounds is None else 'PREVIOUS_READY_OCCURRENCE_LE_SUBMIT',
                stdin_line_available_ns=None, worker_pre_START_demand_ns=None,
                busy_component_decomposition='UNKNOWN_NOT_SEPARATELY_OBSERVED',
                pipe_backpressure_identified=False, pure_stdin_idle_authenticated=False))
        prior[owner] = row
    require(len(records) == cohort['offered_requests'], 'Complete offered denominator changed')
    return fifo.encode(dict(schema=SCHEMA,
        status='OBSERVED_COMPLETE_ATTEMPT_FIFO_DIAGNOSTICS_NOT_A_TRANSPORT_MODEL',
        capture_id=plan['capture_id'], attempt_id=plan['attempt_id'], execution_id=plan['execution_id'],
        counts=counts(records), records=records,
        by_case=[dict(case_id=c['case_id'], **counts([r for r in records if r['case_id'] == c['case_id']])) for c in plan['cases']],
        by_owner=[dict(owner=o, **counts([r for r in records if r['owner'] == o])) for o in sorted({r['owner'] for r in records})],
        case_histories=case_history, priming_rows_used_only_for_history=len(priming),
        complete_offered_denominator_preserved=True, whole_attempt_history_used=True, case_history_reset=False,
        initial_priming_worker_state_assumed=False,
        sender_pressure_definition='Predecessor parent write record still pending at enqueue; not proof of pipe backpressure.',
        READY_processing_gates_worker_START=False, parent_write_record_gates_worker_START=False,
        idle_to_busy_transfer_used=False, busy_submit_to_START_used_as_pre_START_demand=False,
        fitted_transport_law=False, main_queue_provider_eligible=False, final_gain_eligible=False,
        native_evidence_authenticated=False, validation_labels_read=False, GPU_started=False))


def extract(training_review):
    # This existing adapter checks the new schema, completed owned supervisor,
    # every receipt, and exact independent review rederivation before labels.
    macro = macros_v2.extract(training_review)
    review_ref = receipt(training_review); review = checked(review_ref)
    raw = checked(review['raw']); plan = checked(raw['plan'])
    result = summarize(raw, plan, capture_sha256=review['raw']['sha256'])
    require(result['counts']['offered'] == macro['measured_requests'], 'Macro/transport offered denominator differs')
    require(all(h['captured_snapshot_present'] for h in result['case_histories']), 'Actual per-case prior snapshots required')
    sources = strict_references([review_ref, review['raw'], raw['plan'], raw['context'],
        macro['sources'], implementation_sources(__file__)])
    require(all(receipt(r['path']) == r for r in sources), 'Transport source changed during extraction')
    result.update(sources=sources, native_evidence_authenticated=True,
        qualification='Native observed-clock diagnostics only; no learned busy transport law or qualified predictor.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-review', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); result = extract(a.training_review)
    with Path(a.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result['counts']))
