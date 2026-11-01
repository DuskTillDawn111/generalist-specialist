"""Training-only disjoint worker clock, distinct from device and reader clocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage_response_features_v1 import checked, receipt, stamp, training_roles

SCHEMA = 'gsdeploy.disjoint_training_worker_clock.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def availability(event):
    values = [event.get(k) for k in ('perf_ns', 'received_time_ns',
        'collector_dequeued_time_ns', 'collector_processing_time_ns')]
    require(all(stamp(x) for x in values) and values == sorted(values),
        'Complete monotone occurrence/reader/dequeue/processing clock required')
    return dict(zip(('occurrence_ns', 'reader_receipt_ns', 'dequeued_ns', 'processed_ns'), values))


def partition_request(row):
    """Cover START->READY exactly once; never sum overlapping response channels.

    Labels may use the complete training record. None of these future endpoints
    are runtime features. READY occurrence and reader receipt stay separate.
    """
    response = row['worker_response']
    require(row.get('status') == 'completed' and response.get('success') is True,
        'Successful full training request required')
    start, end = response.get('start_perf_ns'), response.get('end_perf_ns')
    ready = row.get('previous_handler_next_request_ready', {})
    require(ready.get('request_id') == row['request_id']
        and ready.get('boundary') == 'previous_handler_next_request_ready'
        and ready.get('scope') == 'after_response_flush_before_next_read',
        'Exact request-bound post-flush READY evidence required')
    ready_clock = availability(ready)
    finish = ready_clock['occurrence_ns']
    received = row.get('response_reader_event_enqueue_time_ns')
    dequeued, processed = (row.get(k) for k in
        ('response_collector_dequeued_time_ns', 'response_collector_processing_time_ns'))
    require(all(stamp(x) for x in (start, end, finish, received, dequeued, processed))
        and row.get('observed_worker_start_time_ns') == start
        and start <= end <= finish and end <= received <= dequeued <= processed
        and row.get('completion_time_ns') == received,
        'Worker response, READY and reader clocks must remain ordered in their own branches')
    starts, intervals, last = {}, [], -1
    for event in row.get('execution_events', []):
        require(event.get('request_id') == row['request_id'], 'Phase request identity changed')
        at = event.get('perf_ns')
        require(stamp(at) and last <= at <= end and start <= at,
            'All worker events must retain source order inside handler')
        last = at
        # Queries are included in the surrounding wall clock, never extra time.
        if event.get('event') not in ('start', 'end'):
            continue
        clock = availability(event)
        key = event.get('kind'), event.get('segment_id')
        require(all(isinstance(k, str) and k for k in key), 'Explicit phase identity required')
        if event['event'] == 'start':
            require(key not in starts, 'Duplicate active phase BEGIN')
            starts[key] = event
        else:
            begin = starts.pop(key, None)
            require(begin is not None and begin['perf_ns'] <= at, 'Unpaired phase END')
            intervals.append(dict(kind=key[0], phase_id=key[1],
                semantic_phase=begin.get('semantic_phase'),
                left_ns=begin['perf_ns'], right_ns=at,
                begin_availability=availability(begin), end_availability=clock))
    require(not starts, 'Unclosed worker phases')
    require(len({(x['kind'], x['phase_id']) for x in intervals}) == len(intervals),
        'Phase identities must be unique within request')
    # Only disjoint or strict nested intervals have a coherent host partition.
    for i, a in enumerate(intervals):
        for b in intervals[i+1:]:
            overlap = max(a['left_ns'], b['left_ns']) < min(a['right_ns'], b['right_ns'])
            nested = ((a['left_ns'] <= b['left_ns'] and b['right_ns'] <= a['right_ns'])
                or (b['left_ns'] <= a['left_ns'] and a['right_ns'] <= b['right_ns']))
            require(not overlap or nested, 'Crossing host phases cannot be partitioned as nested calls')
            require(not overlap or (a['left_ns'], a['right_ns']) != (b['left_ns'], b['right_ns']),
                'Identical nested ranges have ambiguous leaf identity')
    cuts = sorted({start, end, finish, *[v for x in intervals for v in (x['left_ns'], x['right_ns'])]})
    pieces = []
    for left, right in zip(cuts, cuts[1:]):
        if left == right:
            continue
        active = sorted((x for x in intervals if x['left_ns'] <= left and right <= x['right_ns']),
            key=lambda x: (-(x['right_ns']-x['left_ns']), x['phase_id']))
        leaf = active[-1] if active else None
        if left >= end:
            label, kind = 'response_flush_and_ready_tail', 'host_tail'
        elif leaf is None:
            label, kind = 'handler_gap', 'host_gap'
        elif leaf['phase_id'] == 'native_inference':
            label, kind = 'forward_external_gap_in_native_inference', 'host_gap'
        else:
            label, kind = leaf['phase_id'], 'host_call_wall'
        pieces.append(dict(label=label, kind=kind, left_ns=left, right_ns=right,
            duration_ns=right-left, enclosing_phases=[x['phase_id'] for x in active],
            leaf_semantic_phase=leaf.get('semantic_phase') if leaf else None,
            is_device_service=False, synchronous_wait_may_be_included=bool(leaf and leaf['kind']=='sync')))
    total = sum(x['duration_ns'] for x in pieces)
    require(total == finish-start and all(a['right_ns'] == b['left_ns'] for a,b in zip(pieces, pieces[1:])),
        'Disjoint partition must reconstruct the whole worker occupancy without gaps')
    return dict(request_id=row['request_id'], owner=row['model'], task=row['task'], pool_id=row['pool_id'],
        profile_job_id=row['profile_job_id'], pieces=pieces,
        worker_handler_to_ready_ns=finish-start, worker_handler_to_response_end_ns=end-start,
        response_end_to_ready_ns=finish-end, reader_response_end_to_receipt_ns=received-end,
        reader_receipt_to_READY_receipt_ns=ready_clock['reader_receipt_ns']-received,
        response_reader_receipt_ns=received, response_processed_ns=processed,
        READY=ready_clock, nested_raw_intervals=intervals,
        training_labels_only=True, future_endpoints_are_runtime_features=False,
        independent_sync_and_device_wait_addition_permitted=False,
        physical_SM_service_identified=False, main_queue_provider_eligible=False)


def extract(report_path):
    report_ref = receipt(report_path)
    report = checked(report_ref)
    require(report.get('schema') == 'gsdeploy.profile_training_evidence_review.v1'
        and report.get('status') == 'PASS_COMPLETE_TRAINING_COLLECTION_NOT_PREDICTOR_QUALIFICATION'
        and report.get('training_feature_extraction_eligible') is True,
        'Complete reviewed training collection required')
    for reference in report['sources']:
        require(receipt(reference['path']) == reference, 'Training review source changed')
    raw_ref = report['inputs']['raw.json']
    raw = checked(raw_ref)
    plan, _, _ = training_roles(raw)
    rows = [partition_request(row) for row in raw['request_records']]
    sources = [report_ref, raw_ref, receipt(__file__)]
    require(all(receipt(ref['path']) == ref for ref in sources), 'Sources changed during extraction')
    return dict(schema=SCHEMA, status='EXACT_TRAINING_CLOCK_DECOMPOSITION_NOT_PREDICTIVE_MODEL',
        mode=plan['mode'], sources=sources, requests=rows,
        measured_requests=len(rows), priming_requests_excluded=len(raw['priming_request_records']),
        worker_clock_reconstruction_exact=True, fit_performed=False,
        validation_labels_read=False, GPU_started=False, final_gain_eligible=False,
        limitations=['Phase walls and host gaps are response labels, not measured effective SM service.',
            'Sync wall may already include device wait; adding an independent GPU wait is forbidden.',
            'Observed worker response and READY clocks are separate from parent-visible availability.',
            'Exact in-sample partition does not establish a conditional joint execution provider.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = extract(args.training_review)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], requests=result['measured_requests'])))
