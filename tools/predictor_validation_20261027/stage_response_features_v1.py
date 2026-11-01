"""Training-only response labels and causal, parent-visible phase evidence.

CPU/offline only. No predictor fit, CUDA, profiling launch, validation labels,
counter inference or physical-context service clock. This module is not imported
by the frozen profile implementation closure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'gsdeploy.stage_response_training_features.v1'
CONTRACT = {
    'schema': 'gsdeploy.stage_response_fixed_design.v1',
    'candidate_features': {
        'baseline': [],
        'visible_count': ['peer_visible_unconfirmed_owner_count'],
        'visible_phase_aware': ['peer_visible_unconfirmed_owner_count',
                                'peer_same_semantic_phase_owner_count'],
    },
    'response_domains': ['natural_host_call_wall_ns',
                         'natural_recorded_current_stream_elapsed_ns',
                         'profiler_perturbed_host_call_wall_ns',
                         'profiler_perturbed_recorded_current_stream_elapsed_ns'],
    'parameters': 'free real coefficients; exp(linear predictor) is positive; acceleration is allowed',
    'physical_context_is_service': False,
    'last_visible_unconfirmed_is_actual_unfinished': False,
    'same_semantic_phase_is_resource_similarity': False,
    'unsupported_or_rank_deficient': 'UNKNOWN; no coefficient or timing fallback',
    'fit_performed': False,
}
AVAILABILITY = ('received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns')


def stamp(value):
    return type(value) is int and value >= 0


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('Exact path/sha256 receipt required')
    if receipt(reference['path']) != reference:
        raise ValueError('Changed referenced source')
    value = json.loads(Path(reference['path']).read_text())
    if receipt(reference['path']) != reference:
        raise ValueError('Source changed during read')
    return value


def training_roles(raw):
    """Reject nontraining role before inspecting any response/phase label."""
    plan, context, executed = (checked(raw.get(k)) for k in ('plan', 'context', 'executed_jobs'))
    if (plan.get('schema') != 'gsdeploy.paired_profile_campaign.v6'
            or plan.get('split') != 'training' or plan.get('mode') not in ('natural', 'kernel')
            or plan.get('progress_observer') is not True
            or plan.get('mode_switch_at_execute') is not False
            or plan.get('no_retries') is not True
            or plan.get('final_gain_eligible') is not False):
        raise ValueError('Explicit v6 training-only observer plan required; smoke/validation forbidden')
    mode = plan['mode']
    scope = 'startup_through_final_drain_sample_before_worker_close'
    if (context.get('protocol') != 'paired_profile_v6_' + mode
            or context.get('paired_profile_mode') != mode
            or context.get('worker_variant') != 'semantic_progress_worker_v1.py'
            or context.get('monitoring_scope') != scope
            or raw.get('monitoring_scope') != scope
            or raw.get('shutdown_memory_sampled') is not False
            or raw.get('split', 'training') != 'training'
            or raw.get('execution_context_sha256') != plan.get('execution_context_sha256')
            or raw.get('performance_scope') != ('profiler_perturbed' if mode == 'kernel'
                                                else 'unprofiled_with_phase_markers')):
        raise ValueError('Exact training execution/timing-domain context required')
    jobs, rows = plan.get('jobs', []), raw.get('request_records', [])
    if not jobs or any(j.get('split') != 'training' or j.get('validation_view') != 'training' for j in jobs):
        raise ValueError('Every offered job must have training role')
    if executed != dict(split=None, contexts=None, limit=None, trace=mode == 'kernel', jobs=jobs):
        raise ValueError('Complete exact training jobs required')
    by_job = {j.get('id'): j for j in jobs}
    if len(by_job) != len(jobs) or None in by_job:
        raise ValueError('Unique offered training jobs required')
    expected = Counter((j['id'], a, pool, j.get('tasks', {}).get(a, 'ocr' if a == 'ocr#0' else 'text'))
                       for j in jobs for a, pool in j['inputs'].items())
    observed = Counter()
    for row in rows:
        if row.get('split') != 'training' or row.get('validation_view') != 'training':
            raise ValueError('All measured rows must have training role')
        key = row.get('profile_job_id'), row.get('model'), row.get('pool_id'), row.get('task')
        observed[key] += 1
    if observed != expected or len({r.get('request_id') for r in rows}) != len(rows):
        raise ValueError('Exact unique measured training request coverage required')
    if (raw.get('status') != 'completed'
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))
            or any(r.get('status') != 'completed' for r in rows)):
        raise ValueError('Completed clean training collection required; errors cannot be filtered')
    return plan, context, by_job


def ordered_events(rows):
    """Parent processing order is availability; retain worker occurrence too."""
    events = []
    for row in rows:
        last = -1
        for ordinal, event in enumerate(row.get('execution_events', [])):
            times = [event.get(k) for k in ('perf_ns', *AVAILABILITY)]
            if (event.get('request_id') != row['request_id'] or not all(stamp(v) for v in times)
                    or times != sorted(times) or times[-1] < last):
                raise ValueError('Ordered worker/reader/dequeue/processing availability required')
            last = times[-1]
            events.append(dict(event, owner=row['model'], job_id=row['profile_job_id'],
                               request_ordinal=ordinal))
    # Equal parent timestamps across different owners are a simultaneous visible
    # set; a deterministic tie-break does not invent a device service order.
    return sorted(events, key=lambda e: (e['collector_processing_time_ns'], e['owner'],
                                         e['request_id'], e['request_ordinal']))


def visible_prefix(events, cutoff, owner, semantic_phase):
    """Use only processed evidence; a false query never establishes truth-now."""
    pending, host_active, last_query = {}, {}, {}
    for event in events:
        if event['collector_processing_time_ns'] > cutoff:
            continue
        phase = event.get('semantic_phase')
        if phase is None and event.get('segment_id') == 'native_ocr':
            phase = 'ocr_inference'
        key = event['owner'], event['request_id'], event.get('segment_id')
        edge = event.get('event')
        if phase in ('prefill', 'decode', 'ocr_inference') and edge == 'start':
            value = dict(owner=event['owner'], request_id=event['request_id'],
                         phase_id=event['segment_id'], semantic_phase=phase,
                         input_sequence_length=event.get('input_sequence_length'),
                         input_batch_size=event.get('input_batch_size'),
                         forward_index=event.get('forward_index'),
                         begin_observed_ns=event['perf_ns'],
                         begin_available_ns=event['collector_processing_time_ns'])
            host_active[key] = value
            if phase != 'ocr_inference':
                pending[key] = value
        elif edge == 'end':
            host_active.pop(key, None)
        elif edge == 'observation' and event.get('kind') == 'device_event_query':
            if type(event.get('end_event_complete')) is not bool:
                raise ValueError('Device query outcome must be explicit boolean')
            last_query[key] = dict(end_event_complete=event['end_event_complete'],
                                   observed_ns=event['perf_ns'],
                                   available_ns=event['collector_processing_time_ns'])
            if event['end_event_complete']:
                pending.pop(key, None)
    # Uninstrumented OCR contributes only a visible host call, never a certified
    # completed GPU frontier or inferred outstanding-work fraction.
    peers = {k: v for k, v in pending.items() if k[0] != owner}
    peers.update({k: v for k, v in host_active.items()
                  if k[0] != owner and v['semantic_phase'] == 'ocr_inference'})
    peer_owners = {v['owner'] for v in peers.values()}
    same = {v['owner'] for v in peers.values() if v['semantic_phase'] == semantic_phase}
    return dict(cutoff_processing_ns=cutoff,
        peer_visible_unconfirmed_owner_count=len(peer_owners),
        peer_same_semantic_phase_owner_count=len(same),
        peer_visible_unconfirmed_phases=[dict(v, last_processed_query=last_query.get(k))
                                        for k, v in sorted(peers.items())],
        peer_visible_host_active_phases=[v for k, v in sorted(host_active.items()) if k[0] != owner],
        recorded_stream_frontier_only=True,
        all_request_GPU_frontiers_observed=False,
        unconfirmed_is_actual_unfinished=False,
        OCR_device_probe='UNKNOWN_NOT_INSTRUMENTED',
        physical_SM_service=None,
        nominal_flops=None, declared_bytes=None, resource_similarity_proxy=None,
        static_resource_feature_status='UNKNOWN_NO_REVIEWED_STATIC_TOPOLOGY_RECEIPT',
        future_response_fields_used=False)


def matrix_rank(rows):
    """Small fixed-width design diagnostic; no statistical precision claim."""
    if not rows:
        return 0
    a = [[float(x) for x in row] for row in rows]
    width = len(a[0])
    if any(len(row) != width or any(not math.isfinite(x) for x in row) for row in a):
        raise ValueError('Finite rectangular feature matrix required')
    rank = 0
    for col in range(width):
        pivot = max(range(rank, len(a)), key=lambda i: abs(a[i][col]), default=None)
        if pivot is None or abs(a[pivot][col]) <= 1e-10:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        value = a[rank][col]
        a[rank] = [x / value for x in a[rank]]
        for i in range(rank + 1, len(a)):
            value = a[i][col]
            a[i] = [x - value * y for x, y in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def design_check(stages):
    """First necessary feature-rank check; nonlinear fit Jacobian remains due."""
    result = []
    groups = {}
    for stage in stages:
        key = (stage['owner'], stage['semantic_phase'])
        groups.setdefault(key, []).append(stage)
    for (owner, phase), members in sorted(groups.items()):
        blocks = {s['capture_sha256'] for s in members}
        for name, features in CONTRACT['candidate_features'].items():
            matrix = [[state[f] for f in features] for s in members
                      for state in s['processed_covariate_path']]
            rank = matrix_rank(matrix)
            # One row per phase does not make phases independent samples.
            result.append(dict(owner=owner, semantic_phase=phase, candidate=name,
                columns=features, rank=rank, required_rank=len(features),
                full_column_rank=rank == len(features), stage_rows=len(members),
                rank_scope='observed causal-prefix states, not independent phase samples or fitted Jacobian',
                whole_capture_blocks=len(blocks), minimum_initial_repetition_design_met=len(blocks) >= 3,
                status='NECESSARY_DESIGN_CHECK_ONLY_NO_FIT_QUALIFICATION',
                nonlinear_integrated_hazard_Jacobian_checked=False))
    return result


def phase_rows(row, job, events, mode, capture_sha256):
    result, starts = [], {}
    response = row.get('worker_response', {})
    semantic = response.get('semantic_forward_observations', {})
    if semantic.get('instrumentation_errors'):
        raise ValueError('Instrumentation errors cannot be discarded')
    records = {r['id']: r for r in semantic.get('records', [])}
    devices = {r['phase_id']: r for r in semantic.get('device_progress', {}).get('records', [])}
    for event in row.get('execution_events', []):
        key = event.get('kind'), event.get('segment_id')
        if event.get('event') == 'start':
            if key in starts:
                raise ValueError('Duplicate stage BEGIN')
            starts[key] = event
        elif event.get('event') == 'end':
            begin = starts.pop(key, None)
            if begin is None or event['perf_ns'] < begin['perf_ns']:
                raise ValueError('Unpaired/nonmonotone stage END')
            phase = begin.get('semantic_phase')
            stage_id = begin['segment_id']
            if phase not in ('prefill', 'decode'):
                phase = 'ocr_inference' if stage_id == 'native_ocr' else stage_id
            kind = 'language_forward' if phase in ('prefill', 'decode') else 'host_stage'
            cutoff = begin['collector_processing_time_ns']
            features = visible_prefix(events, cutoff, row['model'], phase)
            labels = dict(host_call_wall_ns=event['perf_ns'] - begin['perf_ns'],
                host_label_domain='natural_host_call_wall_ns' if mode == 'natural'
                                  else 'profiler_perturbed_host_call_wall_ns',
                host_observed_interval_ns=[begin['perf_ns'], event['perf_ns']],
                recorded_current_stream_elapsed_ns=None,
                recorded_stream_label_domain='natural_recorded_current_stream_elapsed_ns' if mode == 'natural'
                                             else 'profiler_perturbed_recorded_current_stream_elapsed_ns',
                current_stream_completion_interval_ns=None,
                label_uses_training_future=True, physical_SM_service_ns=None,
                host_and_stream_durations_additive=False,
                current_stream_interval_is_service=False)
            labels['remaining_host_wall_at_BEGIN_available_ns']=max(0,event['perf_ns']-cutoff)
            labels['host_call_ended_before_BEGIN_available']=event['perf_ns']<=cutoff
            if kind == 'language_forward':
                record = records.get(stage_id)
                if (record is None or record.get('start_ns') != begin['perf_ns']
                        or record.get('end_ns') != event['perf_ns']
                        or any(record.get(f) != begin.get(f) for f in
                               ('semantic_phase', 'forward_index', 'input_sequence_length', 'input_batch_size'))):
                    raise ValueError('Exact BEGIN/response semantic record binding required')
                device = devices.get(stage_id)
                queries = [e for e in row['execution_events']
                           if e.get('event') == 'observation' and e.get('phase_id') == stage_id]
                if device is None or 'observation_error' in device:
                    raise ValueError('Each language phase requires an error-free probe record')
                lower = device['end_record_host_bracket_ns'][0]
                upper = None
                for query in queries:
                    if query.get('end_event_complete') is True:
                        upper = query['host_query_end_ns']
                        elapsed = query.get('recorded_stream_elapsed_ms')
                        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                                or elapsed != device.get('recorded_stream_elapsed_ms')):
                            raise ValueError('Finite completed-stream label must match record')
                        labels['recorded_current_stream_elapsed_ns'] = elapsed * 1e6
                        break
                    if query.get('end_event_complete') is False:
                        lower = max(lower, query['host_query_begin_ns'])
                if upper is not None and lower > upper:
                    raise ValueError('Inconsistent device-completion brackets')
                labels['current_stream_completion_interval_ns'] = dict(lower_ns=lower, upper_ns=upper,
                    right_censored=upper is None, scope='recorded_current_stream_event_only')
            changes = sorted({cutoff, *[e['collector_processing_time_ns'] for e in events
                                        if cutoff < e['collector_processing_time_ns'] < event['perf_ns']]})
            path = [visible_prefix(events, at, row['model'], phase) for at in changes]
            horizon=max(cutoff,event['perf_ns'])
            exposure={name:0 for name in CONTRACT['candidate_features']['visible_phase_aware']}
            for index,state in enumerate(path):
                stop=path[index+1]['cutoff_processing_ns'] if index+1<len(path) else horizon
                width=max(0,stop-state['cutoff_processing_ns'])
                for name in exposure:
                    exposure[name]+=width*state[name]
            result.append(dict(capture_sha256=capture_sha256, job_id=job['id'], owner=row['model'],
                request_id=row['request_id'], pool_id=row['pool_id'], task=row['task'], phase_id=stage_id,
                semantic_phase=phase, stage_kind=kind, forward_index=begin.get('forward_index'),
                input_sequence_length=begin.get('input_sequence_length'), input_batch_size=begin.get('input_batch_size'),
                BEGIN_available_ns=cutoff, runtime_features_at_begin=features,
                processed_covariate_path=path,
                path_scope='each state available only at its own cutoff; future changes must be forecast during simulation',
                training_only_design_exposures=dict(window_start_ns=cutoff,window_end_ns=horizon,
                    weighted_evidence_exposure_ns=exposure,
                    future_label_endpoint_used=True,runtime_feature=False),
                offered_offset_ms=job['start_offsets_ms'][row['model']],
                response_labels=labels,
                additive_scope='inclusive_parent_not_additive' if stage_id == 'native_inference'
                               else 'host_call_wall_only_stream_response_is_separate',
                device_frontier_ready=False))
    if starts:
        raise ValueError('Unclosed training stages')
    return result


def extract(path):
    rawref = receipt(path)
    raw = checked(rawref)
    plan, context, jobs = training_roles(raw)
    stages = []
    for job_id, job in jobs.items():
        rows = [r for r in raw['request_records'] if r['profile_job_id'] == job_id]
        events = ordered_events(rows)
        for row in rows:
            stages.extend(phase_rows(row, job, events, plan['mode'], rawref['sha256']))
    sources = [rawref, raw['plan'], raw['context'], raw['executed_jobs'], receipt(__file__)]
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Training source changed during extraction')
    return dict(schema=SCHEMA, status='TRAINING_FEATURES_ONLY_NOT_FIT_OR_QUALIFICATION',
        mode=plan['mode'], sources=sources, fixed_contract=CONTRACT,
        stages=stages, design_checks=design_check(stages),
        priming_rows_excluded=len(raw.get('priming_request_records', [])),
        static_resource_features_status='UNKNOWN_NO_REVIEWED_STATIC_TOPOLOGY_RECEIPT',
        fit_performed=False, validation_labels_read=False, GPU_started=False,
        physical_context_used_as_service=False, unseen_routing_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-raw', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = extract(args.training_raw)
    with Path(args.output).open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(status=result['status'], stages=len(result['stages']))))


if __name__ == '__main__':
    main()
