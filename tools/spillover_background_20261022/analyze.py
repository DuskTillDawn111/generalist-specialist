"""Verify terminal background receipts and report observed context coverage.

This is a development diagnostic. Completed cohorts and descriptive service
tables do not establish open-loop capacity or activate a prediction model.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics
import traceback

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


legacy = module('_background_activity', ROOT/'tools/spillover_capacity_20261022/analyze.py')
independent = module('_background_response_audit', ROOT/'tools/spillover_capacity_20261022/audit_sustained_failure.py')
runner = module('_background_runner_for_analysis', Path(__file__).with_name('run.py'))


def sha(path):
    return independent.sha(path)


def merged_intervals(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def overlap_fraction(begin, end, left, right=None):
    """Observed intersection, counting each alias at most once."""
    if end <= begin:
        raise ValueError('Positive observed handler interval required')
    left = merged_intervals(left)
    if right is None:
        return sum(max(0, min(end, b)-max(begin, a)) for a, b in left)/(end-begin)
    right = merged_intervals(right)
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]; c, d = right[j]
        total += max(0, min(end, b, d)-max(begin, a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total/(end-begin)


def activity_context(fraction, unknown_fraction=0):
    if fraction >= .9:
        return 'BUSY_GE_90PCT'
    if unknown_fraction > 0:
        return 'UNKNOWN_HANDLER_COVERAGE' if fraction == 0 else 'PARTIAL_OBSERVED_UNKNOWN_REMAINDER'
    return 'IDLE' if fraction == 0 else 'PARTIAL'


def score_observed_native(row, response, references, score_result, sample_totals):
    # Match frozen measure_quality's task preprocessing, including page OCR
    # normalization before edit distance. Keep the actual-output hash separate.
    adjusted = dict(response)
    if row['task'] == 'ocr' and 'prediction' in adjusted:
        adjusted['prediction'] = score_result.__globals__['page_ocr_text'](adjusted['prediction'])
    result = legacy.score_native(row, adjusted, references, score_result, sample_totals)
    if 'prediction' in response:
        result['prediction_sha256'] = independent.digest(response['prediction'])
    return result


def resource_summary(raw, config, start, end, offset):
    """Memory monitor uses Unix ns; power monitor uses monotonic ns."""
    memory = raw.get('memory_evidence') or {}; samples = memory.get('samples', [])
    safe = config['catalog']['safe_factor']*max((r['owned_used_mib'] for r in samples), default=0) if samples else None
    if safe is not None:
        assert math.isclose(safe, memory['safe_memory_mib'], abs_tol=1e-7)
    bracket = bool(samples) and min(r['unix_ns'] for r in samples) <= start+offset and max(r['unix_ns'] for r in samples) >= end+offset
    memory_pass = (bracket and memory.get('measured') is True and memory.get('resource_valid') is True
                   and safe <= config['gpu_memory_mib'] and all(not r['foreign_pids'] for r in samples))
    energy = raw.get('energy_evidence') or {}; joules = startup = None; energy_error = None
    try:
        joules = independent.integrate(energy['samples'], start, end)
        startup = independent.integrate(energy['samples'], energy['startup_begin_time_ns'], start)
        assert math.isclose(joules, energy['run_joules'], abs_tol=1e-6)
        assert math.isclose(startup, energy['startup_joules'], abs_tol=1e-6)
    except (AssertionError, KeyError, ValueError, TypeError, IndexError) as exc:
        energy_error = type(exc).__name__+': '+str(exc)
    energy_pass = energy_error is None and energy.get('coverage_valid') is True and not energy.get('sampling_errors')
    return dict(memory_full_window_bracket=bracket, memory_pass=bool(memory_pass), safe_memory_mib=safe,
                memory_clock='unix_ns matched with measurement Unix/monotonic offset', run_joules=joules,
                startup_joules=startup, energy_pass=bool(energy_pass), energy_audit_error=energy_error)


def classify(row, start, cutoff, offset):
    response = row.get('worker_response') or {}
    if row.get('scheduled_time_ns') is None:
        return 'UNRELEASED_UNKNOWN'
    if row.get('status') != 'completed' or response.get('success') is not True:
        return 'RELEASED_WITHOUT_COMPLETION_UNKNOWN'
    if max(row['completion_time_ns'], response['end_ns']-offset) > cutoff:
        return 'AFTER_CUTOFF'
    if response['start_ns'] < start+offset:
        raise ValueError('Measurement handler began before measurement release')
    return 'WITHIN_CUTOFF'


def verify_serial(rows):
    """Raw storage order differs from actual cross-role admission order."""
    admitted = sorted((r for r in rows if r['scheduled_time_ns'] is not None),
                      key=lambda r: r['admission_sequence'])
    assert [r['admission_sequence'] for r in admitted] == list(range(len(admitted)))
    for before, after in zip(admitted, admitted[1:]):
        assert before['status'] == 'completed'
        assert before['completion_time_ns'] <= after['scheduled_time_ns']
        assert before['worker_response']['end_ns'] <= after['worker_response']['start_ns']
    return admitted


def analyze_job(packet, job, raw, packet_sha):
    assert raw['packet_sha256'] == packet_sha and raw['job'] == job
    assert raw['protocol'] == runner.PROTOCOL and raw['status'] in {'completed', 'failed'}
    assert raw['job_identity_sha256'] == runner.job_digest(job)
    assert raw['replica_to_family'] == packet['aliases']
    config = json.loads(Path(packet['system_config']).read_text())
    workers = json.loads(Path(packet['workers']).read_text())
    assert raw['config'] == config
    assert raw['preflight']['ready'] and raw['priming_path_identity']['verified']
    for boundary in ['gpu_boundary_preflight', 'gpu_before_worker_start']:
        assert raw[boundary]['uuid'] == packet['gpu_uuid'] and not raw[boundary]['compute_pids']
    specs = {}
    for alias, family in packet['aliases'].items():
        specs[family] = json.loads(Path(workers[family]['spec']).read_text())
        ready = raw['worker_ready'][alias]
        assert ready['worker'] == family
        assert ready['spec_sha256'] == sha(workers[family]['spec'])
        assert ready['model'] == specs[family]['candidate']['model']
        assert set(ready['tasks']) == set(config['catalog']['models'][family]['supported_tasks'])
    assert len({r['pid'] for r in raw['worker_ready'].values()}) == len(packet['aliases'])
    pool_rows = [json.loads(s) for s in Path(packet['samples']).read_text().splitlines() if s.strip()]
    pool = {r['pool_id']: r for r in pool_rows}
    # Input pool deliberately excludes answers. Score with bound references.
    references = config['catalog']['quality']['references']
    for task in {r['task_type'] for r in pool_rows}:
        assert set(references[task]) == {r['pool_id'] for r in pool_rows if r['task_type'] == task}
    score_result, sample_totals = legacy.frozen_scorers(packet['frozen_source'])
    measured = raw['request_records']
    expected = {r['request_id']: r for values in job['planned_requests'].values() for r in values}
    for alias in job['support_requests']:
        actual = [r for r in measured if r['model'] == alias and r['role'] == 'support']
        for row, template in zip(actual, job['support_requests'][alias]):
            assert all(row[k] == v for k, v in template.items())
            expected[row['request_id']] = template
        assert len(actual) <= 1024
    assert len(expected) == len(measured) == len({r['request_id'] for r in measured})
    for row in measured:
        assert all(row[k] == v for k, v in expected[row['request_id']].items())
    for row in raw['priming_request_records']+raw['warmup_request_records']+measured:
        if row['status'] != 'completed':
            continue
        response = row['worker_response']
        assert response['success'] is True
        for key in ['request_id', 'pool_id', 'arrival_index']:
            assert response[key] == row[key]
        assert response['worker'] == packet['aliases'][row['model']]
        assert response['task_type'] == row['task'] == pool[row['pool_id']]['task_type']
        independent.check_response_protocol(response, specs[packet['aliases'][row['model']]], row['request_id'])
        assert row['scheduled_time_ns'] <= row['enqueued_time_ns'] <= row['actual_submit_time_ns'] <= row['completion_time_ns']
        assert response['submit_ns'] == row['actual_submit_unix_ns']
        assert response['submit_ns'] <= response['start_ns'] < response['end_ns'] <= row['completion_unix_ns']
        assert 0 < response['service_ms'] <= (response['end_ns']-response['start_ns'])/1e6
    assert len(raw['priming_request_records']) == 21
    assert [{k: r[k] for k in ['model', 'task', 'pool_id']} for r in raw['priming_request_records']] == packet['priming_requests']
    window = raw['measurement_window']
    assert window and raw['warmup_window']['duration_seconds'] >= 30
    assert max(r['completion_time_ns'] for r in raw['warmup_request_records']) <= window['start_time_ns']
    start = window['start_time_ns']; cutoff = min(window['end_time_ns'], start+600_000_000_000)
    offset = window['start_unix_ns']-start
    intervals = defaultdict(list); task_intervals = defaultdict(list); unknown_intervals = defaultdict(list)
    for row in measured:
        response = row.get('worker_response') or {}
        begin, end = response.get('start_ns'), response.get('end_ns')
        if type(begin) is int and type(end) is int and begin < end:
            a, b = max(begin, start+offset), min(end, cutoff+offset)
            if b > a:
                intervals[row['model']].append((a, b))
                task_intervals[(row['model'], row['task'])].append((a, b))
        elif row.get('scheduled_time_ns') is not None:
            a = row['scheduled_time_ns']+offset
            if a < cutoff+offset:
                unknown_intervals[row['model']].append((a, cutoff+offset))
    classes = Counter(); details = []; groups = defaultdict(list)
    for alias in packet['aliases']:
        admitted = verify_serial([r for r in measured if r['model'] == alias])
        assert len(admitted) == window['released_by_alias'][alias]
    for row in measured:
        label = classify(row, start, cutoff, offset); classes[(row['model'], row['role'], label)] += 1
        result = {k: row[k] for k in ['request_id', 'model', 'task', 'role', 'pool_id', 'repetition']}
        result.update(deadline_class=label, service_ms=None, both_text_handler_fraction=None, quality=None)
        if label == 'WITHIN_CUTOFF':
            response = row['worker_response']; begin, end = response['start_ns'], response['end_ns']
            result.update(service_ms=response['service_ms'], handler_ms=(end-begin)/1e6,
                both_text_handler_fraction=overlap_fraction(begin, end, intervals['text#0'], intervals['text#1']),
                ocr_handler_fraction=overlap_fraction(begin, end, intervals['ocr#0']),
                generalist_handler_fraction=overlap_fraction(begin, end, intervals['generalist_2#0']),
                quality=score_observed_native(row, response, references, score_result, sample_totals))
            result['generalist_task_handler_fraction'] = {t: overlap_fraction(begin, end, task_intervals[('generalist_2#0', t)])
                                                          for t in ['text', 'image', 'asr']}
            uncertain = {a: overlap_fraction(begin, end, unknown_intervals[a]) for a in packet['aliases']}
            result['unknown_handler_possible_fraction_by_alias'] = uncertain
            result['activity_context'] = dict(both_text=activity_context(result['both_text_handler_fraction'],
                max(uncertain['text#0'], uncertain['text#1'])),
                ocr=activity_context(result['ocr_handler_fraction'], uncertain['ocr#0']),
                generalist=activity_context(result['generalist_handler_fraction'], uncertain['generalist_2#0']))
            groups[(row['model'], row['task'], row['role'], row['pool_id'])].append(result)
        details.append(result)
    per_input = []; coverage = []
    for alias, templates in job['planned_requests'].items():
        for task, pid in sorted({(r['task'], r['pool_id']) for r in templates}):
            rows = groups[(alias, task, 'primary', pid)]
            covered = [r for r in rows if r['both_text_handler_fraction'] >= .9]
            record = dict(alias=alias, task=task, pool_id=pid, completed=len(rows), covered=len(covered),
                          covered_median_service_ms=statistics.median(r['service_ms'] for r in covered) if covered else None)
            record['covered_by_ocr_context'] = {context: dict(count=len(values), median_service_ms=statistics.median(values))
                for context in sorted({r['activity_context']['ocr'] for r in covered})
                if (values := [r['service_ms'] for r in covered if r['activity_context']['ocr'] == context])}
            per_input.append(record); coverage.append(len(covered) >= 2)
    resources = resource_summary(raw, config, start, window['end_time_ns'], offset)
    shutdown_pass = set(raw['shutdown']) == set(packet['aliases']) and all(
        r['exit_code'] == 0 and not r['forced'] and r['bye_received'] for r in raw['shutdown'].values())
    complete = all(r['deadline_class'] == 'WITHIN_CUTOFF' for r in details if r['role'] == 'primary')
    contexts = defaultdict(list)
    for r in details:
        if r['deadline_class'] == 'WITHIN_CUTOFF':
            contexts[(r['model'], r['task'], r['role'], r['activity_context']['both_text'], r['activity_context']['ocr'])].append(r)
    context_summary = [dict(alias=a, task=t, role=role, both_text_context=tc, ocr_context=oc,
        requests=len(rows), unique_inputs=len({r['pool_id'] for r in rows}),
        median_service_ms=statistics.median(r['service_ms'] for r in rows),
        mean_service_ms=statistics.mean(r['service_ms'] for r in rows),
        max_service_ms=max(r['service_ms'] for r in rows)) for (a,t,role,tc,oc),rows in sorted(contexts.items())]
    quality_complete = all(r['quality'] and r['quality']['status'] == 'AVAILABLE' for r in details
                           if r['deadline_class'] == 'WITHIN_CUTOFF')
    return dict(job_id=job['job_id'], block=job['block'], seed=job['seed'], raw_status=raw['status'],
        audit_status='VERIFIED_TERMINAL_EVIDENCE',
        raw_errors=raw['errors'], duration_seconds=(cutoff-start)/1e9,
        classifications=[dict(alias=a, role=r, deadline_class=c, count=n) for (a, r, c), n in sorted(classes.items())],
        primary_complete=complete, full_input_overlap_coverage=bool(coverage) and all(coverage),
        **resources, shutdown_pass=shutdown_pass, native_quality_complete=quality_complete,
        context_service_summary=context_summary, per_input=per_input, requests=details, U=None)


def analyze_preserving_failure(packet, job, raw, packet_sha):
    try:
        return analyze_job(packet, job, raw, packet_sha)
    except (AssertionError, KeyError, ValueError, TypeError, IndexError, OSError) as exc:
        # Startup failure or malformed partial evidence stays receipt-linked.
        return dict(job_id=job['job_id'], block=job['block'], seed=job['seed'], raw_status=raw.get('status'),
            audit_status='AUDIT_INCOMPLETE_OR_FAILED', audit_error=type(exc).__name__+': '+str(exc),
            audit_traceback=traceback.format_exc(), raw_errors=raw.get('errors', []),
            raw_record_counts=dict(Counter((r.get('role','unknown')+':'+r.get('status','unknown')) for r in raw.get('request_records', []))),
            primary_complete=False, full_input_overlap_coverage=False, memory_pass=False, energy_pass=False,
            shutdown_pass=False, native_quality_complete=False, per_input=[], requests=[], U=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True); p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite diagnostic evidence')
    packet = json.loads(args.packet.read_text()); runner.validate_packet(packet)
    queue = json.loads(args.queue.read_text()); packet_sha = sha(args.packet)
    assert queue['packet_sha256'] == packet_sha and queue['status'] != 'RUNNING'
    jobs = []; receipts = []
    for registration in queue['jobs']:
        path = Path(registration['raw']['path']); assert sha(path) == registration['raw']['sha256']
        job = next(j for j in packet['jobs'] if j['job_id'] == registration['job_id'])
        jobs.append(analyze_preserving_failure(packet, job, json.loads(path.read_text()), packet_sha))
        receipts.append(registration['raw'])
    paired = []; repeat_pass = len(jobs) == 2 and all(j['per_input'] for j in jobs)
    if repeat_pass:
        first = {(r['alias'], r['task'], r['pool_id']): r for r in jobs[0]['per_input']}
        for row in jobs[1]['per_input']:
            left = first[(row['alias'], row['task'], row['pool_id'])]
            a = left['covered_median_service_ms']
            b = row['covered_median_service_ms']; ratio = b/a if a and b else None
            passed = ratio is not None and .5 <= ratio <= 2
            repeat_pass &= passed
            strata = []
            for context in sorted(set(left['covered_by_ocr_context']) | set(row['covered_by_ocr_context'])):
                la = left['covered_by_ocr_context'].get(context); rb = row['covered_by_ocr_context'].get(context)
                strata.append(dict(ocr_context=context, left_count=la['count'] if la else 0, right_count=rb['count'] if rb else 0,
                    median_service_ratio=rb['median_service_ms']/la['median_service_ms'] if la and rb else None,
                    scope='DESCRIPTIVE_SAME_OBSERVED_CONTEXT_NOT_NEW_ADMISSION_GATE'))
            paired.append(dict(alias=row['alias'], task=row['task'], pool_id=row['pool_id'], ratio=ratio, passed=passed,
                ocr_context_ratios=strata, pooled_ratio_may_mix_ocr_contexts=True))
    passed = bool(jobs) and repeat_pass and all(j['raw_status'] == 'completed' and j['primary_complete'] and
        j['full_input_overlap_coverage'] and j['memory_pass'] and j['energy_pass'] and j['shutdown_pass'] and j['native_quality_complete'] for j in jobs)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='gsdeploy.background_joint_diagnostic.analysis.v1',
        status='DIAGNOSTIC_COVERAGE_PASS' if passed else 'INCOMPLETE_OR_FAILED_DIAGNOSTIC_COVERAGE',
        jobs=jobs, cross_block_ratios=paired, cross_block_repeatability=repeat_pass,
        predictor_activated=False, formal_capacity=None, U=None,
        source_packet=dict(path=str(args.packet.resolve()), sha256=packet_sha), raw_receipts=receipts,
        limitations=['Finite balanced closed-loop diagnostic; not formal utility or open-loop capacity.',
            'Observed overlap is retrospective, never a prediction feature.',
            'Missing/late observations remain unknown; support does not substitute for primary.'])
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(tool_sha256=sha(__file__),
        report_sha256=sha(args.output/'report.json'), raw_receipts=receipts), indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], jobs=len(jobs), output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
