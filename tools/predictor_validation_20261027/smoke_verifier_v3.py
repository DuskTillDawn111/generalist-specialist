"""Independent CPU review of original four-owner smoke, with complete coverage.

Does not launch/export traces: it re-reads existing SQLite exports and validates
the complete immutable input/response/anchor chain. Output is exclusive. A PASS
qualifies collection/clock correctness only, never task quality or speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

from .native_collect import context_identity, receipt, save
from .profiling import RESIDENTS
from .trace_export import parse_sqlite


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def stamp(value):
    return type(value) is int and value >= 0


def check_receipt(value, path=None):
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and (path is None or Path(value['path']).resolve() == Path(path).resolve())
            and receipt(value['path']) == value)


def validate_coverage(plan, raw, executed):
    issues = []
    jobs = plan.get('jobs', [])
    if not jobs or any(j.get('split') != 'smoke' for j in jobs):
        issues.append('NONEMPTY_SMOKE_PLAN_REQUIRED')
    if executed.get('jobs') != jobs or executed.get('trace') is not True:
        issues.append('EXECUTED_JOBS_NOT_EXACT_TRACED_SMOKE_PLAN')
    expected = Counter((j['id'], a, p, j.get('tasks', {}).get(a, 'ocr' if a == 'ocr#0' else 'text'),
                        j['split'], j['validation_view']) for j in jobs for a, p in j['inputs'].items())
    rows = raw.get('request_records', [])
    observed = Counter((r.get('profile_job_id'), r.get('model'), r.get('pool_id'), r.get('task'),
                        r.get('split'), r.get('validation_view')) for r in rows)
    if expected != observed or len(expected) != len(rows) or {r.get('model') for r in rows} != set(RESIDENTS):
        issues.append('PLANNED_REQUEST_COVERAGE_MISMATCH')
    rids = [r.get('request_id') for r in rows]
    if any(not isinstance(rid, str) or not rid for rid in rids) or len(set(rids)) != len(rids):
        issues.append('DUPLICATE_OR_INVALID_REQUEST_IDS')
    batches = raw.get('batches', [])
    if Counter(b.get('job_id') for b in batches) != Counter(j['id'] for j in jobs):
        issues.append('JOB_BATCH_COVERAGE_MISMATCH')
    for batch in batches:
        matching = [r['request_id'] for r in rows if r.get('profile_job_id') == batch.get('job_id')]
        if Counter(batch.get('request_ids', [])) != Counter(matching):
            issues.append('BATCH_REQUEST_COVERAGE_MISMATCH')
    if Counter(rid for b in batches for rid in b.get('request_ids', [])) != Counter(rids):
        issues.append('BATCH_GLOBAL_REQUEST_COVERAGE_MISMATCH')
    return issues


def validate_request(row, pid):
    """Verify START identity, clocks, next-ready, and pair exact host anchors."""
    rid, issues, anchors, starts = row.get('request_id'), [], [], {}
    response = row.get('worker_response', {})
    begin, end = response.get('start_perf_ns'), response.get('end_perf_ns')
    received = row.get('response_reader_event_enqueue_time_ns')
    stamps = [begin, end, received, row.get('completion_time_ns'),
              row.get('response_collector_dequeued_time_ns'), row.get('response_collector_processing_time_ns')]
    if (row.get('status') != 'completed' or response.get('request_id') != rid
            or response.get('task_type') != row.get('task') or response.get('success') is not True
            or not all(stamp(v) for v in stamps)
            or not begin < end <= received <= stamps[4] <= stamps[5]
            or row['completion_time_ns'] != received):
        return ['HANDLER_RESPONSE_CHAIN_INVALID:'+str(rid)], []
    if not (stamp(row.get('scheduled_time_ns')) and stamp(row.get('actual_submit_time_ns'))
            and row['scheduled_time_ns'] <= row['actual_submit_time_ns'] <= begin):
        issues.append('ARRIVAL_SUBMIT_START_ORDER_INVALID:'+rid)
    start_events = [e for e in row.get('execution_events', []) if 'event' not in e]
    if len(start_events) != 1 or any(e.get('pid') != pid or e.get('perf_ns') != begin
                                   or e.get('request_id') != rid for e in start_events):
        issues.append('START_PID_IDENTITY_MISMATCH:'+rid)
    for event in row.get('execution_events', []):
        host, parent = event.get('perf_ns'), event.get('received_time_ns')
        if (event.get('request_id') != rid or not stamp(host) or not stamp(parent)
                or not begin <= host <= end or host > parent
                or not stamp(event.get('collector_dequeued_time_ns'))
                or not stamp(event.get('collector_processing_time_ns'))
                or not parent <= event['collector_dequeued_time_ns'] <= event['collector_processing_time_ns']):
            issues.append('OBSERVATION_BOUNDARY_CHAIN_INVALID:'+rid)
            continue
        if 'event' not in event:
            continue
        key = (rid, event.get('kind'), event.get('segment_id'))
        if event.get('event') == 'start':
            if key in starts:
                issues.append('DUPLICATE_PHASE_START:'+rid)
            starts[key] = host
        elif event.get('event') == 'end':
            start = starts.pop(key, None)
            if start is None or host <= start:
                issues.append('UNPAIRED_PHASE_END:'+rid)
            else:
                anchors.append(dict(pid=pid, request_id=rid, kind=key[1], segment_id=key[2], start_ns=start, end_ns=host))
        else:
            issues.append('UNKNOWN_PHASE_EDGE:'+rid)
    if starts or not anchors:
        issues.append('INCOMPLETE_PHASE_ANCHORS:'+rid)
    if len({(a['kind'], a['segment_id']) for a in anchors}) != len(anchors):
        issues.append('DUPLICATE_PHASE_IDENTITIES:'+rid)
    nxt = row.get('previous_handler_next_request_ready', {})
    if (nxt.get('request_id') != rid or nxt.get('pid') != pid
            or nxt.get('clock') != 'host_perf_counter_ns'
            or nxt.get('boundary') != 'previous_handler_next_request_ready'
            or not all(stamp(nxt.get(k)) for k in ('perf_ns', 'received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns'))
            or not end <= nxt['perf_ns'] <= nxt['received_time_ns'] <= nxt['collector_dequeued_time_ns'] <= nxt['collector_processing_time_ns']):
        issues.append('WORKER_NEXT_READY_CHAIN_INVALID:'+rid)
    if any(not finite(response.get(k), positive=True) for k in ('service_ms', 'worker_latency_ms')):
        issues.append('SERVICE_OBSERVATION_INVALID:'+rid)
    semantic = response.get('semantic_forward_observations', {})
    if row.get('model') != 'ocr#0' and (not semantic.get('records') or semantic.get('instrumentation_errors')):
        issues.append('SEMANTIC_FORWARD_OBSERVATION_INVALID:'+rid)
    return issues, anchors


def validate_telemetry(raw):
    issues, samples = [], raw.get('board_samples', [])
    start, end = raw.get('observation_start_ns'), raw.get('observation_end_ns')
    if (not stamp(start) or not stamp(end) or start >= end or not samples
            or not samples[0].get('time_ns', end+1) <= start < end <= samples[-1].get('time_ns', 0)):
        return ['TELEMETRY_DOES_NOT_ENVELOPE_OBSERVATION_WINDOW']
    ready = raw.get('ready', {})
    previous = -1
    for sample in samples:
        processes = {int(k): v for k, v in sample.get('processes_mib', {}).items()}
        pids = {r.get('pid') for r in ready.values()}
        if (not stamp(sample.get('time_ns')) or sample['time_ns'] < previous
                or not stamp(sample.get('query_duration_ns')) or not finite(sample.get('watts'), positive=True)
                or set(sample.get('owned_aliases', [])) != set(RESIDENTS) or sample.get('foreign_pids')
                or not pids <= set(processes) or any(not finite(processes.get(pid), positive=True) for pid in pids)
                or not finite(sample.get('owned_used_mib'), positive=True)
                or abs(sum(processes.get(pid, 0) for pid in pids)-sample['owned_used_mib']) > 1e-6):
            issues.append('INVALID_BOARD_OR_OWNED_MEMORY_SAMPLE')
            break
        previous = sample['time_ns']
    return issues


def validate_parsed(parsed, anchors, rows, pid):
    issues = []
    key = lambda a: (a.get('pid'), a.get('request_id'), a.get('kind'), a.get('segment_id'))
    expected, observed = Counter(map(key, anchors)), Counter(map(key, parsed.get('phases', [])))
    if expected != observed or not expected or any(v != 1 for v in observed.values()):
        issues.append('EXACT_CUPTI_ANCHOR_KEY_COVERAGE_FAILED')
    mapping = parsed.get('clock_mappings', {}).get(str(pid), {})
    if (set(parsed.get('clock_mappings', {})) != {str(pid)} or mapping.get('valid') is not True
            or not finite(mapping.get('max_boundary_residual_ns')) or mapping['max_boundary_residual_ns'] > 5_000_000):
        issues.append('PROCESS_CLOCK_MAPPING_FAILED')
    byrid = {r['request_id']: r for r in rows}
    counts, tails = Counter(), {}
    tolerance = max(0, mapping.get('max_boundary_residual_ns', 0)) + 100_000
    for phase in parsed.get('phases', []):
        row = byrid.get(phase.get('request_id'))
        if row is None or phase.get('pid') != pid or phase.get('clock_mapping_valid') is not True:
            issues.append('PHASE_OWNER_OR_CLOCK_INVALID')
            continue
        begin, end = row['worker_response']['start_perf_ns'], row['worker_response']['end_perf_ns']
        for kernel in phase.get('kernels', []):
            ks, ke = kernel.get('start_ns'), kernel.get('end_ns')
            if (kernel.get('pid') != pid or not stamp(ks) or not stamp(ke) or ks >= ke
                    or ks < begin-tolerance or ke > end+tolerance):
                issues.append('KERNEL_OWNER_OR_HANDLER_BOUNDARY_FAILED')
            else:
                counts[phase['request_id']] += 1
                tails[phase['request_id']] = max(tails.get(phase['request_id'], 0), ke-end)
    if any(counts[rid] <= 0 for rid in byrid):
        issues.append('REAL_KERNEL_REQUEST_COVERAGE_FAILED')
    return issues, dict(kernel_counts=dict(counts), max_kernel_past_handler_ns=tails,
                        mapping=mapping, handler_boundary_tolerance_ns=tolerance)


def load_cpu_adapter(binding, spec):
    cmd = binding['command']
    package_root = Path(cmd[cmd.index('--source-root')+1])/'gsdeploy'
    namespace = '_gsdeploy_smoke_v3_'+hashlib.sha256(str(package_root).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__, package.__package__ = [str(package_root)], namespace
        sys.modules[namespace] = package
    module = importlib.import_module(namespace+'.backends.compact')
    resources = spec['resources']
    return module.CompactAdapter(json.loads(Path(resources['protocol']).read_text()),
                                 json.loads(Path(resources['classmap']).read_text()))


def validate_output(row, item, spec, adapter=None):
    response, issues = row.get('worker_response', {}), []
    rid = row.get('request_id')
    if (not isinstance(response.get('raw_output'), str) or not response['raw_output'].strip()
            or not isinstance(response.get('prediction'), str)
            or response.get('sample_id') != item['sample'].get('id')):
        return ['TASK_OUTPUT_STRUCTURE_INVALID:'+str(rid)]
    if adapter is None:
        if response.get('finish_reason') != 'ctc_complete' or not response['prediction'].strip():
            issues.append('OCR_OUTPUT_PROTOCOL_INVALID:'+rid)
    else:
        candidate = spec.get('task_candidates', {}).get(row['task'], spec['candidate'])
        request = adapter.build(candidate, item['sample'])
        observation = {k: response[k] for k in ('finish_reason', 'generated_tokens', 'terminal_token_id', 'eos_token_ids') if k in response}
        checked = adapter.finish(request, item['sample'], response['raw_output'], evaluate_quality=False, **observation)
        for name in ('request_sha256', 'protocol_sha256', 'prediction', 'success', 'finish_reason', 'generated_tokens'):
            if checked.get(name) != response.get(name):
                issues.append('CPU_REPARSED_OUTPUT_MISMATCH:'+rid+':'+name)
        effective = response.get('effective_generation', {})
        for name, value in request['generation'].items():
            if effective.get(name) != value:
                issues.append('GENERATION_POLICY_CHANGED:'+rid+':'+name)
        if not finite(response.get('input_tokens'), positive=True):
            issues.append('INPUT_TOKEN_OBSERVATION_INVALID:'+rid)
    return issues


def review(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    issues, evidence, trace_reports = [], {}, {}
    try:
        plan = json.loads((run/'plan.json').read_text())
        context = json.loads((run/'context.json').read_text())
        raw = json.loads((run/'raw.json').read_text())
        executed = json.loads((run/'executed-jobs.json').read_text())
        for name in ('plan', 'context', 'executed_jobs'):
            actual = run/('executed-jobs.json' if name == 'executed_jobs' else name+'.json')
            if not check_receipt(raw.get(name), actual):
                issues.append('RAW_INPUT_RECEIPT_INVALID:'+name)
        for source in context['execution_sources']:
            if not check_receipt(source['frozen']) or receipt(source['original_path'])['sha256'] != source['frozen']['sha256']:
                issues.append('EXECUTION_SOURCE_DRIFT:'+source['original_path'])
        if context_identity(context) != plan.get('execution_context_sha256') or raw.get('execution_context_sha256') != plan.get('execution_context_sha256'):
            issues.append('EXECUTION_CONTEXT_SHA_MISMATCH')
        if context.get('residents') != RESIDENTS or context.get('execution', {}).get('physical_owners') != 4:
            issues.append('ORIGINAL_FOUR_OWNERS_REQUIRED')
        if raw.get('status') != 'completed' or raw.get('errors') or raw.get('runtime_errors') or raw.get('telemetry_errors'):
            issues.append('COLLECTION_ERRORS_PRESENT')
        ready = raw.get('ready', {})
        if set(ready) != set(RESIDENTS) or len({r.get('pid') for r in ready.values()}) != 4 or any(not stamp(r.get('pid')) for r in ready.values()):
            issues.append('FOUR_DISTINCT_READY_PIDS_REQUIRED')
        if set(raw.get('shutdown', {})) != set(RESIDENTS) or any(s.get('exit_code') != 0 or s.get('forced') is not False or s.get('bye_received') is not True for s in raw.get('shutdown', {}).values()):
            issues.append('OWNED_SHUTDOWN_FAILED')
        issues += validate_coverage(plan, raw, executed)
        issues += validate_telemetry(raw)
        pool_source = next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        pool = {r['pool_id']: r for r in map(json.loads, Path(pool_source).read_text().splitlines())}
        # Confirm measured and priming records both obey the original CPU adapter.
        for alias, family in RESIDENTS.items():
            binding = context['worker_bindings'][family]
            spec = json.loads(Path(binding['spec']).read_text())
            if ready.get(alias, {}).get('spec_sha256') != receipt(binding['spec'])['sha256']:
                issues.append('READY_SPEC_SHA_MISMATCH:'+alias)
            adapter = None if family == 'ocr' else load_cpu_adapter(binding, spec)
            for row in raw.get('priming_request_records', [])+raw.get('request_records', []):
                if row.get('model') == alias:
                    issues += validate_output(row, pool[row['pool_id']], spec, adapter)
            pid = ready[alias]['pid']
            rows = [r for r in raw['request_records'] if r['model'] == alias]
            anchors = []
            for row in rows:
                found, paired = validate_request(row, pid)
                issues += found; anchors += paired
            sql = run/('trace-'+alias.replace('#','-')+'.sqlite')
            parsed = parse_sqlite(sql, anchors)
            found, summary = validate_parsed(parsed, anchors, rows, pid)
            issues += [item+':'+alias for item in found]
            parsed_path = output/('parsed-'+alias.replace('#','-')+'.json')
            save(parsed_path, parsed)
            trace_reports[alias] = dict(sqlite=receipt(sql), nsys=receipt(sql.with_suffix('.nsys-rep')),
                                         parsed=receipt(parsed_path), summary=summary)
        evidence = dict(raw=receipt(run/'raw.json'), plan=receipt(run/'plan.json'), context=receipt(run/'context.json'),
                        executed_jobs=receipt(run/'executed-jobs.json'), execution_context_sha256=plan['execution_context_sha256'],
                        original_sources=context['execution_sources'])
    except Exception as exc:
        issues.append('REVIEW_INCOMPLETE:'+type(exc).__name__+':'+str(exc))
    result = dict(schema='gsdeploy.kernel_smoke_strict_review.v3', status='PASS' if not issues else 'FAIL',
                  issues=issues, evidence=evidence, traces=trace_reports,
                  evaluator_sources=[receipt(__file__), receipt(Path(__file__).with_name('trace_export.py'))],
                  qualification='RUNTIME_PROTOCOL_KERNEL_CLOCK_CORRECTNESS_ONLY',
                  task_quality='UNKNOWN_NO_GOLD_REFERENCES_OR_QUALITY_EVALUATION',
                  quality_note='Legal explicit No Answer is valid protocol output; this does not establish task accuracy.',
                  counters='UNKNOWN_ERR_NVGPUCTRPERM', permit_release='NOT_APPLICABLE_DEFAULT_SHARED_NO_PERMIT',
                  timing_note='Per-process constant offset only; residual bounds do not prove finer cross-process ordering.',
                  deployment_gain_qualified=False, validation_started=False)
    save(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(args.run, args.output)
    print(json.dumps(dict(status=result['status'], issues=result['issues'], report=receipt(Path(args.output)/'report.json'))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
