"""CPU audit of actual stream order, overlap, gaps and host-anchor boundaries.

Canonical kernels come from NVTX ranges plus unattributed kernels, never also
from the repeated phase view. GPU gaps are unclassified, not inferred CPU work.
Smoke may be diagnosed here but can never become training observations.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sqlite3

from gsdeploy.resources import identity
from .native_collect import receipt
from .trace_export import linux_pid

SCHEMA = 'gsdeploy.kernel_sequence_expressibility_audit.v2'
IDENTITY_FIELDS = ('nsys_start_ns', 'nsys_end_ns', 'pid', 'stream', 'correlation_id', 'device', 'context_id')


def verify(ref):
    if not isinstance(ref, dict) or receipt(ref['path']) != ref:
        raise ValueError('Changed/missing source receipt')


def stream_key(kernel):
    return kernel['pid'], kernel.get('device'), kernel.get('context_id'), kernel['stream']


def interval_statistics(intervals):
    """Exact union and simultaneous activity; no sum-as-wall-time shortcut."""
    events = collections.defaultdict(int)
    total = 0
    for start, end in intervals:
        if type(start) is not int or type(end) is not int or start >= end:
            raise ValueError('Positive integer kernel interval required')
        events[start] += 1; events[end] -= 1
        total += end-start
    active = busy = overlap_wall = peak = 0
    previous = None
    for stamp, delta in sorted(events.items()):
        if previous is not None:
            if active:
                busy += stamp-previous
            if active >= 2:
                overlap_wall += stamp-previous
        active += delta
        peak = max(peak, active)
        previous = stamp
    return dict(duration_sum_ns=total, union_busy_ns=busy, overlap_excess_ns=total-busy,
                simultaneous_overlap_wall_ns=overlap_wall, maximum_simultaneous_kernels=peak)


def distribution(values):
    values = sorted(values)
    def quantile(q):
        if not values:
            return None
        position = (len(values)-1)*q
        low = int(position); high = min(low+1, len(values)-1)
        return values[low]+(values[high]-values[low])*(position-low)
    return dict(count=len(values), sum_ns=sum(values), min_ns=min(values) if values else None,
                median_ns=quantile(.5), p95_ns=quantile(.95), max_ns=max(values) if values else None)


def analyze_sequence(kernels, row=None, *, runtime_launches=None, clock_mapping=None):
    """Inspect a request branch without requiring its use as training data."""
    kernels = sorted(kernels, key=lambda k: tuple(k.get(n, 0) or 0 for n in IDENTITY_FIELDS))
    by_stream = collections.defaultdict(list)
    for index, kernel in enumerate(kernels):
        by_stream[stream_key(kernel)].append((index, kernel))
    stats = interval_statistics([(k['nsys_start_ns'], k['nsys_end_ns']) for k in kernels])
    same_stream_overlaps = 0
    stream_sequences = []
    for key, members in sorted(by_stream.items(), key=lambda item: repr(item[0])):
        previous = None
        for _, kernel in members:
            if previous is not None and kernel['nsys_start_ns'] < previous:
                same_stream_overlaps += 1
            previous = max(previous or kernel['nsys_end_ns'], kernel['nsys_end_ns'])
        stream_sequences.append(dict(stream=list(key), kernel_ordinals=[index for index, _ in members],
                                     fifo_edge_count=max(0, len(members)-1)))
    gaps, launches, delays, launch_order_violations = [], [], [], 0
    previous_end = None
    previous_launch_start = None
    runtime_launches = runtime_launches or {}
    for kernel in kernels:
        if previous_end is not None:
            gaps.append(max(0, kernel['nsys_start_ns']-previous_end))
        previous_end = max(previous_end or kernel['nsys_end_ns'], kernel['nsys_end_ns'])
        launch = runtime_launches.get((kernel['pid'], kernel['correlation_id']))
        if launch is not None:
            launches.append(launch['start'])
            delays.append(kernel['nsys_start_ns']-launch['start'])
            if len(by_stream) == 1 and previous_launch_start is not None and launch['start'] < previous_launch_start:
                launch_order_violations += 1
            previous_launch_start = launch['start']
    result = dict(kernel_count=len(kernels), stream_count=len(by_stream), **stats,
        same_stream_overlap_events=same_stream_overlaps, launch_order_violations=launch_order_violations,
        single_serial_stream_expressible=bool(kernels) and len(by_stream) == 1 and stats['overlap_excess_ns'] == 0
                                          and not launch_order_violations,
        stream_dag=dict(stream_sequences=stream_sequences,
            proven_fifo_edge_count=sum(s['fifo_edge_count'] for s in stream_sequences),
            cross_stream_dependencies='UNKNOWN_NO_EVENT_OR_WAIT_EDGE_RECONSTRUCTION',
            timestamp_order_is_not_dependency_evidence=True),
        inter_kernel_unclassified_gaps=distribution(gaps), runtime_launch_count=len(launches),
        kernel_start_minus_runtime_launch_start_ns=distribution(delays),
        gpu_envelope_ns=max((k['nsys_end_ns'] for k in kernels), default=0)-min((k['nsys_start_ns'] for k in kernels), default=0),
        gap_classification='Unclassified non-kernel intervals; not CPU execution, free GPU capacity, or natural launch-delay evidence',
        natural_timing_validated=False, training_observations_created=False)
    if row is not None:
        response = row.get('worker_response', {})
        handler_start, handler_end = response.get('start_perf_ns'), response.get('end_perf_ns')
        valid_clock = bool(clock_mapping and clock_mapping.get('valid'))
        offset = clock_mapping.get('offset_ns') if valid_clock else None
        first = kernels[0]['nsys_start_ns']+offset if kernels and offset is not None else None
        last = max(k['nsys_end_ns'] for k in kernels)+offset if kernels and offset is not None else None
        result['host_boundaries'] = dict(handler_start_ns=handler_start, handler_end_ns=handler_end,
            handler_wall_ns=handler_end-handler_start if type(handler_start) is int and type(handler_end) is int else None,
            mapped_first_kernel_start_ns=first, mapped_last_kernel_end_ns=last,
            prefix_unclassified_ns=first-handler_start if first is not None and type(handler_start) is int else None,
            suffix_unclassified_ns=handler_end-last if last is not None and type(handler_end) is int else None,
            mapping=clock_mapping,
            anchors=[{k: s[k] for k in ('id', 'kind', 'start_ns', 'end_ns') if k in s}
                     for s in response.get('observed_stages', [])],
            warning='Host call anchors may overlap asynchronous GPU work. NVTX minus kernel time is not CPU time.')
    return result


def load_collection(raw_path, parsed_paths):
    """Return verified canonical per-request data plus diagnostic source receipt."""
    raw = json.loads(Path(raw_path).read_text())
    if raw.get('schema') != 'gsdeploy.native_phase_collection.v1' or raw.get('status') != 'completed':
        raise ValueError('Completed native phase collection required')
    if any(raw.get(key) for key in ('errors', 'runtime_errors', 'telemetry_errors')):
        raise ValueError('Failed collection cannot supply sequence evidence')
    verify(raw['plan']); verify(raw['context'])
    plan = json.loads(Path(raw['plan']['path']).read_text())
    context = json.loads(Path(raw['context']['path']).read_text())
    if raw['execution_context_sha256'] != plan['execution_context_sha256']:
        raise ValueError('Raw execution context differs from plan')
    requests = {r['request_id']: dict(row=r, kernels=[], source_traces=[], clock_mapping=None,
                                     runtime_launches={}, auxiliary_gpu_activities={}) for r in raw['request_records']}
    if len(requests) != len(raw['request_records']):
        raise ValueError('Duplicate request identity')
    sources, unassigned, seen = [receipt(raw_path), raw['plan'], raw['context'], receipt(__file__)], [], set()
    trace_details = []
    for parsed_path in parsed_paths:
        trace = json.loads(Path(parsed_path).read_text())
        sqlite_sources = [s for s in trace.get('sources', []) if Path(s['path']).suffix == '.sqlite']
        if trace.get('schema') != 'gsdeploy.nsys_observed_phases.v1' or len(sqlite_sources) != 1:
            raise ValueError('Canonical parsed Nsight trace required')
        for source in trace['sources']:
            verify(source)
            if source not in sqlite_sources and source not in (receipt(raw_path), raw['plan'], raw['context']):
                raise ValueError('Parsed trace has an unbound auxiliary source')
        sql_ref = sqlite_sources[0]
        sources.extend([receipt(parsed_path), *trace['sources']])
        sql = sqlite3.connect(Path(sql_ref['path']).as_uri()+'?mode=ro', uri=True)
        sql.row_factory = sqlite3.Row
        try:
            launches = {(linux_pid(r['globalTid']), r['correlationId']): dict(r)
                        for r in sql.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME') if r['correlationId'] is not None}
            tables = {r[0] for r in sql.execute("select name from sqlite_master where type='table'")}
            auxiliary = {name: sql.execute('select count(*) from '+name).fetchone()[0]
                         for name in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET') if name in tables}
        finally:
            sql.close()
        trace_details.append(dict(parsed=receipt(parsed_path), sqlite=sql_ref,
                                  trace_wide_auxiliary_gpu_activity_counts=auxiliary,
                                  auxiliary_request_attribution='UNKNOWN'))
        canonical = [(region.get('stage'), kernel) for region in trace.get('nvtx_ranges', []) for kernel in region.get('kernels', [])]
        canonical += [(None, kernel) for kernel in trace.get('unattributed_kernels', [])]
        for stage, kernel in canonical:
            key = (sql_ref['sha256'], *(kernel.get(k) for k in IDENTITY_FIELDS))
            if key in seen:
                raise ValueError('Duplicate canonical kernel (do not concatenate phases)')
            seen.add(key)
            rid = stage[0] if stage else None
            if rid not in requests:
                unassigned.append(dict(sqlite=sql_ref, kernel_identity=list(key[1:]), reason='NO_EXPLICIT_REQUEST_STAGE'))
                continue
            request = requests[rid]
            alias = request['row']['model']
            if kernel['pid'] != raw['ready'][alias]['pid']:
                raise ValueError('Request kernel belongs to a different READY process')
            request['kernels'].append(dict(kernel, source_sqlite=sql_ref))
            if sql_ref not in request['source_traces']:
                request['source_traces'].append(sql_ref)
                mapping = trace['clock_mappings'].get(str(kernel['pid']))
                if request['clock_mapping'] is not None and request['clock_mapping'] != mapping:
                    raise ValueError('One request spans incompatible profiler clock mappings')
                request['clock_mapping'] = mapping
            launch_key = kernel['pid'], kernel['correlation_id']
            if launch_key in launches:
                request['runtime_launches'][launch_key] = launches[launch_key]
    report = dict(schema=SCHEMA, raw=receipt(raw_path), source_receipts=sources,
        execution_context_sha256=raw['execution_context_sha256'],
        split_labels=sorted({r['row'].get('split', 'UNKNOWN') for r in requests.values()}),
        requests=[], unassigned_kernels=unassigned, trace_activity_inventory=trace_details,
        GPU_started=False, training_observations_created=False, natural_timing_validated=False)
    families = context.get('residents', {})
    for rid, request in requests.items():
        row = request['row']
        result = analyze_sequence(request['kernels'], row, runtime_launches=request['runtime_launches'],
                                  clock_mapping=request['clock_mapping'])
        result.update(request_id=rid, alias=row['model'], family=families.get(row['model']), task=row['task'],
                      pool_id=row['pool_id'], split=row.get('split'),
                      auxiliary_gpu_activity_attribution='UNKNOWN_TRACE_WIDE_COUNTS_ONLY')
        report['requests'].append(result)
    report['summary'] = dict(request_count=len(requests), canonical_kernel_count=len(seen),
        assigned_kernel_count=sum(len(r['kernels']) for r in requests.values()), unassigned_kernel_count=len(unassigned),
        single_serial_stream_requests=sum(r['single_serial_stream_expressible'] for r in report['requests']),
        overlapping_requests=sum(r['overlap_excess_ns'] > 0 for r in report['requests']),
        smoke_used_for_fitting=False)
    report['audit_sha256'] = identity(report)
    return raw, plan, context, requests, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--parsed', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = load_collection(args.raw, args.parsed)[-1]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report['summary']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
