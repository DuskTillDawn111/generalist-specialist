"""CPU-only audit of observed whole-request admission in native flow runs.

Recorded grant/receipt timestamps are *inside* the actual permit lifetime.
They cannot reconstruct acquire/release boundaries or the FIFO append order.
Passing this audit establishes the observed contract, not GPU isolation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

from gsdeploy.resources import identity, validate_resources


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'gsdeploy.native_flow_mechanism_audit.v1'
RESIDENTS = {'text#0': 'text', 'text#1': 'text', 'generalist_2#0': 'generalist_2', 'ocr#0': 'ocr'}
SEMANTIC_PATHS = ('gsdeploy/runtime/admission.py', 'gsdeploy/runtime/deployment.py', 'gsdeploy/resources.py')


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Changed or invalid source receipt')
    return Path(ref['path'])


def ns(row, key):
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer timestamp required: ' + key)
    return value


def summary(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'max': max(values) if values else None}


def peak(intervals):
    """Half-open concurrency; receipt ends a *recorded* interval at ties."""
    events = sorted((time, change) for start, end in intervals
                    for time, change in ((start, 1), (end, -1)))
    active = highest = 0
    for _, change in events:
        active += change
        highest = max(highest, active)
    return highest


def audit(raw_path):
    raw_path = Path(raw_path).resolve()
    raw = read(raw_path)
    if (raw.get('schema') != 'gsdeploy.native_flow_collection.v1' or raw.get('kind') != 'measured'
            or raw.get('backend') != 'native' or raw.get('status') != 'completed'
            or raw.get('collection_complete') is not True
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors', 'qualification_issues'))):
        raise ValueError('Complete error-free native flow collection required')
    if raw.get('native_raw'):
        native = read(checked(raw['native_raw']))
        if {k: v for k, v in raw.items() if k not in ('native_raw', 'quality_evidence')} != native:
            raise ValueError('Annotated raw differs from its native source')
    plan = read(checked(raw['plan']))
    if (plan.get('schema') != 'gsdeploy.native_flow_plan.v1'
            or identity({k: v for k, v in plan.items() if k != 'plan_sha256'}) != plan.get('plan_sha256')
            or identity(plan['trace']) != plan.get('trace_sha256')
            or raw.get('trace_sha256') != plan['trace_sha256']):
        raise ValueError('Native flow plan schema/hash/trace mismatch')
    candidates = [c for c in plan['candidates'] if c['id'] == raw.get('candidate_id')]
    if len(candidates) != 1:
        raise ValueError('One exact source-plan candidate required')
    candidate = candidates[0]
    aliases = {a: row['family'] for a, row in plan['aliases'].items()}
    if (aliases != RESIDENTS or set(raw.get('resident_aliases', [])) != set(aliases)
            or len(raw['resident_aliases']) != len(aliases) or raw.get('replica_to_family') != aliases):
        raise ValueError('Exact fixed native residency required')
    resources = validate_resources(candidate['resource_configuration'], aliases)
    if (resources != candidate['resource_configuration'] or raw.get('resource_configuration') != resources
            or resources['mode'] not in ('shared', 'request_admission')):
        raise ValueError('Exact normalized shared/admission source configuration required')
    admission = resources['mode'] == 'request_admission'
    if (raw.get('context') != plan['context']
            or raw.get('execution_context_sha256') != plan['execution_context_sha256']
            or candidate['execution_context_sha256'] != plan['execution_context_sha256']):
        raise ValueError('Execution identity or context reference mismatch')
    context = read(checked(plan['context']))
    sources = context.get('execution_sources', [])
    if not sources or context.get('sources') != [s['frozen'] for s in sources]:
        raise ValueError('Immutable execution source snapshots required')
    for source in sources:
        checked(source['frozen'])
    content = {k: v for k, v in context.items() if k not in ('sources', 'execution_sources')}
    content['source_content'] = [{'original_path': s['original_path'], 'sha256': s['frozen']['sha256']}
                                 for s in sources]
    context_hash = identity(content)
    if context_hash != plan['execution_context_sha256'] or context.get('residents') != aliases:
        raise ValueError('Frozen execution context identity mismatch')
    # The interpretation below is tied to the reviewed runtime implementation.
    # Old retained copies do not substitute for the actual project modules.
    semantic_sources = []
    for name in SEMANTIC_PATHS:
        path = ROOT / name
        selected = [s for s in sources if Path(s['original_path']).resolve() == path]
        if len(selected) != 1 or selected[0]['frozen']['sha256'] != receipt(path)['sha256']:
            raise ValueError('Unsupported or missing runtime admission semantics source: ' + name)
        semantic_sources.append(selected[0]['frozen'])
    runner = checked(plan['runner_source'])
    if runner.resolve() != Path(__file__).with_name('native_flow.py').resolve():
        raise ValueError('Reviewed native flow collector source required')
    if plan.get('profile_plan'):
        checked(plan['profile_plan'])
    ready = raw.get('ready', {})
    if set(ready) != set(aliases):
        raise ValueError('READY evidence for every resident required')
    pids = []
    for alias, family in aliases.items():
        info = ready[alias]
        if type(info.get('pid')) is not int or info['pid'] <= 0 or info.get('worker') != family:
            raise ValueError('READY PID/family ownership mismatch')
        pids.append(info['pid'])
        command = context['worker_bindings'][family]['command']
        if command.count('--spec') != 1:
            raise ValueError('Exact bound worker spec required')
        spec_path = Path(command[command.index('--spec') + 1])
        if info.get('spec_sha256') != receipt(spec_path)['sha256']:
            raise ValueError('READY worker spec hash mismatch')
        spec_sources = [s for s in sources if Path(s['original_path']).resolve() == spec_path.resolve()]
        if len(spec_sources) != 1 or spec_sources[0]['frozen']['sha256'] != info['spec_sha256']:
            raise ValueError('Worker spec is not bound by immutable execution sources')
    if len(set(pids)) != len(pids):
        raise ValueError('Resident workers must own distinct PIDs')
    startup = ns(raw, 'startup_begin_ns'), ns(raw, 'startup_end_ns')
    start, deadline, end = ns(raw, 'phase_start_ns'), ns(raw, 'deadline_ns'), ns(raw, 'observation_end_ns')
    if (not startup[0] <= startup[1] <= start < deadline <= ns(raw, 'arrival_loop_end_ns') <= end
            or deadline != start + round(plan['window_seconds'] * 1e9)):
        raise ValueError('Startup/window/drain timing mismatch')
    rows, seen, details = [], set(), []
    for phase, key in (('priming', 'priming_request_records'), ('warmup', 'warmup_request_records'), ('measurement', 'request_records')):
        if not isinstance(raw.get(key), list):
            raise ValueError('All request phases must be explicit lists')
        for row in raw[key]:
            rid, alias = row.get('request_id'), row.get('model')
            if (not isinstance(rid, str) or not rid or rid in seen or row.get('phase') != phase
                    or alias not in aliases or row.get('status') != 'completed'
                    or row.get('task') not in ready[alias].get('tasks', [])):
                raise ValueError('Request identity/status/phase/owner mismatch')
            seen.add(rid)
            response = row.get('worker_response', {})
            if (response.get('request_id') != rid or response.get('task_type') != row['task']
                    or response.get('task') != row['task'] or response.get('success') is not True):
                raise ValueError('Complete matching native RESULT required')
            scheduled, enqueued = ns(row, 'scheduled_time_ns'), ns(row, 'enqueued_time_ns')
            submit, worker_start = ns(row, 'actual_submit_time_ns'), ns(row, 'observed_worker_start_time_ns')
            worker_end, received = ns(response, 'end_perf_ns'), ns(row, 'completion_time_ns')
            if (worker_start != ns(response, 'start_perf_ns')
                    or not scheduled <= enqueued <= submit <= worker_start < worker_end <= received <= end):
                raise ValueError('Request submit/worker/parent receipt order violated')
            events = row.get('execution_events', [])
            starts = [e for e in events if 'event' not in e]
            if (len(starts) != 1 or starts[0].get('request_id') != rid
                    or starts[0].get('pid') != ready[alias]['pid'] or starts[0].get('perf_ns') != worker_start
                    or not worker_start <= ns(starts[0], 'received_time_ns') <= received):
                raise ValueError('Unique native START and READY PID binding required')
            if any(e.get('request_id') != rid for e in events):
                raise ValueError('Foreign execution event in request')
            if 'submit_write_completed_time_ns' in row and ns(row, 'submit_write_completed_time_ns') < submit:
                raise ValueError('Transport write completion predates submission')
            if phase != 'measurement' and received > start:
                raise ValueError('Priming/warmup not drained before measurement')
            if admission:
                wait, grant = ns(row, 'admission_wait_start_time_ns'), ns(row, 'admission_granted_time_ns')
                if not enqueued <= wait <= grant <= submit:
                    raise ValueError('Admission wait/grant/submit order violated')
            else:
                if any(k.startswith('admission_') for k in row):
                    raise ValueError('Shared requests must not fabricate admission permits')
                wait = grant = None
            details.append({'request_id': rid, 'alias': alias, 'phase': phase,
                            'wait_start_ns': wait, 'grant_ns': grant, 'submit_ns': submit,
                            'worker_start_ns': worker_start, 'worker_end_ns': worker_end, 'parent_receipt_ns': received,
                            'enqueued_ns': enqueued})
            rows.append(row)
    offered = {r['request_id']: r for r in plan['trace']}
    actual = {r['request_id']: r for r in raw['request_records']}
    if (len(offered) != len(plan['trace']) or set(offered) != set(actual)
            or set(candidate['routes']) != set(offered)):
        raise ValueError('Complete identical offered cohort required')
    for rid, declared in offered.items():
        row = actual[rid]
        if (row['task'] != declared['task'] or row['pool_id'] != declared['pool_id']
                or row['model'] != candidate['routes'][rid]
                or row['scheduled_time_ns'] != start + declared['offset_ns']):
            raise ValueError('Actual request differs from frozen arrival/input/route')
    shutdown = raw.get('shutdown', {})
    if (set(shutdown) != set(aliases) or any(s.get('forced') is not False or s.get('exit_code') != 0
                                            or s.get('bye_received') is not True for s in shutdown.values())):
        raise ValueError('Complete clean backend shutdown required')
    samples = [s for s in raw.get('board_samples', []) if start <= ns(s, 'time_ns') <= end]
    if not samples:
        raise ValueError('Owned board telemetry during measurement/drain required')
    expected_pids = set(map(str, pids))
    for sample in samples:
        if (set(sample.get('owned_aliases', [])) != set(aliases) or sample.get('foreign_pids')
                or set(sample.get('processes_mib', {})) != expected_pids):
            raise ValueError('Measured resident telemetry ownership mismatch')
    group_peaks, alias_peaks = {}, {}
    observed_peak = 0
    inversions = 0
    if admission:
        intervals = [(r['grant_ns'], r['parent_receipt_ns']) for r in details]
        observed_peak = peak(intervals)
        for alias, allocation in resources['allocations'].items():
            value = peak([(r['grant_ns'], r['parent_receipt_ns']) for r in details if r['alias'] == alias])
            alias_peaks[alias] = {'observed_peak': value, 'max_inflight': allocation['max_inflight']}
            if value > allocation['max_inflight']:
                raise ValueError('Observed per-alias max_inflight exceeded: ' + alias)
        for name, group in resources['groups'].items():
            value = peak([(r['grant_ns'], r['parent_receipt_ns']) for r in details if r['alias'] in group['aliases']])
            group_peaks[name] = {'observed_peak': value, 'max_active': group['max_active']}
            if value > group['max_active']:
                raise ValueError('Observed group max_active exceeded: ' + name)
        snapshot = raw.get('admission_snapshot_after')
        upper = sum(v['max_inflight'] for v in resources['allocations'].values())
        for group in resources['groups'].values():
            if set(group['aliases']) == set(aliases):
                upper = min(upper, group['max_active'])
        if (not isinstance(snapshot, dict) or snapshot.get('closed') is not True
                or snapshot.get('active') != {} or snapshot.get('waiting') != []
                or type(snapshot.get('peak_active')) is not int
                or not observed_peak <= snapshot['peak_active'] <= upper):
            raise ValueError('Admission not fully drained/closed or inconsistent peak snapshot')
        ordered = sorted(details, key=lambda r: (r['grant_ns'], r['request_id']))
        inversions = sum(a['wait_start_ns'] > b['wait_start_ns'] for i, a in enumerate(ordered) for b in ordered[i+1:])
    elif raw.get('admission_snapshot_after') is not None:
        raise ValueError('Shared mode must not claim an admission snapshot')
    for alias in aliases:
        if peak([(r['worker_start_ns'], r['worker_end_ns']) for r in details if r['alias'] == alias]) > 1:
            raise ValueError('One owned native worker has overlapping handler executions: ' + alias)
    return {'schema': SCHEMA, 'status': 'PASS_OBSERVED_CONTRACT', 'GPU_started': False,
            'candidate_id': candidate['id'], 'mode': resources['mode'], 'raw': receipt(raw_path),
            'plan': raw['plan'], 'context': raw['context'], 'runtime_semantics_sources': semantic_sources,
            'audit_source': receipt(__file__),
            'resource_configuration_sha256': identity(resources),
            'mechanism': 'whole_request_host_admission' if admission else 'shared_no_permits',
            'requests_by_phase': dict(collections.Counter(r['phase'] for r in details)),
            'permit_intervals': len(details) if admission else 0,
            'observed_permit_peak': observed_peak if admission else None,
            'group_peaks': group_peaks, 'alias_peaks': alias_peaks,
            'admission_wait_ms': summary([(r['grant_ns']-r['wait_start_ns'])/1e6 for r in details]) if admission else None,
            'grant_to_submit_ms': summary([(r['submit_ns']-r['grant_ns'])/1e6 for r in details]) if admission else None,
            'worker_end_to_parent_receipt_ms': summary([(r['parent_receipt_ns']-r['worker_end_ns'])/1e6 for r in details]),
            'fifo': {'status': 'NOT_IDENTIFIABLE' if admission else 'NOT_APPLICABLE',
                     'observed_wait_start_order_inversions': inversions if admission else None,
                     'interpretation': 'wait_start precedes acquire/append; grant is recorded after acquire. Inversions do not prove FIFO violations.'},
            'ownership': {'distinct_ready_pids': len(pids), 'start_pid_matches_ready': len(details),
                          'owned_telemetry_samples': len(samples), 'foreign_processes': 0,
                          'os_ancestry': 'Not reconstructed after shutdown; collector source checks descendant ownership at READY'},
            'drain': {'all_phases_completed': True, 'backend_shutdown_clean': True,
                      'measurement_drain_ms': (end-deadline)/1e6},
            'observation_limits': [
                'Recorded [grant,parent_receipt) lies inside actual [acquire,release); peak is a lower bound on permit concurrency.',
                'Acquire-to-grant and parent-receipt-to-release gaps are not timestamped; no finite observation-error bound is available.',
                'Passing observed caps plus source/snapshot consistency does not independently prove every instant of permit capacity.',
                'Whole-request host admission does not partition SMs, DRAM, L2, GPU kernels, or memory allocation.',
                'No batch, shared-weight/cache backend or MPS control is enabled or verified by this audit.'],
            'requests': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = audit(args.raw)
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'candidate_id', 'mode', 'permit_intervals', 'GPU_started')}))


if __name__ == '__main__':
    main()
