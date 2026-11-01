"""Joint host-prefix observations, with profiler domains kept separate.

A scenario is one complete job in one actual training capture. These are
observations for fitting and testing a host-supply hypothesis, not exogenous
CUDA releases for a counterfactual execution. In particular, synchronization
and API-return times can already contain GPU waiting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from gsdeploy.forecast_errors import UnknownEvidence
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest
from .trace_export import linux_pid
from .unified_corun_review_v8_1 import load_review, LIGHT, NATURAL

SCHEMA = 'gsdeploy.joint_training_host_prefix_observations.v13'


def public_workload(job):
    owners = set(job['inputs'])
    if (not owners or set(job['tasks']) != owners or set(job['start_offsets_ms']) != owners
            or job.get('split') != 'training'):
        raise ValueError('Complete declared training job required')
    rows = []
    for owner in sorted(owners):
        offset = job['start_offsets_ms'][owner] * 1_000_000
        if offset < 0 or offset != round(offset):
            raise ValueError('Nonnegative integer public arrival nanoseconds required')
        rows.append(dict(alias=owner, task=job['tasks'][owner], pool_id=job['inputs'][owner],
                         arrival_ns=round(offset)))
    return rows


def complete_members(job, records):
    workload = public_workload(job)
    members = [r for r in records if r['profile_job_id'] == job['id']]
    if len(members) != len(workload) or len({r['model'] for r in members}) != len(members):
        raise ValueError('Complete joint job cannot omit or duplicate an owner')
    expected = {(r['alias'], r['task'], r['pool_id']) for r in workload}
    if {(r['model'], r['task'], r['pool_id']) for r in members} != expected:
        raise ValueError('Actual joint job differs from its public workload')
    origins = {r['scheduled_time_ns'] - next(w['arrival_ns'] for w in workload
               if w['alias'] == r['model']) for r in members}
    if len(origins) != 1:
        raise ValueError('One common scheduled job origin required')
    if any(r.get('split') != 'training' or r.get('status') != 'completed' for r in members):
        raise ValueError('Only completed training members accepted')
    return workload, sorted(members, key=lambda r: r['model']), origins.pop()


def natural_prefix(row, origin):
    worker = row['worker_response']
    handler = worker['start_perf_ns']
    semantic = worker.get('semantic_forward_observations', {}).get('records', [])
    first = min(semantic, key=lambda p: p['start_ns']) if semantic else None
    pre_sync = [p for p in worker.get('observed_stages', []) if p['id'] == 'pre_sync']
    if len(pre_sync) > 1:
        raise ValueError('Ambiguous initial synchronization stage')
    sync = pre_sync[0] if pre_sync else None
    if sync is not None and not handler <= sync['start_ns'] <= sync['end_ns']:
        raise ValueError('Ordered initial host synchronization boundaries required')
    return dict(alias=row['model'], request_id=row['request_id'],
        scheduled_offset_ns=row['scheduled_time_ns']-origin,
        submit_offset_ns=row['actual_submit_time_ns']-origin,
        handler_start_offset_ns=handler-origin,
        pre_sync_start_offset_ns=None if sync is None else sync['start_ns']-origin,
        pre_sync_end_offset_ns=None if sync is None else sync['end_ns']-origin,
        pre_sync_missing_reason='NO_EXPLICIT_PRE_SYNC_STAGE' if sync is None else None,
        first_semantic_start_offset_ns=None if first is None else first['start_ns']-origin,
        first_semantic_phase=None if first is None else first['semantic_phase'],
        first_semantic_missing_reason='NO_LANGUAGE_SEMANTIC_BOUNDARY' if first is None else None,
        synchronization_wait_removed=False,
        first_semantic_is_first_CUDA_release=False)


def kernel_prefix(apis, kernels):
    """Distinguish first API submission from first GPU activity across streams."""
    by_correlation = defaultdict(list)
    for a in apis:
        by_correlation[a['correlationId']].append(a)
    mapped = []
    for kernel in kernels:
        candidates = by_correlation[kernel[0]]
        if len(candidates) != 1:
            raise ValueError('Every kernel needs one independently owned runtime API')
        mapped.append((candidates[0], kernel))
    if not mapped:
        raise ValueError('Actual first kernel API must be observed')
    first_launch, launched = min(mapped, key=lambda pair: (pair[0]['start'],pair[0]['end'],pair[1]))
    first_activity_api, activity = min(mapped, key=lambda pair: (pair[1][1],pair[1][2],pair[1][0]))
    return dict(first_kernel_launch_API_entry_ns=first_launch['start'],
        first_kernel_launch_API_return_ns=first_launch['end'],
        first_kernel_launch_activity_start_ns=launched[1],
        first_kernel_activity_start_ns=activity[1],
        first_kernel_activity_API_entry_ns=first_activity_api['start'],
        first_kernel_activity_API_return_ns=first_activity_api['end'],
        launch_and_activity_first_are_same_kernel=launched == activity), first_launch


def light_prefixes(report, raw):
    sql = report['trace']['sqlite']; parsed_ref = report['trace']['parsed']
    if receipt(sql['path']) != sql:
        raise ValueError('Changed native host-prefix SQLite')
    parsed = checked(parsed_ref)
    phases = defaultdict(list)
    for phase in parsed['phases']:
        phases[phase['request_id']].append(phase)
    by_pid = defaultdict(list)
    with sqlite3.connect(Path(sql['path']).as_uri()+'?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        for a in conn.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id'):
            by_pid[linux_pid(a['globalTid'])].append(dict(a))
    for rows in by_pid.values():
        rows.sort(key=lambda a: (a['start'], a['end'], a['correlationId']))
    result = {}
    for row in raw['request_records']:
        ranges = phases[row['request_id']]
        pid = raw['ready'][row['model']]['pid']
        if not ranges or any(p['pid'] != pid for p in ranges):
            raise ValueError('Complete independently owned common-clock request phases required')
        inside = lambda stamp: any(p['nsys_start_ns'] <= stamp <= p['nsys_end_ns'] for p in ranges)
        apis = [a for a in by_pid[pid] if inside(a['start'])]
        kernels = {(k['correlation_id'], k['nsys_start_ns'], k['nsys_end_ns'])
                   for p in ranges for k in p['kernels']}
        timing, first = kernel_prefix(apis,kernels)
        prefix = [a for a in apis if a['start'] <= first['start']]
        roles = {}
        serialized = []
        for a in prefix:
            roles.setdefault(a['globalTid'], len(roles))
            serialized.append(dict(name=a['api_name'], thread_role=roles[a['globalTid']],
                start_ns=a['start'], end_ns=a['end'], correlation_id=a['correlationId']))
        result[row['request_id']] = dict(alias=row['model'], request_id=row['request_id'],
            **timing,
            runtime_prefix=serialized, timestamp_domain='single_native_Nsight_session_ns',
            API_wall_contains_possible_GPU_wait=True,
            observed_prefix_is_counterfactual_release=False)
    return result


def derive(review_paths):
    scenarios = []; sources = [receipt(__file__)]; counts = Counter(); seen = set()
    for path in review_paths:
        ref = receipt(path); doc = checked(ref)
        domain = checked(doc['evidence']['plan.json'])['domain']
        if domain not in (LIGHT, NATURAL):
            raise ValueError('Explicit original light or natural training domain required')
        report, plan, raw = load_review(path, domain)
        raw_ref = report['evidence']['raw.json']
        if raw_ref['sha256'] in seen:
            raise ValueError('One capture cannot be counted twice')
        seen.add(raw_ref['sha256'])
        sources.extend([ref, *report['evidence'].values()])
        light = light_prefixes(report, raw) if domain == LIGHT else None
        if light is not None:
            sources.extend((report['trace']['sqlite'], report['trace']['parsed']))
        covered = []
        for job in plan['jobs']:
            workload, members, origin = complete_members(job, raw['request_records'])
            covered.extend(r['request_id'] for r in members)
            prefixes = ([natural_prefix(r, origin) for r in members] if light is None
                        else [light[r['request_id']] for r in members])
            # One translation for the WHOLE light scenario; never one per-owner
            # median clock map, which would destroy cross-owner intervals.
            if light is not None:
                base = min(a['start_ns'] for r in prefixes for a in r['runtime_prefix'])
                for r in prefixes:
                    for name in ('first_kernel_launch_API_entry_ns','first_kernel_launch_API_return_ns',
                                 'first_kernel_launch_activity_start_ns','first_kernel_activity_start_ns',
                                 'first_kernel_activity_API_entry_ns','first_kernel_activity_API_return_ns'):
                        r[name] -= base
                    for a in r['runtime_prefix']:
                        a['start_ns'] -= base; a['end_ns'] -= base
                event_field = 'first_kernel_launch_API_entry_ns'
            else:
                event_field = 'first_semantic_start_offset_ns'
            order = sorted((r[event_field], r['alias']) for r in prefixes if r[event_field] is not None)
            scenarios.append(dict(domain=domain, source_raw=raw_ref, source_review=ref,
                execution_context_sha256=raw['execution_context_sha256'], job_id=job['id'], job_name=job['name'],
                public_workload=workload, workload_sha256=digest(workload), prefixes=prefixes,
                joint_event_field=event_field, joint_event_order=[p for t,p in order],
                joint_event_gap_from_first_ns={p:t-order[0][0] for t,p in order},
                prefix_clock_origin=dict(domain='native_Nsight_session',ns=base,
                    definition='minimum runtime API entry in whole job') if light is not None
                    else dict(domain='host_perf_counter',ns=origin,definition='declared job scheduled origin'),
                public_arrival_and_light_offsets_share_origin=False if light is not None else True,
                empirical_scenario_probability=None, counterfactual_host_supply_identified=False))
            counts[domain+'_jobs'] += 1; counts[domain+'_requests'] += len(members)
        if (len(covered) != len(set(covered)) or set(covered) != {r['request_id'] for r in raw['request_records']}):
            raise ValueError('Full declared source cohort required')
    sources = list({r['path']:r for r in sources}.values())
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Joint prefix training source changed during extraction')
    return dict(schema=SCHEMA, scenarios=scenarios, sources=sources, counts=dict(counts),
        training_only=True, GPU_started=False, independent_prediction_validation=False,
        whole_job_correlation_preserved=True, per_owner_mean_composition_used=False,
        cross_domain_pairing_identified=False, cross_domain_joint_probabilities_assigned=False,
        synchronization_wait_is_exogenous_supply=False, complete_host_supply_provider=False)


def select_scenarios(bank, domain, workload):
    if bank.get('schema') != SCHEMA or bank.get('training_only') is not True:
        raise ValueError('Explicit training joint-prefix bank required')
    key = digest(workload)
    selected = [r for r in bank['scenarios'] if r['domain'] == domain and r['workload_sha256'] == key
                and r['public_workload'] == workload]
    if not selected:
        raise UnknownEvidence('JOINT_HOST_WORKLOAD_OR_DOMAIN_UNSUPPORTED')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs='+', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    bank = derive(args.reviews); save(out/'observations.json', bank)
    save(out/'report.json', dict(schema=SCHEMA, observations=receipt(out/'observations.json'),
        counts=bank['counts'], complete_host_supply_provider=False, training_only=True, GPU_started=False))
    print(json.dumps(dict(counts=bank['counts'], observations=receipt(out/'observations.json'))))


if __name__ == '__main__':
    main()
