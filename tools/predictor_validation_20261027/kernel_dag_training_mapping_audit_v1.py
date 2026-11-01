"""Read-only full-denominator canonical kernel/DAG audit; optional CPU replay.

This creates evidence about existing isolated training captures. It does not
collect GPU data, fit a response law, or qualify replay timing for deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from .cuda_dependency_dag_v2 import audit_sqlite
from .kernel_domain_bridge_review_v3 import load_report
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS, load_collection, verify
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .ordered_occupancy_resolver_v2 import AuditedOccupancyResolver
from .request_cuda_dag_v2 import request_plan, topological


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require_denominator(expected, observed):
    if len(observed) != len(set(observed)) or set(expected) != set(observed):
        raise ValueError('FULL_OFFERED_REQUEST_DENOMINATOR_REQUIRED')


def validate_mapping(request_id, request, profile, ready_pid):
    """Require a bijection with full device/context/stream/source identities."""
    row = request['row']
    if any(profile.get(k) != v for k, v in dict(request_id=request_id,
            alias=row['model'], task=row['task'], pool_id=row['pool_id'], split='training').items()):
        raise ValueError('REQUEST_OWNER_PROFILE_BINDING_MISMATCH')
    canonical = request['kernels']
    gpu = [n for n in profile['nodes'] if n['kind'] == 'gpu']
    key = lambda k: (k['source_sqlite']['sha256'], *(k.get(f) for f in IDENTITY_FIELDS))
    expected, observed = [key(k) for k in canonical], [key(n['kernel']) for n in gpu]
    if (len(expected) != len(set(expected)) or len(observed) != len(set(observed))
            or Counter(expected) != Counter(observed) or profile['kernel_count'] != len(expected)):
        raise ValueError('EXACT_CANONICAL_KERNEL_BIJECTION_REQUIRED')
    for node in gpu:
        k = node['kernel']
        if (k['pid'] != ready_pid or node['stream_key'] !=
                [k['pid'], k['device'], k['context_id'], k['stream']]
                or (node['correlation_id'], node['observed_start_ns'], node['observed_end_ns']) !=
                (k['correlation_id'], k['nsys_start_ns'], k['nsys_end_ns'])
                or k['source_sqlite'] != profile['source_sqlite']):
            raise ValueError('CUDA_DEVICE_CONTEXT_OWNER_STREAM_BOUNDARY_MISMATCH')
    ids = [n['id'] for n in profile['nodes']]
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_DAG_NODE')
    if profile['unresolved'] or profile['cyclic'] or not topological(profile['nodes'], profile['edges']):
        raise ValueError('REQUEST_DAG_UNRESOLVED_OR_CYCLIC')
    return dict(canonical_kernel_count=len(canonical), mapped_gpu_node_count=len(gpu),
                kernel_identity_sha256=fingerprint(sorted(expected)),
                kernel_stream_keys=sorted({tuple(n['stream_key']) for n in gpu}),
                activity_stream_keys=sorted({tuple(n['stream_key']) for n in profile['nodes'] if 'stream_key' in n}),
                canonical_bijection=True)


def replay_catalog(profiles):
    aliases = {}
    plans, requests, counts = {}, [], Counter()
    for p in profiles:
        alias = aliases.setdefault(p['alias'], dict(family=p['family'], supported_tasks=[]))
        if alias['family'] != p['family']:
            raise ValueError('REPLAY_OWNER_FAMILY_MISMATCH')
        if p['task'] not in alias['supported_tasks']:
            alias['supported_tasks'].append(p['task'])
        key = p['alias'], p['task'], p['pool_id']
        if key in plans:
            raise ValueError('REPLAY_PLAN_KEY_COLLISION')
        plans[key] = p
        requests.append(dict(request_id='known-dag:' + p['request_id'], alias=p['alias'],
                             task=p['task'], pool_id=p['pool_id'], arrival_ns=counts[p['alias']]*50_000_000))
        counts[p['alias']] += 1
    return aliases, plans, requests


def replay_v9(profiles):
    from .cuda_dag_simulator_v9 import DagFullHorizonModel
    aliases, plans, requests = replay_catalog(profiles)
    model = DagFullHorizonModel(aliases, plans, allow_profiled_replay=True,
        allow_independent_copy_hypothesis=True, allow_context_time_slice_hypothesis=True,
        allow_no_implicit_default_stream_hypothesis=True)
    forecast = model.forecast(requests, 0, 10**10)
    require_denominator([r['request_id'] for r in requests], list(forecast['cohort']))
    require_denominator([r['request_id'] for r in requests], list(forecast['finishes']))
    expected_nodes = sum(len(p['nodes']) for p in profiles)
    if forecast['completed_dag_nodes'] != expected_nodes:
        raise ValueError('FULL_DAG_NODE_REPLAY_DENOMINATOR_REQUIRED')
    return dict(status='PASS_PROFILED_REPLAY_ONLY', offered_requests=requests,
        finishes=forecast['finishes'], events=forecast['events'], expected_dag_nodes=expected_nodes,
        completed_dag_nodes=forecast['completed_dag_nodes'],
        max_simultaneous_nodes_per_owner=forecast['max_simultaneous_nodes_per_owner'],
        assumptions=forecast['assumptions'], duration_source=forecast['duration_source'],
        release_source=forecast['release_source'], energy_known=forecast['energy_known'],
        memory_known=forecast['memory_known'], request_start='arrival/owner-ready diagnostic boundary',
        lifecycle='instant handoff diagnostic', physical_accuracy_validated=False,
        dynamic_competition_response_connected=False, deployment_qualified=False)


def authenticated_sources(*documents):
    sources = strict_references(documents)
    for source in sources:
        verify(source)
    return sources


def static_coverage(profiles, audit_path, sources):
    audit_ref = receipt(audit_path)
    declaration = json.loads(Path(audit_path).read_text())
    start_sources = authenticated_sources(sources, audit_ref, declaration)
    resolver = AuditedOccupancyResolver(audit_path)
    rows = []
    for profile in profiles:
        counts = Counter()
        for node in profile['nodes']:
            if node['kind'] != 'gpu':
                continue
            k = node['kernel']
            key = (k['source_sqlite']['sha256'], k['nsys_start_ns'], k['nsys_end_ns'],
                   k['pid'], k['context_id'], k['correlation_id'])
            if key not in resolver.index:
                counts['absent'] += 1
            elif resolver.index[key] is None:
                counts['unresolved_static_allocation'] += 1
            else:
                counts['known_exact_static_allocation'] += 1
        rows.append(dict(request_id=profile['request_id'], alias=profile['alias'],
                         kernel_count=profile['kernel_count'], coverage=dict(counts)))
    authenticated_sources(start_sources, resolver.source_receipts)
    return dict(schema='gsdeploy.kernel_dag_static_coverage.v1',
        status=('PASS_EXACT_STATIC_COVERAGE' if all(r['coverage'].get('known_exact_static_allocation', 0)
            == r['kernel_count'] for r in rows) else 'INCOMPLETE_STATIC_COVERAGE'),
        sources=start_sources, occupancy_audit=audit_ref, hardware=resolver.hardware, requests=rows,
        source_receipts_unchanged_after_audit=True, GPU_started=False, fit_started=False,
        physically_validated=False, deployment_qualified=False)


def audit(review_path, profiles_path, output, *, run_replay=False, occupancy_audit_path=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='gsdeploy.kernel_dag_training_mapping_audit.v1', status='INCOMPLETE',
        offered_request_ids=[], requests=[], traces=[], issues=[], GPU_started=False,
        fit_started=False, training_observations_created=False, physical_accuracy_validated=False,
        deployment_qualified=False, replay={'status': 'NOT_REQUESTED'})
    save = lambda: (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    implementation_start = implementation_sources(__file__)
    sources = authenticated_sources(implementation_start, receipt(review_path), receipt(profiles_path))
    result['sources'] = sources
    try:
        strict, plan, raw = load_report(review_path, 'light-kernel')
        result['offered_request_ids'] = [r['request_id'] for r in raw['request_records']]
        if (any(len(j['inputs']) != 1 or j.get('split') != 'training' for j in plan['jobs'])
                or any(r.get('split') != 'training' for r in raw['request_records'])):
            raise ValueError('ISOLATED_EXPLICIT_TRAINING_REQUIRED')
        parsed = [o['parsed']['path'] for o in strict['owners'].values()]
        raw, _, context, requests, canonical_audit = load_collection(strict['raw']['path'], parsed)
        result['canonical_collection_summary'] = canonical_audit['summary']
        result['execution_context_sha256'] = raw['execution_context_sha256']
        result['raw'] = strict['raw']
        document = json.loads(Path(profiles_path).read_text())
        sources = authenticated_sources(sources, strict, plan, raw, context, document,
                                        canonical_audit['source_receipts'])
        if occupancy_audit_path is not None:
            sources = authenticated_sources(sources, receipt(occupancy_audit_path),
                json.loads(Path(occupancy_audit_path).read_text()))
        result['sources'] = sources
        result['implementation_sources'] = implementation_start
        if (document.get('schema') != 'gsdeploy.request_cuda_dag_profiles.v2'
                or document.get('training_only') is not True
                or document.get('execution_context_sha256') != raw['execution_context_sha256']
                or document.get('denominator') != len(requests)):
            raise ValueError('SAVED_DAG_TRAINING_CONTEXT_DENOMINATOR_MISMATCH')
        for source in document['sources']:
            verify(source)
        require_denominator(requests, [p['request_id'] for p in document['profiles']])
        saved = {p['request_id']: p for p in document['profiles']}
        for alias, owner in strict['owners'].items():
            trace = json.loads(Path(owner['parsed']['path']).read_text())
            sql = owner['sqlite']['path']
            fresh = audit_sqlite(sql)
            with sqlite3.connect(Path(sql).resolve().as_uri()+'?mode=ro', uri=True) as connection:
                tables = [r[0] for r in connection.execute("select name from sqlite_master where type='table'")]
                inventory = {t: connection.execute('select count(*) from "'+t+'"').fetchone()[0]
                             for t in tables if t.startswith('CUPTI_ACTIVITY_KIND_')}
            trace_report = dict(alias=alias, sqlite=owner['sqlite'], parsed=owner['parsed'],
                cup_activity_inventory=inventory, missing_runtime=fresh['missing_runtime'],
                unresolved=fresh['dag']['unresolved'], cyclic=fresh['dag']['cyclic'],
                event_record_count=fresh['event_record_count'], event_wait_count=fresh['event_wait_count'],
                sync_api_inventory=fresh['sync_api_inventory'], remaining_obligations=fresh['remaining_obligations'])
            result['traces'].append(trace_report)
            if fresh['missing_runtime'] or fresh['dag']['unresolved'] or fresh['dag']['cyclic']:
                result['issues'].append('WHOLE_TRACE_DEPENDENCY_INCOMPLETE:'+alias)
            for rid in owner['request_ids']:
                request = requests[rid]
                try:
                    profile = request_plan(rid, request, trace, fresh, sql)
                    profile['family'] = context['residents'][alias]
                    details = validate_mapping(rid, request, profile, raw['ready'][alias]['pid'])
                    validate_mapping(rid, request, saved[rid], raw['ready'][alias]['pid'])
                    if profile != saved[rid]:
                        raise ValueError('FRESH_DAG_DIFFERS_FROM_SOURCE_BOUND_SAVED_PROFILE')
                    result['requests'].append(dict(request_id=rid, alias=alias, pool_id=profile['pool_id'],
                        status='PASS_EXACT_MAPPING', **details, node_count=len(profile['nodes']),
                        edge_count=len(profile['edges']), activity_counts=profile['activity_counts'],
                        edge_counts=dict(Counter(e['kind'] for e in profile['edges'])),
                        profile_sha256=fingerprint(profile), unresolved=profile['unresolved'], cyclic=profile['cyclic'],
                        pre_request_external_dependencies_assumed_completed=profile['pre_request_external_dependencies_assumed_completed'],
                        default_stream_semantics=profile['default_stream_semantics']))
                except Exception as exc:
                    result['requests'].append(dict(request_id=rid, alias=alias, status='FAIL', error=str(exc)))
                    result['issues'].append('REQUEST_MAPPING_FAILED:'+rid)
            save()
            print(json.dumps(dict(trace=alias, requests_audited=len(result['requests']),
                missing_runtime=len(fresh['missing_runtime']), unresolved=len(fresh['dag']['unresolved']))), flush=True)
        require_denominator(requests, [r['request_id'] for r in result['requests']])
        result['mapping_status'] = 'PASS' if not result['issues'] else 'FAIL'
        save()
        if run_replay and not result['issues']:
            try:
                result['replay'] = replay_v9(document['profiles'])
            except Exception as exc:
                result['replay'] = dict(status='FAIL_FULL_COHORT_RETAINED', error=str(exc),
                    offered_request_ids=list(requests))
                result['issues'].append('V9_FULL_COHORT_REPLAY_FAILED')
        if occupancy_audit_path is not None:
            coverage = static_coverage(document['profiles'], occupancy_audit_path, sources)
            with (output/'static-coverage.json').open('x') as stream:
                json.dump(coverage, stream, indent=2, allow_nan=False)
                stream.write('\n')
            result['static_coverage'] = receipt(output/'static-coverage.json')
            if coverage['status'] != 'PASS_EXACT_STATIC_COVERAGE':
                result['issues'].append('INCOMPLETE_STATIC_COVERAGE')
        authenticated_sources(sources)
        if implementation_sources(__file__) != implementation_start:
            raise ValueError('AUDIT_TRANSITIVE_IMPLEMENTATION_CLOSURE_CHANGED')
        load_report(review_path, 'light-kernel')
        result['source_receipts_unchanged_after_audit'] = True
        result['status'] = 'PASS_CPU_MAPPING_ONLY' if not result['issues'] else 'FAIL'
    except Exception as exc:
        result['issues'].append(type(exc).__name__+':'+str(exc))
        result['status'] = 'FAIL'
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True)
    parser.add_argument('--profiles', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay-v9', action='store_true')
    parser.add_argument('--occupancy-audit')
    args = parser.parse_args()
    result = audit(args.review, args.profiles, args.output, run_replay=args.replay_v9,
                   occupancy_audit_path=args.occupancy_audit)
    print(json.dumps(dict(status=result['status'], requests=len(result['requests']),
                         replay=result['replay']['status'], issues=result['issues'])))
    raise SystemExit(0 if result['status'] == 'PASS_CPU_MAPPING_ONLY' else 1)


if __name__ == '__main__':
    main()
