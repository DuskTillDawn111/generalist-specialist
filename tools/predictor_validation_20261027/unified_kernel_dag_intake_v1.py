"""CPU-only native DAG/path intake for the explicit unified-v8_1 schema.

Reparse the existing SQLite; never export/collect, relabel isolated context,
infer SM service, or fit a response law. Native kernel endpoints remain in the
single session's nanosecond coordinate, including same-owner OCR peer nodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from gsdeploy.resources import identity
from . import kernel_dag_response_training_v1 as training
from .cuda_dependency_dag_v2 import audit_sqlite
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .kernel_dag_training_mapping_audit_v1 import validate_mapping, require_denominator
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS, verify
from .native_collect import receipt
from .ordered_occupancy_resolver_v2 import AuditedOccupancyResolver
from .request_cuda_dag_v2 import request_plan
from .smoke_verifier_v3 import validate_request
from .trace_export import parse_sqlite
from .unified_corun_review_v8_1 import LIGHT, REVIEW_SCHEMA, load_review, trace_metadata, RAW_SCHEMA

SCHEMA = 'gsdeploy.unified_kernel_dag_intake.v1'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def authenticated(value):
    refs = strict_references(value)
    for ref in refs:
        verify(ref)
    return refs


def native_key(kernel):
    return tuple(kernel.get(field) for field in IDENTITY_FIELDS)


def canonical_requests(raw, parsed, sql_ref):
    """Attribute the fresh canonical range view once, keeping outside kernels."""
    rows = raw['request_records']
    requests = {row['request_id']: dict(row=row, kernels=[],
        clock_mapping=parsed['clock_mappings'].get(str(raw['ready'][row['model']]['pid']))) for row in rows}
    need(len(requests) == len(rows) and requests, 'Full distinct offered request denominator required')
    seen = set(); outside = []
    entries = [(region.get('stage'), kernel) for region in parsed['nvtx_ranges']
               for kernel in region['kernels']]
    entries.extend((None, kernel) for kernel in parsed['unattributed_kernels'])
    for stage, kernel in entries:
        need(all(type(kernel.get(field)) is int for field in IDENTITY_FIELDS)
             and 0 <= kernel['nsys_start_ns'] < kernel['nsys_end_ns'],
             'Every native measured/outside kernel requires a positive exact integer interval and identities')
        key = native_key(kernel)
        need(key not in seen, 'Duplicate fresh canonical native kernel')
        seen.add(key)
        request = requests.get(stage[0] if stage else None)
        if request is None:
            outside.append(kernel)
            continue
        alias = request['row']['model']
        need(kernel['pid'] == raw['ready'][alias]['pid'], 'Native kernel differs from READY owner')
        request['kernels'].append(dict(kernel, source_sqlite=sql_ref))
    need(all(r['kernels'] for r in requests.values()), 'Offered request missing native kernels')
    # An unmeasured launch can still be an active peer. Never silently omit it.
    events = []
    for request in requests.values():
        for kernel in request['kernels']:
            events.extend(((kernel['nsys_start_ns'], 1, 0), (kernel['nsys_end_ns'], -1, 0)))
    for kernel in outside:
        events.extend(((kernel['nsys_start_ns'], 0, 1), (kernel['nsys_end_ns'], 0, -1)))
    counts = [0, 0]; previous = None; overlap_ns = 0
    deltas = {}
    for stamp, measured, unmeasured in events:
        delta = deltas.setdefault(stamp, [0, 0]);delta[0] += measured;delta[1] += unmeasured
    for stamp, delta in sorted(deltas.items()):
        if previous is not None and all(counts):
            overlap_ns += stamp-previous
        counts = [a+b for a,b in zip(counts,delta)];previous = stamp
    return requests, dict(full_session_kernels=len(seen), outside_measured_kernels=len(outside),
                          outside_measured_overlap_wall_ns=overlap_ns)


def bind_static(profiles, hardware, occupancy_audit=None):
    """Use only exact SQLite member joins; absent/unresolved members stay UNKNOWN."""
    resolver = AuditedOccupancyResolver(occupancy_audit) if occupancy_audit else None
    if resolver:
        need(resolver.hardware == hardware, 'Exact full hardware declaration differs from occupancy audit')
    counts = Counter()
    for profile in profiles:
        for node in profile['nodes']:
            if node['kind'] != 'gpu':
                continue
            kernel = node['kernel']
            key = (kernel['source_sqlite']['sha256'], kernel['nsys_start_ns'], kernel['nsys_end_ns'],
                   kernel['pid'], kernel['context_id'], kernel['correlation_id'])
            patched = resolver.index.get(key) if resolver else None
            if patched is None:
                counts['UNKNOWN_EXACT_STATIC_ALLOCATION'] += 1
                kernel['hardware_id'] = hardware['hardware_id']
                continue
            for field in ('block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster'):
                need(patched.get(field) == kernel.get(field), 'Exact audited native launch metadata mismatch')
            kernel.update(copy.deepcopy(patched));counts['KNOWN_EXACT_STATIC_ALLOCATION'] += 1
    return dict(counts), resolver.source_receipts if resolver else []


def structural_diagnostics(prepared):
    """No solver: diagnose complete paths and the per-node replication barrier."""
    if prepared['rejected_rows']:
        return dict(numerical_qualification=False, reason='INCOMPLETE_SUPPORTED_NODE_COHORT',
                    fit_started=False, columns_dropped=False)
    origins = {};contrasts = [];groups = Counter()
    for row in prepared['rows']:
        groups[row['group']] += 1
        origin = origins.setdefault(row['group'], row['ordered_matrix'][0])
        contrasts.extend(row['ordered_matrix']-origin)
    rank = int(np.linalg.matrix_rank(np.asarray(contrasts)))
    singleton = sum(count == 1 for count in groups.values())
    return dict(numerical_qualification=False, fit_started=False, columns_dropped=False,
        feature_count=len(prepared['features']), within_node_path_covariate_rank=rank,
        baseline_groups=len(groups), singleton_baseline_groups=singleton,
        independent_capture_count=1,
        reason=('ONE_OBSERVATION_PER_EXACT_NODE_BASELINE_NOT_IDENTIFIABLE'
                if singleton == len(groups) else 'NO_NUMERICAL_FIT_REQUESTED'),
        nuisance_adjusted_derivative_rank=0 if singleton == len(groups) else None,
        explanation='One free baseline per single response absorbs each integrated path for every beta; path breakpoints are not independent response replications.')


def intake(review_path, device_attributes, output, *, occupancy_audit=None):
    output = Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    implementation = implementation_sources(__file__)
    strict = json.loads(Path(review_path).read_text())
    need(strict.get('schema') == REVIEW_SCHEMA and strict.get('status') == 'PASS'
         and strict.get('issues') == [] and strict.get('training_only') is True
         and strict.get('validation_started') is False and strict.get('model_fitted') is False,
         'Explicit unmodified unified training-only strict PASS required before label access')
    strict, plan, raw = load_review(review_path,LIGHT)
    need(plan.get('schema') == 'gsdeploy.unified_corun_training_plan.v8_1'
         and plan.get('training_only') is True and plan.get('split') == 'training'
         and raw.get('schema') == RAW_SCHEMA
         and raw.get('measurement_domain') == LIGHT and raw.get('status') == 'completed'
         and all(r.get('split') == 'training' for r in raw['request_records']),
         'Exact unified-v8_1 measured training cohort required')
    context = json.loads(Path(strict['evidence']['context.json']['path']).read_text())
    need(raw['execution_context_sha256'] == plan['execution_context_sha256']
         and context.get('training_only') is True, 'Unified execution context differs')
    hardware_doc = json.loads(Path(device_attributes).read_text())
    devices = hardware_doc['query']['devices']
    need(len(devices) == 1 and devices[0].get('hardware'), 'One evidenced full hardware declaration required')
    hardware = copy.deepcopy(devices[0]['hardware'])
    static_evidence = ([receipt(occupancy_audit),json.loads(Path(occupancy_audit).read_text())]
                       if occupancy_audit else [])
    sources = authenticated([implementation, receipt(review_path), strict, plan, context,
        receipt(device_attributes), static_evidence])
    trace = strict['trace'];sql_ref = trace['sqlite']
    need(trace.get('common_session_clock_identified') is True, 'Native common-session clock required')
    metadata = trace_metadata(sql_ref['path'])
    need(not metadata['issues'], 'Native session metadata/diagnostics no longer qualify')
    anchors = []
    for row in raw['request_records']:
        issues, paired = validate_request(row,raw['ready'][row['model']]['pid'])
        need(not issues, 'Native request/anchor chain failed: '+row['request_id']);anchors.extend(paired)
    parsed = parse_sqlite(sql_ref['path'],anchors)
    saved = json.loads(Path(trace['parsed']['path']).read_text())
    need(saved.get('schema') == 'gsdeploy.unified_measured_phases.v8_1'
         and saved['phases'] == parsed['phases'] and saved['clock_mappings'] == parsed['clock_mappings']
         and saved['unmatched_anchor_count'] == parsed['unmatched_anchor_count'] == 0,
         'Fresh native parsed measured cohort differs from strict source-bound phases')
    requests, inventory = canonical_requests(raw,parsed,sql_ref)
    require_denominator(requests, [r for owner in strict['owners'].values() for r in owner['request_ids']])
    need(all(len(requests[rid]['kernels']) == owner['summary']['kernel_counts'][rid]
             for owner in strict['owners'].values() for rid in owner['request_ids']),
         'Full native canonical kernel denominator differs from strict owner review')
    full = audit_sqlite(sql_ref['path'])
    need(not full['missing_runtime'] and not full['dag']['unresolved'] and not full['dag']['cyclic'],
         'Whole-session CUDA dependency reconstruction incomplete')
    profiles=[];mapping=[]
    for rid, request in requests.items():
        profile=request_plan(rid,request,parsed,full,sql_ref['path'])
        profile['family']=context['residents'][profile['alias']]
        details=validate_mapping(rid,request,profile,raw['ready'][profile['alias']]['pid'])
        profiles.append(profile);mapping.append(dict(request_id=rid,alias=profile['alias'],**details,
            activity_counts=profile['activity_counts'],edge_counts=dict(Counter(e['kind'] for e in profile['edges']))))
    static, static_refs=bind_static(profiles,hardware,occupancy_audit)
    sources=authenticated([sources,static_refs])
    catalog=dict(schema=SCHEMA+'.catalog',training_only=True,validation_labels_read=False,
        execution_context_sha256=raw['execution_context_sha256'],profiles=profiles,sources=sources,
        native_mapping_authenticated=True,context_pooling_with_isolated_allowed=False,
        host_release_coordinates='Descriptive per-owner handler mapping only; never used to align cross-owner GPU lifetime labels')
    capture=dict(schema=training.CAPTURE_SCHEMA,split='training',validation_labels_read=False,
        evidence_kind='declared_measured',capture_id=strict['capture_run_id'],
        raw_sha256=strict['evidence']['raw.json']['sha256'],profiler_id=context['profiler'],
        execution_context_sha256=raw['execution_context_sha256'],hardware_id=hardware['hardware_id'],
        offered_request_ids=list(requests),requests=[],kernels=[])
    for profile in profiles:
        capture['requests'].append(dict(request_id=profile['request_id'],owner=profile['alias'],
            profile_id=profile['request_id'],task=profile['task'],pool_id=profile['pool_id']))
        for node in profile['nodes']:
            if node['kind']=='gpu':
                kernel=node['kernel']
                capture['kernels'].append(dict(request_id=profile['request_id'],owner=profile['alias'],
                    node_id=node['id'],kernel=kernel,start_ns=kernel['nsys_start_ns'],end_ns=kernel['nsys_end_ns'],
                    clock_domain='native-common-session-sqlite:'+sql_ref['sha256']))
    prepared=training.compile_paths(catalog,hardware,[capture])
    if inventory['outside_measured_overlap_wall_ns']:
        # These unlabelled active peers invalidate every row in this capture.
        prepared['rejected_rows']=[dict(capture_id=capture['capture_id'],request_id=k['request_id'],
            owner=k['owner'],node_id=k['node_id'],reason='UNASSIGNED_NATIVE_KERNEL_PEERS_OVERLAP_MEASURED_COHORT')
            for k in capture['kernels']]
        prepared.update(rows=[],observations=[],accepted_kernel_count=0)
    def save(name,value):
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    save('catalog.json',catalog);save('capture.json',capture)
    save('paths.json',dict(schema=prepared['schema'],observations=prepared['observations'],
        rejected_rows=prepared['rejected_rows'],features=prepared['features'],
        offered_kernel_count=prepared['offered_kernel_count'],accepted_kernel_count=prepared['accepted_kernel_count'],
        full_offered_request_registry=prepared['full_offered_request_registry']))
    authenticated(sources);need(implementation_sources(__file__)==implementation,'Intake implementation changed')
    report=dict(schema=SCHEMA,status='COMPLETE_NATIVE_MAPPING_PATH_DIAGNOSTICS_NO_FIT',
        review=receipt(review_path),raw=strict['evidence']['raw.json'],sqlite=sql_ref,
        execution_context_sha256=raw['execution_context_sha256'],hardware_sha256=identity(hardware),
        profiler_id=context['profiler'],full_offered_request_registry=capture['offered_request_ids'],
        offered_requests=len(requests),offered_kernel_count=prepared['offered_kernel_count'],
        accepted_kernel_count=prepared['accepted_kernel_count'],rejected_kernel_count=len(prepared['rejected_rows']),
        rejected_reasons=dict(Counter(r['reason'] for r in prepared['rejected_rows'])),
        full_session_inventory=inventory,requests=mapping,static_coverage=static,
        whole_session_activity_counts=full['activity_counts'],whole_session_missing_runtime=full['missing_runtime'],
        whole_session_unresolved=full['dag']['unresolved'],whole_session_cyclic=full['dag']['cyclic'],
        common_native_session_clock=True,median_offsets_used_for_cross_owner_lifetimes=False,
        same_owner_peer_nodes_included=True,native_mapping_authenticated=True,
        compiler_native_sources_authenticated=prepared['native_sources_authenticated'],
        diagnostics=structural_diagnostics(prepared),sources=sources,source_receipts_unchanged=True,
        artifacts={name:receipt(output/name) for name in ('catalog.json','capture.json','paths.json')},
        isolated_context_pooling_allowed=False,independent_capture_count=1,GPU_started=False,fit_started=False,
        validation_labels_read=False,effective_SM_service_identified=False,
        independent_prediction_accuracy_evaluated=False,physically_validated=False,deployment_qualified=False)
    save('report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True);parser.add_argument('--device-attributes',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--occupancy-audit')
    args=parser.parse_args();report=intake(args.review,args.device_attributes,args.output,occupancy_audit=args.occupancy_audit)
    print(json.dumps({k:report[k] for k in ('status','offered_requests','offered_kernel_count','accepted_kernel_count','rejected_kernel_count','static_coverage','diagnostics')}))


if __name__=='__main__':main()
