"""Complete node-keyed kernel lifetime paths for the dynamic DAG consumer.

This numerical compiler consumes explicit training declarations. Source/native
authentication is separate; profiler lifetime is never labelled SM service.
Unlike the frozen serial compiler, different streams of one owner are peers.
"""
from __future__ import annotations

from collections import defaultdict
import copy

import numpy as np

from gsdeploy.resources import identity
from . import kernel_competition_features_v2 as features
from .implementation_source_closure_v27 import implementation_sources
from .kernel_competition_response_provider_v1 import DOMAIN

SCHEMA = 'gsdeploy.kernel_dag_response_training.v1'
CAPTURE_SCHEMA = SCHEMA + '.capture'
PATH_SCHEMA = SCHEMA + '.paths'
# These are resource/launch declarations, with no response timestamps or runtime
# IDs. Evidence and architecture allocation patches remain part of the match.
STATIC_FIELDS = ('name', 'hardware_id', 'static_evidence', 'block', 'grid',
    'registers_per_thread', 'static_shared_bytes', 'dynamic_shared_bytes',
    'allocated_registers_per_block', 'allocated_shared_bytes',
    'architecture_blocks_per_sm_ceiling', 'allocation_evidence', 'cluster',
    'cooperative_launch')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def static_kernel(kernel):
    return {k: copy.deepcopy(kernel[k]) for k in STATIC_FIELDS if k in kernel}


def catalog_index(document):
    need(document.get('training_only') is True
         and document.get('validation_labels_read', False) is False
         and type(document.get('execution_context_sha256')) is str
         and document['execution_context_sha256'], 'Explicit training DAG catalog required')
    profiles = {}
    for profile in document['profiles']:
        key = profile['alias'], profile['request_id']
        need(key not in profiles and profile.get('split') == 'training',
             'Unique owner/profile training identity required')
        nodes = {n['id']: n for n in profile['nodes']}
        need(len(nodes) == len(profile['nodes']) and not profile['unresolved']
             and profile['cyclic'] is False, 'Complete resolved unique DAG nodes required')
        profiles[key] = (profile, {k: n for k, n in nodes.items() if n['kind'] == 'gpu'})
    need(profiles, 'Nonempty complete training DAG catalog required')
    return profiles


def _capture(capture, catalog, hardware, context):
    """Validate the entire kernel bijection before compiling any timing rows."""
    need(capture.get('execution_context_sha256') == context
         and capture.get('hardware_id') == hardware.get('hardware_id'),
         'Capture context/hardware differs from immutable DAG catalog')
    offered = capture['offered_request_ids']; bindings = capture['requests']
    need(type(offered) is list and offered and all(type(r) is str and r for r in offered)
         and len(set(offered)) == len(offered)
         and [r['request_id'] for r in bindings] == offered,
         'Full ordered distinct offered request registry required')
    by_request = {}; expected = set()
    for row in bindings:
        key = row['owner'], row['profile_id']
        need(key in catalog, 'Exact owner/DAG profile binding required')
        profile, nodes = catalog[key]
        need(row['task'] == profile['task'] and row['pool_id'] == profile['pool_id'],
             'Exact request task/input/profile binding required')
        by_request[row['request_id']] = row
        expected.update((row['request_id'], n) for n in nodes)
        need(nodes, 'Every offered request must retain its canonical GPU cohort')
    actual = set(); launches = []
    for launch in capture['kernels']:
        key = launch['request_id'], launch['node_id']
        need(key in expected and key not in actual,
             'Exact full canonical GPU node bijection required')
        row = by_request[key[0]]; profile, nodes = catalog[row['owner'], row['profile_id']]
        node = nodes[key[1]]
        need(launch['owner'] == row['owner']
             and static_kernel(launch['kernel']) == static_kernel(node['kernel']),
             'Canonical launch static configuration differs from DAG node')
        left, right = launch['start_ns'], launch['end_ns']
        need(type(left) is int and type(right) is int and 0 <= left < right,
             'Positive exact integer profiler lifetime required')
        need(type(launch.get('clock_domain')) is str and launch['clock_domain'],
             'Every launch requires an explicit timeline domain')
        actual.add(key)
        launches.append(dict(launch, binding=row, key=key,
            static=static_kernel(node['kernel'])))
    need(actual == expected, 'Missing canonical kernels cannot be dropped from training')
    return launches


def compile_paths(document, hardware, captures, *, allow_cpu_fixture=False):
    """Sweep full active sets, including other nodes of the same request/owner.

    Exact intervals must share one profiler clock within each capture. A median
    host-offset conversion alone cannot release this gate. Unsupported complete
    captures retain every offered request/kernel with an explicit rejection.
    """
    captures = list(captures)
    need(type(allow_cpu_fixture) is bool, 'Explicit boolean fixture opt-in required')
    # Role gates run before any timing label is accessed.
    for cap in captures:
        need(cap.get('schema') == CAPTURE_SCHEMA and cap.get('split') == 'training'
             and cap.get('validation_labels_read') is False,
             'Training-only role required before kernel label access')
        need(cap.get('evidence_kind') == 'declared_measured'
             or cap.get('evidence_kind') == 'cpu_fixture' and allow_cpu_fixture,
             'Explicit CPU fixture opt-in or declared measured input required')
        need(type(cap.get('capture_id')) is str and cap['capture_id']
             and type(cap.get('raw_sha256')) is str and len(cap['raw_sha256']) == 64
             and type(cap.get('profiler_id')) is str and cap['profiler_id'],
             'Capture/raw/profiler identities required')
    need(captures and len({c['capture_id'] for c in captures}) == len(captures)
         and len({c['raw_sha256'] for c in captures}) == len(captures),
         'Distinct native raw identities required; aliases cannot reweight captures')
    need(len({c['profiler_id'] for c in captures}) == 1,
         'One predeclared profiler configuration domain required')
    catalog = catalog_index(document); rows = []; observations = []; rejected = []; registry = []
    offered_count = 0
    for cap in captures:
        launches = _capture(cap, catalog, hardware, document['execution_context_sha256'])
        registry.extend(dict(capture_id=cap['capture_id'], request_id=r)
                        for r in cap['offered_request_ids'])
        offered_count += len(launches)
        def reject(target, reason):
            rejected.append(dict(capture_id=cap['capture_id'], request_id=target['request_id'],
                owner=target['owner'], node_id=target['node_id'], reason=reason))
        if len({k['clock_domain'] for k in launches}) != 1:
            for target in launches: reject(target, 'UNRESOLVED_CROSS_PROFILER_CLOCK_COUPLING')
            continue
        starts = defaultdict(list); ends = defaultdict(list); paths = defaultdict(list)
        for target in launches:
            starts[target['start_ns']].append(target); ends[target['end_ns']].append(target)
        active = {}; previous = None
        for stamp in sorted(set(starts) | set(ends)):
            if previous is not None and previous < stamp:
                for key, target in active.items():
                    peers = [v['static'] for k, v in active.items() if k != key]
                    state = features.competition_features(target['static'], peers, hardware)
                    vector = (tuple(state['vector'][f] for f in features.FEATURES)
                              if state['known'] else None)
                    pieces = paths[key]
                    reason = None if state['known'] else state['reason']
                    if pieces and pieces[-1]['vector'] == vector and pieces[-1]['reason'] == reason:
                        pieces[-1]['right_ns'] = stamp
                    else:
                        pieces.append(dict(left_ns=previous, right_ns=stamp,
                                           vector=vector, reason=reason))
            # Half-open lifetimes: a kernel ending now is absent for kernels
            # starting now. The ending row's endpoint covariate is its left limit.
            for target in ends[stamp]:
                key = target['key']; pieces = paths.pop(key)
                if any(p['vector'] is None for p in pieces):
                    reject(target, next(p['reason'] for p in pieces if p['vector'] is None))
                else:
                    durations = [(p['right_ns']-p['left_ns']) / 1e6 for p in pieces]
                    vectors = [p['vector'] for p in pieces]; totals = defaultdict(float)
                    for vector, duration in zip(vectors, durations): totals[vector] += duration
                    endpoint = vectors[-1]
                    keys = [v for v in totals if v != endpoint] + [endpoint]
                    binding = target['binding']
                    rows.append(dict(group=(target['owner'], binding['profile_id'], target['node_id'],
                        cap['execution_context_sha256'], cap['profiler_id']), capture=cap['capture_id'],
                        request_id=target['request_id'], macro_id=target['node_id'], owner=target['owner'],
                        phase=target['node_id'], kind='gpu', response_ms=(stamp-target['start_ns'])/1e6,
                        durations=np.asarray([totals[v] for v in keys]), matrix=np.asarray(keys, dtype=float),
                        ordered_durations=np.asarray(durations), ordered_matrix=np.asarray(vectors, dtype=float)))
                    observations.append(dict(capture_id=cap['capture_id'], request_id=target['request_id'],
                        owner=target['owner'], node_id=target['node_id'], path=pieces))
                del active[key]
            for target in starts[stamp]: active[target['key']] = target
            previous = stamp
        need(not active and not paths, 'Sweep must consume every complete lifetime')
    need(len(rows) + len(rejected) == offered_count, 'Full accepted plus UNKNOWN denominator required')
    return dict(schema=PATH_SCHEMA, features=list(features.FEATURES), rows=rows,
        observations=observations, rejected_rows=rejected, offered_kernel_count=offered_count,
        accepted_kernel_count=len(rows), offered_requests=len(registry), full_offered_request_registry=registry,
        native_sources_authenticated=False, all_same_owner_peer_nodes_included=True,
        natural_timing_validated=False, GPU_started=False, validation_labels_read=False)


def fit_paths(prepared):
    """Use the existing lifetime likelihood; preserve all six feature columns."""
    need(prepared.get('schema') == PATH_SCHEMA
         and prepared.get('features') == list(features.FEATURES), 'Exact node-keyed path schema required')
    if prepared['rejected_rows'] or prepared['accepted_kernel_count'] != prepared['offered_kernel_count']:
        return dict(numerical_qualification=False, reason='INCOMPLETE_SUPPORTED_NODE_COHORT',
                    columns_dropped=False)
    rows = prepared['rows']; need(rows, 'Nonempty complete node training cohort required')
    groups = sorted({r['group'] for r in rows}, key=repr)
    # Project feature columns off per-node nuisance intercepts. This has exactly
    # the rank of [group dummies, X] without allocating a quadratic dummy matrix.
    contrasts = []
    origins = {}
    for row in rows:
        origin = origins.setdefault(row['group'], row['ordered_matrix'][0])
        contrasts.extend(row['ordered_matrix'] - origin)
    residual_rank = int(np.linalg.matrix_rank(np.asarray(contrasts, dtype=float)))
    if residual_rank != len(features.FEATURES):
        return dict(features=list(features.FEATURES), numerical_qualification=False,
            reason='STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE',
            structural_design_rank=len(groups)+residual_rank,
            parameters=len(groups)+len(features.FEATURES), columns_dropped=False)
    from .kernel_dag_response_fit_v1 import fit
    info, _, _, _ = fit(rows, features.FEATURES)
    info.update(feature_schema=features.SCHEMA, response_domain=DOMAIN,
        native_sources_authenticated=False, natural_timing_validated=False,
        baseline_nuisance='owner_exact_DAG_profile_node_context_profiler')
    return info


def training_bundle(document, hardware, captures, *, allow_cpu_fixture=False):
    captures = list(captures)
    need(document.get('training_only') is True
         and document.get('validation_labels_read',False) is False,
         'Training-only catalog role required before copying kernel labels')
    for cap in captures:
        need(cap.get('schema') == CAPTURE_SCHEMA and cap.get('split') == 'training'
             and cap.get('validation_labels_read') is False,
             'Training-only role required before copying kernel labels')
    # The caller can retain mutable catalogs, while fitting takes a long time.
    # One snapshot must define both the numeric rows and the final bundle IDs.
    document, hardware, captures = copy.deepcopy((document, hardware, captures))
    prepared = compile_paths(document, hardware, captures, allow_cpu_fixture=allow_cpu_fixture)
    info = fit_paths(prepared)
    result = dict(schema=SCHEMA, split='training', response_domain=DOMAIN,
        feature_schema=features.SCHEMA, features=list(features.FEATURES),
        execution_context_sha256=document['execution_context_sha256'],
        hardware_id=hardware['hardware_id'], hardware_sha256=identity(hardware),
        profiler_id=captures[0]['profiler_id'],
        kernel_catalog_sha256=identity(document), response_fit=info,
        evidence_kind='cpu_fixture' if any(c['evidence_kind']=='cpu_fixture' for c in captures) else 'declared_measured',
        capture_identities=[dict(capture_id=c['capture_id'], raw_sha256=c['raw_sha256'],
                                 sha256=identity(c)) for c in captures],
        offered_kernel_count=prepared['offered_kernel_count'], accepted_kernel_count=prepared['accepted_kernel_count'],
        rejected_kernel_count=len(prepared['rejected_rows']), offered_requests=prepared['offered_requests'],
        sources=implementation_sources(__file__), native_sources_authenticated=False,
        natural_timing_validated=False, independent_prediction_accuracy_evaluated=False,
        effective_SM_service_identified=False, joint_request_distribution_identified=False,
        physically_validated=False, deployment_qualified=False, GPU_started=False,
        validation_labels_read=False)
    result['bundle_sha256'] = identity(result)
    return result, prepared
