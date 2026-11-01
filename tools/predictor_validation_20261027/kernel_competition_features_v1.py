"""Shared kernel-competition covariates for training and generated active sets.

Resource constraints describe static per-block requirements. They never supply
DRAM traffic, achieved SM throughput, cache hits, or a service-rate multiplier.
The training adapter alone reads CUPTI END timestamps; prospective features
use only the immutable catalog and an explicitly supplied active kernel set.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from gsdeploy.forecasting.kernel_resources import static_occupancy, check_resident_blocks
from .kernel_sequence_audit_v2 import IDENTITY_FIELDS, analyze_sequence

SCHEMA = 'gsdeploy.kernel_competition_covariates.v1'
CAPTURE_SCHEMA = 'gsdeploy.kernel_competition_training_capture.v1'
FEATURES = ('peer_kernel_count', 'same_name_peer_count', 'peer_register_block_fraction',
    'peer_shared_block_fraction', 'peer_warp_block_fraction', 'mixed_one_block_infeasible')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def unknown(reason, **extra):
    return dict(schema=SCHEMA, known=False, reason=reason, vector=None,
        actual_concurrent_SM_residency_proven=False, **extra)


def competition_features(target, peers, hardware):
    """Same mapping for training and prediction; consumes no timing labels.

    ``target`` and ``peers`` are static launch declarations already bound to
    training configurations. One notional block from each launch is tested;
    infeasibility is a necessary packing conflict, not a measured slowdown.
    Feasibility does not prove that kernels share an SM or CUDA context.
    """
    kernels = {'target': target, **{'peer:' + str(i): p for i,p in enumerate(peers)}}
    if any(type(k.get('name')) is not str or not k['name'] for k in kernels.values()):
        return unknown('EXPLICIT_KERNEL_NAME_REQUIRED')
    profiles = {k: static_occupancy(v, hardware) for k,v in kernels.items()}
    if any(not p['known'] for p in profiles.values()):
        return unknown('STATIC_CONFIGURATION_UNKNOWN', static_profiles=profiles)
    if any(p['blocks_per_sm'] == 0 for p in profiles.values()):
        return unknown('STATICALLY_INFEASIBLE_KERNEL_CONFIGURATION')
    mixed = check_resident_blocks(kernels, {k:1 for k in kernels}, hardware)
    pp = [profiles['peer:' + str(i)] for i in range(len(peers))]
    vector = dict(peer_kernel_count=len(peers),
        same_name_peer_count=sum(p['name'] == target['name'] for p in peers),
        peer_register_block_fraction=sum(p['allocated_registers_per_block'] for p in pp) / hardware['registers_per_sm'],
        peer_shared_block_fraction=sum(p['allocated_shared_bytes'] for p in pp) / hardware['shared_bytes_per_sm'],
        peer_warp_block_fraction=sum(p['warps_per_block'] for p in pp) / hardware['max_warps_per_sm'],
        mixed_one_block_infeasible=int(not mixed['feasible_under_declared_constraints']))
    return dict(schema=SCHEMA, known=True, feature_names=list(FEATURES), vector=vector,
        mixed_one_block_constraints=mixed,
        interpretation='Per-block static packing covariates; no achieved occupancy or compute/DRAM classification.',
        compute_memory_classification='UNKNOWN', SM_throughput=None, DRAM_bytes=None,
        L2_hit_rate=None, service_rate_multiplier=None,
        actual_concurrent_SM_residency_proven=False, physically_validated=False)


class KernelCompetitionFeatureProvider:
    """Adapter over real KernelResourceProvider ordered plans and active rows.

    Uses its exact family/input/segment/scenario validation. Its analytical
    service method is never called, and no allocation-to-speed assumption is
    enabled by this adapter. This can feed a separate response fit.
    """
    def __init__(self, kernel_provider):
        self.provider = kernel_provider

    def features(self, row, active, aliases, scenario):
        sources = self.provider._active_segments(active, aliases, scenario)
        alias = row['alias']
        need(alias in active and row == active[alias], 'Target must belong to generated active set')
        if sources[alias]['kind'] != 'gpu':
            return unknown('TARGET_NOT_A_GPU_KERNEL')
        peers = [s['kernel'] for a,s in sources.items() if a != alias and s['kind'] == 'gpu']
        return competition_features(sources[alias]['kernel'], peers, self.provider.hardware)

    def training_rows(self, captures, *, allow_cpu_fixture=False):
        """Compile complete training captures to the existing response-fit API.

        Every canonical launch is explicitly bound to an immutable profile and
        segment. All observed overlap breakpoints are used in training only.
        Multiple overlapping/multiple-stream kernels per owner are rejected as
        whole captures: the current single-active-segment forecaster cannot
        express their dependency graph. Rejections retain the full denominator.
        Source declarations here are not a substitute for native intake audit.
        """
        captures = list(captures)
        # Role gates precede any kernel labels, even a malformed target body.
        for cap in captures:
            need(cap.get('schema') == CAPTURE_SCHEMA and cap.get('split') == 'training' and
                 cap.get('validation_labels_read') is False, 'Training role required before kernel access')
            kind = cap.get('evidence_kind')
            need(kind == 'measured' or kind == 'cpu_fixture' and allow_cpu_fixture,
                 'Explicit fixture opt-in or measured declaration required')
            need(cap.get('execution_context_sha256') == self.provider.execution_context_sha256,
                 'Training execution context mismatch')
            need(cap.get('timing_basis') == 'profiled_kernel_lifetime' and
                 type(cap.get('profiler_id')) is str and cap['profiler_id'], 'Explicit profiler timing domain required')
        need(len({c.get('capture_id') for c in captures}) == len(captures), 'Duplicate capture would reweight training')
        rows, rejected, observations = [], [], []
        offered_requests = []
        for cap in captures:
            need(type(cap.get('capture_id')) is str and cap['capture_id'], 'Capture identity required')
            kernels = cap['kernels']
            need(type(kernels) is list and kernels, 'Complete canonical kernel cohort required')
            requests = cap.get('offered_request_ids')
            need(type(requests) is list and requests and all(type(r) is str and r for r in requests)
                 and len(set(requests)) == len(requests), 'Full offered request registry required')
            offered_requests.extend(dict(capture_id=cap['capture_id'],request_id=r) for r in requests)
            need(all(k.get('request_id') in requests for k in kernels), 'Kernel belongs to a foreign request')
            missing_requests = set(requests) - {k['request_id'] for k in kernels}
            if missing_requests:
                rejected.extend(dict(capture_id=cap['capture_id'],request_id=r,
                    reason='OFFERED_REQUEST_WITHOUT_KERNEL_TRACE') for r in sorted(missing_requests))
                rejected.extend(dict(capture_id=cap['capture_id'],request_id=k['request_id'],
                    launch_identity=[k.get(f) for f in IDENTITY_FIELDS],reason='CAPTURE_HAS_UNOBSERVED_REQUEST') for k in kernels)
                continue
            seen, groups, bound = set(), defaultdict(list), []
            for launch in kernels:
                ident = tuple(launch.get(k) for k in IDENTITY_FIELDS)
                need(all(type(t) is int for t in ident) and ident[0] < ident[1] and ident not in seen,
                     'Positive unique canonical kernel identity required')
                seen.add(ident)
                owner = launch['owner']; profile_id, segment_id = launch['profile_id'], launch['segment_id']
                source = self.provider.segments.get((profile_id, segment_id))
                need(source is not None and source['kind'] == 'gpu', 'Exact ordered profile/segment binding required')
                need(launch.get('name') == source['kernel'].get('name') and
                     launch.get('static_kernel') == source['kernel'], 'Exact resolved launch static configuration required')
                need(type(owner) is str and owner and launch['pid'] == cap['owner_pids'].get(owner),
                     'Owner PID lifetime binding required')
                groups[owner].append(launch)
                bound.append((launch, source['kernel']))
            unsupported = [owner for owner, seq in groups.items() if not analyze_sequence(seq)['single_serial_stream_expressible']]
            if unsupported:
                rejected.extend(dict(capture_id=cap['capture_id'], launch_identity=list(k),
                    reason='OWNER_MULTISTREAM_OR_OVERLAP_NOT_EXPRESSIBLE', unsupported_owners=unsupported) for k in seen)
                continue
            capture_rows, capture_observations = [], []
            failed = None
            for ordinal, (target, static) in enumerate(bound):
                left, right = target['nsys_start_ns'], target['nsys_end_ns']
                points = {left, right}
                for peer, _ in bound:
                    for t in (peer['nsys_start_ns'], peer['nsys_end_ns']):
                        if left < t < right: points.add(t)
                pieces = []
                for a,b in zip(sorted(points), sorted(points)[1:]):
                    active_peers = [pstatic for peer,pstatic in bound if peer is not target and
                        peer['nsys_start_ns'] <= a < peer['nsys_end_ns']]
                    features = competition_features(static, active_peers, self.provider.hardware)
                    if not features['known']:
                        failed = features['reason']; break
                    pieces.append(dict(left_ns=a,right_ns=b,features=features))
                if failed: break
                ordered = [tuple(p['features']['vector'][f] for f in FEATURES) for p in pieces]
                durations = [(p['right_ns'] - p['left_ns']) / 1e6 for p in pieces]
                totals = defaultdict(float)
                for vector,duration in zip(ordered,durations): totals[vector] += duration
                endpoint = ordered[-1]; keys = [k for k in totals if k != endpoint] + [endpoint]
                capture_rows.append(dict(group=(target['owner'], target['profile_id'], target['segment_id'],
                    cap['execution_context_sha256'], cap['profiler_id']), capture=cap['capture_id'],
                    request_id=target['request_id'], macro_id='kernel:' + str(ordinal),
                    owner=target['owner'],phase=target['segment_id'],kind='gpu',
                    response_ms=(right-left)/1e6, durations=np.asarray([totals[k] for k in keys]),
                    matrix=np.asarray(keys,dtype=float),ordered_durations=np.asarray(durations),
                    ordered_matrix=np.asarray(ordered,dtype=float)))
                capture_observations.append(dict(capture_id=cap['capture_id'],launch_identity=[target[k] for k in IDENTITY_FIELDS],
                    kernel_response_domain='CUPTI_PROFILED_KERNEL_LIFETIME_NOT_SM_SERVICE', path=pieces))
            if failed:
                rejected.extend(dict(capture_id=cap['capture_id'],launch_identity=list(k),reason=failed) for k in seen)
            else:
                rows.extend(capture_rows); observations.extend(capture_observations)
        return dict(schema=SCHEMA, features=list(FEATURES), rows=rows, observations=observations,
            rejected_rows=rejected, offered_kernel_count=sum(len(c['kernels']) for c in captures),
            full_offered_request_registry=offered_requests,offered_requests=len(offered_requests),
            accepted_kernel_count=len(rows), complete_offered_kernel_denominator_retained=True,
            native_sources_authenticated=False, natural_timing_validated=False,
            prospective_main_chain_wired=False, GPU_started=False, validation_labels_read=False)


def fit_response(prepared):
    """Call the unchanged shared response-likelihood solver, without relabeling.

    This fit estimates a statistical kernel-lifetime response law. It is not
    the isolated operator baseline estimator and has no SM service meaning.
    All six declared covariates are retained; rank/root failures stay failures.
    """
    need(prepared.get('schema') == SCHEMA and prepared.get('features') == list(FEATURES), 'Feature schema mismatch')
    need(prepared.get('rejected_rows') == [] and prepared.get('accepted_kernel_count') ==
         prepared.get('offered_kernel_count') and prepared.get('rows'), 'Complete supported kernel cohort required before fit')
    rows = prepared['rows']
    groups = sorted({r['group'] for r in rows}, key=repr)
    # A feature contrast constant within a baseline group is exactly absorbed
    # by that group's reference time. Reject this global structural invariance
    # before asking the numerical optimizer to choose arbitrary coefficients.
    design = []
    for row in rows:
        group_vector = [int(row['group'] == g) for g in groups]
        design.extend([group_vector + list(vector) for vector in row['ordered_matrix']])
    design = np.asarray(design, dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    dimension = len(groups) + len(FEATURES)
    if rank != dimension:
        return dict(features=list(FEATURES),numerical_qualification=False,
            reason='STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE',
            structural_design_rank=rank,parameters=dimension,columns_dropped=False,
            effective_SM_service_identified=False,prospective_prediction_qualified=False), None, None, None
    from .macro_projection_contrast_training_v1 import fit
    info, beta, mu, variance = fit(rows, FEATURES)
    info.update(baseline_nuisance='owner_exact_ordered_kernel_configuration_execution_context_profiler',
        response_domain='CUPTI_PROFILED_KERNEL_LIFETIME_NOT_SM_SERVICE',
        feature_schema=SCHEMA, native_sources_authenticated=False,
        natural_timing_validated=False)
    return info, beta, mu, variance
