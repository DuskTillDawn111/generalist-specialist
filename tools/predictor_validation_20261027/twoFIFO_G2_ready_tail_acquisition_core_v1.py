"""CPU-only conditional G2-tail acquisition design and exposure arithmetic.

This is deliberately not a producer, native reviewer, fit coordinator or launch
gate. Calendar proposals cannot execute. Pure statistics require separately
authenticated complete macro documents; no Boolean here authenticates native
evidence. Failed/unstarted requests remain in the offered-outcome ledger and
prevent complete-population statistics. No retry, optimizer or GPU API exists.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stage_response_features_v1 import checked, receipt
from .twoFIFO_training_manifest_v1 import OWNERS, digest, require
from .twoFIFO_training_intake_v4 import processed_peers
from . import phase_heterogeneous_response_training_v1 as phase

SCHEMA = 'gsdeploy.G2_ready_tail_conditional_acquisition_CPU_core.v1'
ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT/'artifacts/predictor-G2-ready-tail-targeted-acquisition-v1-20261029-cpu01-prepared'
CALENDAR_SHA = '10a75181e01004cb79bc9a2ed20f12bbf14b55092a14c635d4a031164adfd362'
PROPOSAL_SHA = '79df7faf509e6b4f315203516191c19ceaff1b93feb4d83ed70999bbd4526888'
MATCHING = ROOT/'artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared'
RESOURCE = ROOT/'artifacts/predictor-public181-resource-after-kernel-v1-20261029-cpu01-prepared'
MATCHING_FINAL = MATCHING/'run-v1/report.json'
RESOURCE_FINAL = RESOURCE/'run-v1/report.json'
MATCHING_STATUS = 'COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT'
RESOURCE_STATUS = 'COMPLETE_EIGHT_NATIVE_RESOURCE_TRAINING_AND_INTAKE_NOT_VALIDATION'
FEATURES = phase.DESIGNS['own_family_phase_count_global_same']
TARGET_FEATURE = 'count@generalist_2:ready_tail'
TARGET_POOL = 'tydiqa-new-native-e0ff7b578f9c742a6697469a'
PEER_POOL = 'tydiqa-new-native-02ea54b33b0492f20de11e0f'
OFFSETS = {'quarter': 140710722, 'middle': 281421445, 'three-quarter': 422132168}
ORDERS = [('solo', 'quarter', 'middle', 'three-quarter'),
          ('middle', 'solo', 'three-quarter', 'quarter'),
          ('three-quarter', 'quarter', 'solo', 'middle')]
# Predeclared finite probe, never estimated from outcomes or treated as a root.
FINITE_PROBE_BETA = tuple(.25 for _ in FEATURES)


def verify(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}
            and receipt(ref['path']) == ref, 'Changed exact CPU design receipt')
    return ref


def calendar_contract(proposal_directory=PROPOSAL):
    """Authenticate the existing proposal; return no executable plan."""
    folder = Path(proposal_directory).resolve()
    calendar_ref = receipt(folder/'cases-by-block.json')
    report_ref = receipt(folder/'report.json')
    require(calendar_ref['sha256'] == CALENDAR_SHA and report_ref['sha256'] == PROPOSAL_SHA,
            'Only the unchanged predeclared proposal and calendar are supported')
    blocks = checked(calendar_ref); proposal = checked(report_ref)
    require(proposal['calendar'] == calendar_ref, 'Proposal calendar path/receipt differs')
    bindings = proposal['baseline_bindings']
    for name in ('context', 'pool', 'semantic_progress_worker', 'original_dispatch'):
        verify(bindings[name])
    context = checked(bindings['context'])
    require(len(blocks) == 3 and len({b['capture_id'] for b in blocks}) == 3,
            'Three independently restarted blocks required')
    all_ids = []
    for index, block in enumerate(blocks):
        require(block['each_attempt_independent_startup'] is True
                and block['shards_in_block'] == block['shard_index'] == 1
                and block['resident_owners'] == list(OWNERS)
                and block['runtime_policy'] == bindings['runtime_policy']
                and block['runtime_config'] == bindings['runtime_config']
                and block['context'] == bindings['context'] and block['prepared_pool'] == bindings['pool']
                and block['measurement_ns'] == 10_000_000_000 and block['drain_ns'] == 60_000_000_000
                and block['no_retries'] is True and block['GPU_started'] is False
                and len(block['priming']) == 18 and len(block['cases']) == 4,
                'Unchanged resident/runtime/FIFO/priming/fixed-window contract required')
        require(block['runtime_policy'] == dict(admission=None, batching=False, weight_reuse=False,
                enqueue_all_without_waiting_for_RESULT_or_READY=True), 'No admission cap or batching allowed')
        require(block['priming_semantics']['ordered_inputs'] ==
                [dict(model=r['owner'], task=r['task'], pool_id=r['pool_id'],
                      exact_input_sha256=r['exact_input_sha256']) for r in block['priming']],
                'Exact ordered physical priming semantics required')
        declared = {r['pool_id']: r for r in context['declared_inputs']}
        for row in block['priming']:
            require(declared[row['pool_id']]['exact_input_sha256'] == row['exact_input_sha256'],
                    'Priming physical input changed')
        for case, position in zip(block['cases'], ORDERS[index]):
            arrivals = case['arrivals']
            projected = [(a['owner'], a['task'], a['pool_id'], a['arrival_ns']) for a in arrivals]
            expected = [('generalist_2#0', 'text', TARGET_POOL, 0)]*2 if position == 'solo' else [
                ('text#0', 'text', PEER_POOL, 0), ('text#0', 'text', PEER_POOL, 0),
                ('generalist_2#0', 'text', TARGET_POOL, OFFSETS[position])]
            require(projected == expected and case['offered'] == len(arrivals)
                    and case['public_arrivals_sha256'] == digest(arrivals)
                    and case['positive_exposure_guaranteed'] is False
                    and [a['arrival_index'] for a in arrivals] == list(range(len(arrivals))),
                    'Fixed calendar/order/real queued-peer intent changed')
        all_ids.extend(r['request_id'] for c in block['cases'] for r in c['arrivals'])
        all_ids.extend(r['request_id'] for r in block['priming'])
    require(len(all_ids) == len(set(all_ids)) == 87, 'All33 measured and54 priming IDs must remain disjoint')
    return dict(schema=SCHEMA+'.calendar', proposal=report_ref, calendar=calendar_ref,
        blocks=blocks, measured_offered=33, priming_history_offered=54,
        new_native_producer_implemented=False, executable=False, GPU_started=False,
        positive_exposure_guaranteed=False, numerical_fit_invoked=False)


def outcome_ledger(blocks, raw_by_attempt=None):
    """Retain every offered outcome, even if startup/case/dispatch never begins.

    This is a CPU projection, not a native protocol review. Original rows are
    retained verbatim under observed_row; no successful subset is made labels.
    """
    raw_by_attempt = {} if raw_by_attempt is None else raw_by_attempt
    require(set(raw_by_attempt) <= {b['attempt_id'] for b in blocks}, 'Unknown supplied attempt')
    result = []
    for block in blocks:
        raw = raw_by_attempt.get(block['attempt_id'])
        planned = [('measured', r) for c in block['cases'] for r in c['arrivals']]
        planned += [('history_only_priming', r) for r in block['priming']]
        observed = {}
        if raw is not None:
            require(all(raw.get(k) == block[k] for k in ('capture_id', 'attempt_id')),
                    'Raw projection belongs to another attempt')
            rows = raw['request_records'] + raw['priming_request_records']
            require(len(rows) == len(planned) and len({r['request_id'] for r in rows}) == len(rows)
                    and {r['request_id'] for r in rows} == {r['request_id'] for _, r in planned},
                    'Failures and unstarted offered rows must be retained, never dropped')
            observed = {r['request_id']: r for r in rows}
        for role, public in planned:
            row = observed.get(public['request_id'])
            result.append(dict(capture_id=block['capture_id'], attempt_id=block['attempt_id'],
                request_id=public['request_id'], role=role, public=copy.deepcopy(public),
                status='NOT_OBSERVED' if row is None else row.get('status', 'UNKNOWN'),
                observed_row=copy.deepcopy(row), exposure_support='UNKNOWN_REQUIRES_COMPLETE_NATIVE_REVIEW',
                prediction_label_eligible=False))
    return dict(offered=len(result), measured_offered=sum(r['role'] == 'measured' for r in result),
        priming_offered=sum(r['role'] == 'history_only_priming' for r in result), rows=result,
        incomplete_or_unobserved=sum(r['observed_row'] is None or r['status'] != 'completed' for r in result),
        all_offered_retained=True, survivor_statistics_computed=False, native_evidence_authenticated=False)


def _group(macro):
    shape = macro.get('forward_shape') or {}
    return (macro['owner'], macro['pool_id'], macro['task'], macro['macro_id'],
            macro['semantic_phase'], macro['kind'], *(shape.get(k) for k in
            ('input_sequence_length', 'input_batch_size', 'forward_index')))


def exposure_statistics(documents):
    """All macro rows retained; positivity means processed-prefix exposure only.

    Callers must independently authenticate complete documents. Integer clock
    subtraction precedes floating conversion, including a1ns segment above2**53.
    """
    groups = {}; measured = 0; all_macros = 0; identities = set(); records = []
    for doc in documents:
        require(doc.get('validation_labels_read') is False
                and doc.get('fitted_model_qualified') is False
                and doc.get('whole_attempt_history_used_for_visible_prefix') is True
                and doc.get('case_history_reset') is False,
                'Training whole-attempt processed-prefix documents required')
        require(doc['measured_requests'] == len(doc['requests']), 'All measured requests required')
        for request in doc['requests']:
            key = (doc['capture_id'], request['request_id'])
            require(key not in identities, 'Duplicate request is not another observation')
            identities.add(key); measured += 1
            macros = request['macros']; require(macros, 'Missing request macros are UNKNOWN, not zero exposure')
            require(all(a['right_ns'] == b['left_ns'] for a, b in zip(macros, macros[1:]))
                    and sum(m['response_wall_ns'] for m in macros) == request['worker_start_to_ready_ns'],
                    'Complete disjoint START-to-READY request denominator required')
            for macro in macros:
                all_macros += 1
                path = macro['causal_covariate_path']; left = macro['left_ns']; right = macro['right_ns']
                require(type(left) is int and type(right) is int and right > left
                        and path and path[0]['left_ns'] == left and path[-1]['right_ns'] == right
                        and all(a['right_ns'] == b['left_ns'] for a, b in zip(path, path[1:])),
                        'Positive complete integer macro path required')
                total = 0; zero = 0; positive = 0; weighted_count = 0; pieces = []
                for piece in path:
                    duration = piece['duration_ns']; start = piece['left_ns']; end = piece['right_ns']
                    state = piece['state']
                    require(type(start) is int and type(end) is int and type(duration) is int
                            and duration > 0 and end-start == duration
                            and state['cutoff_processing_ns'] == start
                            and piece['each_state_uses_only_its_processed_prefix'] is True,
                            'No float-clock subtraction, synthetic duration or future state')
                    peers = processed_peers(state, start, macro['owner'], macro['semantic_phase'])
                    count = len(peers); total += duration; weighted_count += duration*count
                    if count: positive += duration
                    else: zero += duration
                    pieces.append(dict(left_ns=start, right_ns=end, duration_ns=duration,
                        peer_count=count, peers=copy.deepcopy(peers)))
                require(total == right-left == macro['response_wall_ns'], 'Exposure mass changed')
                if macro['owner'] != 'generalist_2#0' or macro['semantic_phase'] != 'ready_tail':
                    continue
                group = _group(macro)
                record = groups.setdefault(group, dict(group=list(group), rows=0,
                    n0_exposure_ns=0, positive_peer_exposure_ns=0, total_ns=0, fractions=[]))
                record['rows'] += 1; record['n0_exposure_ns'] += zero
                record['positive_peer_exposure_ns'] += positive; record['total_ns'] += total
                record['fractions'].append(weighted_count/total)
                records.append(dict(capture_id=doc['capture_id'], attempt_id=doc['attempt_id'],
                    request_id=request['request_id'], group=list(group), pieces=pieces,
                    n0_exposure_ns=zero, positive_peer_exposure_ns=positive, total_ns=total,
                    positive_exposure_fraction=positive/total))
    for group in groups.values():
        group['n0_and_positive_observed'] = group['n0_exposure_ns'] > 0 and group['positive_peer_exposure_ns'] > 0
        fractions = group.pop('fractions')
        group['zero_point_within_group_count_derivative_varies'] = len(set(fractions)) > 1
        group['positive_exposure_fraction'] = group['positive_peer_exposure_ns']/group['total_ns']
    positive = sum(g['positive_peer_exposure_ns'] for g in groups.values())
    total = sum(g['total_ns'] for g in groups.values())
    return dict(measured_requests=measured, all_macro_rows=all_macros, G2_ready_tail_rows=len(records),
        total_ns=total, positive_peer_exposure_ns=positive,
        positive_exposure_fraction=None if total == 0 else positive/total,
        any_G2_tail_positive_peer_exposure=positive > 0,
        same_group_n0_and_positive_observed=any(g['n0_and_positive_observed'] for g in groups.values()),
        groups=sorted(groups.values(), key=lambda g:repr(g['group'])), records=records,
        native_evidence_authenticated=False, physical_SM_service_identified=False,
        semantic_positivity_is_numerical_qualification=False)


def derivative_diagnostic(rows, beta):
    """Original12 nuisance-adjusted log-time-change derivatives; no likelihood fit.

    Algebra equals the derivative in stage_response_fit_v1.profile. A finite
    probe is not a root; no stationarity, curvature or accuracy claim follows.
    """
    beta = np.asarray(beta, dtype=float)
    require(beta.shape == (12,) and np.isfinite(beta).all() and rows, 'Finite original12 probe and rows required')
    groups = sorted({tuple(r['group']) for r in rows}, key=repr)
    nuisance = np.zeros((len(rows), len(groups))); derivatives = []
    for index, row in enumerate(rows):
        matrix = np.asarray(row['matrix'], dtype=float); duration = np.asarray(row['durations'], dtype=float)
        require(matrix.ndim == 2 and matrix.shape == (len(duration), 12)
                and np.isfinite(matrix).all() and np.isfinite(duration).all() and (duration > 0).all(),
                'Every original12 row and positive duration retained')
        values = np.log(duration)-matrix@beta; shifted = np.exp(values-np.max(values))
        weights = shifted/sum(shifted); derivatives.append(-(weights@matrix))
        nuisance[index, groups.index(tuple(row['group']))] = -1
    derivatives = np.asarray(derivatives); adjusted = derivatives.copy()
    for group in groups:
        members = [i for i, row in enumerate(rows) if tuple(row['group']) == group]
        adjusted[members] -= np.mean(derivatives[members], axis=0)
    joint = np.column_stack((nuisance, derivatives)); singular = np.linalg.svd(joint, compute_uv=False)
    rank = int(np.linalg.matrix_rank(joint)); target = list(FEATURES).index(TARGET_FEATURE)
    return dict(beta=beta.tolist(), features=list(FEATURES), rows=len(rows), nuisance_groups=len(groups),
        joint_rank=rank, joint_parameters=joint.shape[1], joint_full_column_rank=rank == joint.shape[1],
        nuisance_adjusted_derivative_rank=int(np.linalg.matrix_rank(adjusted)),
        joint_singular_values=singular.tolist(),
        G2_tail_nuisance_adjusted_derivative_l2=float(np.linalg.norm(adjusted[:, target])),
        optimizer_run=False, no_columns_deleted=True, thresholds_changed=False,
        fitted_root_numerical_qualification=False)


def full_and_leaveblock_statistics(documents, *, expected_attempt_ids, rows=None):
    """Pure full/three whole-block fold arithmetic; no native authentication.

    For matching15 invoke only after original intake/native re-review. Missing
    or failed requests never become an implicit successful-population sample.
    """
    blocks = sorted({d['capture_id'] for d in documents})
    require(len(blocks) == 3 and len({d['attempt_id'] for d in documents}) == len(documents),
            'Three whole independent blocks and unique attempts required')
    require(len(documents) in (3, 15) and len(expected_attempt_ids) == len(set(expected_attempt_ids))
            and {d['attempt_id'] for d in documents} == set(expected_attempt_ids),
            'Every predeclared attempt required; missing/failed attempts cannot form a survivor population')
    per_block = len(documents)//3; per_attempt = 11 if per_block == 1 else 26
    require(all(sum(d['capture_id'] == b for d in documents) == per_block for b in blocks)
            and all(d['measured_requests'] == len(d['requests']) == per_attempt
                    and d['priming_rows_excluded_as_labels'] == 18 for d in documents),
            'Exact33+54 targeted or390+270 matching denominators required')
    result = []
    for heldout in [None, *blocks]:
        selected = [d for d in documents if d['capture_id'] != heldout]
        item = dict(heldout_block_id=heldout, training_block_ids=[b for b in blocks if b != heldout],
                    exposure=exposure_statistics(selected))
        if rows is not None:
            selected_rows = [r for r in rows if r['capture'] != heldout]
            require(len(selected_rows) == item['exposure']['all_macro_rows'], 'All original12 macro rows retained')
            item['zero_point'] = derivative_diagnostic(selected_rows, np.zeros(12))
            item['finite_predeclared_probe'] = derivative_diagnostic(selected_rows, FINITE_PROBE_BETA)
        result.append(item)
    return dict(populations=result, whole_block_folds=3, attempt_count=len(documents),
                measured_requests=per_attempt*len(documents), priming_requests=18*len(documents),
                complete_predeclared_population=True, native_evidence_authenticated=False,
                numerical_fit_invoked=False, fitted_model_qualified=False, validation_labels_read=False)


def conditional_design_status(*, matching_complete_and_independently_rederived, statistics,
                              resource_chain_complete_and_rederived=False):
    """Pure decision description; caller assertions never authorize execution.

    A generic rank failure needs attribution; positive exposure alone cannot
    establish usable same-group or numerical support. Independent review, native producer, lease/chain checks and final
    launch authorization do not exist here. All outputs keep executable=False.
    """
    status = 'HOLD_MATCHING15_NOT_COMPLETE_OR_NOT_REDERIVED'
    gaps = []; candidate = None
    if matching_complete_and_independently_rederived:
        require(statistics is not None and len(statistics['populations']) == 4,
                'Full and all three whole-block fold statistics required')
        require(statistics.get('attempt_count') == 15 and statistics.get('measured_requests') == 390
                and statistics.get('priming_requests') == 270
                and statistics.get('complete_predeclared_population') is True,
                'Only complete original matching15 can diagnose need for the future calendar')
        populations = statistics['populations']
        require(populations[0]['heldout_block_id'] is None
                and len({p['heldout_block_id'] for p in populations[1:]}) == 3
                and all(p['heldout_block_id'] is not None for p in populations[1:]),
                'Missing/duplicate full or whole-block fold')
        for population in populations:
            exposure = population['exposure']; reason = None
            if not exposure['any_G2_tail_positive_peer_exposure']:
                reason = 'MISSING_G2_TAIL_POSITIVE_PROCESSED_PEER_EXPOSURE'
            elif not exposure['same_group_n0_and_positive_observed']:
                reason = 'G2_TAIL_POSITIVE_PRESENT_BUT_SAME_NUISANCE_GROUP_N0_POSITIVE_SUPPORT_MISSING'
            if reason:
                gaps.append(dict(heldout_block_id=population['heldout_block_id'], reason=reason))
        missing_positive = any(g['reason'].startswith('MISSING_G2') for g in gaps)
        candidate = bool(gaps)
        if gaps and not missing_positive:
            status = 'HOLD_G2_TAIL_SAME_GROUP_SUPPORT_GAP_REQUIRES_ATTRIBUTION'
        elif missing_positive and not resource_chain_complete_and_rederived:
            status = 'HOLD_MISSING_G2_TAIL_EXPOSURE_BUT_KERNEL_RESOURCE_CHAIN_NOT_COMPLETE'
        elif missing_positive:
            status = 'CONDITIONAL_DESIGN_NEEDED_REQUIRES_NEW_NATIVE_PRODUCER_AND_INDEPENDENT_REVIEW'
        elif any('zero_point' not in p or 'finite_predeclared_probe' not in p for p in populations):
            candidate = None
            status = 'HOLD_G2_TAIL_EXPOSURE_PRESENT_NUMERICAL_SUPPORT_UNASSESSED'
        elif any(not p[key]['joint_full_column_rank'] for p in populations
                 for key in ('zero_point', 'finite_predeclared_probe')):
            candidate = None
            status = 'HOLD_FULL_OR_FOLD_RANK_GAP_REQUIRES_ATTRIBUTION_POSSIBLY_WEAK_G2_EXPOSURE'
        else:
            candidate = False
            status = 'HOLD_TARGETED_NOT_NEEDED_AT_REPORTED_FULL_AND_FOLD_SUPPORT_DIAGNOSTICS'
    return dict(schema=SCHEMA+'.decision', status=status, executable=False,
        targeted_acquisition_candidate=candidate, population_support_gaps=gaps,
        native_ready=False, GPU_started=False, automatic_fit=False, numerical_fit_invoked=False,
        activation_authenticated=False, final_gain_eligible=False)


def dependency_probe():
    """Metadata-only canonical-final availability; never diagnoses partial labels."""
    return dict(schema=SCHEMA+'.dependency_probe', matching_final_path=str(MATCHING_FINAL),
        matching_final_exists=MATCHING_FINAL.is_file(), matching_required_status=MATCHING_STATUS,
        resource_final_path=str(RESOURCE_FINAL), resource_final_exists=RESOURCE_FINAL.is_file(),
        resource_required_status=RESOURCE_STATUS, completed_native_evidence_rederived=False,
        partial_matching_labels_read=False, activation_authenticated=False, executable=False,
        status='HOLD_NO_NATIVE_AUTHENTICATION_OR_EXECUTOR_IN_CPU_CORE', GPU_started=False,
        numerical_fit_invoked=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['calendar', 'dependencies'])
    args = parser.parse_args()
    result = calendar_contract() if args.action == 'calendar' else dependency_probe()
    print(json.dumps(result, indent=2, allow_nan=False))
