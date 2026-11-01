"""Causal owner lifecycle state and a separate active-service mark candidate.

Physical next-context probabilities retain every class. Active reference-service
selection additionally requires visible owned work, never merely residency.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from .context_next_owner_v14 import (
    RESIDENTS, features as old_features, feature_names as old_names,
    fit_matrix, predict_matrix, previous_history, query_from_row)
from .idle_owner_selection_audit_v20 import read
from .profiling import digest

FIELDS = ('handler_active', 'issued_outstanding', 'gpu_ready')
SCHEMA = 'gsdeploy.causal_owner_lifecycle_active_service_mark.v21'


def receipt(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return dict(path=str(path),sha256=hasher.hexdigest())


@dataclass(frozen=True)
class OwnerLifecycle:
    handler_active: bool | None
    issued_outstanding: bool | None
    gpu_ready: bool | None

    def __post_init__(self):
        if any(value is not None and type(value) is not bool for value in asdict(self).values()):
            raise ValueError('Lifecycle facts are explicit true/false/unknown')
        if self.gpu_ready is True and self.issued_outstanding is False:
            raise ValueError('Ready GPU work cannot be certified unissued')

    @property
    def active_service_eligible(self):
        # This is an owned-work domain, not a ready-frontier-only restriction.
        if any(getattr(self, key) is True for key in FIELDS):
            return True
        if any(getattr(self, key) is None for key in FIELDS):
            return None
        return False


def state_features(states, owners=RESIDENTS):
    if set(states) != set(owners):
        raise ValueError('Complete resident lifecycle snapshot required')
    values = []
    for owner in owners:
        state = states[owner]
        if not isinstance(state, OwnerLifecycle):
            raise ValueError('Typed causal lifecycle snapshot required')
        for field in FIELDS:
            value = getattr(state, field)
            values.extend((float(value is True), float(value is None)))
    return values


def feature_names(pools):
    return [*old_names(pools), *[f'{field}:{owner}:{kind}' for owner in RESIDENTS
        for field in FIELDS for kind in ('true', 'unknown')]]


def features(query, states, pools):
    ready={item[0] for item in query['pending_frontier']}
    if any(states[owner].gpu_ready is not (owner in ready) for owner in RESIDENTS):
        raise ValueError('Lifecycle ready state and current GPU frontier disagree')
    return [*old_features(query, pools), *state_features(states)]


def engine_states(owners, running, active):
    """Settled engine state only: unreleased DAG nodes are NOT issued work.

    A running handler proves owned work even while its ready frontier is empty.
    With no handler and no active activity the engine owns no work. A running
    handler without active activities has UNKNOWN issuance because this light
    DAG does not expose the physical CUDA issue queue.
    """
    result = {}
    for owner in owners:
        kinds = [value['kind'] for (alias, _), value in active.items() if alias == owner]
        handler = owner in running
        issued = True if kinds else None if handler else False
        result[owner] = OwnerLifecycle(handler, issued, 'gpu' in kinds)
    return result


class IntervalIndex:
    def __init__(self, spans):
        merged = []
        for begin, end in sorted(spans):
            if end < begin:
                raise ValueError('Reversed lifecycle interval')
            if end == begin:
                continue
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((begin, end))
        self.spans = merged
        self.starts = [begin for begin, _ in merged]

    def contains(self, clock):
        i = bisect_right(self.starts, clock) - 1
        return i >= 0 and clock < self.spans[i][1]


def states_at(index, clock, frontier):
    ready = {item[0] for item in frontier}
    states = {}
    for owner in RESIDENTS:
        handler = (True if index['phase'][owner].contains(clock) or index['certain'][owner].contains(clock)
            else None if index['possible'][owner].contains(clock) else False)
        issued = (True if owner in ready or index['issued'][owner].contains(clock)
            else False if handler is False else None)
        states[owner] = OwnerLifecycle(handler, issued, owner in ready)
    return states


def source_prefix_rows(supervision_path, *, include_indexes=False):
    supervision = read(supervision_path)
    if supervision.get('training_only') is not True or supervision.get('validation_labels_used') is not False:
        raise ValueError('Training-only supervision with no validation labels required')
    rows = supervision['transition_observations']
    history = previous_history(rows)
    indexes, sources = {}, [receipt(supervision_path)]
    for audit in supervision['training_audits']:
        for key in ('raw', 'report'):
            assert receipt(audit[key]['path']) == audit[key]
            sources.append(audit[key])
        raw, report = read(audit['raw']['path']), read(audit['report']['path'])
        known = {row['request_id']: row for row in [*raw.get('priming_request_records', []), *raw['request_records']]}
        sql_ref = next(ref for ref in report['sources'] if Path(ref['path']).suffix == '.sqlite')
        parsed_ref = next(ref for ref in report['sources'] if Path(ref['path']).name == 'parsed-measured.json')
        observation_ref = report['kernel_ready_observations']
        for ref in (sql_ref, parsed_ref, observation_ref):
            assert receipt(ref['path']) == ref
            sources.append(ref)
        parsed = read(parsed_ref['path'])
        exact_phase, certain_handler, possible_handler, issued = (defaultdict(list) for _ in range(4))
        with sqlite3.connect(Path(sql_ref['path']).as_uri() + '?mode=ro', uri=True) as connection:
            for begin, end, text in connection.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'GS_STAGE:%'"):
                match = re.fullmatch(r'GS_STAGE:(.*):([^:]+):([^:]+)', text)
                if match and match[1] in known and end is not None:
                    exact_phase[known[match[1]]['model']].append((begin, end))
        for row in known.values():
            pid = next((event['pid'] for event in row.get('execution_events', []) if 'pid' in event), None)
            mapping = parsed['clock_mappings'].get(str(pid))
            if not mapping or mapping.get('valid') is not True:
                raise UnknownEvidence('SOURCE_HANDLER_CLOCK_MAPPING_REQUIRED')
            offset = mapping['offset_ns']
            response = row['worker_response']
            begin, end = response['start_perf_ns'] - offset, response['end_perf_ns'] - offset
            # Fixed median-clock projection is an explicit observation-domain
            # approximation. Never inspect a future END +/- residual to change
            # state before that END marker occurs in this projected clock.
            # Mapping residual uncertainty remains a source qualification limit.
            certain_handler[row['model']].append((begin,end))
            possible_handler[row['model']].append((begin,end))
        for row in read(observation_ref['path'])['rows']:
            # Prefix: API-return marker has occurred, kernel-END marker has not.
            # A later kernel START is never a current feature or requirement.
            if row['issue_end_ns'] is not None and row['issue_end_ns'] < row['observed_end_ns']:
                issued[known[row['request_id']]['model']].append((row['issue_end_ns'], row['observed_end_ns']))
        indexes[audit['raw']['sha256']] = {
            'phase': {owner: IntervalIndex(exact_phase[owner]) for owner in RESIDENTS},
            'certain': {owner: IntervalIndex(certain_handler[owner]) for owner in RESIDENTS},
            'possible': {owner: IntervalIndex(possible_handler[owner]) for owner in RESIDENTS},
            'issued': {owner: IntervalIndex(issued[owner]) for owner in RESIDENTS}}
    extracted = []
    for row, prior in zip(rows, history):
        index, clock = indexes[row['capture']], row['save_end_ns']
        states = states_at(index, clock, row['pending_frontier'])
        extracted.append(dict(row=row, query=query_from_row(row, prior), states=states))
    return (extracted, sources, indexes) if include_indexes else (extracted, sources)


def probabilities(model, query, states):
    names = feature_names(model['pool_tasks'])
    if model.get('schema') != SCHEMA or model['feature_names'] != names:
        raise ValueError('Explicit lifecycle model schema and feature order required')
    matrix = np.asarray([features(query, states, model['pool_tasks'])])
    raw = predict_matrix(model['fitted'], matrix)[0]
    fitted = model['fitted']
    outside = np.flatnonzero((matrix[0] < np.asarray(fitted['feature_minimum']) - 1e-12)
        | (matrix[0] > np.asarray(fitted['feature_maximum']) + 1e-12))
    flags = ['FEATURE_OUTSIDE_TRAINING_RANGE:' + names[i] for i in outside]
    if state_signature(query, states) not in model.get('lifecycle_support_signatures', []):
        flags.append('HELD_FRONTIER_HISTORY_LIFECYCLE_COMBINATION_UNSEEN')
    eligibility = {owner: states[owner].active_service_eligible for owner in RESIDENTS}
    # Physical mark is never silently filtered. The active-service target has
    # a distinct semantic output; unresolved work identity remains UNKNOWN.
    if any(value is None for value in eligibility.values()):
        status, active, chosen = 'UNKNOWN_OWNER_WORK_DOMAIN', None, None
    else:
        mass = sum(raw[i] for i, owner in enumerate(RESIDENTS) if eligibility[owner])
        if mass <= 0:
            status, active, chosen = 'NO_VISIBLE_ACTIVE_SERVICE_TARGET', None, None
        else:
            active = {owner: float(raw[i] / mass) if eligibility[owner] else 0.
                for i, owner in enumerate(RESIDENTS)}
            chosen = max(active, key=active.get)
            status = 'UNCALIBRATED_ACTIVE_REFERENCE_SERVICE_MARK'
    return dict(status=status, physical_context_owner_probabilities=dict(zip(RESIDENTS, map(float, raw))),
        active_service_owner_probabilities=active, active_service_target=chosen,
        owner_active_service_eligibility=eligibility,
        ood_flags=flags,
        initial_residual_context_modelled=False, background_context_modelled=False,
        service_timing_predicted=False, physical_SM_service_identified=False,
        deployment_qualified=False)


def state_signature(query, states):
    return digest([query['held_owner'], query['previous_held_owner'],
        sorted(query['pending_frontier']),
        [(owner, asdict(states[owner])) for owner in RESIDENTS]])


class LifecycleActiveServiceMark:
    """Explicit bridge for an experimental reference-service context hypothesis.

    It does not claim that active-service selection equals physical CUDA context
    selection. The unfiltered physical probabilities remain in every receipt.
    """
    def __init__(self, model, *, allow_active_service_context_hypothesis=False):
        if allow_active_service_context_hypothesis is not True:
            raise ValueError('Active service and physical context require an explicit hypothesis')
        if (model.get('schema') != SCHEMA or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})):
            raise ValueError('Source-bound lifecycle mark required')
        for ref in [model['implementation'], *model['sources']]:
            if receipt(ref['path']) != ref:
                raise ValueError('Lifecycle mark source changed')
        self.model = model
        self.current_clock_ns = None
        self.states = None

    def observe_view(self, view):
        if not hasattr(view, 'owner_lifecycle'):
            raise UnknownEvidence('SETTLED_OWNER_LIFECYCLE_VIEW_REQUIRED')
        if self.current_clock_ns is not None and view.clock_ns < self.current_clock_ns:
            raise UnknownEvidence('MONOTONIC_OWNER_LIFECYCLE_CLOCK_REQUIRED')
        self.current_clock_ns = view.clock_ns
        self.states = dict(view.owner_lifecycle)
        state_features(self.states)
        ready={node.key[0] for node in view.active_gpu_nodes}
        if any(self.states[owner].gpu_ready is not (owner in ready) for owner in RESIDENTS):
            raise UnknownEvidence('OWNER_LIFECYCLE_READY_FRONTIER_MISMATCH')

    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        if self.states is None:
            raise UnknownEvidence('NO_CAUSAL_LIFECYCLE_SNAPSHOT')
        query = dict(held_owner=held_owner, context_age_ns=context_age_ns,
            pending_frontier=pending_frontier, previous_held_owner=previous_held_owner,
            public_workload=public_workload, pending_age_ns_by_owner=pending_age_ns_by_owner,
            pending_reference_work_ns_by_owner=pending_reference_work_ns_by_owner)
        value = probabilities(self.model, query, self.states)
        if value['active_service_target'] is None:
            raise UnknownEvidence(value['status'])
        return dict(value, most_likely_next_owner=value['active_service_target'],
            next_owner_probabilities=value['active_service_owner_probabilities'],
            source_clock_ns=self.current_clock_ns,
            model_sha256=self.model['model_sha256'],
            active_service_context_hypothesis=True,
            physical_context_probabilities_filtered=False,
            physical_context_replacement_requires_independent_validation=True,
            future_public_workload_used_as_features=False)


class LifecycleSchedulerBridge:
    """Observe the settled view before a pre-existing controller is queried.

    The immutable last view is a rolling-observation boundary, NOT an executable
    checkpoint. Hazard, handoff, DAG progress and controller restore are separate.
    """
    def __init__(self, scheduler, mark):
        self.scheduler, self.mark = scheduler, mark
        if scheduler.oracle.next_owner is not mark:
            raise ValueError('Bridge and controller must share one mark instance')
        self.last_view = None

    def decide(self, view):
        self.mark.observe_view(view)
        self.last_view = view
        return self.scheduler.decide(view)

    def freeze_observation_state(self):
        if self.last_view is None:
            return None
        view = self.last_view
        return dict(clock_ns=view.clock_ns,
            owner_lifecycle={owner:asdict(state) for owner,state in view.owner_lifecycle},
            handler_requests=view.handler_requests, pending_requests=view.pending_requests,
            ready_gpu_nodes=tuple((node.key,node.request_id,node.remaining_reference_work_ns)
                for node in view.active_gpu_nodes),
            is_executable_checkpoint=False, physical_progress_observed=False)

    def provenance(self):
        return dict(self.scheduler.provenance(), owner_lifecycle_bridge=True,
            active_service_context_hypothesis=True,
            lifecycle_model_sha256=self.mark.model['model_sha256'],
            latest_causal_observation=self.freeze_observation_state(),
            scheduler_checkpoint_supported=False)


def metrics(rows, raw, active):
    labels = np.asarray([RESIDENTS.index(row['row']['next_owner']) for row in rows])
    not_pending = np.asarray([not row['row']['next_owner_is_pending'] for row in rows])
    correct = raw.argmax(axis=1) == labels
    chosen = np.asarray(active)
    return dict(n=len(rows), physical_mark_accuracy=float(correct.mean()),
        physical_mark_log_loss=float(-np.log(np.maximum(raw[np.arange(len(rows)), labels], 1e-300)).mean()),
        target_not_pending_n=int(not_pending.sum()),
        target_not_pending_physical_mark_correct=int(correct[not_pending].sum()),
        active_service_target_correct=int((chosen == labels).sum()),
        active_service_target_unknown=int((chosen < 0).sum()),
        active_service_target_accuracy=float((chosen == labels).mean()),
        target_not_pending_active_service_correct=int((chosen[not_pending] == labels[not_pending]).sum()))


def train(supervision_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    extracted, sources = source_prefix_rows(supervision_path)
    # Fixed public task/input schema; input identity is observable at dispatch,
    # labels and observed heldout times do not select features or parameters.
    pools = {item[2]:item[1] for member in extracted for item in member['row']['public_workload']}
    names = feature_names(pools)
    matrix = np.asarray([features(member['query'], member['states'], pools) for member in extracted])
    labels = np.asarray([RESIDENTS.index(member['row']['next_owner']) for member in extracted])
    captures = np.asarray([member['row']['capture'] for member in extracted])
    counts = Counter()
    for member in extracted:
        row, states = member['row'], member['states']
        counts['transitions_retained'] += 1
        counts['target_not_pending_retained'] += not row['next_owner_is_pending']
        counts['target_owned_work_certified'] += states[row['next_owner']].active_service_eligible is True
        counts['target_handler_unknown'] += states[row['next_owner']].handler_active is None
        counts['target_handler_false'] += states[row['next_owner']].handler_active is False
    folds = []
    for capture in sorted(set(captures)):
        training, held = captures != capture, captures == capture
        fitted = fit_matrix(matrix[training], labels[training], list(captures[training]), names)
        raw = predict_matrix(fitted, matrix[held])
        members = [member for member, selected in zip(extracted, held) if selected]
        active = []
        for member, probs in zip(members, raw):
            allowed = [member['states'][owner].active_service_eligible for owner in RESIDENTS]
            if any(value is None for value in allowed) or not any(allowed):
                active.append(-1)
            else:
                active.append(int(np.argmax(np.where(allowed, probs, -1.))))
        folds.append(dict(heldout_training_capture=capture,
            training_captures=sorted(set(captures[training])),
            optimizer=fitted['optimizer'], **metrics(members, raw, active)))
    fitted = fit_matrix(matrix, labels, list(captures), names)
    model = dict(schema=SCHEMA, training_only=True, pool_tasks=pools,
        feature_names=names, fitted=fitted, implementation=receipt(__file__), sources=sources,
        initial_residual_context_model=None, background_context_model=None,
        physical_context_classes=list(RESIDENTS), future_public_arrival_features=False,
        lifecycle_support_signatures=sorted({state_signature(member['query'], member['states']) for member in extracted}),
        model_status='UNCALIBRATED_LIFECYCLE_MARK_CANDIDATE')
    model['model_sha256'] = digest(model)
    report = dict(schema=SCHEMA, status='TRAINING_AND_GROUPED_HOLDOUT_DIAGNOSIS_ONLY',
        counts=dict(counts), holdout_folds=folds, training_optimizer=fitted['optimizer'], sources=sources,
        feature_names=names, GPU_started=False, validation_labels_used=False,
        all_physical_mark_classes_retained=True, ready_only_target_filter=False,
        residual_or_background_context_fabricated=False,
        independent_validation_accuracy=False, final_gain_eligible=False,
        source_clock_uncertainty='Worker lifecycle events use fixed median clock projection; mapping residual uncertainty remains uncalibrated. No future END-residual boundary affects prefix state; exact common-clock NVTX phase is additional active evidence.',
        issued_state_limits='Measured returned-kernel launch witness, otherwise unknown inside a handler; light DAG does not expose all issued CUDA operations',
        holdout_limits='Two training captures, shared input/job templates and correlated adjacent transitions; capture holdout is a training diagnostic, not independent request validation or a calibrated probability claim')
    output.mkdir(parents=True)
    (output/'model.json').write_text(json.dumps(model, indent=2)+'\n')
    report['model'] = receipt(output/'model.json')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = train(args.supervision, args.output)
    print(json.dumps({key: result[key] for key in ('counts', 'holdout_folds', 'training_optimizer')}, indent=2))
