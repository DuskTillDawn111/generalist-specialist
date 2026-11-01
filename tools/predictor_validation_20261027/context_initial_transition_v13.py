"""Extract source-observed initial context states and marked handoff labels.

This is training supervision and support diagnosis, not an availability oracle.
RESTORE_START and a later owned kernel provide only a service-ready bracket.
Exact-cell coverage is reported as a diagnostic, not a prohibition on learning
continuous-age/remaining-work predictors and testing their extrapolation.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import IntervalUnion, context_intervals
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest
from .training_source_guard_v11 import availability_training_sources

SCHEMA = 'gsdeploy.initial_and_marked_context_training_supervision.v13'


def holder_at(intervals, starts, clock_ns):
    if type(clock_ns) is not int or clock_ns < 0:
        raise ValueError('Nonnegative integer source context clock required')
    index = bisect_right(starts, clock_ns) - 1
    if index >= 0 and clock_ns < intervals[index][1]:
        return intervals[index]
    return None


def next_kernel_bound(starts, restore_ns, save_ns):
    """A source witness bounds service readiness above, never identifies it."""
    if (type(restore_ns) is not int or type(save_ns) is not int
            or not 0 <= restore_ns < save_ns):
        raise ValueError('Positive integer paired context envelope required')
    index = bisect_left(starts, restore_ns)
    upper = starts[index] if index < len(starts) and starts[index] < save_ns else None
    return dict(lower_ns=restore_ns, upper_ns=upper,
        meaning='Context service-ready time lies no earlier than RESTORE_START and no later than this observed owned kernel START, when present; SM service is not identified.',
        exact=False, upper_witness='first_owned_measured_kernel_START' if upper is not None else None)


def interval_contains(index, clock):
    i = bisect_right(index.starts, clock) - 1
    return i >= 0 and clock < index.ends[i]


def support_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[digest(key(row))].append(row)
    group_records = []
    covered = 0
    for signature, members in sorted(groups.items()):
        runs = sorted({r['capture'] for r in members})
        covered += len(members) if len(runs) >= 2 else 0
        group_records.append(dict(signature=signature, observations=len(members),
            captures=runs, independent_captures=len(runs), labels=dict(Counter(
                r.get('next_owner', r.get('held_owner', 'UNKNOWN')) for r in members))))
    return dict(groups=group_records, groups_with_two_captures=sum(g['independent_captures'] >= 2 for g in group_records),
        observations_with_two_capture_exact_group=covered, observations=len(rows),
        exact_matching_is_diagnostic_not_final_predictor=True)


def extract(report_references):
    audits = availability_training_sources(report_references)
    initial, transitions, rejects = [], [], []
    counts = Counter()
    sources = [receipt(__file__), receipt(Path(__file__).with_name('context_envelope_diagnostic_v8.py')),
        receipt(Path(__file__).with_name('training_source_guard_v11.py'))]
    for audit in audits:
        report = checked(audit['report'])
        raw, plan = checked(audit['raw']), checked(audit['plan'])
        capture = audit['raw']['sha256']
        observations = checked(report['kernel_ready_observations'])['rows']
        raw_rows = {r['request_id']:r for r in raw['request_records']}
        plan_jobs = {j['id']:j for j in plan['jobs']}
        jobs = defaultdict(list)
        for row in raw_rows.values():
            jobs[row['profile_job_id']].append(row)
        sqls = [r for r in report['sources'] if Path(r['path']).suffix == '.sqlite']
        parseds = [r for r in report['sources'] if Path(r['path']).name == 'parsed-measured.json']
        if len(sqls) != 1 or len(parseds) != 1:
            raise ValueError('Unique common-session context SQLite and measured phase windows required')
        sql, parsed_ref = sqls[0], parseds[0]
        if receipt(sql['path']) != sql:
            raise ValueError('Changed common-session SQLite')
        with sqlite3.connect(Path(sql['path']).as_uri() + '?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            _, brackets, issues = context_intervals(connection)
        mapping = {row['alias']:tuple(row['context'])
            for row in report['controlled_temporal_attribution']['creation_windows']}
        if report['controlled_temporal_attribution']['status'] != 'UNIQUE_CONTROLLED_TEMPORAL_CORRESPONDENCE':
            raise ValueError('Unique controlled context/owner correspondence required')
        intervals = sorted((a, b, owner) for owner, context in mapping.items() for a, b in brackets[context])
        starts = [a for a, b, owner in intervals]
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError('Overlapping context envelopes cannot identify one holder')
        issue_times = sorted(issue['timestamp'] for issue in issues)
        def unresolved(a, b):
            return bisect_right(issue_times, b) > bisect_left(issue_times, a)
        phase_ranges = defaultdict(list)
        kernel_starts = defaultdict(list)
        by_request_kernel_rows = defaultdict(list)
        for row in observations:
            rid = row['request_id']
            if rid not in raw_rows:
                raise ValueError('Foreign request in measured source-ready observations')
            ready, begin, end = row['certified_ready_bound_ns'], row['observed_start_ns'], row['observed_end_ns']
            if any(type(t) is not int for t in (begin, end)) or begin >= end:
                raise ValueError('Positive integer source GPU kernel interval required')
            if ready is not None and type(ready) is not int:
                raise ValueError('Integer source ready upper bound required')
            phase_ranges[rid, row['phase']].append((min(ready, begin) if ready is not None else begin, end))
            kernel_starts[raw_rows[rid]['model']].append(begin)
            by_request_kernel_rows[rid].append(row)
        indexes = {key:IntervalUnion(value) for key, value in phase_ranges.items()}
        kernel_starts = {owner:sorted(value) for owner, value in kernel_starts.items()}
        parsed = checked(parsed_ref)
        request_windows = defaultdict(list)
        for phase in parsed['phases']:
            request_windows[phase['request_id']].append((phase['nsys_start_ns'], phase['nsys_end_ns']))
        for job_id, members in jobs.items():
            job = plan_jobs[job_id]
            public_workload = sorted([[r['model'], r['task'], r['pool_id'],
                round(job['start_offsets_ms'][r['model']] * 1_000_000)] for r in members])
            declared_workload = sorted([[alias, job['tasks'][alias], pool,
                round(job['start_offsets_ms'][alias] * 1_000_000)] for alias, pool in job['inputs'].items()])
            if public_workload != declared_workload:
                raise ValueError('Measured job cohort does not equal the complete public scheduled workload')
            ids = {r['request_id'] for r in members}
            group_indexes = [(raw_rows[rid], phase, index) for (rid, phase), index in indexes.items() if rid in ids]
            cuts = sorted({t for _, _, index in group_indexes for a, b in index.intervals for t in (a, b)})
            if not cuts:
                rejects.append(dict(capture=capture, job_id=job_id, reason='NO_SOURCE_KERNEL_READY_WITNESS'))
                continue
            def frontier(clock):
                pending = defaultdict(list)
                for request, phase, index in group_indexes:
                    if interval_contains(index, clock):
                        pending[request['request_id']].append(phase)
                return sorted([[raw_rows[rid]['model'], raw_rows[rid]['task'], raw_rows[rid]['pool_id'], sorted(phases)]
                    for rid, phases in pending.items()])
            lo = min(a for rid in ids for a, b in request_windows[rid])
            hi = max(b for rid in ids for a, b in request_windows[rid])
            first = cuts[0]
            held = holder_at(intervals, starts, first)
            base = dict(capture=capture, job_id=job_id, job_name=job['name'], public_workload=public_workload,
                clock_ns=first, pending_frontier=frontier(first),
                source_clock_basis='First source DAG-certified queued-or-already-started kernel witness; not an external public arrival clock.')
            if held is None:
                initial.append(dict(base, status='UNKNOWN_INITIAL_CONTEXT_GAP', held_owner=None,
                    restore_start_ns=None, context_age_ns=None, service_ready_bounds=None))
                counts['initial_context_gap'] += 1
            else:
                a, b, owner = held
                if unresolved(a, first):
                    initial.append(dict(base, status='UNKNOWN_INITIAL_CONTEXT_UNPAIRED_EVENT', held_owner=None,
                        restore_start_ns=None, context_age_ns=None, service_ready_bounds=None))
                    counts['initial_unpaired_event'] += 1
                else:
                    initial.append(dict(base, status='OBSERVED_PAIRED_CONTEXT_STATE', held_owner=owner,
                        restore_start_ns=a, context_age_ns=first-a, envelope_save_end_ns=b,
                        service_ready_bounds=next_kernel_bound(kernel_starts.get(owner, []), a, b)))
                    counts['initial_paired'] += 1
                    counts['initial_sole_pending_differs_from_held'] += (
                        len(base['pending_frontier']) == 1 and base['pending_frontier'][0][0] != owner)
            # Extract every complete, observed adjacency in this job window,
            # including a holder whose RESTORE predates the request window.
            first_index = max(0, bisect_right(starts, lo)-1)
            for index in range(first_index, len(intervals)-1):
                a, b, owner = intervals[index]
                if b >= hi:
                    break
                if b < lo:
                    continue
                na, nb, target = intervals[index+1]
                if na >= hi:
                    counts['transition_next_restore_outside_job'] += 1
                    continue
                if unresolved(a, na):
                    counts['transition_unpaired_event_excluded'] += 1
                    continue
                if na < b:
                    raise ValueError('Negative source context handoff gap')
                state = frontier(b)
                if not state:
                    counts['transition_empty_pending_frontier'] += 1
                entry = max(a, lo, cuts[max(0, bisect_right(cuts, b)-1)])
                # Cuts include a witness at b. A zero-width terminal feature
                # state is retained as a label but not as an at-risk interval.
                bounds = next_kernel_bound(kernel_starts.get(target, []), na, nb)
                upper = bounds['upper_ns']
                clean_upper = upper is not None and not unresolved(na, upper)
                if not clean_upper:
                    bounds = dict(bounds, upper_ns=None, upper_witness=None)
                previous_pending = [r[0] for r in state]
                transitions.append(dict(capture=capture, job_id=job_id, job_name=job['name'],
                    public_workload=public_workload, held_owner=owner,
                    current_restore_start_ns=a, save_end_ns=b, context_age_at_save_ns=b-a,
                    frontier_segment_start_ns=entry, context_age_at_entry_ns=entry-a,
                    pending_frontier=state, next_owner=target, next_restore_start_ns=na,
                    save_to_next_restore_ns=na-b, next_envelope_save_end_ns=nb,
                    next_service_ready_bounds=bounds,
                    service_delay_after_save_bounds_ns=dict(lower_ns=na-b,
                        upper_ns=(upper-b) if clean_upper else None),
                    next_owner_is_pending=target in previous_pending,
                    held_owner_is_pending=owner in previous_pending,
                    context_crosses_request_start=a < lo,
                    positive_terminal_risk_interval=entry < b,
                    physical_context_switch_cost_identified=False,
                    physical_SM_service_identified=False))
                counts['observed_transitions'] += 1
                counts['transition_service_upper_witness'] += clean_upper
                counts['transition_same_owner_reacquisition'] += owner == target
                counts['transition_holder_not_pending'] += owner not in previous_pending
                counts['transition_target_not_pending'] += target not in previous_pending
                counts['transition_cross_request_restore_preserved'] += a < lo
        sources.extend([audit['report'], audit['raw'], audit['plan'], audit['context'],
            report['kernel_ready_observations'], *report['sources']])
    sources = list({ref['path']:ref for ref in sources}.values())
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Initial/transition source changed during extraction')
    known_initial = [r for r in initial if r['status'] == 'OBSERVED_PAIRED_CONTEXT_STATE']
    initial_support = support_summary(known_initial, lambda r:dict(workload=r['public_workload'], frontier=r['pending_frontier']))
    switch_support = support_summary(transitions, lambda r:dict(workload=r['public_workload'], held=r['held_owner'], frontier=r['pending_frontier']))
    folds = []
    for capture in sorted({r['capture'] for r in initial}):
        trained_initial = {digest([r['public_workload'], r['pending_frontier']]) for r in known_initial if r['capture'] != capture}
        trained_switch = defaultdict(list)
        for row in transitions:
            if row['capture'] != capture and row['positive_terminal_risk_interval']:
                trained_switch[digest([row['public_workload'], row['held_owner'], row['pending_frontier']])].append(row)
        initial_test = [r for r in known_initial if r['capture'] == capture]
        switch_test = [r for r in transitions if r['capture'] == capture]
        exact_switch = age_switch = 0
        for row in switch_test:
            members = trained_switch.get(digest([row['public_workload'], row['held_owner'], row['pending_frontier']]), [])
            exact_switch += bool(members)
            age = row['context_age_at_save_ns']
            age_switch += any(r['context_age_at_entry_ns'] <= age <= r['context_age_at_save_ns'] for r in members)
        folds.append(dict(excluded_capture=capture, training_captures=1,
            initial_known_test=len(initial_test), initial_exact_state_supported=sum(
                digest([r['public_workload'], r['pending_frontier']]) in trained_initial for r in initial_test),
            transition_test=len(switch_test), transition_exact_state_supported=exact_switch,
            transition_same_state_age_range_supported=age_switch,
            independent_validation=False, numeric_accuracy_evaluated=False,
            scope='Leave-training-capture-out feature-support diagnostic, not a rule that prohibits regression extrapolation.'))
    report = dict(schema=SCHEMA, status='TRAINING_SUPERVISION_EXTRACTED_NOT_AN_AVAILABILITY_ORACLE',
        training_only=True, training_audits=audits, sources=sources, counts=dict(counts),
        initial_observations=initial, transition_observations=transitions, rejected=rejects,
        support=dict(initial=initial_support, transitions=switch_support, leave_capture_out=folds),
        identifiable_training_targets=['Paired initial held owner and original RESTORE age at source-ready frontier',
            'Next restored owner after a paired SAVE', 'SAVE_END to next RESTORE_START envelope gap',
            'Interval-censored service readiness between next RESTORE_START and first owned measured kernel START'],
        unavailable_physical_targets=['Exact restore completion/service-ready timestamp',
            'SM occupancy/demand or context-envelope-as-SM-service', 'Unpaired/overlapping/gap context state'],
        candidate_public_or_cutoff_features=['Current held owner and attained context age',
            'Pending owner/task/input identities and semantic phase',
            'Source-bound predicted remaining reference work and observed/predicted waiting ages (must be independently reconstructed; not supplied by future source endpoints)',
            'Complete declared arrival/input suffix and previous observed context history',
            'Public execution setup and request workload; initial held state is stochastic conditional on prior history'],
        next_models=['Cause-specific marked survival/next-owner model with continuous age and current pending-work/age features',
            'Correlated initial holder/restore-age scenario model using public setup and available prior context observations',
            'Interval-censored service-ready delay model, retaining restore-to-first-kernel witness slack'],
        statistical_limitations=['Only two independent common-session training captures; within-capture thousands of transitions are dependent',
            'Exact-state coverage is a diagnostic. Continuous-feature interpolation/extrapolation is a testable statistical hypothesis, not physical identification',
            'Censoring, full source-ready state and prior hidden driver history may affect mark distribution',
            'Source kernel residual lifetimes are contaminated by scheduling and must not be used as predicted SM work features'],
        field_contract=dict(public_workload='[owner_alias, task, pool_id, declared_start_offset_ns] for every job request',
            pending_frontier='[owner_alias, task, pool_id, sorted_semantic_phases] with source queued-or-started unfinished kernel witness',
            context_age='Integer source clock minus original RESTORE_START, preserving request-boundary carryover',
            next_service_ready='Interval-censored label only; lower RESTORE_START, upper first observed mapped owner measured kernel START if available'),
        default_round_robin=False, arbitrary_context_quantum=False, global_request_cap=False,
        service_ready_equals_restore_start=False, sole_pending_replaces_held=False,
        independent_prediction_accuracy_validated=False, validation_labels_used=False, GPU_started=False)
    report['model_sha256'] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    history = json.loads(Path(args.history).read_text())
    refs = [row['source_report'] for row in history['observed_captures']]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = extract(refs)
    save(output/'report.json', report)
    print(json.dumps(dict(status=report['status'], counts=report['counts'],
        initial_exact_two_capture=report['support']['initial']['observations_with_two_capture_exact_group'],
        transition_exact_two_capture=report['support']['transitions']['observations_with_two_capture_exact_group'],
        leave_capture_out=report['support']['leave_capture_out'], report=receipt(output/'report.json'))))


if __name__ == '__main__':
    main()
