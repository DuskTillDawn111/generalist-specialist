"""Full paired context-risk chronology with causal lifecycle splitting.

All source SAVE events and empty frontiers remain. This predicts physical SAVE
envelopes, not active-service replacement or an invented no-competitor quantum.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
import gzip
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import context_intervals
from .owner_lifecycle_mark_v21 import (
    FIELDS, RESIDENTS, IntervalIndex, source_prefix_rows, states_at, receipt)
from .idle_owner_selection_audit_v20 import read
from .profiling import digest
from .save_hazard_v13 import split_exposure, fit, evaluate

SCHEMA = 'gsdeploy.full_context_lifecycle_SAVE_hazard.v21'


def lifecycle_features(holder, frontier, states):
    values = {'holder:' + holder}
    if not frontier:
        values.add('frontier:empty')
    for owner, task, pool, phases in frontier:
        values.update(('pending_owner:' + owner, 'task:' + task, 'pool:' + pool))
        values.update('phase:' + phase for phase in phases)
    for owner in RESIDENTS:
        for field in FIELDS:
            value = getattr(states[owner], field)
            values.add(f'lifecycle:{owner}:{field}:' + ('unknown' if value is None else str(value).lower()))
    return tuple(sorted(values))


def aggregate(supervision_path, *, risk_stream=None):
    mark_rows, sources, prefix = source_prefix_rows(supervision_path, include_indexes=True)
    supervision = read(supervision_path)
    original_labels = {(member['row']['capture'], member['row']['held_owner'],
        member['row']['current_restore_start_ns'], member['row']['save_end_ns']) for member in mark_rows}
    preserved_labels, full_events = set(), set()
    bins = defaultdict(lambda: [0, 0, 0])
    counts = Counter()
    envelopes = defaultdict(set)
    ages, vocab, exact = [], set(), set()
    state_examples = []
    chronology = []
    event_ledger = {}
    by_capture = defaultdict(Counter)

    for audit in supervision['training_audits']:
        capture = audit['raw']['sha256']
        raw, report = read(audit['raw']['path']), read(audit['report']['path'])
        known = {row['request_id']:row for row in raw['request_records']}
        parsed_ref = next(ref for ref in report['sources'] if Path(ref['path']).name == 'parsed-measured.json')
        sql_ref = next(ref for ref in report['sources'] if Path(ref['path']).suffix == '.sqlite')
        parsed = read(parsed_ref['path'])
        windows = defaultdict(list)
        for phase in parsed['phases']:
            row = known[phase['request_id']]
            windows[row['profile_job_id']].append((phase['nsys_start_ns'], phase['nsys_end_ns']))
        ranges = defaultdict(list)
        for row in read(report['kernel_ready_observations']['path'])['rows']:
            request = known[row['request_id']]
            begin = row['observed_start_ns']
            if row['certified_ready_bound_ns'] is not None:
                begin = min(begin, row['certified_ready_bound_ns'])
            ranges[(request['profile_job_id'], request['model'], request['task'], request['pool_id'], row['phase'])].append((begin,row['observed_end_ns']))
        frontiers = defaultdict(list)
        for (job, owner, task, pool, phase), spans in ranges.items():
            frontiers[job].append((owner, task, pool, phase, IntervalIndex(spans)))
        with sqlite3.connect(Path(sql_ref['path']).as_uri() + '?mode=ro',uri=True) as connection:
            connection.row_factory = sqlite3.Row
            _, brackets, issues = context_intervals(connection)
        mapping = {row['alias']:tuple(row['context'])
            for row in report['controlled_temporal_attribution']['creation_windows']}
        paired = sorted((begin,end,owner) for owner, context in mapping.items() for begin,end in brackets[context])
        if any(left[1] > right[0] for left,right in zip(paired,paired[1:])):
            raise ValueError('Overlapping paired context envelopes')
        issue_times = sorted(row['timestamp'] for row in issues)
        index = prefix[capture]
        state_cuts = sorted({clock for group in index.values() for interval in group.values()
            for span in interval.spans for clock in span})

        for job, spans in sorted(windows.items()):
            lo, hi = min(begin for begin,_ in spans), max(end for _,end in spans)
            job_frontiers = frontiers[job]
            ready_cuts = sorted({clock for *_, interval in job_frontiers for span in interval.spans for clock in span})
            cuts = sorted(set([lo,hi,
                *state_cuts[bisect_right(state_cuts,lo):bisect_left(state_cuts,hi)],
                *ready_cuts[bisect_right(ready_cuts,lo):bisect_left(ready_cuts,hi)]]))
            coverage, qualified_risk = 0, 0

            def frontier_at(clock):
                current = defaultdict(list)
                for owner, task, pool, phase, interval in job_frontiers:
                    if interval.contains(clock):
                        current[owner,task,pool].append(phase)
                return [[owner,task,pool,sorted(phases)] for (owner,task,pool),phases in sorted(current.items())]

            for paired_index,(begin,end,owner) in enumerate(paired):
                start,stop = max(begin,lo),min(end,hi)
                if start >= stop:
                    continue
                width = stop-start
                coverage += width
                # A pairing issue inside this envelope means its holder/risk
                # cannot be certified. Preserve its duration in chronology;
                # never silently treat it as ordinary censoring/exposure.
                if bisect_right(issue_times,end) > bisect_left(issue_times,begin):
                    counts['unresolved_envelope_duration_ns'] += width
                    counts['unresolved_envelope_clips'] += 1
                    continue
                qualified_risk += width
                envelope_id = digest([capture,owner,begin,end])
                envelopes[capture].add(envelope_id)
                envelope_cuts = [start,*cuts[bisect_right(cuts,start):bisect_left(cuts,stop)],stop]
                expected_event = end < hi and end == stop
                event_key = (capture,owner,begin,end)
                if expected_event:
                    full_events.add(event_key)
                    if event_key in original_labels:
                        preserved_labels.add(event_key)
                    next_pair = paired[paired_index+1] if paired_index+1 < len(paired) else None
                    if event_key in original_labels:
                        absent_reason = None
                    elif next_pair is None:
                        absent_reason = 'NO_NEXT_PAIRED_RESTORE'
                    elif next_pair[0] >= hi:
                        absent_reason = 'NEXT_RESTORE_OUTSIDE_CURRENT_JOB_WINDOW'
                    elif bisect_right(issue_times,next_pair[0]) > bisect_left(issue_times,begin):
                        absent_reason = 'UNPAIRED_EVENT_BEFORE_NEXT_RESTORE_TARGET_UNQUALIFIED'
                    else:
                        absent_reason = 'UNRESOLVED_SUPERVISION_DIFFERENCE'
                    event_ledger[event_key] = dict(capture=capture,held_owner=owner,
                        restore_start_ns=begin,save_end_ns=end,job_id=job,
                        original_mark_label_present=event_key in original_labels,
                        missing_from_mark_reason=absent_reason,
                        source_clock='common nsys',target='SAVE_END',age_origin='RESTORE_START',
                        current_restore_precedes_job=begin < lo)
                original_save_ns, original_events = 0, 0
                for left,right in zip(envelope_cuts,envelope_cuts[1:]):
                    frontier = frontier_at(left)
                    states = states_at(index,left,frontier)
                    fs = lifecycle_features(owner,frontier,states)
                    observed = expected_event and right == stop
                    episode = dict(age_entry_ns=left-begin,age_exit_ns=right-begin,save_observed=observed)
                    width = right-left
                    by_capture[capture]['risk_intervals'] += 1
                    by_capture[capture]['exposure_ns'] += width
                    by_capture[capture]['events'] += observed
                    counts['risk_intervals'] += 1
                    counts['exposure_ns'] += width
                    counts['physical_SAVE_events'] += observed
                    counts['right_censored_subintervals'] += not observed
                    counts['empty_frontier_intervals'] += not frontier
                    counts['empty_frontier_exposure_ns'] += width if not frontier else 0
                    counts['empty_frontier_SAVE_events'] += observed and not frontier
                    eligible = {alias:states[alias].active_service_eligible for alias in RESIDENTS}
                    alternative = [alias for alias in RESIDENTS if alias != owner and eligible[alias] is True]
                    no_alternative = not alternative and all(eligible[alias] is False for alias in RESIDENTS if alias != owner)
                    counts['no_active_alternative_intervals'] += no_alternative
                    counts['no_active_alternative_exposure_ns'] += width if no_alternative else 0
                    counts['physical_SAVE_without_active_alternative'] += observed and no_alternative
                    if observed:
                        event_ledger[event_key]['empty_frontier_on_predictable_left_interval'] = not frontier
                        event_ledger[event_key]['active_alternatives_on_predictable_left_interval'] = alternative
                    if risk_stream is not None:
                        risk_stream.write(json.dumps(dict(capture=capture,job_id=job,
                            held_owner=owner,restore_start_ns=begin,start_ns=left,end_ns=right,
                            age_entry_ns=left-begin,age_exit_ns=right-begin,
                            physical_SAVE_END_observed=observed,frontier=frontier,
                            causal_left_features=fs),separators=(',',':'))+'\n')
                    ages.append((left-begin,right-begin))
                    vocab.update(fs)
                    exact.add(digest([owner,frontier]))
                    for age_bin,exposure,event in split_exposure(episode):
                        target = bins[capture,age_bin,fs]
                        target[0] += exposure
                        target[1] += event
                        target[2] += 1
                        original_save_ns += exposure
                        original_events += event
                    if observed and no_alternative and len(state_examples) < 8:
                        state_examples.append(dict(capture=capture,job=job,held_owner=owner,
                            clock_ns=right, interval_start_ns=left, frontier=frontier,
                            lifecycle={alias:{field:getattr(states[alias],field) for field in FIELDS} for alias in RESIDENTS}))
                assert original_save_ns == stop-start
                assert original_events == int(expected_event)
            if coverage > hi-lo:
                raise ValueError('Paired context chronology overlaps in job window')
            chronology.append(dict(capture=capture,job_id=job,start_ns=lo,end_ns=hi,
                paired_context_duration_ns=coverage, qualified_risk_duration_ns=qualified_risk,
                context_gap_duration_ns=hi-lo-coverage))
    rows = [dict(capture=c,age_bin=i,features=list(fs),exposure_ns=v[0],events=v[1],
        episode_bin_intersections=v[2]) for (c,i,fs),v in sorted(bins.items())]
    assert sum(row['exposure_ns'] for row in rows) == counts['exposure_ns']
    assert sum(row['events'] for row in rows) == counts['physical_SAVE_events']
    support = []
    for entry,exit in sorted(ages):
        if support and entry <= support[-1][1]:support[-1][1] = max(support[-1][1],exit)
        else:support.append([entry,exit])
    missing = sorted(original_labels-preserved_labels)
    return dict(rows=rows,counts=dict(counts),capture_envelopes={c:len(values) for c,values in envelopes.items()},
        vocabulary=sorted(vocab),exact_frontier_keys=sorted(exact),age_support_ns=support,
        age_bin_exposure_ns={str(i):sum(row['exposure_ns'] for row in rows if row['age_bin']==i)
            for i in sorted({row['age_bin'] for row in rows})},
        source_receipts=sources, original_marked_SAVE_labels=len(original_labels),
        original_marked_SAVE_labels_preserved=len(preserved_labels),missing_marked_SAVE_labels=missing,
        additional_unmarked_physical_SAVE_events=len(full_events-original_labels),
        job_chronology=chronology,physical_SAVE_without_active_alternative_examples=state_examples,
        event_ledger=list(event_ledger.values()),counts_by_capture={c:dict(v) for c,v in by_capture.items()},
        risk_identifies_active_replacement_timing=False,
        inactive_target_labels_relabelled_or_deleted=False)


def subset(data,capture,include):
    rows = [row for row in data['rows'] if (row['capture']==capture)==include]
    interval_n = sum(value['risk_intervals'] for key,value in data['counts_by_capture'].items() if (key==capture)==include)
    counts = dict(episodes=interval_n,episode_bin_intersections=sum(row['episode_bin_intersections'] for row in rows),
        events=sum(row['events'] for row in rows),exposure_ns=sum(row['exposure_ns'] for row in rows))
    return dict(data,rows=rows,counts=counts,
        capture_envelopes={key:value for key,value in data['capture_envelopes'].items() if (key==capture)==include})


def run(supervision_path,output):
    output = Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    from . import owner_lifecycle_mark_v21
    implementation_before = [receipt(__file__),receipt(owner_lifecycle_mark_v21.__file__)]
    risk_path = output/'risk-chronology.jsonl.gz'
    with gzip.open(risk_path,'wt',encoding='utf-8') as stream:
        data = aggregate(supervision_path,risk_stream=stream)
    declaration = read('artifacts/predictor-save-hazard-v13-20261028-training01/declaration.json')
    old_ref = declaration['source_model']
    if receipt(old_ref['path']) != old_ref:raise ValueError('Old training risk source changed')
    old_model = read(old_ref['path'])
    old_events = {(row['capture'],row['held_owner'],row['original_envelope_start_ns'],row['original_envelope_end_ns'])
        for row in old_model['episodes'] if row['save_observed']}
    del old_model
    old_missing_reasons = Counter()
    for row in data['event_ledger']:
        key=(row['capture'],row['held_owner'],row['restore_start_ns'],row['save_end_ns'])
        row['old_nonempty_frontier_hazard_event_present'] = key in old_events
        row['missing_from_old_hazard_reason'] = (None if key in old_events else
            'EMPTY_READY_FRONTIER_INTERVAL_OMITTED' if row['empty_frontier_on_predictable_left_interval'] else
            'UNRESOLVED_OLD_HAZARD_DIFFERENCE')
        old_missing_reasons[str(row['missing_from_old_hazard_reason'])] += 1
    (output/'physical-SAVE-event-ledger.json').write_text(json.dumps(data.pop('event_ledger'),indent=2)+'\n')
    data['risk_chronology'] = receipt(risk_path)
    data['physical_SAVE_event_ledger'] = receipt(output/'physical-SAVE-event-ledger.json')
    data['old_hazard_event_count'] = len(old_events)
    data['old_hazard_missing_reason_counts'] = dict(old_missing_reasons)
    if data['missing_marked_SAVE_labels']:
        (output/'incomplete-reconstruction.json').write_text(json.dumps(data,indent=2)+'\n')
        raise ValueError('Original marked SAVE chronology not completely recovered')
    # v13 fixed full-exposure Poisson likelihood and regularization, now with
    # empty-frontier and lifecycle covariates. No validation-based selection.
    data['counts']['episodes'] = data['counts']['risk_intervals']
    data['counts']['events'] = data['counts']['physical_SAVE_events']
    folds = []
    for capture in sorted(data['capture_envelopes']):
        train,test = subset(data,capture,False),subset(data,capture,True)
        model = fit(train,'age_additive')
        folds.append(dict(heldout_training_capture=capture,optimizer=model['optimizer'],
            training_counts=train['counts'], heldout=evaluate(model,test)))
    model = fit(data,'age_additive')
    for ref in [*data['source_receipts'], *implementation_before]:
        if receipt(ref['path']) != ref:raise ValueError('Training source bytes changed')
    (output/'sufficient-statistics.json').write_text(json.dumps(data,indent=2)+'\n')
    model = dict(model,lifecycle_schema=SCHEMA,implementation=receipt(__file__),
        source_statistics=receipt(output/'sufficient-statistics.json'),
        prediction_target='Physical SAVE envelope hazard; active replacement is a distinct unqualified target',
        empty_frontier_query_supported=True,
        service_timing_or_SM_capacity_identified=False)
    (output/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    report = dict(schema=SCHEMA,status='SOURCE_PREFIX_FULL_RISK_TRAINING_DIAGNOSTIC',
        counts=data['counts'], original_marked_SAVE_labels=data['original_marked_SAVE_labels'],
        original_marked_SAVE_labels_preserved=data['original_marked_SAVE_labels_preserved'],
        missing_marked_SAVE_labels=data['missing_marked_SAVE_labels'],
        additional_unmarked_physical_SAVE_events=data['additional_unmarked_physical_SAVE_events'],
        old_hazard_event_count=data['old_hazard_event_count'],
        old_hazard_missing_reason_counts=data['old_hazard_missing_reason_counts'],
        risk_chronology=data['risk_chronology'],physical_SAVE_event_ledger=data['physical_SAVE_event_ledger'],
        source_receipts_before=[*data['source_receipts'],*implementation_before],
        source_receipts_after=[receipt(ref['path']) for ref in [*data['source_receipts'],*implementation_before]],
        job_chronology=data['job_chronology'],holdout_folds=folds,full_fit_optimizer=model['optimizer'],
        model=receipt(output/'model.json'),statistics=model['source_statistics'],
        physical_SAVE_without_active_alternative_examples=data['physical_SAVE_without_active_alternative_examples'],
        physical_SAVE_without_active_alternative_retained=True,
        active_transition_events_automatically_censored=False,
        active_replacement_oracle_available=False, GPU_started=False, validation_labels_used=False,
        hazard_target='SAVE_END; attained age from original RESTORE_START',
        feature_cutoff='Predictable left endpoint of each risk interval; only already emitted phase/API/handler/ready events. Future END values are absent from feature vectors.',
        independent_validation_accuracy=False, final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.supervision,args.output)
    print(json.dumps({key:result[key] for key in ('counts','original_marked_SAVE_labels_preserved',
        'additional_unmarked_physical_SAVE_events','holdout_folds','full_fit_optimizer')},indent=2))
