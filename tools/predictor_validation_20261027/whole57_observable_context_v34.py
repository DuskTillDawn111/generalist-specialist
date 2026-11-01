"""Whole-capture training of native observable context transition candidates.

Only attained age and held/previous holder enter the model. Native phase
activity and engine handler activity have different missingness and remain
diagnostic labels, never fitted covariates. Future arrivals belong to the queue engine, not to
physical SAVE/next-context labels. Neither model identifies GPU SM service.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .context_next_owner_v14 import fit_matrix, predict_matrix
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .owner_lifecycle_hazard_v24_fit import normalized_fit
from .owner_lifecycle_mark_v21 import receipt
from .native_collect import context_identity
from .profiling import digest
from .save_hazard_v13 import EDGES_NS, design, split_exposure
from .unknown_preflight_archive_v1 import save_synced
from .whole57_shared_observable_risk_v31 import OWNERS, SCHEMA as DATA_SCHEMA, shared_features

SCHEMA = 'gsdeploy.whole57_shared_observable_context_candidates.v34'
ROUTES = ('balanced', 'skewed', 'spill_low', 'spill_high')
PARAMETERS = dict(
    physical_mark='capture_balanced_L2_multinomial_logistic_v14_fixed_optimizer',
    physical_SAVE='capture_event_normalized_piecewise_exponential_Poisson_v24',
    age_edges_ns=list(EDGES_NS),
    hazard_previous_holder_used=False,
    mark_previous_holder_used=True,
    handler_used_as_feature=False,
    handler_diagnostic_only_reason='Native NVTX phase true and engine handler true are not the same observation variable',
    interaction='held_holder_by_log_attained_age',
    hyperparameters_selected_from_holdout=False,
    holdout='leave_one_whole_capture_out_known_input_family',
    independent_GPU_validation=False,
)


def checked(reference):
    if set(reference) != {'path', 'sha256'} or receipt(reference['path']) != reference:
        raise ValueError('Exact existing path/SHA reference required')
    return json.loads(Path(reference['path']).read_text())


def validate_features(value):
    if set(value) != {'held_owner', 'previous_held_owner', 'context_age_ns', 'handler_active'}:
        raise ValueError('Only declared shared visible fields may enter features')
    return shared_features(**value)


def mask(value):
    validate_features(value)
    return ''.join('1' if value['handler_active'][owner] is True else '?' for owner in OWNERS)


def signature(value):
    validate_features(value)
    return value['held_owner']


def mark_names():
    return ['intercept', 'log_age_ms', 'log_age_ms_squared',
        *['held:' + owner for owner in OWNERS],
        *['held_log_age:' + owner for owner in OWNERS],
        *['previous:' + str(owner) for owner in (None, *OWNERS)]]


def mark_vector(value):
    validate_features(value)
    log_age = math.log1p(value['context_age_ns'] / 1_000_000)
    v = defaultdict(float, intercept=1., log_age_ms=log_age, log_age_ms_squared=log_age**2)
    v['held:' + value['held_owner']] = 1.
    v['held_log_age:' + value['held_owner']] = log_age
    v['previous:' + str(value['previous_held_owner'])] = 1.
    return [v[name] for name in mark_names()]


def hazard_tokens(value):
    validate_features(value)
    return ('holder:' + value['held_owner'],)


def merged_intervals(values):
    result = []
    for lo, hi in sorted(values):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def jsonl(reference):
    if receipt(reference['path']) != reference:
        raise ValueError('Exact compressed training stream required')
    with gzip.open(reference['path'], 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)
    if receipt(reference['path']) != reference:
        raise ValueError('Training stream changed during read')


def validate_risk(row):
    value = validate_features(row['shared_left_features'])
    fields = ('age_entry_ns', 'age_exit_ns', 'start_ns', 'end_ns', 'restore_start_ns')
    if (any(type(row[k]) is not int for k in fields)
            or not 0 <= row['age_entry_ns'] < row['age_exit_ns']
            or not row['restore_start_ns'] <= row['start_ns'] < row['end_ns']
            or row['start_ns']-row['restore_start_ns'] != row['age_entry_ns']
            or row['end_ns']-row['restore_start_ns'] != row['age_exit_ns']
            or value['context_age_ns'] != row['age_entry_ns']
            or row['held_owner'] != value['held_owner']
            or type(row['physical_SAVE_END_observed']) is not bool
            or type(row['exposure_qualified']) is not bool
            or len(row['driver_context']) != 4
            or any(type(x) is not int for x in row['driver_context'])):
        raise ValueError('Every resolved or unresolved native risk row requires consistent identity, positive exact exposure and boolean targets')


def event_identity(timestamp, context, sequence):
    if (type(timestamp) is not int or timestamp<0 or type(sequence) is not int
            or len(context)!=4 or any(type(x) is not int for x in context)):
        raise ValueError('Exact native timestamp, driver context and sequence required')
    return timestamp,tuple(context),sequence


def verify_native_accounting(report, native, marks, risks):
    capture=report['capture'];lo,hi=report['native_window_ns']
    if native['capture'] != capture or native['native_window_ns'] != [lo,hi]:
        raise ValueError('Native/shared capture identity and exact window must agree')
    save_doc=checked(report['physical_SAVE_event_ledger'])
    gap_doc=checked(report['native_context_gaps'])
    native_ledger=checked(native['native_transition_ledger'])
    events=save_doc['events']
    saves={}
    for event in events:
        key=event_identity(event['physical_SAVE_END_ns'],event['driver_context'],event['seqNo'])
        if key in saves or event['capture'] != capture or not lo <= event['physical_SAVE_END_ns'] < hi:
            raise ValueError('Physical SAVE ledger must be unique and entirely from this capture/window')
        saves[key]=event
    if (len(events)!=report['physical_SAVE_event_count']
            or sum(e['next_owner'] is None for e in events)!=report['SAVE_without_next_mark_count']):
        raise ValueError('All physical SAVE and unknown mark counts must reconcile')
    risk_events={};envelopes=defaultdict(list)
    for row in risks:
        validate_risk(row)
        if not lo <= row['start_ns'] < row['end_ns'] <= hi:
            raise ValueError('Risk rows must remain in the original native observation window')
        envelopes[row['held_owner'],tuple(row['driver_context']),row['restore_start_ns']].append(row)
        if row['physical_SAVE_END_observed']:
            key=event_identity(row['end_ns'],row['driver_context'],row['physical_SAVE_seqNo'])
            event=saves.get(key)
            if key in risk_events or event is None or (event['owner'],event['driver_context'],
                    event['restore_start_ns'],event['physical_SAVE_END_ns']) != (
                    row['held_owner'],row['driver_context'],row['restore_start_ns'],row['end_ns']):
                raise ValueError('Every risk SAVE must map exactly once to its original physical ledger event')
            risk_events[key]=row
    expected={event_identity(e['physical_SAVE_END_ns'],e['driver_context'],e['seqNo'])
        for e in events if e['restore_start_ns'] is not None
        and max(lo,e['restore_start_ns']) < e['physical_SAVE_END_ns']}
    if set(risk_events)!=expected:
        raise ValueError('Paired physical SAVE labels cannot disappear from risk chronology')
    intervals=[]
    for rows in envelopes.values():
        rows.sort(key=lambda r:r['start_ns'])
        if any(a['end_ns']!=b['start_ns'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Feature-split risk chronology must remain continuous within every envelope')
        intervals.append([rows[0]['start_ns'],rows[-1]['end_ns']])
    if gap_doc['native_window_ns'] != [lo,hi]:
        raise ValueError('Gap ledger must cover the same original native window')
    all_intervals=sorted([*intervals,*gap_doc['gaps']])
    if (not all_intervals or all_intervals[0][0]!=lo or all_intervals[-1][1]!=hi
            or any(type(t) is not int or not lo<=t<=hi for span in all_intervals for t in span)
            or any(a>=b for a,b in all_intervals)
            or any(a[1]!=b[0] for a,b in zip(all_intervals,all_intervals[1:]))
            or sum(b-a for a,b in gap_doc['gaps'])!=report['gap_duration_ns']):
        raise ValueError('Complete paired exposure plus explicit unknown gaps must partition the original window exactly')
    def native_key(r):
        return (r['held_owner'],tuple(r['held_driver_context']),r['current_restore_start_ns'],r['save_end_ns'],
            r['next_owner'],tuple(r['next_driver_context']),r['next_restore_start_ns'])
    originals=[native_key(r) for r in native_ledger['qualified']]
    derived=[]
    for row in marks:
        value=validate_features(row['features']);end=row['SAVE_END_ns']
        key=(value['held_owner'],tuple(row['held_driver_context']),end-value['context_age_ns'],end,
            row['next_owner'],tuple(row['next_driver_context']),row['next_restore_start_ns'])
        if (row['capture']!=capture or type(end) is not int
                or row['next_owner'] not in OWNERS or not end<=row['next_restore_start_ns']
                or row['same_owner'] is not (value['held_owner']==row['next_owner'])
                or row['same_driver_context'] is not (row['held_driver_context']==row['next_driver_context'])):
            raise ValueError('Physical mark label/identity must remain separate and internally consistent')
        derived.append(key)
    if len(set(originals))!=len(originals) or len(set(derived))!=len(derived) or set(originals)!=set(derived):
        raise ValueError('Every qualified original physical adjacency must occur exactly once as a mark')
    if (Counter(row['next_owner'] for row in marks)!=Counter(report['mark_target_counts'])
            or sum(row['same_owner'] for row in marks)!=report['qualified_same_owner_count']):
        raise ValueError('Physical mark class/same-owner support accounting changed')


def load_dataset(references):
    if len(references) != 4 or {r['route'] for r in references} != set(ROUTES):
        raise ValueError('Four separately collected original light training routes required')
    marks, risks, captures, sources, ledgers = [], [], {}, [], []
    for member in references:
        if set(member) != {'route', 'report'}:
            raise ValueError('Explicit route and final shared-observable report required')
        report = checked(member['report'])
        if (report.get('schema') != DATA_SCHEMA
                or report.get('status') != 'SHARED_OBSERVABLE_RISK_AND_MARK_EXTRACTED_NO_FIT'
                or report.get('training_only') is not True
                or report.get('validation_labels_used') is not False
                or report.get('fit_performed') is not False):
            raise ValueError('Frozen training-only no-fit shared native adapter required')
        capture = report['capture']
        if capture in captures:
            raise ValueError('One physical capture cannot occur in two folds')
        native = checked(report['source_adapter'])
        review = checked(native['review'])
        expected_suffix = '/whole57-joint-light-v23/' + member['route'] + '/report.json'
        if not native['review']['path'].endswith(expected_suffix):
            raise ValueError('Route identity must bind its actual original light review')
        if (review.get('status') != 'PASS' or review.get('issues') != []
                or review.get('requests') != 57 or review.get('training_only') is not True):
            raise ValueError('Complete original 57-request PASS training capture required')
        evidence=review['evidence'];context=checked(evidence['context.json']);plan=checked(evidence['plan.json'])
        raw=checked(evidence['raw.json'])
        if (native['route_id'] != member['route'] or plan['route_id'] != member['route']
                or raw['route_id'] != member['route']
                or context['whole57_training_route_id'] != member['route']
                or plan['domain'] != 'whole57-joint-light-v23' or raw['domain'] != plan['domain']
                or context['measurement_domain'] != plan['domain']
                or context_identity(context) != plan['execution_context_sha256']
                or raw['plan'] != evidence['plan.json'] or raw['context'] != evidence['context.json']
                or raw['execution_context_sha256'] != plan['execution_context_sha256']
                or evidence['raw.json']['sha256'] != capture
                or native['full_original_measured_requests'] != 57 or native['original_priming_requests'] != 12):
            raise ValueError('Actual raw capture, route, context and original 57+12 cohort must agree')
        refs = [member['report'], report['source_adapter'], *report['sources'],
            *report['implementations'], report['risk_chronology'], report['mark_records'],
            report['physical_SAVE_event_ledger'], report['native_context_gaps']]
        refs = strict_references(refs)
        for reference in refs:
            if receipt(reference['path']) != reference:
                raise ValueError('Training dependency changed before fitting')
        sources.extend(refs)
        captures[capture] = dict(route=member['route'], report=member['report'],
            native_window_ns=report['native_window_ns'], full57=True,
            SAVE_without_next_mark_count=report['SAVE_without_next_mark_count'],
            gap_duration_ns=report['gap_duration_ns'])
        cm = list(jsonl(report['mark_records']))
        cr = list(jsonl(report['risk_chronology']))
        if len(cm) != report['mark_record_count'] or len(cr) != report['counts']['all_paired_intervals']:
            raise ValueError('Every recorded mark and risk interval must be retained')
        if any(row['capture'] != capture for row in [*cm, *cr]):
            raise ValueError('Training row cannot cross physical captures')
        if (sum(row['end_ns'] - row['start_ns'] for row in cr) != report['counts']['all_paired_exposure_ns']
                or sum(row['physical_SAVE_END_observed'] for row in cr) != report['counts']['paired_SAVE_events']):
            raise ValueError('Full exposure and physical SAVE accounting must reconcile')
        verify_native_accounting(report,native,cm,cr)
        marks.extend(cm); risks.extend(cr)
        ledgers.append(dict(capture=capture, physical_SAVE=report['physical_SAVE_event_ledger'],
            gaps=report['native_context_gaps'], counts=report['counts']))
    return dict(marks=marks, risks=risks, captures=captures,
                sources=strict_references(sources), ledgers=ledgers)


def aggregate(risks):
    bins = defaultdict(lambda: [0, 0, 0])
    counts = Counter(); by_capture = defaultdict(Counter); envelopes = defaultdict(set)
    support = defaultdict(list); vocabulary = set()
    for row in risks:
        validate_risk(row)
        features = row['shared_left_features']
        start, stop = row['age_entry_ns'], row['age_exit_ns']
        if features['context_age_ns'] != start or row['end_ns'] - row['start_ns'] != stop - start:
            raise ValueError('Attained age and positive native exposure must agree')
        counts['all_risk_intervals'] += 1
        counts['all_exposure_ns'] += stop - start
        counts['all_paired_SAVE'] += int(row['physical_SAVE_END_observed'])
        if row['exposure_qualified'] is not True:
            counts['unresolved_intervals'] += 1
            counts['unresolved_exposure_ns'] += stop - start
            counts['unresolved_paired_SAVE'] += int(row['physical_SAVE_END_observed'])
            continue
        tokens = hazard_tokens(features); vocabulary.update(tokens)
        capture = row['capture']
        envelopes[capture].add((row['held_owner'], tuple(row['driver_context']), row['restore_start_ns']))
        support[signature(features)].append((start, stop))
        pieces = list(split_exposure(dict(age_entry_ns=start, age_exit_ns=stop,
            save_observed=row['physical_SAVE_END_observed'])))
        if (sum(width for _, width, _ in pieces) != stop - start
                or sum(event for _, _, event in pieces) != int(row['physical_SAVE_END_observed'])):
            raise ValueError('Age bin split changed exposure or SAVE targets')
        for index, width, event in pieces:
            value = bins[capture, index, tokens]
            value[0] += width; value[1] += event; value[2] += 1
        for c in (counts, by_capture[capture]):
            c['qualified_intervals'] += 1; c['exposure_ns'] += stop - start
            c['events'] += int(row['physical_SAVE_END_observed'])
    if not bins or not counts['events']:
        raise ValueError('Nonempty qualified physical SAVE risk required')
    rows = [dict(capture=c, age_bin=i, features=list(fs), exposure_ns=v[0],
        events=v[1], episode_bin_intersections=v[2]) for (c, i, fs), v in sorted(bins.items())]
    return dict(rows=rows, counts=dict(counts), by_capture={c:dict(v) for c,v in by_capture.items()},
        capture_envelopes={c:len(v) for c,v in envelopes.items()}, vocabulary=sorted(vocabulary),
        age_support_ns=merged_intervals([span for values in support.values() for span in values]),
        state_age_support_ns={k:merged_intervals(v) for k,v in support.items()})


def fit_models(marks, risks):
    if not marks:
        raise ValueError('Nonempty physical next-owner labels required')
    matrix = np.asarray([mark_vector(row['features']) for row in marks])
    labels = np.asarray([OWNERS.index(row['next_owner']) for row in marks])
    mark = fit_matrix(matrix, labels, [row['capture'] for row in marks], mark_names())
    mark['feature_names'] = mark_names()
    stats = aggregate(risks)
    if any(v.get('events',0) <= 0 for v in stats['by_capture'].values()):
        raise ValueError('Capture-event-normalized hazard requires positive observed SAVE events in every training capture')
    hazard = normalized_fit(stats)
    hazard['state_age_support_ns'] = stats['state_age_support_ns']
    mark_support = defaultdict(list)
    for row in marks:
        mark_support[digest([signature(row['features']), row['features']['previous_held_owner']])].append(
            row['features']['context_age_ns'])
    mark['state_age_ranges_ns'] = {k:[min(v), max(v)] for k,v in mark_support.items()}
    if not mark['optimizer']['converged'] or not hazard['optimizer']['success']:
        raise ValueError('Both declared numerical training fits must converge')
    return dict(mark=mark, hazard=hazard, risk_accounting=stats)


def evaluate_models(models, marks, risks):
    fitted = models['mark']
    probabilities = predict_matrix(fitted, np.asarray([mark_vector(row['features']) for row in marks]))
    labels = np.asarray([OWNERS.index(row['next_owner']) for row in marks])
    chosen = probabilities.argmax(axis=1)
    mark_groups = {}
    for key, indexes in [('all', list(range(len(marks)))),
            *[('target=' + owner, [i for i,r in enumerate(marks) if r['next_owner'] == owner]) for owner in OWNERS],
            *[('held=' + owner, [i for i,r in enumerate(marks) if r['features']['held_owner'] == owner]) for owner in OWNERS]]:
        ix = np.asarray(indexes, dtype=int)
        if len(ix):
            probs = probabilities[ix]; actual = labels[ix]
            mark_groups[key] = dict(n=len(ix), accuracy=float(np.mean(chosen[ix] == actual)),
                NLL=float(-np.log(np.maximum(probs[np.arange(len(ix)), actual], 1e-300)).mean()),
                multiclass_Brier=float(np.sum((probs-np.eye(len(OWNERS))[actual])**2, axis=1).mean()),
                same_held_prediction_count=sum(OWNERS[chosen[i]] == marks[i]['features']['held_owner'] for i in indexes))
        else:
            mark_groups[key] = dict(n=0, accuracy=None, NLL=None, multiclass_Brier=None,
                same_held_prediction_count=0)
    support = fitted['state_age_ranges_ns']
    mark_unknown = 0
    for row in marks:
        value = row['features']; interval = support.get(digest([signature(value), value['previous_held_owner']]))
        mark_unknown += interval is None or not interval[0] <= value['context_age_ns'] <= interval[1]
    data = aggregate(risks); x, _ = design(data['rows'], models['hazard']['columns'], 'age_additive')
    rates = np.exp(x @ np.asarray(models['hazard']['coefficients']))
    groups = defaultdict(lambda:dict(exposure_ns=0, observed_SAVE=0, expected_SAVE=0.))
    for row, rate in zip(data['rows'], rates):
        holder = next(v for v in row['features'] if v.startswith('holder:'))
        for key in ('all', holder, 'capture=' + row['capture']):
            g = groups[key]; g['exposure_ns'] += row['exposure_ns']; g['observed_SAVE'] += row['events']
            g['expected_SAVE'] += float(rate * row['exposure_ns'] / 1e6)
    unseen_hazard_rows = sum(any(f not in models['hazard']['training_vocabulary'] for f in row['features'])
        for row in data['rows'])
    return dict(physical_mark=mark_groups, mark_outside_state_age_support=mark_unknown,
        physical_SAVE=[dict(stratum=k, **v) for k,v in sorted(groups.items())],
        hazard_rows_with_unseen_training_tokens=unseen_hazard_rows,
        full_risk_accounting=data['counts'], same_owner_observed_count=sum(row['same_owner'] for row in marks),
        context_equals_SM_service=False, calibration_certified=False,
        independent_validation=False)


class ObservableContextCandidate:
    """Physical probabilities/rates only; no active-service progress assertion."""
    def __init__(self, model):
        if (model.get('schema') != SCHEMA or model.get('parameters') != PARAMETERS
                or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})):
            raise ValueError('Frozen training-only observable context model required')
        self.model = model
        fitted=model['fitted'];mark=fitted['mark'];hazard=fitted['hazard']
        width=len(mark_names())
        if (mark.get('feature_names')!=mark_names()
                or np.asarray(mark['coefficients']).shape!=(width,len(OWNERS))
                or any(np.asarray(mark[k]).shape!=(width,) for k in (
                    'center','scale','feature_minimum','feature_maximum'))
                or not all(np.isfinite(np.asarray(mark[k],dtype=float)).all() for k in (
                    'coefficients','center','scale','feature_minimum','feature_maximum'))
                or np.any(np.asarray(mark['scale'])<=0)
                or np.any(np.asarray(mark['feature_minimum'])>np.asarray(mark['feature_maximum']))
                or set(mark['label_counts'])!=set(OWNERS)
                or any(type(n) is not int or n<0 for n in mark['label_counts'].values())
                or sum(mark['label_counts'].values())<=0
                or len(hazard['columns'])!=len(set(hazard['columns']))
                or len(hazard['columns'])!=len(hazard['coefficients'])
                or 'intercept' not in hazard['columns']
                or not np.isfinite(np.asarray(hazard['coefficients'],dtype=float)).all()
                or hazard['fixed_age_edges_ns']!=list(EDGES_NS)
                or any(not 0<=lo<hi for spans in hazard['state_age_support_ns'].values() for lo,hi in spans)):
            raise ValueError('Complete finite shape-consistent physical candidate coefficients and support required')
        for reference in model['sources']:
            if receipt(reference['path']) != reference:
                raise ValueError('Observable candidate training dependencies changed')
        self.coefficients = dict(zip(model['fitted']['hazard']['columns'], model['fitted']['hazard']['coefficients']))

    def next_owner(self, value):
        vector = np.asarray([mark_vector(value)])
        fitted = self.model['fitted']['mark']
        probabilities = predict_matrix(fitted, vector)[0]
        state = digest([signature(value), value['previous_held_owner']])
        interval = fitted['state_age_ranges_ns'].get(state)
        flags = []
        if interval is None: flags.append('UNSEEN_HELD_PREVIOUS_STATE')
        elif not interval[0] <= value['context_age_ns'] <= interval[1]:
            flags.append('ATTAINED_AGE_OUTSIDE_STATE_MARK_RANGE')
        absent = [owner for owner in OWNERS if fitted['label_counts'].get(owner, 0) == 0]
        if absent: flags.append('PHYSICAL_TARGET_CLASS_MISSING_IN_TRAINING')
        return dict(physical_owner_probabilities={owner:float(p) for owner,p in zip(OWNERS,probabilities)},
            most_likely_physical_owner=OWNERS[int(probabilities.argmax())], OOD_reasons=flags,
            missing_training_targets=absent, probability_calibrated=False,
            physical_context_equals_active_service=False)

    def save_hazard(self, value, horizon_ns=0):
        if type(horizon_ns) is not int or horizon_ns < 0:
            raise ValueError('Nonnegative integer forecast horizon required')
        tokens = hazard_tokens(value); fitted = self.model['fitted']['hazard']; coefs = self.coefficients
        age = value['context_age_ns']; index = bisect_right(EDGES_NS, age)-1
        base = coefs['intercept'] + sum(coefs.get(t, 0.) for t in tokens)
        rate = math.exp(base + coefs.get('age_bin:' + str(index), 0.))
        flags = []
        if set(tokens)-set(fitted['training_vocabulary']): flags.append('UNSEEN_HOLDER_TOKENS')
        supports = fitted['state_age_support_ns'].get(signature(value), [])
        stop = next((hi for lo,hi in supports if lo <= age < hi), age)
        if stop == age: flags.append('AGE_OUTSIDE_OBSERVED_STATE_RISK')
        if age + horizon_ns > stop: flags.append('HORIZON_OUTSIDE_OBSERVED_STATE_RISK')
        end = min(age+horizon_ns, stop); clock = age; cumulative = 0.
        while clock < end:
            i = bisect_right(EDGES_NS, clock)-1
            right = min(end, EDGES_NS[i+1]) if i+1 < len(EDGES_NS) else end
            cumulative += math.exp(base+coefs.get('age_bin:'+str(i),0.))*(right-clock)/1e6
            clock = right
        return dict(hazard_per_ms=rate, OOD_reasons=flags, restricted_horizon_ns=end-age,
            conditional_CDF=None if flags else -math.expm1(-cumulative),
            restricted_candidate_CDF=-math.expm1(-cumulative),
            target='physical_context_SAVE_END_not_SM_service', probability_calibrated=False,
            state_held_fixed_through_horizon=True)


def train(references, output, *, manifest_reference):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    manifest = checked(manifest_reference)
    if (set(manifest) != {'training_only', 'validation_labels_used', 'captures'}
            or manifest['training_only'] is not True or manifest['validation_labels_used'] is not False
            or manifest['captures'] != references):
        raise ValueError('Exact declared training manifest required')
    data = load_dataset(references)
    implementations = implementation_sources(__file__)
    sources = strict_references([manifest_reference, data['sources'], implementations])
    folds = []
    for capture in sorted(data['captures']):
        train_marks = [r for r in data['marks'] if r['capture'] != capture]
        train_risks = [r for r in data['risks'] if r['capture'] != capture]
        test_marks = [r for r in data['marks'] if r['capture'] == capture]
        test_risks = [r for r in data['risks'] if r['capture'] == capture]
        models = fit_models(train_marks, train_risks)
        folds.append(dict(held_out_capture=capture, held_out_route=data['captures'][capture]['route'],
            training_captures=sorted(c for c in data['captures'] if c != capture),
            metrics=evaluate_models(models, test_marks, test_risks),
            training_optimizers=dict(mark=models['mark']['optimizer'], hazard=models['hazard']['optimizer'])))
    fitted = fit_models(data['marks'], data['risks'])
    model = dict(schema=SCHEMA, parameters=PARAMETERS, fitted=fitted,
        training_only=True, validation_labels_used=False, sources=sources,
        implementation=implementations, captures=data['captures'], ledgers=data['ledgers'],
        physical_context_equals_SM_service=False, active_service_bridge_validated=False)
    model['model_sha256'] = digest(model)
    for ref in sources:
        if receipt(ref['path']) != ref: raise ValueError('Training dependencies changed while fitting')
    out.mkdir(parents=True, exist_ok=False)
    save_synced(out/'model.json', model)
    report = dict(schema=SCHEMA, status='TRAINING_CANDIDATE_FITTED_NOT_INDEPENDENTLY_VALIDATED',
        parameters=PARAMETERS, model=receipt(out/'model.json'),
        whole_capture_holdout=folds, holdout_is_known_input_family=True,
        holdout_is_content_or_recipe_independent=False,
        training_metrics=evaluate_models(fitted, data['marks'], data['risks']),
        original_capture_accounting=data['captures'], all_source_ledgers_retained=True,
        global_request_cap_added=False, six_term_metrics_produced=False,
        independent_GPU_validation=False, GPU_started=False)
    save_synced(out/'report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-manifest', required=True)
    p.add_argument('--training-manifest-sha256', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    manifest_reference = dict(path=str(Path(args.training_manifest).absolute()), sha256=args.training_manifest_sha256)
    doc = checked(manifest_reference)
    if set(doc) != {'training_only', 'validation_labels_used', 'captures'} or doc['training_only'] is not True or doc['validation_labels_used'] is not False:
        p.error('Explicit training-only four-capture manifest required')
    result = train(doc['captures'], args.output, manifest_reference=manifest_reference)
    print(json.dumps(dict(status=result['status'], model=result['model'])))
