"""Training-only event landmark KM; elapsed endpoint time is never SM service."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from . import kernel_endpoint_candidate_v27 as base
from .issued_queue_endpoint_candidate_v28 import IssuedQueuePrefix
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA = 'gsdeploy.kernel_endpoint_conditional_survival.v29'
QUANTILES = (.5, .95, .99)
EVALUATION_HORIZONS_NS = (100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000)
MIN_GROUP_N = 5


def km_fit(observations):
    """Ordinary KM with administrative right censoring and event-before-censor ties."""
    counts = defaultdict(lambda: [0, 0])
    for duration, event in observations:
        if type(duration) is not int or duration <= 0 or type(event) is not bool:
            raise ValueError('Positive followup and explicit observed endpoint required')
        counts[duration][0 if event else 1] += 1
    risk = len(observations)
    survival = 1.
    times, values = [], []
    for duration, (events, censors) in sorted(counts.items()):
        survival *= 1 - events / risk
        times.append(duration)
        values.append(survival)
        risk -= events + censors
    return dict(n=len(observations), events=sum(e for _, e in observations),
                censored=sum(not e for _, e in observations), times_ns=times,
                survival=values, maximum_followup_ns=max(times, default=0))


def survival_at(model, age):
    index = bisect_right(model['times_ns'], age) - 1
    return 1. if index < 0 else model['survival'][index]


def conditional(model, age):
    """Condition on T > age, preserving unidentified quantiles as UNKNOWN."""
    if age < 0:
        raise ValueError('Nonnegative survived age required')
    denominator = survival_at(model, age)
    if denominator <= 0 or age >= model['maximum_followup_ns']:
        return dict(status='UNKNOWN_NO_SURVIVING_TRAINING_SUPPORT', quantiles_ns={}, survival={})
    quantiles = {}
    first = bisect_right(model['times_ns'], age)
    for q in QUANTILES:
        endpoint = next((t for t, s in zip(model['times_ns'][first:], model['survival'][first:])
                         if s / denominator <= 1 - q + 1e-12), None)
        quantiles[str(q)] = None if endpoint is None else endpoint - age
    probabilities = {str(h): survival_at(model, age + h) / denominator
                     if age + h <= model['maximum_followup_ns'] else None
                     for h in EVALUATION_HORIZONS_NS}
    return dict(status='UNCALIBRATED_ENDPOINT_SURVIVAL_CANDIDATE',
                quantiles_ns=quantiles, survival=probabilities,
                conditioning_survival_probability=denominator,
                support_remaining_ns=model['maximum_followup_ns'] - age)


def queue_bucket(count):
    if type(count) is not int or count < 0:
        raise ValueError('Observed issued unfinished count required')
    return '0' if count == 0 else '1' if count == 1 else '2-7' if count < 8 else '8-31' if count < 32 else '32+'


def label_at(endpoint, clock, observation_end):
    # Event already occurred is a state, never a fabricated zero-duration label.
    if endpoint <= clock or observation_end <= clock:
        return None
    return dict(duration_ns=min(endpoint, observation_end) - clock,
                event_observed=endpoint <= observation_end)


def landmark_rows(rows, queue_indexes, issue_times, horizons):
    result = []
    for row in rows:
        prefix = row['prefix_features']; labels = row['observation_labels']
        origin = prefix['cutoff_ns']; capture = row['capture']; end = horizons[capture]
        times = issue_times[capture]; pos = bisect_right(times, origin)
        landmarks = [('launch_API_return', origin)]
        if pos < len(times): landmarks.append(('next_observed_issue', times[pos]))
        for age in (100_000, 1_000_000, 10_000_000, 100_000_000):
            landmarks.append(('elapsed_' + str(age) + 'ns', origin + age))
        start = labels['observed_kernel_start_ns']
        if origin < start:
            landmarks.append(('observed_own_START', start))
        for kind in base.TARGETS:
            endpoint = labels['observed_kernel_start_ns' if kind.startswith('START') else 'observed_kernel_end_ns']
            if endpoint <= origin: continue
            for event_kind, clock in landmarks:
                observation = label_at(endpoint, clock, end)
                if observation is None: continue
                current_queue = queue_indexes[capture].counts(clock)
                same = current_queue.get((prefix['owner'], row['source_stream_id']), 0)
                result.append(dict(capture=capture, source_node_id=row['source_node_id'], target=kind,
                    signature=base.signature(prefix), landmark=event_kind, cutoff_ns=clock,
                    survived_age_ns=clock-origin, observed_same_stream_issued_unfinished=same,
                    queue_bucket=queue_bucket(same), already_started=start <= clock,
                    **observation))
    return result


def group_key(row, mode):
    common = (row['target'], *row['signature'])
    if mode == 'signature_origin_KM': return common
    if mode == 'landmark_queue_KM':
        return (*common, row['landmark'], row['queue_bucket'], row['already_started'])
    raise ValueError('Declared survival mode required')


def fit(rows, mode):
    groups = defaultdict(list)
    for row in rows:
        if mode == 'signature_origin_KM' and row['landmark'] != 'launch_API_return': continue
        groups[group_key(row, mode)].append((row['duration_ns'], row['event_observed']))
    return {key: km_fit(values) for key, values in groups.items()}


def predict(models, row, mode):
    model = models.get(group_key(row, mode))
    if model is None or model['n'] < MIN_GROUP_N:
        return dict(status='UNKNOWN_UNSEEN_OR_SPARSE_GROUP', quantiles_ns={}, survival={})
    age = row['survived_age_ns'] if mode == 'signature_origin_KM' else 0
    return conditional(model, age)


def score(rows, estimates):
    """Censoring-aware coverage bounds; uncensored absolute errors named explicitly."""
    result = dict(n=len(rows), events=sum(r['event_observed'] for r in rows),
                  censors=sum(not r['event_observed'] for r in rows), quantiles={}, horizons={})
    for q in QUANTILES:
        supported = [(r, e.get('quantiles_ns', {}).get(str(q))) for r, e in zip(rows, estimates)]
        supported = [(r, p) for r, p in supported if p is not None]
        success = unknown = 0; errors = []; censored_error_lower_bounds = []
        observed_actual, observed_predictions = [], []
        for row, prediction in supported:
            duration = row['duration_ns']
            if row['event_observed']:
                success += duration <= prediction
                errors.append(abs(duration-prediction))
                observed_actual.append(duration); observed_predictions.append(prediction)
            else:
                unknown += duration <= prediction
                censored_error_lower_bounds.append(max(0, duration-prediction))
        n = len(supported)
        result['quantiles'][str(q)] = dict(supported_n=n, unknown_prediction_n=len(rows)-n,
            coverage_fraction_bounds=None if n == 0 else [success/n, (success+unknown)/n],
            coverage_bounds_do_not_drop_censored_rows=True,
            uncensored_only_n=len(errors),
            uncensored_only_mean_absolute_error_us=None if not errors else float(np.mean(errors)/1e3),
            uncensored_only_median_absolute_error_us=None if not errors else float(np.median(errors)/1e3),
            uncensored_only_p95_absolute_error_us=None if not errors else float(np.percentile(errors,95)/1e3),
            uncensored_only_actual_duration_p95_us=None if not errors else float(np.percentile(observed_actual,95)/1e3),
            uncensored_only_predicted_duration_p95_us=None if not errors else float(np.percentile(observed_predictions,95)/1e3),
            censored_absolute_error_lower_bound_mean_us=None if not censored_error_lower_bounds else float(np.mean(censored_error_lower_bounds)/1e3))
    for horizon in EVALUATION_HORIZONS_NS:
        n=unknown=0; lower=upper=0.
        for row, estimate in zip(rows, estimates):
            p=estimate.get('survival', {}).get(str(horizon))
            if p is None: continue
            n+=1
            if row['event_observed'] and row['duration_ns'] <= horizon:
                loss=p*p; lower+=loss; upper+=loss
            elif row['duration_ns'] >= horizon:
                loss=(1-p)**2; lower+=loss; upper+=loss
            else:
                unknown+=1; lower+=min(p*p,(1-p)**2); upper+=max(p*p,(1-p)**2)
        result['horizons'][str(horizon)]=dict(supported_n=n, early_censored_unknown_n=unknown,
            brier_score_bounds=None if n == 0 else [lower/n,upper/n])
    return result


def run(queue_report_path, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    report=read(queue_report_path)
    if report.get('validation_labels_used') is not False or report.get('integrated_into_forecast') is not False:
        raise ValueError('Unintegrated training-only v28 source required')
    evidence=read(report['source_receipts_before'][0]['path'])
    refs=[receipt(queue_report_path),report['sample'],receipt(report['source_receipts_before'][0]['path']),
          evidence['kernel_rows'],evidence['launch_vocabulary'],receipt(__file__),receipt(base.__file__)]
    for ref in refs:
        if receipt(ref['path']) != ref: raise ValueError('Changed input bytes')
    vocabulary=read(evidence['launch_vocabulary']['path'])
    issued=defaultdict(list); events=defaultdict(list); horizons=defaultdict(int); counts=Counter()
    with gzip.open(evidence['kernel_rows']['path'],'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line); p=row['prefix_features']; labels=row['observation_labels']; capture=row['capture']
            origin=p['cutoff_ns']; end=labels['observed_kernel_end_ns']; owner=p['owner']
            events[capture].append(origin); horizons[capture]=max(horizons[capture],origin)
            if origin < end: issued[capture,owner,vocabulary[p['launch_signature']]['stream']].append((origin,end))
            counts['source_rows']+=1
            counts['START_before_API_return']+=labels['observed_kernel_start_ns'] < origin
            counts['END_before_API_return']+=end < origin
    indexes={capture: IssuedQueuePrefix({(owner,stream): spans for (c,owner,stream),spans in issued.items() if c==capture})
             for capture in events}
    events={capture: sorted(set(times)) for capture,times in events.items()}
    with gzip.open(report['sample']['path'],'rt',encoding='utf-8') as stream:
        sample=[json.loads(line) for line in stream]
    rows=landmark_rows(sample,indexes,events,horizons)
    counts['sampled_kernels']=len(sample); counts['landmark_rows']=len(rows)
    counts['administratively_censored_landmarks']=sum(not r['event_observed'] for r in rows)
    modes=('signature_origin_KM','landmark_queue_KM'); folds=[]; fold_models=[]; paired=[]
    for heldout in sorted(events):
        train=[r for r in rows if r['capture']!=heldout]; test=[r for r in rows if r['capture']==heldout]
        predictions={}
        for mode in modes:
            models=fit(train,mode); estimates=[predict(models,row,mode) for row in test]
            predictions[mode]=estimates
            fold_models.append(dict(heldout_training_capture=heldout,mode=mode,
                groups=[dict(key=list(key),km=value) for key,value in sorted(models.items())]))
            strata=defaultdict(list)
            for index,row in enumerate(test): strata[row['target'],row['landmark']].append(index)
            folds.append(dict(heldout_training_capture=heldout,mode=mode,training_landmarks=len(train),
                training_groups=len(models),unknown_statuses=dict(Counter(e['status'] for e in estimates if e['status'].startswith('UNKNOWN'))),
                scores=[dict(target=kind,landmark=event,**score([test[i] for i in ids],[estimates[i] for i in ids]))
                        for (kind,event),ids in sorted(strata.items())]))
        for (kind,event),ids in sorted(strata.items()):
            for q in QUANTILES:
                common=[i for i in ids if all(predictions[mode][i].get('quantiles_ns',{}).get(str(q)) is not None for mode in modes)]
                paired.append(dict(heldout_training_capture=heldout,target=kind,landmark=event,
                    quantile=q,requested_n=len(ids),common_support_n=len(common),
                    methods={mode:score([test[i] for i in common],[predictions[mode][i] for i in common])['quantiles'][str(q)] for mode in modes}))
    models={mode:[dict(key=list(key),km=value) for key,value in sorted(fit(rows,mode).items())] for mode in modes}
    after=[receipt(ref['path']) for ref in refs]
    if after != refs: raise ValueError('Source bytes changed during candidate evaluation')
    output.mkdir(parents=True)
    (output/'model.json').write_text(json.dumps(dict(schema=SCHEMA,training_only=True,models=models,fold_models=fold_models,
        minimum_group_n=MIN_GROUP_N,source_receipts=refs),indent=2)+'\n')
    with gzip.open(output/'landmark-observations.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row,separators=(',',':'))+'\n')
    result=dict(schema=SCHEMA,status='TRAINING_CAPTURE_HOLDOUT_SURVIVAL_DIAGNOSTIC_NOT_QUALIFIED',
        counts=dict(counts),observation_horizons_ns=dict(horizons),
        administrative_cutoff_rule='Last measured launch API return in each full training capture; later physical endpoints preserved in v26 but right-censored here',
        folds=folds,paired_common_support=paired,model=receipt(output/'model.json'),landmark_observations=receipt(output/'landmark-observations.jsonl.gz'),
        source_receipts_before=refs,source_receipts_after=after,
        limitations=['Two captures share input templates, not independent request validation.',
            'Measured issued unfinished kernel queue is a lower bound, not full CUDA DAG readiness.',
            'Administrative censoring is related to trace position; KM independent-censoring assumption is unverified.',
            'Offline event timestamps do not prove observer delivery availability.',
            'Landmark observations of a kernel are correlated; capture is the holdout unit.',
            'Signature baseline conditions on survived elapsed wall time; landmark candidate conditions on current queue and event class, without dynamic hazard integration.',
            'Unidentified tail quantiles and unsupported horizons are UNKNOWN, never extrapolated.',
            'Observed-only absolute errors exclude censors and cannot alone establish population tail accuracy.',
            'No counterfactual service/rate, request queue, P95, candidate ranking or utility validation.'],
        GPU_started=False,validation_labels_used=False,physical_service_fit=False,
        integrated_into_forecast=False,independent_accuracy_validated=False,final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.queue_report,args.output)
    print(json.dumps(dict(counts=result['counts'],report=result['model']),indent=2))
