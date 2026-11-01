"""Reconstruct the actual v57 fit without its aggregation/posterior helpers."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

from scipy.special import betainc, gammaincinv

from .native_collect import receipt
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced

OWNERS = ('generalist_2#0', 'ocr#0', 'text#0', 'text#1')
KEYS = tuple(owner+':'+format(i,'04b') for owner in OWNERS for i in range(16))
FIELDS = ('rows','exposure_ns','events','unresolved_rows','unresolved_exposure_ns','unresolved_events')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(ref):
    require(receipt(ref['path']) == ref, 'Exact source bytes required')
    return json.loads(Path(ref['path']).read_text())


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo,hi])
    return result


def pooled(counts, captures, key):
    result = {field:sum(counts[c][key][field] for c in captures) for field in FIELDS}
    support = merge([span for c in captures for span in counts[c][key]['age_support_intervals_ns']])
    return dict(**result, age_support_intervals_ns=support,
                age_min_ns=support[0][0] if support else None,
                age_max_ns=support[-1][1] if support else None)


def nb_quantile(q, shape, p):
    # CDF(k) = regularized incomplete beta(shape,k+1,p). Binary search
    # independently checks integer bounds from the author's nbinom.ppf.
    low, high = -1, 1
    while float(betainc(shape, high+1, p)) < q:
        high *= 2
    while high-low > 1:
        middle = (high+low)//2
        if float(betainc(shape, middle+1, p)) >= q:
            high = middle
        else:
            low = middle
    return high


def run(model_path, report_path, output):
    out = Path(output)
    require(not out.exists(), 'Exclusive review output required')
    model_ref, report_ref = receipt(model_path), receipt(report_path)
    model, report = read(model_ref), read(report_ref)
    require(model['schema'] == 'gsdeploy.projected_state_physical_SAVE_counting_process.v57'
            and model['training_only'] is True and model['validation_labels_used'] is False
            and report['model'] == model_ref and report['validation_labels_read'] is False
            and report['GPU_started'] is False, 'Actual training-only v57 model and report required')
    require(model['model_sha256'] == digest({k:v for k,v in model.items() if k != 'model_sha256'}), 'Model digest mismatch')
    require(len(model['captures']) == 4 and set(model['fitted']) == set(KEYS), 'Complete four-capture all64 model required')
    sources = [model_ref,report_ref,receipt(__file__)]
    for ref in model['sources']:
        require(receipt(ref['path']) == ref, 'Training source drift')
    counts, seen, rows_read = {}, set(), 0
    for capture, entry in model['captures'].items():
        projected = read(entry['report'])
        require(projected['capture'] == capture and projected['training_only'] is True
                and projected['validation_labels_used'] is False, 'Source role must be qualified before risk stream read')
        ref = projected['risk_chronology']
        require(receipt(ref['path']) == ref, 'Risk stream drift')
        sources.extend([entry['report'],ref])
        table = {key:dict(**{field:0 for field in FIELDS},age_support_intervals_ns=[]) for key in KEYS}
        with gzip.open(ref['path'],'rt') as stream:
            for line in stream:
                row = json.loads(line); rows_read += 1
                features = row['shared_left_features']; mask = features['handler_active']
                require(row['capture'] == capture and set(mask) == set(OWNERS)
                        and all(type(v) is bool for v in mask.values()), 'Wrong capture or incomplete mask')
                key = row['held_owner']+':'+''.join('1' if mask[o] else '0' for o in OWNERS)
                require(key in table and row['held_owner'] == features['held_owner'], 'Holder domain mismatch')
                width = row['end_ns']-row['start_ns']
                require(width > 0 and width == row['age_exit_ns']-row['age_entry_ns']
                        and row['start_ns']-row['restore_start_ns'] == row['age_entry_ns']
                        and type(row['physical_SAVE_END_observed']) is bool
                        and type(row['exposure_qualified']) is bool, 'Exact positive counting exposure required')
                identity = capture,tuple(row['driver_context']),row['start_ns'],row['end_ns']
                require(identity not in seen, 'Duplicate native exposure'); seen.add(identity)
                cell = table[key]; prefix = '' if row['exposure_qualified'] else 'unresolved_'
                cell[prefix+'rows'] += 1; cell[prefix+'exposure_ns'] += width
                cell[prefix+'events'] += int(row['physical_SAVE_END_observed'])
                if not prefix:
                    cell['age_support_intervals_ns'].append([row['age_entry_ns'],row['age_exit_ns']])
        require(receipt(ref['path']) == ref, 'Risk stream changed during parse')
        for value in table.values():
            support = merge(value['age_support_intervals_ns']); value['age_support_intervals_ns'] = support
            value['age_min_ns'] = support[0][0] if support else None
            value['age_max_ns'] = support[-1][1] if support else None
        require(table == model['by_capture'][capture], 'Independently reconstructed per-capture counts differ')
        counts[capture] = table
    comparisons = Counter()

    def close(actual, expected):
        require(type(actual) in (int,float) and math.isfinite(actual)
                and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-8), 'Independent numeric comparison differs')
        comparisons['numeric'] += 1

    def posterior(saved, stats):
        require(all(saved[k] == v for k,v in stats.items()), 'Pooled exposure/age support differs')
        comparisons['exact'] += len(stats)
        if not stats['exposure_ns']:
            require(saved['status'] == 'UNKNOWN_NO_EXPOSURE'
                    and all(saved[k] is None for k in ('shape','rate_seconds','hazard_per_second','hazard_per_ms','credible95_per_second')),
                    'No-exposure stratum must remain UNKNOWN')
            return
        shape, seconds = stats['events']+.5, stats['exposure_ns']/1e9
        require(saved['status'] == 'FITTED_UNVALIDATED_COUNTING_PROCESS', 'Observed posterior status differs')
        for name, value in (('shape',shape),('rate_seconds',seconds),
                            ('hazard_per_second',shape/seconds),('hazard_per_ms',shape/seconds/1000)):
            close(saved[name],value)
        for q,value in zip((.025,.975),saved['credible95_per_second']):
            close(value,float(gammaincinv(shape,q))/seconds)
        if stats['events'] == 0:
            close(saved['zero_event_frequentist_one_sided95_upper_per_second'],-math.log(.05)/seconds)

    captures = sorted(counts)
    for key in KEYS:
        posterior(model['fitted'][key], pooled(counts,captures,key))
    for fold in report['leave_one_whole_capture_out']:
        heldout = fold['held_out_capture']; training = [c for c in captures if c != heldout]
        require(fold['training_captures'] == training and fold['held_out_route'] == model['captures'][heldout]['route'], 'Whole-capture disjointness differs')
        score = 0.; scored = 0
        for key in KEYS:
            stats = pooled(counts,training,key); posterior(fold['fitted'][key],stats)
            test, d = counts[heldout][key],fold['diagnostics'][key]
            require(d['observed_SAVE'] == test['events'] and d['heldout_exposure_ns'] == test['exposure_ns'], 'Heldout count or exposure differs')
            if not stats['exposure_ns']:
                require(d['status'] == 'UNKNOWN_NO_TRAINING_EXPOSURE', 'Missing training support must remain UNKNOWN'); continue
            if not test['exposure_ns']:
                require(d['status'] == 'NO_HELDOUT_EXPOSURE', 'Empty heldout exposure cannot be scored'); continue
            a, t, h, n = stats['events']+.5,stats['exposure_ns']/1e9,test['exposure_ns']/1e9,test['events']
            p = t/(t+h)
            logprob = math.lgamma(n+a)-math.lgamma(a)-math.lgamma(n+1)+a*math.log(p)+n*math.log1p(-p)
            close(d['log_predictive_probability'],logprob); close(d['predictive_zero_probability'],math.exp(a*math.log(p)))
            close(d['expected_SAVE'],a*h/t)
            bounds = [nb_quantile(q,a,p) for q in (.025,.975)]
            require(d['posterior_predictive95_count'] == bounds and d['observed_within_predictive95'] == (bounds[0] <= n <= bounds[1]), 'Independent NB interval differs')
            score += logprob; scored += 1
        close(fold['sum_log_predictive_probability'],score)
        require(scored == fold['scored_strata'], 'Fold count differs')
    require(len(report['leave_one_whole_capture_out']) == 4, 'Four declared folds required')
    totals = {f:sum(model['fitted'][k][f] for k in KEYS) for f in FIELDS}
    require(totals == model['coverage']['totals'] == report['summary']['totals'], 'Coverage totals differ')
    for ref in sources+model['sources']:
        require(receipt(ref['path']) == ref, 'Review source changed')
    out.mkdir(parents=True,exist_ok=False)
    result = dict(schema='gsdeploy.projected_state_v57_actual_independent_CPU_review.v1',status='PASS',
        model=model_ref,author_report=report_ref,sources=sources,training_source_receipts_checked=len(model['sources']),
        risk_rows_independently_read=rows_read,independent_comparisons=dict(comparisons),totals=totals,
        all64_and_four_capture_counts_and_age_unions_rebuilt=True,
        posterior_and_four_fold_predictive_numbers_independently_recomputed=True,
        author_aggregation_or_posterior_helpers_used=False,
        fit_performed=False,GPU_started=False,validation_labels_read=False,
        independent_prediction_accuracy_validated=False,
        limitations=['This audit checks fit arithmetic and source identities, not Poisson adequacy, time calibration, physical service or queue/utility accuracy.',
                     'Whole-capture count intervals are overlapping training diagnostics. They do not identify event-time predictions.'])
    save_synced(out/'report.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','report','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();r=run(args.model,args.report,args.output)
    print(json.dumps({k:r[k] for k in ('status','risk_rows_independently_read','independent_comparisons','totals')}))
