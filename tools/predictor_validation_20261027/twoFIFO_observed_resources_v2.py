"""Measured fixed case energy and owned peak; no resource forecast or utility.

One attempt has one startup. Each measurement window excludes its drain. Owned
memory retains inactive residents and its cumulative history before that drain.
"""
import argparse
import json
import tempfile
from pathlib import Path

from gsdeploy.energy import integrate_power
from .twoFIFO_owned_training_review_v3 import review as recheck, require
from .stage_response_features_v1 import checked, receipt, stamp
from .implementation_source_closure_v27 import implementation_sources, strict_references


def observed(raw):
    boundaries={b['boundary']:b['time_ns'] for b in raw['lifecycle_boundaries']}
    samples=raw['board_samples'];times=[s['time_ns'] for s in samples]
    require(samples and all(stamp(t) for t in times) and times==sorted(times), 'Original ordered board polls required')
    require(raw.get('shutdown_memory_sampled') is False and times[-1]<=boundaries['monitoring_end']
            <boundaries['shutdown_begin'], 'All sampled memory must precede worker close')
    windows=raw['measurement_windows'];first=windows[0]['start_ns'];startup=boundaries['startup_begin']
    require(startup<=boundaries['priming_end']<=first, 'Startup and priming must precede first case')
    startup_energy=integrate_power(samples,startup,first)
    rows=[];last=None
    for window in windows:
        a,b,d=(window[k] for k in ('start_ns','end_ns','fixed_drain_deadline_ns'))
        require(all(stamp(t) for t in (a,b,d)) and a<b<d and (last is None or last<=a),
                'Distinct fixed measurement and drain windows required')
        before=[s['owned_used_mib'] for s in samples if startup<=s['time_ns']<=d]
        local=[s['owned_used_mib'] for s in samples if a<=s['time_ns']<=d]
        energy=integrate_power(samples,a,b)
        rows.append(dict(case_id=window['case_id'],measurement_start_ns=a,measurement_end_ns=b,
            fixed_drain_deadline_ns=d,measurement_energy_joules=energy,
            measurement_power_watts=None if energy is None else energy/((b-a)/1e9),
            drain_energy_joules_diagnostic_only=integrate_power(samples,b,d),
            startup_through_this_drain_owned_sampled_peak_mib=max(before) if before else None,
            local_measurement_through_drain_owned_sampled_peak_mib=max(local) if local else None,
            startup_energy_assigned_to_case=False,drain_in_measurement_energy=False,
            memory_safety_factor_applied=False,continuous_memory_capacity_bound=False))
        last=d
    require([r['case_id'] for r in rows]==[c['case_id'] for c in raw['cases']], 'Every case resource denominator required')
    return dict(schema='gsdeploy.twoFIFO_observed_case_resources.v2',
        status='MEASURED_CASE_RESOURCES_ONLY_NOT_A_FORECAST',cases=rows,
        startup=dict(occurrences=1,start_ns=startup,end_ns=first,energy_joules=startup_energy,
            priming_included=True,amortization_applied=False,amortization_seconds_for_original_scorer=3600),
        all_startup_through_final_drain_owned_sampled_peak_mib=max(
            (s['owned_used_mib'] for s in samples if startup<=s['time_ns']<=last),default=None),
        original_polls_retained=len(samples),power_interpolation_max_gap_seconds=2.0,
        window_count=len(windows),memory_safety_factor_applied=False,
        resource_forecast_qualified=False,independent_prediction_accuracy_validated=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def extract(review_path):
    review_ref=receipt(review_path);review=checked(review_ref)
    require(review.get('status')=='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
            and review.get('training_feature_extraction_eligible') is True and review.get('issues')==[]
            and review.get('validation_labels_read') is False and review.get('execution') is not None,
            'Complete native new-schema review required')
    with tempfile.TemporaryDirectory(prefix='twoFIFO-resource-review-') as temp:
        rebuilt=recheck(review['raw']['path'],Path(temp)/'review',review['execution']['path'])
        require(checked(rebuilt)==review,'Native resource evidence review must rederive exactly')
    raw=checked(review['raw']);result=observed(raw)
    result.update(training_review=review_ref,raw=review['raw'],capture_id=raw['capture_id'],
        attempt_id=raw['attempt_id'],execution_id=raw['execution_id'])
    result['sources']=strict_references([review_ref,review['sources'],implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in result['sources']), 'Resource source changed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-review',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();result=extract(a.training_review)
    with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(cases=result['window_count'],startup=result['startup'])))
