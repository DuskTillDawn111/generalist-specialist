"""Predeclared age/horizon grid, leave-training-capture-out CDF diagnostics.

The two source captures remain training. Every group/grid query, including
unsupported cells, is retained. The held training capture is used offline only
to describe distribution disagreement; it never updates the fitted provider.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from .context_survival_v12_3 import ContextEnvelopeSurvival
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save

AGES = (0,1_000,10_000,50_000,100_000,500_000,1_000_000,5_000_000)
HORIZONS = (1_000,10_000,50_000,100_000,500_000,1_000_000)


def cdf(result,horizon):
    if horizon > result['restricted_residual_horizon_ns']:
        raise UnknownEvidence('CDF_HORIZON_OUTSIDE_OBSERVED_RISK_SUPPORT')
    return sum(r['probability'] for r in result['residual_event_mass'] if r['residual_ns']<=horizon)


def diagnose(model_path,output):
    out = Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    ref = receipt(model_path);model = checked(ref);began = time.monotonic()
    captures = sorted({r['capture'] for r in model['episodes']})
    if len(captures)!=2:
        raise ValueError('This two-capture diagnostic cannot silently choose a subset')
    # Source rederivation occurs in each provider; no CPU fixture opt-in.
    providers = {excluded:ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,
        allow_single_capture_training_diagnostic=True,exclude_training_capture=excluded) for excluded in captures}
    full = ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True)
    keys = sorted(full.groups,key=str)
    declared = [dict(held_out_training_capture=held,held_owner=owner,
                     frontier=[list(r[:3])+[list(r[3])] for r in frontier],context_age_ns=age)
                for held in captures for owner,frontier in keys for age in AGES]
    save(out/'declared-queries.json',dict(model=ref,age_grid_ns=AGES,horizon_grid_ns=HORIZONS,queries=declared,
        grid_selected_before_diagnostic_results=True,training_only=True))
    results = [];counts = Counter();errors = []
    for query in declared:
        held = query['held_out_training_capture'];other = next(c for c in captures if c!=held)
        values = {};reasons = {}
        # Excluding held => prediction from other run; excluding other =>
        # offline held empirical reference, never an online feature/parameter.
        for name,provider in [('training_prediction',providers[held]),('offline_held_reference',providers[other])]:
            try:
                values[name] = provider.query(query['held_owner'],query['frontier'],query['context_age_ns'])
            except UnknownEvidence as exc:
                reasons[name] = str(exc)
        row = dict(query,status='UNSUPPORTED' if reasons else 'COMMON_EMPIRICAL_SUPPORT',reasons=reasons,horizons=[])
        for horizon in HORIZONS:
            point = dict(horizon_ns=horizon)
            if reasons:
                point.update(status='UNKNOWN',reason='CONDITIONAL_GROUP_OR_AGE_UNSUPPORTED')
            else:
                try:
                    prediction = cdf(values['training_prediction'],horizon)
                    observed = cdf(values['offline_held_reference'],horizon)
                    error = abs(prediction-observed);errors.append(error)
                    point.update(status='EMPIRICAL_CDF_COMPARISON',predicted_CDF=prediction,
                                 offline_held_CDF=observed,absolute_error=error)
                except UnknownEvidence as exc:
                    point.update(status='UNKNOWN',reason=str(exc))
            counts[point['status']] += 1;row['horizons'].append(point)
        row['training_support'] = {name:{k:v for k,v in result.items() if k!='residual_event_mass'}
                                   for name,result in values.items()}
        results.append(row)
    save(out/'query-results.json',results)
    if receipt(ref['path'])!=ref:
        raise ValueError('Training survival model changed during diagnostic')
    report = dict(schema='gsdeploy.leave_training_capture_out_context_survival_diagnostic.v12_3',
        status='TRAINING_IDENTIFIABILITY_DIAGNOSTIC_COMPLETE',model=ref,declared_queries=receipt(out/'declared-queries.json'),
        query_results=receipt(out/'query-results.json'),capture_count=2,exact_groups=len(keys),
        query_count=len(declared),horizon_points=len(declared)*len(HORIZONS),counts=dict(counts),
        supported_CDF_mean_absolute_error=statistics.mean(errors) if errors else None,
        unknown_horizons_not_imputed=True,endpoint_MARE_not_reported_for_censored_data=True,
        supported_cells_are_not_population_accuracy=True,training_only=True,
        empirical_CDF_disagreement_is_not_probability_calibration=True,
        complete_dispatch_oracle_available=False,independent_validation=False,
        physical_service_identified=False,GPU_started=False,elapsed_seconds=time.monotonic()-began)
    save(out/'report.json',report);return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    args = parser.parse_args();result = diagnose(args.model,args.output)
    print(json.dumps({k:result[k] for k in ('status','query_count','horizon_points','counts','supported_CDF_mean_absolute_error')}))


if __name__ == '__main__':
    main()
