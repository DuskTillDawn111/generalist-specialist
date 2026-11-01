"""Holder-specific idle physical SAVE counting-process candidate, CPU only.

The original v41 training projection observes four handler booleans, not the
runtime twelve-field idle state or GPU frontier. Its transfer is explicit and
unvalidated. This module supplies a rate only; marks and cumulative budgets
are untouched. Zero event exposure is never interpreted as a zero hazard.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import gamma, nbinom

from gsdeploy.forecast_errors import UnknownEvidence
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import FIELDS, state_features
from .profiling import digest
from .projected_handler_context_v42 import checked, load_dataset, validate_risk
from .unknown_preflight_archive_v1 import save_synced
from .whole57_shared_observable_risk_v31 import OWNERS

SCHEMA = 'gsdeploy.idle_physical_SAVE_counting_process.v53'
PARAMETERS = dict(
    training_population='v41 qualified paired risk with all four projected handler_active exactly false',
    strata='held_owner only; no pooling or borrowing from active/other owners',
    likelihood='independent_increment_homogeneous_Poisson_in_accumulated_idle_exposure',
    prior='improper_Jeffreys_density_proportional_to_lambda_power_minus_one_half',
    posterior_shape='N+0.5', posterior_rate='T_seconds',
    runtime_point='posterior_mean_in_seconds_inverse_divided_by_1000_for_ms_inverse',
    interval='equal_tailed_95_percent_Gamma_posterior_credible_not_frequentist_confidence',
    no_exposure='UNKNOWN', runtime_domain='all twelve lifecycle fields exactly false AND empty frontier',
    training_to_runtime_transfer='UNVALIDATED_HANDLER_ONLY_TO_COMPLETE_IDLE_BRIDGE',
    age_dependence='constant_rate_hypothesis_not_identified_from_zero_events',
    holdout='four leave_one_whole_training_capture_out; no hyperparameter selection',
    mark_replaced=False, cumulative_budget_replaced=False,
    physical_context_equals_SM_service=False, independent_validation=False)


def aggregate(risks, captures):
    result = {capture:{owner:dict(rows=0, exposure_ns=0, events=0,
        unresolved_rows=0, unresolved_exposure_ns=0, unresolved_events=0,
        age_min_ns=None, age_max_ns=None) for owner in OWNERS} for capture in captures}
    seen = set()
    for row in risks:
        validate_risk(row)
        if row['capture'] not in result:
            raise ValueError('Risk cannot cross declared capture identities')
        if any(v is not False for v in row['shared_left_features']['handler_active'].values()):
            continue
        key = (row['capture'], row['held_owner'], tuple(row['driver_context']), row['start_ns'], row['end_ns'])
        if key in seen:
            raise ValueError('Idle risk exposure cannot be duplicated')
        seen.add(key)
        value = result[row['capture']][row['held_owner']]
        width, event = row['end_ns']-row['start_ns'], int(row['physical_SAVE_END_observed'])
        if not row['exposure_qualified']:
            value['unresolved_rows'] += 1; value['unresolved_exposure_ns'] += width
            value['unresolved_events'] += event
            continue
        value['rows'] += 1; value['exposure_ns'] += width; value['events'] += event
        lo, hi = row['age_entry_ns'], row['age_exit_ns']
        value['age_min_ns'] = lo if value['age_min_ns'] is None else min(value['age_min_ns'], lo)
        value['age_max_ns'] = hi if value['age_max_ns'] is None else max(value['age_max_ns'], hi)
    return result


def pool(counts, captures, owner):
    values = [counts[c][owner] for c in captures]
    result = {k:sum(v[k] for v in values) for k in ('rows', 'exposure_ns', 'events',
        'unresolved_rows', 'unresolved_exposure_ns', 'unresolved_events')}
    lower = [v['age_min_ns'] for v in values if v['age_min_ns'] is not None]
    upper = [v['age_max_ns'] for v in values if v['age_max_ns'] is not None]
    result.update(age_min_ns=min(lower) if lower else None, age_max_ns=max(upper) if upper else None)
    return result


def posterior(stats):
    n, t = stats['events'], stats['exposure_ns']
    if type(n) is not int or n < 0 or type(t) is not int or t < 0 or (t == 0 and n):
        raise ValueError('Nonnegative integer SAVE count and exact exposure ns required')
    if t == 0:
        return dict(status='UNKNOWN_NO_EXPOSURE', **stats, shape=None, rate_seconds=None,
                    hazard_per_second=None, hazard_per_ms=None, credible95_per_second=None)
    seconds, shape = t/1e9, n+.5
    bounds = gamma.ppf([.025, .975], a=shape, scale=1/seconds)
    return dict(status='FITTED_UNVALIDATED_COUNTING_PROCESS', **stats, shape=shape,
                rate_seconds=seconds, hazard_per_second=shape/seconds,
                hazard_per_ms=shape/seconds/1000,
                credible95_per_second=[float(v) for v in bounds],
                zero_event_frequentist_one_sided95_upper_per_second=-math.log(.05)/seconds if n == 0 else None,
                zero_events_do_not_identify_zero_hazard=True,
                credible_interval_lower_positive_is_not_observed_event_evidence=True)


def predictive(fitted, heldout):
    seconds, n = heldout['exposure_ns']/1e9, heldout['events']
    base = dict(observed_SAVE=n, heldout_exposure_ns=heldout['exposure_ns'])
    if fitted['status'] == 'UNKNOWN_NO_EXPOSURE':
        return dict(status='UNKNOWN_NO_TRAINING_EXPOSURE', **base)
    if seconds == 0:
        return dict(status='NO_HELDOUT_EXPOSURE', **base)
    shape, rate = fitted['shape'], fitted['rate_seconds']
    p = rate/(rate+seconds)
    bounds = nbinom.ppf([.025, .975], shape, p)
    return dict(status='COUNT_PREDICTIVE_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION', **base,
                expected_SAVE=fitted['hazard_per_second']*seconds,
                posterior_predictive95_count=[int(x) for x in bounds],
                observed_within_predictive95=bool(bounds[0] <= n <= bounds[1]),
                log_predictive_probability=float(nbinom.logpmf(n, shape, p)),
                predictive_zero_probability=float(nbinom.pmf(0, shape, p)),
                gamma_Poisson_predictive='negative_binomial(shape,p=training_T/(training_T+heldout_T))')


def train(manifest_path, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    manifest_ref = receipt(manifest_path); manifest = checked(manifest_ref)
    if (set(manifest) != {'training_only', 'validation_labels_used', 'captures'}
            or manifest['training_only'] is not True or manifest['validation_labels_used'] is not False):
        raise ValueError('Exact original four-capture training-only manifest required')
    data = load_dataset(manifest['captures'])
    captures = sorted(data['captures'])
    counts = aggregate(data['risks'], captures)
    fitted = {owner:posterior(pool(counts, captures, owner)) for owner in OWNERS}
    folds = []
    for capture in captures:
        training = [c for c in captures if c != capture]
        models = {o:posterior(pool(counts, training, o)) for o in OWNERS}
        folds.append(dict(held_out_capture=capture, held_out_route=data['captures'][capture]['route'],
            training_captures=training, fitted=models,
            diagnostics={o:predictive(models[o], counts[capture][o]) for o in OWNERS}))
    implementations = implementation_sources(__file__)
    sources = strict_references([manifest_ref, data['sources'], implementations])
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Training source changed')
    model = dict(schema=SCHEMA, parameters=PARAMETERS, training_only=True, validation_labels_used=False,
        captures=data['captures'], by_capture=counts, fitted=fitted, sources=sources,
        original_training_manifest=manifest_ref, implementation=implementations,
        software=dict(scipy=__import__('scipy').__version__), idle_physical_mark_identified=False,
        idle_mark_rows=sum(all(v is False for v in r['features']['handler_active'].values()) for r in data['marks']))
    model['model_sha256'] = digest(model)
    summary = dict(rows=sum(v['rows'] for v in fitted.values()), exposure_ns=sum(v['exposure_ns'] for v in fitted.values()),
                   physical_SAVE=sum(v['events'] for v in fitted.values()), idle_marks=model['idle_mark_rows'])
    out.mkdir(parents=True, exist_ok=False)
    save_synced(out/'training-manifest.json', dict(schema=SCHEMA, parameters=PARAMETERS,
        original=manifest_ref, sources=sources, GPU_started=False, validation_labels_read=False))
    save_synced(out/'model.json', model)
    report = dict(schema=SCHEMA, status='CPU_TRAINED_UNVALIDATED_IDLE_HAZARD_CANDIDATE', model=receipt(out/'model.json'),
        training_manifest=receipt(out/'training-manifest.json'), summary=summary, by_owner=fitted,
        leave_one_whole_capture_out=folds, sources=sources, GPU_started=False, validation_labels_read=False,
        independent_prediction_accuracy_validated=False, rate_constancy_testable_with_zero_events=False,
        idle_mark_identified=False, four_folds_are_overlapping_training_diagnostics=True,
        limitations=['Four handler-mask booleans do not measure runtime twelve-field idle state or frontier.',
          'Physical-clock projection is uncalibrated; state/clock error may contaminate idle exposure.',
          'Within-owner constant intensity, independent increments and cross-route exchangeability are hypotheses.',
          'All-zero counts provide upper constraints, not evidence for age shape, rate constancy or positive lower bounds.',
          'Jeffreys improper prior is fixed; positive posterior mean and credible lower bound are prior-dependent.',
          'Unobserved owner exposure is UNKNOWN; no borrowing or mark invention.',
          'Physical SAVE_END is not SM service progress, active target selection, queue accuracy or utility gain.'])
    for ref in sources:
        if receipt(ref['path']) != ref:
            raise ValueError('Training source changed during fit')
    save_synced(out/'report.json', report)
    return report


class IdlePhysicalSaveHazard:
    """Dispatch a new idle rate, preserving the original active v42 query object."""
    def __init__(self, model, fallback):
        if (model.get('schema') != SCHEMA or model.get('parameters') != PARAMETERS
                or model.get('training_only') is not True or model.get('validation_labels_used') is not False
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})
                or set(model.get('fitted', {})) != set(OWNERS)):
            raise ValueError('Exact immutable v53 training candidate required')
        for owner, value in model['fitted'].items():
            stats = {k:value[k] for k in ('rows', 'exposure_ns', 'events', 'unresolved_rows',
                'unresolved_exposure_ns', 'unresolved_events', 'age_min_ns', 'age_max_ns')}
            if posterior(stats) != value:
                raise ValueError('Posterior must match exact unpooled N/T exposure')
        for ref in model['sources']:
            if receipt(ref['path']) != ref:
                raise ValueError('v53 source drift')
        self.model, self.fallback = model, fallback

    def query(self, held_owner, pending_frontier, owner_lifecycle, context_age_ns, horizon_ns=0):
        state_features(owner_lifecycle)
        if (held_owner not in OWNERS or type(context_age_ns) is not int or context_age_ns < 0
                or type(horizon_ns) is not int or horizon_ns < 0):
            raise ValueError('Known held owner and nonnegative integer age/horizon required')
        all_idle = not pending_frontier and all(getattr(owner_lifecycle[o], f) is False for o in OWNERS for f in FIELDS)
        if not all_idle:
            return self.fallback.query(held_owner, pending_frontier, owner_lifecycle, context_age_ns, horizon_ns)
        value = self.model['fitted'][held_owner]
        if value['status'] == 'UNKNOWN_NO_EXPOSURE':
            raise UnknownEvidence('IDLE_PHYSICAL_SAVE_NO_HELD_OWNER_EXPOSURE')
        rate = value['hazard_per_ms']
        flags = ['UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION',
                 'UNVALIDATED_HANDLER_ONLY_TO_COMPLETE_IDLE_BRIDGE',
                 'UNVALIDATED_CONSTANT_IDLE_PHYSICAL_SAVE_RATE']
        if value['events'] == 0:
            flags.append('ZERO_EVENT_IDLE_RATE_USES_FIXED_JEFFREYS_POSTERIOR_MEAN')
        if not value['age_min_ns'] <= context_age_ns <= value['age_max_ns']:
            flags.append('IDLE_AGE_OUTSIDE_OBSERVED_EXPOSURE_RANGE')
        if context_age_ns+horizon_ns > value['age_max_ns']:
            flags.append('IDLE_HORIZON_OUTSIDE_OBSERVED_EXPOSURE_RANGE')
        seconds = horizon_ns/1e9
        return dict(hazard_per_ms=rate, OOD_reasons=flags, restricted_horizon_ns=horizon_ns,
            conditional_CDF=None, restricted_candidate_CDF=-math.expm1(-rate*horizon_ns/1e6),
            posterior_predictive_CDF=-math.expm1(-value['shape']*math.log1p(seconds/value['rate_seconds'])),
            target='physical_context_SAVE_END_not_SM_service', probability_calibrated=False,
            state_held_fixed_through_horizon=True, idle_hazard_version=53,
            model_sha256=self.model['model_sha256'], mark_replaced=False, budget_redrawn=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = train(args.training_manifest, args.output)
    print(json.dumps(dict(status=result['status'], summary=result['summary'])))
