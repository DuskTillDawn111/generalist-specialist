"""Capture-balanced gap and prequeued-witness handoff approximation.

All SAVE->RESTORE gaps train the gap regressor. Missing service witnesses remain
in a separately modeled observation-probability domain, never zero targets.
Predicted witness delay is an approximation, not a counterfactual service bound.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .context_next_owner_v13 import RESIDENTS, features as base_features, feature_names as base_names
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.capture_balanced_handoff_candidate.v14'
LABEL_SCHEMA = 'gsdeploy.prequeued_context_handoff_witness.v14'
READINESS_RULES = ('lower_bound_reference_availability_hypothesis',
    'prequeued_upper_reference_availability_hypothesis')
PARAMETERS = dict(gap_model='L2_ridge_log1p_ns_over_1000',
    witness_delay_model='L2_ridge_log1p_ns_over_1000_observed_witness_only',
    witness_presence_model='L2_binary_logistic_all_transitions', regularization=1.,
    capture_balanced=True, intercept_regularized=True, max_iterations=300,
    optimizer='L-BFGS-B', hyperparameters_selected_from_folds=False)


def names(pool_tasks):
    return [n for n in base_names(pool_tasks) if not n.startswith('workload')
        and n not in ('pending_age_unavailable', 'pending_reference_work_unavailable')] + [
        *[f'next:{owner}' for owner in RESIDENTS],
        *[f'pair:{owner}:{target}' for owner in RESIDENTS for target in RESIDENTS],
        *[f'next_log_age:{owner}' for owner in RESIDENTS]]


def feature_row(*, held_owner, next_owner, context_age_ns, pending_frontier, pool_tasks):
    if next_owner not in RESIDENTS:
        raise ValueError('An already chosen known resident mark is required')
    query = dict(held_owner=held_owner, context_age_ns=context_age_ns, pending_frontier=pending_frontier,
        public_workload=[], pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None)
    original_names = base_names(pool_tasks)
    original = base_features(query, pool_tasks)
    values = [v for n, v in zip(original_names, original) if not n.startswith('workload')
        and n not in ('pending_age_unavailable', 'pending_reference_work_unavailable')]
    values += [float(owner == next_owner) for owner in RESIDENTS]
    values += [float(owner == held_owner and target == next_owner) for owner in RESIDENTS for target in RESIDENTS]
    values += [math.log1p(context_age_ns/1_000_000) if owner == next_owner else 0. for owner in RESIDENTS]
    return values


def row_query(row, pools):
    return dict(held_owner=row['held_owner'], next_owner=row['next_owner'],
        context_age_ns=row['context_age_at_save_ns'], pending_frontier=row['pending_frontier'], pool_tasks=pools)


def weights(captures):
    counts = Counter(captures)
    return np.asarray([1/(len(counts)*counts[c]) for c in captures])


def design_matrix(matrix):
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    center[0], scale[0] = 0., 1.
    scale[scale < 1e-12] = 1.
    return (matrix-center)/scale, center, scale


def ridge_fit(matrix, target_ns, captures):
    if (matrix.ndim != 2 or not len(matrix) or len(target_ns) != len(matrix)
            or len(captures) != len(matrix) or not np.isfinite(matrix).all()
            or not np.isfinite(target_ns).all() or np.any(target_ns < 0)):
        raise ValueError('Observed nonnegative handoff targets required')
    design, center, scale = design_matrix(matrix)
    balanced = weights(captures)
    target = np.log1p(target_ns/1000)
    lhs = design.T@(design*balanced[:,None])
    lhs += PARAMETERS['regularization']/len(matrix)*np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(lhs, design.T@(target*balanced))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist(),
        observed_rows=len(matrix), independent_captures=len(set(captures)))


def logistic_fit(matrix, present, captures):
    if (matrix.ndim != 2 or not len(matrix) or len(present) != len(matrix)
            or len(captures) != len(matrix) or not np.isfinite(matrix).all()
            or not np.isin(present, [0, 1]).all()):
        raise ValueError('Aligned finite features and binary witness-presence labels required')
    design, center, scale = design_matrix(matrix)
    balanced = weights(captures)
    penalty = PARAMETERS['regularization']/len(matrix)
    def objective(beta):
        logits = design@beta
        probability = np.exp(-np.logaddexp(0, -logits))
        loss = np.sum(balanced*(np.logaddexp(0, logits)-present*logits)) + .5*penalty*np.sum(beta**2)
        gradient = design.T@((probability-present)*balanced) + penalty*beta
        return float(loss), gradient
    optimized = minimize(objective, np.zeros(matrix.shape[1]), jac=True, method='L-BFGS-B',
        options=dict(maxiter=PARAMETERS['max_iterations'], gtol=1e-8, ftol=1e-12))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficients=optimized.x.tolist(),
        optimizer=dict(converged=bool(optimized.success), iterations=int(optimized.nit),
            message=str(optimized.message)), observed_rows=len(matrix),
        positive_rows=int(present.sum()), missing_rows=int(len(present)-present.sum()))


def linear_predict(model, matrix):
    return ((matrix-np.asarray(model['center']))/np.asarray(model['scale']))@np.asarray(model['coefficients'])


def ridge_predict(model, matrix):
    value = linear_predict(model, matrix)
    # The statistical centre cannot be a negative delay. A divergent OOD
    # prediction raises rather than wrapping an enormous integer timestamp.
    if not np.isfinite(value).all() or np.any(value > 50):
        raise ValueError('Nonfinite or numerically unsupported handoff extrapolation')
    return np.expm1(np.maximum(0, value))*1000


def presence_predict(model, matrix):
    return np.exp(-np.logaddexp(0, -linear_predict(model, matrix)))


def verify_labels(document):
    if (document.get('schema') != LABEL_SCHEMA or document.get('training_only') is not True
            or document.get('restore_is_exact_service_ready') is not False
            or document.get('observations_are_exogenous_counterfactual_bounds') is not False):
        raise ValueError('Actual training prequeued-witness labels required')
    seen = set()
    for row in document['labels']:
        key = row['capture'], row['save_end_ns']
        if key in seen:
            raise ValueError('Duplicate original handoff transition')
        seen.add(key)
        gap = row['save_to_restore_gap_ns']
        if (type(gap) is not int or gap < 0
                or gap != row['next_restore_start_ns']-row['save_end_ns']
                or row.get('exact_service_ready_observed') is not False):
            raise ValueError('Valid exact source SAVE-to-RESTORE envelope gap required')
        interval = row['service_after_restore_interval_ns']
        witness = row['prequeued_kernel_witness']
        if interval['lower_ns'] != 0:
            raise ValueError('RESTORE_START is only a lower readiness boundary')
        if witness is None:
            if interval['upper_ns'] is not None or not row['missing_reason']:
                raise ValueError('Missing witness must retain an unknown service upper delay')
        elif (row['missing_reason'] is not None or type(interval['upper_ns']) is not int
                or interval['upper_ns'] < 0 or witness['recorded_dependencies_known'] is not True
                or witness['certified_ready_bound_ns'] is None
                or witness['certified_ready_bound_ns'] > row['next_restore_start_ns']
                or witness['observed_start_ns']-row['next_restore_start_ns'] != interval['upper_ns']):
            raise ValueError('Actual source-certified prequeued kernel upper witness required')


def errors(actual, predicted):
    if not len(actual):
        return dict(rows=0, MAE_ns=None, median_absolute_percentage_error=None, p95_absolute_percentage_error=None)
    absolute = np.abs(actual-predicted)
    relative = absolute/np.maximum(actual, 1)
    return dict(rows=len(actual), MAE_ns=float(absolute.mean()),
        median_absolute_percentage_error=float(np.median(relative)),
        p95_absolute_percentage_error=float(np.quantile(relative, .95)))


def baseline_predict(train_rows, target, test_rows, *, pair=False):
    if (not len(train_rows) or len(train_rows) != len(target)
            or not np.isfinite(target).all() or np.any(np.asarray(target) < 0)):
        raise ValueError('Aligned observed nonnegative baseline targets required')
    grouped = defaultdict(list)
    for row, value in zip(train_rows, target):
        grouped[(row['held_owner'],row['next_owner']) if pair else ('all',)].append((row['capture'],value))
    estimates = {}
    for group, members in grouped.items():
        captures = [capture for capture, value in members]
        values = np.asarray([value for capture, value in members])
        estimates[group] = float(np.expm1(np.sum(weights(captures)*np.log1p(values/1000)))*1000)
    global_estimate = float(np.expm1(np.sum(weights([row['capture'] for row in train_rows])
        *np.log1p(np.asarray(target)/1000)))*1000) if pair else None
    output, missing = [], 0
    for row in test_rows:
        group = (row['held_owner'],row['next_owner']) if pair else ('all',)
        if group not in estimates:
            missing += 1
            output.append(global_estimate)
        else:
            output.append(estimates[group])
    return np.asarray(output), missing


class HandoffCandidate:
    def __init__(self, model, *, allow_cpu_fixture=False):
        if (model.get('schema') != SCHEMA or model.get('training_only') is not True
                or model.get('parameters') != PARAMETERS
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})):
            raise ValueError('Source-bound handoff approximation required')
        if model.get('CPU_fixture'):
            if allow_cpu_fixture is not True:
                raise ValueError('CPU fixture cannot qualify measured handoff support')
        else:
            for reference in (model['implementation'], model['feature_implementation'], model['source_labels']):
                if receipt(reference['path']) != reference:
                    raise ValueError('Handoff candidate source changed')
        if model['feature_names'] != names(model['public_pool_tasks']):
            raise ValueError('Handoff feature contract changed')
        width = len(model['feature_names'])
        for key in ('gap_model', 'witness_delay_model', 'witness_presence_model'):
            fitted = model[key]
            for field in ('center', 'scale', 'coefficients'):
                array = np.asarray(fitted[field])
                if array.shape != (width,) or not np.isfinite(array).all():
                    raise ValueError('Finite aligned handoff model arrays required')
            if np.any(np.asarray(fitted['scale']) <= 0):
                raise ValueError('Positive handoff feature scales required')
        minimum, maximum = np.asarray(model['feature_minimum']), np.asarray(model['feature_maximum'])
        if (minimum.shape != (width,) or maximum.shape != (width,)
                or not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
                or np.any(minimum > maximum)):
            raise ValueError('Aligned finite handoff feature ranges required')
        self.model = model

    def query(self, *, held_owner, next_owner, context_age_ns, pending_frontier, readiness_rule):
        if readiness_rule not in READINESS_RULES:
            raise ValueError('Explicit reference-availability readiness hypothesis required')
        feature = np.asarray(feature_row(held_owner=held_owner, next_owner=next_owner,
            context_age_ns=context_age_ns, pending_frontier=pending_frontier,
            pool_tasks=self.model['public_pool_tasks']))
        row = feature[None,:]
        gap = float(ridge_predict(self.model['gap_model'], row)[0])
        upper_proxy = float(ridge_predict(self.model['witness_delay_model'], row)[0])
        witness_probability = float(presence_predict(self.model['witness_presence_model'], row)[0])
        lower, upper = np.asarray(self.model['feature_minimum']), np.asarray(self.model['feature_maximum'])
        outside = np.flatnonzero((feature < lower-1e-12) | (feature > upper+1e-12))
        flags = ['FEATURE_OUTSIDE_TRAINING_RANGE:'+self.model['feature_names'][i] for i in outside]
        pair = held_owner+'->'+next_owner
        support = self.model['owner_pair_counts'].get(pair, dict(rows=0, prequeued_witness=0, missing=0))
        if support['rows'] == 0:
            flags.append('OWNER_PAIR_UNOBSERVED_STATISTICAL_TRANSFER')
        if support['prequeued_witness'] == 0:
            flags.append('NO_OBSERVED_PREQUEUED_WITNESS_FOR_OWNER_PAIR')
        service_delay = 0 if readiness_rule == READINESS_RULES[0] else round(upper_proxy)
        return dict(status='UNCALIBRATED_HANDOFF_REFERENCE_AVAILABILITY_CANDIDATE',
            save_to_restore_gap_ns=round(gap), service_ready_delay_after_restore_ns=service_delay,
            readiness_rule=readiness_rule, readiness_is_unvalidated_hypothesis=True,
            estimated_prequeued_upper_witness_delay_ns=upper_proxy,
            prequeued_witness_probability=witness_probability,
            missing_prequeued_witness_probability=1-witness_probability,
            source_missing_witness_mass_preserved=True, owner_pair_training_support=support,
            ood_flags=flags, model_sha256=self.model['model_sha256'],
            exact_service_ready_predicted=False, predicted_delay_is_counterfactual_bound=False,
            physical_SM_cost_identified=False, calibrated_probability=False,
            source_service_lower_boundary_after_restore_ns=0,
            full_availability_oracle=False)


def train(labels_path, output):
    source = receipt(labels_path)
    document = checked(source)
    verify_labels(document)
    # The producer already completely rederived these immutable source labels.
    # Recheck all its receipts here, without repeating that same expensive
    # whole-DAG extraction again in the same task.
    for reference in document['sources']:
        if receipt(reference['path']) != reference:
            raise ValueError('Prequeued-witness producer source changed')
    supervision_refs = [ref for ref in document['sources'] if Path(ref['path']).name == 'report.json'
        and 'context-initial-transition-v13' in ref['path']]
    if len(supervision_refs) != 1:
        raise ValueError('Unique full source-supervision report required')
    supervision = checked(supervision_refs[0])
    pools = {}
    for audit in supervision['training_audits']:
        plan = checked(audit['plan'])
        if plan.get('split') != 'training' or plan.get('training_only') is not True:
            raise ValueError('Explicit training handoff plans required')
        for row in plan['inputs']:
            previous = pools.setdefault(row['pool_id'], row['task'])
            if previous != row['task']:
                raise ValueError('Public input task changed')
    rows = document['labels']
    matrix = np.asarray([feature_row(**row_query(row,pools)) for row in rows])
    captures = np.asarray([row['capture'] for row in rows])
    gap = np.asarray([row['save_to_restore_gap_ns'] for row in rows])
    present = np.asarray([row['prequeued_kernel_witness'] is not None for row in rows])
    upper = np.asarray([row['service_after_restore_interval_ns']['upper_ns'] if is_present else np.nan
        for row, is_present in zip(rows,present)])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'predeclared-configuration.json', dict(parameters=PARAMETERS, feature_names=names(pools),
        source_labels=source, readiness_rules=list(READINESS_RULES),
        folds='Exclude each complete original raw-SHA capture; no tuning from diagnostics',
        gap_domain='All source transitions including those with no prequeued kernel witness',
        witness_regression_domain='Observed prequeued upper witnesses only; missing labels are not zero',
        missingness_domain='All original transitions',
        baselines='Capture-balanced log-centre constant and outgoing/incoming-holder pair', GPU_started=False))
    folds = []
    for excluded in sorted(set(captures)):
        training, testing = captures != excluded, captures == excluded
        observed_training, observed_testing = training & present, testing & present
        trained_rows = [r for r in rows if r['capture'] != excluded]
        test_rows = [r for r in rows if r['capture'] == excluded]
        trained_observed = [r for r in trained_rows if r['prequeued_kernel_witness'] is not None]
        test_observed = [r for r in test_rows if r['prequeued_kernel_witness'] is not None]
        gap_fit = ridge_fit(matrix[training],gap[training],captures[training])
        upper_fit = ridge_fit(matrix[observed_training],upper[observed_training],captures[observed_training])
        presence_fit = logistic_fit(matrix[training],present[training],captures[training])
        gap_pred = ridge_predict(gap_fit,matrix[testing])
        upper_pred = ridge_predict(upper_fit,matrix[observed_testing])
        presence_pred = presence_predict(presence_fit,matrix[testing])
        baselines = {}
        for name, pair in [('constant',False),('holder_pair',True)]:
            gb, gm = baseline_predict(trained_rows,gap[training],test_rows,pair=pair)
            ub, um = baseline_predict(trained_observed,upper[observed_training],test_observed,pair=pair)
            baselines[name] = dict(gap=errors(gap[testing],gb), witness_upper_proxy=errors(upper[observed_testing],ub),
                gap_pair_fallbacks=gm,witness_pair_fallbacks=um)
        minimum, maximum = matrix[training].min(axis=0), matrix[training].max(axis=0)
        folds.append(dict(excluded_capture=excluded, requested_transitions=len(test_rows),
            gap_predictions=len(test_rows), gap_UNKNOWN=0, witness_observed=int(present[testing].sum()),
            witness_missing=int((~present[testing]).sum()), missing_witness_labels_removed=False,
            gap_candidate=errors(gap[testing],gap_pred),
            observed_witness_upper_proxy_candidate=errors(upper[observed_testing],upper_pred),
            witness_presence_Brier=float(np.mean((presence_pred-present[testing])**2)),
            witness_presence_constant_Brier=float(np.mean((np.mean(present[training])-present[testing])**2)),
            baselines=baselines,witness_presence_optimizer=presence_fit['optimizer'],
            feature_range_OOD_rows=int(np.sum(np.any((matrix[testing]<minimum-1e-12)|(matrix[testing]>maximum+1e-12),axis=1))),
            observed_upper_accuracy_is_not_service_ready_accuracy=True, independent_validation=False))
    pair_counts = {}
    for row in rows:
        key = row['held_owner']+'->'+row['next_owner']
        counts = pair_counts.setdefault(key,dict(rows=0,prequeued_witness=0,missing=0))
        counts['rows'] += 1
        counts['prequeued_witness' if row['prequeued_kernel_witness'] is not None else 'missing'] += 1
    model = dict(schema=SCHEMA, training_only=True, parameters=PARAMETERS, public_pool_tasks=pools,
        feature_names=names(pools), feature_minimum=matrix.min(axis=0).tolist(), feature_maximum=matrix.max(axis=0).tolist(),
        gap_model=ridge_fit(matrix,gap,captures),
        witness_delay_model=ridge_fit(matrix[present],upper[present],captures[present]),
        witness_presence_model=logistic_fit(matrix,present,captures),
        owner_pair_counts=pair_counts, source_labels=source, implementation=receipt(__file__),
        feature_implementation=receipt(Path(__file__).with_name('context_next_owner_v13.py')),
        producer_source_receipts_reverified=True, producer_original_source_rederivation_repeated=False,
        missing_prequeued_witness_rows_retained=int((~present).sum()),
        witness_delay_target_is_interval_upper_not_exact_service=True,
        predicted_witness_delay_is_not_counterfactual_bound=True, independent_accuracy_validated=False,GPU_started=False)
    model['model_sha256'] = digest(model)
    save(output/'model.json', model)
    report = dict(schema='gsdeploy.handoff_candidate_grouped_training_diagnostic.v14',
        status='FIXED_HANDOFF_APPROXIMATION_TRAINING_ONLY_MISSING_WITNESSES_RETAINED',
        model=receipt(output/'model.json'), configuration=receipt(output/'predeclared-configuration.json'),
        requested_transitions=len(rows), observed_witness_rows=int(present.sum()),
        missing_witness_rows=int((~present).sum()), owner_pair_counts=pair_counts,
        independent_captures=len(set(captures)), folds=folds,
        full_witness_presence_optimizer=model['witness_presence_model']['optimizer'],
        six_term_utility_evaluated=False, independent_validation=False,
        full_availability_oracle=False, GPU_started=False)
    save(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = train(args.labels,args.output)
    print(json.dumps(dict(status=report['status'], requested_transitions=report['requested_transitions'],
        observed_witness_rows=report['observed_witness_rows'],missing_witness_rows=report['missing_witness_rows'],
        folds=report['folds'], report=receipt(Path(args.output)/'report.json'))))


if __name__ == '__main__':
    main()
