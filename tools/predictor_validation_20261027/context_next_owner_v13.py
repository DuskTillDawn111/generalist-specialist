"""Fixed CPU multinomial next-context mark candidate, conditional on SAVE now.

This predicts a restored owner, including residents with no pending kernel. It
does not choose a SAVE time, initialize a holder, or invent a service-ready time.
Complete-capture cross checks are training diagnostics, not heldout gate results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .context_initial_transition_v13 import SCHEMA as SUPERVISION_SCHEMA
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.continuous_age_multinomial_next_owner_candidate.v13'
RESIDENTS = ('generalist_2#0', 'ocr#0', 'text#0', 'text#1')
TASKS = ('text', 'image', 'asr', 'ocr')
PHASES = ('outside_language_forwards', 'prefill', 'decode', 'ocr_inference')
PARAMETERS = dict(model='L2_multinomial_logistic', regularization=1.,
    optimizer='L-BFGS-B', max_iterations=300, gradient_tolerance=1e-8,
    capture_balanced_loss=True, age_transform='log1p(context_age_ns/1e6) and square',
    intercept_regularized=True, hyperparameter_selection_from_folds=False)


def feature_names(pools):
    names = ['intercept', 'log_age_ms', 'log_age_ms_squared',
        'pending_age_unavailable', 'pending_reference_work_unavailable']
    for owner in RESIDENTS:
        names += [f'held:{owner}', f'held_log_age:{owner}', f'held_log_age_squared:{owner}',
            f'pending:{owner}', f'workload_count:{owner}', f'workload_max_log_offset_ms:{owner}']
        for task in TASKS:
            names += [f'pending_task:{owner}:{task}', f'workload_task_count:{owner}:{task}']
        for phase in PHASES:
            names += [f'pending_phase:{owner}:{phase}']
        for pool in sorted(pools):
            names += [f'pending_pool:{owner}:{pool}', f'workload_pool_count:{owner}:{pool}']
    return names


def query_from_row(row):
    return dict(held_owner=row['held_owner'], context_age_ns=row['context_age_at_save_ns'],
        pending_frontier=row['pending_frontier'], public_workload=row['public_workload'],
        pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None)


def features(query, pool_tasks):
    required = {'held_owner', 'context_age_ns', 'pending_frontier', 'public_workload',
        'pending_age_ns_by_owner', 'pending_reference_work_ns_by_owner'}
    if set(query) != required:
        raise ValueError('Only explicit cutoff query fields accepted; no future label/timing fields')
    owner, age = query['held_owner'], query['context_age_ns']
    if owner not in RESIDENTS or type(age) is not int or age < 0:
        raise ValueError('Known held resident and nonnegative integer attained context age required')
    if query['pending_age_ns_by_owner'] is not None or query['pending_reference_work_ns_by_owner'] is not None:
        raise ValueError('This first candidate has no independently reconstructed training pending-age/work covariates; do not silently supply future residual work')
    value = defaultdict(float)
    log_age = math.log1p(age / 1_000_000)
    value.update(intercept=1., log_age_ms=log_age, log_age_ms_squared=log_age**2,
        pending_age_unavailable=1., pending_reference_work_unavailable=1.)
    value[f'held:{owner}'] = 1.
    value[f'held_log_age:{owner}'] = log_age
    value[f'held_log_age_squared:{owner}'] = log_age**2
    def identity(alias, task, pool):
        if alias not in RESIDENTS or task not in TASKS or pool_tasks.get(pool) != task:
            raise ValueError('Known public resident/task/input identity required')
        if (alias.startswith('text#') and task != 'text') or (alias == 'ocr#0' and task != 'ocr'):
            raise ValueError('Public owner does not support this task')
    pending_owners = set()
    for row in query['pending_frontier']:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError('Explicit owner/task/pool/semantic-phases frontier required')
        alias, task, pool, phases = row
        identity(alias, task, pool)
        if alias in pending_owners or not phases or any(p not in PHASES for p in phases) or len(set(phases)) != len(phases):
            raise ValueError('Distinct pending owners and supported semantic phases required')
        pending_owners.add(alias)
        value[f'pending:{alias}'] = 1.
        value[f'pending_task:{alias}:{task}'] = 1.
        value[f'pending_pool:{alias}:{pool}'] = 1.
        for phase in phases:
            value[f'pending_phase:{alias}:{phase}'] = 1.
    for row in query['public_workload']:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError('Full public owner/task/input/declared-offset workload required')
        alias, task, pool, offset = row
        identity(alias, task, pool)
        if type(offset) is not int or offset < 0:
            raise ValueError('Nonnegative integer declared workload offset required')
        value[f'workload_count:{alias}'] += 1.
        value[f'workload_task_count:{alias}:{task}'] += 1.
        value[f'workload_pool_count:{alias}:{pool}'] += 1.
        key = f'workload_max_log_offset_ms:{alias}'
        value[key] = max(value[key], math.log1p(offset / 1_000_000))
    return [value[name] for name in feature_names(pool_tasks)]


def softmax(logits):
    values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_matrix(matrix, labels, captures):
    n, width = matrix.shape
    if not n or len(labels) != n or len(captures) != n:
        raise ValueError('Nonempty aligned training feature/label/capture rows required')
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    center[0], scale[0] = 0., 1.
    scale[scale < 1e-12] = 1.
    design = (matrix-center)/scale
    counts = Counter(captures)
    weights = np.asarray([1/(len(counts)*counts[c]) for c in captures])
    targets = np.eye(len(RESIDENTS))[labels]
    regularization = PARAMETERS['regularization']/n
    def objective(flat):
        coefficients = flat.reshape(width, len(RESIDENTS))
        logits = design@coefficients
        maximum = np.max(logits, axis=1)
        normalizer = maximum + np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
        loss = np.sum(weights*(normalizer-logits[np.arange(n), labels]))
        probabilities = softmax(logits)
        loss += .5*regularization*np.sum(coefficients**2)
        gradient = design.T@((probabilities-targets)*weights[:,None]) + regularization*coefficients
        return float(loss), gradient.ravel()
    optimized = minimize(objective, np.zeros(width*len(RESIDENTS)), jac=True, method='L-BFGS-B',
        options=dict(maxiter=PARAMETERS['max_iterations'], gtol=PARAMETERS['gradient_tolerance'], ftol=1e-12))
    coefficients = optimized.x.reshape(width, len(RESIDENTS))
    if not np.isfinite(coefficients).all():
        raise ValueError('Nonfinite multinomial training fit')
    ages = defaultdict(list)
    for row in matrix:
        for index, owner in enumerate(RESIDENTS):
            # Feature names make the location stable; do not infer owner from label.
            held_column = 5 + index*((width-5)//len(RESIDENTS))
            if row[held_column] == 1:
                ages[owner].append(float(row[1]))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist(),
        feature_minimum=matrix.min(axis=0).tolist(), feature_maximum=matrix.max(axis=0).tolist(),
        held_log_age_ranges={owner:[min(values),max(values)] for owner,values in ages.items()},
        label_counts={owner:int(np.sum(labels == i)) for i,owner in enumerate(RESIDENTS)},
        optimizer=dict(converged=bool(optimized.success), iterations=int(optimized.nit),
            message=str(optimized.message), objective=float(optimized.fun)))


def predict_matrix(fitted, matrix):
    return softmax(((matrix-np.asarray(fitted['center']))/np.asarray(fitted['scale']))@
        np.asarray(fitted['coefficients']))


def evaluate(labels, predicted, rows):
    correct = predicted == labels
    not_pending = np.asarray([not r['next_owner_is_pending'] for r in rows])
    confusion = {owner:{other:0 for other in (*RESIDENTS, 'UNKNOWN')} for owner in RESIDENTS}
    for actual, chosen in zip(labels, predicted):
        confusion[RESIDENTS[int(actual)]][RESIDENTS[int(chosen)] if chosen >= 0 else 'UNKNOWN'] += 1
    return dict(requested_transitions=len(rows), predicted_transitions=int(np.sum(predicted >= 0)),
        unknown_transitions=int(np.sum(predicted < 0)),
        correct=int(correct.sum()), accuracy=float(correct.mean()),
        target_not_pending_count=int(not_pending.sum()),
        target_not_pending_correct=int(correct[not_pending].sum()),
        target_not_pending_accuracy=float(correct[not_pending].mean()) if not_pending.any() else None,
        confusion=confusion)


class NextOwnerCandidate:
    """Mark-only API for a scheduler composing separate survival/service models."""
    def __init__(self, model, *, allow_cpu_fixture=False):
        if (model.get('schema') != SCHEMA or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})
                or model.get('parameters') != PARAMETERS):
            raise ValueError('Source-bound fixed multinomial candidate required')
        if model.get('CPU_fixture'):
            if allow_cpu_fixture is not True:
                raise ValueError('CPU fixture cannot qualify a measured next-owner candidate')
        else:
            if receipt(model['implementation']['path']) != model['implementation']:
                raise ValueError('Next-owner candidate implementation changed')
            checked(model['source_supervision'])
        if (model.get('feature_names') != feature_names(model['public_pool_tasks'])
                or model.get('residents') != list(RESIDENTS)):
            raise ValueError('Next-owner feature/class contract changed')
        fitted = model['fitted']
        width = len(model['feature_names'])
        if (np.asarray(fitted['coefficients']).shape != (width, len(RESIDENTS))
                or any(np.asarray(fitted[k]).shape != (width,) for k in (
                    'center', 'scale', 'feature_minimum', 'feature_maximum'))
                or any(not np.isfinite(np.asarray(fitted[k])).all() for k in (
                    'coefficients', 'center', 'scale', 'feature_minimum', 'feature_maximum'))
                or np.any(np.asarray(fitted['scale']) <= 0)):
            raise ValueError('Finite aligned multinomial model arrays required')
        self.model = model

    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            public_workload, pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        query = dict(held_owner=held_owner, context_age_ns=context_age_ns, pending_frontier=pending_frontier,
            public_workload=public_workload, pending_age_ns_by_owner=pending_age_ns_by_owner,
            pending_reference_work_ns_by_owner=pending_reference_work_ns_by_owner)
        row = np.asarray(features(query, self.model['public_pool_tasks']))
        fitted = self.model['fitted']
        minimum, maximum = np.asarray(fitted['feature_minimum']), np.asarray(fitted['feature_maximum'])
        outside = np.flatnonzero((row < minimum-1e-12) | (row > maximum+1e-12))
        names = self.model['feature_names']
        warnings = ['FEATURE_OUTSIDE_TRAINING_RANGE:'+names[i] for i in outside]
        ranges = fitted['held_log_age_ranges'].get(held_owner)
        if ranges is None or not ranges[0] <= row[1] <= ranges[1]:
            warnings.append('HELD_OWNER_AGE_OUTSIDE_TRAINING_SUPPORT')
        probability = predict_matrix(fitted, row[None,:])[0]
        return dict(status='UNCALIBRATED_MARK_CANDIDATE_WITH_EXTRAPOLATION' if warnings else 'UNCALIBRATED_MARK_CANDIDATE',
            next_owner_probabilities={owner:float(probability[i]) for i,owner in enumerate(RESIDENTS)},
            most_likely_next_owner=RESIDENTS[int(probability.argmax())],
            zero_observed_training_target_classes=[owner for owner,count in fitted['label_counts'].items() if count == 0],
            observed_training_target_counts=dict(fitted['label_counts']),
            ood_flags=warnings, model_sha256=self.model['model_sha256'],
            conditioned_on_observed_or_separately_predicted_SAVE_now=True,
            save_time_predicted=False, restore_time_predicted=False, service_ready_time_predicted=False,
            physical_SM_service_identified=False, calibrated_probability=False,
            pending_age_and_reference_work_used=False, complete_availability_oracle=False)


def train(supervision_path, output):
    source = receipt(supervision_path)
    supervision = checked(source)
    if (supervision.get('schema') != SUPERVISION_SCHEMA or supervision.get('training_only') is not True
            or supervision.get('model_sha256') != digest({k:v for k,v in supervision.items() if k != 'model_sha256'})):
        raise ValueError('Unmodified source-bound training supervision required')
    # Plan receipt checks happen before reading any numerical training rows.
    pools = {}
    for audit in supervision['training_audits']:
        plan = checked(audit['plan'])
        if plan.get('split') != 'training' or plan.get('training_only') is not True:
            raise ValueError('Explicit training plans required')
        for row in plan['inputs']:
            previous = pools.setdefault(row['pool_id'], row['task'])
            if previous != row['task']:
                raise ValueError('Public pool identity changed task')
    for ref in supervision['sources']:
        if receipt(ref['path']) != ref:
            raise ValueError('Training supervision source changed')
    rows = supervision['transition_observations']
    matrix = np.asarray([features(query_from_row(row), pools) for row in rows])
    labels = np.asarray([RESIDENTS.index(row['next_owner']) for row in rows])
    captures = np.asarray([row['capture'] for row in rows])
    independent = sorted(set(captures))
    if len(independent) < 2:
        raise ValueError('At least two independent source captures required for grouped diagnostics')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'predeclared-configuration.json', dict(parameters=PARAMETERS, residents=list(RESIDENTS),
        public_pool_tasks=pools, feature_names=feature_names(pools), supervision=source,
        folds='Exclude each entire original raw-SHA capture; no tuning from these folds',
        baselines=['training_global_majority', 'last_held_owner', 'training_holder_conditional_majority'],
        additional_explanatory_baselines=['holder_public_workload_majority', 'holder_public_workload_frontier_lookup'],
        additional_baselines_used_for_tuning=False,
        target='Next paired RESTORE owner conditional on current SAVE_END; all residents and all transitions retained',
        missing_covariates=['pending_reference_work_ns', 'pending_age_ns'],
        numeric_service_time_fitted=False, GPU_started=False))
    folds = []
    with (output/'capture-heldout-predictions.jsonl').open('w') as stream:
        for excluded in independent:
            testing, training = captures == excluded, captures != excluded
            fitted = fit_matrix(matrix[training], labels[training], captures[training])
            probability = predict_matrix(fitted, matrix[testing])
            predicted = probability.argmax(axis=1)
            tested_rows = [r for r in rows if r['capture'] == excluded]
            trained_rows = [r for r in rows if r['capture'] != excluded]
            training_labels = labels[training]
            majority = int(np.bincount(training_labels, minlength=len(RESIDENTS)).argmax())
            conditional = {}
            for owner in RESIDENTS:
                members = [RESIDENTS.index(r['next_owner']) for r in trained_rows if r['held_owner'] == owner]
                conditional[owner] = int(np.bincount(members, minlength=len(RESIDENTS)).argmax()) if members else majority
            last = np.asarray([RESIDENTS.index(r['held_owner']) for r in tested_rows])
            conditional_predictions = np.asarray([conditional[r['held_owner']] for r in tested_rows])
            explanatory = {}
            for name, fields in [('holder_public_workload_majority', ('held_owner', 'public_workload')),
                    ('holder_public_workload_frontier_lookup', ('held_owner', 'public_workload', 'pending_frontier'))]:
                groups = defaultdict(Counter)
                for row in trained_rows:
                    groups[digest([row[key] for key in fields])][row['next_owner']] += 1
                chosen = []
                for row in tested_rows:
                    group = groups.get(digest([row[key] for key in fields]))
                    chosen.append(max(range(len(RESIDENTS)), key=lambda i:group[RESIDENTS[i]]) if group else -1)
                explanatory[name] = np.asarray(chosen)
            tests = labels[testing]
            numeric = evaluate(tests, predicted, tested_rows)
            folds.append(dict(excluded_capture=excluded, training_captures=sorted(set(captures[training])),
                candidate=numeric, optimizer=fitted['optimizer'],
                training_global_majority=evaluate(tests, np.full(len(tests), majority), tested_rows),
                last_held_owner=evaluate(tests, last, tested_rows),
                training_holder_conditional_majority=evaluate(tests, conditional_predictions, tested_rows),
                **{name:evaluate(tests, values, tested_rows) for name, values in explanatory.items()},
                out_of_training_feature_range_rows=int(np.sum(np.any(
                    (matrix[testing] < np.asarray(fitted['feature_minimum'])-1e-12)
                    | (matrix[testing] > np.asarray(fitted['feature_maximum'])+1e-12), axis=1))),
                all_target_not_pending_rows_retained=True, independent_prediction_gate=False))
            for row, chosen, p in zip(tested_rows, predicted, probability):
                stream.write(json.dumps(dict(capture=row['capture'], job_id=row['job_id'], save_end_ns=row['save_end_ns'],
                    true_next_owner=row['next_owner'], predicted_next_owner=RESIDENTS[int(chosen)],
                    probabilities={owner:float(p[i]) for i,owner in enumerate(RESIDENTS)},
                    next_owner_is_pending=row['next_owner_is_pending'], training_only=True))+'\n')
    fitted = fit_matrix(matrix, labels, captures)
    model = dict(schema=SCHEMA, training_only=True, parameters=PARAMETERS,
        public_pool_tasks=pools, residents=list(RESIDENTS), feature_names=feature_names(pools), fitted=fitted,
        source_supervision=source, source_capture_ids=independent,
        implementation=receipt(__file__), pending_reference_work_training_available=False,
        pending_age_training_available=False, independent_accuracy_validated=False,
        target='NEXT_RESTORED_OWNER_GIVEN_SAVE_NOW_NOT_A_SERVICE_OR_SAVE_TIME_MODEL',
        numerical_targets_from_future_kernel_residuals=False, physical_SM_service_identified=False,
        full_availability_oracle=False, GPU_started=False)
    model['model_sha256'] = digest(model)
    save(output/'model.json', model)
    summary = dict(schema='gsdeploy.next_owner_candidate_grouped_training_diagnostic.v13',
        status='FITTED_FIXED_MARK_CANDIDATE_TRAINING_DIAGNOSTIC_ONLY',
        model=receipt(output/'model.json'), configuration=receipt(output/'predeclared-configuration.json'),
        grouped_predictions=receipt(output/'capture-heldout-predictions.jsonl'),
        transitions=len(rows), independent_captures=len(independent),
        all_target_not_pending_rows_retained=sum(not r['next_owner_is_pending'] for r in rows),
        observed_next_owner_counts=dict(Counter(r['next_owner'] for r in rows)),
        resident_classes_without_observed_next_owner=[r for r in RESIDENTS if not any(row['next_owner'] == r for row in rows)],
        folds=folds, full_fit_optimizer=fitted['optimizer'],
        scopes=['Whole-capture leave-out is training diagnostics; no new independent outcomes read',
            'Continuous age and public workload features permit statistical interpolation/extrapolation; OOD is reported, not hidden as an exact-cell lookup',
            'Target counts are reported for every resident; very rare classes are not evidence of reliable class-specific accuracy',
            'No row is discarded because target owner is not pending; all four resident classes remain possible',
            'High source accuracy mostly reflects stable transfers within repeated job cohorts; compare workload-conditioned majority/lookup baselines before attributing improvement to age',
            'Pending ages/reference work remain explicitly unavailable rather than using future source kernel lifetimes',
            'A scheduler still needs separate survival/SAVE, initial-holder, restore and interval-censored service models'],
        independent_validation=False, calibrated_probability=False, full_availability_oracle=False,
        six_term_utility_evaluated=False, GPU_started=False)
    save(output/'report.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = train(args.supervision, args.output)
    print(json.dumps(dict(status=report['status'], transitions=report['transitions'],
        folds=[dict(excluded_capture=f['excluded_capture'], optimizer=f['optimizer'],
            accuracies={name:f[name]['accuracy'] for name in ('candidate', 'training_global_majority',
                'last_held_owner', 'training_holder_conditional_majority')},
            target_not_pending_accuracy=f['candidate']['target_not_pending_accuracy']) for f in report['folds']],
        full_fit_optimizer=report['full_fit_optimizer'], report=receipt(Path(args.output)/'report.json'))))


if __name__ == '__main__':
    main()
