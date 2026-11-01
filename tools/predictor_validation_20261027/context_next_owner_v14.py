"""Current-frontier/history next-context mark candidate, conditional on SAVE now.

No future job composition, request count, scheduled offset, or residual kernel
label is a feature. Previous holder is source-linked within one capture; missing
history remains missing. This mark model does not supply a SAVE or service time.
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
from .training_source_guard_v11 import availability_training_sources

SCHEMA = 'gsdeploy.current_frontier_previous_holder_next_owner_candidate.v14'
RESIDENTS = ('generalist_2#0', 'ocr#0', 'text#0', 'text#1')
TASKS = ('text', 'image', 'asr', 'ocr')
PHASES = ('outside_language_forwards', 'prefill', 'decode', 'ocr_inference')
PARAMETERS = dict(model='L2_multinomial_logistic', regularization=1.,
    optimizer='L-BFGS-B', max_iterations=1000, gradient_tolerance=1e-8,
    capture_balanced_loss=True, age_transform='log1p(context_age_ns/1e6) and square',
    intercept_regularized=True, hyperparameter_selection_from_folds=False,
    penalty_normalization='regularization / training transition count, retaining the fixed v13 convention',
    future_public_workload_features=False,
    previous_history='Unique same-capture prior next_restore/current_restore and next_owner/current_holder match; otherwise missing')


def feature_names(pools, *, include_previous_history=True):
    names = ['intercept', 'log_age_ms', 'log_age_ms_squared',
        'pending_age_unavailable', 'pending_reference_work_unavailable']
    for owner in RESIDENTS:
        names += [f'held:{owner}', f'held_log_age:{owner}', f'held_log_age_squared:{owner}', f'pending:{owner}']
        names += [f'pending_task:{owner}:{task}' for task in TASKS]
        names += [f'pending_phase:{owner}:{phase}' for phase in PHASES]
        names += [f'pending_pool:{owner}:{pool}' for pool in sorted(pools)]
    if include_previous_history:
        names += ['previous_held_owner_missing', *[f'previous_held:{owner}' for owner in RESIDENTS]]
    return names


def previous_history(rows):
    """Only a unique already-completed same-capture handoff identifies history."""
    incoming = defaultdict(list)
    for row in rows:
        a,b,restore = (row[k] for k in ('current_restore_start_ns','save_end_ns','next_restore_start_ns'))
        if (any(type(t) is not int for t in (a,b,restore)) or not a < b <= restore
                or row['context_age_at_save_ns'] != b-a
                or row['held_owner'] not in RESIDENTS or row['next_owner'] not in RESIDENTS):
            raise ValueError('Ordered original context transitions and resident identities required')
        incoming[row['capture'],restore,row['next_owner']].append(row)
    result = []
    for row in rows:
        candidates = incoming[row['capture'],row['current_restore_start_ns'],row['held_owner']]
        if len(candidates) == 1:
            prior = candidates[0]
            if prior['save_end_ns'] > row['current_restore_start_ns']:
                raise ValueError('Previous holder cannot use a future handoff')
            result.append(dict(previous_held_owner=prior['held_owner'], status='UNIQUE_SAME_CAPTURE_PRIOR_HANDOFF',
                prior_save_end_ns=prior['save_end_ns'], prior_restore_start_ns=prior['next_restore_start_ns']))
        else:
            result.append(dict(previous_held_owner=None,
                status='MISSING_PRIOR_HANDOFF' if not candidates else 'AMBIGUOUS_PRIOR_HANDOFF',
                prior_save_end_ns=None, prior_restore_start_ns=None))
    return result


def query_from_row(row, history):
    return dict(held_owner=row['held_owner'], context_age_ns=row['context_age_at_save_ns'],
        pending_frontier=row['pending_frontier'], previous_held_owner=history['previous_held_owner'],
        public_workload=None, pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None)


def features(query, pool_tasks, *, include_previous_history=True):
    required = {'held_owner','context_age_ns','pending_frontier','previous_held_owner','public_workload',
        'pending_age_ns_by_owner','pending_reference_work_ns_by_owner'}
    if set(query) != required:
        raise ValueError('Only explicit cutoff query fields accepted; no future label/timing fields')
    owner, age, previous = query['held_owner'],query['context_age_ns'],query['previous_held_owner']
    if owner not in RESIDENTS or type(age) is not int or age < 0:
        raise ValueError('Known held resident and nonnegative integer attained context age required')
    if previous is not None and previous not in RESIDENTS:
        raise ValueError('Previous held owner must be a resident or explicitly missing')
    if query['pending_age_ns_by_owner'] is not None or query['pending_reference_work_ns_by_owner'] is not None:
        raise ValueError('Pending-age/work covariates are not independently reconstructed training features')
    def identity(alias, task, pool):
        if alias not in RESIDENTS or task not in TASKS or pool_tasks.get(pool) != task:
            raise ValueError('Known public resident/task/input identity required')
        if (alias.startswith('text#') and task != 'text') or (alias == 'ocr#0' and task != 'ocr'):
            raise ValueError('Public owner does not support this task')
    value=defaultdict(float);log_age=math.log1p(age/1_000_000)
    value.update(intercept=1.,log_age_ms=log_age,log_age_ms_squared=log_age**2,
        pending_age_unavailable=1.,pending_reference_work_unavailable=1.)
    value[f'held:{owner}']=1.;value[f'held_log_age:{owner}']=log_age;value[f'held_log_age_squared:{owner}']=log_age**2
    seen=set()
    for row in query['pending_frontier']:
        if not isinstance(row,list) or len(row)!=4:
            raise ValueError('Explicit current owner/task/pool/semantic-phases frontier required')
        alias,task,pool,phases=row;identity(alias,task,pool)
        if alias in seen or not isinstance(phases,list) or not phases or any(p not in PHASES for p in phases) or len(set(phases))!=len(phases):
            raise ValueError('One FIFO current request per owner and supported distinct phases required')
        seen.add(alias);value[f'pending:{alias}']=1.;value[f'pending_task:{alias}:{task}']=1.;value[f'pending_pool:{alias}:{pool}']=1.
        for phase in phases:value[f'pending_phase:{alias}:{phase}']=1.
    # Optional queue-public data is validated, then deliberately discarded.
    # Changes in future counts, composition, offsets cannot alter mark features.
    if query['public_workload'] is not None:
        for row in query['public_workload']:
            if not isinstance(row,list) or len(row)!=4:
                raise ValueError('Public workload verification needs owner/task/input/offset rows')
            alias,task,pool,offset=row;identity(alias,task,pool)
            if type(offset) is not int or offset<0:
                raise ValueError('Nonnegative integer public workload offset required')
    if previous is None:value['previous_held_owner_missing']=1.
    else:value[f'previous_held:{previous}']=1.
    return [value[name] for name in feature_names(pool_tasks,include_previous_history=include_previous_history)]


def cutoff_signature(query):
    frontier=sorted([a,t,p,sorted(phases)] for a,t,p,phases in query['pending_frontier'])
    return digest([query['held_owner'],frontier,query['previous_held_owner']])


def softmax(logits):
    values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fit_matrix(matrix, labels, captures, names):
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
            held_column = names.index('held:'+owner)
            if row[held_column] == 1:
                ages[owner].append(float(row[1]))
    return dict(center=center.tolist(), scale=scale.tolist(), coefficients=coefficients.tolist(),
        feature_minimum=matrix.min(axis=0).tolist(), feature_maximum=matrix.max(axis=0).tolist(),
        held_log_age_ranges={owner:[min(values),max(values)] for owner,values in ages.items()},
        label_counts={owner:int(np.sum(labels == i)) for i,owner in enumerate(RESIDENTS)},
        optimizer=dict(converged=bool(optimized.success), iterations=int(optimized.nit),
            message=str(optimized.message), objective=float(optimized.fun),
            maximum_absolute_gradient=float(np.abs(optimized.jac).max())))


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
    """Current-cutoff mark model; future workload is verification-only input."""
    def __init__(self, model, *, allow_cpu_fixture=False):
        if (model.get('schema') != SCHEMA or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k!='model_sha256'})
                or model.get('parameters') != PARAMETERS):
            raise ValueError('Source-bound fixed current-frontier mark candidate required')
        if model.get('CPU_fixture'):
            if allow_cpu_fixture is not True:
                raise ValueError('CPU fixture cannot qualify a measured mark candidate')
        else:
            if receipt(model['implementation']['path']) != model['implementation']:
                raise ValueError('Next-owner candidate implementation changed')
            checked(model['source_supervision'])
            if receipt(model['history_observations']['path']) != model['history_observations']:
                raise ValueError('Previous-holder source-link observations changed')
        names=feature_names(model['public_pool_tasks']);fitted=model['fitted'];width=len(names)
        if model.get('feature_names')!=names or model.get('residents')!=list(RESIDENTS):
            raise ValueError('Current-cutoff feature/class contract changed')
        if (np.asarray(fitted['coefficients']).shape!=(width,len(RESIDENTS))
                or any(np.asarray(fitted[k]).shape!=(width,) for k in ('center','scale','feature_minimum','feature_maximum'))
                or any(not np.isfinite(np.asarray(fitted[k])).all() for k in ('coefficients','center','scale','feature_minimum','feature_maximum'))
                or np.any(np.asarray(fitted['scale'])<=0)):
            raise ValueError('Finite aligned multinomial arrays required')
        self.model=model
        self.support=set(model['cutoff_state_signatures'])

    def predict_given_save_now(self, *, held_owner, context_age_ns, pending_frontier,
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None, pending_reference_work_ns_by_owner=None):
        query=dict(held_owner=held_owner,context_age_ns=context_age_ns,pending_frontier=pending_frontier,
            previous_held_owner=previous_held_owner,public_workload=public_workload,
            pending_age_ns_by_owner=pending_age_ns_by_owner,pending_reference_work_ns_by_owner=pending_reference_work_ns_by_owner)
        row=np.asarray(features(query,self.model['public_pool_tasks']));fitted=self.model['fitted']
        outside=np.flatnonzero((row<np.asarray(fitted['feature_minimum'])-1e-12)|(row>np.asarray(fitted['feature_maximum'])+1e-12))
        warnings=['FEATURE_OUTSIDE_TRAINING_RANGE:'+self.model['feature_names'][i] for i in outside]
        ages=fitted['held_log_age_ranges'].get(held_owner)
        if ages is None or not ages[0]<=row[1]<=ages[1]:warnings.append('HELD_OWNER_AGE_OUTSIDE_TRAINING_SUPPORT')
        if cutoff_signature(query) not in self.support:warnings.append('HELD_FRONTIER_PREVIOUS_COMBINATION_UNSEEN')
        missing=previous_held_owner is None
        probability=predict_matrix(fitted,row[None,:])[0]
        return dict(status='UNCALIBRATED_MARK_CANDIDATE_WITH_SUPPORT_WARNINGS' if warnings or missing else 'UNCALIBRATED_MARK_CANDIDATE',
            next_owner_probabilities={owner:float(probability[i]) for i,owner in enumerate(RESIDENTS)},
            most_likely_next_owner=RESIDENTS[int(probability.argmax())],
            observed_training_target_counts=dict(fitted['label_counts']),
            zero_observed_training_target_classes=[owner for owner,count in fitted['label_counts'].items() if count==0],
            previous_held_owner_missing=missing,
            history_warning='NO_OBSERVED_OR_MODELLED_PREVIOUS_HOLDER_SUPPLIED' if missing else None,
            ood_flags=warnings,model_sha256=self.model['model_sha256'],
            future_public_workload_used_as_features=False,public_workload_verification_only=True,
            conditioned_on_observed_or_separately_predicted_SAVE_now=True,
            save_time_predicted=False,restore_time_predicted=False,service_ready_time_predicted=False,
            physical_SM_service_identified=False,calibrated_probability=False,
            pending_age_and_reference_work_used=False,complete_availability_oracle=False)


def baseline_predictions(trained_rows, tested_rows, histories, trained_histories):
    """Every lookup falls back to train holder majority, then global majority."""
    global_counts=Counter(r['next_owner'] for r in trained_rows)
    majority=max(RESIDENTS,key=lambda owner:global_counts[owner])
    holders=defaultdict(Counter);frontiers=defaultdict(Counter);previous=defaultdict(Counter)
    for row,history in zip(trained_rows,trained_histories):
        holders[row['held_owner']][row['next_owner']]+=1
        key=digest([row['held_owner'],row['pending_frontier']])
        frontiers[key][row['next_owner']]+=1
        previous[digest([key,history['previous_held_owner']])][row['next_owner']]+=1
    chosen=defaultdict(list);fallbacks=Counter()
    for row,history in zip(tested_rows,histories):
        held=holders.get(row['held_owner']);fallback=max(RESIDENTS,key=lambda owner:held[owner]) if held else majority
        key=digest([row['held_owner'],row['pending_frontier']]);group=frontiers.get(key)
        chosen['training_holder_only_majority'].append(RESIDENTS.index(fallback))
        chosen['training_global_majority'].append(RESIDENTS.index(majority))
        chosen['holder_frontier_lookup_with_fallback'].append(RESIDENTS.index(max(RESIDENTS,key=lambda owner:group[owner]) if group else fallback))
        fallbacks['holder_frontier_lookup_with_fallback']+=not bool(group)
        group=previous.get(digest([key,history['previous_held_owner']]))
        chosen['holder_frontier_previous_lookup_with_fallback'].append(RESIDENTS.index(max(RESIDENTS,key=lambda owner:group[owner]) if group else fallback))
        fallbacks['holder_frontier_previous_lookup_with_fallback']+=not bool(group)
    return {key:np.asarray(values) for key,values in chosen.items()},dict(fallbacks)


def train(supervision_path, output):
    source=receipt(supervision_path);supervision=checked(source)
    if (supervision.get('schema')!=SUPERVISION_SCHEMA or supervision.get('training_only') is not True
            or supervision.get('validation_labels_used') is not False
            or supervision.get('model_sha256')!=digest({k:v for k,v in supervision.items() if k!='model_sha256'})):
        raise ValueError('Unmodified source-bound training supervision required')
    audits=availability_training_sources([r['report'] for r in supervision['training_audits']])
    if audits!=supervision['training_audits']:
        raise ValueError('Supervision training lineage differs from guarded original captures')
    pools={}
    for audit in audits:
        for row in checked(audit['plan'])['inputs']:
            previous=pools.setdefault(row['pool_id'],row['task'])
            if previous!=row['task']:raise ValueError('Public pool task identity changed')
    for ref in supervision['sources']:
        if receipt(ref['path'])!=ref:raise ValueError('Training supervision source changed')
    rows=supervision['transition_observations'];histories=previous_history(rows)
    captures=np.asarray([r['capture'] for r in rows]);independent=sorted(set(captures))
    if set(independent)!={a['raw']['sha256'] for a in audits}:
        raise ValueError('Complete guarded original capture cohort required')
    queries=[query_from_row(row,history) for row,history in zip(rows,histories)]
    names=feature_names(pools);ablation_names=feature_names(pools,include_previous_history=False)
    matrix=np.asarray([features(query,pools) for query in queries])
    ablation=np.asarray([features(query,pools,include_previous_history=False) for query in queries])
    unavailable=np.asarray([features(dict(query,previous_held_owner=None),pools) for query in queries])
    labels=np.asarray([RESIDENTS.index(r['next_owner']) for r in rows])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'predeclared-configuration.json',dict(parameters=PARAMETERS,residents=list(RESIDENTS),
        feature_names=names,no_previous_history_ablation_feature_names=ablation_names,public_pool_tasks=pools,
        supervision=source,folds='Exclude each complete original raw-SHA capture; fixed configuration and no fold tuning',
        baselines=['training_global_majority','training_holder_only_majority','holder_frontier_lookup_with_fallback','holder_frontier_previous_lookup_with_fallback'],
        ablation='Same fixed L2 multinomial model without any previous-holder or missing-history feature',
        public_future_workload_in_mark_features=False,initial_previous_holder_default=None,
        future_arrivals_are_advanced_only_by_the_DAG=True,GPU_started=False))
    with (output/'previous-holder-observations.jsonl').open('x') as stream:
        for row,history in zip(rows,histories):
            stream.write(json.dumps(dict(capture=row['capture'],job_id=row['job_id'],save_end_ns=row['save_end_ns'],
                current_restore_start_ns=row['current_restore_start_ns'],held_owner=row['held_owner'],**history))+'\n')
    folds=[]
    with (output/'capture-heldout-predictions.jsonl').open('x') as stream:
        for excluded in independent:
            training,testing=captures!=excluded,captures==excluded
            fitted=fit_matrix(matrix[training],labels[training],captures[training],names)
            fitted_ablation=fit_matrix(ablation[training],labels[training],captures[training],ablation_names)
            probability=predict_matrix(fitted,matrix[testing]);predicted=probability.argmax(axis=1)
            predicted_ablation=predict_matrix(fitted_ablation,ablation[testing]).argmax(axis=1)
            predicted_missing=predict_matrix(fitted,unavailable[testing]).argmax(axis=1)
            trained_rows=[r for r in rows if r['capture']!=excluded];tested_rows=[r for r in rows if r['capture']==excluded]
            train_histories=[h for r,h in zip(rows,histories) if r['capture']!=excluded]
            test_histories=[h for r,h in zip(rows,histories) if r['capture']==excluded]
            baselines,fallbacks=baseline_predictions(trained_rows,tested_rows,test_histories,train_histories)
            missing=np.asarray([h['previous_held_owner'] is None for h in test_histories])
            subset_rows=[r for r,m in zip(tested_rows,missing) if m]
            support={cutoff_signature(q) for q,c in zip(queries,captures) if c!=excluded}
            folds.append(dict(excluded_capture=excluded,training_captures=sorted(set(captures[training])),
                candidate=evaluate(labels[testing],predicted,tested_rows),
                no_previous_history_ablation=evaluate(labels[testing],predicted_ablation,tested_rows),
                candidate_with_all_previous_history_missing=evaluate(labels[testing],predicted_missing,tested_rows),
                candidate_on_naturally_missing_history=evaluate(labels[testing][missing],predicted[missing],subset_rows) if subset_rows else None,
                **{name:evaluate(labels[testing],values,tested_rows) for name,values in baselines.items()},
                baseline_fallback_counts=fallbacks,optimizer=fitted['optimizer'],ablation_optimizer=fitted_ablation['optimizer'],
                previous_history_status_counts=dict(Counter(h['status'] for h in test_histories)),
                out_of_training_feature_range_rows=int(np.any((matrix[testing]<np.asarray(fitted['feature_minimum'])-1e-12)
                    |(matrix[testing]>np.asarray(fitted['feature_maximum'])+1e-12),axis=1).sum()),
                unseen_held_frontier_previous_combinations=sum(cutoff_signature(q) not in support for q,c in zip(queries,captures) if c==excluded),
                all_target_not_pending_rows_retained=True,independent_prediction_gate=False))
            for row,history,chosen,without,missing_choice,p in zip(tested_rows,test_histories,predicted,predicted_ablation,predicted_missing,probability):
                stream.write(json.dumps(dict(capture=row['capture'],job_id=row['job_id'],save_end_ns=row['save_end_ns'],
                    true_next_owner=row['next_owner'],predicted_next_owner=RESIDENTS[int(chosen)],
                    previous_held_owner=history['previous_held_owner'],previous_history_status=history['status'],
                    no_history_ablation_prediction=RESIDENTS[int(without)],all_history_missing_prediction=RESIDENTS[int(missing_choice)],
                    probabilities={owner:float(p[i]) for i,owner in enumerate(RESIDENTS)},
                    next_owner_is_pending=row['next_owner_is_pending'],training_only=True))+'\n')
    fitted=fit_matrix(matrix,labels,captures,names)
    model=dict(schema=SCHEMA,training_only=True,parameters=PARAMETERS,public_pool_tasks=pools,
        residents=list(RESIDENTS),feature_names=names,fitted=fitted,source_supervision=source,
        history_observations=receipt(output/'previous-holder-observations.jsonl'),
        source_capture_ids=independent,implementation=receipt(__file__),
        cutoff_state_signatures=sorted({cutoff_signature(query) for query in queries}),
        future_public_workload_features=False,pending_reference_work_training_available=False,
        pending_age_training_available=False,independent_accuracy_validated=False,
        target='NEXT_RESTORED_OWNER_GIVEN_SAVE_NOW_CURRENT_FRONTIER_AND_PAST_HOLDER_ONLY',
        numerical_targets_from_future_kernel_residuals=False,physical_SM_service_identified=False,
        full_availability_oracle=False,GPU_started=False)
    model['model_sha256']=digest(model);save(output/'model.json',model)
    known_initial=[r for r in supervision['initial_observations'] if r['status']=='OBSERVED_PAIRED_CONTEXT_STATE']
    summary=dict(schema='gsdeploy.current_frontier_mark_grouped_training_diagnostic.v14',
        status='FITTED_FIXED_MARK_CANDIDATE_TRAINING_DIAGNOSTIC_ONLY',model=receipt(output/'model.json'),
        configuration=receipt(output/'predeclared-configuration.json'),history_observations=model['history_observations'],
        grouped_predictions=receipt(output/'capture-heldout-predictions.jsonl'),transitions=len(rows),
        independent_captures=len(independent),all_target_not_pending_rows_retained=sum(not r['next_owner_is_pending'] for r in rows),
        observed_next_owner_counts=dict(Counter(r['next_owner'] for r in rows)),
        previous_history_status_counts=dict(Counter(h['status'] for h in histories)),
        paired_initial_observations=len(known_initial),prospective_initial_previous_holder_default=None,
        paired_initial_candidates_automatically_assigned_previous_holder=0,
        prospective_initial_previous_holder_unavailable_without_separate_history=len(known_initial),
        folds=folds,full_fit_optimizer=fitted['optimizer'],
        future_workload_features_removed=True,public_full_future_workload_verification_only=True,
        scopes=['All original transition labels, nonpending targets and missing previous histories are retained',
            'Previous holder uses one unique already-completed same-capture handoff, never a cross-source pairing',
            'At a new simulated initial state previous holder remains unknown unless separately observed/modelled; source paired initial states do not grant prospective prior history',
            'Only current pending identity/phase, held owner, continuous attained age, and past holder enter the candidate',
            'A 57-request future cohort cannot change features or support warnings merely by future counts or offsets',
            'Rare target classes and missing-history predictions remain uncalibrated; compare full-coverage baselines and no-history ablation',
            'Whole-capture CV on two training captures is a dependent-data diagnostic, not independent accuracy or a full scheduling oracle'],
        independent_validation=False,calibrated_probability=False,full_availability_oracle=False,
        six_term_utility_evaluated=False,GPU_started=False)
    save(output/'report.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=train(args.supervision,args.output)
    print(json.dumps(dict(report=receipt(Path(args.output)/'report.json'),transitions=report['transitions'],
        previous_history_status_counts=report['previous_history_status_counts'],
        folds=[dict(excluded_capture=f['excluded_capture'],accuracies={name:f[name]['accuracy'] for name in
            ('candidate','no_previous_history_ablation','candidate_with_all_previous_history_missing','training_holder_only_majority','holder_frontier_lookup_with_fallback','holder_frontier_previous_lookup_with_fallback')},
            optimizer=f['optimizer']) for f in report['folds']],full_fit_optimizer=report['full_fit_optimizer'])))


if __name__=='__main__':main()


