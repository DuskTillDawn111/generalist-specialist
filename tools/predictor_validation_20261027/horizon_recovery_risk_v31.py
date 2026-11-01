"""Same-family whole-capture horizon-recovery diagnostic, separate from utility.

For positive reader-visible backlog, predict first queue clearance later than H.
Right censoring of exact clearance does not hide a known binary target at H.
"""
from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import recovery_risk as legacy
from gsdeploy.resources import identity

SCHEMA='gsdeploy.horizon_recovery_training_diagnostic.v31'
HORIZONS_NS=(1_000_000_000,5_000_000_000,10_000_000_000)
PHASES=('prefill','decode','sync','cleanup')


def binary_target(prefix,labels):
    if labels['window_start_ns']!=prefix['cutoff_ns'] or labels['horizon_ns'] not in HORIZONS_NS:
        raise ValueError('Matching v30 cutoff and fixed horizon required')
    if prefix['waiting_count']<=0:return dict(status='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY',label=None)
    horizon=labels['horizon_ns'];recovery=labels['recovery'];elapsed=recovery['elapsed_or_censor_ns']
    if type(elapsed) is not int or elapsed<0:raise ValueError('Nonnegative elapsed/censor observation required')
    if recovery['event_observed']:
        if elapsed==0:raise ValueError('Positive current backlog cannot clear at elapsed zero')
        if elapsed>labels['observed_until_ns']-prefix['cutoff_ns']:
            raise ValueError('Recovery event outside observed label interval')
        return dict(status='KNOWN_OBSERVED_CLEARANCE',label=int(elapsed>horizon))
    coverage=labels['observed_until_ns']-prefix['cutoff_ns']
    if coverage>=horizon and elapsed>=horizon:
        return dict(status='KNOWN_NOT_CLEARED_BY_H_DESPITE_EXACT_RECOVERY_CENSOR',label=1)
    return dict(status='UNKNOWN_ADMINISTRATIVE_CENSOR_BEFORE_H',label=None)


def vector(prefix,aliases,tasks):
    """Allowlisted causal fields; never read the isolated labels or GPU cursor."""
    names=['waiting_count','oldest_waiting_ms','visible_HOL_ms']
    values=[prefix['waiting_count'],prefix['oldest_waiting_age_ns']/1e6,prefix['visible_head_of_line_age_ns']/1e6]
    registry={r['request_id']:r for r in prefix['full_public_arrival_input_registry']}
    waiting=[registry[rid] for rid in prefix['waiting_request_ids']]
    if len(waiting)!=prefix['waiting_count'] or any(r['arrival_ns']>prefix['cutoff_ns'] for r in waiting):
        raise ValueError('Complete cutoff-visible waiting set required')
    details=prefix['observed_requests']
    for alias in aliases:
        owner=prefix['per_owner'][alias]
        fields=dict(waiting_count=len(owner['waiting_request_ids']),
            oldest_waiting_ms=owner['oldest_waiting_age_ns']/1e6,
            observed_handler_active=len(owner['handler_active_request_ids']),
            observed_awaiting_next_ready=len(owner['awaiting_next_ready_request_ids']))
        for key,value in fields.items():names.append(alias+':'+key);values.append(value)
    for phase in PHASES:
        count=sum(any(p.get('semantic_phase',p['kind'])==phase for p in row['active_phase_segments']) for row in details)
        names.append('visible_active_phase_requests:'+phase);values.append(count)
    names.extend(['observed_pipe_write_unfinished_count','maximum_visible_unfinished_write_elapsed_ms'])
    elapsed=[row['elapsed_since_dispatch_without_write_completion_ns'] for row in details]
    elapsed=[value for value in elapsed if value is not None]
    values.extend([len(elapsed),max(elapsed,default=0)/1e6])
    future=[r for r in registry.values() if r['arrival_ns']>prefix['cutoff_ns']]
    for task in tasks:
        own=[r for r in waiting if r['task']==task];remaining=[r for r in future if r['task']==task]
        # Raw input_size is a covariate, never interpreted as GPU time/capacity.
        fields=dict(waiting_input_size_sum=sum(r['input_features']['input_size'] for r in own),
            waiting_input_size_max=max((r['input_features']['input_size'] for r in own),default=0),
            remaining_declared_count=len(remaining),
            remaining_declared_input_size_sum=sum(r['input_features']['input_size'] for r in remaining))
        for horizon in HORIZONS_NS:
            fields['declared_arrivals_next_'+str(horizon)+'ns']=sum(r['arrival_ns']<prefix['cutoff_ns']+horizon for r in remaining)
        for key,value in fields.items():names.append(task+':'+key);values.append(value)
    for envelope in prefix['future_count_envelopes']:
        names.append('future_suffix_max_count@'+str(envelope['window_ns']))
        values.append(envelope['maximum_future_declared_request_count'])
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in values):
        raise ValueError('Finite nonnegative explicitly observed/public feature coordinates required')
    return names,np.asarray(values,dtype=float)


def training_row(prefix_row,label_row,domain):
    check_domain(domain,require_actual=False)
    if not domain.get('prepared_execution_projection_exact_equal',False):
        raise UnknownEvidence('EXECUTION_PROJECTION_DIFFERENT_REQUIRES_SEPARATE_DOMAIN')
    for key in ('id','run_id','group_id','kind','execution_context_sha256','decision_time_ns'):
        if prefix_row[key]!=label_row[key]:raise ValueError('Prefix/label identity mismatch')
    if prefix_row.get('split')!='training' or label_row.get('split')!='training':raise ValueError('Training-only source required')
    original=prefix_row['execution_context_sha256']
    if original not in domain['member_execution_context_sha256']:raise UnknownEvidence('UNBRIDGED_ORIGINAL_EXECUTION_CONTEXT')
    prefix=prefix_row['prefix_features'];labels=label_row['offline_labels']
    if prefix['execution_context_sha256']!=original:raise ValueError('Original context must be preserved')
    aliases=sorted(domain['common_execution_projection']['context']['residents'])
    tasks=sorted({r['task'] for r in domain['common_execution_projection']['original_input_registry']})
    names,values=vector(prefix,aliases,tasks);target=binary_target(prefix,labels)
    return dict(id=prefix_row['id']+':H'+str(labels['horizon_ns']),source_sample_id=prefix_row['id'],
        run_id=prefix_row['run_id'],group_id=prefix_row['group_id'],kind=prefix_row['kind'],split='training',
        original_execution_context_sha256=original,risk_domain_sha256=domain['risk_domain_sha256'],
        decision_time_ns=prefix_row['decision_time_ns'],horizon_ns=labels['horizon_ns'],
        feature_names=names,feature_values=values.tolist(),label=target['label'],label_status=target['status'])


def legacy_stumps_kernel(matrix,labels,*,n_estimators=40,learning_rate=.2,minimum_leaf=2):
    """Numerical loop ported verbatim from legacy fit_recovery_risk, API unchanged.

    Only the numerical kernel is reused. No fake recovery duration or legacy
    target adapter is involved. Tests compare against the legacy CPU fit.
    """
    X=np.asarray(matrix,dtype=float);y=np.asarray(labels,dtype=float)
    if X.ndim!=2 or X.shape[0]!=len(y) or not np.isfinite(X).all() or set(y)-{0.,1.}:
        raise ValueError('Finite 2D feature matrix and binary labels required')
    if type(n_estimators) is not int or not 1<=n_estimators<=200 or type(minimum_leaf) is not int or minimum_leaf<2 or not 0<learning_rate<=1:
        raise ValueError('Bounded fixed stumps hyperparameters required')
    if min(np.sum(y==0),np.sum(y==1))<2:raise UnknownEvidence('BOTH_HORIZON_CLASSES_REQUIRE_SUPPORT')
    base=math.log(float(np.mean(y))/(1.-float(np.mean(y))))
    scores,trees=np.full(len(y),base),[]
    for _ in range(n_estimators):
        residual,best=y-legacy._sigmoid(scores),None
        for feature in range(X.shape[1]):
            unique=np.unique(X[:,feature])
            if len(unique)>33:unique=unique[np.unique(np.linspace(0,len(unique)-1,33).astype(int))]
            for left,right in zip(unique,unique[1:]):
                threshold=float(left+(right-left)/2.)
                mask=X[:,feature]<=threshold
                if min(np.sum(mask),np.sum(~mask))<minimum_leaf:continue
                low,high=float(np.mean(residual[mask])),float(np.mean(residual[~mask]))
                prediction=np.where(mask,low,high);loss=float(np.sum((residual-prediction)**2))
                if best is None or loss<best[0]:best=(loss,feature,threshold,low,high,prediction)
        if best is None:raise UnknownEvidence('NO_SUPPORTED_HORIZON_TREE_SPLIT')
        _,feature,threshold,low,high,prediction=best
        trees.append(dict(feature=feature,threshold=threshold,left=learning_rate*low,right=learning_rate*high))
        scores+=learning_rate*prediction
    return dict(base_score=base,trees=trees)


def candidate_fit(rows,domain,*,horizon_ns,n_estimators=40,learning_rate=.2,minimum_leaf=2,allow_cpu_fixture=False):
    check_domain(domain,require_actual=domain.get('kind')!='cpu_fixture')
    if domain['kind']=='cpu_fixture' and not allow_cpu_fixture:raise ValueError('CPU fixture is not physical evidence')
    if horizon_ns not in HORIZONS_NS:raise ValueError('Predeclared horizon required')
    if any(r['split']!='training' or r['horizon_ns']!=horizon_ns for r in rows):raise ValueError('Exact training horizon only')
    if any(r['original_execution_context_sha256'] not in domain['member_execution_context_sha256'] for r in rows):
        raise UnknownEvidence('UNBRIDGED_ORIGINAL_EXECUTION_CONTEXT')
    if any(r['risk_domain_sha256']!=domain['risk_domain_sha256'] for r in rows):raise ValueError('Frozen row risk domain mismatch')
    if any(r['kind']!=('cpu_fixture' if domain['kind']=='cpu_fixture' else 'measured') for r in rows):
        raise ValueError('Evidence kind differs from risk domain')
    runs={r['run_id'] for r in rows}
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate risk samples')
    eligible=[r for r in rows if r['label'] is not None]
    if not eligible:raise UnknownEvidence('NO_KNOWN_HORIZON_TARGETS')
    if len({r['run_id'] for r in eligible})<3:raise UnknownEvidence('THREE_WHOLE_TRAINING_CAPTURES_WITH_KNOWN_TARGETS_REQUIRED')
    feature_names=eligible[0]['feature_names']
    if any(r['feature_names']!=feature_names for r in eligible):raise ValueError('Feature schema differs')
    X=np.asarray([r['feature_values'] for r in eligible]);y=np.asarray([r['label'] for r in eligible])
    fitted=legacy_stumps_kernel(X,y,n_estimators=n_estimators,learning_rate=learning_rate,minimum_leaf=minimum_leaf)
    model=dict(schema=SCHEMA+'.model',risk_domain_sha256=domain['risk_domain_sha256'],horizon_ns=horizon_ns,
        feature_names=feature_names,support_min=X.min(axis=0).tolist(),support_max=X.max(axis=0).tolist(),**fitted,
        provenance=dict(kind='cpu_fixture' if domain['kind']=='cpu_fixture' else 'measured',
            run_ids=sorted(runs),row_ids=sorted(r['id'] for r in rows),
            shared_recipe_content_group_ids=sorted({r['group_id'] for r in rows}),
            original_execution_context_sha256=sorted({r['original_execution_context_sha256'] for r in rows})),
        training_class_counts=dict(Counter(str(r['label']) for r in eligible)),
        unknown_rows_excluded=len(rows)-len(eligible),
        evaluation_scope='same_recipe_content_whole_capture_training_diagnostic',
        calibrated_probability=False,input_generalization_claimed=False,physical_predictor_qualified=False,
        six_term_utility_changed=False,rho_changed=False,rolling_forecast_implemented=False)
    model['model_sha256']=identity(model)
    return model


def check_domain(domain,*,require_actual=True):
    if domain.get('schema')!=SCHEMA+'.risk_domain' or domain.get('risk_domain_sha256')!=identity({k:v for k,v in domain.items() if k!='risk_domain_sha256'}):
        raise ValueError('Frozen explicit risk domain required')
    if require_actual and (domain.get('kind')!='measured' or domain.get('actual_execution_qualification')!='ALL_MEMBERS_PASS_EXACT_RUNTIME_PROJECTION'):
        raise UnknownEvidence('ALL_ACTUAL_EXECUTION_MEMBERS_NOT_YET_QUALIFIED')


def score_candidate(model,row,domain,*,allow_cpu_fixture=False):
    check_domain(domain,require_actual=domain.get('kind')!='cpu_fixture')
    if model.get('model_sha256')!=identity({k:v for k,v in model.items() if k!='model_sha256'}):raise ValueError('Frozen candidate mutated')
    if model['risk_domain_sha256']!=domain['risk_domain_sha256'] or row['original_execution_context_sha256'] not in domain['member_execution_context_sha256']:
        raise UnknownEvidence('UNBRIDGED_ORIGINAL_EXECUTION_CONTEXT')
    if row['risk_domain_sha256']!=domain['risk_domain_sha256']:raise ValueError('Frozen prediction row risk domain mismatch')
    if model['provenance']['kind']=='cpu_fixture' and not allow_cpu_fixture:raise ValueError('CPU fixture is not physical evidence')
    if row['horizon_ns']!=model['horizon_ns'] or row['feature_names']!=model['feature_names']:raise ValueError('Candidate feature/horizon mismatch')
    values=np.asarray(row['feature_values'])
    if values.shape!=(len(model['feature_names']),) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Finite nonnegative fixed-dimensional public feature vector required')
    extrapolated=[model['feature_names'][i] for i in np.flatnonzero((values<model['support_min'])|(values>model['support_max']))]
    if extrapolated:return dict(status='UNKNOWN_OUTSIDE_TRAINING_SUPPORT',risk_score=None,extrapolated_features=extrapolated)
    value=model['base_score']
    for tree in model['trees']:
        value+=tree['left'] if values[tree['feature']]<=tree['threshold'] else tree['right']
    return dict(status='UNCALIBRATED_TRAINING_DIAGNOSTIC',risk_score=float(legacy._sigmoid(value)),
        calibrated_probability=False,six_term_utility_changed=False,rho_changed=False)


def capture_holdout(model,rows,domain,*,allow_cpu_fixture=False):
    if not rows or any(r['split']!='training_capture_holdout' for r in rows):raise ValueError('Explicit diagnostic whole-capture holdout required')
    if {r['run_id'] for r in rows}&set(model['provenance']['run_ids']) or {r['id'] for r in rows}&set(model['provenance']['row_ids']):
        raise ValueError('Held-out whole run overlaps model training')
    predictions=[];known=[];unknown=Counter()
    for row in rows:
        estimate=score_candidate(model,row,domain,allow_cpu_fixture=allow_cpu_fixture)
        predictions.append(dict(id=row['id'],run_id=row['run_id'],label=row['label'],**estimate))
        if row['label'] is None:unknown['UNKNOWN_OR_NOT_AT_RISK_TARGET']+=1
        elif estimate['risk_score'] is None:unknown[estimate['status']]+=1
        else:known.append((row['label'],estimate['risk_score']))
    positive=[p for label,p in known if label];negative=[p for label,p in known if not label]
    auc=None if not positive or not negative else sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
    return dict(schema=SCHEMA+'.capture_holdout',evaluation_scope='same_recipe_content_whole_capture_training_diagnostic',
        requested_n=len(rows),scored_known_n=len(known),unknown_counts=dict(unknown),
        brier_score=None if not known else float(np.mean([(p-label)**2 for label,p in known])),rank_auc=auc,
        predictions=predictions,shared_recipe_content_groups=sorted({r['group_id'] for r in rows}&set(model['provenance']['shared_recipe_content_group_ids'])),
        input_generalization_claimed=False,probability_calibrated=False,physical_accuracy_qualified=False,
        confidence_interval=None,training_refit=False,six_term_utility_changed=False,rho_changed=False)


def whole_capture_partition(rows,heldout_run):
    if heldout_run not in {r['run_id'] for r in rows}:raise ValueError('Held-out run absent')
    train=[copy.deepcopy(r) for r in rows if r['run_id']!=heldout_run]
    test=[dict(copy.deepcopy(r),split='training_capture_holdout') for r in rows if r['run_id']==heldout_run]
    if not train:raise ValueError('Training captures must remain')
    return train,test
