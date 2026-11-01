"""CPU training-only macro response fit and whole-training-capture leaveout.

This fits response clocks conditional on observed peer notification paths. It
does not generate those paths, internal host END/query events or future work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from .composable_response_macros_v1 import extract as extract_macros, SCHEMA as MACRO_SCHEMA
from .device_progress_smoke_raw_verify_v4 import run as recheck_raw
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .profile_training_evidence_review_v1 import review as recheck_review
from .stage_response_features_v1 import checked, receipt
from .stage_response_fit_v1 import profile, require

SCHEMA = 'gsdeploy.macro_joint_response_training_leaveout.v1'
FEATURES = ('peer_visible_macro_owner_count', 'peer_same_visible_macro_phase_owner_count')
CANDIDATES = {'baseline': (), 'visible_count': FEATURES[:1], 'visible_phase_aware': FEATURES}
DOMAIN = 'natural_forward_and_async_tail_wall'


def verified_document(path):
    """Recompute both review levels and exact macros, not just stored hashes."""
    ref = receipt(path)
    doc = checked(ref)
    require(doc.get('schema') == MACRO_SCHEMA and doc.get('mode') == 'natural'
        and doc.get('status') == 'DISJOINT_TRAINING_RESPONSE_MACROS_NOT_FITTED_PROVIDER'
        and doc.get('validation_labels_read') is False, 'Natural training macro domain required')
    for source in doc['sources']:
        require(receipt(source['path']) == source, 'Macro source changed')
    review_ref = doc['sources'][0]
    review = checked(review_ref)
    require(review.get('training_feature_extraction_eligible') is True
        and review.get('validation_labels_read') is False, 'Qualified training review required')
    raw_review = checked(review['raw_review'])
    closure = strict_references([review, raw_review])
    for source in closure:
        require(receipt(source['path']) == source, 'Complete review closure changed')
    with tempfile.TemporaryDirectory(prefix='macro-training-recheck-') as directory:
        raw_ref = recheck_raw(review['inputs']['raw.json']['path'], Path(directory)/'raw')
        require(checked(raw_ref) == raw_review, 'Raw review must rederive exactly')
        new_ref = recheck_review(Path(review['inputs']['raw.json']['path']).parent,
            review['execution']['path'], review['raw_review']['path'], Path(directory)/'review')
        require(checked(new_ref) == review, 'Training review must rederive exactly')
    require(extract_macros(review_ref['path']) == doc,
        'Macros must rederive exactly from complete unchanged review closure')
    require(receipt(path) == ref, 'Macro document changed during rederivation')
    return doc, strict_references([ref, doc['sources'], closure])


def prepared(documents, features):
    require(all(f in FEATURES for f in features), 'Only fixed macro evidence features supported')
    rows = []
    for doc in documents:
        require(doc.get('schema') == MACRO_SCHEMA and doc.get('mode') == 'natural'
            and doc.get('validation_labels_read') is False, 'Natural macro training only')
        for request in doc['requests']:
            macros = request['macros']
            require(macros and all(a['right_ns'] == b['left_ns'] for a,b in zip(macros,macros[1:]))
                and sum(m['response_wall_ns'] for m in macros) == request['worker_start_to_ready_ns'],
                'Every request must have exact complete disjoint macro coverage')
            for macro in macros:
                require(macro['response_domain'] == DOMAIN
                    and macro['additional_stream_elapsed_or_sync_wait_may_be_added'] is False
                    and macro['true_SM_service_identified'] is False, 'Explicit macro response wall domain required')
                left, right = macro['left_ns'], macro['right_ns']
                path = macro['causal_covariate_path']
                require(type(left) is int and type(right) is int and right > left
                    and path and path[0]['left_ns'] == left and path[-1]['right_ns'] == right
                    and all(a['right_ns'] == b['left_ns'] for a,b in zip(path,path[1:])),
                    'Exact occurrence-origin complete macro path required')
                durations, matrix = [], []
                for piece in path:
                    duration = piece['right_ns']-piece['left_ns']
                    require(type(duration) is int and duration > 0 and duration == piece['duration_ns']
                        and piece['state']['cutoff_processing_ns'] == piece['left_ns']
                        and piece['each_state_uses_only_its_processed_prefix'] is True,
                        'Positive disjoint processed-prefix exposure required')
                    vector = [piece['state'][f] for f in features]
                    require(all(type(x) is int and x >= 0 for x in vector), 'Nonnegative integer macro evidence required')
                    durations.append(duration/1e6); matrix.append(vector)
                require(sum(p['duration_ns'] for p in path) == right-left == macro['response_wall_ns'],
                    'Macro response and exposure clocks differ')
                shape = macro['forward_shape'] or {}
                group = (macro['owner'], macro['pool_id'], macro['task'], macro['macro_id'],
                    macro['semantic_phase'], macro['kind'], *(shape.get(k) for k in
                        ('input_sequence_length','input_batch_size','forward_index')))
                # Aggregate equal states for likelihood only; ordered original
                # path is retained for exact conditional quantile inversion.
                totals = defaultdict(float)
                for duration, vector in zip(durations,matrix): totals[tuple(vector)] += duration
                endpoint = tuple(matrix[-1])
                keys = [key for key in totals if key != endpoint] + [endpoint]
                rows.append(dict(group=group, capture=macro['capture_sha256'],
                    request_id=macro['request_id'], macro_id=macro['macro_id'], owner=macro['owner'],
                    phase=macro['semantic_phase'],kind=macro['kind'],
                    response_ms=(right-left)/1e6,
                    durations=np.asarray([totals[k] for k in keys]),
                    matrix=np.asarray(keys,dtype=float).reshape(len(keys),len(features)),
                    ordered_durations=np.asarray(durations),
                    ordered_matrix=np.asarray(matrix,dtype=float).reshape(len(matrix),len(features))))
    require(bool(rows), 'Complete macro rows required')
    return rows


def objective(beta, rows):
    value, _, variance, derivative, residual = profile(beta, rows)
    endpoint = np.asarray([row['matrix'][-1] for row in rows])
    gradient = np.sum(derivative*(1+residual[:,None]/variance),axis=0)+np.sum(endpoint,axis=0)
    return value/len(rows), gradient/len(rows)


def fit(rows, candidate):
    width = len(CANDIDATES[candidate]); attempts = []
    if width:
        for start in (np.zeros(width), np.full(width,.25), np.full(width,-.25)):
            result = minimize(objective,start,args=(rows,),method='BFGS',jac=True,
                options={'gtol':1e-8,'maxiter':1000})
            attempts.append(result)
        valid = [x for x in attempts if math.isfinite(float(x.fun)) and np.all(np.isfinite(x.x))]
        require(bool(valid),'No finite optimization attempt')
        best = min(valid,key=lambda x:x.fun); beta = best.x
        converged, gradient_norm = bool(best.success), float(np.linalg.norm(best.jac))
    else: beta=np.empty(0); converged=True; gradient_norm=0.
    nll, mu, variance, derivative, residual = profile(beta,rows)
    groups = sorted(mu, key=repr); indices={g:i for i,g in enumerate(groups)}
    baseline=np.zeros((len(rows),len(groups)))
    for i,row in enumerate(rows):baseline[i,indices[row['group']]]=-1
    joint=np.column_stack((baseline,derivative)); singular=np.linalg.svd(joint,compute_uv=False)
    rank=int(np.linalg.matrix_rank(joint)); dimension=len(groups)+width
    info=dict(candidate=candidate,features=list(CANDIDATES[candidate]),coefficients=beta.tolist(),
        coefficient_constraints='free_real_acceleration_allowed',nll=nll,mean_nll=nll/len(rows),
        objective_scaling='exact_nll_divided_by_training_row_count_same_argmin_improved_gradient_scale',
        lognormal_reference_sigma=math.sqrt(variance),macro_rows=len(rows),
        captures=sorted({r['capture'] for r in rows}),optimizer_converged=converged,
        optimizer_mean_nll_gradient_norm=gradient_norm,
        optimizer_attempts=[dict(success=bool(x.success),message=str(x.message),mean_nll=float(x.fun),
            gradient_norm=float(np.linalg.norm(x.jac))) for x in attempts],
        baseline_nuisance='owner_input_task_macro_id_phase_kind_forward_shape',
        baselines=[dict(group=list(g),log_reference_ms=mu[g],rows=sum(r['group']==g for r in rows)) for g in groups],
        joint_response_jacobian=dict(rank=rank,parameters=dimension,full_column_rank=rank==dimension,
            includes_baseline_nuisance_means=True,includes_sigma=False,singular_values=singular.tolist(),
            condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None),
        training_log_reference_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        qualified=False)
    return info, beta, mu, variance


def conditional_median(row,beta,reference_ms):
    """Invert observed path; beyond end explicitly freeze last observed state."""
    remaining=reference_ms; elapsed=0.
    for duration,vector in zip(row['ordered_durations'],row['ordered_matrix']):
        rate=math.exp(-float(vector@beta)); work=duration*rate
        if remaining <= work or math.isclose(remaining,work,rel_tol=1e-14):
            return elapsed+min(remaining,work)/rate,False
        remaining-=work;elapsed+=duration
    return elapsed+remaining/math.exp(-float(row['ordered_matrix'][-1]@beta)),True


def summarize(records):
    known=[r for r in records if r['support']=='KNOWN_TRAINING_NUISANCE_GROUP']
    result=dict(rows=len(records),supported_rows=len(known),unknown_rows=len(records)-len(known))
    if known:
        error=np.asarray([r['conditional_median_error_ms'] for r in known])
        result.update(mean_nll=float(np.mean([r['nll'] for r in known])),
            conditional_median_mae_ms=float(np.mean(np.abs(error))),
            conditional_median_rmse_ms=float(np.sqrt(np.mean(error**2))),
            conditional_median_mape=float(np.mean([abs(r['conditional_median_error_ms'])/r['actual_ms'] for r in known])),
            conditional_90pct_interval_empirical_coverage=float(np.mean([.05<=r['PIT']<=.95 for r in known])),
            endpoint_state_continuation_rows=sum(r['endpoint_state_continuation_assumed'] for r in known))
    return result


def evaluate(rows,beta,mu,variance):
    records=[]
    for row in rows:
        record=dict(capture_sha256=row['capture'],request_id=row['request_id'],owner=row['owner'],
            macro_id=row['macro_id'],phase=row['phase'],kind=row['kind'],group=list(row['group']),actual_ms=row['response_ms'])
        if row['group'] not in mu:
            record['support']='UNKNOWN_UNSEEN_OWNER_INPUT_MACRO_SHAPE';records.append(record);continue
        logv=float(np.logaddexp.reduce(np.log(row['durations'])-row['matrix']@beta))
        delta=logv-mu[row['group']]
        nll=logv+.5*math.log(2*math.pi*variance)+delta**2/(2*variance)+float(row['matrix'][-1]@beta)
        predicted,extension=conditional_median(row,beta,math.exp(mu[row['group']]))
        record.update(support='KNOWN_TRAINING_NUISANCE_GROUP',nll=nll,log_reference_residual=delta,
            PIT=float(ndtr(delta/math.sqrt(variance))),conditional_median_ms=predicted,
            conditional_median_error_ms=predicted-row['response_ms'],endpoint_state_continuation_assumed=extension,
            future_peer_path_generated=False,prospective_prediction=False)
        records.append(record)
    groups=defaultdict(list)
    for record in records:groups[(record['owner'],record['macro_id'],record['phase'])].append(record)
    return dict(summary=summarize(records),per_macro=[dict(owner=k[0],macro_id=k[1],phase=k[2],**summarize(v))
        for k,v in sorted(groups.items())],records=records)


def run(paths):
    docs=[]; sources=[]
    for path in paths:
        doc,closure=verified_document(path);docs.append(doc);sources.extend(closure)
    captures=[{m['capture_sha256'] for r in d['requests'] for m in r['macros']} for d in docs]
    require(len(docs)==3 and all(len(x)==1 for x in captures)
        and len(set.union(*captures))==3, 'Exactly three distinct whole natural training captures required')
    captures=sorted(set.union(*captures));results=[]
    for candidate,features in CANDIDATES.items():
        rows=prepared(docs,features);full,beta,mu,variance=fit(rows,candidate);folds=[];all_records=[]
        for heldout in captures:
            train=[r for r in rows if r['capture']!=heldout];test=[r for r in rows if r['capture']==heldout]
            fitted,b,m,v=fit(train,candidate);scored=evaluate(test,b,m,v);all_records.extend(scored['records'])
            folds.append(dict(heldout_capture_sha256=heldout,training_fit=fitted,evaluation=scored))
        eligible=all(f['training_fit']['optimizer_converged']
            and f['training_fit']['joint_response_jacobian']['full_column_rank']
            and f['evaluation']['summary']['unknown_rows']==0 for f in folds)
        full.update(whole_capture_leaveout=folds,pooled_leaveout=summarize(all_records),
            leaveout_selection_eligible=eligible,
            coefficient_capture_stability=dict(fold_coefficients=[f['training_fit']['coefficients'] for f in folds],
                max_minus_min=(np.ptp(np.asarray([f['training_fit']['coefficients'] for f in folds]),axis=0)).tolist()),
            all_capture_conditional_fit=evaluate(rows,beta,mu,variance))
        results.append(full)
    eligible=[r for r in results if r['leaveout_selection_eligible']]
    selected=min(eligible,key=lambda r:np.mean([f['evaluation']['summary']['mean_nll']
        for f in r['whole_capture_leaveout']]))['candidate'] if eligible else None
    closure=strict_references([sources,implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in closure),'Source changed during fit')
    return dict(schema=SCHEMA,status='THREE_TRAINING_CAPTURE_CONDITIONAL_MACRO_RESPONSE_DIAGNOSTICS',
        mode='natural',response_domain=DOMAIN,sources=closure,candidates=results,selected_candidate=selected,
        selection_rule='equal_capture_mean_heldout_conditional_NLL_among_converged_full_rank_supported_folds',
        selection_scope='three_training_captures_only_not_independent_validation',
        endpoint_density_jacobian_included=True,baseline_and_interference_jointly_fitted=True,
        all_entry_forward_final_READY_and_OCR_nodes_retained=True,
        response_error_scope='conditional_on_observed_heldout_peer_path_with_explicit_endpoint_state_continuation',
        internal_host_END_law='UNKNOWN_NOT_FITTED',query_outcome_law='UNKNOWN_NOT_FITTED',
        delayed_notification_law='UNKNOWN_NOT_FITTED',future_work_template_law='UNKNOWN_NOT_FITTED',
        residual_dependence_across_macros='UNKNOWN_NOT_FITTED',
        capture_robust_parameter_uncertainty_estimated=False,joint_provider_complete=False,
        native_CUDA_API_numeric_authentication='UNQUALIFIED',main_queue_provider_eligible=False,
        independent_prediction_accuracy_validated=False,final_gain_eligible=False,GPU_started=False,
        validation_labels_read=False,
        limitations=['Whole captures, not macro rows, are replication units; three blocks do not establish robust precision.',
            'Heldout likelihood is a conditional path score; endogenous peer paths and internal notifications are not generated.',
            'Wall median errors assume frozen final peer state if reference work exceeds observed macro path.',
            'Owner/input/macro-id/shape nuisance support does not extrapolate to unseen inputs or generation branches.',
            'Macro response already includes asynchronous GPU waits, selection and model-specific final work; add no independent wait.',
            'No SM service, causal hardware interference, queue policy, P95 accuracy or final utility gain is qualified.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.training_macros)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],selected_candidate=result['selected_candidate'])))
