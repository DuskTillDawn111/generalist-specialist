"""CPU-only partial decoder projection contrast in complete macro training.

Nominal decoder projection intensity contrast excludes attention/KV, measured
DRAM/cache and quantized execution costs. Unknown pairs remain explicit, and
all response nodes are retained. This is not a prospective queue provider.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root

from . import macro_joint_response_fit_v1 as base
from .macro_joint_response_second_order_v1 import differential, local_minimum_certificate
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import ordered_events
from .static_language_work_v1 import (build_binding, features_from_visible_begin,
    training_begin_features, validate_binding_sources)

SCHEMA='gsdeploy.partial_decoder_projection_contrast_macro_training.v1'
COUNT='peer_visible_macro_owner_count'
PHASE='peer_same_visible_macro_phase_owner_count'
UNKNOWN='unknown_projection_pair_count'
OWN_UNKNOWN='own_projection_unsupported_indicator'
CONTRAST='known_partial_decoder_projection_contrast_sum'
CANDIDATES={
    'baseline':(),
    'visible_phase_aware':(COUNT,PHASE),
    'phase_and_projection_missingness':(COUNT,PHASE,UNKNOWN,OWN_UNKNOWN),
    'phase_missingness_projection_contrast':(COUNT,PHASE,UNKNOWN,OWN_UNKNOWN,CONTRAST),
}


def checked_static_inventory(path):
    ref=base.receipt(path);report=base.checked(ref)
    base.require(report['schema']=='gsdeploy.static_language_work_training_CPU_report.v1'
        and report['validation_labels_read'] is False and report['GPU_started'] is False,
        'Source bound CPU static projection inventory required')
    for source in report['sources']:
        base.require(base.receipt(source['path'])==source,'Static inventory source changed')
    bindings={}
    for family,binding_ref in report['bindings'].items():
        binding=base.checked(binding_ref);validate_binding_sources(binding)
        sources=binding['sources']
        rebuilt=build_binding(sources[0]['path'],family,language_source=sources[3]['path'],
            quantization_source=sources[4]['path'],loader_source=sources[5]['path'],
            lora_source=sources[-1]['path'] if family=='text' else None)
        base.require(rebuilt==binding,'Static binding must rederive exactly from declared sources/headers')
        bindings[family]=binding
    return report,bindings,strict_references([ref,report['sources'],report['bindings'],
        [b['sources'] for b in bindings.values()]])


def descriptor(feature,event,capture):
    return dict(partial_decoder_projection_intensity=feature[
        'base_dense_equivalent_structural_intensity_flops_per_logical_byte'],
        binding_sha256=feature['binding_sha256'],phase=feature['phase'],
        batch_size=feature['batch_size'],sequence_length=feature['sequence_length'],
        available_at_processing_ns=event['collector_processing_time_ns'],
        BEGIN_occurrence_ns=event['perf_ns'],owner=event['owner'],request_id=event['request_id'],
        phase_id=event['segment_id'],raw_event_ordinal=event['request_ordinal'],capture_sha256=capture,
        attention_KV_and_quantized_runtime_costs='UNKNOWN_EXCLUDED_FROM_PARTIAL_PROXY')


def pair_state(macro,piece,begins):
    """Use only the exact current macro BEGIN if it has been processed."""
    cutoff=piece['left_ns'];state=piece['state'];peers=state['peer_macros']
    own=begins.get((macro['owner'],macro['request_id'],macro['macro_id']))
    if own and own['available_at_processing_ns']>cutoff:own=None
    own_supported=own is not None and macro['kind']=='language_forward_macro'
    if not own_supported:own=None
    pairs=[];contrast_sum=0.;similarity_sum=0.;known=0
    for peer in peers:
        other=begins.get((peer['owner'],peer['request_id'],peer['phase_id']))
        if other and other['available_at_processing_ns']>cutoff:other=None
        supported=own_supported and other is not None
        contrast=similarity=None
        if supported:
            # log intensity contrast is bounded without a hardware threshold.
            logcontrast=abs(math.log(own['partial_decoder_projection_intensity']
                /other['partial_decoder_projection_intensity']))
            contrast=logcontrast/(1+logcontrast);similarity=1/(1+logcontrast)
            contrast_sum+=contrast;similarity_sum+=similarity;known+=1
        pairs.append(dict(peer_owner=peer['owner'],peer_request_id=peer['request_id'],
            peer_phase_id=peer['phase_id'],pair_supported=supported,
            unknown_reasons=[] if supported else
                ([] if own_supported else ['OWN_CURRENT_MACRO_PROJECTION_UNSUPPORTED_OR_BEGIN_NOT_PROCESSED'])
                +([] if other else ['PEER_ENTRY_OCR_OR_BEGIN_SHAPE_UNSUPPORTED']),
            peer_BEGIN_descriptor=other,partial_decoder_projection_contrast=contrast,
            partial_decoder_projection_similarity=similarity))
    n=len(peers);unknown=n-known
    base.require(n==state[COUNT] and known+unknown==n,'Exact visible peer denominator required')
    return dict(own_supported=own_supported,own_BEGIN_descriptor=own,peer_denominator_N=n,
        known_pair_denominator_K=known,unknown_pair_denominator_U=unknown,
        all_peer_contrast_sum=contrast_sum if unknown==0 else None,
        all_peer_similarity_sum=similarity_sum if unknown==0 else None,
        all_peer_mean_contrast=contrast_sum/n if unknown==0 and n else None,
        known_partial_decoder_projection_contrast_sum=contrast_sum,
        known_partial_decoder_projection_similarity_sum=similarity_sum,pairs=pairs,
        own_projection_unsupported_indicator=int(not own_supported),unknown_projection_pair_count=unknown,
        complementarity_or_hardware_classification='UNKNOWN_NOT_ESTABLISHED',
        proxy_is_actual_execution_resource_state=False,
        zero_known_pair_sum_means_empty_known_set_not_unknown_pair_imputation=True)


def enrich(doc,bindings,inventory):
    review=base.checked(doc['sources'][0]);raw=base.checked(review['inputs']['raw.json'])
    context=base.checked(raw['context']);capture=review['inputs']['raw.json']['sha256']
    natural_refs=[x['features'] for x in inventory['captures'] if x['mode']=='natural']
    prior_features=None
    for ref in natural_refs:
        existing=base.checked(ref)
        if existing['full_training_review']==doc['sources'][0]:prior_features=existing;break
    base.require(prior_features is not None,'Exact natural training BEGIN static features required')
    base.require(training_begin_features(doc['sources'][0]['path'],bindings)==prior_features,
        'Training BEGIN projection features must rederive exactly')
    events_by_job={}
    for row in raw['request_records']:events_by_job.setdefault(row['profile_job_id'],[]).append(row)
    begins={}
    for rows in events_by_job.values():
        for event in ordered_events(rows):
            family=context['residents'][event['owner']]
            if family in bindings and event.get('event')=='start' and event.get('semantic_phase') in ('prefill','decode'):
                feature=features_from_visible_begin(bindings[family],event,
                    parent_now_ns=event['collector_processing_time_ns'])
                key=event['owner'],event['request_id'],event['segment_id']
                base.require(key not in begins,'Unique capture/owner/request/BEGIN identity required')
                begins[key]=descriptor(feature,event,capture)
    records=[]
    for request in doc['requests']:
        for macro in request['macros']:
            path=[]
            for piece in macro['causal_covariate_path']:
                state=pair_state(macro,piece,begins)
                path.append(dict(left_ns=piece['left_ns'],right_ns=piece['right_ns'],
                    duration_ns=piece['duration_ns'],cutoff_processing_ns=piece['left_ns'],
                    peer_visible_macro_owner_count=piece['state'][COUNT],
                    peer_same_visible_macro_phase_owner_count=piece['state'][PHASE],projection_state=state))
            records.append(dict(capture_sha256=capture,request_id=macro['request_id'],owner=macro['owner'],
                macro_id=macro['macro_id'],kind=macro['kind'],response_wall_ns=macro['response_wall_ns'],path=path))
    return records,strict_references([natural_refs])


def coverage(records):
    result=dict(macros=len(records),language_macros=sum(r['kind']=='language_forward_macro' for r in records),
        time_pieces=0,total_response_ns=0,own_supported_ns=0,own_unsupported_ns=0,
        no_visible_peer_and_own_unsupported_ns=0,visible_peer_exposure_ns=0,
        known_pair_exposure_ns=0,unknown_pair_exposure_ns=0,all_peer_aggregate_unknown_ns=0)
    for record in records:
        for piece in record['path']:
            state=piece['projection_state'];duration=piece['duration_ns'];result['time_pieces']+=1
            result['total_response_ns']+=duration
            result['own_supported_ns' if state['own_supported'] else 'own_unsupported_ns']+=duration
            result['visible_peer_exposure_ns']+=duration*state['peer_denominator_N']
            result['known_pair_exposure_ns']+=duration*state['known_pair_denominator_K']
            result['unknown_pair_exposure_ns']+=duration*state['unknown_pair_denominator_U']
            if state['unknown_pair_denominator_U']:result['all_peer_aggregate_unknown_ns']+=duration
            if not state['own_supported'] and not state['peer_denominator_N']:
                result['no_visible_peer_and_own_unsupported_ns']+=duration
    result['known_pair_exposure_fraction']=result['known_pair_exposure_ns']/result['visible_peer_exposure_ns']
    return result


def prepared(documents,records,features):
    rows=base.prepared(documents,())
    base.require(len(rows)==len(records),'No unsupported macro may be dropped')
    for row,record in zip(rows,records):
        base.require((row['capture'],row['request_id'],row['macro_id'])==
            (record['capture_sha256'],record['request_id'],record['macro_id']),'Exact macro feature identity required')
        totals=defaultdict(float);ordered=[];durations=[]
        for piece in record['path']:
            state=dict(piece['projection_state'],**{COUNT:piece[COUNT],PHASE:piece[PHASE]})
            vector=tuple(state[f] for f in features)
            base.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in vector),
                'Finite nonnegative observed/count/masked-sum covariates required')
            duration=piece['duration_ns']/1e6;totals[vector]+=duration;ordered.append(vector);durations.append(duration)
        endpoint=ordered[-1];keys=[k for k in totals if k!=endpoint]+[endpoint]
        row.update(durations=np.asarray([totals[k] for k in keys]),
            matrix=np.asarray(keys,dtype=float).reshape(len(keys),len(features)),
            ordered_durations=np.asarray(durations),
            ordered_matrix=np.asarray(ordered,dtype=float).reshape(len(ordered),len(features)))
    return rows


def fit(rows,features):
    width=len(features);attempts=[];roots=[]
    if width:
        for start in (np.zeros(width),np.full(width,.25),np.full(width,-.25)):
            solved=minimize(lambda b:differential(b,rows)[:2],start,jac=True,
                hess=lambda b:differential(b,rows)[2],method='trust-exact',options={'gtol':1e-10,'maxiter':300})
            value,gradient,hessian=differential(solved.x,rows)
            attempts.append(dict(start=start.tolist(),solution=solved.x.tolist(),success=bool(solved.success),
                message=str(solved.message),mean_nll=value,**local_minimum_certificate(gradient,hessian)))
            stationary=root(lambda b:differential(b,rows)[1],solved.x,jac=lambda b:differential(b,rows)[2],
                method='lm',options={'ftol':1e-12,'xtol':1e-12,'gtol':1e-12,'maxiter':300})
            value,gradient,hessian=differential(stationary.x,rows)
            roots.append(dict(start=solved.x.tolist(),solution=stationary.x.tolist(),success=bool(stationary.success),
                message=str(stationary.message),mean_nll=value,**local_minimum_certificate(gradient,hessian)))
        valid=[r for r in roots if r['success'] and r['stationary_strict_local_minimum']]
        finite=[r for r in [*roots,*attempts] if math.isfinite(r['mean_nll'])]
        base.require(bool(finite),'No finite diagnostic solution')
        best=min(valid or finite,key=lambda r:r['mean_nll']);beta=np.asarray(best['solution'])
        disagreement=max(float(np.linalg.norm(np.asarray(r['solution'])-beta)) for r in roots)
        qualified=len(valid)==3 and disagreement<1e-7 and all(
            r['strictly_positive_curvature'] and np.linalg.norm(np.asarray(r['solution'])-beta)<1e-7 for r in attempts)
    else:beta=np.empty(0);qualified=True;disagreement=0.
    nll,mu,variance,derivatives,residual=base.profile(beta,rows)
    groups=sorted(mu,key=repr);indices={g:i for i,g in enumerate(groups)}
    nuisance=np.zeros((len(rows),len(groups)))
    for i,row in enumerate(rows):nuisance[i,indices[row['group']]]=-1
    joint=np.column_stack((nuisance,derivatives));singular=np.linalg.svd(joint,compute_uv=False)
    rank=int(np.linalg.matrix_rank(joint));dimension=len(groups)+width
    residual_derivatives=derivatives.copy()
    for group in groups:
        mask=[i for i,r in enumerate(rows) if r['group']==group]
        residual_derivatives[mask]-=np.mean(derivatives[mask],axis=0)
    return dict(features=list(features),coefficients=beta.tolist(),macro_rows=len(rows),nll=nll,
        mean_nll=nll/len(rows),lognormal_reference_sigma=math.sqrt(variance),
        baseline_nuisance='owner_input_task_macro_id_phase_kind_forward_shape',
        baselines=[dict(group=list(g),log_reference_ms=mu[g]) for g in groups],
        captures=sorted({r['capture'] for r in rows}),trust_exact_attempts=attempts,gradient_root_attempts=roots,
        numerical_qualification=bool(qualified and rank==dimension),root_max_solution_disagreement=disagreement,
        stationarity_certificate=local_minimum_certificate(*differential(beta,rows)[1:]) if width else None,
        nuisance_adjusted_derivative_rank=int(np.linalg.matrix_rank(residual_derivatives)) if width else 0,
        nuisance_adjusted_features=len(features),
        joint_response_jacobian=dict(rank=rank,parameters=dimension,full_column_rank=rank==dimension,
            singular_values=singular.tolist(),includes_baseline_nuisance_means=True,includes_sigma=False),
        effective_SM_service_identified=False,prospective_prediction_qualified=False),beta,mu,variance


def run(paths,inventory_path):
    inventory,bindings,sources=checked_static_inventory(inventory_path);docs=[];records=[]
    for path in paths:
        doc,closure=base.verified_document(path);docs.append(doc);sources.extend(closure)
        enriched,refs=enrich(doc,bindings,inventory);records.extend(enriched);sources.extend(refs)
    captures={r['capture_sha256'] for r in records}
    base.require(len(docs)==3 and len(captures)==3 and len(records)==1002,
        'Exactly three complete natural training captures and all 1002 macros required')
    results=[]
    for candidate,features in CANDIDATES.items():
        rows=prepared(docs,records,features);full,beta,mu,variance=fit(rows,features);folds=[];all_records=[]
        for heldout in sorted(captures):
            fitted,b,m,v=fit([r for r in rows if r['capture']!=heldout],features)
            evaluated=base.evaluate([r for r in rows if r['capture']==heldout],b,m,v)
            all_records.extend(evaluated['records']);folds.append(dict(heldout_capture_sha256=heldout,
                training_fit=fitted,evaluation=evaluated))
        full.update(candidate=candidate,whole_capture_leaveout=folds,pooled_leaveout=base.summarize(all_records),
            leaveout_selection_eligible=bool(full['numerical_qualification'] and all(f['training_fit']['numerical_qualification']
                and f['evaluation']['summary']['unknown_rows']==0 for f in folds)),
            all_capture_conditional_fit=base.evaluate(rows,beta,mu,variance))
        results.append(full)
    eligible=[r for r in results if r['leaveout_selection_eligible']]
    selected=min(eligible,key=lambda r:r['pooled_leaveout']['mean_nll'])['candidate'] if eligible else None
    closure=strict_references([sources,implementation_sources(__file__)])
    base.require(all(base.receipt(r['path'])==r for r in closure),'Source changed during contrast training')
    return dict(schema=SCHEMA,status='FULL_DOMAIN_PARTIAL_DECODER_PROJECTION_CONTRAST_CONDITIONAL_TRAINING_DIAGNOSTICS',
        sources=closure,mode='natural',response_domain=base.DOMAIN,coverage=coverage(records),
        candidates=results,selected_candidate=selected,feature_records=records,
        caption='Partial decoder projection intensity contrast; excludes attention/KV, DRAM/cache and quantized execution costs.',
        similarity_plus_contrast_equals_known_pair_count=True,
        known_pair_count_equals_visible_peer_count_minus_unknown_pairs=True,
        similarity_and_contrast_joint_fit_forbidden_due_exact_collinearity=True,
        unknown_pairs_imputed_as_zero=False,unsupported_macros_dropped=0,
        full_response_provider_complete=False,missingness_law_and_future_peer_paths='UNKNOWN_NOT_FITTED',
        physical_compute_memory_classification='UNKNOWN_NOT_ESTABLISHED',
        complementarity_or_interference_reduction_proven=False,measured_DRAM_cache_features_used=False,
        native_CUDA_API_numeric_authentication='UNQUALIFIED',independent_capture_robust_uncertainty=False,
        GPU_started=False,validation_labels_read=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        conditional_to_prospective_required_inputs=[
            'Actual receipted model/precision/topology binding and execution identity.',
            'Own candidate BEGIN shape/batch/phase qualified before dispatch, or generate its processing notice delay.',
            'Current peer owner/request/phase identity and each latest actually processed BEGIN shape with notice age.',
            'Generate future peer BEGIN/native-END and processing notifications causally, including cross-macro delivery.',
            'Joint generation stop/shape and missingness laws; preserve unsupported owner states instead of resource zero.',
            'Separate internal H/query/selection/RESULT/READY notification law and residual dependence.',
            'Whole independent capture validation of queue/policy predictions, beyond conditional path scores.'],
        limitations=[
            'Logical byte operands include repeated/aliased reads and hypothetical dense activation-dtype weights, never actual VRAM or DRAM.',
            'Last visible BEGIN shape describes a declared projection proxy during the entire visible macro tail, not current GPU resource activity.',
            'Own unsupported sets all N pairs unknown even with known peers; own_supported is retained even when N=0.',
            'Known-pair sums are sums over K observed pairs, with N/K/U explicit; all-peer aggregate is null whenever U>0.',
            'Free coefficients associate partial structure with response conditional on observed paths; no proven hardware complementarity.',
            'Three captures are the replication units; macro rows and 230 forward rows per capture are not independent captures.',
            'Conditional wall median inversion may assume endpoint-state continuation, as explicitly marked per row.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True)
    parser.add_argument('--static-inventory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.training_macros,args.static_inventory)
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],selected_candidate=result['selected_candidate'],coverage=result['coverage'])))
