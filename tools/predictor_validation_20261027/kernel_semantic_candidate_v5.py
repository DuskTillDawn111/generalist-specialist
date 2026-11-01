"""Instantiate rich-feature duration hypotheses on full numeric prefill templates.

Projects only exact constant/integer-linear state/operator relations checked
against all retained public training lengths. Never reads measured durations
from a query sequence or invents non-affine FLOPs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .kernel_baseline_intake_v3 import checked,_verify
from .kernel_launch_pooling_v4 import geometry
from .kernel_semantic_pooling_v5 import predict_group
from .native_collect import receipt,save
from .profiling import digest


def scalar_relation(m0,v0,m1,v1):
    if v0==v1:return dict(kind='constant',value=v0)
    if v0%m0==0 and v1%m1==0 and v0//m0==v1//m1:return dict(kind='integer_linear_public_tokens',coefficient=v0//m0)
    raise ValueError('Actual feature relation is not identified by an exact supported public-shape rule')


def evaluate(rule,m):
    return rule['value'] if rule['kind']=='constant' else rule['coefficient']*m


def instantiate(fit_path,template_path,candidate_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    fit_source=receipt(fit_path);fit=checked(fit_source)
    for source in fit['implementation_sources']:_verify(source)
    doc=checked(fit['models']);requests=checked(fit['request_sequences'])
    template_source=receipt(template_path);template=checked(template_source)
    for source in template['implementation_sources']:_verify(source)
    candidate_source=receipt(candidate_path);candidate=checked(candidate_source)
    if candidate_source not in template['candidate_instances']:raise ValueError('Exact model-bound numeric candidate required')
    m=candidate['input_tokens'];lengths=template['training_lengths'];m0,m1=lengths
    if not m0<=m<=m1:raise ValueError('Observed public length bracket required')
    training=[]
    for request in requests:
        if request['family']!='text':continue
        rows=[r for r in request['ordered_rows'] if r['semantic_group'].startswith('prefill_')]
        if not rows:continue
        length=rows[0]['features']['observed_public_input_tokens']
        training.append((length,request,rows))
    left=next(rows for length,_,rows in training if length==m0)
    right=next(rows for length,_,rows in training if length==m1)
    if any(len(rows)!=len(left) for _,_,rows in training) or len(left)!=len(candidate['ordered_launch_signatures']):
        raise ValueError('Different ordered topology requires a new feature projection')
    geometry_names=set(geometry(candidate['ordered_launch_signatures'][0]));relations=[]
    for position,(a,b) in enumerate(zip(left,right)):
        ga,gb=doc['groups'][a['group_id']],doc['groups'][b['group_id']]
        excluded={'kernel_name','launch_settings'}
        if {k:v for k,v in ga.items() if k not in excluded}!={k:v for k,v in gb.items() if k not in excluded}:
            raise ValueError('Public-shape feature domains differ beyond observed kernel/resource dispatch')
        if set(a['features'])!=set(b['features']):raise ValueError('Shape feature schema changed')
        rules={f:scalar_relation(m0,a['features'][f],m1,b['features'][f]) for f in a['features'] if f not in geometry_names}
        for length,request,rows in training:
            if any(evaluate(rule,length)!=rows[position]['features'][f] for f,rule in rules.items()):
                raise ValueError('Feature projection fails a real retained training shape')
        relations.append(rules)
    predictions=[];reasons=Counter()
    for position,signature in enumerate(candidate['ordered_launch_signatures']):
        base=doc['groups'][left[position]['group_id']]
        group=dict(base,kernel_name=signature['name'],launch_settings={k:v for k,v in signature.items() if k not in ('name','grid')})
        group_id=digest(group);features={f:evaluate(rule,m) for f,rule in relations[position].items()};features.update(geometry(signature))
        if group_id in doc['models']:prediction=predict_group(doc['models'][group_id],features)
        else:prediction=dict(known=False,duration_hypothesis_ns=None,reasons=['EXACT_NATIVE_PROJECTION_KERNEL_DOMAIN_NOT_OBSERVED'])
        pattern=template['patterns'][template['slot_pattern_indices'][position]]
        if prediction['duration_hypothesis_ns'] is None and pattern['kind']=='REDUCTION_BLOCK_RESOURCE_MODEL' and pattern['source_formula_verified']:
            anchor=m0 if candidate['dispatch_policy']=='lower_domain' or candidate['dispatch_policy']=='nearest_anchor' and m-m0<=m1-m else m1
            anchor_rows=left if anchor==m0 else right;source_group_id=anchor_rows[position]['group_id']
            anchor_prediction=predict_group(doc['models'][source_group_id],features)
            prediction=dict(anchor_prediction,known=False,
                reasons=anchor_prediction['reasons']+['SOURCE_CONDITIONAL_REDUCTION_RESOURCE_TRANSFER_UNVALIDATED'],
                resource_transfer_hypothesis=True,source_group_id=source_group_id,
                target_resources_are_a_different_domain=True)
        for reason in prediction['reasons']:reasons[reason]+=1
        predictions.append(dict(position=position,group_id=group_id,features=features,**prediction))
    numeric=[p['duration_hypothesis_ns'] for p in predictions];complete=all(v is not None for v in numeric)
    relations_path=output/'feature-projections.json';save(relations_path,relations)
    result=dict(schema='gsdeploy.semantic_shape_prefill_duration_candidate.v5',fit=fit_source,template=template_source,candidate=candidate_source,
        input_tokens=m,dispatch_policy=candidate['dispatch_policy'],kernel_count=len(predictions),
        numeric_hypothesis_count=sum(v is not None for v in numeric),accepted_fit_and_feature_hull_count=sum(p['known'] for p in predictions),
        complete_numeric_sequence=complete,candidate_kernel_duration_sum_ns=sum(numeric) if complete else None,
        unknown_reason_counts=dict(reasons),ordered_predictions=predictions,feature_projections=receipt(relations_path),
        source_training_lengths=sorted({length for length,_,_ in training}),feature_projection_checked_on_all_retained_training_shapes=True,
        candidate_sequence_and_dispatch_are_hypotheses=True,non_affine_flops_invented=False,
        actual_future_kv_cache_length_claimed=False,actual_generated_token_count_predicted=False,
        kernel_duration_sum_is_not_request_wall_time=True,measured_query_target_fallback_used=False,
        full_context_equality_claimed=False,native_projection_pooling_is_hypothesis=True,
        independent_new_shape_validation_passed=False,natural_timing_validated=False,deployment_qualified=False,
        GPU_workload_executed=False,model_weights_loaded=False,
        implementation_sources=[receipt(__file__),receipt(Path(__file__).with_name('kernel_semantic_pooling_v5.py'))])
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','template','candidate','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=instantiate(a.fit,a.template,a.candidate,a.output)
    print({k:r[k] for k in ('kernel_count','numeric_hypothesis_count','accepted_fit_and_feature_hull_count','candidate_kernel_duration_sum_ns','unknown_reason_counts')})


if __name__=='__main__':main()
