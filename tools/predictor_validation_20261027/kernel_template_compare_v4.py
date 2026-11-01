"""CPU comparison of numeric template hypotheses to receipt-bound new captures.

Changed contexts remain explicit cross-context diagnostics. This comparator
never pools durations or changes any template coefficients.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import kernel_template_rules_v3 as rules
from .chatqa_public_shapes_v3 import public_content_sha
from .kernel_baseline_intake_v3 import checked, _verify
from .kernel_domain_bridge_review_v3 import load_report as load_v3, request_kernels
from .small_training_domain_bridge_review_v4 import load_report as load_v4
from .native_collect import context_identity, receipt, save
from .profiling import digest


def signature_comparison(predicted,actual):
    changed=Counter();first=None;matched=0
    for index,(left,right) in enumerate(zip(predicted,actual)):
        if left==right:matched+=1
        else:
            if first is None:first=index
            for key in left.keys()|right.keys():
                if left.get(key)!=right.get(key):changed[key]+=1
    if first is None and len(predicted)!=len(actual):first=min(len(predicted),len(actual))
    return dict(predicted_kernel_count=len(predicted),actual_kernel_count=len(actual),
        matched_position_count=matched,all_positions_match=predicted==actual,
        first_mismatch_index=first,changed_field_counts=dict(changed),
        predicted_sequence_sha256=digest(predicted),actual_sequence_sha256=digest(actual))


def compare(model_path,light_review_path,selection_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    model_source=receipt(model_path);model=checked(model_source)
    for source in model['implementation_sources']:_verify(source)
    public=checked(model['public_shape_source'])
    training_raw={e['raw']['sha256'] for t in public['observed_prefill_templates'].values() for e in t['evidence']}
    selection_source=receipt(selection_path);selection=checked(selection_source)
    selected=checked(selection['selected_input']);m=selection['selected']['input_tokens']
    if public_content_sha(selected['sample'])!=selection['selected']['public_content_sha256']:
        raise ValueError('Selected actual public content changed')
    source=receipt(light_review_path);strict=checked(source)
    loader=load_v4 if strict['schema']=='gsdeploy.startup_training_collection_strict_review.v4' else load_v3
    report,plan,raw=loader(light_review_path,'light-kernel')
    context=checked(report['context']);context_sha=context_identity(context)
    if context_sha!=plan['execution_context_sha256'] or context_sha!=raw['execution_context_sha256']:
        raise ValueError('Actual capture canonical context mismatch')
    independent=report['raw']['sha256'] not in training_raw
    results=[]
    for row in raw['request_records']:
        if row['pool_id']!=selected['pool_id'] or context['residents'][row['model']]!='text':continue
        if row['worker_response']['input_tokens']!=m:raise ValueError('Actual token count differs from frozen CPU tokenizer')
        owner=report['owners'][row['model']];parsed=checked(owner['parsed']);_verify(owner['sqlite'])
        actual=[item['signature'] for item in request_kernels(parsed,owner['sqlite']['path'],row)
            if item['group'].startswith('prefill_')]
        actual_path=output/(row['model'].replace('#','-')+'-'+row['request_id'].replace(':','-')+'-observed.json')
        save(actual_path,dict(ordered_launch_signatures=actual,input_tokens=m,
            raw=report['raw'],parsed=owner['parsed'],sqlite=owner['sqlite'],request_id=row['request_id'],
            context_sha256=context_sha,observation_domain='light-kernel',duration_targets_exported=False))
        for policy in ('nearest_anchor','lower_domain','upper_domain'):
            candidate=rules.instantiate_model(model,m,policy)
            result=signature_comparison(candidate['ordered_launch_signatures'],actual)
            result.update(alias=row['model'],request_id=row['request_id'],input_tokens=m,policy=policy,
                observation=receipt(actual_path),same_execution_context=context_sha==model['legal_support']['context'],
                independent_raw_capture=independent,unseen_training_shape=m not in model['training_lengths'],timing_used=False)
            result['same_domain_new_shape_validation_passed']=bool(result['all_positions_match'] and
                result['same_execution_context'] and independent and result['unseen_training_shape'])
            results.append(result)
    if not results:raise ValueError('No selected actual ChatQA request in this capture')
    result=dict(schema='gsdeploy.kernel_template_capture_comparison.v4',model=model_source,
        actual_light_review=source,training_selection=selection_source,comparisons=results,
        context_relationship='EXACT_CANONICAL_MATCH' if context_sha==model['legal_support']['context'] else 'CHANGED_CONTEXT_DIAGNOSTIC_ONLY',
        GPU_workload_executed=False,measured_durations_used=False,model_refitted=False,
        implementation_sources=[receipt(__file__),receipt(rules.__file__)],
        natural_timing_validated=False,deployment_qualified=False)
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','light-review','selection','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();r=compare(args.model,args.light_review,args.selection,args.output)
    print(json.dumps(dict(context_relationship=r['context_relationship'],comparisons=r['comparisons'])))


if __name__=='__main__':main()
