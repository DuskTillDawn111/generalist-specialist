"""Explicit common risk domain: preserve original context SHA and exact proof."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.resources import identity
from .background_collect import context_identity
from .horizon_recovery_risk_v31 import SCHEMA
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

ROUTES=('balanced','skewed','spill_low','spill_high')
ALLOWED_METADATA_DIFFERENCES=('execution_nonce','whole57_training_route_id')


def leaf_differences(left,right,path=''):
    if isinstance(left,dict) and isinstance(right,dict):
        result=[]
        for key in sorted(set(left)|set(right)):
            child=path+'/'+str(key)
            if key not in left or key not in right:result.append(dict(path=child,left=left.get(key),right=right.get(key),missing_side='left' if key not in left else 'right'))
            else:result.extend(leaf_differences(left[key],right[key],child))
        return result
    if isinstance(left,list) and isinstance(right,list):
        result=[]
        for index in range(max(len(left),len(right))):
            child=path+'/'+str(index)
            if index>=len(left) or index>=len(right):result.append(dict(path=child,left=left[index] if index<len(left) else None,right=right[index] if index<len(right) else None,missing_side='left' if index>=len(left) else 'right'))
            else:result.extend(leaf_differences(left[index],right[index],child))
        return result
    return [] if type(left)==type(right) and left==right else [dict(path=path,left=left,right=right)]


def execution_projection(context,plan):
    return dict(context={k:copy.deepcopy(v) for k,v in context.items() if k not in ALLOWED_METADATA_DIFFERENCES},
        resource_configuration=copy.deepcopy(plan['resource_configuration']),
        original_input_registry=copy.deepcopy(plan['inputs']),recipe=copy.deepcopy(plan['recipe']))


def actual_runtime_projection(raw):
    if raw.get('final_source_audit')!='PASS' or raw.get('errors') or raw.get('runtime_errors'):
        raise ValueError('Actual runtime source/collector error audit must PASS')
    if len(raw.get('request_records',[]))!=57 or len(raw.get('priming_request_records',[]))!=12:
        raise ValueError('Complete actual 57+12 cohort required')
    if any(row.get('status')!='completed' for row in raw['request_records']+raw['priming_request_records']):
        raise ValueError('All actual measured and priming requests must complete')
    return dict(domain=raw['domain'],ready={alias:{k:copy.deepcopy(v) for k,v in worker.items() if k!='pid'}
        for alias,worker in sorted(raw['ready'].items())})


def build(prepared_root,actual_review_paths=()):
    prepared_root=Path(prepared_root);refs=[receipt(__file__)];members=[];contexts=[];plans=[]
    for route in ROUTES:
        context_ref=receipt(prepared_root/route/'context.json');plan_ref=receipt(prepared_root/route/'plan.json')
        refs.extend([context_ref,plan_ref]);context=read(context_ref['path']);plan=read(plan_ref['path'])
        if context_identity(context)!=plan['execution_context_sha256'] or plan.get('training_only') is not True:
            raise ValueError('Frozen original execution context identity mismatch')
        if plan['route_id']!=route or context['whole57_training_route_id']!=route or not context['execution_nonce']:
            raise ValueError('Predeclared route/nonce mismatch')
        if context['measurement_domain']!='whole57-joint-natural-v23':raise ValueError('Natural domain required; do not bridge profiling domains')
        contexts.append(context);plans.append(plan)
        members.append(dict(route_id=route,context=context_ref,plan=plan_ref,
            original_execution_context_sha256=plan['execution_context_sha256'],
            omitted_metadata={key:context[key] for key in ALLOWED_METADATA_DIFFERENCES}))
    projection=execution_projection(contexts[0],plans[0]);comparisons=[];compatible=True
    for route,context,plan in zip(ROUTES[1:],contexts[1:],plans[1:]):
        differences=leaf_differences(projection,execution_projection(context,plan));compatible&=not differences
        comparisons.append(dict(reference_route=ROUTES[0],compared_route=route,
            complete_context_differences=leaf_differences(contexts[0],context),
            execution_projection_exact_equal=not differences,unexpected_execution_differences=differences))
    prepared_compatible=compatible
    actual_by_route={};runtime_comparisons=[]
    for review_path in actual_review_paths:
        review_ref=receipt(review_path);review=read(review_path)
        if review.get('status')!='PASS' or review.get('training_only') is not True:raise ValueError('Passed actual training review required')
        raw_ref=review['evidence']['raw.json'];plan_ref=review['evidence']['plan.json'];context_ref=review['evidence']['context.json']
        for ref in (raw_ref,plan_ref,context_ref):
            if receipt(ref['path'])!=ref:raise ValueError('Changed reviewed runtime evidence')
        raw=read(raw_ref['path']);route=raw['route_id']
        if route not in ROUTES or route in actual_by_route:raise ValueError('Duplicate/unknown actual route')
        member=next(m for m in members if m['route_id']==route)
        if member['plan']!=plan_ref or member['context']!=context_ref or raw['execution_context_sha256']!=member['original_execution_context_sha256']:
            raise ValueError('Actual raw/plan/context does not match original member identity')
        refs.extend([review_ref,raw_ref]);actual=actual_runtime_projection(raw)
        actual_by_route[route]=dict(review=review_ref,raw=raw_ref,projection=actual,
            actual_READY_PIDs={alias:worker['pid'] for alias,worker in raw['ready'].items()})
    if actual_by_route:
        reference=sorted(actual_by_route)[0]
        for route,item in sorted(actual_by_route.items()):
            differences=leaf_differences(actual_by_route[reference]['projection'],item['projection'])
            compatible&=not differences
            runtime_comparisons.append(dict(reference_route=reference,compared_route=route,
                actual_runtime_projection_exact_equal=not differences,differences=differences))
    all_actual=set(actual_by_route)==set(ROUTES)
    qualification=('ALL_MEMBERS_PASS_EXACT_RUNTIME_PROJECTION' if compatible and all_actual else
                   'UNKNOWN_RUNTIME_OR_PREPARED_EXECUTION_DIFFERENCE' if not compatible else
                   'PREPARED_PROJECTION_EQUAL_WAITING_FOR_ALL_ACTUAL_MEMBERS')
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Domain evidence bytes changed during audit')
    domain=dict(schema=SCHEMA+'.risk_domain',kind='measured',
        actual_execution_qualification=qualification,
        member_execution_context_sha256=[m['original_execution_context_sha256'] for m in members],members=members,
        allowed_metadata_differences=list(ALLOWED_METADATA_DIFFERENCES),
        complete_context_comparisons=comparisons,common_execution_projection=projection,
        common_execution_projection_sha256=identity(projection),
        actual_members=actual_by_route,actual_runtime_comparisons=runtime_comparisons,
        prepared_execution_projection_exact_equal=prepared_compatible,
        source_receipts_before=refs,source_receipts_after=after,
        scope='same_recipe_content_natural_training_capture_diagnostic_only',
        original_execution_context_SHA_rewritten=False,profiler_domain_transfer_claimed=False,
        input_generalization_claimed=False,probability_calibrated=False,physical_accuracy_qualified=False)
    domain['risk_domain_sha256']=identity(domain)
    return domain


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared-root',required=True)
    parser.add_argument('--actual-review',action='append',default=[]);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    result=build(args.prepared_root,args.actual_review);output.mkdir(parents=True)
    (output/'risk-domain.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(actual_execution_qualification=result['actual_execution_qualification'],risk_domain_sha256=result['risk_domain_sha256']),indent=2))
