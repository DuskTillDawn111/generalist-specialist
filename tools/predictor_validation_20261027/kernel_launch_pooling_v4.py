"""Alternative CPU launch-geometry duration hypothesis, including non-affine work.

This separate schema pools observed identical typed kernel implementations
across layers. Geometry is launch geometry, never inferred FLOPs or bytes.
Original affine models and evidence remain unchanged. Training fit does not
establish the cross-layer pooling or new-shape generalization hypotheses.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog,nnls

from .kernel_baseline_intake_v3 import checked,_verify,capture_identity,sqlite_identity
from .kernel_domain_bridge_review_v3 import load_report,request_kernels
from .native_collect import receipt,save
from .profiling import digest


FEATURES=('grid_x','grid_y','grid_z','grid_blocks')
SCHEMA='gsdeploy.real_launch_geometry_pooled_baseline.v4'


def geometry(signature):
    grid=signature.get('grid');block=signature.get('block')
    if not isinstance(grid,list) or not isinstance(block,list) or len(grid)!=3 or len(block)!=3:
        raise ValueError('Explicit three-dimensional grid and block required')
    if any(type(x) is not int or x<1 for x in grid+block):raise ValueError('Positive integer launch dimensions required')
    return dict(grid_x=grid[0],grid_y=grid[1],grid_z=grid[2],grid_blocks=math.prod(grid))


def pooling_group(domain,family,phase,batch,configured_precision,signature):
    geometry(signature)
    return dict(**domain,family=family,semantic_phase=phase,semantic_batch=batch,
        configured_precision=configured_precision,kernel_name=signature['name'],
        launch_settings={k:v for k,v in signature.items() if k not in ('name','grid')},
        feature_schema_id='gsdeploy.real_grid_axes_and_grid_block_count.v1',
        grouping_hypothesis='Same typed implementation/resources/phase/configured precision across layers; module and local ordinal omitted explicitly')


def fit_group(rows):
    runs=sorted({r['run_id'] for r in rows})
    result=dict(observation_count=len(rows),run_count=len(runs),run_ids=runs,
        min_observations=3,min_runs=2,max_relative_training_error_limit=0.25,
        independent_validation_passed=False)
    if len(rows)<3 or len(runs)<2:return dict(**result,fit_accepted=False,reasons=['INSUFFICIENT_OBSERVATIONS_OR_RUNS'])
    values=np.array([[r['features'][f] for f in FEATURES] for r in rows],dtype=float)
    # Positive features and nonnegative coefficients guarantee positive
    # numeric hypotheses without inventing a duration floor or trace fallback.
    center=np.zeros(len(FEATURES));scale=values.max(axis=0);scale[scale==0]=1
    design=np.column_stack((np.ones(len(rows)),(values-center)/scale))
    targets=np.array([r['duration_ns'] for r in rows],dtype=float)
    if np.any(~np.isfinite(targets)) or np.any(targets<=0):raise ValueError('Positive finite actual CUPTI targets required')
    coefficients,_=nnls(design,targets,maxiter=1000);rank=np.linalg.matrix_rank(design)
    relative=np.abs(design@coefficients-targets)/np.maximum(targets,1)
    worst=float(relative.max())
    return dict(**result,fit_accepted=worst<=0.25,
        reasons=[] if worst<=0.25 else ['TRAINING_RESIDUAL_EXCEEDS_25_PERCENT'],
        center=center.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
        feature_points=np.unique(values,axis=0).tolist(),affine_design_rank=int(rank),
        coefficient_count=1+len(FEATURES),max_relative_training_error=worst,
        regression='nonnegative_linear_duration_in_scaled_positive_launch_geometry',
        mean_relative_training_error=float(relative.mean()),
        training_error_is_generalization_bound=False)


def predict_group(model,features):
    if 'coefficients' not in model:return dict(known=False,duration_hypothesis_ns=None,reasons=model['reasons'])
    values=np.array([features[f] for f in FEATURES],dtype=float)
    center=np.array(model['center']);scale=np.array(model['scale'])
    query=np.r_[1.,(values-center)/scale]
    points=np.array(model['feature_points']);design=np.column_stack((np.ones(len(points)),(points-center)/scale))
    weights=np.linalg.lstsq(design.T,query,rcond=None)[0]
    affine=bool(np.max(np.abs(design.T@weights-query))<=1e-8)
    hull=False
    if affine:
        solution=linprog(np.zeros(len(points)),A_eq=design.T,b_eq=query,bounds=(0,None),method='highs')
        hull=bool(solution.success and np.max(np.abs(design.T@solution.x-query))<=1e-7)
    value=float(query@np.array(model['coefficients']))
    physical=math.isfinite(value) and value>0
    reasons=list(model['reasons'])
    if not affine:reasons.append('OUTSIDE_TRAINING_GEOMETRY_AFFINE_SUBSPACE')
    elif not hull:reasons.append('OUTSIDE_TRAINING_GEOMETRY_CONVEX_HULL')
    if not physical:reasons.append('NONPHYSICAL_REGRESSION')
    accepted=bool(model['fit_accepted'] and hull and physical)
    return dict(known=accepted,duration_hypothesis_ns=value if physical else None,
        accepted_duration_ns=value if accepted else None,reasons=reasons,
        inside_training_geometry_hull=hull,fit_accepted=model['fit_accepted'],
        cross_layer_pooling_independently_validated=False,new_shape_independently_validated=False,
        natural_timing_validated=False,deployment_qualified=False,measured_target_fallback_used=False)


def fit(intake_paths,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    groups={};group_rows=defaultdict(list);sources=[];requests=[];runs=set();count=0
    affine_counts=Counter();old_execution_keys=defaultdict(set);identities=set()
    rows_path=output/'observations.jsonl'
    with rows_path.open('w') as target:
        for intake_path in intake_paths:
            intake_source=receipt(intake_path);intake=checked(intake_source);sources.append(intake_source)
            for source in intake['sources']:_verify(source)
            bridge=checked(intake['bridge'])
            if bridge['status']!='PASS':raise ValueError('Passing exact bridge required')
            strict,plan,raw=load_report(intake['light_review']['path'],'light-kernel')
            context=checked(strict['context']);run,context_sha=capture_identity(strict,plan,raw,context)
            if run in runs:raise ValueError('Real raw capture SHA repeated; not a new independent run')
            runs.add(run)
            if run!=intake['capture_run_id'] or context_sha!=intake['execution_context_sha256']:
                raise ValueError('Prior intake differs from canonical capture identity')
            _verify(intake['rows']);affine={}
            with Path(intake['rows']['path']).open() as handle:
                for line in handle:
                    row=json.loads(line);affine[row['id']]=row
            first=next(iter(affine.values()));_verify(first['isolation_evidence'])
            isolation=checked(first['isolation_evidence'])
            if not all(isolation[k] is True for k in ('actual_request_intervals_disjoint','explicit_solo_jobs','sampled_telemetry_foreign_pids_absent')):
                raise ValueError('Strict source-bound isolated solo evidence required')
            owner_cache={}
            for alias,owner in strict['owners'].items():
                parsed=checked(owner['parsed']);_verify(owner['sqlite']);hardware,versions=sqlite_identity(owner['sqlite'])
                profiler='NVIDIA Nsight Systems:'+versions['EXPORT_PRODUCT_VERSION']+':light-kernel'
                if hardware!=first['hardware_id'] or profiler!=first['profiler_id']:raise ValueError('Actual hardware/profiler changed')
                owner_cache[alias]=(parsed,owner['sqlite'],dict(hardware_id=hardware,profiler_id=profiler,
                    execution_context_sha256=context_sha,timing_basis='profiled_isolated',
                    profiler_perturbed=True,perturbation_flags=sorted(first['perturbation_flags'])))
            for row in raw['request_records']:
                alias=row['model'];parsed,sqlite_source,domain=owner_cache[alias];family=context['residents'][alias]
                semantics={r['id']:r for r in row['worker_response'].get('semantic_forward_observations',{}).get('records',[])}
                req=dict(run_id=run,alias=alias,request_id=row['request_id'],pool_id=row['pool_id'],family=family,
                    raw=strict['raw'],parsed=strict['owners'][alias]['parsed'],sqlite=sqlite_source,
                    isolation=first['isolation_evidence'],context=strict['context'],ordered_rows=[])
                launches=request_kernels(parsed,sqlite_source['path'],row)
                for item in launches:
                    kernel=item['kernel'];signature=item['signature'];semantic=semantics.get(item['group'])
                    phase=semantic['semantic_phase'] if semantic else '__outside_language_forwards__'
                    batch=semantic['input_batch_size'] if semantic else None
                    group=pooling_group(domain,family,phase,batch,context['worker_bindings'][family]['execution'],signature)
                    group_id=digest(group);groups[group_id]=group
                    identity=f'{run}:{alias}:{row["request_id"]}:{kernel["correlation_id"]}:{kernel["nsys_start_ns"]}'
                    if identity in identities:raise ValueError('Duplicate exact CUPTI identity')
                    identities.add(identity)
                    duration=kernel['nsys_end_ns']-kernel['nsys_start_ns']
                    if duration<=0:raise ValueError('Actual positive CUPTI interval required')
                    observed=dict(id=identity,run_id=run,group_id=group_id,features=geometry(signature),duration_ns=duration,
                        exact_launch=dict(pid=kernel['pid'],context_id=kernel['context_id'],correlation_id=kernel['correlation_id'],
                            nsys_start_ns=kernel['nsys_start_ns'],nsys_end_ns=kernel['nsys_end_ns']),
                        previously_affine_bound=identity in affine)
                    target.write(json.dumps(observed,separators=(',',':'))+'\n')
                    group_rows[group_id].append(observed);req['ordered_rows'].append(dict(group_id=group_id,features=observed['features'],id=identity))
                    affine_counts[identity in affine]+=1;count+=1
                    if identity in affine:old_execution_keys[group_id].add(affine[identity]['execution_key'])
                requests.append(req)
    models={group_id:fit_group(rows) for group_id,rows in group_rows.items()}
    fit_counts=Counter();observation_counts=Counter();phase_counts=Counter();family_counts=Counter()
    for group_id,model in models.items():
        label='ACCEPTED_TRAINING_FIT' if model['fit_accepted'] else '+'.join(model['reasons'])
        fit_counts[label]+=1;observation_counts[label]+=model['observation_count']
        phase_counts[groups[group_id]['semantic_phase']]+=model['observation_count']
        family_counts[groups[group_id]['family']]+=model['observation_count']
    model_path=output/'models.json';save(model_path,dict(schema=SCHEMA,groups=groups,models=models,features=FEATURES))
    request_path=output/'request-sequences.json';save(request_path,requests)
    result=dict(schema=SCHEMA,status='CPU_ALTERNATIVE_POOLING_HYPOTHESIS_FITTED',intakes=sources,
        models=receipt(model_path),observations=receipt(rows_path),request_sequences=receipt(request_path),
        capture_run_ids=sorted(runs),independent_capture_count=len(runs),observation_count=count,
        group_count=len(groups),fit_group_counts=dict(fit_counts),fit_observation_counts=dict(observation_counts),
        prior_affine_observations=affine_counts[True],new_non_affine_observations=affine_counts[False],
        phase_observation_counts=dict(phase_counts),family_observation_counts=dict(family_counts),
        pooled_groups_with_multiple_old_affine_execution_keys=sum(len(v)>1 for v in old_execution_keys.values()),
        max_old_affine_execution_keys_in_one_group=max(map(len,old_execution_keys.values()),default=0),
        features=dict(names=FEATURES,meaning='Observed CUDA grid axes and product; block/resource/type/name remain exact group fields; never FLOPs or bytes'),
        precision_scope='Frozen configured model precision plus exact demangled typed kernel symbol, no claim of measured accumulator precision',
        group_policy='Canonical context/hardware/profiler/flags/family/phase/semantic batch/configured precision/name/launch attributes exact; module and local ordinal omitted as an explicit pooling hypothesis',
        thresholds=dict(min_observations=3,min_runs=2,max_relative_training_error=0.25),
        source_derived_work_features_added=False,geometry_only_proxy_limitations='Dynamic loop trip counts, KV length, operand strides and cache state are not established by launch geometry alone',
        intermediate_peak_model_known=False,cross_layer_pooling_independently_validated=False,
        heldout_validation_performed=False,natural_timing_validated=False,deployment_qualified=False,
        GPU_workload_executed=False,model_weights_loaded=False,implementation_sources=[receipt(__file__)])
    save(output/'report.json',result);return result


def instantiate_candidate(fit_report_path,candidate_path,output,family='text',template_model_path=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    fit_source=receipt(fit_report_path);report=checked(fit_source)
    for source in report['implementation_sources']:_verify(source)
    doc=checked(report['models']);candidate_source=receipt(candidate_path);candidate=checked(candidate_source)
    template=None;template_source=None
    if template_model_path:
        template_source=receipt(template_model_path);template=checked(template_source)
        for source in template['implementation_sources']:_verify(source)
        if candidate_source not in template['candidate_instances']:raise ValueError('Exact template candidate receipt required')
    # Select an explicitly observed prefill domain. A changed context is never
    # introduced by substituting the candidate's new capture context here.
    matches={key:group for key,group in doc['groups'].items() if group['family']==family and group['semantic_phase']=='prefill'}
    bases={digest({k:v for k,v in group.items() if k not in ('kernel_name','launch_settings')}):group for group in matches.values()}
    if len(bases)!=1:raise ValueError('Exactly one canonical observed prefill execution domain required')
    base=next(iter(bases.values()));predictions=[];cache={};reasons=Counter()
    for position,signature in enumerate(candidate['ordered_launch_signatures']):
        group=pooling_group({k:base[k] for k in ('hardware_id','profiler_id','execution_context_sha256','timing_basis','profiler_perturbed','perturbation_flags')},
            family,'prefill',base['semantic_batch'],base['configured_precision'],signature)
        group_id=digest(group);features=geometry(signature);key=(group_id,tuple(features.values()))
        if key not in cache:
            cache[key]=predict_group(doc['models'][group_id],features) if group_id in doc['models'] else dict(
                known=False,duration_hypothesis_ns=None,reasons=['EXACT_POOLED_KERNEL_DOMAIN_NOT_OBSERVED'])
        prediction=cache[key]
        if group_id not in doc['models'] and template is not None:
            pattern=template['patterns'][template['slot_pattern_indices'][position]]
            if pattern['kind']=='REDUCTION_BLOCK_RESOURCE_MODEL' and pattern['source_formula_verified']:
                anchors=sorted(map(int,pattern['anchor_signatures']));m=candidate['input_tokens'];policy=candidate['dispatch_policy']
                anchor=anchors[0] if policy=='lower_domain' or policy=='nearest_anchor' and m-anchors[0]<=anchors[1]-m else anchors[1]
                anchor_signature=pattern['anchor_signatures'][str(anchor)]
                transferred_group=pooling_group({k:base[k] for k in ('hardware_id','profiler_id','execution_context_sha256','timing_basis','profiler_perturbed','perturbation_flags')},
                    family,'prefill',base['semantic_batch'],base['configured_precision'],anchor_signature)
                transferred_id=digest(transferred_group)
                if transferred_id in doc['models']:
                    anchor_prediction=predict_group(doc['models'][transferred_id],geometry(anchor_signature))
                    prediction=dict(anchor_prediction,known=False,accepted_duration_ns=None,
                        reasons=anchor_prediction['reasons']+['SOURCE_CONDITIONAL_BLOCK_RESOURCE_TRANSFER_UNVALIDATED'],
                        resource_transfer_hypothesis=True,source_model_group_id=transferred_id,
                        source_template_model=template_source,source_anchor_tokens=anchor,
                        target_resources_are_a_different_domain=True)
        for reason in prediction['reasons']:reasons[reason]+=1
        predictions.append(dict(position=position,group_id=group_id,features=features,**prediction))
    values=[p['duration_hypothesis_ns'] for p in predictions]
    result=dict(schema='gsdeploy.pooled_launch_duration_sequence_candidate.v4',fit=fit_source,candidate=candidate_source,
        input_tokens=candidate['input_tokens'],dispatch_policy=candidate['dispatch_policy'],family=family,
        source_conditional_template_resource_transfer_model=template_source,
        execution_context_sha256=base['execution_context_sha256'],predictions=predictions,
        kernel_count=len(predictions),numeric_hypothesis_count=sum(v is not None for v in values),
        accepted_training_fit_and_hull_count=sum(p['known'] for p in predictions),unknown_reason_counts=dict(reasons),
        complete_numeric_sequence=all(v is not None for v in values),
        candidate_kernel_duration_sum_ns=sum(values) if all(v is not None for v in values) else None,
        sum_is_not_request_wall_time=True,natural_host_gaps_and_overlap_included=False,
        generated_decode_count_predicted=False,candidate_template_is_unvalidated=True,
        cross_layer_pooling_independently_validated=False,new_shape_independently_validated=False,
        natural_timing_validated=False,deployment_qualified=False,measured_target_fallback_used=False,
        implementation_sources=[receipt(__file__)])
    save(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('fit');f.add_argument('--intake',action='append',required=True);f.add_argument('--output',required=True)
    q=sub.add_parser('instantiate');q.add_argument('--fit',required=True);q.add_argument('--candidate',required=True);q.add_argument('--output',required=True)
    q.add_argument('--template-model')
    args=p.parse_args()
    if args.action=='fit':result=fit(args.intake,args.output);keys=('observation_count','group_count','fit_group_counts','fit_observation_counts')
    else:result=instantiate_candidate(args.fit,args.candidate,args.output,template_model_path=args.template_model);keys=('kernel_count','numeric_hypothesis_count','accepted_training_fit_and_hull_count','unknown_reason_counts')
    print(json.dumps({k:result[k] for k in keys}))


if __name__=='__main__':main()
