"""Separate real-shape/state feature schema over complete observed CUDA sequences.

CPU training only. Native-projection pooling is an explicit hypothesis; original
full contexts and raw captures remain identifiable. Affine FLOPs are declared
operator covariates, never assigned to non-affine kernels as invented work.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls,linprog

from .kernel_baseline_intake_v3 import WEIGHT_SELECTION,checked,_verify,sqlite_identity
from .kernel_domain_bridge_review_v3 import load_report as load_v3,request_kernels
from .small_training_domain_bridge_review_v4 import load_report as load_v4
from .kernel_launch_pooling_v4 import geometry
from .native_execution_projection_v5 import native_execution_projection
from .native_collect import context_identity,receipt,save
from .profiling import digest


def affine_features(operator):
    name=WEIGHT_SELECTION.get(operator['implementation'])
    weights=[t for t in operator['direct_tensors'] if t['name']==name]
    if len(weights)!=1:raise ValueError('Explicit actual primary weight tensor required')
    inp,out=operator['input'],operator['output'];batch,positions,k=inp['shape'];n=operator['out_features']
    if k!=operator['in_features'] or out['shape']!=[batch,positions,n]:raise ValueError('Unsupported actual affine shape')
    features=dict(affine_batch=batch,affine_M=batch*positions,affine_N=n,affine_K=k,
        affine_weight_logical_bytes=weights[0]['logical_bytes'],affine_input_logical_bytes=inp['logical_bytes'],
        affine_output_logical_bytes=out['logical_bytes'],affine_declared_dense_operator_flops=operator['dense_gemm_flops'])
    if features['affine_declared_dense_operator_flops']!=2*batch*positions*n*k:raise ValueError('Actual declared affine feature mismatch')
    precision=dict(input=inp['dtype'],output=out['dtype'],weight=weights[0]['dtype'],weight_tensor_name=name)
    return features,dict(implementation=operator['implementation'],tensor_storage_precision=precision)


def semantic_features(semantic,semantics,public_tokens,request_ordinal,phase_ordinal):
    result=dict(request_launch_ordinal=request_ordinal,phase_launch_ordinal=phase_ordinal)
    if type(public_tokens) is int and public_tokens>=0:result['observed_public_input_tokens']=public_tokens
    if semantic is not None:
        result.update(semantic_input_sequence_length=semantic['input_sequence_length'],
            semantic_input_batch_size=semantic['input_batch_size'],semantic_forward_index=semantic['forward_index'],
            semantic_input_token_positions=semantic['input_token_positions'],
            cumulative_declared_input_positions=sum(s['input_token_positions'] for s in semantics.values()
                if s['forward_index']<=semantic['forward_index']))
    if any(type(v) is not int or v<0 for v in result.values()):raise ValueError('Actual nonnegative integer state features required')
    return result


def fit_group(rows,names):
    runs=sorted({r['run_id'] for r in rows})
    result=dict(observation_count=len(rows),run_count=len(runs),run_ids=runs,feature_names=names,
        fit_accepted=False,min_observations=3,min_runs=2,max_relative_training_error_limit=0.25)
    if len(rows)<3 or len(runs)<2:return dict(**result,reasons=['INSUFFICIENT_OBSERVATIONS_OR_RUNS'])
    values=np.array([[r['features'][f] for f in names] for r in rows],dtype=float)
    scale=values.max(axis=0);scale[scale==0]=1
    design=np.column_stack((np.ones(len(rows)),values/scale));targets=np.array([r['duration_ns'] for r in rows],dtype=float)
    coefficients,_=nnls(design,targets,maxiter=1000);fitted=design@coefficients
    relative=np.abs(fitted-targets)/np.maximum(targets,1);worst=float(relative.max())
    result.update(fit_accepted=worst<=0.25,reasons=[] if worst<=0.25 else ['TRAINING_RESIDUAL_EXCEEDS_25_PERCENT'],
        scale=scale.tolist(),coefficients=coefficients.tolist(),affine_design_rank=int(np.linalg.matrix_rank(design)),
        feature_points=np.unique(values,axis=0).tolist(),
        coefficient_count=len(names)+1,max_relative_training_error=worst,mean_relative_training_error=float(relative.mean()),
        regression='nonnegative_linear_duration_with_actual_shape_state_covariates',training_error_is_generalization_bound=False,
        feature_ranges={f:[int(values[:,i].min()),int(values[:,i].max())] for i,f in enumerate(names)})
    return result


def predict_group(model,features):
    if 'coefficients' not in model:return dict(known=False,duration_hypothesis_ns=None,reasons=model['reasons'])
    if set(features)!=set(model['feature_names']):raise ValueError('Exact explicit feature schema required; missing values are not imputed')
    vector=np.r_[1.,np.array([features[f] for f in model['feature_names']],dtype=float)/np.array(model['scale'])]
    value=float(vector@np.array(model['coefficients']))
    physical=math.isfinite(value) and value>0
    points=model.get('_feature_point_set')
    if points is None:
        points={tuple(point) for point in model['feature_points']};model['_feature_point_set']=points
    query=tuple(features[f] for f in model['feature_names']);hull=query in points;affine=True
    if not hull:
        design=np.column_stack((np.ones(len(points)),np.array(list(points))/np.array(model['scale'])))
        weights=np.linalg.lstsq(design.T,vector,rcond=None)[0]
        affine=bool(np.max(np.abs(design.T@weights-vector))<=1e-8)
        if affine:
            result=linprog(np.zeros(len(points)),A_eq=design.T,b_eq=vector,bounds=(0,None),method='highs')
            hull=bool(result.success and np.max(np.abs(design.T@result.x-vector))<=1e-7)
    reasons=list(model['reasons'])+([] if physical else ['NONPHYSICAL_REGRESSION'])
    if not affine:reasons.append('OUTSIDE_ACTUAL_FEATURE_AFFINE_SUBSPACE')
    elif not hull:reasons.append('OUTSIDE_ACTUAL_FEATURE_CONVEX_HULL')
    return dict(known=bool(model['fit_accepted'] and hull and physical),duration_hypothesis_ns=value if physical else None,
        reasons=reasons,training_fit_accepted=model['fit_accepted'],inside_actual_feature_convex_hull=hull,
        prospective_convex_hull_support_checked=True,
        native_projection_pooling_is_hypothesis=True,measured_target_fallback_used=False)


def load_domain(source,domain):
    report=checked(source)
    loader=load_v4 if report['schema']=='gsdeploy.startup_training_collection_strict_review.v4' else load_v3
    return loader(source['path'],domain)


def fit(bridge_paths,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=[];contexts=[];groups={};group_rows=defaultdict(list);requests=[];seen=set();runs=set();counts=Counter()
    rows_path=output/'observations.jsonl'
    with rows_path.open('w') as target:
        for bridge_path in bridge_paths:
            bridge_source=receipt(bridge_path);bridge=checked(bridge_source);sources.append(bridge_source)
            if bridge['status']!='PASS' or not bridge['training_only']:raise ValueError('Passing training-only bridge required')
            _verify(bridge['evaluator'])
            mr,mp,mraw=load_domain(bridge['metadata_review'],'metadata')
            lr,lp,lraw=load_domain(bridge['light_review'],'light-kernel')
            context=checked(lr['context']);full=context_identity(context);projection=native_execution_projection(context)
            if full!=lp['execution_context_sha256'] or full!=lraw['execution_context_sha256']:raise ValueError('Actual canonical full context mismatch')
            if context['affine_observation'] is not False:raise ValueError('Kernel timing must not use affine hooks')
            run='raw-sha256:'+lr['raw']['sha256']
            if run in runs or lr['capture_run_id']!=run:raise ValueError('Actual independent raw capture identities required')
            runs.add(run);contexts.append(dict(context=lr['context'],projection=projection))
            jobs={j['id']:j for j in lp['jobs']}
            intervals=sorted((r['worker_response']['start_perf_ns'],r['worker_response']['end_perf_ns'])
                for r in lraw['request_records']+lraw.get('priming_request_records',[]))
            if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Concurrent requests invalidate isolated hypothesis')
            metadata={(r['profile_job_id'],r['model'],r['pool_id']):r for r in mraw['request_records']}
            mapped={(r['trial_job_id'],r['alias'],r['pool_id']):r for r in bridge['requests']}
            owners={}
            for alias,owner in lr['owners'].items():
                if not owner.get('request_ids'):continue
                parsed=checked(owner['parsed']);_verify(owner['sqlite']);hardware,versions=sqlite_identity(owner['sqlite'])
                owners[alias]=(parsed,owner,hardware,'NVIDIA Nsight Systems:'+versions['EXPORT_PRODUCT_VERSION']+':light-kernel')
            for lrow in lraw['request_records']:
                alias=lrow['model'];family=context['residents'][alias];identity=(lrow['profile_job_id'],alias,lrow['pool_id'])
                if list(jobs[lrow['profile_job_id']]['inputs'])!=[alias]:raise ValueError('Explicit one-owner request required')
                mrow=metadata[identity];bridge_request=mapped[identity]
                if not bridge_request['exact_work_matched'] or not bridge_request['all_observed_kernel_groups_matched']:
                    raise ValueError('Exact observed metadata/light work required')
                mapping=checked(bridge_request['mapping']);op_for_launch={}
                for semantic_group in mapping:
                    for pair in semantic_group['mapped_operator_kernels']:
                        launch=pair['light_kernel'];key=(launch['correlation_id'],launch['nsys_start_ns'])
                        if key in op_for_launch:raise ValueError('Ambiguous exact operator binding')
                        op_for_launch[key]=pair['operator_id']
                obs=mrow['worker_response'].get('semantic_forward_observations',{})
                operators={r['id']:r for r in obs.get('affine_operator_observations',{}).get('records',[])}
                semantics={r['id']:r for r in lrow['worker_response'].get('semantic_forward_observations',{}).get('records',[])}
                parsed,owner,hardware,profiler=owners[alias];launches=request_kernels(parsed,owner['sqlite']['path'],lrow)
                phase_ordinals=Counter();request=dict(run_id=run,alias=alias,family=family,pool_id=lrow['pool_id'],
                    request_id=lrow['request_id'],source_raw=lr['raw'],source_sqlite=owner['sqlite'],source_parsed=owner['parsed'],
                    source_mapping=bridge_request['mapping'],source_metadata=mr['raw'],full_context_sha256=full,
                    native_projection_sha256=projection['projection_sha256'],ordered_rows=[])
                for ordinal,item in enumerate(launches):
                    k=item['kernel'];signature=item['signature'];semantic=semantics.get(item['group'])
                    phase=semantic['semantic_phase'] if semantic else '__outside_language_forwards__'
                    features=geometry(signature)
                    features.update(semantic_features(semantic,semantics,lrow['worker_response'].get('input_tokens'),ordinal,phase_ordinals[item['group']]))
                    phase_ordinals[item['group']]+=1
                    opid=op_for_launch.get((k['correlation_id'],k['nsys_start_ns']));affine=None
                    if opid is not None:
                        actual,affine=affine_features(operators[opid]);features.update(actual)
                    group=dict(native_projection_sha256=projection['projection_sha256'],hardware_id=hardware,profiler_id=profiler,
                        timing_basis='profiled_isolated',family=family,semantic_phase=phase,
                        semantic_batch=semantic['input_batch_size'] if semantic else None,
                        configured_precision=context['worker_bindings'][family]['execution'],kernel_name=signature['name'],
                        launch_settings={key:value for key,value in signature.items() if key not in ('name','grid')},
                        affine_configuration=affine,feature_names=sorted(features),
                        feature_schema_id='gsdeploy.actual_geometry_affine_shape_and_semantic_state.v5')
                    group_id=digest(group);groups[group_id]=group
                    obsid=f'{run}:{alias}:{lrow["request_id"]}:{k["correlation_id"]}:{k["nsys_start_ns"]}'
                    if obsid in seen:raise ValueError('Duplicate real launch observation')
                    seen.add(obsid);duration=k['nsys_end_ns']-k['nsys_start_ns']
                    if duration<=0:raise ValueError('Actual positive CUPTI interval required')
                    row=dict(id=obsid,run_id=run,group_id=group_id,features=features,duration_ns=duration,
                        actual_full_context_sha256=full,semantic_group=item['group'],
                        exact_launch=dict(pid=k['pid'],context_id=k['context_id'],correlation_id=k['correlation_id'],
                            nsys_start_ns=k['nsys_start_ns'],nsys_end_ns=k['nsys_end_ns']))
                    target.write(json.dumps(row,separators=(',',':'))+'\n');group_rows[group_id].append(row)
                    request['ordered_rows'].append(dict(id=obsid,group_id=group_id,features=features,semantic_group=item['group']))
                    counts['affine' if affine else 'non_affine']+=1
                requests.append(request)
    models={key:fit_group(rows,groups[key]['feature_names']) for key,rows in group_rows.items()}
    model_path=output/'models.json';save(model_path,dict(schema='gsdeploy.semantic_shape_kernel_models.v5',groups=groups,models=models))
    # Numeric predictions are independent of row targets. Targets below are
    # used only for explicitly labelled training aggregate diagnostics.
    targets={r['id']:r['duration_ns'] for rows in group_rows.values() for r in rows}
    aggregate=[];fit_counts=Counter();fit_rows=Counter();candidate_count=0
    for key,model in models.items():
        label='ACCEPTED_TRAINING_FIT' if model['fit_accepted'] else '+'.join(model['reasons'])
        fit_counts[label]+=1;fit_rows[label]+=model['observation_count']
    for request in requests:
        predicted=[];bucket=defaultdict(list)
        for row in request['ordered_rows']:
            model=models[row['group_id']];prediction=predict_group(model,row['features']);actual=targets[row['id']]
            candidate_count+=prediction['duration_hypothesis_ns'] is not None
            predicted.append(dict(id=row['id'],group_id=row['group_id'],**prediction))
            bucket[row['semantic_group']].append((prediction['duration_hypothesis_ns'],actual))
        path=output/(request['run_id'].split(':')[1][:12]+'-'+request['alias'].replace('#','-')+'-'+request['request_id'].replace(':','-')+'-predictions.json')
        save(path,dict(ordered_predictions=predicted,training_diagnostics_only=True))
        values=[(p['duration_hypothesis_ns'],targets[p['id']]) for p in predicted]
        def summary(pairs):
            available=[p for p in pairs if p[0] is not None];actual=sum(a for _,a in pairs)
            predicted_sum=sum(p for p,_ in available);covered_actual=sum(a for _,a in available)
            return dict(kernel_count=len(pairs),numeric_count=len(available),complete_numeric_sequence=len(available)==len(pairs),
                actual_kernel_sum_ns=actual,predicted_available_kernel_sum_ns=predicted_sum,
                relative_sum_error_on_covered_kernels=abs(predicted_sum-covered_actual)/covered_actual if covered_actual else None,
                relative_complete_kernel_sum_error=abs(predicted_sum-actual)/actual if len(available)==len(pairs) else None,
                sum_absolute_kernel_errors_ns=sum(abs(p-a) for p,a in available))
        aggregate.append(dict(run_id=request['run_id'],alias=request['alias'],request_id=request['request_id'],family=request['family'],
            pool_id=request['pool_id'],predictions=receipt(path),all_kernels=summary(values),
            phase_sums={phase:summary(pairs) for phase,pairs in bucket.items()},
            tiny_kernel_diagnostic_threshold_ns=10000,tiny_kernels=summary([p for p in values if p[1]<=10000]),
            larger_kernels=summary([p for p in values if p[1]>10000]),training_not_independent_validation=True,
            kernel_sum_is_not_request_wall_time=True))
    request_path=output/'request-sequences.json';save(request_path,requests)
    aggregate_path=output/'training-aggregate-diagnostics.json';save(aggregate_path,aggregate)
    report=dict(schema='gsdeploy.semantic_shape_kernel_baseline_fit.v5',bridges=sources,contexts=contexts,
        models=receipt(model_path),observations=receipt(rows_path),request_sequences=receipt(request_path),
        training_aggregate_diagnostics=receipt(aggregate_path),capture_run_ids=sorted(runs),independent_capture_count=len(runs),
        observation_count=len(seen),affine_feature_observation_count=counts['affine'],non_affine_observation_count=counts['non_affine'],
        group_count=len(groups),fit_group_counts=dict(fit_counts),fit_observation_counts=dict(fit_rows),
        numeric_training_hypothesis_count=candidate_count,
        thresholds=dict(min_observations=3,min_runs=2,max_relative_training_error=0.25),
        actual_kv_cache_length_observed=False,
        cache_proxy_scope='Cumulative declared input positions from actual semantic forward records, not measured KV tensor length',
        affine_flops_scope='Actual declared dense operator FLOPs associated by exact observer/kernel bridge, not per-kernel executed FLOPs',
        non_affine_flops_or_bytes_invented=False,missing_features_imputed=False,
        full_contexts_preserved=True,native_projection_pooling_is_unvalidated_hypothesis=True,
        heldout_validation_performed=False,natural_timing_validated=False,deployment_qualified=False,
        GPU_workload_executed=False,model_weights_loaded=False,
        implementation_sources=[receipt(__file__),receipt(Path(__file__).with_name('native_execution_projection_v5.py')),
            receipt(Path(__file__).with_name('kernel_launch_pooling_v4.py')),
            receipt(Path(__file__).with_name('kernel_baseline_intake_v3.py')),
            receipt(Path(__file__).with_name('kernel_domain_bridge_review_v3.py')),
            receipt(Path(__file__).with_name('small_training_domain_bridge_review_v4.py'))])
    save(output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bridge',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=fit(a.bridge,a.output)
    print(json.dumps({k:r[k] for k in ('observation_count','group_count','fit_group_counts','fit_observation_counts','numeric_training_hypothesis_count')}))


if __name__=='__main__':main()
