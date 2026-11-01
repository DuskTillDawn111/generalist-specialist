"""One fixed Gamma alternative on the original whole-H600-seed training split."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

from . import run_aware_training as training, run_aware_report as report
from . import gamma_profiles_h600_v1 as engine
from . import h600_event_predict_v2 as numerical
from .background_statistical import SCHEMA, family_task, DEFINITION as CONTEXT
from .duration_selection import evaluate
from .ocr_trained_validation import LIMITS
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

POLICY=dict(estimator='gamma_reciprocal_mean.v1',evidence='Original native raw plus strict source-bound mapper; no transient materialization dependency',variant='ocr_handler',handler_alpha=.1,
    semantic_alpha=.1,progress_bounds=[.05,8.],training_flows=16,
    CV='Two entire H600 seeds held out, four routes each; twelve retained flows plus old two profile collections',
    repeats_fitted=False,independent_outcomes_used=False,hyperparameter_sweep=False,
    numerical_engine='selected_completion_event_v2',
    selection='Decision errors including missing, temporal failed/unknown counts, normalized temporal mean; exact tie retains WLS',
    interpretation='Conditional mean-targeting score, endogenous observed fractions; no exact joint likelihood or physical identification')


def sources():
    return [receipt(p) for p in (__file__,engine.__file__,numerical.__file__,
        Path(__file__).with_name('h600_event_simulator_v2.py'))]


def prepare(training_protocol,baseline_report,resource_manifest,quality,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    training._verify_protocol(training_protocol)
    baseline=read(baseline_report)
    if (baseline.get('schema')!='gsdeploy.run_aware_H600_numerical_v2_training_report.v1'
            or baseline['run_aware_protocol']!=receipt(training_protocol)
            or baseline['resource_manifest']!=receipt(resource_manifest)
            or baseline.get('independent_validation') is not False):
        raise ValueError('Exact numerical-v2 original training baseline required')
    refs=[*sources(),receipt(training_protocol),receipt(baseline_report),receipt(resource_manifest),receipt(quality),
        *baseline['source_receipts']]
    for ref in refs:checked(ref)
    value=dict(schema='gsdeploy.fixed_Gamma_H600_training_protocol.v2',created_ns=time.perf_counter_ns(),
        policy=POLICY,training_protocol=receipt(training_protocol),baseline_report=receipt(baseline_report),
        resource_manifest=receipt(resource_manifest),quality=receipt(quality),source_receipts=refs,
        independent_validation=False,final_gain_eligible=False,GPU_started=False)
    value['protocol_sha256']=digest(value)
    output.mkdir(parents=True,exist_ok=False);save(output/'protocol.json',value)
    return value


def protocol(path):
    value=read(path)
    if (value.get('policy')!=POLICY or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed fixed Gamma protocol')
    for ref in value['source_receipts']:checked(ref)
    for ref in sources():
        if ref not in value['source_receipts']:raise ValueError('Gamma source version changed')
    return value


def fit_groups(bundle,groups,protocol_ref,seed):
    refs=training._unique_refs([protocol_ref,*sources(),bundle['protocol_receipt'],bundle['run_protocol_receipt'],
        *[g['raw'] for g in groups],*[r for g in groups for r in g['transitive_sources']]])
    traces=copy.deepcopy([t for g in groups for t in g['traces']])
    for trace in traces:
        if trace.get('split')!='training' or trace.get('not_for_training') is True:
            raise ValueError('Training-only observations required')
        for row in trace['requests']:row['family']=family_task(row['family'],row['task'])
    base=engine.fit_traces(traces,sources=refs,variant='ocr_handler')
    definition=copy.deepcopy(CONTEXT);definition.update(base_numerics=engine.variant_definition('ocr_handler'),
        estimator=POLICY['estimator'],variant='ocr_handler',scope=POLICY['interpretation'])
    groups_metadata=[{k:copy.deepcopy(g[k]) for k in
        ('group_id','role','raw','plan','seed','candidate_id','transitive_sources','run_metadata') if k in g} for g in groups]
    model=dict(schema=SCHEMA,definition=definition,definition_sha256=digest(definition),
        execution_context_sha256=base['execution_context_sha256'],source_receipts=refs,base_model=base,
        coverage=base['coverage'],core_enabled=False,GPU_started=False,
        prediction_status=dict(point_prediction_available=bool(base['profiles']),independently_validated=False,physically_identified=False),
        training_task_mixtures=[[dict(family=json.loads(m['family'])[0],task=json.loads(m['family'])[1],count=m['count'])
            for m in members] for members in base['family_contexts'].values()],
        gamma_training_evidence=dict(protocol=protocol_ref,heldout_seed=seed,groups=groups_metadata,
            training_flow_count=sum(g['role']=='continuous_training' for g in groups),
            original_profile_collections=sum(g['role']=='old_profile_training' for g in groups),
            diagnostic_repeats_fitted=False,validation_sources_used=[]))
    model['model_sha256']=digest(model)
    return model


def verify_model(path,bundle=None):
    model=read(path);evidence=model['gamma_training_evidence'];p=protocol(checked(evidence['protocol']))
    bundle=bundle or training.load_training(checked(p['training_protocol']))
    seed=evidence['heldout_seed']
    if seed is not None and seed not in bundle['run_protocol']['H600_seeds']:raise ValueError('Unknown omitted seed')
    groups=[g for g in bundle['groups'] if seed is None or not(g['role']=='continuous_training' and
        g['run_metadata']['source_role']=='H600_primary' and g['seed']==seed)]
    expected=fit_groups(bundle,groups,evidence['protocol'],seed)
    if model!=expected:raise ValueError('Gamma numerical model differs from recomputed source-bound fit')
    if evidence['training_flow_count']!=(16 if seed is None else 12):raise ValueError('Incorrect full/whole-seed flow count')
    return dict(verified=True,numerical_fit_recomputed=True,model=receipt(path),heldout_seed=seed,
        training_flow_count=evidence['training_flow_count'],diagnostic_repeats_fitted=False,validation_sources_used=[])


def run(protocol_path,output,full=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    p=protocol(protocol_path);bundle=training.load_training(checked(p['training_protocol']))
    output.mkdir(parents=True,exist_ok=False);save(output/'started.json',dict(protocol=receipt(protocol_path),full=full,created_ns=time.perf_counter_ns()))
    if full:
        model=fit_groups(bundle,bundle['groups'],receipt(protocol_path),None);path=output/'model.json';save(path,model)
        result=verify_model(path,bundle);save(output/'verification.json',result);return result
    resources,resource_refs=report.resource_folds(checked(p['resource_manifest']),bundle['run_protocol']['H600_seeds'])
    quality_doc=read(checked(p['quality']));quality=quality_doc.get('loss_by_family_task',quality_doc)
    rows=[];contracts={};models=[];all_refs=[receipt(protocol_path),*p['source_receipts'],*resource_refs]
    for seed in bundle['run_protocol']['H600_seeds']:
        heldout=[g for g in bundle['groups'] if g['role']=='continuous_training' and g['run_metadata']['source_role']=='H600_primary' and g['seed']==seed]
        omitted={g['group_id'] for g in heldout}
        groups=[g for g in bundle['groups'] if g['group_id'] not in omitted]
        if len(heldout)!=4:raise ValueError('Exactly four entire-seed heldout flows required')
        model=fit_groups(bundle,groups,receipt(protocol_path),seed);path=output/('seed-'+str(seed)+'-model.json');save(path,model)
        verification=verify_model(path,bundle);models.append(verification)
        fold=resources[seed];document=read(checked(fold['resources']))
        for group in heldout:
            plan=read(checked(group['plan']));candidate=next(c for c in plan['candidates'] if c['id']==group['candidate_id'])
            proposed=copy.deepcopy(candidate)
            for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):proposed[key]=fold['candidate_overrides'][candidate['id']][key]
            prediction=numerical.predict(plan,proposed,model,document,quality)
            raw,scored=report.core.scored_heldout(group['raw'])
            row=report.core.compare(plan,candidate,prediction,raw)
            row.update(seed=seed,flow_id=group['group_id'],model=receipt(path),heldout_raw=group['raw'],heldout_plan=group['plan'],heldout_scored=scored,
                independent_validation=False,predicted_candidate_resource_estimates=fold['candidate_overrides'][candidate['id']],resource_prediction_source=fold['resources'])
            rows.append(row);all_refs.extend([receipt(path),group['raw'],group['plan'],scored])
            contracts[seed]=dict(baseline=plan['baseline_candidate_id'],tie_tolerance=plan.get('utility_tie_tolerance',0.),
                candidate_ids=sorted(c['id'] for c in plan['candidates']),tasks=sorted(plan['workload_rps']))
            save(output/(group['group_id']+'-predictions.json'),prediction)
            print('GAMMA_CV_CASE',group['group_id'],row['known'],flush=True)
    value=dict(schema='gsdeploy.Gamma_H600_training_CV.v1',created_ns=time.perf_counter_ns(),policy=POLICY,protocol=receipt(protocol_path),
        candidates=rows,by_seed={str(seed):report.core.ranking([r for r in rows if r['seed']==seed],contract['baseline'],tie_tolerance=contract['tie_tolerance']) for seed,contract in contracts.items()},
        requested=sum(r['requested'] for r in rows),known=sum(r['known'] for r in rows),model_verifications=models,
        complete_prediction_coverage=all(r['known']==r['requested'] for r in rows),source_receipts=all_refs,
        independent_validation=False,independent_results_read=False,thresholds_tuned=False,final_gain_eligible=False,GPU_started=False)
    units=dict(seeds=sorted(contracts),routes=sorted(report.ROUTES));value['original_threshold_assessment']=evaluate(value,units,LIMITS,contracts)
    baseline=read(checked(p['baseline_report']))
    def score(r):
        a=r['original_threshold_assessment'];return [a['decision_errors'],a['temporal_failed_or_unknown'],a['normalized_temporal_error_mean'] if a['normalized_temporal_error_mean'] is not None else float('inf')]
    selected='gamma' if score(value)<score(baseline) else 'WLS'
    selection=dict(baseline=p['baseline_report'],gamma_score=score(value),baseline_score=score(baseline),selected=selected,
        independent_validation=False,original_thresholds_passed=value['original_threshold_assessment']['original_training_cv_thresholds_passed'] if selected=='gamma' else baseline['original_threshold_assessment']['original_training_cv_thresholds_passed'])
    for ref in all_refs:checked(ref)
    save(output/'report.json',value);save(output/'selection.json',selection)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='op',required=True)
    a=subs.add_parser('prepare')
    for name in ('training-protocol','baseline-report','resource-manifest','quality','output'):a.add_argument('--'+name,required=True)
    a=subs.add_parser('fit');a.add_argument('--protocol',required=True);a.add_argument('--output',required=True);a.add_argument('--full',action='store_true')
    a=subs.add_parser('verify');a.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.op=='prepare':value=prepare(args.training_protocol,args.baseline_report,args.resource_manifest,args.quality,args.output)
    elif args.op=='fit':value=run(args.protocol,args.output,args.full)
    else:value=verify_model(args.model)
    print(json.dumps(value if args.op!='prepare' else dict(protocol_sha256=value['protocol_sha256'])))
