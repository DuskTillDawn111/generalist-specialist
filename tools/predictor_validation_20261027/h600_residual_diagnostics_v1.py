"""Fresh training-CV residual diagnostics, prepared separately from execution.

No fit or GPU. Predictions accept public trace/routes and frozen fold evidence
only. Actual worker/parent clocks are joined after prediction for diagnostics.
"""
from __future__ import annotations

import argparse
import collections
import copy
import math
from pathlib import Path
import statistics
import time

from gsdeploy.forecast_errors import UnknownEvidence
from . import run_aware_report as frozen
from .native_flow import checked,receipt,save
from .probe_validation import read
from .validation import percentile

SCHEMA='gsdeploy.H600_training_CV_residual_diagnostics.v1'
MIN_INPUT_N=5
POLICY=dict(population='Every public offered request, including forecast UNKNOWN; exactly 1200 per case',
    sign='predicted minus actual',minimum_input_count=MIN_INPUT_N,
    sum_service='Sum of paired predicted minus actual worker service; complete-cohort sum UNKNOWN if any prediction missing; not a causal queue decomposition',
    e2e='Forecast worker finish minus public arrival versus actual parent receipt minus public arrival; parent tail separately observed',
    submit_wait='Actual host send-start actual_submit_time_ns through actual worker START, separately from scheduled-arrival wait',
    P95='Full-cohort predicted P95 only when all forecast values known; observed P95 retains every request; paired-subset P95 explicitly separate',
    arrival_thirds='Equal thirds of public scheduled arrival horizon; ordinal 0,1,2 independent of observed finish',
    W='Observed-phase versus frozen provider plan vocabulary audit only; no observed-context work integration. Vocabulary UNKNOWN does not imply fresh forecast UNKNOWN',
    scope='Training CV only; no fitting, selection, causal attribution, independent outcomes or gain qualification')


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def stats(values):
    values=list(values)
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        sum=sum(values) if values else None,p95=percentile(values) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None)


def residual_stats(rows,metric):
    pairs=[(r['predicted_'+metric+'_ms'],r['actual_'+metric+'_ms']) for r in rows
        if finite(r.get('predicted_'+metric+'_ms')) and finite(r.get('actual_'+metric+'_ms'))]
    observed=[r['actual_'+metric+'_ms'] for r in rows if finite(r.get('actual_'+metric+'_ms'))]
    predicted=[p for p,a in pairs];actual_paired=[a for p,a in pairs];errors=[p-a for p,a in pairs]
    complete=len(pairs)==len(rows) and bool(rows)
    full_predicted_p95=percentile(predicted) if complete else None
    observed_p95=percentile(observed) if len(observed)==len(rows) and rows else None
    return dict(offered_count=len(rows),paired_known_count=len(pairs),unknown_count=len(rows)-len(pairs),
        complete_cohort_prediction=complete,paired_signed_bias_ms=statistics.mean(errors) if pairs else None,
        paired_sum_predicted_minus_actual_ms=sum(errors) if pairs else None,
        full_cohort_sum_predicted_minus_actual_ms=sum(errors) if complete else None,
        paired_mean_absolute_error_ms=statistics.mean(abs(e) for e in errors) if pairs else None,
        full_predicted_p95_ms=full_predicted_p95,full_actual_p95_ms=observed_p95,
        full_p95_signed_difference_ms=full_predicted_p95-observed_p95 if full_predicted_p95 is not None and observed_p95 is not None else None,
        paired_subset_predicted_p95_ms=percentile(predicted),paired_subset_actual_p95_ms=percentile(actual_paired),
        paired_error_p95_ms=percentile(errors),observed=stats(observed))


def summarize(rows,*,input_group=False):
    if not rows:raise ValueError('Nonempty offered group required')
    known=sum(finite(r.get('predicted_e2e_ms')) for r in rows)
    return dict(offered_count=len(rows),fresh_forecast_known_count=known,unknown_count=len(rows)-known,
        input_minimum_n=MIN_INPUT_N if input_group else None,
        sparse_input_group=input_group and len(rows)<MIN_INPUT_N,
        service=residual_stats(rows,'service'),e2e=residual_stats(rows,'e2e'),wait=residual_stats(rows,'wait'),
        parent_tail_ms=stats(r['parent_tail_ms'] for r in rows if finite(r.get('parent_tail_ms'))),
        actual_submit_to_START_ms=stats(r['submit_to_START_ms'] for r in rows if finite(r.get('submit_to_START_ms'))),
        scheduled_to_actual_submit_ms=stats(r['scheduled_to_submit_ms'] for r in rows if finite(r.get('scheduled_to_submit_ms'))),
        W_vocabulary_status_counts=dict(collections.Counter(r['W_vocabulary']['status'] for r in rows)),
        interpretation='Signed residual association; service sums are not a causal decomposition of queue delay')


def grouped(rows):
    result={'all':summarize(rows)}
    definitions={
        'by_alias_task_input':('alias','task','pool_id'),
        'by_alias_task_arrival_third':('alias','task','arrival_third'),
        'by_alias_task_input_arrival_third':('alias','task','pool_id','arrival_third'),
        'by_task':('task',),
    }
    for name,keys in definitions.items():
        groups=collections.defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        result[name]=[{**dict(zip(keys,key)),**summarize(items,input_group='pool_id' in keys)} for key,items in sorted(groups.items())]
    return result


def vocabulary_audit(provider,aliases,row):
    """Keep retrospective phase labels separate from public-input forecast."""
    phases=[s['phase'] for s in row['segments']]
    try:
        plan=provider.plan({k:row[k] for k in ('alias','task','pool_id')},aliases,'mean')
        predicted=[s['id'] for s in plan['segments']]
    except (UnknownEvidence,ValueError,KeyError) as exc:
        return dict(status='UNKNOWN_W_INPUT_PLAN',observed_phases=phases,provider_phases=None,reason=str(exc),
            reconstructed_W=None,fresh_forecast_implication=None)
    matched=phases==predicted
    return dict(status='VOCABULARY_MATCH_W_NOT_COMPUTED' if matched else 'UNKNOWN_W_VOCABULARY',
        observed_phases=phases,provider_phases=predicted,reconstructed_W=None,
        fresh_forecast_implication=None,scope=POLICY['W'])


def join_rows(plan,candidate,prediction,raw,mapped,provider):
    offered=plan['trace'];actual={r['request_id']:r for r in raw['request_records']}
    forecast={r['request_id']:r for r in prediction['requests']}
    observed={r.get('native_request_id',r['request_id']):r for r in mapped['traces'][0]['requests']}
    ids={r['request_id'] for r in offered}
    if len(ids)!=len(offered) or any(set(x)!=ids for x in (actual,forecast,observed)):
        raise ValueError('Every offered request must remain in actual, mapped and forecast residual populations')
    rows=[];origin=raw['phase_start_ns'];horizon=round(plan['window_seconds']*1e9)
    for public in offered:
        rid=public['request_id'];a=actual[rid];p=forecast[rid];o=observed[rid]
        alias=candidate['routes'][rid];scheduled=origin+public['offset_ns']
        start=a['observed_worker_start_time_ns'];end=a['worker_response']['end_perf_ns'];parent=a['completion_time_ns']
        if (a['status']!='completed' or a['model']!=alias or a['task']!=public['task'] or a['pool_id']!=public['pool_id'] or
                a['scheduled_time_ns']!=scheduled or not scheduled<=start<end<=parent<=raw['observation_end_ns']):
            raise ValueError('Actual complete native timing/input/route identity mismatch')
        submit=a.get('actual_submit_time_ns')
        if submit is not None and (type(submit) is not int or not scheduled<=submit<=start):
            raise ValueError('Actual host submit must be between public scheduled arrival and worker START')
        rows.append(dict(request_id=rid,alias=alias,family=plan['aliases'][alias]['family'],task=public['task'],pool_id=public['pool_id'],
            arrival_third=min(2,3*public['offset_ns']//horizon),offset_ns=public['offset_ns'],
            predicted_service_ms=p['service_ms'],actual_service_ms=(end-start)/1e6,
            predicted_wait_ms=p['wait_ms'],actual_wait_ms=(start-scheduled)/1e6,
            predicted_e2e_ms=p['e2e_ms'],actual_e2e_ms=(parent-scheduled)/1e6,
            parent_tail_ms=(parent-end)/1e6,submit_to_START_ms=(start-submit)/1e6 if submit is not None else None,
            scheduled_to_submit_ms=(submit-scheduled)/1e6 if submit is not None else None,
            fresh_forecast_status='KNOWN' if finite(p['e2e_ms']) else 'UNKNOWN',
            fresh_forecast_unknown_reason=prediction['unknown_reason'],W_vocabulary=vocabulary_audit(provider,plan['aliases'],o)))
    return rows


def unique_refs(refs):
    result={}
    for ref in refs:
        checked(ref)
        if ref['path'] in result and result[ref['path']]!=ref:raise ValueError('Conflicting source receipts')
        result[ref['path']]=ref
    return list(result.values())


def run(report_path,quality_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if not Path(report_path).is_file():raise ValueError('Complete frozen H600 training CV report required before running')
    report=read(report_path)
    if (report.get('schema')!='gsdeploy.run_aware_H600_training_seed_crossfit_error_report.v1' or
            report.get('independent_validation') is not False or report.get('independent_results_read') is not False or
            report.get('predictive_fitting_performed') is not False or report.get('resource_transfer_status')!='H600_WHOLE_SEED_TRAINING_CV_ONLY'):
        raise ValueError('Frozen H600 whole-seed training CV with matching resources required')
    refs=unique_refs([receipt(report_path),receipt(__file__),*report['source_receipts'],*report['predictor_source_receipts']])
    quality_ref=receipt(quality_path)
    if quality_ref not in report['source_receipts']:raise ValueError('Quality must be exact original report training proxy')
    quality_doc=read(checked(quality_ref));quality=quality_doc.get('loss_by_family_task',quality_doc)
    if quality_doc.get('split')!='training' or quality_doc.get('heldout_observations_used') is not False:
        raise ValueError('Original explicit training-only fixed quality proxy required')
    audit=frozen.audit_manifest(checked(report['manifest']))
    if audit['model_verifications']!=report['model_verifications']:raise ValueError('Twelve-train/four-heldout model verification changed')
    resources,resource_refs=frozen.resource_folds(checked(report['resource_manifest']),audit['contracts'])
    refs.extend(resource_refs);refs.extend(audit['source_receipts'])
    original={r['flow_id']:r for r in report['candidates']}
    if len(original)!=8 or set(original)!={c['flow_id'] for c in audit['cases']}:raise ValueError('Exactly eight H600 heldout cases required')
    from .flow_profiles import map_flow_collection
    from .flow_validation import _provider
    from .training_run_audit import verify_quality_annotation
    from .duration_selection import evaluate
    fresh_rows=[];cases=[]
    for case in audit['cases']:
        plan=read(checked(case['heldout_plan']));candidate=next(c for c in plan['candidates'] if c['id']==case['candidate_id'])
        model=read(checked(case['model']));fold=resources[case['seed']]
        prediction_candidate=copy.deepcopy(candidate)
        for key in ('predicted_startup_joules','predicted_prephase_peak_mib'):
            prediction_candidate[key]=fold['candidate_overrides'][candidate['id']][key]
        # Predict before loading heldout actuals. core.predict accepts only
        # public plan, frozen training models/resources and fixed training Q.
        prediction=frozen.core.predict(plan,prediction_candidate,model,read(checked(fold['resources'])),quality)
        raw,scored_ref=frozen.core.scored_heldout(case['heldout_raw'])
        verify_quality_annotation(raw,plan,read(checked(plan['catalog'])))
        mapped=map_flow_collection(case['heldout_raw']['path'])
        row=frozen.core.compare(plan,candidate,prediction,raw)
        row.update(**case,heldout_scored=scored_ref,independent_validation=False)
        row['resource_prediction_source']=fold['resources']
        row['predicted_candidate_resource_estimates']={k:prediction_candidate[k] for k in ('predicted_startup_joules','predicted_prephase_peak_mib')}
        if row!=original[case['flow_id']]:raise ValueError('Fresh prediction/observed comparison differs from immutable original H600 report case')
        fresh_rows.append(row)
        residuals=join_rows(plan,candidate,prediction,raw,mapped,_provider(model,candidate['execution_context_sha256']))
        if len(residuals)!=1200:raise ValueError('Exactly 1200 offered residual rows per H600 case required')
        cases.append(dict(flow_id=case['flow_id'],seed=case['seed'],candidate_id=candidate['id'],model=case['model'],
            resource=fold['resources'],candidate_resource_overrides=row['predicted_candidate_resource_estimates'],
            heldout_raw=case['heldout_raw'],heldout_scored=scored_ref,original_case_numeric_equality=True,
            rows=residuals,summary=grouped(residuals)))
        refs.extend([scored_ref,*mapped['mapping_report']['sources']])
    requested=sum(r['requested'] for r in fresh_rows);known=sum(r['known'] for r in fresh_rows)
    rankings={str(seed):frozen.core.ranking([r for r in fresh_rows if r['seed']==seed],contract['baseline'],tie_tolerance=contract['tie_tolerance']) for seed,contract in sorted(audit['contracts'].items())}
    assessment=evaluate(dict(candidates=fresh_rows),dict(seeds=sorted(audit['contracts']),routes=sorted(frozen.ROUTES)),
        read(checked(report['threshold_addendum']))['thresholds'],audit['contracts'])
    if (requested!=report['requested'] or known!=report['known'] or rankings!=report['by_seed'] or
            assessment!=report['original_threshold_assessment'] or requested!=9600):
        raise ValueError('Fresh report summary, decision checks or original threshold assessment differs')
    refs=unique_refs([*refs,*frozen.predictor_sources(),receipt(map_flow_collection.__code__.co_filename),receipt(__file__)])
    value=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),policy=POLICY,original_report=receipt(report_path),
        source=receipt(__file__),cases=cases,requested=requested,known=known,
        original_summary_numeric_equality=True,recomputed_original_threshold_assessment=assessment,
        model_verifications=audit['model_verifications'],source_receipts=refs,
        original_failures_preserved=True,independent_validation=False,independent_results_read=False,
        fitting_performed=False,parameter_selection_performed=False,GPU_started=False,final_gain_eligible=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'report.json',value)
    lines=['H600 training-CV residual diagnostics','',
        'All eight fresh predictions, observed comparisons, rankings and original threshold assessments match the immutable report. Each model excludes all four heldout H600 routes and retains 12 continuous training flows. Resource/startup/peak evidence excludes the same seed. All 9600 offered requests remain in residual denominators.',
        '', '| Flow | Offered / known E2E | Service signed bias ms | Complete service error sum ms | E2E signed bias ms |', '|---|---:|---:|---:|---:|']
    for c in cases:
        s=c['summary']['all'];lines.append(f"| {c['flow_id']} | {s['offered_count']}/{s['fresh_forecast_known_count']} | {s['service']['paired_signed_bias_ms']} | {s['service']['full_cohort_sum_predicted_minus_actual_ms']} | {s['e2e']['paired_signed_bias_ms']} |")
    lines+=['','Alias/task/input and public arrival-third groups, full-population P95, sparse input counts (<5), parent tail and host-submit-to-START observations are retained in report.json. Missing predictions remain UNKNOWN and never disappear from denominators. Paired subset quantities are explicitly labeled.',
        '', 'Observed W vocabulary checks are separate from fresh forecast coverage. OCR vocabulary mismatches retain UNKNOWN_W_VOCABULARY; no observed phase or future runtime enters prediction. No observed-context W integral is calculated.',
        '', 'Service error sums and grouped associations do not identify a causal decomposition of queue error. No model, score, gate, selection or independent result was changed.']
    (output/'results.md').write_text('\n'.join(lines)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('report','quality','output'):p.add_argument('--'+key,required=True)
    args=p.parse_args();run(args.report,args.quality,args.output)
