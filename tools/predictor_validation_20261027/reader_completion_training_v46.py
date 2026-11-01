"""Fit an observable completion candidate on four original natural captures.

This is a known-input-family diagnostic, never a replacement physical service
model or an independent gate. Priming is retained as history, excluded from fit.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import shutil

from . import reader_completion_hazard_v46 as hazard
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import context_identity, receipt
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced


def read_stream(reference):
    if receipt(reference['path'])!=reference:raise ValueError('Immutable compressed training stream required')
    with gzip.open(reference['path'],'rt',encoding='utf-8') as stream:
        result=[json.loads(line) for line in stream]
    if receipt(reference['path'])!=reference:raise ValueError('Training stream changed during read')
    return result


def raw_interval(row):
    starts=[e for e in row['execution_events'] if 'event' not in e]
    if len(starts)!=1:raise ValueError('One original actually received START required')
    lo=starts[0]['received_time_ns'];hi=row['response_reader_event_enqueue_time_ns']
    if type(lo) is not int or type(hi) is not int or not 0<=lo<hi or row['completion_time_ns']!=hi:
        raise ValueError('Positive original reader interval required')
    return lo,hi


def verify_rows(report,raw,risks,labels):
    if len(raw['request_records'])!=57 or len(raw['priming_request_records'])!=12:
        raise ValueError('Separate original measured57 and priming12 populations required')
    original=[('priming',r) for r in raw['priming_request_records']]+[('measured',r) for r in raw['request_records']]
    if len(original)!=69 or len({r['request_id'] for _,r in original})!=69:
        raise ValueError('Exact original 57 measurement plus 12 priming identities required')
    spans={o:sorted(raw_interval(r) for _,r in original if r['model']==o) for o in hazard.OWNERS}
    starts={o:[a for a,b in values] for o,values in spans.items()}
    if any(a[1]>b[0] for values in spans.values() for a,b in zip(values,values[1:])):
        raise ValueError('Reader-visible serial owner intervals required')
    def mask(clock):
        result={}
        for o in hazard.OWNERS:
            i=bisect_right(starts[o],clock)-1
            result[o]=i>=0 and spans[o][i][0]<=clock<spans[o][i][1]
        return result
    cuts=sorted({t for values in spans.values() for pair in values for t in pair})
    expected=[];expected_labels={}
    for cohort,row in original:
        if row['status']!='completed':raise ValueError('Full original completed cohort required')
        lo,hi=raw_interval(row);boundaries=[lo,*cuts[bisect_right(cuts,lo):bisect_left(cuts,hi)],hi]
        for a,b in zip(boundaries,boundaries[1:]):
            expected.append(dict(capture=report['capture'],request_id=row['request_id'],cohort=cohort,
                origin_reader_START_ns=lo,start_ns=a,end_ns=b,age_entry_ns=a-lo,age_exit_ns=b-lo,
                completion_observed=b==hi,features=dict(alias=row['model'],task=row['task'],pool_id=row['pool_id'],
                    attained_reader_handler_age_ns=a-lo,active_handler_by_owner=mask(a))))
        expected_labels[row['request_id']]=(cohort,row,lo,hi)
    if risks!=expected:raise ValueError('Every exact risk row/mask/cut/event must reconstruct from original raw reader history')
    groups=hazard.episodes(risks)
    if len(groups)!=69:raise ValueError('All 69 complete reader episodes required')
    counts=Counter()
    for r in risks:
        counts[r['cohort']+'_risk_intervals']+=1;counts[r['cohort']+'_exposure_ns']+=r['end_ns']-r['start_ns']
        counts[r['cohort']+'_completion_events']+=r['completion_observed']
    counts.update({cohort+'_requests':sum(c==cohort for c,r in original) for cohort in ('measured','priming')})
    if dict(counts)!=report['counts']:raise ValueError('Full separate measured/priming counts must reconcile exactly')
    if len(labels)!=69 or len({r['request_id'] for r in labels})!=69 or {r['request_id'] for r in labels}!=set(expected_labels):
        raise ValueError('All original request boundary labels required')
    for label in labels:
        cohort,row,lo,hi=expected_labels[label['request_id']]
        expected_scalars=dict(scheduled_time_ns=row['scheduled_time_ns'],enqueued_time_ns=row['enqueued_time_ns'],
            dispatch_begin_ns=row['actual_submit_time_ns'],dispatch_write_completed_ns=row['submit_write_completed_time_ns'],
            worker_handler_start_perf_ns=row['worker_response']['start_perf_ns'],
            worker_handler_end_perf_ns=row['worker_response']['end_perf_ns'],
            worker_next_ready_perf_ns=row['previous_handler_next_request_ready']['perf_ns'],
            worker_next_ready_reader_received_ns=row['previous_handler_next_request_ready']['received_time_ns'])
        if (label['capture']!=report['capture'] or label['cohort']!=cohort
                or (label['alias'],label['task'],label['pool_id'])!=(row['model'],row['task'],row['pool_id'])
                or label['origin_reader_START_ns']!=lo or label['RESULT_reader_received_ns']!=hi
                or label['worker_response_labels']!=row['worker_response']
                or label['worker_next_ready_record']!=row['previous_handler_next_request_ready']
                or any(label.get(k)!=v for k,v in expected_scalars.items())):
            raise ValueError('Independent worker/READY labels and reader targets cannot be rewritten')
    return dict(counts)


def load_dataset(manifest_reference):
    manifest=hazard.checked(manifest_reference)
    if (set(manifest)!={'training_only','validation_labels_used','captures'}
            or manifest['training_only'] is not True or manifest['validation_labels_used'] is not False
            or len(manifest['captures'])!=4 or {r['route'] for r in manifest['captures']}!=set(hazard.ROUTES)):
        raise ValueError('Explicit four distinct actual training routes required')
    rows=[];sources=[manifest_reference];captures={};projection=None;ready_projection=None;execution_recipe=None
    for member in manifest['captures']:
        if set(member)!={'route','report'}:raise ValueError('Explicit actual route/report required')
        report=hazard.checked(member['report'])
        if (report.get('schema')!=hazard.DATA_SCHEMA
                or report.get('status')!='NATURAL_READER_HANDLER_COMPLETION_RISK_EXTRACTED_NO_FIT'
                or report.get('training_only') is not True or report.get('validation_labels_used') is not False
                or report.get('fit_performed') is not False or report.get('route')!=member['route']):
            raise ValueError('Frozen training-only no-fit natural reader adapter required')
        review=hazard.checked(report['source_review'])
        if (review.get('schema')!='gsdeploy.whole57_joint_training_review.v23'
                or review.get('status')!='PASS' or review.get('training_only') is not True
                or review.get('model_fitted') is not False or review.get('validation_started') is not False):
            raise ValueError('Actual training review role required before opening any raw outcome')
        raw=hazard.checked(report['source_raw'])
        plan=hazard.checked(report['source_plan']);context=hazard.checked(report['source_context'])
        if (review.get('schema')!='gsdeploy.whole57_joint_training_review.v23'
                or review.get('status')!='PASS' or review.get('issues')!=[] or review.get('training_only') is not True
                or review.get('requests')!=57 or review.get('trace') is not None
                or raw.get('training_only') is not True or plan.get('training_only') is not True
                or context.get('training_only') is not True
                or raw.get('schema')!='gsdeploy.whole57_joint_host_lifecycle_training_raw.v23'
                or plan.get('schema')!='gsdeploy.whole57_joint_host_lifecycle_training_plan.v23'
                or raw.get('status')!='completed' or raw.get('errors') or raw.get('runtime_errors')
                or raw.get('telemetry_errors') or raw.get('final_source_audit')!='PASS'
                or plan.get('split')!='training' or raw.get('split')!='training'
                or raw['domain']!='whole57-joint-natural-v23' or plan['domain']!=raw['domain']
                or context['measurement_domain']!=raw['domain'] or raw['route_id']!=member['route']
                or plan['route_id']!=member['route'] or context['whole57_training_route_id']!=member['route']
                or raw['plan']!=report['source_plan'] or raw['context']!=report['source_context']
                or raw['execution_context_sha256']!=context_identity(context)
                or raw['execution_context_sha256']!=plan['execution_context_sha256']
                or report['execution_context_sha256']!=raw['execution_context_sha256']
                or review['evidence']['raw.json']!=report['source_raw']
                or review['evidence']['plan.json']!=report['source_plan'] or review['evidence']['context.json']!=report['source_context']
                or report['capture']!=report['source_raw']['sha256'] or report['capture'] in captures
                or raw['planned_public_cohort']!=plan['trace']
                or [r['request_id'] for r in raw['request_records']]!=[r['request_id'] for r in plan['trace']]):
            raise ValueError('Full raw/context/plan/review/domain/cohort identity must agree')
        for row,public in zip(raw['request_records'],plan['trace']):
            if ((row['model'],row['task'],row['pool_id'],row['scheduled_time_ns']-raw['phase_start_ns'])
                    !=(plan['routes'][public['request_id']],public['task'],public['pool_id'],public['offset_ns'])):
                raise ValueError('Offered arrivals/input/routes cannot be changed')
        cp={k:v for k,v in context.items() if k not in ('execution_nonce','whole57_training_route_id')}
        rp={a:{k:v for k,v in worker.items() if k!='pid'} for a,worker in raw['ready'].items()}
        recipe={k:plan[k] for k in ('recipe','resource_configuration','inputs','weights')}
        if projection is None:projection=cp;ready_projection=rp;execution_recipe=recipe
        if cp!=projection or rp!=ready_projection or recipe!=execution_recipe:
            raise ValueError('Exact original natural execution context/actual READY/input/recipe projection differs')
        actual_ready=hazard.checked(report['actual_ready'])
        if (actual_ready['ready']!=raw['ready'] or actual_ready['source_raw']!=report['source_raw']
                or actual_ready.get('capture')!=report['capture'] or set(raw['ready'])!=set(hazard.OWNERS)
                or any(type(w.get('pid')) is not int or w['pid']<=0 or w.get('clock')!='time.perf_counter_ns'
                    or w.get('worker')!=a.split('#')[0] or not isinstance(w.get('tasks'),list)
                    or not w['tasks'] or not isinstance(w.get('spec_sha256'),str) or len(w['spec_sha256'])!=64
                    for a,w in raw['ready'].items())):
            raise ValueError('Actual original READY evidence mismatch')
        refs=strict_references([member['report'],report['sources'],report['implementations'],
            report['risk_chronology'],report['request_boundaries'],report['received_events'],report['actual_ready']])
        for ref in refs:
            if receipt(ref['path'])!=ref:raise ValueError('Training source drift')
        risks=read_stream(report['risk_chronology']);labels=read_stream(report['request_boundaries'])
        counts=verify_rows(report,raw,risks,labels)
        captures[report['capture']]=dict(route=member['route'],adapter=member['report'],raw=report['source_raw'],
            original_context=report['source_context'],original_execution_context_sha256=raw['execution_context_sha256'],
            counts=counts,offered_cohort=57,retained_priming_history=12)
        rows.extend(r for r in risks if r['cohort']=='measured');sources.extend(refs)
    return dict(rows=rows,captures=captures,sources=strict_references(sources),
        context_projection_sha256=digest(projection),ready_projection_sha256=digest(ready_projection))


def write_gzip(path,value):
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'));stream.write('\n')


def train(manifest_reference,output):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    data=load_dataset(manifest_reference);implementations=implementation_sources(__file__)
    sources=strict_references([data['sources'],implementations]);out.mkdir(parents=True)
    save_synced(out/'intent.json',dict(schema=hazard.SCHEMA,parameters=hazard.PARAMETERS,sources=sources,
        training_only=True,validation_labels_used=False,GPU_started=False))
    summaries=[];fold_models=[]
    for capture,identity in sorted(data['captures'].items()):
        fit_rows=[r for r in data['rows'] if r['capture']!=capture];test_rows=[r for r in data['rows'] if r['capture']==capture]
        model=hazard.fit(fit_rows,include_peers=True);baseline=hazard.fit(fit_rows,include_peers=False)
        frozen=dict(heldout_capture=capture,heldout_route=identity['route'],
            training_captures=sorted(c for c in data['captures'] if c!=capture),candidate=model,baseline=baseline)
        model_path=out/(identity['route']+'-frozen-fold-models.json');save_synced(model_path,frozen)
        evaluation=hazard.evaluate(model,baseline,test_rows)
        predictions=evaluation.pop('predictions');prediction_path=out/(identity['route']+'-landmark-diagnostic.json.gz')
        write_gzip(prediction_path,predictions)
        summaries.append(dict(heldout_capture=capture,heldout_route=identity['route'],training_captures=frozen['training_captures'],
            frozen_models=receipt(model_path),predictions_and_offline_labels=receipt(prediction_path),**evaluation))
        fold_models.append(receipt(model_path))
        print(json.dumps(dict(event='READER_COMPLETION_FOLD_FINISHED',route=identity['route'],landmarks=evaluation['landmarks'],
            quantiles=evaluation['full_population_quantile_diagnostic'])),flush=True)
    candidate=hazard.fit(data['rows'],include_peers=True);baseline=hazard.fit(data['rows'],include_peers=False)
    model=dict(schema=hazard.SCHEMA,parameters=hazard.PARAMETERS,candidate=candidate,baseline=baseline,
        captures=data['captures'],sources=sources,implementations=implementations,training_only=True,
        validation_labels_used=False,context_projection_sha256=data['context_projection_sha256'],
        ready_projection_sha256=data['ready_projection_sha256'],physical_service_identified=False,
        original_utility_or_stability_changed=False)
    model['model_sha256']=digest(model)
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Frozen source changed during training')
    save_synced(out/'model.json',model)
    project=Path(__file__).resolve().parents[2];snapshots=[]
    for ref in implementations:
        source=Path(ref['path']);target=out/'source-snapshot'/source.relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if receipt(target)['sha256']!=ref['sha256']:raise ValueError('Exact source snapshot required')
        snapshots.append(dict(original=ref,snapshot=receipt(target)))
    report=dict(schema=hazard.SCHEMA,status='FOUR_CAPTURE_OBSERVABLE_COMPLETION_TRAINING_DIAGNOSTIC_NOT_QUALIFIED',
        model=receipt(out/'model.json'),parameters=hazard.PARAMETERS,whole_capture_holdout=summaries,
        original_capture_accounting=data['captures'],complete_frozen_fold_models=fold_models,
        source_snapshots=snapshots,training_only=True,validation_labels_used=False,GPU_started=False,
        final_training_predictions_evaluated=False,holdout_is_content_or_recipe_independent=False,
        physical_SM_service_or_counterfactual_slowdown_identified=False,
        full_cohort_queue_forecast_executed=False,rolling_full_forecast_implemented=False,
        original_six_terms_produced=False,independent_prediction_accuracy_validated=False,
        sameholder_rejection_removed=False,global_request_cap_added=False,
        limitations=['Only four distinct captures share the exact input/recipe family.',
            'Reader START/RESULT/READY and physical worker boundaries remain separate.',
            'Future reader peer state is held fixed for these conditional landmark candidates.',
            'Dense landmarks are correlated; no independent-window confidence interval is claimed.',
            'Full-population metrics retain flagged extrapolation; common-support denominators are also explicit.',
            'This candidate does not replace the original service, utility, resource or stability gate.'])
    save_synced(out/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-manifest',required=True)
    parser.add_argument('--training-manifest-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=train(dict(path=str(Path(args.training_manifest).absolute()),sha256=args.training_manifest_sha256),args.output)
    print(json.dumps(dict(status=result['status'],model=result['model'])),flush=True)
