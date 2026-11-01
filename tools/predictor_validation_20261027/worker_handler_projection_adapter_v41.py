"""Explicit uncalibrated fixed-median worker-handler training projection.

Only original projected worker begin/end events define the handler mask.
Native GS_STAGE and GPU readiness do not repair or override that projection.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left,bisect_right
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

from .implementation_source_closure_v27 import implementation_sources,strict_references
from .owner_lifecycle_mark_v21 import IntervalIndex,receipt
from .unknown_preflight_archive_v1 import save_synced

SCHEMA='gsdeploy.whole57_fixed_median_worker_handler_context_risk.v41'
OWNERS=('generalist_2#0','ocr#0','text#0','text#1')
ASSUMPTION='UNCALIBRATED_FIXED_MEDIAN_WORKER_BOUNDARY_PROJECTION'


def checked(ref):
    if set(ref)!={'path','sha256'} or receipt(ref['path'])!=ref:raise ValueError('Exact source reference required')
    return json.loads(Path(ref['path']).read_text())


def features(held_owner,previous_held_owner,context_age_ns,handler_active):
    if (held_owner not in OWNERS or previous_held_owner is not None and previous_held_owner not in OWNERS
            or type(context_age_ns) is not int or context_age_ns<0 or set(handler_active)!=set(OWNERS)
            or any(type(v) is not bool for v in handler_active.values())):
        raise ValueError('Explicit complete projected worker-handler boolean mask required')
    return dict(held_owner=held_owner,previous_held_owner=previous_held_owner,
        context_age_ns=context_age_ns,handler_active=dict(handler_active))


def projection(indexes,clock):
    return {owner:indexes[owner].contains(clock) for owner in OWNERS}


def proximity_diagnostic(indexes,residuals,clock):
    result={}
    for owner in OWNERS:
        boundaries=sorted(t for span in indexes[owner].spans for t in span)
        index=bisect_left(boundaries,clock)
        nearby=boundaries[max(0,index-1):index+1]
        result[owner]=None if any(abs(clock-t)<=residuals[owner] for t in nearby) else indexes[owner].contains(clock)
    return result


def split_risk(row,indexes,cuts,residuals):
    lo,hi=row['start_ns'],row['end_ns']
    if not lo<hi:raise ValueError('Positive original physical exposure required')
    boundaries=[lo,*cuts[bisect_right(cuts,lo):bisect_left(cuts,hi)],hi]
    for left,right in zip(boundaries,boundaries[1:]):
        value=copy.deepcopy(row);old=row['shared_left_features']
        observed=row['physical_SAVE_END_observed'] and right==hi
        value.update(start_ns=left,end_ns=right,age_entry_ns=left-row['restore_start_ns'],
            age_exit_ns=right-row['restore_start_ns'],physical_SAVE_END_observed=observed,
            physical_SAVE_seqNo=row['physical_SAVE_seqNo'] if observed else None,
            shared_left_features=features(row['held_owner'],old['previous_held_owner'],left-row['restore_start_ns'],projection(indexes,left)),
            handler_projection_assumption=ASSUMPTION,
            handler_boundary_proximity_diagnostic=proximity_diagnostic(indexes,residuals,left),
            handler_projection_is_exact_native_fact=False,
            right_boundary_is_feature=False)
        yield value


def stream(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Original compressed training stream changed')
    with gzip.open(ref['path'],'rt',encoding='utf-8') as handle:
        for line in handle:yield json.loads(line)
    if receipt(ref['path'])!=ref:raise ValueError('Original compressed training stream changed during read')


def extract(shared_reference,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    source=checked(shared_reference);native=checked(source['source_adapter']);review=checked(native['review'])
    if (source.get('schema')!='gsdeploy.whole57_shared_observable_native_context_risk.v31'
            or source.get('status')!='SHARED_OBSERVABLE_RISK_AND_MARK_EXTRACTED_NO_FIT'
            or source.get('training_only') is not True or source.get('validation_labels_used') is not False
            or source.get('fit_performed') is not False):raise ValueError('Completed no-fit original shared capture required')
    indexes_doc=checked(native['lifecycle_indexes']);raw=checked(review['evidence']['raw.json'])
    indexes={o:IntervalIndex(indexes_doc['certain'][o]) for o in OWNERS}
    mappings=native['original_handler_projection_by_pid'];residuals={}
    for owner in OWNERS:
        value=mappings[str(raw['ready'][owner]['pid'])]
        if (value['method']!='fixed_median_worker_boundary_projection_not_calibrated_common_clock'
                or type(value['max_boundary_residual_ns']) is not int or value['max_boundary_residual_ns']<0):
            raise ValueError('Original fixed-median mapping and descriptive residual required')
        residuals[owner]=value['max_boundary_residual_ns']
    # Independently reconstruct projected handler intervals from original raw
    # worker boundaries. The misleading historical key "certain" is not a claim.
    reconstructed={o:[] for o in OWNERS}
    for row in raw['priming_request_records']+raw['request_records']:
        owner=row['model'];offset=mappings[str(raw['ready'][owner]['pid'])]['offset_ns'];worker=row['worker_response']
        reconstructed[owner].append((worker['start_perf_ns']-offset,worker['end_perf_ns']-offset))
    if any(IntervalIndex(reconstructed[o]).spans!=indexes[o].spans for o in OWNERS):
        raise ValueError('Historical handler index differs from original projected worker begin/end')
    cuts=sorted({t for index in indexes.values() for span in index.spans for t in span})
    implementation=implementation_sources(__file__)
    refs=strict_references([shared_reference,source['source_adapter'],source['sources'],source['implementations'],
        native['lifecycle_indexes'],native['native_transition_ledger'],source['risk_chronology'],source['mark_records'],
        source['physical_SAVE_event_ledger'],source['native_context_gaps'],review['evidence']['raw.json'],implementation])
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Projected-handler dependency changed')
    out.mkdir(parents=True,exist_ok=False);counts=Counter();mark_counts=Counter();mask_counts=Counter();n=0
    risk_path=out/'risk-chronology.jsonl.gz';mark_path=out/'worker-handler-mark-records.jsonl.gz'
    with gzip.open(risk_path,'wt',encoding='utf-8') as handle:
        for row in stream(source['risk_chronology']):
            for value in split_risk(row,indexes,cuts,residuals):
                handle.write(json.dumps(value,separators=(',',':'))+'\n');width=value['end_ns']-value['start_ns']
                counts['all_paired_intervals']+=1;counts['all_paired_exposure_ns']+=width
                counts['paired_SAVE_events']+=value['physical_SAVE_END_observed']
                if value['exposure_qualified']:
                    counts['qualified_risk_intervals']+=1;counts['qualified_exposure_ns']+=width
                    counts['qualified_SAVE_events']+=value['physical_SAVE_END_observed']
                else:counts['unresolved_exposure_ns']+=width
                mask_counts[''.join('1' if value['shared_left_features']['handler_active'][o] else '0' for o in OWNERS)]+=1
    with gzip.open(mark_path,'wt',encoding='utf-8') as handle:
        for row in stream(source['mark_records']):
            old=row['features'];clock=row['SAVE_END_ns']
            row['features']=features(old['held_owner'],old['previous_held_owner'],old['context_age_ns'],projection(indexes,clock))
            row.update(handler_projection_assumption=ASSUMPTION,
                handler_boundary_proximity_diagnostic=proximity_diagnostic(indexes,residuals,clock),
                handler_projection_is_exact_native_fact=False)
            handle.write(json.dumps(row,separators=(',',':'))+'\n');n+=1;mark_counts[row['next_owner']]+=1
    for key in ['all_paired_exposure_ns','paired_SAVE_events','qualified_exposure_ns','qualified_SAVE_events']:
        if counts[key]!=source['counts'].get(key,0):raise ValueError('Handler splitting changed original exposure or physical SAVE population')
    if n!=source['mark_record_count'] or dict(mark_counts)!=source['mark_target_counts']:
        raise ValueError('Projected handler features changed original mark labels')
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Projected-handler dependency changed during extraction')
    result=copy.deepcopy(source)
    result.update(schema=SCHEMA,adapter_version=41,status='PROJECTED_WORKER_HANDLER_RISK_AND_MARK_EXTRACTED_NO_FIT',
        source_shared_adapter=shared_reference,sources=refs,implementations=strict_references([source['implementations'],implementation]),
        counts=dict(counts),risk_chronology=receipt(risk_path),mark_records=receipt(mark_path),
        projection_assumption=ASSUMPTION,original_handler_projection_by_pid=mappings,
        worker_handler_observed_residual_by_owner_ns=residuals,worker_handler_mask_counts=dict(mask_counts),
        handler_projection='Fixed original worker begin/end minus frozen per-PID median offset; half-open intervals; no native stage OR.',
        handler_model_use='Explicit uncalibrated training-to-engine worker-handler projection hypothesis',
        no_exact_clock_claim=True,calibrated_cross_clock_error_bound_ns=None,
        native_stage_activity_used_as_handler=False,projection_residual_is_certified_error_bound=False,
        handler_boundary_proximity_diagnostic='Three-valued sensitivity diagnostic only; observed residual is not a timing error bound; never a feature or interval cut.',
        future_END_residual_boundary_used=False,future_END_used_before_fixed_projected_END=False)
    result.pop('matched_engine_projection',None)
    save_synced(out/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shared-report',required=True)
    p.add_argument('--shared-report-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=extract(dict(path=str(Path(args.shared_report).absolute()),sha256=args.shared_report_sha256),args.output)
    print(json.dumps(dict(status=result['status'],capture=result['capture'],counts=result['counts'])))
