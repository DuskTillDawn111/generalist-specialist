"""All-public-case structural rank witness, never a fitted queue/M prediction.

First observed source profile per public key is transplanted without fitting.
The witness establishes no forced public-design column degeneracy; numerical
rank of the eventual donor-generated full/fold matrices remains unobserved.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path

from . import twoFIFO_case_memory_feature_audit_v1 as features
from . import twoFIFO_training_intake_v4 as intake
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import implementation_sources,strict_references


def profiles_from_feature_only_raw(raw):
    profiles={}
    for row in raw['priming_request_records']+raw['request_records']:
        key=(row['model'],row['task'],row['pool_id'])
        if key in profiles:continue
        events=[copy.deepcopy(e) for e in row['execution_events'] if
            (e.get('event')=='start' and (e.get('segment_id') in ('native_inference','native_ocr') or e.get('semantic_phase') in ('prefill','decode')))
            or e.get('event')=='end' and e.get('segment_id') in ('native_inference','native_ocr')]
        if not events:continue
        origin=min(e['collector_processing_time_ns'] for e in events)
        profiles[key]=dict(events=[dict(e,offset_ns=e['collector_processing_time_ns']-origin) for e in events],
            duration_ns=max(e['collector_processing_time_ns']-origin for e in events)+1,
            selection='first_source_priming_then_measured_row_for_exact_public_key_no_target_used')
    return profiles


def case_witness(case,profiles):
    available=defaultdict(int);rows=[]
    for index,arrival in sorted(enumerate(case['arrivals']),key=lambda x:(x[1]['arrival_ns'],x[0])):
        key=(arrival['owner'],arrival['task'],arrival['pool_id'])
        if key not in profiles:raise ValueError('UNKNOWN_SOURCE_PUBLIC_KEY_PROFILE:'+repr(key))
        profile=profiles[key];begin=max(arrival['arrival_ns'],available[arrival['owner']]);events=[]
        for template in profile['events']:
            at=begin+template['offset_ns'];event=copy.deepcopy(template);event.pop('offset_ns')
            event.update(request_id=arrival['request_id'],perf_ns=at,received_time_ns=at,
                collector_dequeued_time_ns=at,collector_processing_time_ns=at);events.append(event)
        available[arrival['owner']]=begin+profile['duration_ns']
        rows.append(dict(request_id=arrival['request_id'],model=arrival['owner'],task=arrival['task'],
            pool_id=arrival['pool_id'],profile_job_id=case['case_id'],execution_events=events))
    raw=dict(priming_request_records=[],request_records=rows,measurement_windows=[dict(case_id=case['case_id'],
        start_ns=0,end_ns=10_000_000_000,fixed_drain_deadline_ns=70_000_000_000)])
    return features.feature_rows(raw)['rows'][0]


def run(campaign_path,source_raw_path,output):
    campaign=intake.campaign_v2.audit(campaign_path);plans=[checked(r) for r in campaign['attempts']]
    source_ref=receipt(source_raw_path);source_raw=checked(source_ref)
    profiles=profiles_from_feature_only_raw(source_raw);rows=[]
    for plan in plans:
        for case in plan['cases']:
            row=case_witness(case,profiles);row.update(capture_id=plan['capture_id'],attempt_id=plan['attempt_id']);rows.append(row)
    assert len(rows)==150
    blocks=sorted({r['capture_id'] for r in rows});full=features.matrix_audit(rows,'case_demand_C_R');folds=[]
    for block in blocks:
        donor=[r for r in rows if r['capture_id']!=block];heldout=[r for r in rows if r['capture_id']==block]
        folds.append(dict(heldout_block_id=block,training_case_rows=len(donor),heldout_case_rows=len(heldout),
            training_matrix=features.matrix_audit(donor,'case_demand_C_R'),
            heldout_matrix=features.matrix_audit(heldout,'case_demand_C_R')))
    result=dict(schema='gsdeploy.twoFIFO_case_memory_public_structural_rank_witness.v1',
        status='FEATURE_ONLY_TRANSPLANT_WITNESS_NOT_GENERATED_FORECAST_QUALIFICATION',
        campaign=receipt(campaign_path),source_profile_raw=source_ref,all_public_case_rows=150,
        all_public_attempts=12,full_matrix=full,whole_block_leaveout=folds,rows=rows,
        profiles=[dict(public_key=list(k),**v) for k,v in sorted(profiles.items())],
        profile_transplant_uses_public_arrivals_and_per_owner_FIFO_only=True,
        source_clock_and_shape_transfer_qualified=False,crossvalidation_donor_independence_claimed=False,
        copied_source_profile_is_not_eventual_fold_generated_trajectory=True,
        actual_generated_full_and_fold_numeric_rank='UNKNOWN_UNTIL_COMPLETE_DONOR_ONLY_GENERATION',
        witness_proves_actual_full_or_fold_identifiability=False,
        source_memory_or_power_target_accessed=False,board_samples_accessed=False,
        native_queue_or_memory_fit_run=False,validation_labels_read=False,GPU_started=False,
        sources=strict_references([receipt(campaign_path),campaign['attempts'],implementation_sources(__file__)]))
    out=Path(output);out.mkdir(parents=True,exist_ok=False);path=out/'report.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return receipt(path)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--source-raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.source_raw,args.output)))
