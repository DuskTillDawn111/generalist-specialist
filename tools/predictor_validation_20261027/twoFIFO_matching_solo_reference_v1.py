"""Authenticated matching solo host references; no physical slowdown inference.

Three capture blocks, two same-owner burst requests per exact key. Every
candidate's no-peer property is checked against its whole observed attempt,
including priming and other cases. No validation outcome or fit is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
import json
from pathlib import Path
from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_matching_prospective_bridge_v1 as matching
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard
from .twoFIFO_training_manifest_v1 import require,digest
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked,stamp
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode

SCHEMA='gsdeploy.twoFIFO_matching_exact_input_solo_host_reference.v1'
H_SOLO=dict(id='Hsolo_matching_host_reference_v1',
    statement='The source-authenticated matching10s/60s solo START-to-READY samples are descriptive host occupancy references for the same worker specification and exact prepared input, despite different case horizon, prior history and CPU activity. Their ratio to native90s/120s validation occupancy is conditional and does not isolate causal interference or physical GPU service.',
    empirical_transfer_qualified=False,physical_SM_service_identified=False,
    independent_replication_unit='whole_capture_block',samples_within_block_are_not_independent_replicates=True)


def _pool(plan):
    response.verify(plan['prepared_pool'])
    return {row['pool_id']:row for row in (json.loads(line) for line in Path(plan['prepared_pool']['path']).read_text().splitlines())}


def _worker_identity(context,owner):
    binding=context['worker_bindings'][owner.split('#')[0]]
    return dict(spec_sha256=binding['spec_sha256'],command=binding['command'])


def extract_rows(plans,raws):
    """Pure evidence algebra. The caller must authenticate all native15 first."""
    require(len(plans)==len(raws)==15,'Exact matching15 solo reference population required')
    rows=[];blocks=sorted({p['capture_id'] for p in plans})
    require(len(blocks)==3,'Three whole training capture blocks required')
    for plan,raw in zip(plans,raws):
        require(raw['attempt_id']==plan['attempt_id'] and raw['execution_id']==plan['execution_id'],
            'Actual same-attempt solo source required')
        all_rows=raw['priming_request_records']+raw['request_records'];by_id={r['request_id']:r for r in all_rows}
        require(len(by_id)==len(all_rows),'No history row may be dropped or duplicated')
        intervals=[]
        for row in all_rows:
            start=row.get('observed_worker_start_time_ns');ready=(row.get('previous_handler_next_request_ready') or {}).get('perf_ns')
            require(stamp(start) and stamp(ready) and start<=ready,'Complete START/READY full-attempt source required')
            intervals.append((row['model'],row['request_id'],start,ready))
        pool=_pool(plan);context=checked(response.verify(plan['context']))
        for case in plan['cases']:
            if case.get('mode')!='solo_same_owner_burst':continue
            require(len(case['arrivals'])==2 and len({a['owner'] for a in case['arrivals']})==1,
                'Frozen explicit two-request same-owner solo design required')
            for index,arrival in enumerate(case['arrivals']):
                row=by_id[arrival['request_id']];start=row['observed_worker_start_time_ns'];end=row['previous_handler_next_request_ready']['perf_ns']
                peers=[dict(owner=owner,request_id=rid,start_ns=a,READY_ns=b) for owner,rid,a,b in intervals
                    if owner!=arrival['owner'] and a<end and b>start]
                key=(arrival['owner'],arrival['task'],arrival['pool_id'])
                require((row['model'],row['task'],row['pool_id'])==key,'Exact solo source key required')
                rows.append(dict(owner=key[0],task=key[1],pool_id=key[2],request_id=row['request_id'],
                    capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],
                    case_id=case['case_id'],same_owner_burst_position=index,
                    worker_identity=_worker_identity(context,key[0]),sample_sha256=digest(pool[key[2]]['sample']),
                    host_START_to_READY_ns=end-start,START_ns=start,READY_ns=end,
                    other_owner_overlaps=peers,solo_supported=not peers,
                    full_attempt_history_requests=len(all_rows),priming_excluded_as_reference_labels=True))
    groups=defaultdict(list)
    for row in rows:groups[(row['owner'],row['task'],row['pool_id'])].append(row)
    summaries=[]
    for key,members in sorted(groups.items()):
        supported=[r for r in members if r['solo_supported']];per_block=[]
        for block in blocks:
            planned=[r for r in members if r['capture_id']==block]
            values=[r['host_START_to_READY_ns'] for r in supported if r['capture_id']==block]
            complete_block=len(planned)==len(values)==2 and all(v>0 for v in values)
            per_block.append(dict(capture_id=block,planned_samples=len(planned),required_samples=2,samples=len(values),
                mean_ns=F(sum(values),2) if complete_block else None,
                conditional_supported_sample_mean_ns=F(sum(values),len(values)) if values else None,
                complete_planned_block=complete_block,
                minimum_ns=min(values) if complete_block else None,maximum_ns=max(values) if complete_block else None))
        complete=all(r['mean_ns'] is not None for r in per_block)
        mean=sum((r['mean_ns'] for r in per_block),F(0))/3 if complete else None
        summaries.append(dict(owner=key[0],task=key[1],pool_id=key[2],planned_samples=len(members),
            supported_samples=len(supported),planned_blocks=3,supported_blocks=sum(r['mean_ns'] is not None for r in per_block),
            equally_weighted_block_mean_host_ns=mean,block_summaries=per_block,
            block_mean_range_ns=[min(r['mean_ns'] for r in per_block),max(r['mean_ns'] for r in per_block)] if complete else None,
            uncertainty_scope='DESCRIPTIVE_THREE_BLOCK_RANGE_NOT_POPULATION_CONFIDENCE_INTERVAL',
            samples_within_block_renormalized=False))
    return dict(rows=rows,keys=summaries,whole_blocks=blocks,reference_scope='OBSERVED_HOST_START_TO_READY_WITH_NO_OBSERVED_OTHER_OWNER_HANDLER_OVERLAP',
        physical_SM_service_identified=False,causal_interference_isolated=False,history_or_cache_state_equated=False)


def build(fit_report_path,target_plan_path):
    """Reauthenticate exact saved fit/native15; caller supplies canonical target."""
    fit_ref=receipt(fit_report_path);fit=response.audit_fit(fit_ref)['report'];target_ref=receipt(target_plan_path)
    from . import twoFIFO_fresh181_native_v4 as target_executor
    target=target_executor.audit_plan(target_plan_path)
    require(checked(target_ref)==target,'Canonical native4 target plan required for solo binding')
    watched=strict_references([fit_ref,fit,target_ref,target,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        target_pool=_pool(target);target_context=checked(response.verify(target['context']))
        def extract(bundle,intake):
            require(intake==fit['intake_evidence'],'Solo references must come from the exact saved-fit source15')
            result=extract_rows(bundle['plans'],matching._raws(bundle));binding=[]
            summaries={(item['owner'],item['task'],item['pool_id']):item for item in result['keys']}
            for key in matching.required_keys(checked(response.verify(target['public_contract']))):
                item=summaries.get(key)
                members=[r for r in result['rows'] if (r['owner'],r['task'],r['pool_id'])==key]
                compatible=bool(members) and all(r['sample_sha256']==digest(target_pool[key[2]]['sample'])
                    and r['worker_identity']==_worker_identity(target_context,key[0]) for r in members)
                binding.append(dict(owner=key[0],task=key[1],pool_id=key[2],exact_input_worker_match=compatible,
                    three_block_reference_available=compatible and item is not None and item['supported_blocks']==3
                        and item['equally_weighted_block_mean_host_ns']>0))
            result.update(target_bindings=binding,sources=strict_references([watched,bundle['review_receipts'],bundle['documents']]))
            return result
        value=response.with_verified_bundle(fit['manifest']['path'],fit['continuation']['path'],extract)
        agenda=[dict(owner=r['owner'],task=r['task'],pool_id=r['pool_id'],reason='NO_MATCHED_THREE_BLOCK_OBSERVED_NOPEER_REFERENCE',
            collection='NEW_DISTINCT_PROSPECTIVE_SOLO_REFERENCE_NOT_TARGET_COHORT_RETRY',
            predeclare_before_execution=dict(resident_workers=4,priming_requests=18,epoch_anchor='snapshot_time_ns_plus300s',
                READY_settle_timeout_seconds=10,measurement_seconds=90,drain_seconds=120,
                target_key_only=True,global_admission_cap=None,repetitions_in_independent_blocks=3),
            GPU_started=False) for r in value['target_bindings'] if not r['three_block_reference_available']]
        guard.check();return encode(dict(schema=SCHEMA,fit=fit_ref,target_plan=target_ref,hypothesis=H_SOLO,
            **value,missing_reference_collection_agenda=agenda,validation_labels_read=False,used_for_fit=False,
            source_evidence_authenticated=True,empirical_transfer_qualified=False,final_gain_eligible=False,GPU_started=False))


def slowdown_labels(observed,reference,forecast_errors=None):
    reference=decode(reference);require(reference['schema']==SCHEMA and reference['source_evidence_authenticated'] is True,
        'Rebuilt authenticated exact matching solo reference required')
    summaries={(r['owner'],r['task'],r['pool_id']):r for r in reference['keys']}
    supported={(r['owner'],r['task'],r['pool_id']) for r in reference['target_bindings'] if r['three_block_reference_available']}
    predictions={} if forecast_errors is None else {r['request_id']:r for r in decode(forecast_errors)['requests']}
    result=[]
    for row in observed['requests']:
        key=(row['owner'],row['task'],row['pool_id']);item=summaries.get(key);denom=None if key not in supported else item['equally_weighted_block_mean_host_ns']
        value=row['actual_START_to_READY_ns'];ratio=F(value,1)/denom if value is not None and denom is not None and denom>0 else None
        predicted=predictions.get(row['request_id'],{}).get('predicted_mean_START_to_READY_ns')
        predicted_ratio=predicted/denom if predicted is not None and denom is not None and denom>0 else None
        span=None if item is None else item['block_mean_range_ns']
        result.append(dict(request_id=row['request_id'],conditional_host_solo_reference_ratio=ratio,
            predicted_conditional_host_solo_reference_ratio=predicted_ratio,
            conditional_ratio_error=None if predicted_ratio is None or ratio is None else predicted_ratio-ratio,
            descriptive_ratio_range=None if ratio is None or span[0]<=0 else [F(value,1)/span[1],F(value,1)/span[0]],
            reference_block_mean_host_ns=denom,reference_block_mean_range_ns=span,
            reference_supported_blocks=0 if item is None else item['supported_blocks'],
            unknown_reason=None if ratio is not None else 'MISSING_EXACT_NOPEER_REFERENCE_OR_OBSERVED_READY',
            physical_slowdown_qualified=False,causal_interference_isolated=False))
    return encode(dict(schema=SCHEMA+'.conditional_host_ratios',requests=result,hypothesis=H_SOLO,
        offered_requests=len(result),missing_reference_collection_agenda=reference['missing_reference_collection_agenda'],
        validation_labels_read=True,used_for_fit=False,empirical_transfer_qualified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit-report','target-plan','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();print(json.dumps(matching.prefix.save(args.output,build(args.fit_report,args.target_plan))))
