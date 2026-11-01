"""Audit causal native worker boundary visibility; never fabricate device progress.

CPU-only adapter over original training captures. Host PHASE/end observations
are useful facts but do not measure a current fractional CUDA work cursor.
This module does not implement main lifecycle continuation or start workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .rolling_visible_queue_v1 import advance_runtime_records, new_session, require, validate

SCHEMA='gsdeploy.rolling_native_visible_evidence_audit.v2'
ORIGINAL_MANIFEST_SHA256='4cbb2725b4ede8e8fc948844feb83daa38b97cd0e8cac81aa16eda93258735b4'
OFFSETS_NS=tuple(t*1_000_000_000 for t in (1,5,10,15,20,25,30))
DEVICE_EVIDENCE_CONTRACT=dict(
    status='REQUIRED_NOT_IMPLEMENTED_BY_CURRENT_NATIVE_WORKER',
    identity=['execution_context_sha256','worker_pid','request_id','alias',
              'frozen_profile_sha256','profile_node_id','stream_id','event_id'],
    progress='Completed device event established by nonblocking device-event query; host forward return is insufficient',
    timestamp=['host_query_begin_ns','host_query_end_ns','parent_reader_receipt_ns','persistent_per_stream_sequence'],
    mapping='Exact node/stream mapping declared before the request; no mapping selected from future outputs',
    observation_delay='Preserve worker-query bracket and later parent receipt separately',
    in_flight_node_fraction='UNKNOWN; event completion never identifies fractional progress inside an unfinished node',
    current_remaining='Only bounds under an explicitly bound immutable DAG/progress model; stale fraction is not a current point',
    dynamic_decode='Future EOS/token count unknown; public maximum tokens is an upper bound, not observed remaining tokens',
    waiting='Atomic native queue/start acknowledgement needed; missing or delayed START does not prove waiting',
    no_forced_cuda_sync=True,no_kernel_privilege_claim=True,no_request_cap=True,
    exact_current_cursor_available=False,main_lifecycle_continuation_implemented=False)


def read(ref):
    require(set(ref)=={'path','sha256'} and receipt(ref['path'])==ref,'Exact immutable training source required')
    value=json.loads(Path(ref['path']).read_text())
    require(receipt(ref['path'])==ref,'Training source changed during read')
    return value


def snapshot(session,records,cutoff_ns):
    """Select by parent reader visibility before touching any future body."""
    current=advance_runtime_records(session,cutoff_ns,records);validate(current)
    statuses={};requests={};counts=Counter()
    for rid,public in current['full_public_cohort'].items():
        observed=current['observer']['requests'].get(rid,{})
        if public['arrival_ns']>cutoff_ns:status='FUTURE_PUBLIC_ARRIVAL'
        elif observed.get('handler_end_ns') is not None:
            status='COMPLETED_READY_VISIBLE' if observed.get('worker_next_ready_ns') is not None else 'HANDOFF_RESIDUAL_UNKNOWN'
        elif observed.get('handler_start_ns') is not None:status='BUSY_CURRENT_DEVICE_PROGRESS_UNKNOWN'
        else:status='ARRIVED_START_VISIBILITY_UNKNOWN'
        statuses[rid]=status;counts[status]+=1
        phases=observed.get('phases',{})
        requests[rid]=dict(status=status,handler_start_ns=observed.get('handler_start_ns'),
            handler_end_ns=observed.get('handler_end_ns'),worker_next_ready_ns=observed.get('worker_next_ready_ns'),
            visible_host_phase_end_ids=[key for key,value in phases.items() if value['end_ns'] is not None],
            visible_host_open_phase_ids=[key for key,value in phases.items() if value['end_ns'] is None],
            current_device_fraction=None,exact_current_remaining_work=None,
            no_START_is_waiting_proof=False,host_phase_end_is_device_completion=False)
    delays=[event['received_time_ns']-event['payload']['perf_ns'] for event in current['feedback_prefix']]
    return current,dict(cutoff_ns=cutoff_ns,session_sha256=current['session_sha256'],
        public_cohort_size=len(statuses),status_counts=dict(counts),request_states=requests,
        visible_feedback_events=len(delays),max_visible_reader_delay_ns=max(delays,default=None),
        positive_delay_visible_events=sum(d>0 for d in delays),
        exact_current_progress_receipts=0,exact_waiting_proofs=0,
        numeric_main_continuation_eligible=False,GPU_started=False)


def qualify_capture(plan_ref,context_ref,raw_ref):
    # Role and identities are checked before opening any outcome body.
    plan,context=read(plan_ref),read(context_ref)
    require(plan.get('split')=='training' and plan.get('training_only') is True
            and context.get('training_only') is True,'Explicit training role required before raw read')
    raw=read(raw_ref)
    require(raw.get('status')=='completed' and raw.get('split')=='training'
            and raw.get('training_only') is True and raw.get('plan')==plan_ref and raw.get('context')==context_ref
            and raw['execution_context_sha256']==plan['execution_context_sha256']
            and raw['route_id']==plan['route_id'] and raw['planned_public_cohort']==plan['trace']
            and len(raw['request_records'])==57 and len(raw['priming_request_records'])==12,
            'Exact original completed four-route 57+12 training capture required')
    aliases={alias:dict(supported_tasks=list(value['tasks'])) for alias,value in raw['ready'].items()}
    start=raw['phase_start_ns'];deadline=start+round(plan['recipe']['window_seconds']*1e9)
    require(raw['deadline_ns']==deadline and plan['recipe']['admission']=='uncapped_shared'
            and plan['recipe']['batch_size']==1,'Original uncapped unbatched windows required')
    rows=[dict(request_id=r['request_id'],task=r['task'],pool_id=r['pool_id'],
        alias=plan['routes'][r['request_id']],arrival_ns=start+r['offset_ns']) for r in plan['trace']]
    expected={r['request_id']:r for r in rows}
    require({r['request_id'] for r in raw['request_records']}==set(expected),'Complete original request registry required')
    for record in raw['request_records']:
        public=expected[record['request_id']]
        require(record.get('split')=='training' and record.get('status')=='completed'
                and record['model']==public['alias'] and record['task']==public['task']
                and record['pool_id']==public['pool_id'] and record['scheduled_time_ns']==public['arrival_ns'],
                'Original public request and execution identity changed')
    session=new_session(rows,aliases,plan['execution_context_sha256'],start,deadline,
        deadline+round(plan['recipe']['drain_seconds']*1e9))
    return plan,raw,session


def audit(manifest_path,output):
    out=Path(output).absolute();require(not out.exists(),'Exclusive CPU visibility-audit directory required')
    manifest_ref=receipt(manifest_path)
    require(manifest_ref['sha256']==ORIGINAL_MANIFEST_SHA256,'Exact original v41 training manifest required')
    manifest=read(manifest_ref)
    require(manifest.get('training_only') is True and manifest.get('validation_labels_used') is False
            and len(manifest['captures'])==4,'Four original training-only capture projections required')
    implementation=implementation_sources(__file__);sources=[manifest_ref,*implementation];captures=[]
    for entry in manifest['captures']:
        projection=read(entry['report']);sources.append(entry['report'])
        require(projection.get('training_only') is True and projection.get('validation_labels_used') is False,
                'Qualified training projection required')
        refs={}
        for filename in ('plan.json','context.json','raw.json'):
            matches=[r for r in projection['sources'] if r['path'].endswith('/whole57-joint-light-v23/'+entry['route']+'/'+filename)]
            require(len(matches)==1,'Exact one original training source required: '+filename)
            refs[filename]=matches[0]
        plan,raw,session=qualify_capture(refs['plan.json'],refs['context.json'],refs['raw.json'])
        require(entry['route']==plan['route_id'],'Capture route must equal original manifest route')
        sources.extend(refs.values());ticks=[]
        for offset in OFFSETS_NS:
            session,tick=snapshot(session,raw['request_records'],raw['phase_start_ns']+offset)
            tick['offset_ns']=offset;ticks.append(tick)
        captures.append(dict(route=entry['route'],sources=refs,snapshots=ticks,
            priming_records=12,measured_records=57,priming_not_inserted_in_public_cohort=True))
    sources=strict_references(sources)
    require(all(receipt(ref['path'])==ref for ref in sources),'Visibility audit source bytes changed')
    out.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,status='CPU_NATIVE_TRAINING_VISIBILITY_AUDITED_NO_CURRENT_DEVICE_CURSOR',
        captures=captures,cutoffs_per_capture=len(OFFSETS_NS),sources=sources,
        required_device_evidence=DEVICE_EVIDENCE_CONTRACT,
        GPU_started=False,validation_labels_read=False,training_labels_read=True,
        current_native_cursor_collector_implemented=False,main_lifecycle_continuation_implemented=False,
        numeric_queue_P95_or_six_term_accuracy_claimed=False,
        limitations=['Offline causal replay of original training captures; not an online worker snapshot implementation.',
            'START/PHASE/end/READY are host facts selected by parent reader receipt; no future worker response body is consumed by snapshot.',
            'Visible host semantic ends are not kernel/device completions, and do not identify a current fractional progress.',
            'No START can mean delayed visibility. Waiting depth and GPU progress remain unknown without additional native observations.',
            'The device-evidence contract is a requirement for future instrumentation, not an implemented or validated collector.'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.training_manifest,args.output)
    print(json.dumps(dict(status=result['status'],captures=len(result['captures']),cutoffs=result['cutoffs_per_capture'])))
