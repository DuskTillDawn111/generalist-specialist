"""Whole57 native lifecycle adapter v35: tolerate unchanged startup polling snapshots.

This external adapter leaves legacy sources frozen. A STARTed unfinished kernel
is a ready witness; absent queued-readiness evidence stays unknown. Context
envelopes label context selection and never identify SM service.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sqlite3
import statistics

from . import owner_lifecycle_mark_v21 as legacy
from . import whole57_joint_review_v23 as review_code
from .context_envelope_diagnostic_v8 import context_intervals
from .context_next_owner_v14 import features as base_features, previous_history, query_from_row
from .lossless_derived_reader_v30 import verify_mapping
from .nvtx_interval_index_v2 import NVTXRangeIndex
from .trace_export import linux_pid, _stage_label
from .unknown_preflight_archive_v1 import save_synced

SCHEMA='gsdeploy.whole57_native_lifecycle_training_adapter.v35'
RESIDENTS=legacy.RESIDENTS


def checked(ref):
    if legacy.receipt(ref['path'])!=ref:raise ValueError('Changed explicit source receipt: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def native_visible_features(member,pools):
    """Legacy feature order, preserving unknown gpu_ready in native-visible rows.

    New records are explicitly a different observation scope. A missing native
    readiness witness is not encoded as false. These are physical mark training
    features, not a ready-complete engine snapshot or a qualified oracle.
    """
    query,states=member['query'],member['states'];ready={r[0] for r in query['pending_frontier']}
    if any((states[o].gpu_ready is True)!=(o in ready) for o in RESIDENTS):
        raise ValueError('Visible ready witnesses and frontier differ')
    return [*base_features(query,pools),*legacy.state_features(states)]


def request_ranges(connection,raw,anchors):
    known={r['request_id']:r for r in raw['priming_request_records']+raw['request_records']}
    anchor_map={(a['pid'],a['request_id'],a['kind'],a['segment_id']):a for a in anchors}
    ranges=[];offsets=defaultdict(list);seen=set()
    for row in connection.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL ORDER BY n.rowid'):
        stage=_stage_label(row['label'] or '')
        if stage is None or stage[0] not in known:continue
        pid=linux_pid(row['globalTid']);rid,kind,segment=stage;key=(pid,rid,kind,segment)
        if key not in anchor_map or key in seen or raw['ready'][known[rid]['model']]['pid']!=pid:
            raise ValueError('Native stage and original worker anchor/PID identity differ')
        seen.add(key);anchor=anchor_map[key]
        if not row['start']<row['end']:raise ValueError('Positive native stage required')
        offsets[pid].extend((anchor['start_ns']-row['start'],anchor['end_ns']-row['end']))
        ranges.append(dict(nsys_start_ns=row['start'],nsys_end_ns=row['end'],global_tid=row['globalTid'],
            pid=pid,stage=stage,request_id=rid,kind=kind,segment_id=segment,owner=known[rid]['model']))
    if seen!=set(anchor_map):raise ValueError('All original 69-request phase anchors must be present exactly once')
    mappings={pid:dict(offset_ns=round(statistics.median(v)),max_boundary_residual_ns=max(abs(x-round(statistics.median(v))) for x in v),
        method='fixed_median_worker_boundary_projection_not_calibrated_common_clock') for pid,v in offsets.items()}
    if any(m['max_boundary_residual_ns']>5_000_000 for m in mappings.values()):raise ValueError('Legacy source mapping tolerance exceeded')
    return ranges,mappings


def context_mapping(connection,raw,context,events,brackets):
    """Serial creation -> primary driver process, independently checked by prime.

    A second driver context from that exact same scheduler globalPid stays a
    distinct context. No arithmetic join to the CUPTI/container PID is used.
    """
    aliases=set(raw['ready']);identities=raw['live_identity_events'];base=connection.execute('SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME').fetchall()
    if len(base)!=1:raise ValueError('One native session required')
    baseline=identities[0]
    if baseline['phase']!='pre_startup' or baseline['launchers'] or baseline['ready'] or baseline['nvml_processes_mib']:
        raise ValueError('Observed empty pre-startup baseline required')
    startup=[]
    for event in identities:
        if event['phase']!='startup_begin' or set(event['ready'])==aliases:continue
        if startup and set(event['launchers'])==set(startup[-1]['launchers']):
            # Polling may repeat a creation stage while a worker initializes.
            # It cannot relabel a process or shrink/advance READY spuriously.
            earlier=startup[-1]
            for owner,identity in event['launchers'].items():
                if any(identity.get(k)!=earlier['launchers'][owner].get(k) for k in ('pid','start_ticks','execution_nonce')):
                    raise ValueError('Repeated startup snapshot changed owned process identity')
            if set(event['ready'])!=set(earlier['ready']):
                raise ValueError('Repeated startup stage has inconsistent READY set')
            continue
        startup.append(event)
    all_ready=next(x for x in identities if set(x['ready'])==aliases)
    if len(startup)!=4:raise ValueError('Four original serial startup windows required')
    first={key:min(x['timestamp'] for x in values) for key,values in events.items()};primary={};windows=[];previous=set()
    for i,row in enumerate(startup):
        current=set(row['launchers'])
        if not current>=previous or len(current-previous)!=1 or set(row['ready'])!=previous:raise ValueError('Serial owned startup identity required')
        owner=next(iter(current-previous));identity=row['launchers'][owner]
        if (identity['pid']!=raw['ready'][owner]['pid'] or identity['start_ticks']!=raw['live_owned_processes'][owner]['start_ticks']
                or identity['execution_nonce']!=context['execution_nonce']):raise ValueError('Original owner process identity changed')
        begin=row['time_ns'];end=(startup[i+1] if i+1<len(startup) else all_ready)['time_ns']
        candidates=[key for key,stamp in first.items() if begin<base[0][0]+stamp<end]
        if len(candidates)!=1:raise ValueError('Primary context creation is ambiguous')
        primary[owner]=candidates[0];windows.append(dict(owner=owner,primary_context=list(candidates[0]),begin_perf_ns=begin,end_perf_ns=end))
        previous=current
    process_owner={key[3]:owner for owner,key in primary.items()}
    if len(process_owner)!=4:raise ValueError('Four distinct primary scheduler processes required')
    mapping={key:process_owner.get(key[3]) for key in events}
    if any(owner is None for owner in mapping.values()):raise ValueError('Unattributed scheduler context must remain unresolved')
    return mapping,dict(primary_creation_windows=windows,context_owner_mapping=[dict(context=list(key),owner=owner,
        basis='serial_primary_creation_and_independent_priming' if key==primary[owner] else 'additional_context_same_exact_scheduler_process_as_priming_verified_primary') for key,owner in mapping.items()],
        direct_OS_namespace_join=False,arithmetic_PID_join=False,driver_and_CUPTI_context_ID_equality_assumed=False,
        original_primary_context_count=4,driver_context_count=len(mapping),priming_cross_checks=[])


def transition_records(pairs,issues,lo,hi,capture,public):
    """Keep raw adjacency/context identities; unresolved boundaries are labelled."""
    issue_times=sorted(x['timestamp'] for x in issues);clean=[];unresolved=[]
    if any(y[0]<x[1] for x,y in zip(pairs,pairs[1:])):raise ValueError('Overlapping paired context envelopes')
    for current,nxt in zip(pairs,pairs[1:]):
        begin,end,owner,key=current;next_begin,next_end,target,next_key=nxt
        if not lo<=end<hi or next_begin>=hi:continue
        row=dict(capture=capture,job_id='whole57:'+capture,job_name='original_full57_one_capture',public_workload=public,
            held_owner=owner,current_restore_start_ns=begin,save_end_ns=end,context_age_at_save_ns=end-begin,
            next_owner=target,next_restore_start_ns=next_begin,save_to_next_restore_ns=next_begin-end,
            next_envelope_save_end_ns=next_end,held_driver_context=list(key),next_driver_context=list(next_key),
            same_owner=owner==target,same_driver_context=key==next_key,physical_SM_service_identified=False)
        if bisect_right(issue_times,next_begin)>bisect_left(issue_times,begin):
            unresolved.append(dict(row,reason='UNPAIRED_OR_REPEATED_EVENT_IN_CURRENT_ENVELOPE_OR_HANDOFF'))
        else:clean.append(row)
    return clean,unresolved


def extract(review_reference,archive_reference,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    review=checked(review_reference)
    if (review.get('schema')!='gsdeploy.whole57_joint_training_review.v23' or review.get('status')!='PASS'
            or review.get('issues')!=[] or review.get('requests')!=57 or review.get('training_only') is not True
            or review.get('validation_started') is not False or review.get('model_fitted') is not False or not review.get('trace')):
        raise ValueError('Completed light whole57 training review required')
    refs=[review_reference,*review['evidence'].values(),review['host_lifecycle'],review['reviewer'],review['trace']['sqlite'],
        review['trace']['export']['source'],review['trace']['session_clock'],review['trace']['session_metadata'],review['trace']['native_first_api_kernel']]
    for ref in refs:
        if legacy.receipt(ref['path'])!=ref:raise ValueError('Original whole57 source changed')
    # The logical parsed path is never opened, even when it happens to exist.
    storage=verify_mapping(archive_reference,review['trace']['parsed'])
    raw=checked(review['evidence']['raw.json']);plan=checked(review['evidence']['plan.json']);context=checked(review['evidence']['context.json'])
    review_code.collect.audit(Path(review['evidence']['plan.json']['path']).parent)
    host_issues,host_rows,anchors=review_code.host_lifecycle_rows(plan,raw)
    if host_issues or host_rows!=checked(review['host_lifecycle'])['rows']:raise ValueError('Original full57 host lifecycle mismatch')
    if set(raw['ready'])!=set(RESIDENTS) or len(raw['priming_request_records'])!=12:raise ValueError('Four owners and original 12 primers required')
    output.mkdir(parents=True,exist_ok=False)
    source_code=[legacy.receipt(p) for p in (__file__,legacy.__file__,review_code.__file__,Path(__file__).with_name('lossless_derived_reader_v30.py'))]
    capture=review['evidence']['raw.json']['sha256'];known={r['request_id']:r for r in raw['priming_request_records']+raw['request_records']};measured={r['request_id'] for r in raw['request_records']}
    con=sqlite3.connect(Path(review['trace']['sqlite']['path']).as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    try:
        ranges,mappings=request_ranges(con,raw,anchors);range_index=NVTXRangeIndex(ranges)
        events,brackets,pairing_issues=context_intervals(con);mapping,attribution=context_mapping(con,raw,context,events,brackets)
        unions={key:legacy.IntervalIndex(spans) for key,spans in brackets.items()}
        phase_spans=defaultdict(list);handler_spans=defaultdict(list);issued_spans=defaultdict(list);ready_spans=defaultdict(list)
        for r in ranges:phase_spans[r['owner']].append((r['nsys_start_ns'],r['nsys_end_ns']))
        for r in known.values():
            pid=raw['ready'][r['model']]['pid'];offset=mappings[pid]['offset_ns'];response=r['worker_response']
            handler_spans[r['model']].append((response['start_perf_ns']-offset,response['end_perf_ns']-offset))
        prime_hits=defaultdict(Counter);prime_totals=Counter();request_kernels=Counter();frontier_spans=defaultdict(list);counts=Counter()
        semantic={rid:{s['id']:s['semantic_phase'] for s in r['worker_response'].get('semantic_forward_observations',{}).get('records',[])} for rid,r in known.items()}
        semantic_ranges={rid:[r for r in ranges if r['request_id']==rid and r['segment_id'] in semantic[rid]] for rid in known}
        query='''SELECT k.rowid kernel_rowid,k.start,k.end,k.globalPid,k.contextId,k.streamId,k.correlationId,
                 r.start api_start,r.end api_end,r.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL k
                 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
                 AND (r.globalTid >> 24)=(k.globalPid >> 24) ORDER BY k.start,k.end,k.rowid'''
        witness_path=output/'native-kernel-witnesses.jsonl.gz'
        with gzip.open(witness_path,'wt',encoding='utf-8') as stream:
            last_kernel=None
            for row in con.execute(query):
                identity=row['kernel_rowid']
                if identity==last_kernel:raise ValueError('Ambiguous native process/correlation launch')
                last_kernel=identity
                matched=range_index.owner_for_launch(row['globalTid'],row['api_start'],row['api_end'])
                if matched is None:continue
                rid=matched['request_id'];request=known[rid];owner=request['model'];pid=linux_pid(row['globalPid']);begin,end=row['start'],row['end']
                if pid!=raw['ready'][owner]['pid'] or begin>=end:raise ValueError('Owned native kernel identity/endpoints differ')
                if rid not in measured:
                    prime_totals[rid]+=2
                    for key,index in unions.items():
                        for stamp in (begin,end):
                            i=bisect_right(index.starts,stamp)-1
                            if i>=0 and stamp<=index.spans[i][1]:prime_hits[rid][key]+=1
                    continue
                request_kernels[rid]+=1;ready_spans[owner].append((begin,end))
                if row['api_end']<end:issued_spans[owner].append((row['api_end'],end))
                labels=[(r['nsys_end_ns']-r['nsys_start_ns'],semantic[rid][r['segment_id']]) for r in semantic_ranges[rid] if r['nsys_start_ns']<=row['api_start'] and row['api_end']<=r['nsys_end_ns']]
                phase=min(labels)[1] if labels else ('ocr_inference' if request['task']=='ocr' else 'outside_language_forwards')
                frontier_spans[owner,request['task'],request['pool_id'],phase].append((begin,end))
                value=dict(request_id=rid,owner=owner,pid=pid,correlation_id=row['correlationId'],CUPTI_context_id=row['contextId'],stream_id=row['streamId'],
                    API_entry_ns=row['api_start'],API_return_ns=row['api_end'],observed_start_ns=begin,observed_end_ns=end,phase=phase,
                    certified_queued_ready_bound_ns=None,ready_witness='STARTED_UNFINISHED_ONLY',all_native_dependencies_identified=False)
                stream.write(json.dumps(value,separators=(',',':'))+'\n');counts['measured_kernels']+=1
        if set(request_kernels)!=measured:raise ValueError('Original full57 native kernel denominator incomplete')
        for row in raw['priming_request_records']:
            rid,owner=row['request_id'],row['model'];hits=prime_hits[rid];total=prime_totals[rid]
            own=sum(count for key,count in hits.items() if mapping[key]==owner)
            if not total or own!=total or any(count for key,count in hits.items() if mapping[key]!=owner):
                raise ValueError('Independent original priming endpoint/context owner check failed')
            attribution['priming_cross_checks'].append(dict(request_id=rid,owner=owner,kernel_endpoints=total,own_context_hits=own))
    finally:con.close()
    native_ranges=[r for r in ranges if r['request_id'] in measured];lo=min(r['nsys_start_ns'] for r in native_ranges);hi=max(r['nsys_end_ns'] for r in native_ranges)
    pairs=sorted((a,b,mapping[key],key) for key,spans in brackets.items() for a,b in spans)
    public=sorted([[plan['routes'][r['request_id']],r['task'],r['pool_id'],r['offset_ns']] for r in plan['trace']])
    transitions,rejected=transition_records(pairs,pairing_issues,lo,hi,capture,public)
    indexes={name:{owner:legacy.IntervalIndex(spans[owner]) for owner in RESIDENTS} for name,spans in
        (('phase',phase_spans),('certain',handler_spans),('possible',handler_spans),('issued',issued_spans),('started',ready_spans))}
    frontier_indexes=[(*key,legacy.IntervalIndex(spans)) for key,spans in sorted(frontier_spans.items())]
    def snapshot(clock):
        visible=defaultdict(list)
        for owner,task,pool,phase,index in frontier_indexes:
            if index.contains(clock):visible[owner,task,pool].append(phase)
        frontier=[[owner,task,pool,sorted(phases)] for (owner,task,pool),phases in sorted(visible.items())]
        ready={r[0] for r in frontier};states={}
        for owner in RESIDENTS:
            handler=indexes['phase'][owner].contains(clock) or indexes['certain'][owner].contains(clock)
            issued=True if indexes['issued'][owner].contains(clock) or owner in ready else None if handler else False
            states[owner]=legacy.OwnerLifecycle(handler,issued,True if owner in ready else None if handler else False)
        return frontier,states
    histories=previous_history(transitions);dataset=output/'mark-records.jsonl.gz';state_counts=defaultdict(Counter)
    with gzip.open(dataset,'wt',encoding='utf-8') as stream:
        for row,history in zip(transitions,histories):
            frontier,states=snapshot(row['save_end_ns']);pending={r[0] for r in frontier}
            row.update(pending_frontier=frontier,next_owner_is_pending=row['next_owner'] in pending,held_owner_is_pending=row['held_owner'] in pending,
                frontier_observation_scope='native_started_unfinished_only_queued_readiness_unknown',positive_terminal_risk_interval=False)
            query=query_from_row(row,history)
            serialized=dict(row=row,query=query,states={o:asdict(s) for o,s in states.items()},
                observation_domain='native_visible_with_explicit_unknown_queued_readiness',handler_clock_projection='fixed_median_uncalibrated')
            stream.write(json.dumps(serialized,separators=(',',':'))+'\n')
            for owner,state in states.items():state_counts[owner][json.dumps(asdict(state),sort_keys=True)]+=1
    save_synced(output/'native-transition-ledger.json',dict(qualified=transitions,unresolved=rejected,pairing_issues=pairing_issues))
    save_synced(output/'native-lifecycle-indexes.json',{name:{owner:index.spans for owner,index in group.items()} for name,group in indexes.items()})
    for ref in refs+source_code:
        if legacy.receipt(ref['path'])!=ref:raise ValueError('Source changed during native extraction')
    if legacy.receipt(archive_reference['path'])!=archive_reference or legacy.receipt(storage['compressed']['path'])!=storage['compressed']:
        raise ValueError('Explicit parsed archive changed during native extraction')
    result=dict(schema=SCHEMA,adapter_version=35,status='NATIVE_VISIBLE_MARK_DATASET_EXTRACTED_NO_FIT',training_only=True,validation_labels_used=False,
        review=review_reference,parsed_storage=storage,sources=refs,implementations=source_code,capture=capture,route_id=plan['route_id'],
        native_window_ns=[lo,hi],full_original_measured_requests=57,original_priming_requests=12,
        measured_requests_by_owner=dict(Counter(r['model'] for r in raw['request_records'])),
        kernel_count_by_request=dict(request_kernels),counts=dict(counts),controlled_temporal_attribution=attribution,
        label_counts=dict(Counter(r['next_owner'] for r in transitions)),held_counts=dict(Counter(r['held_owner'] for r in transitions)),
        transition_count=len(transitions),unresolved_transition_count=len(rejected),pairing_issue_count=len(pairing_issues),
        qualified_same_owner_count=sum(r['same_owner'] for r in transitions),qualified_same_driver_context_count=sum(r['same_driver_context'] for r in transitions),
        unresolved_same_owner_count=sum(r['same_owner'] for r in rejected),lifecycle_state_counts={o:dict(c) for o,c in state_counts.items()},
        mark_records=legacy.receipt(dataset),native_transition_ledger=legacy.receipt(output/'native-transition-ledger.json'),
        native_kernel_witnesses=legacy.receipt(witness_path),lifecycle_indexes=legacy.receipt(output/'native-lifecycle-indexes.json'),
        original_handler_projection_by_pid=mappings,calibrated_cross_clock_error_bound_ns=None,
        ready_scope='Native STARTED unfinished witness only. Queued dependency readiness is UNKNOWN, not false.',
        legacy_feature_order_compatible=True,legacy_strict_ready_complete_snapshot=False,
        legacy_source_prefix_record_shape_compatible=True,source_DAG_queued_readiness_fabricated=False,
        leave_out_unit='entire raw capture SHA, including every route/request/adjacent transition from that capture',
        independent_capture_count=1,independent_validation=False,fit_performed=False,physical_SM_service_identified=False,
        context_envelope_equals_service=False,GPU_started=False)
    save_synced(output/'report.json',result);return result


def source_prefix_rows(report_reference,*,include_indexes=False):
    """External API matching legacy record shape, with native-visible semantics."""
    report=checked(report_reference)
    if report.get('schema')!=SCHEMA or report.get('training_only') is not True or report.get('validation_labels_used') is not False:
        raise ValueError('Explicit whole57 native-visible training dataset required')
    for ref in report['sources']+report['implementations']:
        if legacy.receipt(ref['path'])!=ref:raise ValueError('Dataset bound source changed')
    ref=report['mark_records']
    if legacy.receipt(ref['path'])!=ref:raise ValueError('Native mark records changed')
    with gzip.open(ref['path'],'rt') as stream:members=[json.loads(line) for line in stream]
    for member in members:member['states']={o:legacy.OwnerLifecycle(**s) for o,s in member['states'].items()}
    if len(members)!=report['transition_count']:raise ValueError('Complete native transition denominator required')
    sources=[report_reference,*report['sources'],*report['implementations'],report['parsed_storage']['archive_manifest'],report['parsed_storage']['compressed'],ref]
    if not include_indexes:return members,sources
    saved=checked(report['lifecycle_indexes'])
    indexes={report['capture']:{name:{owner:legacy.IntervalIndex(spans) for owner,spans in group.items()} for name,group in saved.items()}}
    return members,sources,indexes


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('review','review-sha256','archive-manifest','archive-manifest-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=extract(dict(path=str(Path(args.review).absolute()),sha256=args.review_sha256),
        dict(path=str(Path(args.archive_manifest).absolute()),sha256=args.archive_manifest_sha256),args.output)
    print(json.dumps({k:result[k] for k in ('status','transition_count','label_counts','qualified_same_owner_count','unresolved_transition_count')}))
