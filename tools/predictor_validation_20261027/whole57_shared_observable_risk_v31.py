"""Native context-risk chronology with conservative shared handler observations.

Native GS_STAGE proves a handler active. Outside those ranges, true handler
status is unknown; no worker median projection is promoted to a native fact.
An engine can match this projection by mapping inactive handlers to unknown.
Every physical SAVE is retained independently of marked adjacency qualification.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sqlite3

from .whole57_lifecycle_training_adapter_v30 import checked,SCHEMA as ADAPTER_SCHEMA
from .owner_lifecycle_mark_v21 import IntervalIndex,receipt
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced
from .implementation_source_closure_v27 import implementation_sources

SCHEMA='gsdeploy.whole57_shared_observable_native_context_risk.v31'
OWNERS=('generalist_2#0','ocr#0','text#0','text#1')


def handler_projection(indexes,clock):
    return {owner:True if indexes[owner].contains(clock) else None for owner in OWNERS}


def engine_projection(handler_active):
    if set(handler_active)!=set(OWNERS) or any(type(x) is not bool for x in handler_active.values()):
        raise ValueError('Complete settled engine handler booleans required')
    return {owner:True if handler_active[owner] else None for owner in OWNERS}


def shared_features(held_owner,previous_held_owner,context_age_ns,handler_active):
    if (held_owner not in OWNERS or previous_held_owner is not None and previous_held_owner not in OWNERS
            or type(context_age_ns) is not int or context_age_ns<0 or set(handler_active)!=set(OWNERS)
            or any(x is not None and x is not True for x in handler_active.values())):
        raise ValueError('Shared observable context/age/true-or-unknown handler projection required')
    return dict(held_owner=held_owner,previous_held_owner=previous_held_owner,context_age_ns=context_age_ns,
        handler_active=dict(handler_active))


def extract(adapter_reference,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    adapter=checked(adapter_reference)
    if (adapter.get('schema')!=ADAPTER_SCHEMA or adapter.get('status')!='NATIVE_VISIBLE_MARK_DATASET_EXTRACTED_NO_FIT'
            or adapter.get('training_only') is not True or adapter.get('validation_labels_used') is not False):
        raise ValueError('Explicit completed native whole57 training adapter required')
    implementation=implementation_sources(__file__)
    refs=[adapter_reference,*adapter['sources'],*adapter['implementations'],adapter['lifecycle_indexes'],adapter['native_transition_ledger'],
        adapter['parsed_storage']['archive_manifest'],adapter['parsed_storage']['compressed'],*implementation]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Source changed before risk extraction')
    indexes=checked(adapter['lifecycle_indexes']);phase={owner:IntervalIndex(spans) for owner,spans in indexes['phase'].items()}
    ledger=checked(adapter['native_transition_ledger']);mark_by_save={(r['held_owner'],r['current_restore_start_ns'],r['save_end_ns']):r for r in ledger['qualified']}
    issues=ledger['pairing_issues'];issue_times=sorted(x['timestamp'] for x in issues)
    mapping={tuple(r['context']):r['owner'] for r in adapter['controlled_temporal_attribution']['context_owner_mapping']}
    review=checked(adapter['review']);sql=review['trace']['sqlite'];lo,hi=adapter['native_window_ns'];capture=adapter['capture']
    events=[];tags={};pairs=[]
    with sqlite3.connect(Path(sql['path']).as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        tags={r['id']:r['name'] for r in c.execute('SELECT * FROM ENUM_GPU_CTX_SWITCH')}
        for r in c.execute('SELECT * FROM GPU_CONTEXT_SWITCH_EVENTS ORDER BY timestamp,seqNo'):
            event=dict(r);event['tag_name']=tags[r['tag']];event['context_key']=[r[k] for k in ('vmId','gpuId','contextId','globalPid')]
            event['owner']=mapping[tuple(event['context_key'])];events.append(event)
    opened={}
    for event in events:
        key=tuple(event['context_key'])
        if event['tag_name']=='RESTORE_START':opened[key]=event
        elif event['tag_name']=='SAVE_END':
            if key in opened:
                begin=opened.pop(key)
                pairs.append(dict(begin=begin,end=event))
        else:raise ValueError('Unknown native scheduler tag')
    pairs.sort(key=lambda x:x['begin']['timestamp']);save_pairs={p['end']['seqNo']:p for p in pairs}
    incoming=defaultdict(list)
    for row in ledger['qualified']:incoming[row['next_owner'],row['next_restore_start_ns']].append(row)
    allcuts=sorted({t for index in phase.values() for span in index.spans for t in span})
    out.mkdir(parents=True,exist_ok=False);counts=Counter();save_rows=[];risk_path=out/'risk-chronology.jsonl.gz';mark_path=out/'shared-mark-records.jsonl.gz'
    with gzip.open(risk_path,'wt',encoding='utf-8') as stream:
        for pair in pairs:
            begin,end=pair['begin']['timestamp'],pair['end']['timestamp'];left,right=max(begin,lo),min(end,hi)
            if left>=right:continue
            owner=pair['begin']['owner'];key=pair['begin']['context_key'];prior=incoming[owner,begin]
            previous=prior[0]['held_owner'] if len(prior)==1 else None
            history_status='UNIQUE_ALREADY_COMPLETED_SAME_CAPTURE_HANDOFF' if len(prior)==1 else 'UNKNOWN_PRIOR_HANDOFF'
            unresolved=bisect_right(issue_times,end)>bisect_left(issue_times,begin)
            qualified=not unresolved
            cuts=[left,*allcuts[bisect_right(allcuts,left):bisect_left(allcuts,right)],right]
            for a,b in zip(cuts,cuts[1:]):
                event_observed=b==end and end<hi
                features=shared_features(owner,previous,a-begin,handler_projection(phase,a))
                row=dict(capture=capture,held_owner=owner,driver_context=key,restore_start_ns=begin,
                    start_ns=a,end_ns=b,age_entry_ns=a-begin,age_exit_ns=b-begin,
                    physical_SAVE_END_observed=event_observed,physical_SAVE_seqNo=pair['end']['seqNo'] if event_observed else None,
                    exposure_qualified=qualified,unresolved_reason='UNPAIRED_OR_REPEATED_EVENT_IN_ENVELOPE' if unresolved else None,
                    shared_left_features=features,previous_holder_status=history_status,
                    hazard_target='physical_context_SAVE_envelope_not_SM_service',right_boundary_is_feature=False)
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
                counts['all_paired_intervals']+=1;counts['all_paired_exposure_ns']+=b-a;counts['paired_SAVE_events']+=event_observed
                if qualified:
                    counts['qualified_risk_intervals']+=1;counts['qualified_exposure_ns']+=b-a;counts['qualified_SAVE_events']+=event_observed
                else:counts['unresolved_exposure_ns']+=b-a
    for event in events:
        if event['tag_name']!='SAVE_END' or not lo<=event['timestamp']<hi:continue
        pair=save_pairs.get(event['seqNo']);begin=pair['begin']['timestamp'] if pair else None
        mark=mark_by_save.get((event['owner'],begin,event['timestamp'])) if pair else None
        save_rows.append(dict(capture=capture,owner=event['owner'],driver_context=event['context_key'],
            physical_SAVE_END_ns=event['timestamp'],seqNo=event['seqNo'],restore_start_ns=begin,
            context_age_ns=event['timestamp']-begin if begin is not None else None,
            next_owner=mark['next_owner'] if mark else None,next_driver_context=mark['next_driver_context'] if mark else None,
            next_restore_start_ns=mark['next_restore_start_ns'] if mark else None,
            next_mark_status='QUALIFIED_ORIGINAL_ADJACENCY' if mark else 'UNKNOWN_NEXT_MARK',
            missing_mark_reason=None if mark else 'UNPAIRED_SAVE' if pair is None else 'UNRESOLVED_HANDOFF_OR_NO_NEXT_PAIRED_RESTORE_IN_WINDOW',
            same_owner=mark['same_owner'] if mark else None,same_driver_context=mark['same_driver_context'] if mark else None))
    with gzip.open(mark_path,'wt',encoding='utf-8') as stream:
        for row in ledger['qualified']:
            owner,begin,end=row['held_owner'],row['current_restore_start_ns'],row['save_end_ns'];prior=incoming[owner,begin]
            previous=prior[0]['held_owner'] if len(prior)==1 else None
            value=dict(capture=capture,features=shared_features(owner,previous,end-begin,handler_projection(phase,end)),
                next_owner=row['next_owner'],held_driver_context=row['held_driver_context'],next_driver_context=row['next_driver_context'],
                SAVE_END_ns=end,next_restore_start_ns=row['next_restore_start_ns'],same_owner=row['same_owner'],
                same_driver_context=row['same_driver_context'],training_only=True,
                future_label_fields_not_features=True)
            stream.write(json.dumps(value,separators=(',',':'))+'\n')
    save_synced(out/'physical-SAVE-event-ledger.json',dict(events=save_rows,pairing_issues=issues))
    # Native gaps are retained without holder or fabricated RESTORE age.
    covered=sorted((max(p['begin']['timestamp'],lo),min(p['end']['timestamp'],hi)) for p in pairs if p['begin']['timestamp']<hi and p['end']['timestamp']>lo)
    gaps=[];cursor=lo
    for a,b in covered:
        if a<cursor:raise ValueError('Overlapping native paired context risk')
        if a>cursor:gaps.append([cursor,a])
        cursor=b
    if cursor<hi:gaps.append([cursor,hi])
    save_synced(out/'native-context-gaps.json',dict(native_window_ns=[lo,hi],gaps=gaps,held_owner=None,restore_age=None))
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Source changed during risk extraction')
    summary=dict(schema=SCHEMA,status='SHARED_OBSERVABLE_RISK_AND_MARK_EXTRACTED_NO_FIT',training_only=True,
        validation_labels_used=False,source_adapter=adapter_reference,sources=refs,implementations=implementation,capture=capture,native_window_ns=[lo,hi],
        original_full57_retained=True,original_twelve_primers_retained=True,
        native_window_limits='First measured native GS_STAGE start through last measured native GS_STAGE end. Original perf offer/deadline/drain retained in source; no exact native conversion of perf arrival clock is claimed.',
        counts=dict(counts),physical_SAVE_event_count=len(save_rows),physical_SAVE_by_owner=dict(Counter(r['owner'] for r in save_rows)),
        SAVE_without_next_mark_count=sum(r['next_owner'] is None for r in save_rows),mark_record_count=len(ledger['qualified']),
        mark_target_counts=dict(Counter(r['next_owner'] for r in ledger['qualified'])),
        qualified_same_owner_count=sum(r['same_owner'] for r in ledger['qualified']),
        risk_chronology=receipt(risk_path),mark_records=receipt(mark_path),physical_SAVE_event_ledger=receipt(out/'physical-SAVE-event-ledger.json'),
        native_context_gaps=receipt(out/'native-context-gaps.json'),gap_duration_ns=sum(b-a for a,b in gaps),
        handler_projection='Native GS_STAGE containment proves true; all other states unknown. No false idle inference or median-clock future END used.',
        matched_engine_projection='Engine running handler -> true; inactive -> unknown. This is deliberately coarser than full handler state.',
        excluded_features=['pending_frontier','issued_queue','GPU_ready','remaining_work','future_request_count','future_public_workload'],
        fit_performed=False,calibrated_cross_clock_error_bound_ns=None,context_envelope_equals_SM_service=False,
        physical_SM_service_identified=False,independent_validation=False,GPU_started=False)
    save_synced(out/'report.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True);p.add_argument('--adapter-sha256',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=extract(dict(path=str(Path(args.adapter).absolute()),sha256=args.adapter_sha256),args.output)
    print(json.dumps({k:r[k] for k in ('status','counts','physical_SAVE_event_count','SAVE_without_next_mark_count','mark_target_counts')}))
