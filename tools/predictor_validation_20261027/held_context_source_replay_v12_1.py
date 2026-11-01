"""CPU replay of original paired context envelopes through the v12 state.

Only known paired RESTORE/SAVE events are replayed. Gaps and unpaired events
remain reported; no prediction, SM service, or GPU progress is validated.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sqlite3

from .context_envelope_diagnostic_v8 import context_intervals
from .dag_host_release_v7 import checked
from .dag_phase_scheduler_v8 import ActiveGpuNode,SchedulingView
from .held_context_scheduler_v12_1 import HeldContextScheduler
from .native_collect import receipt,save


def replay(history_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    history_ref=receipt(history_path);history=checked(history_ref);captures=[]
    sources=[history_ref,receipt(__file__),receipt(Path(__file__).with_name('held_context_scheduler_v12_1.py'))]
    for capture in history['observed_captures']:
        pending=checked(capture['source_report']);sql=next(s for s in pending['sources'] if s['path'].endswith('.sqlite'))
        with sqlite3.connect(Path(sql['path']).as_uri()+'?mode=ro',uri=True) as c:
            c.row_factory=sqlite3.Row;_,brackets,issues=context_intervals(c)
        mapping={w['alias']:tuple(w['context']) for w in pending['controlled_temporal_attribution']['creation_windows']}
        intervals=sorted((a,b,owner) for owner,k in mapping.items() for a,b in brackets[k]);starts=[a for a,b,p in intervals]
        # Reject overlap rather than choose a synthetic physical holder.
        overlaps=[]
        for left,right in zip(intervals,intervals[1:]):
            if right[0]<left[1]:overlaps.append(dict(left=left,right=right))
        if overlaps:raise ValueError('Paired context-envelope overlap cannot identify one held owner')
        events=sorted([(a,1,'RESTORE_START',owner) for a,b,owner in intervals]+
                      [(b,0,'SAVE_END',owner) for a,b,owner in intervals])
        scheduler=HeldContextScheduler(mapping,allow_source_replay=True)
        samples=[];event_index=0;different=0
        for span in capture['first_five_ms_frontier_spans']:
            t=capture['ready_timeline_origin_trace_ns']+span['start_offset_ns']
            while event_index<len(events) and events[event_index][0]<=t:
                stamp,_,kind,owner=events[event_index]
                scheduler.observe_context_event(stamp,kind,owner);event_index+=1
            nodes=[]
            for owner,task,pool,phases in span['frontier']:
                for phase in phases:nodes.append(ActiveGpuNode((owner,phase),1.,phase,owner,task,pool))
            query=scheduler.snapshot(SchedulingView(t,tuple(nodes)))
            index=bisect_right(starts,t)-1
            expected=intervals[index] if index>=0 and t<intervals[index][1] else None
            if query.held_owner!=(expected[2] if expected else None):raise ValueError('Source replay holder mismatch')
            if expected and query.context_age_ns!=t-expected[0]:raise ValueError('Source replay attained age mismatch')
            owners={r[0] for r in span['frontier']}
            mismatch=bool(len(owners)==1 and query.held_owner is not None and query.held_owner not in owners)
            different+=mismatch
            samples.append(dict(offset_ns=span['start_offset_ns'],pending_frontier=span['frontier'],
                held_owner=query.held_owner,context_age_ns=query.context_age_ns,
                sole_ready_differs_from_held=mismatch,paired_envelope_state_exact=True))
        captures.append(dict(source_report=capture['source_report'],source_sqlite=sql,
            replayed_context_events=event_index,checked_frontier_points=len(samples),
            sole_ready_differs_from_held_points=different,rows=samples,
            unresolved_context_pairing_issues=issues,unpaired_events_filled=False,
            remaining_model_availability_policy='UNKNOWN_NOT_IDENTIFIED'))
        sources.extend((capture['source_report'],sql,*pending['sources']))
    sources=list({r['path']:r for r in sources}.values())
    if any(receipt(r['path'])!=r for r in sources):raise ValueError('Context source changed during replay')
    result=dict(schema='gsdeploy.held_context_source_replay.v12_1',status='PAIRED_ENVELOPE_STATE_REPLAY_PASS',
        sources=sources,captures=captures,default_availability_fallback=False,
        sole_ready_changes_held_context=False,context_age_resets_on_host_event=False,
        numeric_work_progress_evaluated=False,independent_prediction_validation=False,
        physical_SM_service_identified=False,GPU_started=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=replay(a.history,a.output)
    print(json.dumps(dict(status=r['status'],report=receipt(Path(a.output)/'report.json'),
        checked_points=sum(c['checked_frontier_points'] for c in r['captures']),
        held_ready_mismatch_points=sum(c['sole_ready_differs_from_held_points'] for c in r['captures']))))

if __name__=='__main__':main()
