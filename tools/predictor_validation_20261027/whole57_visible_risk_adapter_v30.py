"""Whole-run training observations -> receipt-causal features + isolated risk labels.

This adapter never makes a GPU checkpoint or fits a recovery-risk model.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from gsdeploy.forecasting import observed_state
from gsdeploy.resources import identity
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

SCHEMA='gsdeploy.whole57_reader_visible_risk_training.v30'
HORIZONS_NS=(1_000_000_000,5_000_000_000,10_000_000_000)
INPUT_FIELDS=('task','pool_id','input_size','exact_input_sha256')
VISIBLE_PHASE_FIELDS=('semantic_phase','forward_index','input_sequence_length','input_batch_size')
HOST_BOUNDARIES=('enqueued_time_ns','actual_submit_time_ns','submit_write_completed_time_ns')


def public_registry(plan,phase_start):
    inputs={(r['task'],r['pool_id']):{k:r[k] for k in INPUT_FIELDS} for r in plan['inputs']}
    result=[]
    for request in plan['trace']:
        task,pool=request['task'],request['pool_id'];rid=request['request_id']
        result.append(dict(request_id=rid,alias=plan['routes'][rid],task=task,pool_id=pool,
            arrival_ns=phase_start+request['offset_ns'],input_features=copy.deepcopy(inputs[task,pool])))
    if len({r['request_id'] for r in result})!=len(result): raise ValueError('Duplicate public request')
    return sorted(result,key=lambda r:(r['arrival_ns'],r['request_id']))


def arrivals_through(registry,cutoff,since=0):
    return [dict(sequence=i,request_id=r['request_id'],alias=r['alias'],task=r['task'],pool_id=r['pool_id'],
                 arrival_ns=r['arrival_ns'],observed_time_ns=r['arrival_ns'])
            for i,r in enumerate(registry,1) if i>since and r['arrival_ns']<=cutoff]


def received_phase_metadata(records,cutoff):
    """Extra fields are visible only on actual received semantic boundary events."""
    result={}
    for record in records:
        for event in record.get('execution_events',[]):
            if event['received_time_ns']>cutoff or 'event' not in event: continue
            key=record['request_id'],event['segment_id']
            result.setdefault(key,{}).update({k:event[k] for k in VISIBLE_PHASE_FIELDS if k in event})
    return result


def visible_host_pipeline(records,cutoff):
    """Parent-owned event times are visible when reached, not before a write returns."""
    return {row['request_id']:{field:row[field] for field in HOST_BOUNDARIES
            if row.get(field) is not None and row[field]<=cutoff} for row in records}


def prefix_features(state,registry,*,cutoff,declared_at,metadata,host_pipeline):
    """Accept a reconciled public observer, never a full outcome record."""
    if state['observed_at_ns']!=cutoff: raise ValueError('Current observer cutoff required')
    by_id={r['request_id']:r for r in registry}; waiting=[]; active=[]; handoff=[]; details=[]
    for public in registry:
        rid=public['request_id'];row=state['requests'].get(rid)
        if public['arrival_ns']>cutoff: continue
        if row is None or 'arrival_ns' not in row: raise ValueError('Complete public arrival feed required')
        if row['handler_start_ns'] is None: status='WAITING_FOR_READER_VISIBLE_START';waiting.append(public)
        elif row['handler_end_ns'] is None: status='HANDLER_STARTED_NOT_OBSERVED_ENDED';active.append(public)
        elif row['worker_next_ready_ns'] is None: status='RESULT_SEEN_NEXT_READY_UNOBSERVED';handoff.append(public)
        else: status='WORKER_READY_OBSERVED'
        phases=[]
        for segment in row['phase_order']:
            phase=row['phases'][segment]
            phases.append(dict(segment_id=segment,**phase,**metadata.get((rid,segment),{})))
        host=copy.deepcopy(host_pipeline.get(rid,{}))
        pipeline=('NOT_OBSERVED_ENQUEUED' if 'enqueued_time_ns' not in host else
                  'ENQUEUED_NOT_OBSERVED_DISPATCHED' if 'actual_submit_time_ns' not in host else
                  'DISPATCHED_PIPE_WRITE_NOT_OBSERVED_COMPLETED' if 'submit_write_completed_time_ns' not in host else
                  'PIPE_WRITE_COMPLETION_OBSERVED')
        details.append(dict(request_id=rid,alias=public['alias'],status=status,
            visible_host_boundaries=host,host_pipeline_status=pipeline,
            elapsed_since_dispatch_without_write_completion_ns=cutoff-host['actual_submit_time_ns']
                if 'actual_submit_time_ns' in host and 'submit_write_completed_time_ns' not in host else None,
            handler_start_ns=row['handler_start_ns'],handler_end_ns=row['handler_end_ns'],
            worker_next_ready_ns=row['worker_next_ready_ns'],
            active_phase_segments=[p for p in phases if p['end_ns'] is None],
            most_recent_started_phase=None if not phases else phases[-1],
            kernel_cursor=None,kernel_cursor_status='UNKNOWN'))
    busy_owners={r['alias'] for r in active+handoff}
    future=[copy.deepcopy(r) for r in registry if r['arrival_ns']>cutoff]
    envelopes=[]
    for window in HORIZONS_NS:
        maximum=0
        for row in future:
            maximum=max(maximum,sum(row['arrival_ns']<=other['arrival_ns']<row['arrival_ns']+window for other in future))
        envelopes.append(dict(window_ns=window,maximum_future_declared_request_count=maximum))
    per_owner={}
    for alias in state['aliases']:
        own=[r for r in waiting if r['alias']==alias]
        per_owner[alias]=dict(waiting_request_ids=[r['request_id'] for r in own],
            oldest_waiting_age_ns=max((cutoff-r['arrival_ns'] for r in own),default=0),
            handler_active_request_ids=[r['request_id'] for r in active if r['alias']==alias],
            awaiting_next_ready_request_ids=[r['request_id'] for r in handoff if r['alias']==alias])
    result=dict(schema=SCHEMA+'.prefix',clock='host_perf_counter_ns',cutoff_ns=cutoff,
        declared_at_ns=declared_at,execution_context_sha256=state['execution_context_sha256'],
        visible_state_sha256=state['state_sha256'],feedback_sequence=state['last_feedback_sequence'],
        full_public_arrival_input_registry=copy.deepcopy(registry),future_declared_arrivals=future,
        future_source='declared_schedule',future_count_envelopes=envelopes,
        waiting_request_ids=[r['request_id'] for r in waiting],waiting_count=len(waiting),
        oldest_waiting_age_ns=max((cutoff-r['arrival_ns'] for r in waiting),default=0),
        visible_head_of_line_age_ns=max((cutoff-r['arrival_ns'] for r in waiting if r['alias'] in busy_owners),default=0),
        per_owner=per_owner,observed_requests=details,
        queue_reference_work_ms=None,queue_reference_work_status='UNKNOWN_NO_SOURCE_BOUND_SOLO_WORK_PROVIDER',
        GPUcursor=None,GPUcursor_status='UNKNOWN_NOT_A_DAG_CHECKPOINT',
        resource_share=None,resource_share_status='UNKNOWN',
        observed_state_unknowns=copy.deepcopy(state['unknowns']),
        reader_visible_host_state=copy.deepcopy(state),
        visible_phase_metadata_proof=[dict(request_id=rid,segment_id=segment,fields=copy.deepcopy(fields))
            for (rid,segment),fields in sorted(metadata.items())],
        visible_host_pipeline_proof=copy.deepcopy(host_pipeline))
    result['prefix_content_sha256']=identity(result)
    return result


def verify_prefix(prefix):
    """Verify host observer state hash and exact derived prefix, never GPU state."""
    if prefix.get('prefix_content_sha256')!=identity({k:v for k,v in prefix.items() if k!='prefix_content_sha256'}):
        raise ValueError('Reader-visible prefix content digest mismatch')
    state=prefix['reader_visible_host_state'];now=prefix['cutoff_ns'];context=prefix['execution_context_sha256']
    registry=prefix['full_public_arrival_input_registry'];aliases={alias:{} for alias in state['aliases']}
    for row in registry:
        if set(row)!={'request_id','alias','task','pool_id','arrival_ns','input_features'} or set(row['input_features'])!=set(INPUT_FIELDS):
            raise ValueError('Exact public original-input registry required')
    observed_state.forecast_snapshot_from_observed(state,aliases,context,now,sorted({r['task'] for r in registry}))
    if state['state_sha256']!=prefix['visible_state_sha256']:raise ValueError('Public host-observer hash differs from prefix')
    metadata={}
    for row in prefix['visible_phase_metadata_proof']:
        if set(row)!={'request_id','segment_id','fields'} or set(row['fields'])-set(VISIBLE_PHASE_FIELDS):
            raise ValueError('Allowlisted received phase metadata proof required')
        if row['segment_id'] not in state['requests'].get(row['request_id'],{}).get('phases',{}):
            raise ValueError('Metadata lacks corresponding visible phase')
        key=row['request_id'],row['segment_id']
        if key in metadata:raise ValueError('Duplicate visible phase metadata')
        metadata[key]=row['fields']
    host=prefix['visible_host_pipeline_proof']
    if set(host)-{r['request_id'] for r in registry}:raise ValueError('Foreign visible host request')
    for fields in host.values():
        if set(fields)-set(HOST_BOUNDARIES) or any(type(t) is not int or not 0<=t<=now for t in fields.values()):
            raise ValueError('Host boundary must be a cutoff-visible allowlisted timestamp')
    reconstructed=prefix_features(state,registry,cutoff=now,declared_at=prefix['declared_at_ns'],metadata=metadata,host_pipeline=host)
    if reconstructed!=prefix:raise ValueError('Derived visible prefix differs from verified host state')
    return state


def snapshot_at(plan,records,phase_start,context,cutoff):
    """Convenient independent prefix replay, used by leakage tests."""
    registry=public_registry(plan,phase_start)
    aliases=plan['resource_configuration']['allocations']
    state=observed_state.new_observed_state(aliases,context,mode='frozen',frozen_cutoff_ns=cutoff)
    feedback=observed_state.feedback_from_runtime_records(records,context,cutoff)
    state=observed_state.advance_observed_state(state,cutoff,feedback=feedback,
        arrivals=arrivals_through(registry,cutoff))
    return prefix_features(state,registry,cutoff=cutoff,declared_at=phase_start,
        metadata=received_phase_metadata(records,cutoff),host_pipeline=visible_host_pipeline(records,cutoff))


class OfflineQueueLabels:
    """Isolated future-label path. No result is fed back into prefix construction."""
    def __init__(self,registry,records,observation_end):
        self.registry=registry;self.end=observation_end;self.records={r['request_id']:r for r in records}
        if len(records)!=len(self.records) or set(self.records)!={r['request_id'] for r in registry}:
            raise ValueError('Full original cohort is required, including uncompleted requests')
        deltas=defaultdict(lambda:[0,0]);self.receipts={};self.start_receipts={}
        for public in registry:
            rid=public['request_id'];arrival=public['arrival_ns'];row=self.records[rid]
            if arrival<=observation_end: deltas[arrival][0]+=1;deltas[arrival][1]+=1
            starts=[e for e in row.get('execution_events',[]) if 'event' not in e]
            if len(starts)>1: raise ValueError('Duplicate START label')
            start=starts[0]['received_time_ns'] if starts else None
            end=row.get('response_reader_event_enqueue_time_ns')
            for boundary in (start,end):
                if boundary is not None and boundary<arrival: raise ValueError('Receipt precedes public arrival')
            if start is not None and start<=observation_end:
                deltas[start][0]-=1;self.start_receipts[rid]=start
            if end is not None and end<=observation_end:
                deltas[end][1]-=1;self.receipts[rid]=end
        self.times=[];self.waiting=[];self.outstanding=[];w=o=0
        for time,delta in sorted(deltas.items()):
            w+=delta[0];o+=delta[1]
            if min(w,o)<0:raise ValueError('Negative label queue')
            self.times.append(time);self.waiting.append(w);self.outstanding.append(o)

    def _at(self,time,series):
        index=bisect_right(self.times,time)-1
        return 0 if index<0 else series[index]

    def outcome(self,cutoff,horizon):
        stop=min(cutoff+horizon,self.end)
        points=[t for t in self.times if cutoff<t<=stop]
        initial=self._at(cutoff,self.waiting)
        recovery=next((t for t in points if self._at(t,self.waiting)==0),None) if initial else cutoff
        recovery_observed=recovery is not None
        area=0;last=cutoff;count=initial;peak=initial;peak_outstanding=self._at(cutoff,self.outstanding)
        for time in points:
            area+=count*(time-last);count=self._at(time,self.waiting);last=time
            peak=max(peak,count);peak_outstanding=max(peak_outstanding,self._at(time,self.outstanding))
        area+=count*(stop-last)
        tail=[]
        for public in self.registry:
            rid=public['request_id'];arrival=public['arrival_ns'];receipt=self.receipts.get(rid)
            pending_at_cutoff=arrival<=cutoff and (receipt is None or receipt>cutoff)
            arriving_in_horizon=cutoff<arrival<cutoff+horizon
            if not (pending_at_cutoff or arriving_in_horizon):continue
            arrived_by_end=arrival<=self.end
            tail.append(dict(request_id=rid,arrival_ns=arrival,reader_receipt_ns=receipt,
                event_observed=receipt is not None,
                arrival_to_reader_receipt_ns=None if receipt is None else receipt-arrival,
                right_censor_followup_ns=max(0,self.end-arrival) if arrived_by_end else None,
                observation_status='EVENT' if receipt is not None else 'RIGHT_CENSORED' if arrived_by_end else 'NOT_YET_ARRIVED_AT_OBSERVATION_END'))
        complete=bool(tail) and all(r['event_observed'] for r in tail)
        return dict(schema=SCHEMA+'.offline_labels',window_start_ns=cutoff,horizon_ns=horizon,
            declared_window_end_ns=cutoff+horizon,observed_until_ns=stop,full_horizon_observed=cutoff+horizon<=self.end,
            queue_clock='reader receipt visibility; not worker physical START',
            observed_waiting_peak=peak,waiting_peak_is_lower_bound=cutoff+horizon>self.end,
            observed_waiting_area_request_ns=area,end_of_observed_window_waiting_count=count,
            observed_outstanding_peak=peak_outstanding,
            recovery=dict(event_observed=recovery_observed,elapsed_or_censor_ns=(recovery if recovery_observed else stop)-cutoff,
                scope='first empty reader-visible waiting queue within requested horizon; zero when initially empty'),
            tail_cohort=tail,tail_complete=complete,
            exact_tail_p95_ms=None if not complete else float(np.percentile([r['arrival_to_reader_receipt_ns']/1e6 for r in tail],95)),
            latency_violation=None,latency_violation_status='UNKNOWN_NO_PREDECLARED_LATENCY_THRESHOLDS',
            target_threshold_tuned_from_labels=False)


def run(review_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    review=read(review_path)
    if review.get('status')!='PASS' or review.get('training_only') is not True:
        raise ValueError('Passed whole57 training review required')
    refs=[receipt(review_path),review['evidence']['raw.json'],review['evidence']['plan.json'],
          receipt(__file__),receipt(observed_state.__file__)]
    for ref in refs:
        if receipt(ref['path'])!=ref:raise ValueError('Changed training source')
    raw=read(refs[1]['path']);plan=read(refs[2]['path'])
    if raw.get('split')!='training' or raw.get('training_only') is not True or raw['execution_context_sha256']!=plan['execution_context_sha256']:
        raise ValueError('Exact-context training raw and plan required')
    records=raw['request_records'];begin=raw['phase_start_ns'];end=raw['observation_end_ns'];context=raw['execution_context_sha256']
    registry=public_registry(plan,begin)
    if len(registry)!=57 or len(records)!=57 or raw['planned_public_cohort']!=plan['trace']:
        raise ValueError('Complete unchanged whole57 public cohort required')
    by_id={r['request_id']:r for r in registry}
    for record in records:
        public=by_id[record['request_id']]
        if any(record[k]!=public[v] for k,v in [('model','alias'),('task','task'),('pool_id','pool_id'),('scheduled_time_ns','arrival_ns')]):
            raise ValueError('Runtime record changed public route/input/arrival')
    all_feedback=observed_state.feedback_from_runtime_records(records,context,end)
    cutoffs=sorted({begin,end,*[r['arrival_ns'] for r in registry if r['arrival_ns']<=end],
        *[e['received_time_ns'] for e in all_feedback],*range(begin,end+1,1_000_000_000)})
    labels=OfflineQueueLabels(registry,records,end)
    run_id=plan['training_id']+':'+plan['domain']+':'+plan['route_id']+':'+refs[1]['sha256']
    group_id=identity(dict(recipe=plan['recipe'],input_family=plan['input_selection']))
    state=observed_state.new_observed_state(plan['resource_configuration']['allocations'],context)
    output.mkdir(parents=True);counts=Counter();maximum_waiting=0;source_rows=[]
    with gzip.open(output/'prefix-features.jsonl.gz','wt',encoding='utf-8') as fs, gzip.open(output/'offline-risk-labels.jsonl.gz','wt',encoding='utf-8') as ls:
        for cutoff in cutoffs:
            feedback=[e for e in all_feedback if e['sequence']>state['last_feedback_sequence'] and e['received_time_ns']<=cutoff]
            state=observed_state.advance_observed_state(state,cutoff,feedback=feedback,
                arrivals=arrivals_through(registry,cutoff,state['last_arrival_sequence']))
            features=prefix_features(state,registry,cutoff=cutoff,declared_at=begin,
                metadata=received_phase_metadata(records,cutoff),host_pipeline=visible_host_pipeline(records,cutoff))
            row_id=run_id+':'+str(cutoff-begin)
            envelope=dict(id=row_id,run_id=run_id,group_id=group_id,split='training',kind='measured',domain=plan['domain'],route_id=plan['route_id'],execution_context_sha256=context,decision_time_ns=cutoff)
            fs.write(json.dumps(dict(**envelope,prefix_features=features),separators=(',',':'))+'\n')
            maximum_waiting=max(maximum_waiting,features['waiting_count']);counts['prefix_snapshots']+=1
            for horizon in HORIZONS_NS:
                outcome=labels.outcome(cutoff,horizon)
                ls.write(json.dumps(dict(**envelope,offline_labels=outcome),separators=(',',':'))+'\n')
                counts['label_rows']+=1;counts['full_horizon_labels']+=outcome['full_horizon_observed']
                counts['recovery_right_censored']+=not outcome['recovery']['event_observed']
                counts['tail_incomplete']+=bool(outcome['tail_cohort']) and not outcome['tail_complete']
            source_rows.append(dict(id=row_id,run_id=run_id,group_id=group_id))
    after=[receipt(ref['path']) for ref in refs]
    if after!=refs:raise ValueError('Training source bytes changed during extraction')
    manifest=dict(schema=SCHEMA+'.whole_run_split_manifest',runs=[dict(run_id=run_id,group_id=group_id,domain=plan['domain'],route_id=plan['route_id'],source_raw=refs[1],row_count=len(source_rows))],
        required_split_unit='whole run_id; preserve recipe/input group when reserving independent family holdout',
        request_random_split_allowed=False,window_random_split_allowed=False,
        minimum_training_runs_for_existing_risk_fitter=3,minimum_training_groups_for_existing_risk_fitter=2,
        model_fit_status='NOT_FIT_INSUFFICIENT_INDEPENDENT_RUNS_AND_NO_PREDECLARED_LATENCY_THRESHOLD')
    (output/'split-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(schema=SCHEMA,status='ACTUAL_READER_OBSERVATIONS_TO_ISOLATED_RISK_ROWS_READY_NO_MODEL',
        source_receipts_before=refs,source_receipts_after=after,counts=dict(counts),maximum_visible_waiting_count=maximum_waiting,
        run_id=run_id,group_id=group_id,original_cohort=57,complete_public_future_trace_preserved=True,
        horizons_ns=list(HORIZONS_NS),cutoff_rule='Every visible reader feedback receipt, declared arrival, and fixed one-second grid through observation_end; ties atomic',
        features=receipt(output/'prefix-features.jsonl.gz'),labels=receipt(output/'offline-risk-labels.jsonl.gz'),split_manifest=receipt(output/'split-manifest.json'),
        visible_interface=observed_state.STATE_SCHEMA,training_only=True,GPU_started=False,model_fitted=False,
        validation_labels_used=False,rolling_forecast_implemented=False,DAG_checkpoint_implemented=False,
        original_six_term_utility_changed=False,original_stability_rule_changed=False,
        limitations=['Only one whole training run: correlated request/event windows cannot create independent captures.',
            'Waiting means no reader-visible START; delayed receipts can overstate physical waiting.',
            'Public input size/hash are known from plan; token features appear only after received semantic events.',
            'No source-bound solo-work provider, physical GPU cursor, ready DAG, throughput or resource share is fabricated.',
            'Recovery is first empty visible waiting queue within each label horizon, not guaranteed sustained stability.',
            'This v30 prefix schema is not directly accepted by the old recovery_risk v2 fitter; requires an explicit frozen feature projection and target protocol.'])
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.review,args.output)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],maximum_visible_waiting_count=result['maximum_visible_waiting_count']),indent=2))
