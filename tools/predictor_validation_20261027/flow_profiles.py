"""Strict, source-bound continuous native-flow semantic training observations.

Training provenance is derived only from a prospectively frozen training plan;
old validation flows cannot become training examples. All actual measurement
handlers stay together, preserving cross-request and multiway interference.
Semantic walls remain mixed host/transfer/launch/wait intervals, not kernels.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.energy import integrate_power
from gsdeploy.resources import validate_resources
from .background_profiles import ALLOWED, RESIDENTS, _events, checked_receipt, ns, positive, receipt
from .flow_validation import validate_plan
from .profiling_analysis import digest
from .semantic_profiles import parse_semantic_request

SCHEMA='gsdeploy.observed_continuous_flow_execution.v1'


def _read(ref):return json.loads(checked_receipt(ref).read_text())


def _request(row,ready,context,*,split,scope_id,left,right):
    rid=row['request_id'];alias=row['model'];family=RESIDENTS[alias];task=row['task']
    if row.get('status')!='completed' or task not in ALLOWED[family] or task not in ready[alias]['tasks']:
        raise ValueError('Complete supported actual request required')
    response=row['worker_response']
    if (response.get('request_id')!=rid or response.get('task')!=task or response.get('task_type')!=task
            or row.get('sample_id',row['pool_id'])!=row['pool_id']):
        raise ValueError('Response request/task/input identity mismatch')
    start,end=ns(response['start_perf_ns']),ns(response['end_perf_ns'])
    arrival,submit,completion=(ns(row[k]) for k in ('scheduled_time_ns','actual_submit_time_ns','completion_time_ns'))
    if start!=row.get('observed_worker_start_time_ns') or not left<=arrival<=submit<=start<end<=completion<=right:
        raise ValueError('Actual scheduled/submit/START/END/parent chronology mismatch')
    if row.get('enqueued_time_ns') is not None and not arrival<=ns(row['enqueued_time_ns'])<=submit:
        raise ValueError('Actual enqueue chronology mismatch')
    if row.get('submit_write_completed_time_ns') is not None and not submit<=ns(row['submit_write_completed_time_ns'])<=completion:
        raise ValueError('Actual write completion chronology mismatch')
    stages=response.get('observed_stages',[])
    if not stages:raise ValueError('Actual observed native stages required')
    cursor=start;seen=set()
    for stage in stages:
        a,b=ns(stage['start_ns']),ns(stage['end_ns']);key=stage['id'],stage['kind']
        if key in seen or stage['kind'] not in {'gpu','host','sync','cleanup'} or not cursor<=a<b<=end:
            raise ValueError('Actual native stages must be distinct, ordered and nonoverlapping')
        seen.add(key);cursor=b
    _events(row,response,ready[alias]['pid'])
    # _events checks every required edge. Also reject unmatched extra events,
    # wrong clocks or foreign request observations instead of silently dropping.
    observations=stages+response.get('semantic_forward_observations',{}).get('records',[])
    expected={(s['id'],s['kind'],edge,s[edge+'_ns']) for s in observations for edge in ('start','end')}
    actual=[];previous=-1;starts=0
    for event in row.get('execution_events',[]):
        timestamp=ns(event['perf_ns'])
        if event.get('request_id')!=rid or timestamp<previous or not start<=timestamp<=end:
            raise ValueError('Actual execution events have foreign request or invalid chronological order')
        previous=timestamp
        received=event.get('received_time_ns')
        if received is not None and not timestamp<=ns(received)<=completion:
            raise ValueError('Actual event parent receipt chronology mismatch')
        if 'event' not in event:
            starts+=1
            if event.get('pid')!=ready[alias]['pid'] or timestamp!=start:
                raise ValueError('Unmatched actual START event')
        else:actual.append((event.get('segment_id'),event.get('kind'),event.get('event'),timestamp))
    if starts!=1 or len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Unmatched or duplicate actual START/PHASE event')
    semantic=parse_semantic_request({'request_id':rid,'alias':alias,'family':family,'task':task,
        'pool_id':row['pool_id'],'handler_start_ns':start,'handler_end_ns':end},response)
    cursor=start
    for segment in semantic['segments']:
        if segment['start_ns']!=cursor or not cursor<segment['end_ns']<=end:
            raise ValueError('Continuous semantic handler coverage required')
        segment.update(kind='mixed',demand={'sm':None,'dram':None,'l2':None},
            observation_basis='Actual START, language-forward and post-sync wall boundaries; no kernel-only or counter attribution')
        cursor=segment['end_ns']
    if cursor!=end:raise ValueError('Semantic handler end not covered')
    return {**semantic,'request_id':scope_id+':'+rid,'native_request_id':rid,'split':split,
        'split_basis':'Frozen prospective plan.phase; never inferred from outcome',
        'arrival_ns':arrival,'actual_submit_ns':submit,'parent_receipt_ns':completion,
        'native_stages':copy.deepcopy(stages),'execution':copy.deepcopy(context['execution']),
        'start_basis':'observed_worker_START','source_response_in_training_features':False}


def map_flow_collection(raw_path,*,allow_validation_diagnostics=False):
    """Return {traces:[one semantic continuous cohort], mapping_report, resource_evidence}.

    Explicit validation diagnostics keep split=validation and not_for_training;
    default mapping rejects all non-training phases before touching outcomes.
    Priming/warmup are audited as resource history and excluded from fit traces.
    """
    raw_path=Path(raw_path).resolve();raw=json.loads(raw_path.read_text())
    if (raw.get('schema')!='gsdeploy.native_flow_collection.v1' or raw.get('kind')!='measured'
            or raw.get('backend')!='native' or raw.get('status')!='completed'):
        raise ValueError('Complete measured native full-flow collection required')
    plan=validate_plan(_read(raw['plan']));phase=plan.get('phase')
    if phase=='predictor_training':split='training'
    elif allow_validation_diagnostics and phase=='predictor_validation':split='validation'
    else:raise ValueError('Frozen plan.phase must be predictor_training; validation cannot enter training')
    if plan.get('split',split)!=split:raise ValueError('Frozen plan phase/split disagreement')
    if any(raw.get(k) for k in ('errors','runtime_errors','telemetry_errors','qualification_issues')) or raw.get('collection_complete') is not True:
        raise ValueError('Error-free, complete native collection protocol required')
    context=_read(raw['context'])
    if raw['context']!=plan.get('context') or context.get('residents')!=RESIDENTS:
        raise ValueError('Frozen context and exact residency mismatch')
    if ({a:v['family'] for a,v in plan['aliases'].items()}!=RESIDENTS or raw.get('replica_to_family')!=RESIDENTS
            or len(raw.get('resident_aliases',[]))!=len(RESIDENTS) or set(raw['resident_aliases'])!=set(RESIDENTS)):
        raise ValueError('Exact frozen resident aliases required')
    snapshots=context.get('execution_sources',[])
    if not snapshots or context.get('sources')!=[s['frozen'] for s in snapshots]:
        raise ValueError('Explicit consistent execution source snapshot required')
    if len({s['original_path'] for s in snapshots})!=len(snapshots):raise ValueError('Duplicate execution snapshot paths')
    for source in snapshots:checked_receipt(source['frozen'])
    content={k:v for k,v in context.items() if k not in {'sources','execution_sources'}}
    content['source_content']=[{'original_path':s['original_path'],'sha256':s['frozen']['sha256']} for s in snapshots]
    identity=digest(content)
    if identity!=raw.get('execution_context_sha256') or identity!=plan.get('execution_context_sha256'):
        raise ValueError('Snapshot execution identity mismatch')
    candidate=next((c for c in plan['candidates'] if c['id']==raw.get('candidate_id')),None)
    if candidate is None or candidate['execution_context_sha256']!=identity:
        raise ValueError('Frozen candidate execution identity mismatch')
    resources=validate_resources(candidate['resource_configuration'],RESIDENTS)
    if raw.get('resource_configuration')!=resources:raise ValueError('Actual resource configuration differs from frozen candidate')
    context_resources=context.get('resource_configuration',context.get('execution',{}).get('resource_configuration'))
    if ((context_resources is not None and context_resources!=resources)
            or resources['mode']!='shared' and context_resources!=resources):
        raise ValueError('Nonshared resources require a matching distinct execution-context resource binding')
    refs=[receipt(raw_path),raw['plan'],raw['context']]
    for key in ('profile_plan','catalog','runner_source'):
        checked_receipt(plan[key]);refs.append(plan[key])
    profile_plan=_read(plan['profile_plan'])
    if profile_plan.get('execution_context_sha256')!=identity:raise ValueError('Public profile input domain execution mismatch')
    if profile_plan.get('plan_sha256')!=digest({k:v for k,v in profile_plan.items() if k!='plan_sha256'}):
        raise ValueError('Public profile input plan self hash mismatch')
    public={(r['task'],r['pool_id']) for r in profile_plan.get('input_assignments',[]) if r.get('view')=='known_input'}
    if not public:raise ValueError('Explicit known-input training public domain required')
    invocation_path=raw_path.parent/'invocation.json';invocation=json.loads(invocation_path.read_text())
    if (invocation.get('plan')!=raw['plan'] or invocation.get('candidate_id')!=candidate['id']
            or invocation.get('runner')!=plan['runner_source']):
        raise ValueError('Prelaunch invocation differs from frozen plan/candidate/runner')
    refs.append(receipt(invocation_path))
    startup,ready_time,left,deadline,right=(ns(raw[k]) for k in
        ('startup_begin_ns','startup_end_ns','phase_start_ns','deadline_ns','observation_end_ns'))
    if not startup<=ready_time<=left<deadline<=right or deadline-left!=round(plan['window_seconds']*1e9):
        raise ValueError('Actual startup/ready/measurement/drain chronology mismatch')
    # The old frozen runner lacks per-row split. A prospectively timestamped
    # source-bound plan is the sole permitted derivation for new training rows.
    if split=='training' and ('created_ns' not in plan or ns(plan['created_ns'])>=startup):
        raise ValueError('Training plan must be explicitly frozen before native startup')
    ready=raw.get('ready',{})
    if set(ready)!=set(RESIDENTS) or len({r.get('pid') for r in ready.values()})!=len(RESIDENTS):
        raise ValueError('Distinct actual READY PID for every resident required')
    for alias,row in ready.items():
        family=RESIDENTS[alias]
        if (type(row.get('pid')) is not int or row['pid']<=0 or row.get('worker')!=family
                or row.get('clock')!='time.perf_counter_ns'
                or row.get('spec_sha256')!=context['worker_bindings'][family]['spec_sha256']
                or not set(row.get('tasks',[]))<=ALLOWED[family]):
            raise ValueError('READY PID/clock/spec/task identity mismatch')
    shutdown=raw.get('shutdown',{})
    if set(shutdown)!=set(RESIDENTS) or any(v.get('exit_code')!=0 or v.get('forced') is not False or v.get('bye_received') is not True for v in shutdown.values()):
        raise ValueError('Complete clean shutdown/BYE evidence required')
    samples=raw['board_samples'];previous=-1;maxgap=0;pid_alias={str(v['pid']):a for a,v in ready.items()}
    for sample in samples:
        timestamp=ns(sample['time_ns'])
        if timestamp<=previous:raise ValueError('Strictly increasing actual NVML timestamps required')
        if previous>=0:maxgap=max(maxgap,timestamp-previous)
        previous=timestamp;positive(sample['watts']);positive(sample['owned_used_mib'])
        processes={str(k):positive(v) for k,v in sample['processes_mib'].items()}
        if (sample.get('foreign_pids') or not set(processes)<=set(pid_alias)
                or set(sample.get('owned_aliases',[]))!={pid_alias[p] for p in processes}
                or abs(sum(processes.values())-sample['owned_used_mib'])>1e-6):
            raise ValueError('NVML owned PID/alias/sum or foreign-process mismatch')
        if timestamp>=ready_time and set(processes)!=set(pid_alias):raise ValueError('All resident NVML telemetry required after READY')
    if len(samples)<2 or maxgap>2_000_000_000 or not samples[0]['time_ns']<=startup<=right<=samples[-1]['time_ns']:
        raise ValueError('Full startup-through-drain telemetry bracketing and unchanged gap limit required')
    trace_hash=digest(plan['trace'])
    if raw.get('trace_sha256',raw.get('cohort_sha256'))!=trace_hash:raise ValueError('Actual offered trace SHA mismatch')
    records=raw['request_records'];by_id={r['request_id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!={r['request_id'] for r in plan['trace']}:
        raise ValueError('Full offered cohort has duplicate/missing/foreign requests')
    source=receipt(raw_path);scope_id='flow:'+source['sha256']
    requests=[]
    for request in plan['trace']:
        row=by_id[request['request_id']];task,pool=request['task'],request['pool_id']
        if ((task,pool) not in public or (row['task'],row['pool_id'],row['model'],row['scheduled_time_ns'])!=
                (task,pool,candidate['routes'][request['request_id']],left+request['offset_ns'])
                or row.get('phase')!='measurement' or row.get('split',split)!=split):
            raise ValueError('Actual measurement route/input/arrival/split differs from frozen training domain')
        requests.append(_request(row,ready,context,split=split,scope_id=scope_id,left=left,right=right))
    if any(r['parent_receipt_ns']>deadline+round(plan['drain_seconds']*1e9) for r in requests):
        raise ValueError('Actual parent completion exceeds declared bounded drain')
    for alias in RESIDENTS:
        ordered=[r for r in requests if r['alias']==alias]
        if ordered!=sorted(ordered,key=lambda r:r['handler_start_ns']) or any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('Same-alias handlers must be FIFO and nonoverlapping in continuous cohort')
    history=[];history_ids=set(by_id)
    for key,phase_name in (('priming_request_records','priming'),('warmup_request_records','warmup')):
        for row in raw.get(key,[]):
            if row['request_id'] in history_ids or row.get('phase')!=phase_name:
                raise ValueError('Duplicate or mislabelled prephase history request')
            history_ids.add(row['request_id'])
            item=_request(row,ready,context,split='excluded_prephase_history',scope_id=scope_id,left=startup,right=left)
            item['not_for_service_training']=True;history.append(item)
    all_rows=history+requests
    for alias in RESIDENTS:
        rows=sorted((r for r in all_rows if r['alias']==alias),key=lambda r:r['handler_start_ns'])
        if any(a['handler_end_ns']>b['handler_start_ns'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Same-alias actual history handlers overlap')
    before=[s for s in samples if s['time_ns']<=left];after=[s for s in samples if s['time_ns']>=right]
    selected=[before[-1]]+[s for s in samples if left<s['time_ns']<right]+[after[0]]
    refs.extend(receipt(Path(__file__).with_name(name)) for name in ('flow_profiles.py','background_profiles.py','semantic_profiles.py','profiling_analysis.py','flow_validation.py'))
    trace={'schema':SCHEMA,'kind':'measured','backend':'native','id':scope_id,'split':split,
        'not_for_training':split!='training','split_basis':'Prospectively frozen plan.phase',
        'clock':'host_monotonic_ns','execution_context_sha256':identity,'sources':refs,
        'observation_window_ns':{'start':left,'end':right},'admission_enabled':resources['mode']=='request_admission',
        'resource_configuration':resources,'resource_configuration_sha256':digest(resources),'requests':requests,
        'power_samples':[{'time_ns':s['time_ns'],'watts':s['watts']} for s in selected],
        'memory_samples':[{k:s[k] for k in ('time_ns','owned_used_mib','owned_aliases','foreign_pids')} for s in selected],
        'scope':'One intact continuous measurement cohort including full drained handlers; mixed semantic walls, no artificial per-request solo trials'}
    resource_evidence={'sources':refs,'split':split,'execution_context_sha256':identity,'resource_configuration':resources,
        'resource_configuration_sha256':digest(resources),
        'measurement_window_ns':{'start':left,'deadline':deadline,'observation_end':right},
        'startup_window_ns':{'start':startup,'ready':ready_time,'end':left},
        'startup_joules':integrate_power(samples,startup,left),'measurement_joules':integrate_power(samples,left,deadline),
        'drain_joules':integrate_power(samples,deadline,right) if deadline<right else 0.,
        'raw_all_sampled_owned_peak_mib':max(s['owned_used_mib'] for s in samples),
        'raw_prephase_sampled_owned_peak_mib':max(s['owned_used_mib'] for s in samples if s['time_ns']<=left),
        'raw_measurement_and_drain_sampled_owned_peak_mib':max(s['owned_used_mib'] for s in selected),
        'safe_factor_applied':False,'capacity_bound':False,'stage_history':history+requests,
        'board_samples':copy.deepcopy(samples),'scope':'Actual full startup/prime/warmup/measurement/drain history; drain energy separate from original measurement energy'}
    return {'traces':[trace],'resource_evidence':resource_evidence,'mapping_report':{
        'trace_count':1,'continuous_cohort':True,'request_count':len(requests),'prephase_history_requests_excluded':len(history),
        'split':split,'not_for_training':split!='training','phase':phase,'candidate_id':candidate['id'],'sources':refs,
        'snapshot_verified':True,'actual_START_PHASE_events_verified':True,'full_offered_cohort_verified':True,
        'same_alias_FIFO_nonoverlap_verified':True,'all_resident_telemetry_after_READY_verified':True,
        'max_telemetry_gap_ns':maxgap,'kernel_evidence_available':False,'GPU_started':False}}


def flow_semantic_traces(raw_path,**kwargs):
    mapped=map_flow_collection(raw_path,**kwargs)
    return mapped['traces'],mapped['mapping_report']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--allow-validation-diagnostics',action='store_true')
    a=p.parse_args();value=map_flow_collection(a.raw,allow_validation_diagnostics=a.allow_validation_diagnostics)
    with a.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
