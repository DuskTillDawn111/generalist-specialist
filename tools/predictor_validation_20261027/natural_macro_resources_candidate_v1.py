"""New-domain training-only board resources; whole-capture replay diagnostics.

Actual START-to-READY phase/input states are covariates, never SM activity.
All residents retain memory. Only polls wholly inside a state label that state;
power training uses adjacent poll pairs wholly inside the same state. Missing
states or poll support remain UNKNOWN. No native/accuracy qualification.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from gsdeploy.energy import integrate_power
from .native_collect import context_identity, receipt, save

SCHEMA='gsdeploy.natural_macro_resource_training_candidate.v1'
TABLE_SCHEMA='gsdeploy.natural_macro_resource_capture_table.v1'
LAG_LIMIT_NS=2_000_000_000
MIN_CAPTURES=2


def require(condition,message):
    if not condition:raise ValueError(message)


def checked(source):
    require(receipt(source['path'])==source,'Changed immutable resource source')
    return json.loads(Path(source['path']).read_text())


def finite(value):return type(value) in (float,int) and math.isfinite(value) and value>=0


def key(members,resident_owners):
    require(type(resident_owners) in (tuple,list) and len(set(resident_owners))==len(resident_owners)
        and bool(resident_owners),'Complete distinct resident roster required')
    result=[];seen=set()
    for member in members:
        owner=member['owner']
        require(owner in resident_owners and owner not in seen,'Distinct resident handler identity required')
        seen.add(owner)
        values={k:member[k] for k in ('owner','task','pool_id','response_phase')}
        require(all(type(v) is str and v for v in values.values()),'Explicit actual phase/public input state required')
        result.append(values)
    result.sort(key=lambda r:r['owner'])
    payload=dict(resident_owners=sorted(resident_owners),actual_START_to_READY_handlers=result)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),result


def poll_bracket(sample):
    at,duration=sample.get('time_ns'),sample.get('query_duration_ns')
    require(type(at) is int and at>=0 and type(duration) is int and duration>=0,
        'Actual midpoint and complete nonnegative poll duration required')
    start=at-duration//2
    require(start>=0,'Poll bracket must be nonnegative')
    return start,start+duration


def priming_identity(raw,context):
    priming=raw['priming_request_records']
    require(len(priming)==18 and all(r['status']=='completed' for r in priming),
        'Exactly eighteen actual successful new-domain priming requests required')
    rows=[];declared={r['pool_id']:r for r in context['declared_inputs']}
    for row in priming:
        entry={k:row[k] for k in ('model','task','pool_id')}
        public=declared[row['pool_id']]
        require(public['task']==row['task'],'Priming public input task differs from exact declaration')
        entry['exact_input_sha256']=public['exact_input_sha256']
        require(all(type(v) is str and v for v in entry.values()),'Actual priming input/protocol hashes required')
        rows.append(entry)
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def observed_table(raw,macros,resident_owners):
    """No I/O; caller must authenticate raw/review/macro sources first."""
    left,right=raw['observation_start_ns'],raw['observation_end_ns']
    require(type(left) is int and type(right) is int and 0<=left<right,'Original observation window required')
    samples=raw['board_samples'];stamps=[s['time_ns'] for s in samples]
    require(len(stamps)>=2 and stamps==sorted(set(stamps)),'Strictly ordered complete board polls required')
    require(stamps[0]<=left<right<=stamps[-1],'Polls must bracket original observation window')
    boundaries={r['boundary']:r['time_ns'] for r in raw['lifecycle_boundaries']}
    startup=boundaries['startup_begin']
    require(stamps[0]<=startup<left and boundaries['measurement_and_drain_end']==right,
        'Actual startup-to-observation and final-drain boundaries required')
    monitoring_end=boundaries['monitoring_end'];shutdown=boundaries['shutdown_begin']
    require(right<=stamps[-1]<=monitoring_end<shutdown
        and raw.get('shutdown_memory_sampled') is False,
        'Complete monitored startup-through-drain scope must end before worker close')
    monitor_samples=[s for s in samples if s['time_ns']<=monitoring_end]
    require(len(monitor_samples)==len(samples),'Future close/shutdown polls cannot enter owned peak')
    intervals=[];cuts={left,right}
    request_rows={r['request_id']:r for r in raw['request_records']}
    require(len(request_rows)==len(raw['request_records'])==len(macros['requests']),
        'Every offered training request must enter macro resource table')
    for request in macros['requests']:
        row=request_rows[request['request_id']]
        own=request['macros']
        require(bool(own) and own[0]['left_ns']==row['observed_worker_start_time_ns']
            and own[-1]['right_ns']==row['previous_handler_next_request_ready']['perf_ns']
            and all(a['right_ns']==b['left_ns'] for a,b in zip(own,own[1:])),
            'Complete disjoint actual START-to-READY macro coverage required')
        for macro in own:
            a,b=macro['left_ns'],macro['right_ns']
            require(left<=a<b<=right,'Actual macro must lie within full observation')
            member=dict(owner=row['model'],task=row['task'],pool_id=row['pool_id'],response_phase=macro['semantic_phase'])
            intervals.append((a,b,member));cuts.update((a,b))
    slices=[];ordered=sorted(cuts)
    for a,b in zip(ordered,ordered[1:]):
        state,members=key([m for x,y,m in intervals if x<=a and b<=y],resident_owners)
        slices.append(dict(left_ns=a,right_ns=b,duration_ns=b-a,state_key=state,members=members,
            resident_owners=list(resident_owners),actual_SM_work_known=False,
            inactive_resident_memory_is_zero=False,interior_poll_ordinals=[],power_pair_ordinals=[],
            power_training_duration_ns=0,power_training_joules=0.,sampled_owned_peak_mib=None,
            observed_piecewise_linear_board_joules=integrate_power(samples,a,b)))
    rows=[];starts=[s['left_ns'] for s in slices]
    for i,sample in enumerate(samples):
        begin,end=poll_bracket(sample);at=sample['time_ns']
        require(finite(sample.get('watts')) and finite(sample.get('owned_used_mib'))
            and not sample.get('foreign_pids') and not sample.get('unresolved_nvml_pids')
            and not sample.get('unresolved_ownership_reasons'),'Finite entirely-owned resource polls required')
        if left<=at<=right:
            require(set(sample['owned_aliases'])==set(resident_owners),'All resident memory required including inactive owners')
        index=bisect_right(starts,at)-1
        safe=(0<=index<len(slices) and slices[index]['left_ns']<=begin and end<slices[index]['right_ns'])
        row=dict(raw_poll_ordinal=i,midpoint_ns=at,poll_begin_ns=begin,poll_end_ns=end,
            query_duration_ns=sample['query_duration_ns'],watts=sample['watts'],
            owned_used_mib=sample['owned_used_mib'],owned_aliases=sample['owned_aliases'],raw_phase=sample['phase'],
            state_key=slices[index]['state_key'] if safe else None,
            state_slice_ordinal=index if safe else None,
            state_label_status='POLL_BRACKET_INSIDE_ONE_ACTUAL_STATE' if safe else 'UNKNOWN_STATE_OR_OUTSIDE_OBSERVATION')
        rows.append(row)
        if safe:
            target=slices[index];target['interior_poll_ordinals'].append(i)
            peak=target['sampled_owned_peak_mib']
            target['sampled_owned_peak_mib']=sample['owned_used_mib'] if peak is None else max(peak,sample['owned_used_mib'])
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        if (a['state_slice_ordinal'] is None or a['state_slice_ordinal']!=b['state_slice_ordinal']
                or b['midpoint_ns']-a['midpoint_ns']>LAG_LIMIT_NS):continue
        target=slices[a['state_slice_ordinal']];duration=b['midpoint_ns']-a['midpoint_ns']
        target['power_pair_ordinals'].append([i,i+1]);target['power_training_duration_ns']+=duration
        target['power_training_joules']+=(a['watts']+b['watts'])/2*duration/1e9
    before=[s['owned_used_mib'] for s in samples if startup<=s['time_ns']<=left]
    startup_j=integrate_power(samples,startup,left)
    observation_j=integrate_power(samples,left,right)
    require(bool(before),'Nonempty premeasurement owned memory support required')
    return dict(schema=TABLE_SCHEMA,resident_owners=list(resident_owners),
        observation_window_ns=[left,right],observation_duration_ns=right-left,
        startup_window_ns=[startup,left],priming_requests=len(raw['priming_request_records']),
        measured_requests=len(request_rows),macro_nodes=len(intervals),slices=slices,polls=rows,
        startup_joules=startup_j,premeasurement_owned_sampled_peak_mib=max(before),
        observation_joules=observation_j,scoring_measurement_energy_joules=None,
        scoring_energy_unknown_reason='PROFILE_ONLY_HAS_OBSERVATION_AND_COMBINED_MEASUREMENT_DRAIN_END_NO_SCORING_DEADLINE',
        raw_monitor_owned_sampled_peak_mib=max(s['owned_used_mib'] for s in monitor_samples),
        owned_peak_monitor_window_ns=[stamps[0],monitoring_end],
        raw_monitor_sample_count=len(samples),polls_without_state_label=sum(p['state_key'] is None for p in rows),
        state_power_training_duration_ns=sum(s['power_training_duration_ns'] for s in slices),
        observation_power_training_uncovered_duration_ns=right-left-sum(s['power_training_duration_ns'] for s in slices),
        piecewise_linear_power_is_sampling_assumption=True,poll_brackets_are_not_GPU_counter_windows=True,
        sampled_peak_is_continuous_peak=False,shutdown_memory_observed=False)


def fit_tables(tables):
    require(bool(tables),'Nonempty new-domain training capture tables required')
    captures=[table['capture_sha256'] for table in tables]
    require(len(captures)==len(set(captures)),'Repeated capture is not independent resource support')
    require(len({t['execution_context_sha256'] for t in tables})==1,'Exact new natural execution identity required')
    roster=tables[0]['resident_owners'];priming=tables[0]['priming_content_identity']
    require(all(t['resident_owners']==roster and t['priming_content_identity']==priming
        and t['priming_requests']==18 for t in tables),'Same four residents and exact eighteen priming contents required')
    groups=defaultdict(dict)
    for table in tables:
        local=defaultdict(lambda:dict(power_joules=0.,power_duration_ns=0,owned_peak_mib=None))
        for row in table['slices']:
            x=local[row['state_key']];x['members']=row['members']
            x['power_joules']+=row['power_training_joules'];x['power_duration_ns']+=row['power_training_duration_ns']
            peak=row['sampled_owned_peak_mib']
            if peak is not None:x['owned_peak_mib']=peak if x['owned_peak_mib'] is None else max(peak,x['owned_peak_mib'])
        for k,v in local.items():groups[k][table['capture_sha256']]=v
    states={}
    for k,rows in sorted(groups.items()):
        power=[v['power_joules']/(v['power_duration_ns']/1e9) for v in rows.values() if v['power_duration_ns']>0]
        memory=[v['owned_peak_mib'] for v in rows.values() if v['owned_peak_mib'] is not None]
        states[k]=dict(members=next(iter(rows.values()))['members'],per_capture=rows,
            power_capture_count=len(power),memory_capture_count=len(memory),
            power_known=len(power)>=MIN_CAPTURES,memory_known=len(memory)>=MIN_CAPTURES,
            mean=dict(watts=statistics.mean(power) if len(power)>=MIN_CAPTURES else None,
                peak_memory_mib=statistics.mean(memory) if len(memory)>=MIN_CAPTURES else None),
            max=dict(watts=max(power) if len(power)>=MIN_CAPTURES else None,
                peak_memory_mib=max(memory) if len(memory)>=MIN_CAPTURES else None))
    startup=[t['startup_joules'] for t in tables]
    peaks=[t['premeasurement_owned_sampled_peak_mib'] for t in tables]
    startup_known=len(startup)>=MIN_CAPTURES and all(v is not None for v in startup)
    return dict(schema=SCHEMA,execution_context_sha256=tables[0]['execution_context_sha256'],
        capture_sha256s=captures,resident_owners=roster,priming_content_identity=priming,states=states,
        minimum_independent_captures=MIN_CAPTURES,equal_capture_means=True,
        startup=dict(known=startup_known,joules=statistics.mean(startup) if startup_known else None,
            premeasurement_owned_sampled_peak_mib=statistics.mean(peaks) if startup_known else None,
            priming_requests=18,energy_includes_priming=True,startup_amortization_seconds=3600),
        empirical_full_raw_monitor_peaks=[dict(capture_sha256=t['capture_sha256'],peak_mib=t['raw_monitor_owned_sampled_peak_mib']) for t in tables],
        state_scope='Exact owner/task/public-input and actual START-to-READY response phase; not SM activity',
        notice_processing_and_physical_context_are_not_actual_handler_work=True,
        within_phase_forward_index_shape_history_omitted_is_assumption=True,
        core_provider_eligible=False,physical_accuracy_validated=False,capacity_bound=False,
        memory_safe_factor_applied=False,validation_labels_read=False,GPU_started=False)


class ResourceTrainingProvider:
    """Source-bound hypothesis interface; explicit opt-in, never native eligible."""
    def __init__(self,model_path,target_context,*,allow_training_resource_hypothesis=False,scenario='mean'):
        require(allow_training_resource_hypothesis is True,'Explicit new-domain training resource hypothesis opt-in required')
        require(scenario in ('mean','max'),'Explicit empirical resource scenario required')
        source=receipt(model_path);doc=checked(source)
        require(doc['schema']==SCHEMA and context_identity(target_context)==doc['execution_context_sha256'],
            'Exact new natural execution context required; old worker/domain cannot transfer')
        for r in doc['sources']:checked(r) if r['path'].endswith('.json') else require(receipt(r['path'])==r,'Resource implementation changed')
        tables=[checked(r) for r in doc['training_tables']]
        for table in tables:
            from .macro_joint_response_fit_v1 import verified_document
            macros,closure=verified_document(table['training_macros']['path'])
            require(all(r in doc['sources'] for r in closure),
                'Provider must retain complete rederived raw/review/macro source closure')
            raw=checked(table['raw']);context=checked(table['context'])
            review=checked(macros['sources'][0])
            require(table['raw']==review['inputs']['raw.json']
                and table['context']==review['inputs']['context.json'],
                'Actual table raw/context must bind rederived macro review')
            actual=observed_table(raw,macros,tuple(doc['resident_owners']))
            require(all(table.get(k)==v for k,v in actual.items())
                and table['capture_sha256']==table['raw']['sha256']
                and table['execution_context_sha256']==raw['execution_context_sha256']==context_identity(context)
                and table['priming_content_identity']==priming_identity(raw,context),
                'Resource table must rederive from exact raw polls and macros')
        expected=fit_tables(tables)
        require(all(doc.get(k)==v for k,v in expected.items()),'Training candidate must match exact source-bound tables')
        self.document=copy.deepcopy(doc);self.scenario=scenario
        self.execution_context_sha256=doc['execution_context_sha256']

    def _lookup(self,active,resident_owners):
        require(sorted(resident_owners)==sorted(self.document['resident_owners']),'Complete resident roster required for resource prediction')
        state,_=key(active,resident_owners)
        return self.document['states'].get(state)

    def power(self,active,*,resident_owners):
        row=self._lookup(active,resident_owners);known=bool(row and row['power_known'])
        return dict(known=known,watts=row[self.scenario]['watts'] if known else None,
            evidence='New natural full-state bracket-contained training polls only',native_provider_eligible=False)

    def memory(self,active,*,resident_owners):
        row=self._lookup(active,resident_owners);known=bool(row and row['memory_known'])
        return dict(known=known,peak_memory_mib=row[self.scenario]['peak_memory_mib'] if known else None,
            inactive_resident_memory_is_zero=False,safe_factor_applied=False,capacity_bound=False)

    def startup(self):return copy.deepcopy(self.document['startup'])


def predict_timeline(timeline,provider,*,target_context,execution_binding):
    """Integrate exactly one global clock; existing scorer owns normalization."""
    from .macro_resource_timeline_v1 import clock
    require(timeline.get('schema')=='gsdeploy.macro_global_resource_timeline.v1',
        'Explicit global macro resource timeline required')
    require(context_identity(target_context)==provider.execution_context_sha256,
        'Explicit exact target execution context required')
    timeline_id=hashlib.sha256(json.dumps(timeline,sort_keys=True).encode()).hexdigest()
    require(type(execution_binding) is dict
        and execution_binding.get('execution_context_sha256')==provider.execution_context_sha256
        and execution_binding.get('resource_timeline_sha256')==timeline_id
        and execution_binding.get('basis')=='caller_declared_explicit_reference_execution_binding',
        'Explicit caller binding of this exact global timeline to target execution required')
    total=clock(timeline['total_wall_ns']);measure=clock(timeline['total_measurement_overlap_ns'])
    duration=measurement=0;known_j=0.;known_peak=None;power_unknown=0;memory_unknown=0
    for row in timeline['slices']:
        delta=clock(row['duration_ns']);overlap=clock(row['measurement_overlap_ns'])
        require(delta>0 and overlap<=delta,'Positive global slice and bounded original measurement overlap required')
        duration+=delta;measurement+=overlap
        members=row['actual_START_to_READY_handlers'];residents=row['resident_owners']
        power=provider.power(members,resident_owners=residents)
        if power['known']:known_j+=power['watts']*float(overlap)/1e9
        else:power_unknown+=overlap
        memory=provider.memory(members,resident_owners=residents)
        if memory['known']:
            peak=memory['peak_memory_mib'];known_peak=peak if known_peak is None else max(known_peak,peak)
        else:memory_unknown+=delta
    require(duration==total and measurement==measure,'All global idle/parallel/drain slices must remain in denominator')
    startup=provider.startup();pre=startup['premeasurement_owned_sampled_peak_mib']
    whole_peak=None if memory_unknown or pre is None or known_peak is None else max(pre,known_peak)
    from .macro_response_queue_reference_v1 import encode
    return encode(dict(schema='gsdeploy.natural_macro_resource_timeline_prediction.v1',
        board_energy_joules=known_j if power_unknown==0 else None,partial_known_measurement_joules=known_j,
        peak_memory_mib=whole_peak,partial_known_peak_memory_mib=known_peak,
        startup_energy_joules=startup['joules'],premeasurement_owned_sampled_peak_mib=pre,
        power_unknown_measurement_duration_ns=power_unknown,memory_unknown_observation_duration_ns=memory_unknown,
        total_observation_duration_ns=duration,total_measurement_duration_ns=measurement,
        execution_context_sha256=provider.execution_context_sha256,
        resource_timeline_sha256=timeline_id,execution_binding=copy.deepcopy(execution_binding),
        caller_execution_binding_independently_authenticated=False,
        startup_amortization_seconds=3600,startup_energy_already_amortized=False,safe_factor_already_applied=False,
        resource_prediction_qualified=False,main_queue_provider_eligible=False,validation_labels_read=False,GPU_started=False))


def replay_diagnostic(table,model):
    """Actual state durations are supplied; this is not future-path prediction."""
    known_j=0.;known_actual_j=0.;known_actual=True;known_ns=0;peak=None;memory_known=True;unknown=[]
    for row in table['slices']:
        state=model['states'].get(row['state_key'])
        if state and state['power_known']:
            known_j+=state['mean']['watts']*row['duration_ns']/1e9;known_ns+=row['duration_ns']
            actual=row['observed_piecewise_linear_board_joules']
            if actual is None:known_actual=False
            else:known_actual_j+=actual
        else:unknown.append(row['state_key'])
        if state and state['memory_known']:
            x=state['mean']['peak_memory_mib'];peak=x if peak is None else max(peak,x)
        else:memory_known=False
    startup=model['startup'];pred_startup=startup['joules']
    pre=startup['premeasurement_owned_sampled_peak_mib']
    if pre is None:memory_known=False
    predicted_peak=max(pre,peak) if memory_known and peak is not None else None
    predicted_j=known_j if known_ns==table['observation_duration_ns'] else None
    def error(p,a):return None if p is None or a is None else dict(predicted=p,actual=a,
        signed_error=p-a,absolute_relative_error=abs(p-a)/a if a>0 else None)
    return dict(heldout_capture_sha256=table['capture_sha256'],training_captures=model['capture_sha256s'],
        heldout_capture_in_training=table['capture_sha256'] in model['capture_sha256s'],
        complete_actual_state_durations_supplied=True,forward_resource_prediction=False,
        observation_duration_ns=table['observation_duration_ns'],power_known_duration_ns=known_ns,
        power_unknown_duration_ns=table['observation_duration_ns']-known_ns,
        partial_known_observation_joules=known_j,power_unknown_state_keys=sorted(set(unknown)),
        known_state_duration_energy_error=error(known_j,known_actual_j) if known_actual else None,
        observation_energy=error(predicted_j,table['observation_joules']),
        startup_energy=error(pred_startup,table['startup_joules']),
        raw_monitor_owned_sampled_peak=error(predicted_peak,table['raw_monitor_owned_sampled_peak_mib']),
        scoring_measurement_energy=None,scoring_E_unknown_reason=table['scoring_energy_unknown_reason'],
        resource_accuracy_qualified=False)


def run(paths,output):
    from .macro_joint_response_fit_v1 import verified_document
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    tables=[];sources=[];refs=[]
    for index,path in enumerate(paths):
        doc,closure=verified_document(path)
        review=checked(doc['sources'][0]);raw_ref=review['inputs']['raw.json'];raw=checked(raw_ref)
        context_ref=review['inputs']['context.json'];context=checked(context_ref)
        require(context['paired_profile_mode']=='natural' and context['worker_variant']=='semantic_progress_worker_v1.py',
            'Only new natural progress worker resource sources supported')
        table=observed_table(raw,doc,tuple(context['residents']))
        table.update(capture_sha256=raw_ref['sha256'],execution_context_sha256=raw['execution_context_sha256'],
            context=context_ref,raw=raw_ref,training_macros=receipt(path),
            priming_content_identity=priming_identity(raw,context))
        destination=out/f'capture{index+1:02d}-training-table.json';save(destination,table)
        refs.append(receipt(destination));tables.append(table);sources.extend(closure)
    require(len(tables)==3,'Exactly three independent new natural training captures required')
    model=fit_tables(tables)
    for name in ('natural_macro_resources_candidate_v1.py','macro_joint_response_fit_v1.py','composable_response_macros_v1.py'):
        sources.append(receipt(Path(__file__).with_name(name)))
    sources.append(receipt(Path(__file__).resolve().parents[2]/'gsdeploy/energy.py'))
    model.update(training_tables=refs,sources=list({r['path']:r for r in sources}.values()))
    save(out/'model.json',model)
    folds=[]
    for heldout in tables:
        train=[t for t in tables if t['capture_sha256']!=heldout['capture_sha256']]
        fold=replay_diagnostic(heldout,fit_tables(train));require(not fold['heldout_capture_in_training'],'Capture leaveout contamination')
        folds.append(fold)
    save(out/'whole-capture-resource-replay.json',folds)
    report=dict(schema='gsdeploy.natural_macro_resource_training_report.v1',
        status='THREE_CAPTURE_TRAINING_TABLES_AND_RESOURCE_REPLAY_ONLY_NOT_QUALIFICATION',
        model=receipt(out/'model.json'),tables=refs,diagnostics=receipt(out/'whole-capture-resource-replay.json'),
        sources=model['sources'],captures=len(tables),requests=sum(t['measured_requests'] for t in tables),
        polls=sum(t['raw_monitor_sample_count'] for t in tables),macro_nodes=sum(t['macro_nodes'] for t in tables),
        states=len(model['states']),power_supported_states=sum(s['power_known'] for s in model['states'].values()),
        memory_supported_states=sum(s['memory_known'] for s in model['states'].values()),
        whole_capture_leaveout=folds,
        startup_amortization_seconds=3600,startup_energy_already_amortized=False,safe_factor_already_applied=False,
        unchanged_scorer='Existing scorer alone applies startup_J / 3600 and memory safety factor once',
        limitations=['Three natural training captures only; no validation labels or independent resource gate.',
            'State contains actual host macro phase/public input, not SM work or per-model attribution.',
            'NVML query brackets retained; boundary polls have UNKNOWN state labels and no state-training imputation.',
            'Poll midpoint piecewise-linear board energy is a sampling model; sampled owned maximum is not a continuous peak or bound.',
            'Per-state phase pooling omits forward index, history and shape; stationary resource transport remains a hypothesis.',
            'Profile observation endpoint combines measurement and drain; no separate scoring deadline exists, so final E remains UNKNOWN.',
            'Leaveout energy uses actual heldout state durations; it does not qualify prospective timing, queue/P95, resources or utility.'],
        resource_provider_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False)
    require(all(receipt(r['path'])==r for r in sources),'Training source changed during fit')
    save(out/'report.json',report);return receipt(out/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-macros',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.training_macros,args.output)))
