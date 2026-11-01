"""Isolated generated-shape sidecar and absolute sampled-memory hypothesis.

No filtered poll increments, future heldout state, capacity bound or provider
promotion. Exact duration and explicitly declared observation support are
required for an empirical absolute demand; otherwise the rollout is UNKNOWN.
"""
from bisect import bisect_left,bisect_right
from collections import defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path
import statistics

from . import twoFIFO_joint_request_law_v4 as joint
from . import twoFIFO_resource_provider_v3 as resource
from . import twoFIFO_memory_envelope_candidate_v1 as shapes
from . import actual_worker_twoFIFO_reference_v5_2 as engine
from .macro_response_queue_reference_v1 import encode,rational
from .stage_response_features_v1 import checked,receipt,ordered_events
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_prospective_absolute_sampled_memory.v1'
SIDECAR_SCHEMA=SCHEMA+'.selected_vector_shape_sidecar'
TABLE_SCHEMA=SCHEMA+'.native_structural_intervals'
require=resource.require


def digest(value):return resource.digest(encode(value))


def selected_shape_sidecar(selections,*,physical_execution_identity):
    """Preserve every selected complete node and generated mark, before use."""
    require(isinstance(physical_execution_identity,str) and len(physical_execution_identity)==64,
        'Explicit physical execution identity required')
    slots=[];seen=set()
    for selected in selections:
        require(set(selected)=={'vector','public_key_arrival_ordinal'},'Explicit selected complete vector and public ordinal required')
        vector=selected['vector'];ordinal=selected['public_key_arrival_ordinal']
        require(isinstance(vector.get('training_block_id'),str) and vector['training_block_id'],
            'Selected vector must retain its whole-block donor')
        compiled=joint.compile_draw_v4(vector,ordinal);key=tuple(vector['public_key'])
        require(len(key)==3 and all(isinstance(v,str) and v for v in key),'Exact owner/task/input key required')
        require((*key,ordinal) not in seen,'Distinct complete public slots required');seen.add((*key,ordinal))
        require(len(vector['nodes'])==len(compiled['plan'].macros),'Complete selected macro list required')
        nodes=[]
        for node,macro in zip(vector['nodes'],compiled['plan'].macros):
            require(node['macro_id']==macro.identity,'Selected node and compiled macro differ')
            shape=copy.deepcopy(node['forward_shape'])
            if shape is not None:
                require(type(shape) is dict and set(shape)=={'input_sequence_length','input_batch_size','forward_index'},
                    'Exact selected BEGIN shape axes required')
                shapes.shape(shape)
            marks=[]
            for source,generated in zip(node['marks'],macro.marks):
                require(compiled['generated_to_source_message_identity'][generated.identity]==source['identity'],
                    'Generated mark must bind its exact selected source mark')
                marks.append(dict(source_mark_identity=source['identity'],generated_mark_identity=generated.identity,
                    action=source['action'],phase=source['phase'],macro_id=macro.identity,
                    forward_shape=copy.deepcopy(shape)))
            nodes.append(dict(macro_id=macro.identity,phase=node['phase'],kind=node['kind'],
                forward_shape=shape,marks=marks))
        slots.append(dict(public_key=list(key),public_key_arrival_ordinal=ordinal,
            selected_vector_id=vector['vector_id'],selected_complete_vector_sha256=digest(vector),
            training_block_id=vector['training_block_id'],nodes=nodes))
    return dict(schema=SIDECAR_SCHEMA,physical_execution_identity=physical_execution_identity,slots=slots,
        selected_complete_vectors=len(slots),all_nodes_and_marks_preserved=True,
        shape_visibility_endpoint='generated_parent_processing_not_latent_occurrence',
        sidecar_is_independently_authenticated=False,provider_replaced=False)


class ShapeState:
    def __init__(self,rows,owners,descriptor=None):
        self.state=shapes.CausalState(rows,owners);self.initial_known=descriptor is not None
        if descriptor is not None:
            require(descriptor['resident_owners']==list(owners),'Initial shape resident roster differs')
            self.state.active={x['owner']:copy.deepcopy(x) for x in descriptor['current_processed_owner_task_exact_input_phase_shape']}
            self.state.highwater={(x['owner'],x['task']):{k:copy.deepcopy(v) for k,v in x.items() if k not in ('owner','task')}
                for x in descriptor['prior_processed_shape_highwater']}
            self.state.maximum_visible_concurrency=descriptor['maximum_prior_visible_concurrency']
            require(self.state.descriptor()==descriptor,'Exact causal initial shape descriptor required')

    def descriptor(self):return dict(shape_state=self.state.descriptor(),initial_shape_history_known=self.initial_known)

    def native(self,event):self.state.process(event)

    def generated(self,body,mark):
        action=body['action'];phase=body.get('phase');rid=body['request_id']
        if action=='phase':
            event=dict(request_id=rid,event='start',semantic_phase=phase,
                segment_id='native_inference' if phase=='entry' else 'native_ocr' if phase=='ocr_inference' else phase)
            if phase in ('prefill','decode'):event.update(shapes.shape(mark['forward_shape'] or {}))
            self.state.process(event)
        elif action in ('clear','ready'):
            self.state.process(dict(request_id=rid,event='end',segment_id='native_inference'))


def _coalesce(intervals):
    result=[]
    for row in intervals:
        if result and result[-1]['right_ns']==row['left_ns'] and result[-1]['state']==row['state'] and result[-1].get('case_id')==row.get('case_id'):
            result[-1]['right_ns']=row['right_ns']
        else:result.append(copy.deepcopy(row))
    for index,row in enumerate(result):row.update(interval_id=index,duration_ns=row['right_ns']-row['left_ns'],state_key=digest(row['state']))
    return result


def generated_shape_intervals(sidecar,world,public_requests,*,resident_owners,initial_descriptor=None):
    """Only processed notifications update shape; future sidecar nodes stay latent."""
    require(sidecar.get('schema')==SIDECAR_SCHEMA and world.get('schema')==engine.SCHEMA,'Exact sidecar and v5.2 generated world required')
    world=engine.decode(world);requests={};ordinals=defaultdict(int);request_slots={}
    for _,row in sorted(enumerate(public_requests),key=lambda x:(rational(x[1]['arrival_ns']),x[0])):
        require(set(row)=={'request_id','owner','task','pool_id','arrival_ns'} and row['request_id'] not in requests,
            'Complete public-only offered rows required')
        require(row['owner'] in resident_owners,'Offered owner outside resident roster')
        key=(row['owner'],row['task'],row['pool_id']);ordinal=ordinals[key];ordinals[key]+=1
        requests[row['request_id']]=row;request_slots[row['request_id']]=(*key,ordinal)
    require(len(requests)==world['offered_requests'] and set(requests)==set(world['RESULT_receipts'])|set(world['unfinished_request_ids']),
        'Exact complete generated offered cohort required')
    slots={(*s['public_key'],s['public_key_arrival_ordinal']):s for s in sidecar['slots']}
    require(len(slots)==len(sidecar['slots']) and set(slots)==set(request_slots.values()),'Sidecar must cover every offered public slot exactly')
    marks={key:{m['generated_mark_identity']:m for n in slot['nodes'] for m in n['marks']} for key,slot in slots.items()}
    state=ShapeState([dict(request_id=k,model=r['owner'],task=r['task'],pool_id=r['pool_id']) for k,r in requests.items()],resident_owners,initial_descriptor)
    start=rational(world['original_window'][0]);end=rational(world['cutoff_ns']);cursor=start;intervals=[];processed=[];seen=set();last=start
    for index,body in enumerate(world['trace']):
        if body.get('kind')!='processed_notification':continue
        at=rational(body['available_ns']);require(start<=last<=at<=end,'Generated processing chronology must remain exact');last=at
        occurrence=rational(body['occurrence_ns'])
        require(start<=occurrence<=at,'Generated occurrence cannot follow its processing time')
        if 'reader_receipt_ns' in body:
            reader=rational(body['reader_receipt_ns'])
            require(occurrence<=reader<=at,'Generated reader/processing chronology differs')
        rid=body['request_id'];require(rid in requests and requests[rid]['owner']==body['owner'],'Generated notification public owner mismatch')
        key=request_slots[rid];identity=body['identity'];duplicate=(key,identity)
        require(duplicate not in seen,'Duplicate processed generated mark');seen.add(duplicate)
        if body['action']=='ready':
            require(identity=='worker_READY','Original generated READY identity required');mark=None
        else:
            require(identity in marks[key],'Unbound generated mark cannot update shape')
            mark=marks[key][identity]
            require(all(body.get(k)==mark[k] for k in ('action','phase','macro_id')),'Generated action/phase/macro differs from selected vector')
        if body['action'] not in ('phase','clear','ready'):continue
        if cursor<at:intervals.append(dict(left_ns=cursor,right_ns=at,state=state.descriptor()));cursor=at
        before=state.descriptor();state.generated(body,mark)
        processed.append(dict(trace_ordinal=index,request_id=rid,public_slot=list(key),generated_mark_identity=identity,
            macro_id=body.get('macro_id'),processing_ns=at,state_before=before,state_after=state.descriptor()))
    if cursor<end:intervals.append(dict(left_ns=cursor,right_ns=end,state=state.descriptor()))
    return encode(dict(schema=SCHEMA+'.generated_shape_intervals',physical_execution_identity=sidecar['physical_execution_identity'],
        intervals=_coalesce(intervals),processed_shape_marks=processed,initial_shape_history_known=state.initial_known,
        final_shape_descriptor=state.state.descriptor(),all_offered_requests_retained=len(requests),
        pending_or_latent_shape_used=False,sidecar_sha256=digest(sidecar),generated_world_sha256=digest(world),
        shape_state_sufficiency_established=False))


def attach_observation_support(intervals,polls):
    """Retain every poll; boundary-touching/crossing intervals stay UNKNOWN."""
    intervals=engine.decode(copy.deepcopy(intervals));polls=engine.decode(polls)
    if not intervals:return [],[dict(raw_poll_ordinal=i,status='UNKNOWN_OUTSIDE_STRUCTURAL_INTERVAL') for i in range(len(polls))]
    cuts=[r['left_ns'] for r in intervals]+[intervals[-1]['right_ns']]
    for row in intervals:row.update(contained_polls=[],crossing_poll_ordinals=[])
    assignments=[]
    for ordinal,poll in enumerate(polls):
        begin,end,at=(rational(poll[k]) for k in ('poll_begin_ns','poll_end_ns','sample_ns'))
        require(begin<=at<=end,'Complete poll/sample bracket required')
        index=bisect_right(cuts,begin)-1
        contained=bool(0<=index<len(intervals) and cuts[index]<begin<=at<=end<cuts[index+1])
        if contained:
            intervals[index]['contained_polls'].append(dict(raw_poll_ordinal=ordinal,**poll))
            assignments.append(dict(raw_poll_ordinal=ordinal,status='CONTAINED_IN_STRUCTURAL_INTERVAL',interval_ids=[index]))
        else:
            touched=list(range(max(0,bisect_left(cuts,begin)-1),min(len(intervals)-1,bisect_right(cuts,end)-1)+1)) if begin<=cuts[-1] and end>=cuts[0] else []
            for j in touched:intervals[j]['crossing_poll_ordinals'].append(ordinal)
            assignments.append(dict(raw_poll_ordinal=ordinal,status='UNKNOWN_BOUNDARY_CROSSING_OR_OUTSIDE',interval_ids=touched))
    for row in intervals:
        row['observation_support_status']='SUPPORTED_EXACT_INTERVAL_AND_POLL_GEOMETRY' if row['contained_polls'] and not row['crossing_poll_ordinals'] else 'UNKNOWN_CROSSING_OR_NO_CONTAINED_POLL'
        geometry=[{k:rational(p[k]-row['left_ns']) for k in ('poll_begin_ns','poll_end_ns','sample_ns')} for p in row['contained_polls']]
        row['support']=dict(state=row['state'],duration_ns=rational(row['duration_ns']),relative_poll_brackets=geometry)
        row['support_key']=digest(row['support'])
    return encode(intervals),encode(assignments)


def _relevant(event):
    return (event.get('event')=='start' and (event.get('segment_id') in ('native_inference','native_ocr') or event.get('semantic_phase') in ('prefill','decode'))
        or event.get('event')=='end' and event.get('segment_id') in ('native_inference','native_ocr'))


def absolute_table_from_verified(raw,plan,context,resource_table):
    """Pure post-native-review adapter; no poll increments enter these targets."""
    # Preserve native identities; annotate only an absent compatibility axis in
    # a temporary view. Current v3 capture already records profile_job_id.
    priming_rows=[dict(r,profile_job_id=r.get('profile_job_id','__priming_compatibility__')) for r in raw['priming_request_records']]
    rows=priming_rows+raw['request_records'];events=[e for e in ordered_events(rows) if _relevant(e)]
    windows=raw['measurement_windows'];start=windows[0]['start_ns'];end=windows[-1]['fixed_drain_deadline_ns']
    initial=shapes.candidate_at(priming_rows,plan['resident_owners'],start)
    state=ShapeState(rows,plan['resident_owners'],initial);cuts={start,end}
    cuts.update(e['collector_processing_time_ns'] for e in events if start<e['collector_processing_time_ns']<end)
    cuts.update(t for w in windows for t in (w['start_ns'],w['fixed_drain_deadline_ns']))
    events=[e for e in events if e['collector_processing_time_ns']>=start];index=0;intervals=[]
    ordered=sorted(cuts)
    for left,right in zip(ordered,ordered[1:]):
        while index<len(events) and events[index]['collector_processing_time_ns']<=left:state.native(events[index]);index+=1
        window=next((w for w in windows if w['start_ns']<=left<right<=w['fixed_drain_deadline_ns']),None)
        intervals.append(dict(left_ns=left,right_ns=right,state=state.descriptor(),case_id=window['case_id'] if window else None))
    intervals=_coalesce(intervals);polls=[]
    for sample in raw['board_samples']:
        begin,finish=resource.legacy.poll_bracket(sample)
        polls.append(dict(poll_begin_ns=begin,poll_end_ns=finish,sample_ns=sample['time_ns'],sampled_owned_mib=sample['owned_used_mib']))
    intervals,assignments=attach_observation_support(intervals,polls)
    for row in intervals:
        known=row['case_id'] is not None and row['observation_support_status'].startswith('SUPPORTED')
        values=[p['sampled_owned_mib'] for p in row['contained_polls']]
        known=bool(known and all(resource.legacy.finite(v) and v>=0 for v in values))
        row.update(absolute_sampled_peak_target_mib=max(values) if known else None,
            target_status='ABSOLUTE_INTERVAL_SAMPLED_PEAK' if known else 'UNKNOWN_UNSUPPORTED_INTERVAL')
    return dict(schema=TABLE_SCHEMA,capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],
        physical_execution_identity=resource.physical_execution_identity(plan,context),
        resident_owners=plan['resident_owners'],initial_shape_descriptor=initial,
        initial_after_priming_owned_sampled_peak_mib=resource_table['premeasurement_owned_sampled_peak_mib'],
        intervals=intervals,poll_assignments=assignments,cases=resource_table['measured_case_resources']['cases'],
        compatibility_priming_job_annotation_used=any('profile_job_id' not in r for r in raw['priming_request_records']),
        native_priming_job_identity_preserved=True,
        all_original_polls_retained=len(assignments)==len(polls),poll_increments_used=False,
        allocator_state_sufficiency_established=False,capacity_bound=False)


def fit_absolute_tables(tables):
    require(tables and all(t.get('schema')==TABLE_SCHEMA for t in tables),'Exact absolute-target tables required')
    require(len({t['attempt_id'] for t in tables})==len(tables),'Distinct absolute-target attempts required')
    blocks=sorted({t['capture_id'] for t in tables});require(len(blocks)>=2,'At least two whole training blocks required')
    require(all({t['attempt_id'] for t in tables if t['capture_id']==b}=={b+'-shard-'+str(i) for i in (1,2,3,4)} for b in blocks),'All four shards per training block required')
    require(len({t['physical_execution_identity'] for t in tables})==1,'Physical execution identity differs')
    groups=defaultdict(lambda:defaultdict(list));supports={}
    for table in tables:
        for row in table['intervals']:
            if row['absolute_sampled_peak_target_mib'] is None:continue
            require(resource.legacy.finite(row['absolute_sampled_peak_target_mib']),
                'Nonnegative finite absolute sampled-memory target required')
            require(row['support_key']==digest(row['support']) and row['support']['state']==row['state'],
                'Absolute interval support body/key/state differs')
            require(row['observation_support_status']=='SUPPORTED_EXACT_INTERVAL_AND_POLL_GEOMETRY'
                and row['contained_polls'] and not row['crossing_poll_ordinals'],
                'Absolute target needs supported contained polls without boundary crossing')
            decoded=engine.decode(row)
            geometry=[{k:rational(p[k]-decoded['left_ns']) for k in ('poll_begin_ns','poll_end_ns','sample_ns')} for p in decoded['contained_polls']]
            require(decoded['support']['duration_ns']==rational(decoded['right_ns']-decoded['left_ns'])
                and decoded['support']['relative_poll_brackets']==geometry,
                'Absolute support must match the complete interval and contained polling geometry')
            key=row['support_key'];supports[key]=row['support'];groups[key][table['capture_id']].append(row['absolute_sampled_peak_target_mib'])
    states={}
    for key,by_block in groups.items():
        peaks={b:max(v) for b,v in by_block.items()};known=len(peaks)>=2
        states[key]=dict(support=supports[key],per_block_absolute_peak_mib=peaks,
            per_block_all_absolute_targets_mib=dict(by_block),known=known,
            mean=statistics.mean(peaks.values()) if known else None,max=max(peaks.values()) if known else None)
    return dict(schema=SCHEMA+'.absolute_model',training_block_ids=blocks,training_attempt_ids=sorted(t['attempt_id'] for t in tables),
        physical_execution_identity=tables[0]['physical_execution_identity'],resident_owners=tables[0]['resident_owners'],states=states,
        donor_initial_states={t['attempt_id']:dict(training_block_id=t['capture_id'],shape_descriptor=t['initial_shape_descriptor'],
            owned_sampled_peak_mib=t['initial_after_priming_owned_sampled_peak_mib']) for t in tables},
        prediction_is_absolute_D_not_poll_increment=True,exact_duration_and_observation_geometry_required=True,
        minimum_supported_whole_blocks=2,equal_block_empirical_scenario_not_probability_law=True,
        memory_state_sufficiency_established=False,sampled_peak_is_capacity_bound=False,
        provider_replaced=False,resource_prediction_qualified=False,main_queue_provider_eligible=False,
        final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def train_absolute_campaign(campaign_path,review_paths,output):
    native_tables,sources=resource.load_complete(campaign_path,review_paths)
    require(len(native_tables)==12 and len({t['capture_id'] for t in native_tables})==3,'Exactly twelve complete native attempts required')
    tables=[]
    for native in native_tables:
        raw=checked(native['raw']);plan=checked(native['plan']);context=checked(native['context'])
        table=absolute_table_from_verified(raw,plan,context,native)
        table.update(raw=native['raw'],plan=native['plan'],training_review=native['training_review']);tables.append(table)
    model=fit_absolute_tables(tables);folds=[]
    for block in model['training_block_ids']:
        folds.append(dict(heldout_block_id=block,training_model=fit_absolute_tables([t for t in tables if t['capture_id']!=block]),
            heldout_attempt_ids=[t['attempt_id'] for t in tables if t['capture_id']==block]))
    out=Path(output);out.mkdir(parents=True,exist_ok=False);refs=[]
    for index,table in enumerate(tables):
        path=out/('table-'+str(index+1)+'.json');path.write_text(json.dumps(encode(table),indent=2,allow_nan=False)+'\n');refs.append(receipt(path))
    result=dict(schema=SCHEMA+'.training_report',model=model,whole_block_leaveout=folds,tables=refs,
        all_cases_intervals_polls_and_folds_retained=True,prospective_native_accuracy_validated=False,
        sources=strict_references([sources,implementation_sources(__file__)]))
    require(all(receipt(r['path'])==r for r in result['sources']),'Absolute training source changed')
    path=out/'report.json';path.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n');return receipt(path)


def rollout(sidecar,world,public_requests,*,resident_owners,physical_execution_identity,
            absolute_model=None,memory_origin=None,observation_plan=None,scenario='max',query_times=()):
    """Compose max(P,D) once per structural interval; queries never alter it."""
    require(sidecar['physical_execution_identity']==physical_execution_identity,'Sidecar target physical identity differs')
    require(scenario in ('mean','max'),'Explicit empirical absolute-demand scenario required')
    initial=None;prefix=None;model=absolute_model;initial_reason='UNKNOWN_NO_ABSOLUTE_MODEL_OR_GENERATED_MEMORY_ORIGIN'
    if model is not None:
        require(model.get('schema')==SCHEMA+'.absolute_model' and model['physical_execution_identity']==physical_execution_identity,
            'Exact absolute model and physical domain required')
        require(model['resident_owners']==list(resident_owners),'Absolute model resident roster differs')
        require(all(s['training_block_id'] in model['training_block_ids'] for s in sidecar['slots']),
            'Selected request vector block outside absolute training donors')
        if memory_origin is not None:
            require(set(memory_origin)=={'basis','donor_attempt_id'} and memory_origin['basis']=='selected_training_donor_after_original_priming',
                'Only explicit training-donor original-priming origin supported; settled seed is not memory state')
            donor=model['donor_initial_states'].get(memory_origin['donor_attempt_id'])
            require(donor is not None and donor['training_block_id'] in model['training_block_ids'],'Initial memory donor outside training fold')
            require({s['training_block_id'] for s in sidecar['slots']}=={donor['training_block_id']},
                'Original-priming donor and selected request vectors must share one whole-block latent')
            initial=donor['shape_descriptor'];prefix=donor['owned_sampled_peak_mib']
            if prefix is not None:require(resource.legacy.finite(prefix) and prefix>=0,'Nonnegative initial sampled peak required')
            initial_reason=None if prefix is not None else 'UNKNOWN_TRAINING_DONOR_INITIAL_PEAK'
    generated=generated_shape_intervals(sidecar,world,public_requests,resident_owners=resident_owners,initial_descriptor=initial)
    intervals=generated['intervals'];observation_reason='UNKNOWN_NO_DECLARED_GENERATED_OBSERVATION_PLAN'
    if observation_plan is not None:
        require(set(observation_plan)=={'basis','polls'} and observation_plan['basis']=='caller_declared_generated_observation_schedule',
            'Explicit prospective observation schedule required')
        require(all(set(p)=={'poll_begin_ns','poll_end_ns','sample_ns'} for p in observation_plan['polls']),
            'Prospective observation plan must contain clocks only, no realized memory labels')
        intervals,_=attach_observation_support(intervals,observation_plan['polls']);observation_reason=None
    known_peak=prefix;all_known=prefix is not None;predictions=[]
    for row in intervals:
        state=model['states'].get(row.get('support_key')) if model is not None else None
        if state is not None:
            require(digest(state['support'])==row['support_key'] and state['support']==row['support'],
                'Absolute model support body/key differs from generated interval')
        supported=bool(state and state['known'] and row.get('observation_support_status','').startswith('SUPPORTED'))
        demand=state[scenario] if supported else None
        if demand is not None:require(resource.legacy.finite(demand),'Nonnegative finite absolute demand required')
        if demand is None:all_known=False
        else:known_peak=demand if known_peak is None else max(known_peak,demand)
        predictions.append(dict(interval_id=row['interval_id'],left_ns=row['left_ns'],right_ns=row['right_ns'],
            state=row['state'],support_key=row.get('support_key'),absolute_D_mib=demand,
            predicted_prefix_peak_mib=known_peak if all_known else None,partial_known_peak_mib=known_peak,
            status='SUPPORTED_EMPIRICAL_ABSOLUTE_DEMAND' if supported else 'UNKNOWN_STATE_DURATION_OR_OBSERVATION_SUPPORT'))
    queries=[]
    for at in query_times:
        at=rational(engine.decode(at));matches=[r for r in predictions if rational(engine.decode(r['left_ns']))<=at<rational(engine.decode(r['right_ns']))]
        queries.append(dict(at_ns=at,prediction=copy.deepcopy(matches[0]) if matches else None))
    return encode(dict(schema=SCHEMA+'.rollout',intervals=predictions,queries=queries,
        peak_memory_mib=known_peak if all_known and predictions else None,partial_known_peak_mib=known_peak,
        generated_shape_intervals=generated,memory_origin=copy.deepcopy(memory_origin),
        initial_memory_unknown_reason=initial_reason,observation_support_unknown_reason=observation_reason,
        sidecar_sha256=digest(sidecar),absolute_model_sha256=digest(model) if model is not None else None,
        caller_observation_and_origin_independently_authenticated=False,
        supplied_absolute_model_independently_rederived_at_rollout=False,
        recurrence='P_next=max(P_previous,absolute_D)',poll_increments_used=False,
        repeated_queries_change_peak=False,realized_heldout_past_samples_used=False,
        state_sufficiency_established=False,capacity_bound=False,probability_law_qualified=False,
        startup_and_absolute_D_joint_law_qualified=False,
        original_memory_safe_factor_applied=False,provider_replaced=False,
        resource_prediction_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False))
