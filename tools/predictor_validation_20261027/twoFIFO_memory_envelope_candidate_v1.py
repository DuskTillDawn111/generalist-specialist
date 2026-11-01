"""Isolated causal shape/high-water memory hypothesis, never capacity proof.

Complete native v3 campaign only. Labels are sampled positive increments above
the actually observed prior owned peak. Evaluation conditions on heldout past
polls/processed shapes; it is filtered resource replay, not future-path rollout.
"""
from collections import defaultdict
import copy
import json
from pathlib import Path

from . import twoFIFO_resource_provider_v3 as resource
from . import twoFIFO_owned_training_review_v3 as review
from . import twoFIFO_observed_resources_v2 as observed
from . import twoFIFO_training_intake_v4 as intake
from .stage_response_features_v1 import receipt,checked,ordered_events
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_causal_shape_memory_envelope_candidate.v1'
TABLE_SCHEMA=SCHEMA+'.table'
require=review.require


def shape(event):
    result={k:event.get(k) for k in ('input_batch_size','input_sequence_length','forward_index')}
    require(all(v is None or type(v) is int and v>=0 for v in result.values()),'Explicit nonnegative actual BEGIN shape axes required')
    return result


class CausalState:
    def __init__(self,rows,resident_owners):
        self.rows={r['request_id']:r for r in rows};self.owners=list(resident_owners)
        self.active={};self.highwater={};self.maximum_visible_concurrency=0

    def process(self,event):
        row=self.rows[event['request_id']];owner=row['model'];edge=event.get('event');segment=event.get('segment_id')
        if edge=='start' and segment in ('native_inference','native_ocr'):
            self.active[owner]=dict(owner=owner,task=row['task'],pool_id=row['pool_id'],
                phase='entry' if segment=='native_inference' else 'ocr_inference',actual_BEGIN_shape=None)
        elif edge=='start' and event.get('semantic_phase') in ('prefill','decode'):
            actual=shape(event);key=(owner,row['task']);high=self.highwater.setdefault(key,
                dict(max_input_batch_size=None,max_input_sequence_length=None,max_forward_index=None))
            for axis,value in actual.items():
                if value is not None:
                    k='max_'+axis;high[k]=value if high[k] is None else max(high[k],value)
            self.active[owner]=dict(owner=owner,task=row['task'],pool_id=row['pool_id'],
                phase=event['semantic_phase'],actual_BEGIN_shape=actual)
        elif edge=='end' and segment in ('native_inference','native_ocr'):
            self.active.pop(owner,None)
        self.maximum_visible_concurrency=max(self.maximum_visible_concurrency,len(self.active))

    def descriptor(self):
        return dict(resident_owners=self.owners,
            current_processed_owner_task_exact_input_phase_shape=[copy.deepcopy(self.active[k]) for k in sorted(self.active)],
            prior_processed_shape_highwater=[dict(owner=k[0],task=k[1],**copy.deepcopy(self.highwater[k])) for k in sorted(self.highwater)],
            current_visible_concurrency=len(self.active),maximum_prior_visible_concurrency=self.maximum_visible_concurrency,
            visible_concurrency_is_not_actual_SM_or_live_allocation=True,
            unknown_entry_OCR_KV_allocator_axes_not_imputed=True)


def candidate_at(rows,owners,cutoff):
    state=CausalState(rows,owners)
    for event in ordered_events(rows):
        if event['collector_processing_time_ns']<=cutoff:state.process(event)
    return state.descriptor()


def table_from_verified(raw,plan,context):
    """Caller authenticates native review first; no labels discarded."""
    rows=raw['priming_request_records']+raw['request_records'];events=ordered_events(rows)
    samples=raw['board_samples'];startup=next(b['time_ns'] for b in raw['lifecycle_boundaries'] if b['boundary']=='startup_begin')
    state=CausalState(rows,plan['resident_owners']);event_index=0;prior_peak=None;records=[];poll_index=0
    facts=observed.observed(raw);windows=raw['measurement_windows']
    for ordinal,sample in enumerate(samples):
        begin,end=resource.legacy.poll_bracket(sample)
        while event_index<len(events) and events[event_index]['collector_processing_time_ns']<=begin:
            state.process(events[event_index]);event_index+=1
        # An earlier complete poll, never this label/current or future poll,
        # supplies the high-water anchor available before this query begins.
        while poll_index<ordinal:
            previous=samples[poll_index];_,finish=resource.legacy.poll_bracket(previous)
            if finish>begin:break
            if previous['time_ns']>=startup:
                prior_peak=previous['owned_used_mib'] if prior_peak is None else max(prior_peak,previous['owned_used_mib'])
            poll_index+=1
        window=next((w for w in windows if w['start_ns']<=sample['time_ns']<=w['fixed_drain_deadline_ns']),None)
        descriptor=state.descriptor();known=window is not None and prior_peak is not None
        reason=None if known else 'OUTSIDE_CASE' if window is None else 'NO_CAUSAL_PRIOR_OWNED_POLL'
        records.append(dict(raw_poll_ordinal=ordinal,midpoint_ns=sample['time_ns'],poll_begin_ns=begin,poll_end_ns=end,
            case_id=window['case_id'] if window else None,
            label_status='CONDITIONAL_SAMPLED_INCREMENT' if known else 'UNKNOWN_'+reason,
            state=descriptor,state_key=resource.digest(descriptor),prior_observed_owned_peak_mib=prior_peak,
            sampled_owned_mib=sample['owned_used_mib'],positive_increment_mib=max(0,sample['owned_used_mib']-prior_peak) if known else None,
            state_uses_processed_prefix_only=True,prior_peak_uses_earlier_completed_polls_only=True))
    return dict(schema=TABLE_SCHEMA,capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],
        execution_id=plan['execution_id'],physical_execution_identity=resource.physical_execution_identity(plan,context),
        measured_requests=len(raw['request_records']),priming_history_requests=len(raw['priming_request_records']),
        cases=facts['cases'],polls=records,all_original_poll_rows_retained=len(records)==len(samples),
        future_generated_shape_used=False,shape_history_is_sufficiency_hypothesis_only=True,
        sampled_memory_is_capacity_bound=False)


def fit_tables(tables):
    require(tables and len({t['attempt_id'] for t in tables})==len(tables),'Distinct attempt tables required')
    require(all(t['schema']==TABLE_SCHEMA for t in tables),'Exact candidate schema required')
    blocks=sorted({t['capture_id'] for t in tables});require(len(blocks)>=2,'Two whole training blocks required')
    require(all({t['attempt_id'] for t in tables if t['capture_id']==b}==
        {b+'-shard-'+str(i) for i in (1,2,3,4)} for b in blocks),'Complete four-shard blocks required')
    require(len({t['physical_execution_identity'] for t in tables})==1,'Physical worker/configuration/priming identity differs')
    groups=defaultdict(lambda:defaultdict(list));descriptors={}
    for table in tables:
        for poll in table['polls']:
            if poll['positive_increment_mib'] is None:continue
            key=poll['state_key'];descriptors[key]=poll['state']
            groups[key][table['capture_id']].append(poll['positive_increment_mib'])
    states={}
    for key,by_block in groups.items():
        known=len(by_block)>=2
        states[key]=dict(state=descriptors[key],whole_block_count=len(by_block),known=known,
            per_block_sampled_increment_mib=dict(by_block),equal_block_weights={b:1/len(by_block) for b in by_block},
            empirical_conditional_max_increment_mib=max(max(v) for v in by_block.values()) if known else None,
            empirical_upper_is_probability_or_capacity_bound=False)
    return dict(schema=SCHEMA,status='CAUSAL_SHAPE_ENVELOPE_HYPOTHESIS_NOT_SUFFICIENT_STATE_OR_PROVIDER',
        training_block_ids=blocks,training_attempt_ids=sorted(t['attempt_id'] for t in tables),
        physical_execution_identity=tables[0]['physical_execution_identity'],states=states,
        minimum_supported_whole_blocks=2,equal_whole_block_mixture=True,
        conditioned_on_exact_public_input_and_processed_shape=True,
        allocator_free_bins_KV_lengths_workspaces_and_release_history_UNKNOWN=True,
        no_priming_single_request_dominates_concurrency_assumption=True,
        sampled_peak_is_capacity_bound=False,memory_safety_factor_applied=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,validation_labels_read=False,GPU_started=False)


def evaluate_table(table,model):
    require(table.get('schema')==TABLE_SCHEMA and model.get('schema')==SCHEMA,'Exact candidate table and model schemas required')
    require(table.get('physical_execution_identity')==model.get('physical_execution_identity'),
        'Heldout physical worker/configuration/priming identity differs')
    require(table['capture_id'] not in model['training_block_ids'],'Whole heldout block leaked into training')
    records=[];case_records=[]
    for poll in table['polls']:
        if poll['case_id'] is None:continue
        state=model['states'].get(poll['state_key']);known=bool(state and state['known'] and poll['prior_observed_owned_peak_mib'] is not None)
        predicted=poll['prior_observed_owned_peak_mib']+state['empirical_conditional_max_increment_mib'] if known else None
        actual=max(poll['prior_observed_owned_peak_mib'],poll['sampled_owned_mib']) if poll['prior_observed_owned_peak_mib'] is not None else None
        records.append(dict(case_id=poll['case_id'],raw_poll_ordinal=poll['raw_poll_ordinal'],state_key=poll['state_key'],
            predicted_cumulative_sampled_peak_mib=predicted,actual_cumulative_sampled_peak_mib=actual,
            signed_error_mib=predicted-actual if known else None,underestimated=predicted<actual if known else None,
            status='SUPPORTED_CONDITIONAL_PAST_FILTERED_REPLAY' if known else 'UNKNOWN_STATE_OR_CAUSAL_PREFIX_SUPPORT'))
    for case in table['cases']:
        rows=[r for r in records if r['case_id']==case['case_id']];supported=bool(rows) and all(r['predicted_cumulative_sampled_peak_mib'] is not None for r in rows)
        predicted=max(r['predicted_cumulative_sampled_peak_mib'] for r in rows) if supported else None
        actual=case['startup_through_this_drain_owned_sampled_peak_mib']
        case_records.append(dict(case_id=case['case_id'],polls=len(rows),supported_polls=sum(r['signed_error_mib'] is not None for r in rows),
            all_polls_supported=supported,predicted_owned_sampled_peak_mib=predicted,actual_owned_sampled_peak_mib=actual,
            signed_error_mib=predicted-actual if supported and actual is not None else None,
            underestimated=predicted<actual if supported and actual is not None else None))
    known=[r for r in records if r['signed_error_mib'] is not None]
    return dict(heldout_block_id=table['capture_id'],attempt_id=table['attempt_id'],cases=case_records,polls=records,
        offered_cases=len(table['cases']),all_original_cases_retained=True,supported_polls=len(known),unknown_polls=len(records)-len(known),
        supported_poll_fraction=len(known)/len(records) if records else None,
        underestimation_count=sum(r['underestimated'] for r in known),
        empirical_supported_poll_nonunderestimate_fraction=sum(not r['underestimated'] for r in known)/len(known) if known else None,
        actual_heldout_past_poll_and_processed_shape_conditioned=True,heldout_future_shape_or_peak_used_as_feature=False,
        prospective_generated_resource_prediction=False,capacity_bound=False)


def run(campaign_path,review_paths,output):
    metadata=checked(receipt(campaign_path))
    require(metadata.get('schema')==intake.campaign_v2.SCHEMA,'Only new clean v3 campaign can enter candidate training; legacy v2 HOLD')
    campaign=intake.campaign_v2.audit(campaign_path);plans=[checked(r) for r in campaign['attempts']]
    reviews=[(receipt(p),checked(receipt(p))) for p in review_paths]
    require(intake.planned_intake(plans,reviews)['complete'],'All twelve complete native attempts required; no partial memory fit')
    by_attempt={checked(r['inputs']['plan.json'])['attempt_id']:ref for ref,r in reviews};tables=[];documents=[];sources=[]
    for plan in plans:
        ref=by_attempt[plan['attempt_id']];r=checked(ref)
        doc=intake.macros_v2.extract(ref['path']);documents.append(doc)
        raw=checked(r['raw']);context=checked(raw['context']);table=table_from_verified(raw,plan,context)
        table.update(raw=r['raw'],training_review=ref,plan=raw['plan'],raw_SHA_provenance_only=True)
        tables.append(table);sources.extend(doc['sources'])
    denominator=intake.complete_documents(plans,documents);model=fit_tables(tables);folds=[]
    for block in model['training_block_ids']:
        train=[t for t in tables if t['capture_id']!=block];heldout=[t for t in tables if t['capture_id']==block]
        fitted=fit_tables(train)
        folds.append(dict(heldout_block_id=block,training_attempt_ids=fitted['training_attempt_ids'],
            heldout_attempt_ids=[t['attempt_id'] for t in heldout],
            model=fitted,
            evaluations=[evaluate_table(t,fitted) for t in heldout]))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    refs=[]
    for i,table in enumerate(tables):
        p=out/('table-'+str(i+1)+'.json');p.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n');refs.append(receipt(p))
    result=dict(model=model,whole_block_leaveout=folds,denominator=denominator,tables=refs,
        all_cases_all_states_all_folds_retained=True,no_successful_fold_selection=True,
        independent_prediction_accuracy_validated=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        sources=strict_references([receipt(campaign_path),campaign['attempts'],sources,implementation_sources(__file__)]))
    require(all(receipt(s['path'])==s for s in result['sources']),'Memory candidate source changed')
    p=out/'report.json';p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return receipt(p)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True)
    parser.add_argument('--training-reviews',nargs='*',default=[]);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.training_reviews,args.output)))
