"""Source-bound multi-window resources; three whole-block training hypotheses.

Physical worker/configuration identity is distinct from arrival/plan identity.
Power states use original bracket-contained polls. Memory also binds the exact
ordered prior state path; unseen history remains UNKNOWN. No scorer is changed.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import statistics

from . import twoFIFO_training_intake_v4 as intake
from . import twoFIFO_composable_macros_v3 as macros_v2
from . import twoFIFO_observed_resources_v2 as observed
from . import natural_macro_resources_candidate_v1 as legacy
from .stage_response_features_v1 import checked,receipt
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_multiwindow_resource_training.v3'
TABLE_SCHEMA='gsdeploy.twoFIFO_multiwindow_resource_table.v3'
V2_CAPTURE_HOLD='HOLD_V2_IN_WINDOW_CHECKPOINT_PERTURBS_COLLECTOR_AND_MONITOR_NOT_PRODUCTION_TRAINING'


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_execution_payload(plan,context):
    """Explicit retained physical execution axes; plan/arrivals remain provenance.

    Whole context bytes retain worker/checkpoint/device/header/thread/protocol
    bindings. Runtime knobs affecting execution are copied separately; rates,
    routing, random seed and window lengths describe offered work, not workers.
    Input bytes and exact priming remain bound, despite excluding public arrivals.
    """
    config=plan['runtime_config']
    physical_keys=('deployment','startup_timeout_seconds','shutdown_timeout_seconds',
                   'max_pending_per_model','idle_residency')
    offered_keys=('workload_rps','routing_rps','warmup_seconds','measurement_seconds',
                  'arrival_process','random_seed','routing_source','drain_timeout_seconds')
    require(set(config)==set(physical_keys+offered_keys),'Unreviewed runtime knob cannot silently escape physical identity')
    require(plan['resident_owners']==list(intake.campaign_v2.OWNERS),'Exact physical resident roster required')
    return dict(schema='gsdeploy.twoFIFO_physical_execution_identity.v3',context=context,
        runtime_physical_config={k:config[k] for k in physical_keys},runtime_policy=plan['runtime_policy'],
        resident_owners=plan['resident_owners'],priming=plan['priming'],
        implementation_sources=plan['sources'],source_freeze=plan['source_freeze'],
        plan_hash_and_arrivals_are_provenance_not_physical_axes=True)


def physical_execution_identity(plan,context):return digest(physical_execution_payload(plan,context))


def next_history(previous,state_key):
    return digest(dict(previous=previous,actual_state_key=state_key))


def history_start(priming_content_identity):return digest(dict(priming_content_identity=priming_content_identity))


def memory_key(state_key,history):return digest(dict(state_key=state_key,ordered_prior_state_history=history))


def table_from_verified(raw,macros,plan,context):
    """Pure adapter; public loader authenticates native macros/review first."""
    table=legacy.observed_table(raw,macros,tuple(plan['resident_owners']))
    measurements=observed.observed(raw)
    priming=legacy.priming_identity(raw,context);history=history_start(priming);previous=None
    for row in table['slices']:
        state=row['state_key']
        if state!=previous:history=next_history(history,state);previous=state
        row['memory_state_key']=memory_key(state,history)
        row['ordered_state_history_identity']=history
        row['measurement_overlap_ns']=sum(max(0,min(row['right_ns'],w['end_ns'])-
            max(row['left_ns'],w['start_ns'])) for w in raw['measurement_windows'])
    table.update(schema=TABLE_SCHEMA,capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],
        execution_id=plan['execution_id'],physical_execution_identity=physical_execution_identity(plan,context),
        physical_execution_payload=physical_execution_payload(plan,context),priming_content_identity=priming,
        measured_case_resources=measurements,measurement_windows=copy.deepcopy(raw['measurement_windows']),
        measurement_duration_ns=sum(w['end_ns']-w['start_ns'] for w in raw['measurement_windows']),
        startup_joules=measurements['startup']['energy_joules'],
        raw_monitor_owned_sampled_peak_mib=measurements['all_startup_through_final_drain_owned_sampled_peak_mib'],
        scoring_measurement_energy_joules=(sum(r['measurement_energy_joules'] for r in measurements['cases'])
            if all(r['measurement_energy_joules'] is not None for r in measurements['cases']) else None),
        scoring_energy_unknown_reason=None,one_startup_per_attempt=True,
        prior_cases_and_idle_gaps_retained=True,memory_history_compressed_ordered_state_path=True)
    return table


def fit_tables(tables):
    require(tables and len({t['attempt_id'] for t in tables})==len(tables),'Distinct attempt resources required')
    require(all(t['schema']==TABLE_SCHEMA for t in tables),'New multi-window table required')
    physical={t['physical_execution_identity'] for t in tables};require(len(physical)==1,'Physical execution axes differ')
    require(len({t['priming_content_identity'] for t in tables})==1,'Exact same priming contents required')
    blocks=sorted({t['capture_id'] for t in tables});require(len(blocks)>=2,'At least two independent whole blocks required')
    require(all({t['attempt_id'] for t in tables if t['capture_id']==b}==
        {b+'-shard-'+str(i) for i in (1,2,3,4)} for b in blocks),'All four shards required per training block')
    power=defaultdict(lambda:defaultdict(lambda:[0.,0]));memory=defaultdict(dict)
    for table in tables:
        block=table['capture_id']
        for row in table['slices']:
            values=power[row['state_key']][block]
            values[0]+=row['power_training_joules'];values[1]+=row['power_training_duration_ns']
            if row['sampled_owned_peak_mib'] is not None:
                k=row['memory_state_key'];memory[k][block]=max(memory[k].get(block,0),row['sampled_owned_peak_mib'])
    states={}
    for k,by_block in power.items():
        values={b:j/(n/1e9) for b,(j,n) in by_block.items() if n>0}
        states[k]=dict(per_block_watts=values,known=len(values)>=2,
            mean=statistics.mean(values.values()) if len(values)>=2 else None,
            max=max(values.values()) if len(values)>=2 else None)
    memories={k:dict(per_block_peak_mib=values,known=len(values)>=2,
        mean=statistics.mean(values.values()) if len(values)>=2 else None,
        max=max(values.values()) if len(values)>=2 else None) for k,values in memory.items()}
    startup_j={};startup_m={}
    for b in blocks:
        subset=[t for t in tables if t['capture_id']==b];energies=[t['startup_joules'] for t in subset]
        peaks=[t['premeasurement_owned_sampled_peak_mib'] for t in subset]
        startup_j[b]=statistics.mean(energies) if all(v is not None for v in energies) else None
        startup_m[b]=statistics.mean(peaks) if all(v is not None for v in peaks) else None
    def aggregate(values):return statistics.mean(values.values()) if all(v is not None for v in values.values()) else None
    return dict(schema=SCHEMA,physical_execution_identity=next(iter(physical)),
        physical_execution_payload=tables[0]['physical_execution_payload'],
        resident_owners=tables[0]['resident_owners'],priming_content_identity=tables[0]['priming_content_identity'],
        training_block_ids=blocks,training_attempt_ids=sorted(t['attempt_id'] for t in tables),
        power_states=states,memory_history_states=memories,
        startup=dict(joules=aggregate(startup_j),premeasurement_owned_sampled_peak_mib=aggregate(startup_m),
            per_block_joules=startup_j,occurrences_per_predicted_attempt=1,priming_included=True),
        equal_block_aggregation=True,raw_SHA_used_as_replication_unit=False,
        minimum_supported_whole_blocks=2,resource_prediction_qualified=False,
        main_queue_provider_eligible=False,final_gain_eligible=False,
        startup_energy_already_amortized=False,safe_factor_already_applied=False,
        state_pooling_is_empirical_stationarity_hypothesis=True,
        ordered_state_history_omits_elapsed_time_and_tensor_shapes=True,
        validation_labels_read=False,GPU_started=False)


def load_complete(campaign_path,review_paths):
    metadata=checked(receipt(campaign_path))
    require(metadata.get('schema')!='gsdeploy.original_twoFIFO_owned_training_campaign.v2',V2_CAPTURE_HOLD)
    campaign=intake.campaign_v2.audit(campaign_path);plans=[checked(r) for r in campaign['attempts']]
    # v2 writes a growing full-raw checkpoint inside every measurement window.
    # Completion/protocol correctness cannot qualify this perturbed execution
    # as production training. Preserve the old receipts; a new capture/version
    # must be reviewed separately instead of weakening this gate.
    require(campaign.get('schema')!='gsdeploy.original_twoFIFO_owned_training_campaign.v2',V2_CAPTURE_HOLD)
    reviews=[(receipt(p),checked(receipt(p))) for p in review_paths];status=intake.planned_intake(plans,reviews)
    require(status['complete'],'All twelve complete native attempts required before reading resource labels')
    refs={checked(d['inputs']['plan.json'])['attempt_id']:ref for ref,d in reviews};tables=[];sources=[];documents=[]
    for plan in plans:
        ref=refs[plan['attempt_id']];review=checked(ref);macros=macros_v2.extract(ref['path'])
        documents.append(macros)
        raw=checked(review['raw']);context=checked(raw['context'])
        table=table_from_verified(raw,macros,plan,context)
        table.update(raw=review['raw'],context=raw['context'],plan=raw['plan'],training_review=ref,
            capture_sha256=review['raw']['sha256'])
        tables.append(table);sources.extend(macros['sources'])
    intake.complete_documents(plans,documents)
    return tables,strict_references([receipt(campaign_path),campaign['attempts'],sources,implementation_sources(__file__)])


def run(campaign_path,review_paths,output):
    tables,sources=load_complete(campaign_path,review_paths);model=fit_tables(tables)
    folds=[dict(heldout_block_id=b,training_model=fit_tables([t for t in tables if t['capture_id']!=b]),
        heldout_attempt_ids=[t['attempt_id'] for t in tables if t['capture_id']==b],
        independent_prediction_accuracy_validated=False) for b in model['training_block_ids']]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model.update(campaign=receipt(campaign_path),training_reviews=[receipt(p) for p in review_paths],
        sources=sources,whole_block_leaveout=folds)
    require(all(receipt(r['path'])==r for r in sources),'Resource training source changed')
    with (out/'model.json').open('x') as stream:json.dump(model,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt(out/'model.json')


class ResourceTrainingProvider:
    def __init__(self,model_path,target_plan,target_context,*,allow_training_resource_hypothesis=False,scenario='mean'):
        require(allow_training_resource_hypothesis is True,'Explicit training resource hypothesis opt-in required')
        require(scenario in ('mean','max'),'Explicit empirical resource scenario required')
        doc=checked(receipt(model_path));require(doc['schema']==SCHEMA,'New multi-window resource model required')
        for r in doc['sources']:require(receipt(r['path'])==r,'Resource source changed')
        tables,sources=load_complete(doc['campaign']['path'],[r['path'] for r in doc['training_reviews']])
        actual=fit_tables(tables);require(all(doc.get(k)==v for k,v in actual.items()),'Resource law must rederive from native polls')
        require(all(r in doc['sources'] for r in sources),'Complete rederived source closure required')
        identity=physical_execution_identity(target_plan,target_context)
        require(identity==doc['physical_execution_identity'],'Target physical worker/config/priming axes differ')
        self.document=copy.deepcopy(doc);self.scenario=scenario;self.target_plan=copy.deepcopy(target_plan)
        self.target_context=copy.deepcopy(target_context);self.physical_execution_identity=identity

    def power(self,active,*,resident_owners):
        state,_=legacy.key(active,resident_owners);require(sorted(resident_owners)==sorted(self.document['resident_owners']),'Resident roster differs')
        row=self.document['power_states'].get(state);known=bool(row and row['known'])
        return dict(known=known,watts=row[self.scenario] if known else None)

    def memory(self,active,*,resident_owners,ordered_state_history_identity):
        state,_=legacy.key(active,resident_owners);require(sorted(resident_owners)==sorted(self.document['resident_owners']),'Resident roster differs')
        row=self.document['memory_history_states'].get(memory_key(state,ordered_state_history_identity));known=bool(row and row['known'])
        return dict(known=known,peak_memory_mib=row[self.scenario] if known else None,
            inactive_resident_memory_is_zero=False,safe_factor_applied=False,capacity_bound=False)

    def startup(self):return copy.deepcopy(self.document['startup'])


def predict_timeline(timeline,provider,*,target_context,execution_binding):
    from .macro_resource_timeline_v1 import clock
    from .macro_response_queue_reference_v1 import encode
    require(timeline.get('schema')=='gsdeploy.macro_global_resource_timeline.v1','Explicit global resource timeline required')
    identity=physical_execution_identity(provider.target_plan,target_context)
    require(identity==provider.physical_execution_identity,'Target physical execution differs')
    timeline_id=digest(timeline)
    require(execution_binding.get('physical_execution_identity')==identity
        and execution_binding.get('resource_timeline_sha256')==timeline_id
        and execution_binding.get('basis')=='caller_declared_explicit_reference_execution_binding',
        'Explicit timeline and physical execution binding required')
    duration=measurement=0;known_j=0.;peak=None;power_unknown=memory_unknown=0
    initial=execution_binding.get('memory_initial_state');history=previous=None;prefix_peak=None
    fresh=bool(isinstance(initial,dict) and initial.get('basis')=='fresh_after_original_priming')
    if fresh:
        require(set(initial)=={'basis'},'Exact explicit fresh-after-priming state required')
        history=history_start(provider.document['priming_content_identity'])
    elif initial is not None:
        require(isinstance(initial,dict) and set(initial)=={'basis','ordered_state_history_identity',
            'last_state_key','prior_owned_sampled_peak_mib'} and initial['basis']=='explicit_prior_attempt_history'
            and isinstance(initial['ordered_state_history_identity'],str) and len(initial['ordered_state_history_identity'])==64
            and isinstance(initial['last_state_key'],str) and len(initial['last_state_key'])==64
            and legacy.finite(initial['prior_owned_sampled_peak_mib']),
            'Nonfresh case requires explicit prior state history identity, last state and cumulative owned peak')
        history=initial['ordered_state_history_identity'];previous=initial['last_state_key']
        prefix_peak=initial['prior_owned_sampled_peak_mib']
    for row in timeline['slices']:
        delta=clock(row['duration_ns']);overlap=clock(row['measurement_overlap_ns'])
        require(delta>0 and overlap<=delta,'Positive bounded global resource exposure required')
        duration+=delta;measurement+=overlap;members=row['actual_START_to_READY_handlers'];owners=row['resident_owners']
        state,_=legacy.key(members,owners)
        if history is not None and state!=previous:history=next_history(history,state);previous=state
        power=provider.power(members,resident_owners=owners)
        if power['known']:known_j+=power['watts']*float(overlap)/1e9
        else:power_unknown+=overlap
        memory=(provider.memory(members,resident_owners=owners,ordered_state_history_identity=history)
            if history is not None else dict(known=False,peak_memory_mib=None))
        if memory['known']:peak=memory['peak_memory_mib'] if peak is None else max(peak,memory['peak_memory_mib'])
        else:memory_unknown+=delta
    require(duration==clock(timeline['total_wall_ns']) and measurement==clock(timeline['total_measurement_overlap_ns']),
        'Every idle/active/drain slice must remain in the global denominator')
    startup=provider.startup();pre=startup['premeasurement_owned_sampled_peak_mib']
    if prefix_peak is not None:pre=max(pre,prefix_peak) if pre is not None else prefix_peak
    return encode(dict(schema='gsdeploy.twoFIFO_multiwindow_resource_prediction.v3',
        board_energy_joules=known_j if power_unknown==0 else None,partial_known_measurement_joules=known_j,
        peak_memory_mib=max(pre,peak) if memory_unknown==0 and pre is not None and peak is not None else None,
        partial_known_peak_memory_mib=peak,startup_energy_joules=startup['joules'],
        premeasurement_owned_sampled_peak_mib=pre,power_unknown_measurement_duration_ns=power_unknown,
        memory_unknown_observation_duration_ns=memory_unknown,total_observation_duration_ns=duration,
        total_measurement_duration_ns=measurement,physical_execution_identity=identity,
        memory_initial_history_supplied=history is not None,
        memory_initial_history_missing_reason=None if history is not None else 'NO_EXPLICIT_INITIAL_ATTEMPT_HISTORY_DO_NOT_RESET_TO_PRIMING',
        resource_timeline_sha256=timeline_id,execution_binding=copy.deepcopy(execution_binding),
        caller_execution_binding_independently_authenticated=False,startup_amortization_seconds=3600,
        startup_energy_already_amortized=False,safe_factor_already_applied=False,
        resource_prediction_qualified=False,main_queue_provider_eligible=False,
        legacy_v2_capture_training_status=V2_CAPTURE_HOLD,
        validation_labels_read=False,GPU_started=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True);parser.add_argument('--training-reviews',nargs='*',default=[])
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.campaign,args.training_reviews,args.output)))
