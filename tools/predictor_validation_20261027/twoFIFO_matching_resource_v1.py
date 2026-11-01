"""Matching exact15 matching resource model and whole-block donor provider.

Native inputs retain their new schema. Only the existing pure resource table,
whole-block averaging and timeline mathematics are reused. Preparation reads
source metadata only; complete matching exact15 intake precedes resource label access.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_resource_provider_v3 as resource
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_matching_resource_training.v1'
require, receipt, checked = response.require, response.receipt, response.checked
DOMAIN = dict(measurement_seconds=10, fixed_drain_seconds=60,
    physical_context='EXACT_MATCHING_CONTEXT_UNION22',
    state_support='AT_LEAST_TWO_DONOR_WHOLE_BLOCKS',
    memory_support='EXACT_ORDERED_PRIOR_STATE_HISTORY',
    longer_horizon_transfer_qualified=False)
DESIGN = dict(whole_block_count=3, attempts_per_block=5, measured_requests=390,
    history_only_priming_requests=270, training_attempts_per_fold=10,
    heldout_attempts_per_fold=5, minimum_supported_whole_blocks=2,
    scenario_choices=['mean', 'max'], equal_block_aggregation=True,
    memory_key='EXACT_STATE_AND_ORDERED_PRIOR_STATE_PATH',
    memory_history_reset_between_cases=False,
    energy_scope='ORIGINAL_MEASUREMENT_ONLY_EXCLUDES_DRAIN',
    memory_scope='OWNED_RESIDENT_PEAK_STARTUP_THROUGH_CURRENT_DRAIN',
    startup_occurrences_per_attempt=1, startup_amortized_in_provider=False,
    memory_safety_factor_applied_in_provider=False,
    native_schema_relabeling=False, heldout_unit='whole_capture_block_all_five_shards',
    resource_domain=DOMAIN)


legacy=resource.legacy
observed=resource.observed
digest=resource.digest
next_history=resource.next_history
history_start=resource.history_start
memory_key=resource.memory_key
TABLE_SCHEMA=SCHEMA+'.table'


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
    require(plan['resident_owners']==list(response.executor.OWNERS),'Exact physical resident roster required')
    return dict(schema=SCHEMA+'.physical_execution_identity',context=context,
        runtime_physical_config={k:config[k] for k in physical_keys},runtime_policy=plan['runtime_policy'],
        resident_owners=plan['resident_owners'],priming=response.executor.planned_priming_semantics(plan,context)['ordered_inputs'],
        implementation_sources=plan['sources'],source_freeze=plan['source_freeze'],
        plan_hash_and_arrivals_are_provenance_not_physical_axes=True)


def physical_execution_identity(plan,context):return digest(physical_execution_payload(plan,context))


def table_from_verified(raw,macros,plan,context):
    """Pure adapter; public loader authenticates native macros/review first."""
    require(plan['runtime_config']['measurement_seconds']==DOMAIN['measurement_seconds']
        and plan['runtime_config']['drain_timeout_seconds']==DOMAIN['fixed_drain_seconds'],
        'Only matching10+60 resource windows are supported')
    require([w['case_id'] for w in raw['measurement_windows']]==[c['case_id'] for c in plan['cases']]
        and all(w['end_ns']-w['start_ns']==10_000_000_000
            and w['fixed_drain_deadline_ns']-w['end_ns']==60_000_000_000
            for w in raw['measurement_windows']), 'Actual measured resource windows must match the fixed horizon')
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
    table.update(schema=TABLE_SCHEMA,resource_domain=copy.deepcopy(DOMAIN),
        capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],
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
    require(all(t.get('resource_domain')==DOMAIN for t in tables),'Resource tables must retain the observed10+60 domain')
    physical={t['physical_execution_identity'] for t in tables};require(len(physical)==1,'Physical execution axes differ')
    require(len({t['priming_content_identity'] for t in tables})==1,'Exact same priming contents required')
    blocks=sorted({t['capture_id'] for t in tables});require(len(blocks)>=2,'At least two independent whole blocks required')
    require(all({t['attempt_id'] for t in tables if t['capture_id']==b}==
        {b+'-shard-'+str(i) for i in (1,2,3,4,5)} for b in blocks),'All five shards required per training block')
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
    return dict(schema=SCHEMA,resource_domain=copy.deepcopy(DOMAIN),physical_execution_identity=next(iter(physical)),
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


def _predict_supported_timeline(timeline,provider,*,target_context,execution_binding):
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
        legacy_v2_capture_training_status=resource.V2_CAPTURE_HOLD,
        validation_labels_read=False,GPU_started=False))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt(path)


def prepare(fit_manifest_path, output):
    """Predeclare before final matching exact15 exists; do not inspect native labels."""
    manifest_ref = receipt(fit_manifest_path)
    manifest = response.audit(fit_manifest_path)
    require(not Path(manifest['expected_continuation_report_path']).exists(),
            'Resource design must be fixed before final matching exact15 continuation exists')
    require(not (Path(fit_manifest_path).resolve().parent / 'phase12-fit-claim.json').exists(),
            'Resource design must be fixed before native response fitting')
    sources = strict_references([manifest_ref, manifest['sources'], implementation_sources(__file__)])
    for ref in sources:
        response.verify(ref)
    document = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_BEFORE_MATCHING_EXACT15_FINAL',
        fit_manifest=manifest_ref, expected_continuation_report_path=manifest['expected_continuation_report_path'],
        design=copy.deepcopy(DESIGN), sources=sources, resource_fit_run=False,
        native_training_resource_labels_read=False, numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return save(output / 'design.json', document)


def audit(design_path):
    design = checked(receipt(design_path))
    require(design.get('schema') == SCHEMA + '.design'
        and design.get('status') == 'CPU_PREDECLARED_BEFORE_MATCHING_EXACT15_FINAL'
        and design.get('design') == DESIGN and design.get('resource_fit_run') is False
        and design.get('native_training_resource_labels_read') is False
        and design.get('numerical_fit_run') is False and design.get('validation_labels_read') is False
        and design.get('GPU_started') is False, 'Exact predeclared resource design required')
    response.verify(design['fit_manifest'])
    manifest = response.audit(design['fit_manifest']['path'])
    require(design['expected_continuation_report_path'] == manifest['expected_continuation_report_path'],
            'Expected matching exact15 continuation changed')
    expected = strict_references([design['fit_manifest'], manifest['sources'], implementation_sources(__file__)])
    require(design.get('sources') == expected, 'Resource design implementation closure changed')
    for ref in expected:
        response.verify(ref)
    return design


def _fits(tables):
    """Use unchanged mathematics, retaining the complete full/fold contract."""
    require(len(tables) == 15 and len({t['attempt_id'] for t in tables}) == 15,
            'Exactly fifteen distinct resource attempts required')
    full = fit_tables(tables)
    require(len(full['training_block_ids']) == 3 and len(full['training_attempt_ids']) == 15,
            'Exactly three whole resource blocks required')
    folds = []
    for block in full['training_block_ids']:
        donor_tables = [t for t in tables if t['capture_id'] != block]
        donor = fit_tables(donor_tables)
        heldout = sorted(t['attempt_id'] for t in tables if t['capture_id'] == block)
        require(len(donor['training_block_ids']) == 2 and len(donor['training_attempt_ids']) == 10
            and len(heldout) == 5 and not set(heldout) & set(donor['training_attempt_ids']),
            'Every fold must exclude all five heldout attempts')
        folds.append(dict(heldout_block_id=block, training_model=donor,
            heldout_attempt_ids=heldout, independent_prediction_accuracy_validated=False))
    return full, folds


def build_from_bundle(bundle, intake_document):
    """Pure callback after with_verified_bundle; native new schema stays intact."""
    response._validate_bundle(bundle)
    require(intake_document.get('schema') == response.intake.SCHEMA
        and intake_document.get('status') == 'COMPLETE_MATCHING_EXACT15_TRAINING_INTAKE_NOT_FIT'
        and intake_document.get('attempt_count') == 15 and intake_document.get('measured_requests') == 390
        and intake_document.get('priming_requests') == 270
        and intake_document.get('validation_labels_read') is False,
        'Complete true matching exact15 intake required before resource labels')
    implementation_start = implementation_sources(__file__)
    start_sources = strict_references([intake_document['sources'], bundle['review_receipts'],
        implementation_start])
    for ref in start_sources:
        response.verify(ref)
    plans = {p['attempt_id']: p for p in bundle['plans']}
    documents = {d['attempt_id']: d for d in bundle['documents']}
    require(len(documents) == 15 and set(documents) == set(plans), 'Exact fifteen macro identities required')
    reviews = {}
    for ref in bundle['review_receipts']:
        review = checked(ref)
        require(review.get('schema') == response.intake.native_review.SCHEMA
            and review.get('status') == 'PASS_COMPLETE_MATCHING_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
            and review.get('training_feature_extraction_eligible') is True
            and review.get('issues') == [] and review.get('validation_labels_read') is False
            and review.get('matching_manifest') == intake_document['manifest'],
            'Only complete matching native reviews accepted')
        plan = checked(review['inputs']['plan.json'])
        require(plan['attempt_id'] not in reviews and plans.get(plan['attempt_id']) == plan,
                'Reviewed resource plan identity differs')
        reviews[plan['attempt_id']] = (ref, review)
    require(set(reviews) == set(plans), 'Every resource attempt must have one reviewed native source')
    tables, sources = [], start_sources
    for aid, plan in plans.items():
        review_ref, review = reviews[aid]
        raw_ref = response.verify(review['raw'])
        raw = checked(raw_ref)
        require(raw.get('schema') == response.executor.RAW_SCHEMA and raw.get('backend') == 'native'
            and raw.get('status') == 'completed' and raw.get('split') == 'training'
            and raw.get('attempt_id') == aid and raw.get('capture_id') == plan['capture_id']
            and raw.get('execution_id') == plan['execution_id']
            and checked(raw['plan']) == plan and raw['context'] == plan['context'],
            'Exact source-matching raw/context identity required; no old raw-schema relabeling')
        context = checked(raw['context'])
        doc = documents[aid]
        require(doc.get('schema') == response.adapter.MACRO_SCHEMA
            and doc.get('capture_id') == plan['capture_id'] and doc.get('execution_id') == plan['execution_id']
            and doc.get('matching_manifest') == intake_document['manifest'],
            'Exact source-matching macro identity required')
        table = table_from_verified(raw, doc, plan, context)
        table.update(raw=raw_ref, context=raw['context'], plan=raw['plan'],
            training_review=review_ref, capture_sha256=raw_ref['sha256'])
        tables.append(table)
        sources = strict_references([sources, raw_ref, raw['context'], raw['plan'], review['sources'], doc['sources']])
    require(sum(t['measured_requests'] for t in tables) == 390
        and sum(t['priming_requests'] for t in tables) == 270,
        'Resource tables must retain 390 measured requests and 270 priming history requests')
    full, folds = _fits(tables)
    require(implementation_sources(__file__) == implementation_start, 'Resource implementation closure changed')
    for ref in sources:
        response.verify(ref)
    return dict(resource_model=full, whole_block_leaveout=folds, sources=sources,
        denominator=copy.deepcopy(bundle['denominator']), measured_requests=390,
        history_only_priming_requests=270, training_attempts=15, resource_fit_run=True,
        native_training_resource_labels_read=True, response_numerical_fit_run=False,
        validation_labels_read=False, GPU_started=False, resource_prediction_qualified=False,
        main_queue_provider_eligible=False, final_gain_eligible=False)


def run(design_path, output, *, fit_enabled=False):
    """Gate all native label access on complete matching exact15 and explicit opt-in."""
    design_ref = receipt(design_path)
    design = audit(design_path)
    manifest = response.audit(design['fit_manifest']['path'])
    final_path = Path(design['expected_continuation_report_path'])
    result = dict(schema=SCHEMA, design=design_ref, status='WAITING_FOR_MATCHING_EXACT15_CONTINUATION_FINAL',
        resource_fit_run=False, native_training_resource_labels_read=False,
        validation_labels_read=False, GPU_started=False, resource_prediction_qualified=False,
        main_queue_provider_eligible=False, final_gain_eligible=False,
        sources=strict_references([design_ref, design['sources']]))
    if final_path.is_file():
        final_ref = receipt(final_path)
        result['continuation'] = final_ref
        if not response._complete_final(manifest, checked(final_ref)):
            result['status'] = 'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL'
        elif not fit_enabled:
            result['status'] = 'READY_EXPLICIT_RESOURCE_FIT_REQUIRED'
        else:
            save(Path(design_path).resolve().parent/'resource-fit-claim.json',
                dict(design=design_ref,continuation=final_ref,no_retries=True))
            def callback(bundle, intake_doc):
                fitted = build_from_bundle(bundle, intake_doc)
                fitted['intake_evidence'] = copy.deepcopy(intake_doc)
                return fitted
            result.update(response.with_verified_bundle(design['fit_manifest']['path'], final_path, callback))
            result.update(status='COMPLETE_MATCHING_RESOURCE_TRAINING_NOT_QUALIFICATION')
        result['sources'] = strict_references([result['sources'], design_ref, design['sources'], final_ref])
    audit(design_path)
    response.verify(design_ref)
    for ref in result['sources']:
        response.verify(ref)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return save(output / 'model.json', result)


class FoldResourceTrainingProvider(resource.ResourceTrainingProvider):
    """Reauthenticate matching exact15 and rederive full model plus all three folds."""
    def __init__(self, model_path, target_plan, target_context, *, heldout_block_id,
                 allow_training_resource_hypothesis=False, scenario='mean'):
        require(allow_training_resource_hypothesis is True, 'Explicit training resource hypothesis opt-in required')
        require(scenario in ('mean', 'max'), 'Explicit empirical resource scenario required')
        model_ref = receipt(model_path)
        doc = checked(model_ref)
        require(doc.get('schema') == SCHEMA
            and doc.get('status') == 'COMPLETE_MATCHING_RESOURCE_TRAINING_NOT_QUALIFICATION'
            and doc.get('resource_fit_run') is True and doc.get('validation_labels_read') is False
            and doc.get('GPU_started') is False, 'Complete new-schema resource model required')
        for ref in doc['sources']:
            response.verify(ref)
        design = audit(doc['design']['path'])
        response.verify(doc['design'])
        response.verify(doc['continuation'])
        require(doc['continuation']['path'] == design['expected_continuation_report_path'],
                'Resource model continuation differs from design')
        def rederive(bundle, intake_doc):
            require(intake_doc == doc['intake_evidence'], 'Resource native intake changed')
            return build_from_bundle(bundle, intake_doc)
        actual = response.with_verified_bundle(design['fit_manifest']['path'],
            doc['continuation']['path'], rederive)
        require(all(doc.get(k) == v for k, v in actual.items() if k != 'sources'),
                'Full resource law and all three folds must rederive from native polls')
        expected_sources = strict_references([actual['sources'], doc['design'], design['sources'], doc['continuation']])
        require(doc['sources'] == expected_sources, 'Resource model source closure differs')
        selected = next((f for f in actual['whole_block_leaveout'] if f['heldout_block_id'] == heldout_block_id), None)
        require(selected is not None, 'Exact whole heldout block required')
        require(target_plan.get('capture_id') == heldout_block_id
            and target_plan.get('attempt_id') in selected['heldout_attempt_ids'],
            'Target must belong to the selected five-attempt heldout block')
        identity = physical_execution_identity(target_plan, target_context)
        require(identity == selected['training_model']['physical_execution_identity'],
                'Target physical worker/config/priming axes differ')
        self.document = copy.deepcopy(selected['training_model'])
        self.scenario = scenario
        self.target_plan, self.target_context = copy.deepcopy(target_plan), copy.deepcopy(target_context)
        self.physical_execution_identity = identity
        self.fold_binding = dict(schema=SCHEMA + '.fold_provider', full_native_model=model_ref,
            selected_fold_model_sha256=resource.digest(self.document), heldout_block_id=heldout_block_id,
            training_block_ids=copy.deepcopy(self.document['training_block_ids']),
            training_attempt_ids=copy.deepcopy(self.document['training_attempt_ids']),
            heldout_attempt_ids=copy.deepcopy(selected['heldout_attempt_ids']),
            sources=strict_references([doc['sources'], model_ref, implementation_sources(__file__)]),
            all_three_folds_rederived=True, complete_fifteen_native_attempt_gate=True,
            startup_recomputed_from_only_ten_donor_attempts=True,
            resource_prediction_qualified=False, main_queue_provider_eligible=False,
            final_gain_eligible=False, validation_labels_read=False, GPU_started=False)
        for ref in self.fold_binding['sources']:
            response.verify(ref)


def predict_timeline(timeline, provider, *, target_context, execution_binding):
    require(isinstance(provider, FoldResourceTrainingProvider), 'Explicit matching whole-block resource provider required')
    from .macro_resource_timeline_v1 import clock
    from .macro_response_queue_reference_v1 import encode
    require(timeline.get('schema')=='gsdeploy.macro_global_resource_timeline.v1',
        'Explicit global resource timeline required')
    duration=measurement=0
    for row in timeline['slices']:
        delta=clock(row['duration_ns']);overlap=clock(row['measurement_overlap_ns'])
        require(delta>0 and overlap<=delta,'Positive bounded global resource exposure required')
        duration+=delta;measurement+=overlap
    require(duration==clock(timeline['total_wall_ns'])
        and measurement==clock(timeline['total_measurement_overlap_ns']),
        'Every idle/active/drain slice must remain in the global denominator')
    identity=physical_execution_identity(provider.target_plan,target_context)
    timeline_id=digest(timeline)
    require(execution_binding.get('physical_execution_identity')==identity
        and execution_binding.get('resource_timeline_sha256')==timeline_id
        and execution_binding.get('basis')=='caller_declared_explicit_reference_execution_binding',
        'Explicit timeline and physical execution binding required')
    window=timeline.get('original_window');end=timeline.get('original_drain_end_ns')
    horizon_known=(isinstance(window,list) and len(window)==2 and end is not None)
    if horizon_known:
        left,right=map(clock,window);drain_end=clock(end)
        horizon_known=(right-left==10_000_000_000 and drain_end-right==60_000_000_000
            and duration==drain_end-left and measurement==right-left)
    physical_known=identity==provider.physical_execution_identity
    domain_known=provider.document.get('resource_domain')==DOMAIN
    if not (horizon_known and physical_known and domain_known):
        result=encode(dict(schema='gsdeploy.twoFIFO_multiwindow_resource_prediction.v3',
            board_energy_joules=None,partial_known_measurement_joules=0.,peak_memory_mib=None,
            partial_known_peak_memory_mib=None,startup_energy_joules=None,
            premeasurement_owned_sampled_peak_mib=None,
            power_unknown_measurement_duration_ns=measurement,
            memory_unknown_observation_duration_ns=duration,total_observation_duration_ns=duration,
            total_measurement_duration_ns=measurement,physical_execution_identity=identity,
            resource_timeline_sha256=timeline_id,execution_binding=copy.deepcopy(execution_binding),
            unknown_reason='UNKNOWN_RESOURCE_HORIZON_OR_PHYSICAL_CONTEXT_DOMAIN',
            resource_domain=copy.deepcopy(DOMAIN),horizon_transfer_qualified=False,
            resource_prediction_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
            startup_energy_already_amortized=False,safe_factor_already_applied=False,
            validation_labels_read=False,GPU_started=False))
        result['resource_fold_binding']=copy.deepcopy(provider.fold_binding)
        return result
    result = _predict_supported_timeline(timeline, provider, target_context=target_context,
        execution_binding=execution_binding)
    result['resource_domain']=copy.deepcopy(DOMAIN)
    result['resource_fold_binding'] = copy.deepcopy(provider.fold_binding)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--fit-manifest', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('audit'); p.add_argument('--design', required=True)
    p = sub.add_parser('run'); p.add_argument('--design', required=True); p.add_argument('--output', required=True)
    p.add_argument('--fit', action='store_true', default=False)
    args = parser.parse_args()
    value = prepare(args.fit_manifest, args.output) if args.command == 'prepare' else (
        audit(args.design) if args.command == 'audit' else run(args.design, args.output, fit_enabled=args.fit))
    print(json.dumps(value))
