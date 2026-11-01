"""Prepare eight exact-public181 resource training attempts, CPU only.

The four frozen fresh181 plans remain validation references. This module has
no collector, GPU entry point, fit, model promotion, or permission-Boolean
escape. Its pure producer helpers check shape/lineage, not native authenticity.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import uuid

from . import twoFIFO_fresh181_execute_v1 as fresh
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .stage_response_features_v1 import checked, receipt
from .twoFIFO_training_manifest_v1 import digest, require

SCHEMA = 'gsdeploy.twoFIFO_public181_resource_campaign.v1'
MEASUREMENT_NS = 90_000_000_000
DRAIN_NS = 120_000_000_000
TUPLE_FIELDS = ('startup_joules', 'premeasurement_owned_sampled_peak_mib',
                'measurement_through_drain_owned_sampled_peak_mib', 'measurement_board_joules')
COLLECTION = dict(measurement_ns=MEASUREMENT_NS, fixed_drain_ns=DRAIN_NS,
    residents_per_attempt=4, startup_occurrences_per_attempt=1, priming_requests_per_attempt=18,
    fresh_workers_per_attempt=True, preserve_original_sender_stdin_FIFO=True,
    global_admission_cap=None, checkpoint_during_measurement_or_drain=False,
    monitor='live_owned_launcher_descendants_NVML_board', nominal_poll_interval_ns=20_000_000,
    monitor_begins_before_startup=True, monitor_ends_before_worker_close=True,
    energy_scope='measurement_only_board_integral_plus_startup_to_first_arrival_once',
    startup_includes_priming_and_precohort_wait=True, startup_amortization_seconds=3600,
    startup_amortization_applied_in_producer=False, memory_safety_factor_applied_in_producer=False,
    memory_scope='owned_resident_sampled_peak_startup_through_fixed_drain',
    memory_combination='max(premeasurement_peak,measurement_through_drain_peak)',
    drain_energy_is_diagnostic_only=True, retain_every_offered_request=True,
    retain_every_owned_poll=True, no_retry_or_overwrite=True)
PRODUCER_INTERFACE = dict(schema=SCHEMA+'.producer_interface',
    support='EXACT_FOUR_FROZEN_PUBLIC_CASES_ONLY', request_schema=SCHEMA+'.forecast_request',
    training_row_schema=SCHEMA+'.training_joint_tuple',
    output_schema=SCHEMA+'.paired_resource_draws', joint_tuple_fields=list(TUPLE_FIELDS),
    initial_memory_modes=['training_donor_joint_startup_tuple', 'actual_owned_prefix_completed_before_cutoff'],
    default_initial_memory_mode='training_donor_joint_startup_tuple',
    resource_draw_unit='one_complete_training_block_with_all_four_candidates',
    resource_block_weights='equal_two_training_blocks',
    startup_P0_future_peak_energy_must_share_donor_attempt=True,
    heldout_future_rows_or_polls_allowed=False, split_or_capture_hashes_are_features=False,
    exact_ordered_future_history_lookup_required=False,
    fixed_case_resource_marginal_is_not_generated_queue_joint_law=True,
    queue_resource_coupling_required_for_probability_claims=True,
    missing_term_or_probability_mass_policy='UNKNOWN_FULL_U_NO_RENORMALIZATION',
    native_authentication='REQUIRES_SEPARATE_EXECUTION_REVIEW_NOT_A_CALLER_BOOLEAN')


def verify(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'} and receipt(ref['path'])==ref,
            'Immutable source or campaign artifact changed')
    return ref


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt(path)


def public_case_payload(plan):
    """Exact known public case; absolute clocks and execution IDs are provenance."""
    require(plan['measurement_ns']==MEASUREMENT_NS and plan['drain_ns']==DRAIN_NS,
            'Original90+120 horizon required')
    require(len(plan['cases'])==1 and len(plan['cases'][0]['arrivals'])==181,
            'One complete public181 case required')
    return dict(candidate_id=plan['candidate_id'], candidate=plan['candidate'],
        cases=plan['cases'], runtime_config=plan['runtime_config'],
        measurement_ns=plan['measurement_ns'], drain_ns=plan['drain_ns'],
        objective=plan['objective'], public_contract=plan['public_contract'],
        context_sha256=plan['context']['sha256'], resident_owners=plan['resident_owners'],
        runtime_policy=plan['runtime_policy'], priming_semantics=plan['priming_semantics'],
        prepared_pool=plan['prepared_pool'])


def planned_worker_registry(context):
    """Same command/env derivation as the frozen fresh181 collector, without Popen."""
    registry={}
    for owner in fresh.OWNERS:
        binding=copy.deepcopy(context['worker_bindings'][owner.split('#')[0]])
        binding['command'][1]=str(Path(fresh.__file__).with_name('semantic_progress_worker_v1.py'))
        binding['env'].update(context['threads'])
        binding['env']['PYTHONUNBUFFERED']='1'
        registry[owner]=binding
    return registry


def _inputs(fresh_manifest_path):
    manifest_ref=receipt(fresh_manifest_path)
    manifest=fresh.audit(fresh_manifest_path)
    require(manifest['candidate_count']==4 and len(manifest['attempts'])==4,
            'Exactly four frozen fresh validation references required')
    plans=[checked(verify(row['plan'])) for row in manifest['attempts']]
    require(all(p['split']=='validation' for p in plans), 'Existing fresh plans must remain validation')
    context=checked(verify(manifest['context']))
    require(all(p['context']==manifest['context'] and p['priming_semantics']==plans[0]['priming_semantics']
        and p['runtime_policy']==plans[0]['runtime_policy'] and p['resident_owners']==plans[0]['resident_owners']
        for p in plans), 'One exact context/priming/backend roster required')
    require(len({p['candidate_id'] for p in plans})==4, 'Distinct complete candidate roster required')
    for plan in plans:
        public_case_payload(plan)
        require(len(plan['priming'])==18, 'All eighteen original priming inputs required')
    return manifest_ref,manifest,plans,context


def _build(fresh_manifest_path, namespace, output,precohort_lead_seconds):
    require(type(namespace) is str and re.fullmatch('[a-f0-9]{32}',namespace),
            'Explicit independent128-bit training namespace required')
    require(type(precohort_lead_seconds) in (int,float) and math.isfinite(precohort_lead_seconds)
        and 1<=precohort_lead_seconds<=1800,
        'Explicit new collection/prediction budget in1..1800 seconds required; never inherit unresolved60s')
    parent_ref,parent,validation,context=_inputs(fresh_manifest_path)
    output=Path(output).resolve()
    implementation=implementation_sources(__file__)
    registry=planned_worker_registry(context)
    transfer=dict(schema=SCHEMA+'.source_transfer_declaration', parent_fresh_manifest=parent_ref,
        parent_context=parent['context'], parent_source_closure=parent['sources'],
        target_preparation_sources=implementation, planned_worker_registry=registry,
        planned_worker_registry_sha256=digest(registry), context_bytes_changed=False,
        collection_protocol=COLLECTION, target_precohort_lead_seconds=precohort_lead_seconds,
        parent_reference_precohort_lead_seconds=parent['lead_seconds'],
        parent_validation_native_execution_reusable=False,
        validation_backend_must_be_prepared_with_target_protocol=True,
        old_matching_physical_identity_reused=False,
        basis='DERIVED_SAME_WORKER_COMMANDS_CONTEXT_PRIMING_NEW_ROLE_AND_COLLECTION_PROTOCOL',
        execution_review_requirements=['exact executed worker command/env/source and READY specification',
            'exact context and planned physical/runtime policy',
            'live PID/start-ticks launcher ancestry and every owned monitor poll',
            'actual ordered eighteen successful priming inputs and fixed90+120 boundaries',
            'no protected-interval checkpoint and monitor stops before worker close'],
        native_backend_implemented=False, native_source_transfer_authenticated=False)
    protocol_identity=digest(transfer)
    sources=strict_references([parent_ref,parent['sources'],[a['plan'] for a in parent['attempts']],implementation])
    for source in sources:
        verify(source)
    reserved={a['execution_id'] for a in parent['attempts']}
    matching=checked(verify(parent['matching_producer']))
    reserved.update(a['execution_id'] for a in matching['attempts'])
    blocks=[];attempts=[]
    for block_index in (1,2):
        block_id='public181-resource-'+namespace+'-block-'+str(block_index)
        ordered=validation if block_index==1 else list(reversed(validation))
        blocks.append(dict(block_id=block_id,split='training',candidate_order=[p['candidate_id'] for p in ordered]))
        for index,reference in enumerate(ordered):
            cid=reference['candidate_id']
            aid=block_id+'-'+cid
            execution_id=uuid.uuid5(uuid.NAMESPACE_URL,SCHEMA+':'+namespace+':'+protocol_identity+':'+str(block_index)+':'+cid).hex
            require(execution_id not in reserved, 'Training execution reused matching or fresh validation identity')
            reserved.add(execution_id)
            priming=[dict(row,request_id=aid+':priming:'+str(i)) for i,row in enumerate(reference['priming'])]
            plan=dict(schema=SCHEMA+'.training_attempt', split='training',namespace=namespace,
                capture_id=block_id,attempt_id=aid,execution_id=execution_id,candidate_id=cid,
                candidate_ordinal_in_block=index,reference_validation_plan=receipt(
                    next(a['plan']['path'] for a in parent['attempts'] if a['candidate_id']==cid)),
                public_case=public_case_payload(reference),public_case_sha256=digest(public_case_payload(reference)),
                context=reference['context'],prepared_pool=reference['prepared_pool'],
                resident_owners=reference['resident_owners'],runtime_policy=reference['runtime_policy'],
                runtime_config=reference['runtime_config'],priming=priming,
                priming_semantics=reference['priming_semantics'],cases=reference['cases'],
                measurement_ns=MEASUREMENT_NS,drain_ns=DRAIN_NS,collection_contract=COLLECTION,
                precohort_lead_seconds=precohort_lead_seconds,collection_protocol_identity=protocol_identity,
                source_transfer_declaration_sha256=digest(transfer),
                historical_public_request_ids_retained=True,request_identity_scope='execution_id_and_request_id',
                sources=sources,training_labels_read=False,GPU_started=False,numerical_fit_run=False,
                native_executor_entrypoint=None)
            attempts.append(dict(plan=plan,path=output/aid/'plan.json'))
    return dict(parent_ref=parent_ref,parent=parent,validation=validation,namespace=namespace,
        blocks=blocks,attempts=attempts,source_transfer=transfer,sources=sources,
        collection_protocol_identity=protocol_identity,precohort_lead_seconds=precohort_lead_seconds)


def _manifest(built):
    return dict(schema=SCHEMA+'.manifest',status='CPU_PREPARED_EXACT_PUBLIC181_RESOURCE_TRAINING',
        fresh_validation_manifest=built['parent_ref'],namespace=built['namespace'],
        training_blocks=built['blocks'],training_attempts=[dict(plan=receipt(a['path']),
            attempt_id=a['plan']['attempt_id'],execution_id=a['plan']['execution_id'],
            candidate_id=a['plan']['candidate_id'],capture_id=a['plan']['capture_id']) for a in built['attempts']],
        independent_validation_references=built['parent']['attempts'],
        source_transfer=built['source_transfer'],collection_protocol_identity=built['collection_protocol_identity'],
        target_precohort_lead_seconds=built['precohort_lead_seconds'],
        collection_contract=COLLECTION,producer_interface=PRODUCER_INTERFACE,sources=built['sources'],
        train_attempts=8,validation_attempts=4,training_measured_requests=8*181,
        training_priming_requests=8*18,validation_measured_requests=4*181,
        validation_priming_requests=4*18,training_plus_validation_attempts=12,
        fixed_candidate_count=4,training_block_count=2,
        training_role='ONLY_NEW_EIGHT_NATIVE_ATTEMPTS',validation_role='FROZEN_FRESH_FOUR_PLANS_UNMODIFIED',
        support='EXACT_FOUR_PUBLIC_CASES_NO_DFS_OR_NEW_CALENDAR_TRANSFER',
        training_labels_read=False,validation_labels_read=False,GPU_started=False,numerical_fit_run=False,
        native_executor_implemented=False,resource_horizon_empirically_validated=False,
        validation_native_backend_requires_new_preparation=True,
        full_U_prediction_evaluated=False,final_gain_eligible=False)


def prepare(fresh_manifest_path,namespace,output,*,precohort_lead_seconds):
    output=Path(output).resolve()
    require(not output.exists(), 'A preparation cannot overwrite existing evidence')
    built=_build(fresh_manifest_path,namespace,output,precohort_lead_seconds)
    output.mkdir(parents=True,exist_ok=False)
    for attempt in built['attempts']:
        attempt['path'].parent.mkdir()
        save(attempt['path'],attempt['plan'])
    result=save(output/'manifest.json',_manifest(built))
    audit(result['path'])
    return result


def audit(manifest_path):
    document=checked(receipt(manifest_path))
    require(document.get('schema')==SCHEMA+'.manifest', 'Exact new public181 resource campaign required')
    verify(document['fresh_validation_manifest'])
    built=_build(document['fresh_validation_manifest']['path'],document['namespace'],
        Path(manifest_path).resolve().parent,document['target_precohort_lead_seconds'])
    for attempt in built['attempts']:
        require(checked(receipt(attempt['path']))==attempt['plan'], 'Rehashed training plan changed original public/backend/role contract')
    require(document==_manifest(built), 'Exact immutable eight-training/four-validation campaign differs')
    return document


def forecast_request(campaign, candidate_id, *, queue_model_sources, resource_model_source):
    """Reviewable public producer input shape; contains no heldout raw/path.

    This pure builder is not a native review or a runtime authorization gate.
    The caller must pass a campaign returned by ``audit`` for artifact use.
    """
    require(campaign.get('schema')==SCHEMA+'.manifest', 'Explicit campaign required')
    candidates={a['candidate_id'] for a in campaign['training_attempts']}
    require(candidate_id in candidates, 'Resource producer supports only the frozen four candidates')
    reference=next(a for a in campaign['independent_validation_references'] if a['candidate_id']==candidate_id)
    plan=checked(verify(reference['plan']))
    for source in list(queue_model_sources)+[resource_model_source]:
        verify(source)
    return dict(schema=SCHEMA+'.forecast_request',candidate_id=candidate_id,
        public_case=public_case_payload(plan),public_case_sha256=digest(public_case_payload(plan)),
        collection_protocol_identity=campaign['collection_protocol_identity'],
        training_attempts=campaign['training_attempts'],queue_model_sources=queue_model_sources,
        resource_model_source=resource_model_source,
        memory_origin=dict(basis='training_donor_joint_startup_tuple'),
        resource_tuple_fields=list(TUPLE_FIELDS),target_future_labels_read=False,
        native_authentication='NOT_ESTABLISHED_BY_INTERFACE_BUILDER')


def paired_resource_draws(rows,campaign):
    """Pure eight-row interface checker preserving each complete joint tuple.

    No fitted probabilities, source authentication, queue/resource independence
    or native collection qualification is established by this table helper.
    """
    expected={a['attempt_id']:a for a in campaign['training_attempts']}
    require(type(rows) is list and len(rows)==8, 'All eight training joint tuples required')
    require(all(type(row) is dict and set(row)=={'schema','split','attempt_id','capture_id',
        'execution_id','candidate_id','plan','collection_protocol_identity','public_case_sha256',
        'joint_tuple','native_review','raw'} for row in rows), 'Closed training tuple interface required')
    require(all(row['schema']==SCHEMA+'.training_joint_tuple' and row['split']=='training'
        for row in rows), 'Every donor role must qualify before any joint resource label access')
    require({row['attempt_id'] for row in rows}==set(expected)
        and len({row['attempt_id'] for row in rows})==8,
        'Full distinct eight-training denominator required before resource labels')
    seen=set();by_block={}
    heldout={a['execution_id'] for a in campaign['independent_validation_references']}
    for row in rows:
        require(row['schema']==SCHEMA+'.training_joint_tuple' and row['split']=='training',
                'Training role must precede joint resource label access')
        aid=row['attempt_id'];require(aid in expected and aid not in seen, 'Unknown, missing or duplicated training attempt')
        seen.add(aid);declared=expected[aid]
        require(all(row[field]==declared[field] for field in ('capture_id','execution_id','candidate_id','plan'))
            and row['execution_id'] not in heldout, 'Heldout/future or foreign tuple cannot become a donor')
        require(row['collection_protocol_identity']==campaign['collection_protocol_identity'], 'Resource backend protocol differs')
        plan=checked(verify(declared['plan']))
        require(row['public_case_sha256']==plan['public_case_sha256'], 'Resource tuple has different calendar/routes/context')
        values=row['joint_tuple']
        require(type(values) is dict and set(values)==set(TUPLE_FIELDS)
            and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values.values()),
            'Complete finite unamortized unsafened joint resource tuple required')
        require(type(row['native_review']) is dict and type(row['raw']) is dict,
                'Source receipts required; authenticity still needs native re-review')
        # A structural check binds exact bytes but cannot certify native execution.
        verify(row['native_review']);verify(row['raw'])
        by_block.setdefault(row['capture_id'],{})[row['candidate_id']]=dict(
            donor_attempt_id=aid,donor_execution_id=row['execution_id'],joint_tuple=copy.deepcopy(values),
            native_review=row['native_review'],raw=row['raw'])
    require(seen==set(expected) and len(by_block)==2
        and all(set(value)=={a['candidate_id'] for a in expected.values()} for value in by_block.values()),
        'Each donor whole block must retain all four candidates')
    return dict(schema=SCHEMA+'.paired_resource_draws',draws=[dict(draw_id=block,block_id=block,
        weight=dict(numerator=1,denominator=2),candidates=by_block[block]) for block in sorted(by_block)],
        startup_energy_already_amortized=False,safe_factor_already_applied=False,
        same_attempt_joint_tuple_preserved=True,training_attempts=8,
        support='EXACT_FOUR_PUBLIC_CASES_ONLY',native_authentication='NOT_ESTABLISHED_BY_STRUCTURAL_HELPER',
        queue_resource_joint_law_qualified=False,probability_calibrated=False,
        validation_labels_read=False,GPU_started=False,full_U_accuracy_evaluated=False)


def resource_bridge_payload(draw,candidate_id,*,window_start_ns,resident_owners):
    """Project one intact donor tuple to the unchanged original scorer interface.

    This numerical projection makes no source/probability qualification claim.
    In particular, never average P0 and future peak separately before max().
    """
    require(type(window_start_ns) is int and window_start_ns>=0, 'Exact integer forecast origin required')
    require(type(resident_owners) is list and len(set(resident_owners))==len(resident_owners)==4,
            'All four resident owners required')
    candidate=draw['candidates'].get(candidate_id)
    require(candidate is not None, 'Frozen candidate absent from complete donor block')
    values=candidate['joint_tuple']
    require(type(values) is dict and set(values)==set(TUPLE_FIELDS)
        and all(type(value) in (int,float) and math.isfinite(value) and value>=0
                for value in values.values()), 'Complete finite joint tuple required')
    return dict(original_window_ns=[window_start_ns,window_start_ns+MEASUREMENT_NS],
        original_drain_end_ns=window_start_ns+MEASUREMENT_NS+DRAIN_NS,
        resident_owners=list(resident_owners),startup_energy_joules=values['startup_joules'],
        startup_through_phase_start_owned_peak_mib=values['premeasurement_owned_sampled_peak_mib'],
        phase_start_through_drain_owned_peak_mib=values['measurement_through_drain_owned_sampled_peak_mib'],
        measurement_board_energy_joules=values['measurement_board_joules'],
        measurement_energy_includes_drain=False,memory_safety_factor_already_applied=False,
        startup_energy_already_amortized=False,
        source_evidence=[candidate['native_review'],candidate['raw'],
            dict(donor_attempt_id=candidate['donor_attempt_id'],block_id=draw['block_id'],
                 numerical_projection_is_native_authentication=False)])


def audit_causal_prefix_polls(polls,*,cutoff_ns,forecast_origin_ns):
    """Validate clocks before looking at any owned-memory value."""
    require(type(cutoff_ns) is int and type(forecast_origin_ns) is int
        and cutoff_ns<forecast_origin_ns, 'Strict pre-origin causal cutoff required')
    for poll in polls:
        require(type(poll) is dict and set(poll)=={'poll_begin_ns','poll_end_ns','sample_ns','owned_mib'},
                'Closed causal owned-poll interface required')
        begin,end,at=(poll[k] for k in ('poll_begin_ns','poll_end_ns','sample_ns'))
        require(all(type(t) is int for t in (begin,end,at)) and 0<=begin<=at<=end<=cutoff_ns,
                'Heldout future or boundary-crossing poll cannot initialize memory')
    require(polls and all(type(p['owned_mib']) in (int,float) and math.isfinite(p['owned_mib'])
        and p['owned_mib']>=0 for p in polls), 'Complete nonnegative causal owned values required')
    return max(p['owned_mib'] for p in polls)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-validation-manifest',required=True)
    parser.add_argument('--namespace',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--precohort-lead-seconds',required=True,type=float)
    args=parser.parse_args()
    print(json.dumps(prepare(args.fresh_validation_manifest,args.namespace,args.output,
        precohort_lead_seconds=args.precohort_lead_seconds)))
