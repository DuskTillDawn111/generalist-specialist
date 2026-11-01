"""Typed matching-law replay for an owned fresh181 pre-cohort prefix.

CPU-only, no fit and no target future labels. This module does not change the
frozen fresh181 native HOLD. A future collector can release a validation gate
only by independently replaying this typed report, not by reading its Booleans.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
import math
import json
import time
from pathlib import Path

from . import twoFIFO_matching_phase12_v1 as response
from . import twoFIFO_matching_joint_law_v1 as joint
from . import twoFIFO_matching_execute_v1 as matching
from . import twoFIFO_fresh181_execute_v1 as fresh
from . import twoFIFO_fresh181_prefix_v1 as prefix
from . import twoFIFO_generated_wholeblock_evaluation_v1 as generated
from . import twoFIFO_prospective_forecast_v1 as prospective
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .macro_response_queue_reference_v1 import encode
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.twoFIFO_matching_typed_prospective_forecast_bridge.v1'
require=response.require
receipt=response.receipt
checked=response.checked
verify=response.verify
PHYSICAL_KEYS=('deployment','startup_timeout_seconds','shutdown_timeout_seconds','max_pending_per_model','idle_residency')
OFFERED_KEYS=('workload_rps','routing_rps','warmup_seconds','measurement_seconds','arrival_process','random_seed','routing_source','drain_timeout_seconds')
ASSUMPTIONS=dict(
    internal_work_template_stationary_across_changed_schedule=True,
    fitted_visible_phase_rate_response_transfers_to_original181_schedule=True,
    donor_notice_slacks_transfer_to_fresh_owned_stdout_lifetimes=True,
    paired_transport_feasible_set_transfers_but_is_not_fitted_probability_law=True,
    conditional_request_vectors_independent_given_common_whole_block=True,
    finite_slots_and_equal_whole_block_weights_predeclared=True,
    relative_tracked_message_sequence_excludes_startup_READY=True,
    shared_source_identity_does_not_certify_assumptions_or_accuracy=True)


def required_keys(contract):
    keys=sorted({(candidate['routes'][r['request_id']],r['task'],r['pool_id'])
        for candidate in contract['candidates'] for r in contract['trace']})
    require(len(keys)==16,'Original181 must retain all sixteen routed exact-input keys')
    return keys


def physical_axes(plans,target,prefix_document):
    """Producer-specific axes; never force new source lists into old hash."""
    require(len(plans)==15,'All fifteen authenticated matching plans required')
    context=checked(verify(target['context']));expected=matching.planned_priming_semantics(target,context)
    require(prefix_document['actual_priming_semantics_sha256']==expected['sha256'],
        'Actual fresh prefix priming differs from its declared ordered content')
    rows=[]
    for plan in plans:
        donor=checked(verify(plan['context']))
        require(plan['context']['sha256']==target['context']['sha256'] and donor==context,
            'Whole matching worker context bytes differ across source and target')
        require(set(plan['runtime_config'])==set(target['runtime_config'])==set(PHYSICAL_KEYS+OFFERED_KEYS),
            'Unreviewed runtime knob cannot disappear from the producer-specific bridge')
        require(all(plan['runtime_config'][k]==target['runtime_config'][k] for k in PHYSICAL_KEYS)
            and plan['runtime_policy']==target['runtime_policy']
            and plan['resident_owners']==target['resident_owners']==list(fresh.OWNERS),
            'Physical runtime, policy or resident axes differ')
        donor_prime=matching.planned_priming_semantics(plan,donor)
        require(donor_prime['sha256']==expected['sha256'],'Ordered matching priming content differs')
        # The matching plan is separately authenticated, as is the entire fresh
        # closure. All immutable donor sources must remain explicitly bound;
        # extra collector/model sources are reported, not hidden in a hash.
        require(all(ref in target['sources'] for ref in plan['sources']),
            'A matching backend source is absent from the fresh executable binding')
        rows.append(dict(attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],
            context_sha256=plan['context']['sha256'],priming_semantics_sha256=donor_prime['sha256'],
            runtime_physical={k:copy.deepcopy(plan['runtime_config'][k]) for k in PHYSICAL_KEYS}))
    return dict(schema=SCHEMA+'.physical_axes',all15_donor_axes=rows,
        full_context_bytes_equal=True,worker_bindings_spec_command_env_checkpoint_equal=True,
        threads_device_protocol_equal=True,ordered_priming_content_equal=True,
        actual_target_priming_semantics_sha256=expected['sha256'],
        new_collector_and_model_sources_additional=True,old_physical_identity_hash_reused=False,
        changed_offered_axes=[k for k in OFFERED_KEYS if any(p['runtime_config'][k]!=target['runtime_config'][k] for p in plans)],
        request_and_capture_ids_are_provenance=True)


def support(law,contract):
    keys=required_keys(contract);blocks=law['blocks']
    require(len(blocks)==3 and len(set(blocks))==3 and law.get('heldout_block_id') is None,
        'All three full-law donor blocks required; leaveout law cannot impersonate prospective root')
    rows=[]
    for block in blocks:
        for key in keys:
            donors=[v for v in law['vectors'] if v['training_block_id']==block and v['public_key']==list(key)]
            rows.append(dict(block_id=block,public_key=list(key),vector_ids=[v['vector_id'] for v in donors],supported=bool(donors)))
    return dict(required_public_keys=[list(k) for k in keys],checks=rows,expected_checks=48,
        complete_all_keys_in_every_donor_block=all(r['supported'] for r in rows))


def _raws(bundle):
    reviews={checked(checked(ref)['inputs']['plan.json'])['attempt_id']:checked(ref) for ref in bundle['review_receipts']}
    require(set(reviews)=={p['attempt_id'] for p in bundle['plans']},'Every matching reviewed raw is required')
    return [checked(reviews[p['attempt_id']]['raw']) for p in bundle['plans']]


def bind_owned_producers(draws,seed):
    """Map symbolic donor producers to actual target lifetime coordinates.

    The original compiler intentionally emits modeled lifetime names. Reusing
    them with an actual owned prefix would discard its station/sequence binding.
    Only these producer labels change; work, phase marks, transport, slacks and
    probabilities remain the compiler's original values.
    """
    output=[]
    for draw in draws:
        if type(draw['inputs']) is prospective.ensemble.UnknownForecastInputs:
            output.append(draw);continue
        inputs=draw['inputs'];plans={};renamed={}
        for key,versions in inputs['plans'].items():
            producer=seed['owner_producers'][key[0]];values=[]
            for plan in versions:
                updated=replace(plan,ready_producer_id=producer,
                    macros=tuple(replace(macro,marks=tuple(replace(mark,producer_id=producer)
                        for mark in macro.marks)) for macro in plan.macros))
                renamed[plan]=updated;values.append(updated)
            plans[key]=tuple(values)
        require(set(renamed)==set(inputs['protocol_slacks']),'Every joint plan protocol vector must survive lifetime binding')
        output.append(dict(draw,inputs=dict(inputs,plans=plans,
            protocol_slacks={renamed[plan]:value for plan,value in inputs['protocol_slacks'].items()})))
    return output


def propagate(requests,draws,seed,contract,gate):
    """Original full-calendar ensemble, unchanged mathematical/scoring entry.

    prospective.public_calendars rejects retained historical IDs by design.
    Fresh181 instead scopes them with execution_id and calls the very same
    compiler/evaluator directly; it never renames source documents as old ones.
    """
    require(len(requests)==181 and len(draws)==12,'The original181 cohort and all three-by-four worlds are required')
    value=prospective.ensemble.evaluate(requests,draws,scoring_plan=contract,quality={},
        window_start_ns=gate['forecast_origin_ns'],window_end_ns=gate['measurement_end_ns'],
        drain_end_ns=gate['fixed_drain_deadline_ns'],settled_initial_state=seed)
    value=decode(value);summaries=[];projection=[]
    for row in value['draws']:
        forecast=row['forecast'];summary={k:copy.deepcopy(v) for k,v in row.items() if k!='forecast'}
        if forecast is None:
            summary.update(forecast_sha256=None,world_sha256=None,query_fingerprint_sha256=None)
            results={}
        else:
            world=forecast['world'];results=world['RESULT_receipts']
            require(world['offered_requests']==181 and set(results)|set(world['unfinished_request_ids'])==
                {r['request_id'] for r in requests},'A generated world lost offered requests')
            summary.update(forecast_sha256=fresh.digest(encode(forecast)),world_sha256=fresh.digest(encode(world)),
                query_fingerprint_sha256=forecast['query_fingerprint_sha256'],
                queue_diagnostics=copy.deepcopy(forecast['queue_diagnostics']),
                scoring=copy.deepcopy(forecast['scoring']),
                RESULT_receipts=copy.deepcopy(results),unfinished_request_ids=world['unfinished_request_ids'])
        summaries.append(summary)
        projection.append(dict(draw_id=row['draw_id'],block_id=row['block_id'],weight=row['weight'],status=row['status'],
            requests=[dict(request_id=r['request_id'],completion_time_ns=math.ceil(results[r['request_id']])
                if r['request_id'] in results else None) for r in requests]))
    value['draws']=summaries
    return encode(dict(ensemble=value,complete_draw_projection=projection,
        projection_method='ceil_exact_rational_ns_same_as_original_six_term_scorer',
        full_world_and_query_hashes_retained=True,full_generated_traces_omitted_after_deterministic_hashing=True,
        Q_actual_quality_UNKNOWN=True,M_E_resource_horizon_UNKNOWN=True,full_U_UNKNOWN=True))


def replay_bundle(bundle,fitted,plan,prefix_document,gate):
    """Pure numerical continuation; authentication is mandatory in produce/audit."""
    response._validate_bundle(bundle)
    require(fitted.get('denominator')==bundle['denominator'] and fitted.get('features')==list(response.FEATURES),
        'Saved matching root and source row bundle differ')
    law=None
    if fitted.get('response_law_compilation_eligible') is not True:
        reason='UNKNOWN_UNQUALIFIED_SAVED_MATCHING_PHASE12_ROOT_OR_FOLDS'
    else:
        law=joint.build_joint_law(bundle['documents'],bundle['plans'],_raws(bundle),fitted,heldout_block_id=None)
        reason=None
    return replay_law(bundle['plans'],law,reason,plan,prefix_document,gate)


def replay_law(plans,law,reason,plan,prefix_document,gate):
    """Same math for a freshly authenticated or guarded cached complete law."""
    contract=checked(verify(plan['public_contract']));axes=physical_axes(plans,plan,prefix_document)
    blocks=sorted({p['capture_id'] for p in plans});coverage=None;selection=[]
    require(len(blocks)==3,'Three whole matching blocks required')
    if law is not None:
        require(law['blocks']==blocks,'Cached law and exact15 plans differ in donor blocks')
        coverage=support(law,contract)
        if not coverage['complete_all_keys_in_every_donor_block']:reason='UNKNOWN_MISSING_EXACT_TEMPLATE_IN_DONOR_BLOCK'
    else:require(type(reason) is str and reason,'A missing law must retain its explicit UNKNOWN reason')
    if reason is not None:draws=generated.unknown_draws(blocks,reason)
    else:draws,selection=generated.compile_case_draws(gate['public_requests'],law)
    draws=bind_owned_producers(draws,prefix_document['seed'])
    result=propagate(gate['public_requests'],draws,prefix_document['seed'],contract,gate)
    decoded=decode(result);unknown=decoded['ensemble']['world_unknown_probability_mass']
    generated_count=sum(r['status']=='GENERATED_WORLD' for r in decoded['ensemble']['draws'])
    return dict(schema=SCHEMA+'.replayed_hypothesis',physical_axes=axes,template_support=coverage,
        law_sha256=None if law is None else fresh.digest(law),law_blocks=blocks,
        measured_request_vectors=0 if law is None else law['measured_request_vectors'],
        joint_law_compilation_reused_saved_fit=True,numerical_fit_run=False,
        selections=encode(selection),selections_sha256=fresh.digest(encode(selection)),forecast=result,
        forecast_sha256=fresh.digest(result),generated_worlds=generated_count,
        world_unknown_probability_mass=encode(unknown),all12_draws_and181_requests_retained=True,
        source_supported_hypothesis_available=reason is None and generated_count>0,
        unknown_reason=reason,assumptions=copy.deepcopy(ASSUMPTIONS),
        independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False,
        target_future_labels_read=False,GPU_started=False)


def _replay(fit_ref,gate_ref,*,allow_cpu_fixture=False,session=None):
    if session is None:
        fit_audit=response.audit_fit(verify(fit_ref));fit=fit_audit['report']
    else:
        require(type(session) is AuthenticatedLawSession and session.fit_ref==fit_ref,'Active typed law session required')
        payload=session.checked_payload();fit=copy.deepcopy(session.fit_report)
    gate=prefix.audit_gate(verify(gate_ref),allow_cpu_fixture=allow_cpu_fixture)
    plan=fresh.audit_plan(gate['plan']['path']);prefix_document=prefix.audit_prefix(gate['prefix'],gate['plan'],plan,
        gate['manifest'],allow_cpu_fixture=allow_cpu_fixture)
    require(fit_ref in gate['model_sources'] and receipt(__file__) in gate['forecast_generator_sources'],
        'Exact saved matching fit and typed producer must be predeclared before prefix collection')
    fit_manifest=checked(verify(fit['manifest']))
    require(fit_manifest['executor_manifest']==plan['matching_producer'],
        'The saved fit must belong to the exact matching campaign bound by the fresh plan')
    def continue_bundle(bundle,intake_document):
        require(intake_document==fit['intake_evidence'],'Saved fit intake does not rederive from exact15 native continuation')
        return replay_bundle(bundle,fit['fit'],plan,prefix_document,gate)
    if session is None:
        value=response.with_verified_bundle(fit['manifest']['path'],fit['continuation']['path'],continue_bundle)
    else:
        value=replay_law(payload['plans'],payload['law'],payload['unknown_reason'],plan,prefix_document,gate)
    sources=strict_references([fit_ref,fit['sources'],gate_ref,gate['plan'],gate['manifest'],gate['prefix'],
        plan['sources'],gate['model_sources'],gate['forecast_generator_sources'],implementation_sources(__file__)])
    for ref in sources:
        if session is None or ref not in session.watched:verify(ref)
    if session is not None:session.checked_payload()
    native=prefix_document['backend']=='native'
    return dict(schema=SCHEMA,status='SOURCE_AUTHENTICATED_PREDECLARED_HYPOTHESIS_NOT_ACCURACY_QUALIFICATION',
        saved_matching_fit=fit_ref,gate=gate_ref,plan=gate['plan'],prefix=gate['prefix'],
        execution_id=gate['execution_id'],candidate_id=gate['candidate_id'],
        public_calendar_sha256=fresh.digest(gate['public_requests']),
        prefix_snapshot_sha256=fresh.digest(prefix_document['prefix_snapshot']),
        forecast_origin_ns=gate['forecast_origin_ns'],measurement_end_ns=gate['measurement_end_ns'],
        fixed_drain_deadline_ns=gate['fixed_drain_deadline_ns'],replay=value,sources=sources,
        CPU_TEST_ONLY=not native,source_supported_native_hypothesis=native and value['source_supported_hypothesis_available'],
        actual_owned_prefix_source_binding='SOURCE_REVIEWED_COLLECTOR_RUNTIME_SNAPSHOT_NOT_POSTHOC_TARGET_LABELS',
        native_protocol_stdout_independently_reparsed=False,
        no_future_target_labels_read=True,model_fit=False,GPU_started=False,
        independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False,
        original_stability_evaluated=False,final_gain_eligible=False)


def produce(fit_report_path,gate_path,output,*,allow_cpu_fixture=False):
    fit_ref=receipt(fit_report_path);gate_ref=receipt(gate_path)
    watched=_watched(fit_ref,gate_ref)
    with fresh.SourceChangeGuard(watched) as guard:
        guard.check();value=_replay(fit_ref,gate_ref,allow_cpu_fixture=allow_cpu_fixture);guard.check()
        result=prefix.save(output,value);guard.check();return result


def _watched(fit_ref,gate_ref):
    fit=checked(verify(fit_ref));gate=checked(verify(gate_ref));plan=checked(verify(gate['plan']))
    manifest=checked(verify(gate['manifest']))
    return strict_references([fit_ref,fit,gate_ref,gate,plan,manifest,implementation_sources(__file__)])


_SESSION_FACTORY=object()


class AuthenticatedLawSession:
    """In-process pre-authentication cache, valid only while its watch is live.

    This is not a serializable caller qualification. Only open_law_session can
    construct it after exact15 source authentication and complete law building.
    Its full content hash and active source-change watcher are checked each use.
    """
    def __init__(self,factory,payload,watched,guard,timings):
        require(factory is _SESSION_FACTORY,'Use exact15 authenticated law-session context manager')
        self._json=json.dumps(payload,sort_keys=True,allow_nan=False)
        self._hash=fresh.digest(payload);self._watched=copy.deepcopy(watched);self._guard=guard
        self._active=True;self.timings=copy.deepcopy(timings)
    def checked_payload(self):
        require(self._active,'Preauthenticated law session is inactive or expired')
        self._guard.check();value=json.loads(self._json)
        require(fresh.digest(value)==self._hash,'Preauthenticated complete law/cache contents changed')
        return value
    @property
    def fit_ref(self):return self.checked_payload()['fit_ref']
    @property
    def fit_report(self):return self.checked_payload()['fit_report']
    @property
    def watched(self):return copy.deepcopy(self._watched)


@contextmanager
def open_law_session(fit_report_path):
    """Authenticate/compile before worker startup, keep the guard through seal.

    The live producer and live auditor can each pre-authenticate once before
    prefix publication. Live prefix replay still executes every original world;
    it does not reread/review15 native raws or redo the saved fit.
    """
    begin=time.perf_counter();fit_ref=receipt(fit_report_path);snapshot=checked(verify(fit_ref))
    watched=strict_references([fit_ref,snapshot,implementation_sources(__file__)])
    with fresh.SourceChangeGuard(watched) as guard:
        guard.check();at=time.perf_counter();fit_audit=response.audit_fit(fit_ref);fit=fit_audit['report']
        timings=dict(saved_fit_audit_seconds=time.perf_counter()-at)
        def authenticate(bundle,intake_document):
            require(intake_document==fit['intake_evidence'],'Cached law must retain the saved fit exact15 intake')
            response._validate_bundle(bundle);law=None;reason='UNKNOWN_UNQUALIFIED_SAVED_MATCHING_PHASE12_ROOT_OR_FOLDS'
            require(fit['fit'].get('denominator')==bundle['denominator']
                and fit['fit'].get('features')==list(response.FEATURES),'Saved root and cached source bundle differ')
            at=time.perf_counter()
            if fit['fit'].get('response_law_compilation_eligible') is True:
                law=joint.build_joint_law(bundle['documents'],bundle['plans'],_raws(bundle),fit['fit'],heldout_block_id=None)
                reason=None
            timings['joint_law_compile_seconds']=time.perf_counter()-at
            return dict(fit_ref=fit_ref,fit_report=fit,plans=copy.deepcopy(bundle['plans']),law=law,unknown_reason=reason)
        at=time.perf_counter()
        payload=response.with_verified_bundle(fit['manifest']['path'],fit['continuation']['path'],authenticate)
        timings['verified_bundle_plus_law_seconds']=time.perf_counter()-at
        for ref in watched:verify(ref)
        guard.check();timings['preauthentication_total_seconds']=time.perf_counter()-begin
        session=AuthenticatedLawSession(_SESSION_FACTORY,payload,watched,guard,timings)
        try:yield session
        finally:
            session._active=False;guard.check()


def produce_from_session(session,gate_path,output,*,allow_cpu_fixture=False):
    require(type(session) is AuthenticatedLawSession,'Typed active preauthenticated law session required')
    gate_ref=receipt(gate_path);fit_ref=session.fit_ref
    with fresh.SourceChangeGuard(_watched(fit_ref,gate_ref)) as guard:
        guard.check();value=_replay(fit_ref,gate_ref,allow_cpu_fixture=allow_cpu_fixture,session=session)
        session.checked_payload();guard.check();result=prefix.save(output,value);guard.check();return result


def audit_from_session(session,report_path,*,require_native=False):
    require(type(session) is AuthenticatedLawSession,'Typed active preauthenticated law session required')
    ref=receipt(report_path);document=checked(verify(ref))
    require(document.get('schema')==SCHEMA and document.get('saved_matching_fit')==session.fit_ref,
        'Typed report and preauthenticated saved matching fit must match')
    with fresh.SourceChangeGuard(strict_references([ref,_watched(session.fit_ref,document['gate'])])) as guard:
        guard.check();expected=_replay(session.fit_ref,document['gate'],allow_cpu_fixture=not require_native,session=session)
        require(document==expected,'Cached-source typed audit must replay every actual prefix-bound forecast draw')
        if require_native:require(expected['source_supported_native_hypothesis'] is True,'Native gate HOLD for unknown/CPU hypothesis')
        session.checked_payload();verify(ref);guard.check();return expected


def audit_report(report_path,*,require_native=False):
    """Reauthenticate exact15 and replay every draw; caller flags cannot release."""
    ref=receipt(report_path);document=checked(verify(ref))
    require(document.get('schema')==SCHEMA,'Typed matching prospective report required')
    watched=strict_references([ref,_watched(document['saved_matching_fit'],document['gate'])])
    with fresh.SourceChangeGuard(watched) as guard:
        guard.check();expected=_replay(document['saved_matching_fit'],document['gate'],allow_cpu_fixture=not require_native)
        require(document==expected,'Typed prospective report must rederive complete source/fit/prefix/calendar/all-world replay')
        if require_native:
            require(expected['source_supported_native_hypothesis'] is True,
                'Native gate HOLD: missing qualified matching root, exact per-block support or generated worlds')
        verify(ref);guard.check();return expected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('produce')
    for key in ('fit-report','gate','output'):cmd.add_argument('--'+key,required=True)
    cmd=sub.add_parser('audit');cmd.add_argument('--report',required=True);cmd.add_argument('--require-native',action='store_true')
    args=parser.parse_args()
    print(produce(args.fit_report,args.gate,args.output) if args.op=='produce' else audit_report(args.report,require_native=args.require_native))
