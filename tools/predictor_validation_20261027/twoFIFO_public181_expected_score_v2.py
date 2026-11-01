"""Prospective six-term conditional mean under explicit uncalibrated Hcpu.

Structural support and mathematical completeness are mandatory. Empirical
accuracy is a distinct, deliberately false qualification: uncalibrated does
not mean that a supported, predeclared hypothesis cannot make predictions.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from fractions import Fraction as F
import json
import itertools
import math
from pathlib import Path
import time

from . import twoFIFO_public181_resource_execute_v1 as resource_executor
from . import twoFIFO_public181_resource_intake_v1 as resource_intake
from . import twoFIFO_public181_resource_campaign_v1 as resource_campaign
from . import twoFIFO_matching_prospective_bridge_v4 as typed
from . import twoFIFO_expected_marginal_score_v1 as marginal
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_fresh181_native_v4 as target
from .macro_six_term_bridge_v1 import whole_score
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked
from .twoFIFO_training_manifest_v1 import require,digest
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.twoFIFO_public181_authenticated_expected_score.v2'
_FACTORY=object()
HCPU=dict(id='Hcpu_passive_training_to_live_prediction_v1',
    statement='Conditional on identical workers, initial actual18 READY history, public calendar, snapshot+300s, 10s settle, measurement90s/drain120s and board sampling policy, the additional CPU producer/auditor/IO activity in the startup wait does not materially change future Q, startup energy, premeasurement owned peak or future owned peak/measurement energy.',
    training_activity='PASSIVE_STARTUP_WAIT_WITH_MONITOR_AND_CHECKPOINTS',
    validation_activity='FULL_TWO_PASS_SIX_TERM_CPU_REPLAY_IO_AND_SEAL_IN_SAME_STARTUP_WAIT',
    empirical_transfer_qualified=False,accuracy_qualified=False,
    startup_energy_omitted_or_subtracted=False,assumed_full_protocol_equality=False)
# Frozen before any validation output. These are engineering acceptance bounds,
# not learned confidence intervals or claims of population accuracy.
EVALUATION_PROTOCOL=dict(schema=SCHEMA+'.prospective_evaluation.v1',
    hypothesis=HCPU,terms=['Q','T','L','P','M','E'],engineering_tolerance_terms=['Q','M','E'],descriptive_error_terms=['T','L','P'],
    term_error_rule='abs(predicted_mean-observed)<=max(absolute_tolerance,relative_tolerance*abs(observed))',
    term_tolerances=dict(Q=dict(absolute_tolerance=.02,relative_tolerance=.10),
        M=dict(absolute_tolerance=.05,relative_tolerance=.10),
        E=dict(absolute_tolerance=.05,relative_tolerance=.10)),
    primitive_tolerances=dict(startup_joules=dict(absolute_tolerance=30.,relative_tolerance=.10),
        premeasurement_owned_sampled_peak_mib=dict(absolute_tolerance=64.,relative_tolerance=.10),
        measurement_through_drain_owned_sampled_peak_mib=dict(absolute_tolerance=64.,relative_tolerance=.10),
        measurement_board_joules=dict(absolute_tolerance=90.,relative_tolerance=.10)),
    evaluation_population='ALL_FOUR_PREDECLARED_CANDIDATES_NO_SURVIVOR_SELECTION',
    score_terms=['Q','T','L','P','M','E'],ranking='ALL_SIX_CANDIDATE_PAIRS',
    delta_indifference_absolute=.01,
    delta_rule='REPORT_ALL_SIX_PAIRS_WITH_EXACT_POSITIVE_ZERO_NEGATIVE_SIGNS_AND_SIGN_AGREEMENT_EVEN_NEAR_ZERO; INDIFFERENCE_IS_A_SEPARATE_ENGINEERING_CLASSIFICATION_ONLY',
    pairwise_missing_candidate_rule='RETAIN_ALL_SIX_PAIRS_WITH_UNKNOWN_LABEL_NO_DENOMINATOR_CHANGE',
    missing_label_rule='UNKNOWN_RETAIN_ALL_PLANNED_ROWS_NEVER_ZERO_OR_RENORMALIZE',
    cpu_activity_diagnostics=['prediction_spans','board_samples','premeasurement_owned_peak','startup_joules','sampling_gaps'],
    validation_future_labels_used_to_choose_tolerances=False,model_updated_after_validation=False,
    one_four_candidate_block_establishes_general_accuracy=False,original_stability_rule_unchanged=True,
    tolerance_basis='PREDECLARED_ENGINEERING_BOUNDS_NOT_PAPER_GUARANTEES_OR_STATISTICAL_CONFIDENCE',
    automatic_final_gain_qualification=False)
OBSERVATION_AXES=dict(residents=4,measurement_ns=90_000_000_000,drain_ns=120_000_000_000,
    startup_occurrences=1,precohort_lead_ns=300_000_000_000,
    startup_amortization_seconds=3600,monitor='live_owned_launcher_descendants_NVML_board',
    nominal_poll_interval_ns=20_000_000,measurement_includes_drain=False,
    memory='owned_resident_sampled_peak_startup_through_fixed_drain')


def verify(ref):
    require(receipt(ref['path'])==ref,'Expected-score source receipt changed');return ref


def shared_dispatch_protocol():
    return dict(module='tools.predictor_validation_20261027.twoFIFO_public181_resource_execute_v1',
        source=receipt(resource_executor.__file__),function='dispatch_case',epoch_anchor='snapshot_time_ns_plus300s',
        READY_settle_timeout_ns=resource_executor.SETTLE_NS,
        measurement_ns=resource_executor.MEASUREMENT_NS,drain_ns=resource_executor.DRAIN_NS)


def save(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush()
        import os
        os.fsync(stream.fileno())
    return receipt(path)


class ResourceSession:
    def __init__(self,factory,payload,guard):
        require(factory is _FACTORY,'Use exact native resource re-review session')
        self._json=json.dumps(payload,sort_keys=True,allow_nan=False);self._digest=digest(payload)
        self._guard=guard;self._active=True
    def checked_payload(self):
        require(self._active,'Native resource session expired');self._guard.check()
        payload=json.loads(self._json);require(digest(payload)==self._digest,'Native resource cache changed')
        return payload


@contextmanager
def open_resource_session(manifest_path,review_paths):
    """Re-review every supplied native source before startup; retain missing8."""
    manifest_ref=receipt(manifest_path);snapshot=checked(verify(manifest_ref))
    refs=[receipt(p) for p in review_paths]
    documents=[checked(verify(ref)) for ref in refs]
    watched=strict_references([manifest_ref,snapshot,refs,documents,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        guard.check();manifest=resource_executor.audit(manifest_path)
        intake=resource_intake.load_complete(manifest_path,review_paths)
        require(manifest==snapshot and len(intake['rows'])==8,'All8 native resource plans must survive authentication')
        guard.check();session=ResourceSession(_FACTORY,dict(manifest=manifest_ref,manifest_document=manifest,
            review_refs=refs,intake=intake,source_bindings=watched),guard)
        try:yield session
        finally:session._active=False;guard.check()


class ExpectedSession:
    def __init__(self,factory,configuration_ref,configuration,queue_session,resource_session,guard):
        require(factory is _FACTORY,'Only source-bound expected preparation can open this session')
        self.configuration_ref=configuration_ref;self.configuration=copy.deepcopy(configuration)
        self._digest=digest(configuration);self.queue=queue_session;self.resources=resource_session
        self._guard=guard;self._active=True
    def check(self):
        require(self._active and digest(self.configuration)==self._digest,'Expected-score session expired or mutated')
        self._guard.check();self.queue.checked_payload();self.resources.checked_payload()


def prepare(target_manifest_path,target_plan_path,resource_manifest_path,resource_review_paths,output):
    """New CPU configuration only; does not enable an old native collector."""
    manifest=target.audit(target_manifest_path);plan=target.audit_plan(target_plan_path)
    target_ref=receipt(target_manifest_path);plan_ref=receipt(target_plan_path)
    require(plan_ref in [r['plan'] for r in manifest['attempts']],'Exact native-v4 target plan required')
    resources=resource_executor.audit(resource_manifest_path)
    refs=[receipt(path) for path in resource_review_paths]
    require(plan['expected_score_resources']==dict(manifest=receipt(resource_manifest_path),reviews=refs)
        and plan['evaluation_protocol']==EVALUATION_PROTOCOL,'Resources and prospective protocol must be bound in native4 plan')
    require(len(refs)<=8 and len({r['path'] for r in refs})==len(refs),'Distinct planned resource source reviews required')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    closure=implementation_sources(__file__);frozen=[]
    for source in closure:
        relative=Path(source['path']).relative_to(resource_executor.ROOT)
        path=out/'source-freeze'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(source['path']).read_bytes());frozen.append(dict(original_path=source['path'],frozen=receipt(path)))
    require(all(a['sha256']==b['frozen']['sha256'] for a,b in zip(closure,frozen)),'Expected source changed during freeze')
    value=_declaration(manifest,plan,resources,target_ref,plan_ref,receipt(resource_manifest_path),refs,closure,frozen)
    ref=save(out/'preparation.json',value);audit_preparation(ref['path']);return ref


def _declaration(manifest,plan,resources,target_ref,plan_ref,resource_ref,refs,closure,frozen):
    return dict(schema=SCHEMA+'.preparation',target_manifest=target_ref,target_plan=plan_ref,
        fit_report=plan['saved_matching_fit'],resource_manifest=resource_ref,resource_reviews=refs,
        candidate_id=plan['candidate_id'],expected_queue_worlds=12,expected_resource_attempts=8,
        original_score_contract=plan['public_contract'],original_observation_axes=OBSERVATION_AXES,
        sources=strict_references([target_ref,plan_ref,manifest['sources'],resources['sources'],refs,closure,frozen]),
        implementation_sources=closure,source_freeze=frozen,
        native_collector_integration='EXPLICIT_NATIVE_V4_SIX_TERM_CONDITIONAL_HYPOTHESIS',
        hypothesis=HCPU,evaluation_protocol=EVALUATION_PROTOCOL,
        total_live_pipeline_budget_qualified=False,full_U_seal_authorizes_target_enqueue=False,
        GPU_started=False,validation_labels_read=False,final_gain_eligible=False)


def audit_preparation(path):
    ref=receipt(path);document=checked(verify(ref))
    require(document.get('schema')==SCHEMA+'.preparation','Exact expected-score preparation required')
    require(document['implementation_sources']==implementation_sources(__file__),'Expected implementation closure changed')
    require(all(receipt(r['frozen']['path'])==r['frozen'] and receipt(r['original_path'])['sha256']==r['frozen']['sha256']
        for r in document['source_freeze']),'Expected source freeze changed')
    for source in document['sources']:verify(source)
    manifest=target.audit(document['target_manifest']['path']);plan=target.audit_plan(document['target_plan']['path'])
    resources=resource_executor.audit(document['resource_manifest']['path'])
    require(plan['expected_score_resources']==dict(manifest=document['resource_manifest'],reviews=document['resource_reviews'])
        and plan['evaluation_protocol']==EVALUATION_PROTOCOL,
        'Expected preparation resource binding differs from the exact native4 plan')
    require(type(document['resource_reviews']) is list and len(document['resource_reviews'])<=8
        and len({r['path'] for r in document['resource_reviews']})==len(document['resource_reviews']),
        'Distinct source-bound resource review declarations required')
    require(document['target_plan'] in [r['plan'] for r in manifest['attempts']]
        and document['fit_report']==plan['saved_matching_fit'] and document['candidate_id']==plan['candidate_id']
        and document['original_score_contract']==plan['public_contract']
        and document['expected_queue_worlds']==12 and document['expected_resource_attempts']==8
        and len(resources['attempts'])==8 and document['original_observation_axes']==OBSERVATION_AXES
        and document['total_live_pipeline_budget_qualified'] is False
        and document['full_U_seal_authorizes_target_enqueue'] is False
        and document['native_collector_integration']=='EXPLICIT_NATIVE_V4_SIX_TERM_CONDITIONAL_HYPOTHESIS',
        'Expected-score preparation domain or execution limitation changed')
    closure=implementation_sources(__file__)
    frozen=[dict(original_path=source['path'],frozen=receipt(Path(path).resolve().parent/'source-freeze'/
        Path(source['path']).relative_to(resource_executor.ROOT))) for source in closure]
    require(all(source['sha256']==item['frozen']['sha256'] for source,item in zip(closure,frozen)),
            'Complete expected source closure freeze required')
    expected=_declaration(manifest,plan,resources,document['target_manifest'],document['target_plan'],
        document['resource_manifest'],document['resource_reviews'],closure,frozen)
    require(document==expected,'Complete expected preparation declaration must reconstruct')
    verify(ref);return document


@contextmanager
def open_session(preparation_path):
    ref=receipt(preparation_path);snapshot=checked(verify(ref))
    watched=strict_references([ref,snapshot,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        configuration=audit_preparation(preparation_path);require(configuration==snapshot,'Expected preparation changed')
        with typed.open_target_session(configuration['fit_report']['path'],configuration['target_manifest']['path'],
                configuration['target_plan']['path']) as queue_session:
            with open_resource_session(configuration['resource_manifest']['path'],[r['path'] for r in configuration['resource_reviews']]) as resources:
                session=ExpectedSession(_FACTORY,ref,configuration,queue_session,resources,guard)
                try:yield session
                finally:session._active=False;guard.check()


@contextmanager
def open_target_session(fit_report_path,manifest_path,plan_path):
    """Exact native4 entrypoint: all training authentication precedes workers."""
    manifest_ref=receipt(manifest_path);plan_ref=receipt(plan_path)
    snapshot=checked(verify(manifest_ref));plan=checked(verify(plan_ref))
    watched=strict_references([manifest_ref,snapshot,plan_ref,plan,implementation_sources(__file__)])
    with SourceChangeGuard(watched) as guard:
        with typed.open_target_session(fit_report_path,manifest_path,plan_path) as queue_session:
            cached=queue_session.checked_payload();require(cached['plan']==plan,'Preauthenticated target changed')
            binding=plan['expected_score_resources']
            require(plan['evaluation_protocol']==EVALUATION_PROTOCOL
                and plan['shared_dispatch_protocol']==shared_dispatch_protocol(),
                'Native4 must bind frozen prospective hypothesis and shared dispatcher')
            with open_resource_session(binding['manifest']['path'],[r['path'] for r in binding['reviews']]) as resources:
                configuration=dict(target_manifest=manifest_ref,target_plan=plan_ref,fit_report=receipt(fit_report_path),
                    resource_manifest=binding['manifest'],resource_reviews=binding['reviews'],
                    hypothesis=HCPU,evaluation_protocol=EVALUATION_PROTOCOL)
                session=ExpectedSession(_FACTORY,plan_ref,configuration,queue_session,resources,guard)
                try:yield session
                finally:session._active=False;guard.check()


def transfer_axes(resource_payload,plan,prefix_document,gate):
    """Actual per-axis evidence; closure hashes are provenance, not physics.

    Hard structural differences block a numerical transfer. Hcpu is separately
    declared uncalibrated: it permits testable conditional means, never accuracy.
    """
    manifest=resource_payload['manifest_document'];rows=[];all_issues=[]
    context=checked(verify(plan['context']));target_registry=resource_campaign.planned_worker_registry(context)
    expected_priming=typed.core.matching.planned_priming_semantics(plan,context)
    actual_gap=gate['forecast_origin_ns']-prefix_document['prefix_snapshot']['snapshot_time_ns']
    require(prefix_document['actual_priming_semantics_sha256']==expected_priming['sha256'],
            'Actual target priming differs from planned content')
    for item in manifest['attempts']:
        donor=checked(verify(item['plan']));donor_context=checked(verify(donor['context']));issues=[]
        equalities=dict(worker_context=donor_context==context,
            worker_registry=resource_campaign.planned_worker_registry(donor_context)==target_registry,
            runtime_config=donor['runtime_config']==plan['runtime_config'],
            runtime_policy=donor['runtime_policy']==plan['runtime_policy'],
            resident_owners=donor['resident_owners']==plan['resident_owners'],
            priming_semantics=donor['priming_semantics']==expected_priming,
            prepared_pool=donor['prepared_pool']==plan['prepared_pool'],
            original_score_contract=donor['public_case']['public_contract']==plan['public_contract'],
            objective=donor['public_case']['objective']==plan['objective'],
            measurement=donor['measurement_ns']==plan['measurement_ns']==OBSERVATION_AXES['measurement_ns'],
            drain=donor['drain_ns']==plan['drain_ns']==OBSERVATION_AXES['drain_ns'],
            declared_lead=donor['precohort_lead_seconds']==plan['forecast_barrier']['lead_seconds']==300,
            READY_settle_budget=plan['forecast_barrier']['prior_READY_settle_timeout_seconds']*10**9==resource_executor.SETTLE_NS,
            actual_snapshot_lead=actual_gap==resource_executor.LEAD_NS)
        equalities['shared_dispatch_protocol']=plan.get('shared_dispatch_protocol')==shared_dispatch_protocol()
        equalities['prospective_evaluation_protocol']=plan.get('evaluation_protocol')==EVALUATION_PROTOCOL
        # Candidate calendar equality is only required for its two donor rows.
        if donor['candidate_id']==plan['candidate_id']:
            equalities['exact_candidate_and_ordered_calendar']=(donor['public_case']['candidate']==plan['candidate']
                and donor['cases']==plan['cases'])
        issues.extend('TRANSFER_AXIS_DIFFERS:'+name for name,value in equalities.items() if not value)
        rows.append(dict(attempt_id=donor['attempt_id'],candidate_id=donor['candidate_id'],equalities=equalities,
            issues=issues,training_protocol_identity=donor['collection_protocol_identity'],
            target_execution_context_sha256=plan['execution_context_sha256']))
        if donor['candidate_id']==plan['candidate_id']:all_issues.extend(issues)
    return dict(schema=SCHEMA+'.explicit_resource_transfer',all8_axes=rows,
        actual_snapshot_to_origin_ns=actual_gap,declared_target_lead_ns=plan['forecast_barrier']['lead_seconds']*10**9,
        snapshot_copy_anchor_excess_ns=actual_gap-resource_executor.LEAD_NS,
        training_anchor='snapshot_time_ns_plus300s',target_anchor='snapshot_time_ns_plus300s',
        collector_activity_difference_hypothesis=HCPU,
        hypotheses_empirically_validated=False,protocol_hash_equality_used_as_physical_proof=False,
        closure_hash_difference_ignored=False,structural_transfer_supported=not all_issues,
        conditional_resource_hypothesis_available=not all_issues,
        resource_marginal_transfer_qualified=False,empirical_transfer_qualified=False,
        unknown_reasons=sorted(set(all_issues)))


def resource_marginals(intake,plan,gate,transfer,domain,sources):
    """Two whole-block Q/M/E marginals, no resource/queue pairing."""
    rows=intake['rows'];require(len(rows)==8,'All8 planned resource denominators required')
    blocks=sorted({r['capture_id'] for r in rows});require(len(blocks)==2,'Two original whole resource blocks required')
    contract=checked(verify(plan['public_contract']));requests=gate['public_requests'];a=gate['forecast_origin_ns']
    frame=dict(original_window=encode([F(a),F(gate['measurement_end_ns'])]),
        original_drain_end_ns=encode(F(gate['fixed_drain_deadline_ns'])),cutoff_ns=encode(F(gate['fixed_drain_deadline_ns'])),
        RESULT_receipts={},unfinished_request_ids=[r['request_id'] for r in requests],offered_requests=181)
    worlds=[]
    for block in blocks:
        matches=[row for row in rows if row['capture_id']==block and row['candidate_id']==plan['candidate_id']]
        require(len(matches)==1,'Each block must retain the exact candidate even if UNKNOWN')
        donor=matches[0];values={term:None for term in 'QME'}
        if transfer['conditional_resource_hypothesis_available'] and donor['joint_tuple'] is not None:
            draw=dict(block_id=block,candidates={plan['candidate_id']:dict(donor_attempt_id=donor['attempt_id'],
                joint_tuple=donor['joint_tuple'],native_review=donor['native_review'],raw=donor['raw'])})
            resources=resource_campaign.resource_bridge_payload(draw,plan['candidate_id'],
                window_start_ns=a,resident_owners=plan['resident_owners'])
            original=whole_score(frame,requests,contract,{},resources)['score']['terms']
            values.update(M=original['M'],E=original['E'])
        quality=donor['actual_native_quality']
        # Q is also a hypothesis transfer to future native outcomes. Keep the
        # same physical/calendrical gate rather than using a frozen constant.
        if transfer['conditional_resource_hypothesis_available'] and quality is not None:
            require(quality['original_offered_rates']==contract['workload_rps'],'Original quality rate weights differ')
            values['Q']=quality['measured_Q']
        worlds.append(dict(draw_id=block,block_id=block,weight=F(1,2),status='GENERATED_WORLD',forecast=values,
            donor_attempt_id=donor['attempt_id'],source_resource_status=donor['resource_status']))
    return {term:dict(domain=copy.deepcopy(domain),sources=sources,
        distribution=ensemble.distribution(worlds,lambda f,t=term:f[t])) for term in 'QME'},worlds


def primitive_predictions(intake,plan,transfer):
    rows=[r for r in intake['rows'] if r['candidate_id']==plan['candidate_id']]
    require(len(rows)==2,'Two original candidate donors required for primitive diagnostics')
    result={}
    for key in EVALUATION_PROTOCOL['primitive_tolerances']:
        values=[r['joint_tuple'].get(key) if r['joint_tuple'] is not None else None for r in rows]
        known=transfer['conditional_resource_hypothesis_available'] and all(v is not None for v in values)
        result[key]=dict(predicted_mean=math.fsum(values)/2 if known else None,
            original_donor_count=2,donor_values=values,empirical_accuracy_qualified=False)
    return result


def compose(resource_payload,plan,prefix,gate,queue,*,configuration_ref,target_manifest_ref):
    """Shared numerical chain for live reports and a TRAINING_ONLY CPU proxy.

    This function authenticates nothing. Only replay(active ExpectedSession)
    can construct a native report; budget proxies remain explicitly nonnative.
    """
    forecast=decode(queue['replay']['forecast'])['ensemble'];require(len(forecast['draws'])==12,'All12 original queue worlds required')
    payload=resource_payload;transfer=transfer_axes(payload,plan,prefix,gate)
    domain=dict(public_case_sha256=digest(dict(candidate=plan['candidate'],cases=plan['cases'],
        prepared_pool=plan['prepared_pool'],context=plan['context'])),
        target_execution_context_sha256=plan['execution_context_sha256'],
        original_score_contract=plan['public_contract'],observation_axes=OBSERVATION_AXES)
    contract=checked(verify(plan['public_contract']));comparison=dict(
        public_population_sha256=digest(contract['trace']),original_score_contract=plan['public_contract'],
        observation_axes=OBSERVATION_AXES,target_manifest=target_manifest_ref)
    refs=strict_references([configuration_ref,payload['source_bindings'],queue['sources']])
    marginals,resource_worlds=resource_marginals(payload['intake'],plan,gate,transfer,domain,refs)
    marginals.update({term:dict(domain=copy.deepcopy(domain),sources=queue['sources'],
        distribution=forecast['six_terms'][term]) for term in 'TLP'})
    score=marginal.combine(marginals,candidate_id=plan['candidate_id'],candidate_domain=domain,
        comparison_contract=comparison,weights=contract['weights'])
    return dict(score=score,transfer=transfer,resource_worlds=resource_worlds,sources=refs,
        primitive_predictions=primitive_predictions(payload['intake'],plan,transfer))


def replay(session,gate_ref,*,allow_cpu_fixture=False):
    require(type(session) is ExpectedSession,'Exact active expected-score session required');session.check()
    cached,plan,prefix,gate=typed._prefix_and_gate(session.queue,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    require(cached['plan_ref']==session.configuration['target_plan'],'Expected preparation and live target differ')
    queue=typed.replay(session.queue,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    composed=compose(session.resources.checked_payload(),plan,prefix,gate,queue,
        configuration_ref=session.configuration_ref,target_manifest_ref=session.configuration['target_manifest'])
    score=composed['score'];complete=score['all_six_marginal_expectations_complete']
    session.check()
    return encode(dict(schema=SCHEMA,status='SOURCE_AUTHENTICATED_SEPARATE_MARGINAL_HYPOTHESIS_NOT_LAUNCH_OR_ACCURACY_QUALIFICATION',
        preparation=session.configuration_ref,gate=gate_ref,prefix=gate['prefix'],target_plan=cached['plan_ref'],
        candidate_id=plan['candidate_id'],source_queue_forecast=queue,resource_transfer=composed['transfer'],
        resource_marginal_worlds=composed['resource_worlds'],expected_score=score,all12_queue_worlds_retained=True,
        resource_primitive_predictions=composed['primitive_predictions'],
        all8_resource_plans_retained=True,queue_resource_draws_paired=False,sources=composed['sources'],
        source_evidence_authenticated=prefix['backend']=='native',
        conditional_mean_available=complete,conditional_mean_hypothesis=HCPU,evaluation_protocol=EVALUATION_PROTOCOL,
        empirical_transfer_qualified=False,independent_accuracy_qualified=False,full_U_accuracy_qualified=False,
        source_authentication_does_not_qualify_transfer_or_accuracy=True,
        total_live_pipeline_budget_qualified=False,existing_queue_seal_reused_as_full_U_seal=False,
        native_collector_integrated=True,full_U_qualified=False,positive_delta_probability=None,
        original_stability_evaluated=False,validation_labels_read=False,GPU_started=False,final_gain_eligible=False))


def produce(session,gate_path,output,*,allow_cpu_fixture=False):
    gate_ref=receipt(gate_path);gate=checked(verify(gate_ref))
    with SourceChangeGuard([gate_ref,gate['prefix']]) as guard:
        guard.check();value=replay(session,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
        result=save(output,value);guard.check();session.check();return result


def audit_report(session,report_path,*,require_native=False):
    require(type(session) is ExpectedSession,'Independent active expected-score auditor required')
    ref=receipt(report_path);document=checked(verify(ref))
    require(document.get('schema')==SCHEMA,'Exact expected-score report required')
    with SourceChangeGuard([ref,document['gate'],document['prefix']]) as guard:
        expected=replay(session,document['gate'],allow_cpu_fixture=not require_native)
        require(expected==document,'Complete queue replay and all six marginal scores must rederive')
        if require_native:require(expected['source_evidence_authenticated'] is True,'Native actual prefix required')
        guard.check();session.check();return expected


def _seal_document(cached,plan,gate,report_ref,report,*,begin,cpu_only):
    return dict(schema=SCHEMA+'.prediction_seal',expected_report=report_ref,gate=report['gate'],
        plan=cached['plan_ref'],prefix=gate['prefix'],execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
        forecast_origin_ns=gate['forecast_origin_ns'],seal_begin_ns=begin,
        complete_six_term_report_sha256=report_ref['sha256'],
        conditional_mean_available=report['conditional_mean_available'],
        source_evidence_authenticated=report['source_evidence_authenticated'],
        hypothesis=HCPU,evaluation_protocol=EVALUATION_PROTOCOL,CPU_TEST_ONLY=cpu_only,
        validation_labels_read=False,empirical_transfer_qualified=False,independent_accuracy_qualified=False,
        full_U_accuracy_qualified=False,full_U_qualified=False,original_stability_evaluated=False,final_gain_eligible=False)


def seal(session,gate_path,report_path,output,*,clock_ns=time.perf_counter_ns,report_reference=None):
    """Independent full six-term replay, durable seal, strictly before epoch.

    Call this with a separately preauthenticated auditor session. It performs
    the second replay itself; do not add a third audit replay outside the budget.
    """
    gate_ref=receipt(gate_path);gate_document=checked(verify(gate_ref))
    report_ref=receipt(report_path) if report_reference is None else verify(report_reference)
    require(report_ref['path']==str(Path(report_path).resolve()),'Exact producer receipt required')
    with SourceChangeGuard([gate_ref,gate_document['prefix'],report_ref]) as guard:
        cached,plan,prefix,gate=typed._prefix_and_gate(session.queue,gate_ref,
            allow_cpu_fixture=clock_ns is not time.perf_counter_ns)
        require(str(Path(report_path).resolve())==plan['forecast_barrier']['expected_report_path']
            and str(Path(output).resolve())==plan['forecast_barrier']['seal_path'],
            'Only predeclared six-term report/seal paths accepted')
        report=audit_report(session,report_path,require_native=clock_ns is time.perf_counter_ns)
        require(report['conditional_mean_available'] is True
            and report['resource_transfer']['structural_transfer_supported'] is True,
            'Missing mathematical support or native resource/quality labels: zero target requests')
        begin=clock_ns();require(type(begin) is int and begin<gate['forecast_origin_ns'],
            'Full six-term prediction and independent audit must finish before origin')
        value=_seal_document(cached,plan,gate,report_ref,report,begin=begin,
            cpu_only=prefix['backend']!='native' or clock_ns is not time.perf_counter_ns)
        guard.check();verify(report_ref);result=save(output,value)
        require(clock_ns()<gate['forecast_origin_ns'],'Six-term seal crossed fixed epoch; zero targets')
        guard.check();session.check();return result


def audit_seal_structure(session,gate_ref,seal_ref,*,now_ns,allow_cpu_fixture=False):
    """Cheap source-bound acceptance after the concrete independent seal call."""
    session.check();cached,plan,prefix,gate=typed._prefix_and_gate(session.queue,gate_ref,
        allow_cpu_fixture=allow_cpu_fixture)
    document=checked(verify(seal_ref));report_ref=document['expected_report'];report=checked(verify(report_ref))
    require(str(Path(seal_ref['path']).resolve())==plan['forecast_barrier']['seal_path']
        and str(Path(report_ref['path']).resolve())==plan['forecast_barrier']['expected_report_path'],
        'Canonical full-sixterm seal/report required')
    expected=_seal_document(cached,plan,gate,report_ref,report,begin=document['seal_begin_ns'],cpu_only=document['CPU_TEST_ONLY'])
    require(document==expected and report['schema']==SCHEMA and report['gate']==gate_ref
        and report['prefix']==gate['prefix'] and report['target_plan']==cached['plan_ref']
        and report['evaluation_protocol']==EVALUATION_PROTOCOL and report['conditional_mean_hypothesis']==HCPU
        and document['conditional_mean_available'] is True
        and type(now_ns) is int and type(document['seal_begin_ns']) is int
        and document['seal_begin_ns']<=now_ns<gate['forecast_origin_ns']
        and (allow_cpu_fixture or document['CPU_TEST_ONLY'] is False and document['source_evidence_authenticated'] is True),
        'Exact six-term conditional-mean seal must be accepted before epoch')
    session.check();return document


def evaluate_validation(predictions,observed,*,candidate_ids):
    """Pure error arithmetic over an independently authenticated caller input.

    This helper grants no source or accuracy qualification. All four original
    candidates and six pairs survive missing outcomes. Actual terms must be
    derived by the unchanged native scorer before calling this function.
    """
    require(type(candidate_ids) is list and len(candidate_ids)==4 and len(set(candidate_ids))==4,
        'Frozen four-candidate evaluation population required')
    require(set(predictions)<=set(candidate_ids) and set(observed)<=set(candidate_ids),'Foreign evaluation candidate')
    def finite(value):return type(value) in (int,float,F) and math.isfinite(float(value))
    def error(pred,actual,tolerance=None):
        if pred is None or actual is None:return dict(predicted=pred,observed=actual,absolute_error=None,within_engineering_tolerance=None)
        require(finite(pred) and finite(actual),'Finite prediction/actual required')
        if tolerance is None:return dict(predicted=pred,observed=actual,absolute_error=abs(pred-actual),
            engineering_tolerance=None,within_engineering_tolerance=None)
        bound=max(tolerance['absolute_tolerance'],tolerance['relative_tolerance']*abs(actual));difference=abs(pred-actual)
        return dict(predicted=pred,observed=actual,absolute_error=difference,engineering_tolerance=bound,
            within_engineering_tolerance=difference<=bound)
    rows=[];predicted_U={};actual_U={}
    for cid in candidate_ids:
        document=decode(predictions[cid]) if cid in predictions else None
        actual=decode(observed[cid]) if cid in observed else None
        if document is not None:
            require(document['evaluation_protocol']==EVALUATION_PROTOCOL and document['conditional_mean_hypothesis']==HCPU,
                'Prediction must predeclare this exact evaluation protocol')
            score=marginal._reconstruct(document['expected_score']);require(score['candidate_id']==cid,'Prediction candidate differs')
            predicted_U[cid]=score['expected_U_under_supplied_marginals'];terms=score['expected_terms']
        else:predicted_U[cid]=None;terms={}
        actual_terms=actual.get('terms',{}) if actual is not None else {}
        require(set(actual_terms)<=set('QTLPME'),'Unknown actual score term')
        require(all(v is None or finite(v) and v>=0 for v in actual_terms.values()),
            'Actual original loss terms must be finite and nonnegative when observed')
        complete=all(finite(actual_terms.get(t)) for t in 'QTLPME')
        weights=actual.get('weights') if actual is not None else None
        actual_U[cid]=actual.get('full_U') if actual is not None else None
        crosscheck=None
        if weights is not None:
            require(weights=={term:1/6 for term in 'QTLPME'},'Original scorer weights required')
            if complete:
                # Preserve the exact unchanged scorer operation/order. This is
                # a cross-check, never a replacement observed utility label.
                crosscheck=-sum(weights[term]*value for term,value in actual_terms.items())
        require(actual_U[cid] is None or finite(actual_U[cid]) and complete and actual_U[cid]==crosscheck,
            'Observed full_U must be the unchanged original scorer label with its weights and terms')
        term_errors={term:error(terms.get(term),actual_terms.get(term),EVALUATION_PROTOCOL['term_tolerances'].get(term))
            for term in 'QTLPME'}
        primitive_errors={key:error(document.get('resource_primitive_predictions',{}).get(key,{}).get('predicted_mean') if document else None,
            actual.get('primitives',{}).get(key) if actual else None,tol)
            for key,tol in EVALUATION_PROTOCOL['primitive_tolerances'].items()}
        rows.append(dict(candidate_id=cid,term_errors=term_errors,primitive_errors=primitive_errors,
            predicted_U=predicted_U[cid],observed_U=actual_U[cid],observed_U_original_arithmetic_crosscheck=crosscheck,
            observed_U_uses_original_scorer_label=True))
    pairs=[]
    def sign(value):return None if value is None else (1 if value>0 else -1 if value<0 else 0)
    for left,right in itertools.combinations(candidate_ids,2):
        pred=None if predicted_U[left] is None or predicted_U[right] is None else predicted_U[right]-predicted_U[left]
        actual=None if actual_U[left] is None or actual_U[right] is None else actual_U[right]-actual_U[left]
        pairs.append(dict(baseline=left,candidate=right,predicted_delta_U=pred,observed_delta_U=actual,
            predicted_sign=sign(pred),observed_sign=sign(actual),
            exact_sign_agreement=None if pred is None or actual is None else sign(pred)==sign(actual),
            predicted_engineering_indifference=None if pred is None else abs(pred)<=.01,
            observed_engineering_indifference=None if actual is None else abs(actual)<=.01))
    return encode(dict(schema=SCHEMA+'.pure_error_arithmetic',evaluation_protocol=EVALUATION_PROTOCOL,
        candidates=rows,pairs=pairs,candidate_denominator=4,pair_denominator=6,
        known_pair_signs=sum(p['exact_sign_agreement'] is not None for p in pairs),
        source_evidence_authenticated=False,empirical_transfer_qualified=False,independent_accuracy_qualified=False,
        full_U_accuracy_qualified=False,final_gain_eligible=False,model_updated_after_validation=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    cmd=sub.add_parser('prepare')
    for name in ('target-manifest','target-plan','resource-manifest','output'):cmd.add_argument('--'+name,required=True)
    cmd.add_argument('--resource-review',action='append',default=[])
    cmd=sub.add_parser('audit');cmd.add_argument('--preparation',required=True)
    args=parser.parse_args()
    if args.op=='prepare':value=prepare(args.target_manifest,args.target_plan,args.resource_manifest,args.resource_review,args.output)
    else:value=audit_preparation(args.preparation)
    print(json.dumps(value))
