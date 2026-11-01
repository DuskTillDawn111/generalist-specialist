"""Source-bound original expected utility from separate queue/resource marginals.

No GPU launcher and no replacement for the frozen queue-only native-v3 seal.
Separate active training sessions authenticate source evidence before startup;
all12 queue worlds and all8 planned resource rows remain at their original
mass. Known source provenance never implies transfer or accuracy qualification.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import time

from . import twoFIFO_public181_resource_execute_v1 as resource_executor
from . import twoFIFO_public181_resource_intake_v1 as resource_intake
from . import twoFIFO_public181_resource_campaign_v1 as resource_campaign
from . import twoFIFO_matching_prospective_bridge_v3 as typed
from . import twoFIFO_expected_marginal_score_v1 as marginal
from . import twoFIFO_forecast_ensemble_v1 as ensemble
from . import twoFIFO_fresh181_native_v3 as target
from .macro_six_term_bridge_v1 import whole_score
from .macro_response_queue_reference_v1 import encode
from .actual_worker_twoFIFO_reference_v5_2 import decode
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked
from .twoFIFO_training_manifest_v1 import require,digest
from .twoFIFO_source_rebound_execute_v1 import SourceChangeGuard

SCHEMA='gsdeploy.twoFIFO_public181_authenticated_expected_score.v1'
_FACTORY=object()
OBSERVATION_AXES=dict(residents=4,measurement_ns=90_000_000_000,drain_ns=120_000_000_000,
    startup_occurrences=1,precohort_lead_ns=300_000_000_000,
    startup_amortization_seconds=3600,monitor='live_owned_launcher_descendants_NVML_board',
    nominal_poll_interval_ns=20_000_000,measurement_includes_drain=False,
    memory='owned_resident_sampled_peak_startup_through_fixed_drain')


def verify(ref):
    require(receipt(ref['path'])==ref,'Expected-score source receipt changed');return ref


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
    require(plan_ref in [r['plan'] for r in manifest['attempts']],'Exact native-v3 target plan required')
    resources=resource_executor.audit(resource_manifest_path)
    refs=[receipt(path) for path in resource_review_paths]
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
        native_collector_integration='NOT_IMPLEMENTED_FROZEN_NATIVE_V3_REMAINS_QUEUE_ONLY',
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
        and document['native_collector_integration']=='NOT_IMPLEMENTED_FROZEN_NATIVE_V3_REMAINS_QUEUE_ONLY',
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


def transfer_axes(resource_payload,plan,prefix_document,gate):
    """Actual per-axis evidence; closure hashes are provenance, not physics.

    Current native-v3 differs in READY-settle limit, epoch anchor and live CPU
    work. Those material differences remain explicit UNKNOWN transfer gates.
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
        # Candidate calendar equality is only required for its two donor rows.
        if donor['candidate_id']==plan['candidate_id']:
            equalities['exact_candidate_and_ordered_calendar']=(donor['public_case']['candidate']==plan['candidate']
                and donor['cases']==plan['cases'])
        issues.extend('TRANSFER_AXIS_DIFFERS:'+name for name,value in equalities.items() if not value)
        # v3 source explicitly selects origin after returning the snapshot and
        # performs live prediction/audit CPU activity during the startup wait;
        # resource source has snapshot anchor and passive wait. No measured
        # evidence currently bounds their effect on startup energy or P0.
        issues.extend(['UNQUALIFIED_EPOCH_ANCHOR_TRANSFER_SNAPSHOT_VS_POSTSNAPSHOT_CLOCK',
            'UNVALIDATED_STARTUP_CPU_PREDICTION_AUDIT_ACTIVITY_EFFECT_ON_BOARD_ENERGY_AND_P0'])
        rows.append(dict(attempt_id=donor['attempt_id'],candidate_id=donor['candidate_id'],equalities=equalities,
            issues=issues,training_protocol_identity=donor['collection_protocol_identity'],
            target_execution_context_sha256=plan['execution_context_sha256']))
        if donor['candidate_id']==plan['candidate_id']:all_issues.extend(issues)
    return dict(schema=SCHEMA+'.explicit_resource_transfer',all8_axes=rows,
        actual_snapshot_to_origin_ns=actual_gap,declared_target_lead_ns=plan['forecast_barrier']['lead_seconds']*10**9,
        snapshot_copy_anchor_excess_ns=actual_gap-resource_executor.LEAD_NS,
        training_anchor='snapshot_time_ns_plus300s',target_anchor='post_snapshot_clock_ns_plus_declared_lead',
        collector_activity_difference_hypothesis='Unvalidated equality of startup board energy and P0 under passive wait versus full prediction and audit CPU activity',
        hypotheses_empirically_validated=False,protocol_hash_equality_used_as_physical_proof=False,
        closure_hash_difference_ignored=False,resource_marginal_transfer_qualified=not all_issues,
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
        if transfer['resource_marginal_transfer_qualified'] and donor['joint_tuple'] is not None:
            draw=dict(block_id=block,candidates={plan['candidate_id']:dict(donor_attempt_id=donor['attempt_id'],
                joint_tuple=donor['joint_tuple'],native_review=donor['native_review'],raw=donor['raw'])})
            resources=resource_campaign.resource_bridge_payload(draw,plan['candidate_id'],
                window_start_ns=a,resident_owners=plan['resident_owners'])
            original=whole_score(frame,requests,contract,{},resources)['score']['terms']
            values.update(M=original['M'],E=original['E'])
        quality=donor['actual_native_quality']
        # Q is also a hypothesis transfer to future native outcomes. Keep the
        # same physical/calendrical gate rather than using a frozen constant.
        if transfer['resource_marginal_transfer_qualified'] and quality is not None:
            require(quality['original_offered_rates']==contract['workload_rps'],'Original quality rate weights differ')
            values['Q']=quality['measured_Q']
        worlds.append(dict(draw_id=block,block_id=block,weight=F(1,2),status='GENERATED_WORLD',forecast=values,
            donor_attempt_id=donor['attempt_id'],source_resource_status=donor['resource_status']))
    return {term:dict(domain=copy.deepcopy(domain),sources=sources,
        distribution=ensemble.distribution(worlds,lambda f,t=term:f[t])) for term in 'QME'},worlds


def replay(session,gate_ref,*,allow_cpu_fixture=False):
    require(type(session) is ExpectedSession,'Exact active expected-score session required');session.check()
    cached,plan,prefix,gate=typed._prefix_and_gate(session.queue,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    require(cached['plan_ref']==session.configuration['target_plan'],'Expected preparation and live target differ')
    queue=typed.replay(session.queue,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    forecast=decode(queue['replay']['forecast'])['ensemble'];require(len(forecast['draws'])==12,'All12 original queue worlds required')
    payload=session.resources.checked_payload();transfer=transfer_axes(payload,plan,prefix,gate)
    domain=dict(public_case_sha256=digest(dict(candidate=plan['candidate'],cases=plan['cases'],
        prepared_pool=plan['prepared_pool'],context=plan['context'])),
        target_execution_context_sha256=plan['execution_context_sha256'],
        original_score_contract=plan['public_contract'],observation_axes=OBSERVATION_AXES)
    contract=checked(verify(plan['public_contract']));comparison=dict(
        public_population_sha256=digest(contract['trace']),original_score_contract=plan['public_contract'],
        observation_axes=OBSERVATION_AXES,target_manifest=session.configuration['target_manifest'])
    refs=strict_references([session.configuration_ref,payload['source_bindings'],queue['sources']])
    marginals,resource_worlds=resource_marginals(payload['intake'],plan,gate,transfer,domain,refs)
    marginals.update({term:dict(domain=copy.deepcopy(domain),sources=queue['sources'],
        distribution=forecast['six_terms'][term]) for term in 'TLP'})
    score=marginal.combine(marginals,candidate_id=plan['candidate_id'],candidate_domain=domain,
        comparison_contract=comparison,weights=contract['weights'])
    session.check()
    return encode(dict(schema=SCHEMA,status='SOURCE_AUTHENTICATED_SEPARATE_MARGINAL_HYPOTHESIS_NOT_LAUNCH_OR_ACCURACY_QUALIFICATION',
        preparation=session.configuration_ref,gate=gate_ref,prefix=gate['prefix'],target_plan=cached['plan_ref'],
        candidate_id=plan['candidate_id'],source_queue_forecast=queue,resource_transfer=transfer,
        resource_marginal_worlds=resource_worlds,expected_score=score,all12_queue_worlds_retained=True,
        all8_resource_plans_retained=True,queue_resource_draws_paired=False,sources=refs,
        source_evidence_authenticated=prefix['backend']=='native',
        source_authentication_does_not_qualify_transfer_or_accuracy=True,
        total_live_pipeline_budget_qualified=False,existing_queue_seal_reused_as_full_U_seal=False,
        native_collector_integrated=False,full_U_qualified=False,positive_delta_probability=None,
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
