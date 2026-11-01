"""Exact-native-v3 target registry and preauthenticated live replay.

This explicit v3 wrapper preserves frozen-v1 behavior. Only its source-bound
native-v3 plan auditor can construct an active target session before workers.
The collector separately enforces real fit, measured budget and review gates.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
import time
from pathlib import Path

from . import twoFIFO_matching_prospective_bridge_v1 as core
from . import twoFIFO_fresh181_native_v3 as target
from . import twoFIFO_fresh181_prefix_v1 as prefix
from .implementation_source_closure_v27 import implementation_sources,strict_references
from .stage_response_features_v1 import receipt,checked

SCHEMA='gsdeploy.twoFIFO_matching_typed_prospective_forecast_bridge.v3'
TARGET_REGISTRY='tools.predictor_validation_20261027.twoFIFO_fresh181_native_v3'
require=core.require
verify=core.verify
_FACTORY=object()


class TargetSession:
    def __init__(self,factory,payload,guard,law_session):
        require(factory is _FACTORY,'Only the exact native-v3 auditor can open a target session')
        self._json=json.dumps(payload,sort_keys=True,allow_nan=False);self._hash=core.fresh.digest(payload)
        self._guard=guard;self._law=law_session;self._active=True
    def checked_payload(self):
        require(self._active,'Typed native-v3 target session has expired')
        self._guard.check();self._law.checked_payload();value=json.loads(self._json)
        require(core.fresh.digest(value)==self._hash,'Preauthenticated native-v3 target plan changed')
        return value


@contextmanager
def open_target_session(fit_report_path,manifest_path,plan_path):
    """Authenticate exact target/15-source law before runtime startup.

    A collector must keep two separately opened producer/auditor sessions alive
    through live prefix/forecast/seal. The native collector separately verifies
    its source-bound measured budget and independent review before launch.
    """
    manifest_ref=receipt(manifest_path);snapshot=checked(verify(manifest_ref));plan_ref=receipt(plan_path)
    watched=strict_references([manifest_ref,snapshot,plan_ref,implementation_sources(__file__)])
    with core.fresh.SourceChangeGuard(watched) as guard:
        guard.check();manifest=target.audit(manifest_path);plan=target.audit_plan(plan_path)
        require(snapshot==manifest and plan_ref in [a['plan'] for a in manifest['attempts']],
            'Exact preauthenticated native-v3 target manifest/plan required')
        require(plan.get('schema')==target.SCHEMA+'.attempt' and plan['target_plan_auditor']==dict(
            module=TARGET_REGISTRY,source=receipt(target.__file__),schema=target.SCHEMA+'.attempt'),
            'Only the fixed source-bound native-v3 target auditor is registered')
        fit_ref=receipt(fit_report_path)
        require(plan['saved_matching_fit']==fit_ref and plan['forecast_barrier']['typed_producer_source']==receipt(__file__)
            and type(plan['forecast_barrier']['lead_seconds']) is int and 1<=plan['forecast_barrier']['lead_seconds']<=1200,
            'Exact predeclared saved matching fit, typed producer and predeclared live lead required')
        with core.open_law_session(fit_report_path) as law_session:
            payload=law_session.checked_payload();fit_manifest=checked(verify(payload['fit_report']['manifest']))
            require(fit_manifest['executor_manifest']==plan['matching_producer'],
                'Native-v3 target and cached matching fit belong to different campaigns')
            guard.check();session=TargetSession(_FACTORY,dict(manifest_ref=manifest_ref,plan_ref=plan_ref,
                plan=plan,fit_ref=fit_ref,source_bindings=watched),guard,law_session)
            try:yield session
            finally:session._active=False;guard.check()


def _prefix_and_gate(session,gate_ref,*,allow_cpu_fixture=False):
    """Recheck new evidence against a live preaudited exact target.

    Immutable training/target sources stay under pre-startup watchers; this
    avoids rehashing/reviewing15 raw captures during the live prefix budget.
    Actual prefix/gate/context bytes and strict snapshot derivation are checked
    each time. No mutable caller auditor or arbitrary plan schema is accepted.
    """
    require(type(session) is TargetSession,'Exact active typed native-v3 target session required')
    cached=session.checked_payload();plan=cached['plan'];gate=checked(verify(gate_ref))
    require(str(Path(gate_ref['path']).resolve())==plan['forecast_barrier']['gate_request_path']
        and gate['plan']==cached['plan_ref'] and gate['manifest']==cached['manifest_ref'],
        'Actual gate must name the preauthenticated canonical native-v3 target')
    prefix_ref=gate['prefix'];document=checked(verify(prefix_ref))
    require(str(Path(prefix_ref['path']).resolve())==plan['forecast_barrier']['prefix_path'],
        'Actual pre-cohort prefix must use the native-v3 target canonical path')
    derived=prefix.derive(plan,document['prefix_snapshot'],document['forecast_origin_ns'])
    expected=dict(schema=prefix.SCHEMA,status='OWNED_SETTLED_PRECOHORT_PREFIX_NOT_ACCURACY_QUALIFICATION',
        plan=cached['plan_ref'],manifest=cached['manifest_ref'],context=plan['context'],execution_id=plan['execution_id'],
        candidate_id=plan['candidate_id'],backend=document['backend'],prefix_snapshot=document['prefix_snapshot'],
        forecast_origin_ns=document['forecast_origin_ns'],public_calendar_sha256=plan['cases'][0]['public_arrivals_sha256'],
        ready_ownership_evidence=document['ready_ownership_evidence'],**derived,
        prefix_source_binding='ACTUAL_OWNED_RUNTIME_SNAPSHOT_BEFORE_ANY_COHORT_SEND',
        target_future_labels_read=False,current_cohort_contains_prior_labels=False,
        physical_progress_observed=False,independent_accuracy_qualified=False,
        source_authenticated_by_independent_execution_review=False,
        resource_horizon_support='UNKNOWN_MATCHING_TRAINING10_PLUS60_DOES_NOT_CERTIFY90_PLUS120',source_bindings=plan['sources'])
    require(document==expected and document['backend'] in (('native','CPU_TEST_ONLY') if allow_cpu_fixture else ('native',)),
        'Exact prefix contents and observed backend must bind the real native-v3 target')
    require(set(document['ready_ownership_evidence'])==set(core.fresh.OWNERS)
        and all(document['ready_ownership_evidence'][o] for o in core.fresh.OWNERS),
        'Complete actual READY ancestry evidence required')
    if document['backend']=='native':prefix.inspect_native_ownership(document['prefix_snapshot'],document['ready_ownership_evidence'])
    require(gate==prefix._gate_document(prefix_ref,cached['plan_ref'],plan,cached['manifest_ref'],document),
        'Complete actual gate calendar/objective must rederive against the registered target')
    session.checked_payload();return cached,plan,document,gate


def replay(session,gate_ref,*,allow_cpu_fixture=False):
    cached,plan,document,gate=_prefix_and_gate(session,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    law_payload=session._law.checked_payload()
    value=core.replay_law(law_payload['plans'],law_payload['law'],law_payload['unknown_reason'],plan,document,gate)
    session.checked_payload()
    return dict(schema=SCHEMA,status='SOURCE_AUTHENTICATED_NATIVE_V3_HYPOTHESIS_NOT_ACCURACY_OR_LAUNCH_QUALIFICATION',
        target_registry=TARGET_REGISTRY,target_auditor_source=receipt(target.__file__),
        saved_matching_fit=cached['fit_ref'],manifest=cached['manifest_ref'],plan=cached['plan_ref'],
        gate=gate_ref,prefix=gate['prefix'],execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
        public_calendar_sha256=core.fresh.digest(gate['public_requests']),
        prefix_snapshot_sha256=core.fresh.digest(document['prefix_snapshot']),replay=value,
        forecast_origin_ns=gate['forecast_origin_ns'],measurement_end_ns=gate['measurement_end_ns'],
        fixed_drain_deadline_ns=gate['fixed_drain_deadline_ns'],
        source_supported_native_hypothesis=document['backend']=='native' and value['source_supported_hypothesis_available'],
        CPU_TEST_ONLY=document['backend']!='native',
        sources=strict_references([cached['source_bindings'],session._law.watched,gate_ref,gate['prefix']]),
        native_execution_enabled=False,native_collector_status=plan['native_collector_status'],
        no_future_target_labels_read=True,model_fit=False,GPU_started=False,
        independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False,
        original_stability_evaluated=False,final_gain_eligible=False)


def produce(session,gate_path,output,*,allow_cpu_fixture=False):
    gate_ref=receipt(gate_path);snapshot=checked(verify(gate_ref))
    with core.fresh.SourceChangeGuard([gate_ref,snapshot['prefix']]) as guard:
        guard.check();value=replay(session,gate_ref,allow_cpu_fixture=allow_cpu_fixture);guard.check()
        ref=prefix.save(output,value);guard.check();session.checked_payload();return ref


def audit_report(session,report_path,*,require_native=False):
    ref=receipt(report_path);document=checked(verify(ref))
    require(document.get('schema')==SCHEMA,'Exact typed-v3 report required; typed-v1 is not relabeled')
    with core.fresh.SourceChangeGuard([ref,document['gate'],document['prefix']]) as guard:
        guard.check();expected=replay(session,document['gate'],allow_cpu_fixture=not require_native)
        require(document==expected,'All typed-v3 target/prefix/law/181-world outputs must exactly replay')
        if require_native:require(expected['source_supported_native_hypothesis'] is True,'Native-v3 seal HOLD for missing/unknown source hypothesis')
        guard.check();session.checked_payload();verify(ref);return expected


def seal(session,gate_path,report_path,output,*,clock_ns=time.perf_counter_ns,report_reference=None):
    """Full typed replay and durable seal; never consume generic completions."""
    gate_ref=receipt(gate_path);gate_snapshot=checked(verify(gate_ref))
    report_ref=receipt(report_path) if report_reference is None else verify(report_reference)
    require(report_ref['path']==str(Path(report_path).resolve()),'Exact fixed typed report receipt required')
    with core.fresh.SourceChangeGuard([report_ref,gate_ref,gate_snapshot['prefix']]) as outer_guard:
        outer_guard.check()
        cached,plan,document,gate=_prefix_and_gate(session,gate_ref,
            allow_cpu_fixture=clock_ns is not time.perf_counter_ns)
        require(str(Path(report_path).resolve())==plan['forecast_barrier']['typed_report_path']
            and str(Path(output).resolve())==plan['forecast_barrier']['seal_path'],
            'Only canonical predeclared typed report/seal paths may release the barrier')
        report=audit_report(session,report_path,require_native=clock_ns is time.perf_counter_ns)
        begin=clock_ns();require(type(begin) is int and begin<gate['forecast_origin_ns'],
            'Typed prediction and full audit must finish strictly before the origin')
        value=dict(schema=SCHEMA+'.prediction_seal',typed_report=report_ref,gate=gate_ref,
            plan=cached['plan_ref'],prefix=gate['prefix'],execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
            forecast_origin_ns=gate['forecast_origin_ns'],seal_begin_ns=begin,
            complete_replay_forecast_sha256=report['replay']['forecast_sha256'],
            source_supported_native_hypothesis=report['source_supported_native_hypothesis'],
            CPU_TEST_ONLY=document['backend']!='native' or clock_ns is not time.perf_counter_ns,
            validation_labels_read=False,independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False)
        outer_guard.check();verify(report_ref);result=prefix.save(output,value)
        require(clock_ns()<gate['forecast_origin_ns'],'Typed seal persistence crossed the origin; zero targets required')
        session.checked_payload();verify(report_ref);outer_guard.check();return result


def audit_seal_structure(session,gate_ref,seal_ref,*,now_ns,allow_cpu_fixture=False):
    """Cheap acceptance after seal's independent full replay, still source-bound.

    The collector must call seal with its separately authenticated auditor
    session first. This verifies the immutable result before its final
    acceptance checkpoint; it cannot replace that full auditor call.
    """
    cached,plan,document,gate=_prefix_and_gate(session,gate_ref,allow_cpu_fixture=allow_cpu_fixture)
    seal_document=checked(verify(seal_ref));report_ref=seal_document['typed_report'];report=checked(verify(report_ref))
    require(str(Path(seal_ref['path']).resolve())==plan['forecast_barrier']['seal_path']
        and str(Path(report_ref['path']).resolve())==plan['forecast_barrier']['typed_report_path'],
        'Canonical typed prediction paths required')
    require(seal_document==dict(schema=SCHEMA+'.prediction_seal',typed_report=report_ref,gate=gate_ref,
        plan=cached['plan_ref'],prefix=gate['prefix'],execution_id=plan['execution_id'],candidate_id=plan['candidate_id'],
        forecast_origin_ns=gate['forecast_origin_ns'],seal_begin_ns=seal_document['seal_begin_ns'],
        complete_replay_forecast_sha256=report['replay']['forecast_sha256'],
        source_supported_native_hypothesis=report['source_supported_native_hypothesis'],
        CPU_TEST_ONLY=seal_document['CPU_TEST_ONLY'],validation_labels_read=False,
        independent_accuracy_qualified=False,resource_horizon_qualified=False,full_U_qualified=False)
        and report['schema']==SCHEMA and report['gate']==gate_ref and report['prefix']==gate['prefix']
        and type(seal_document['seal_begin_ns']) is int and seal_document['seal_begin_ns']<=now_ns<gate['forecast_origin_ns']
        and (allow_cpu_fixture or seal_document['CPU_TEST_ONLY'] is False and report['source_supported_native_hypothesis'] is True),
        'Full source-bound typed seal must be accepted strictly before the origin')
    session.checked_payload();return seal_document
