"""Measured dynamic routing evaluation with original six-term/stability rules.

Only cohort/identity verification changes: original arrivals and SamplePool are
replayed, while aliases must follow the saved causal policy decisions rather
than a static SWRR fiction. Observed route shares never prove prior capacity.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import importlib
import hashlib
import json

from tools.spillover_capacity_20261022.exposure import frozen_api
_API=frozen_api();_PACKAGE=_API['trace'].__package__
energy_norm=importlib.import_module(_PACKAGE+'.energy').energy_norm
summarize=importlib.import_module(_PACKAGE+'.metrics').summarize
components=importlib.import_module(_PACKAGE+'.prediction').components
measure_quality=_API['quality'].measure_quality;sample_totals=_API['quality'].sample_totals
SamplePool=_API['workload'].SamplePool
_stability=importlib.import_module(_PACKAGE+'.stability')
DEFAULT_THRESHOLDS=_stability.DEFAULT_THRESHOLDS;reconstruct=_stability.reconstruct;classify=_stability.classify
make_trace=_API['trace'].make_trace;trace_digest=_API['trace'].trace_digest
from tools.queue_refined_20261023.admission import canonical_digest,policy_identity,measurement_identity,source_checks,capacity_envelope,quality_identity_covers,potential_paths,digest

def decision_digest(records):
    payload=[dict(request_id=r['request_id'],model=r['model'],routing_decision=r['routing_decision']) for r in records]
    # Exact runtime_window encoding (including ASCII escaping) is preserved.
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _sample_path(config,raw):
    path=raw.get('samples_path',raw.get('sample_pool_path'))
    return Path(path) if path else Path(config.catalog['quality']['identity']['source'])/'runtime-inputs.jsonl'

def _contract_inputs(config,raw):
    c=raw['policy_contract'];issues=[]
    if raw.get('policy_identity')!=policy_identity(c):issues.append('POLICY_IDENTITY_MISMATCH')
    if raw.get('strategy')!=c.get('strategy'):issues.append('STRATEGY_MISMATCH')
    okay,source_issues=source_checks(c);issues+=source_issues
    if c.get('weights')!=config.weights or c.get('latency_refs_ms')!=config.latency_reference_ms:issues.append('ORIGINAL_OBJECTIVE_CONFIG_MISMATCH')
    if {a:s['family'] for a,s in c['aliases'].items()}!=raw['replica_to_family']:issues.append('ALIAS_IDENTITY_MISMATCH')
    paths=potential_paths(config,raw['plan'],c)
    # Policy gets historical aggregate loss only. Per-request references remain
    # exclusively in measure_quality after routing decisions have been checked.
    for m,t in paths:
        scores=config.catalog['quality']['paths'][m][t]['samples'].values()
        totals=[sample_totals(t,s) for s in scores];expected=sum(a for a,b in totals)/sum(b for a,b in totals)
        value=c.get('quality_loss',{}).get(m,{}).get(t)
        if type(value) not in (int,float) or not math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-12):issues.append('AGGREGATE_QUALITY_IDENTITY_MISMATCH:'+m+'/'+t)
    return issues

def audit_phase(config,raw,phase='measurement',dispatcher_factory=None):
    """Replay phase-local policy state and every visible queue/completion event.

    completion_time_ns is host receive time, not collector application time.
    newly_observed_completions records when that result was actually visible to
    routing. The audit checks a monotone causal processing order; it does not
    invent unrecorded worker start or collector application timestamps.
    """
    errors=[]
    try:
        contract=raw['policy_contract'];window=raw['measurement_window' if phase=='measurement' else phase+'_window']
        rows=raw['request_records' if phase=='measurement' else phase+'_request_records'];transport=raw['transport_config']
        if dispatcher_factory is None:
            from tools.queue_refined_20261023.protocol import make_dispatcher
            dispatcher_factory=make_dispatcher
        dispatcher=dispatcher_factory(contract) # fresh SWRR/token bucket per phase
        trace=make_trace(config.workload,window['duration_seconds'],transport['random_seed'],transport['arrival_process'])
        pool=SamplePool(_sample_path(config,raw),list(config.workload),transport['random_seed'])
        if digest(_sample_path(config,raw))!=config.catalog['quality']['identity']['input_pool_sha256']:errors.append('INPUT_POOL_SHA_MISMATCH')
        if len(rows)!=len(trace) or window.get('offered_requests')!=len(trace):errors.append('COHORT_COUNT_MISMATCH')
        if window.get('trace_sha256')!=trace_digest(trace):errors.append('TRACE_IDENTITY_MISMATCH')
        if window.get('dispatch_decisions_sha256')!=decision_digest(rows):errors.append('DECISION_DIGEST_MISMATCH')
        byid={r['request_id']:r for r in rows}
        if len(byid)!=len(rows):errors.append('DUPLICATE_REQUEST_IDS')
        prior={};observed=set();last_now=-1;aliases=raw['replica_to_family'];guard_events=0
        for i,((offset,task),row) in enumerate(zip(trace,rows)):
            item=pool.next(task);rid=f'{phase}:{i}';routing=row['routing_decision'];now=routing['decision_time_ns'];snapshot=routing['snapshot']
            if (row.get('request_id')!=rid or row.get('phase')!=phase or row.get('arrival_index')!=i or row.get('task')!=task
                or row.get('pool_id')!=item['pool_id'] or row.get('sample_id')!=item['sample'].get('id')
                or row.get('scheduled_time_ns')!=window['start_time_ns']+round(offset*1e9)):errors.append('COHORT_ROW_MISMATCH:'+rid)
            enqueued=row.get('enqueued_time_ns');submit=row.get('actual_submit_time_ns');finish=row.get('completion_time_ns')
            if (type(now) is not int or now<last_now or now<row['scheduled_time_ns']
                or type(enqueued) is not int or enqueued<now or (submit is not None and (type(submit) is not int or submit<enqueued))
                or (finish is not None and (submit is None or type(finish) is not int or finish<submit))):
                errors.append('NONCAUSAL_TIMESTAMPS:'+rid)
            last_now=now
            newly=routing.get('newly_observed_completions')
            if not isinstance(newly,list):raise ValueError('MISSING_OBSERVED_COMPLETION_VISIBILITY_LOG')
            for event in newly:
                cid=event['request_id'];previous=prior.get(cid)
                if (cid in observed or previous is None or previous.get('status')!='completed'
                    or event.get('model')!=previous.get('model') or event.get('completion_time_ns')!=previous.get('completion_time_ns')
                    or type(event.get('completion_time_ns')) is not int or event['completion_time_ns']>now):
                    errors.append('INVALID_OBSERVED_COMPLETION:'+rid+':'+str(cid))
                observed.add(cid)
            expected={a:dict(outstanding=[],last_completed_ns=None) for a in aliases}
            for cid,previous in prior.items():
                alias=previous['model'];complete=previous.get('completion_time_ns')
                if cid in observed:
                    old=expected[alias]['last_completed_ns'];expected[alias]['last_completed_ns']=max(old or complete,complete)
                else:
                    sent=previous.get('actual_submit_time_ns')
                    expected[alias]['outstanding'].append(dict(request_id=cid,task=previous['task'],pool_id=previous['pool_id'],
                        enqueued_ns=previous['enqueued_time_ns'],actual_submit_ns=sent if sent is not None and sent<=now else None))
            for state in expected.values():state['outstanding'].sort(key=lambda r:(r['enqueued_ns'],r['request_id']))
            if snapshot!=expected:errors.append('QUEUE_SNAPSHOT_MISMATCH:'+rid)
            decision=dispatcher.choose(task,row['pool_id'],now,deepcopy(snapshot))
            if decision!=routing['decision']:errors.append('POLICY_REPLAY_MISMATCH:'+rid)
            if decision.get('alias')!=row.get('model'):errors.append('CHOSEN_ALIAS_MISMATCH:'+rid)
            if 'rate_guard' in decision:guard_events+=1
            prior[rid]=row
        return dict(verified=not errors,errors=errors,phase=phase,requests=len(rows),trace_requests=len(trace),
            observed_completion_visibility='phase-local causal monotone host receive/application visibility, not worker true finish',
            guard_events_replayed=guard_events,static_SWRR_fiction_used=False)
    except (KeyError,TypeError,ValueError,OSError,ImportError) as exc:
        return dict(verified=False,errors=errors+['AUDIT_EXCEPTION:'+str(exc)],phase=phase,static_SWRR_fiction_used=False)

def evaluate_run(config,raw):
    """Original measured arithmetic and stability; dynamic identity/route audit."""
    if hasattr(config,'config') and not hasattr(config,'catalog'):config=config.config
    try:
        errors=_contract_inputs(config,raw);audit=audit_phase(config,raw)
        if raw.get('warmup_window') and raw['warmup_window'].get('duration_seconds',0)>0:
            warmup=audit_phase(config,raw,'warmup');audit['warmup']=warmup
            if not warmup['verified']:errors.append('WARMUP_POLICY_AUDIT_FAILED')
        window=raw['measurement_window']
        if window.get('aborted') or window['drain_end_time_ns']<window['end_time_ns']:raise ValueError('WINDOW_ABORTED')
        evidence=reconstruct(raw,{a:SimpleNamespace(service_time_ms=config.catalog['models'][m]['service_time_ms']) for a,m in raw['replica_to_family'].items()},DEFAULT_THRESHOLDS)
        stability=classify(evidence,DEFAULT_THRESHOLDS)
        measurement=SimpleNamespace(measurement_seconds=window['duration_seconds'],workload_rps=config.workload,
            routing_rps={t:{a:0. for a in raw['replica_to_family']} for t in config.workload},deployment={a:1 for a in raw['replica_to_family']})
        per_task,system,_=summarize(deepcopy(raw['request_records']),measurement,window)
        stats={t:dict(mean_ms=per_task[t]['mean_e2e_latency_ms'],p95_ms=per_task[t]['p95_e2e_latency_ms']) for t in config.workload}
        if not raw['request_records'] or any(s['mean_ms'] is None or s['p95_ms'] is None for s in stats.values()):raise ValueError('MISSING_TASK_OBSERVATIONS')
        identity=measurement_identity(config,raw)
        identity_ok=raw.get('measurement_identity')==identity
        memory=raw.get('memory_evidence',{});energy=raw.get('energy_evidence',{})
        memory_ok=(memory.get('measured') is True and memory.get('resource_valid') is True and memory.get('measurement_identity')==identity
            and type(memory.get('safe_memory_mib')) in (int,float) and math.isfinite(memory['safe_memory_mib']) and memory['safe_memory_mib']>=0)
        M=memory['safe_memory_mib']/config.gpu_memory_mib if memory_ok else None
        E=energy_norm(energy,window['duration_seconds'],config.energy_reference_watts,config.startup_amortization_seconds)
        energy_ok=energy.get('kind')=='measured' and energy.get('profile_key')==identity and energy.get('measurement_identity')==identity and E is not None
        if not energy_ok:E=None
        # Mock collection deliberately has no native preflight. Preserve the
        # original scorer's fail-closed identity rule without dereferencing
        # None or replacing missing native output evidence with historical Q.
        scoring_raw=raw if isinstance(raw.get('preflight'),dict) else dict(raw,preflight={})
        Q,quality=measure_quality(config.catalog,scoring_raw,config.workload)
        arithmetic=components(stats,config.workload,config.latency_reference_ms,1-system['completed_within_window']/len(raw['request_records']),M,config.weights,E,Q)
        diagnostic=arithmetic.pop('U');capacity=capacity_envelope(config,raw['plan'],raw['policy_contract'])
        # Replaying the guarded dispatcher is required for its prior bound to
        # apply. Merely attaching a guard declaration never verifies capacity.
        constraints=capacity['constraints_verified'] and audit['verified']
        quality_ok=quality_identity_covers(config,raw);cohort=audit['verified'] and not errors
        eligible=(raw.get('status')=='completed' and raw.get('backend')=='native' and memory_ok
            and M<=1+1e-6/config.gpu_memory_mib and stability['stability_state_v2'] in {'steady','recoverable_congested'}
            and quality_ok and arithmetic['utility_complete'] and cohort and constraints and identity_ok
            and system['completion_ratio']==1 and window.get('offered_requests')==len(raw['request_records'])
            and ('E' not in config.weights or energy_ok))
        return dict(stability=stability,stability_evidence=evidence,dynamic_audit=dict(**audit,identity_errors=errors,
            measurement_identity_verified=identity_ok,expected_measurement_identity=identity,capacity_evidence=capacity),
            utility=dict(**arithmetic,U=diagnostic if eligible else None,diagnostic_U=diagnostic,eligible=bool(eligible),
                memory_scope=memory.get('scope'),memory_verified=memory_ok,quality_admission_verified=quality_ok,
                quality_identity_verified=quality_ok,quality_evidence=quality,cohort_verified=bool(cohort),
                constraints_verified=bool(constraints),energy_verified=energy_ok,absolute_quality_gate=False,
                routing_semantics='causal_policy_replay_v1',original_stability_thresholds_unchanged=True))
    except (KeyError,TypeError,ValueError,OSError) as exc:
        return dict(stability={'stability_state_v2':'invalid_observation'},stability_evidence={'reason':str(exc)},
            dynamic_audit=dict(verified=False,errors=[str(exc)]),utility=dict(U=None,diagnostic_U=None,eligible=False,reason=str(exc)))
