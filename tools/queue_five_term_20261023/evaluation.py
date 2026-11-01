"""Original metric primitives with fixed five-term U and diagnostic P replay."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import math

from tools.queue_refined_20261023 import evaluation as original
from tools.queue_refined_20261023.admission import measurement_identity, capacity_envelope, quality_identity_covers
from tools.queue_coupled_utility_20261023.runtime_window import feedback
from tools.queue_five_term_20261023.policy import FIXED_WEIGHTS, OBJECTIVE_SCHEMA, validate_weights


def audit_phase(config, raw, phase='measurement'):
    from tools.queue_five_term_20261023.protocol import make_dispatcher, validate_contract
    rows = raw['request_records' if phase == 'measurement' else phase + '_request_records']
    window = raw['measurement_window' if phase == 'measurement' else phase + '_window']
    errors = []
    prior = {}
    try:
        validate_contract(raw['policy_contract'])
        for row in rows:
            routing = row['routing_decision']
            for event in routing['newly_observed_completions']:
                previous = prior.get(event.get('request_id'))
                if previous is None or previous.get('status') != 'completed' or event != feedback(previous):
                    errors.append('FEEDBACK_ALLOWLIST_OR_IDENTITY_MISMATCH:' + row['request_id'])
            prior[row['request_id']] = row
        if errors:
            return dict(verified=False, errors=errors, phase=phase)

        class ReplayDispatcher:
            def __init__(self, contract):
                self.inner = make_dispatcher(contract)
                self.inner.start_phase(window['start_time_ns'], window['end_time_ns'])
                self.index = 0

            def choose(self, task, pool_id, now_ns, snapshot):
                row = rows[self.index]
                self.index += 1
                self.inner.observe_completed(deepcopy(row['routing_decision']['newly_observed_completions']), now_ns)
                self.inner.set_arrival(row['request_id'], row['scheduled_time_ns'])
                return self.inner.choose(task, pool_id, now_ns, snapshot)

        audit = original.audit_phase(config, raw, phase, dispatcher_factory=ReplayDispatcher)
        audit['completion_feedback_allowlist_verified'] = True
        audit['shared_gpu_policy_replayed'] = True
        return audit
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return dict(verified=False, errors=['COUPLED_AUDIT_EXCEPTION:' + str(exc)], phase=phase)


def evaluate_run(config, raw):
    if hasattr(config, 'config') and not hasattr(config, 'catalog'):
        config = config.config
    audit = None
    try:
        validate_weights(config.weights)
        validate_weights(raw['policy_contract'].get('weights'))
        if raw['policy_contract'].get('objective_schema') != OBJECTIVE_SCHEMA:
            raise ValueError('FIVE_TERM_OBJECTIVE_IDENTITY_MISMATCH')
        errors = original._contract_inputs(config, raw)
        audit = audit_phase(config, raw)
        if raw.get('warmup_window') and raw['warmup_window'].get('duration_seconds', 0) > 0:
            audit['warmup'] = audit_phase(config, raw, 'warmup')
            if not audit['warmup']['verified']:
                errors.append('WARMUP_POLICY_AUDIT_FAILED')
        window = raw['measurement_window']
        if window.get('aborted') or window['drain_end_time_ns'] < window['end_time_ns']:
            raise ValueError('WINDOW_ABORTED')
        evidence = original.reconstruct(raw, {a: SimpleNamespace(service_time_ms=config.catalog['models'][m]['service_time_ms'])
                                             for a, m in raw['replica_to_family'].items()}, original.DEFAULT_THRESHOLDS)
        stability = original.classify(evidence, original.DEFAULT_THRESHOLDS)
        measurement = SimpleNamespace(measurement_seconds=window['duration_seconds'], workload_rps=config.workload,
                                      routing_rps={t: {a: 0. for a in raw['replica_to_family']} for t in config.workload},
                                      deployment={a: 1 for a in raw['replica_to_family']})
        per_task, system, _ = original.summarize(deepcopy(raw['request_records']), measurement, window)
        stats = {t: dict(mean_ms=per_task[t]['mean_e2e_latency_ms'], p95_ms=per_task[t]['p95_e2e_latency_ms'])
                 for t in config.workload}
        if not raw['request_records'] or any(s['mean_ms'] is None or s['p95_ms'] is None for s in stats.values()):
            raise ValueError('MISSING_TASK_OBSERVATIONS')
        identity = measurement_identity(config, raw)
        identity_ok = raw.get('measurement_identity') == identity
        memory = raw.get('memory_evidence', {})
        energy = raw.get('energy_evidence', {})
        memory_ok = (memory.get('measured') is True and memory.get('resource_valid') is True
                     and memory.get('measurement_identity') == identity
                     and type(memory.get('safe_memory_mib')) in (int, float)
                     and math.isfinite(memory['safe_memory_mib']) and memory['safe_memory_mib'] >= 0)
        M = memory['safe_memory_mib'] / config.gpu_memory_mib if memory_ok else None
        E = original.energy_norm(energy, window['duration_seconds'], config.energy_reference_watts, config.startup_amortization_seconds)
        energy_ok = (energy.get('kind') == 'measured' and energy.get('profile_key') == identity
                     and energy.get('measurement_identity') == identity and E is not None)
        if not energy_ok:
            E = None
        scoring_raw = raw if isinstance(raw.get('preflight'), dict) else dict(raw, preflight={})
        Q, quality = original.measure_quality(config.catalog, scoring_raw, config.workload)
        arithmetic = original.components(stats, config.workload, config.latency_reference_ms,
                                         1 - system['completed_within_window'] / len(raw['request_records']), M, config.weights, E, Q)
        diagnostic = arithmetic.pop('U')
        capacity = capacity_envelope(config, raw['plan'], raw['policy_contract'])
        constraints = capacity['constraints_verified'] and audit['verified']
        quality_ok = quality_identity_covers(config, raw)
        cohort = audit['verified'] and not errors
        eligible = (raw.get('status') == 'completed' and raw.get('backend') == 'native' and memory_ok
                    and M <= 1 + 1e-6 / config.gpu_memory_mib
                    and stability['stability_state_v2'] in {'steady', 'recoverable_congested'}
                    and quality_ok and arithmetic['utility_complete'] and cohort and constraints and identity_ok
                    and system['completion_ratio'] == 1 and window.get('offered_requests') == len(raw['request_records'])
                    and energy_ok)
        return dict(stability=stability, stability_evidence=evidence,
                    dynamic_audit=dict(**audit, identity_errors=errors, measurement_identity_verified=identity_ok,
                                       capacity_evidence=capacity),
                    utility=dict(**arithmetic, U=diagnostic if eligible else None, diagnostic_U=diagnostic,
                                 objective_schema=OBJECTIVE_SCHEMA, objective_weights=dict(FIXED_WEIGHTS),
                                 active_utility_terms=['Q','T','L','M','E'], P_evaluation_role='diagnostic_only',
                                 eligible=bool(eligible), memory_verified=memory_ok, quality_identity_verified=quality_ok,
                                 quality_evidence=quality, cohort_verified=bool(cohort), constraints_verified=bool(constraints),
                                 energy_verified=energy_ok, absolute_quality_gate=False,
                                 original_stability_thresholds_unchanged=True,
                                 routing_semantics='causal_shared_gpu_feedback_v1'))
    except (KeyError, ValueError, TypeError, OSError) as exc:
        retained_audit = audit if audit is not None else dict(verified=False, errors=[str(exc)])
        retained_audit['metric_evaluation_errors'] = [str(exc)]
        return dict(stability={'stability_state_v2': 'invalid_observation'},
                    dynamic_audit=retained_audit,
                    utility=dict(U=None, diagnostic_U=None, eligible=False, reason=str(exc)))
