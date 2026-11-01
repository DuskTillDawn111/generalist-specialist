"""Explicit new policy contracts over immutable worker/admission primitives."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.queue_refined_20261023 import protocol as original
from tools.queue_five_term_20261023.policy import CoupledUtilityPolicy, FIXED_WEIGHTS, OBJECTIVE_SCHEMA, validate_weights

digest = original.digest
receipt = original.receipt
write = original.write
nominal_plan = original.nominal_plan
SCHEMA = 'gsdeploy.five_term_shared_gpu_policy.v1'
BASELINE_SWITCH_SCHEMA = 'gsdeploy.five_term_legacy_route_switch.v1'


def validate_contract(contract):
    validate_weights(contract.get('weights'))
    if contract.get('objective_schema') != OBJECTIVE_SCHEMA:
        raise ValueError('Explicit five-term objective identity required')
    original.validate_contract(contract)
    required = {str(p.resolve()) for p in Path(__file__).parent.glob('*.py')}
    if not required.issubset(contract['source_fingerprints']):
        raise ValueError('Missing five-term implementation source bindings')
    switch = contract.get('baseline_switch')
    if switch is not None:
        if (set(switch) != {'schema', 'task', 'mode', 'allow_text_generalist'}
            or switch['schema'] != BASELINE_SWITCH_SCHEMA or switch['task'] != 'text'
            or switch['mode'] != 'latency' or type(switch['allow_text_generalist']) is not bool
            or contract['strategy'] != 'queue_latency' or contract['mode'] != 'latency'
            or contract.get('coupled') is not None or contract.get('refinement') is not None):
            raise ValueError('Explicit legacy latency route-switch contract required')
    value = contract.get('coupled')
    if value is None:
        return True
    if value.get('schema') != SCHEMA:
        raise ValueError('Explicit coupled policy schema required')
    if receipt(value['resource_model']['path']) != value['resource_model']:
        raise ValueError('Resource evidence receipt changed')
    if value['allow_spill'] != (contract['strategy'] != 'queue_specialist_only'):
        raise ValueError('Coupled spill/control strategy mismatch')
    if contract.get('refinement') is not None:
        raise ValueError('Select one online policy implementation')
    if value['params']['exposure_rate_rps'] != contract['guard']['rate_rps']:
        raise ValueError('Online bucket differs from prior capacity envelope')
    if value['params']['exposure_burst'] != contract['guard']['burst']:
        raise ValueError('Online burst differs from prior capacity envelope')
    from tools.queue_coupled_utility_20261023.resources import load_provider
    provider = load_provider(json.loads(Path(value['resource_model']['path']).read_text()), contract['deployment'])
    workload = {'text': contract['text_rate'], **{t: sum(v.values()) for t, v in contract['background_routing'].items()}}
    CoupledUtilityPolicy(contract['aliases'], provider, contract['quality_loss'], contract['weights'],
                         contract['latency_refs_ms'], workload, allow_spill=value['allow_spill'], **value['params'])
    return True


def make_contract(template, resource_model, *, allow_spill=True, params=None, sources=None):
    contract = copy.deepcopy(template)
    contract.pop('probe', None)
    contract.pop('refinement', None)
    contract.pop('coupled', None)
    contract.pop('baseline_switch', None)
    contract['strategy'] = 'queue_quality' if allow_spill else 'queue_specialist_only'
    contract['mode'] = 'quality_aware'
    contract['weights'] = dict(FIXED_WEIGHTS)
    contract['objective_schema'] = OBJECTIVE_SCHEMA
    contract['P_evaluation_role'] = 'diagnostic_only'
    contract['source_fingerprints'] = dict(sources or contract['source_fingerprints'])
    rate = contract['text_rate']
    # The same declared budget applies to both arms; the control never spends it.
    contract['guard'] = dict(schema='queue_aware_token_bucket_v1', task='text', family='generalist_2', rate_rps=rate / 8, burst=1.)
    defaults = dict(scenarios=['mean', 'p90'], utility_margin=0., energy_reference_watts=450.,
                    exposure_rate_rps=rate / 8, exposure_burst=1., cooldown_ms=30000.,
                    drift_ratio_limit=2., memory_delta_penalty_upper_norm=None,
                    maximum_pending_for_forecast=32)
    defaults.update(params or {})
    contract['coupled'] = dict(schema=SCHEMA, resource_model=receipt(resource_model), allow_spill=allow_spill, params=defaults)
    contract['local_objective'] = 'Observed-prefix Q/T/L and remaining whole-board E conditional proxy with Q/T/L/M/E=.2, P=0 diagnostic only; physical M and future counterfactual U unknown.'
    validate_contract(contract)
    return contract


def make_legacy_contract(template, *, allow_spill=True, sources=None):
    """Keep the legacy latency scorer identical while toggling text G2 access."""
    contract = copy.deepcopy(template)
    for key in ('probe', 'refinement', 'coupled', 'baseline_switch'):
        contract.pop(key, None)
    contract.update(strategy='queue_latency', mode='latency', weights=dict(FIXED_WEIGHTS),
                    objective_schema=OBJECTIVE_SCHEMA, P_evaluation_role='diagnostic_only')
    contract['source_fingerprints'] = dict(sources or contract['source_fingerprints'])
    contract['baseline_switch'] = dict(schema=BASELINE_SWITCH_SCHEMA, task='text', mode='latency',
                                       allow_text_generalist=allow_spill)
    contract['local_objective'] = 'Legacy latency-only routing baseline; evaluated under fixed Q/T/L/M/E=.2, P=0.'
    validate_contract(contract)
    return contract


class Dispatcher:
    def __init__(self, contract):
        validate_contract(contract)
        self.contract = contract
        self.aliases = contract['aliases']
        self.arrival = None
        if contract.get('coupled') is None:
            self.legacy = original.make_dispatcher(contract)
            switch = contract.get('baseline_switch')
            if switch is not None:
                aliases = contract['aliases']
                if not switch['allow_text_generalist']:
                    aliases = {a: s for a, s in aliases.items() if s['family'] == 'text'}
                policy = original.QueueAwarePolicy(aliases, contract['service_ms'], contract['quality_loss'],
                                                   contract['weights'], contract['latency_refs_ms'], mode='latency',
                                                   overdue_remaining_ms=contract['overdue_remaining_ms'])
                if contract.get('guard'):
                    from tools.queue_refined_20261023.guarded_policy import RateLimitedPolicy
                    policy = RateLimitedPolicy(policy, contract['guard'])
                self.legacy.policy = policy
            self.policy = None
            return
        self.legacy = None
        from tools.queue_coupled_utility_20261023.resources import load_provider
        value = contract['coupled']
        provider = load_provider(json.loads(Path(value['resource_model']['path']).read_text()), contract['deployment'])
        workload = {'text': contract['text_rate'], **{t: sum(v.values()) for t, v in contract['background_routing'].items()}}
        self.policy = CoupledUtilityPolicy(self.aliases, provider, contract['quality_loss'], contract['weights'],
                                            contract['latency_refs_ms'], workload, allow_spill=value['allow_spill'], **value['params'])
        api = original.frozen_api()
        plan = nominal_plan(contract)
        _, routes = api['trace'].expand_routes(plan['deployment'], plan['routing'])
        self.router = api['routing'].Router(routes)

    def start_phase(self, start_ns, deadline_ns):
        if self.policy:
            self.policy.start_phase(start_ns, deadline_ns)

    def set_arrival(self, request_id, scheduled_ns):
        self.arrival = dict(request_id=request_id, arrival_ns=scheduled_ns)
        if self.policy:
            self.policy.set_arrival(request_id, scheduled_ns)

    def observe_completed(self, records, now_ns):
        if any(r['completion_time_ns'] > now_ns for r in records):
            raise ValueError('Future completion feedback')
        if self.policy:
            self.policy.observe_completed(records)

    def choose(self, task, pool_id, now_ns, snapshot):
        if self.legacy:
            result = self.legacy.choose(task, pool_id, now_ns, snapshot)
            result['baseline_switch'] = copy.deepcopy(self.contract.get('baseline_switch'))
            result['objective_schema'] = OBJECTIVE_SCHEMA
            result['P_evaluation_role'] = 'diagnostic_only'
            result['online_objective_scope'] = 'Frozen legacy/refined surrogate baseline, evaluated with new five-term weights'
            return result
        if self.arrival is None:
            raise ValueError('Current arrival identity is required')
        if task != 'text':
            alias = self.router.choose(task)
            self.policy.current = None
            result = dict(alias=alias, task=task, pool_id=pool_id, now_ns=now_ns,
                          status='FROZEN_SWRR_BACKGROUND', trigger='STATIC_ROUTE', audit=[])
        else:
            result = self.policy.choose(task, pool_id, now_ns, snapshot)
        self.policy.observe_admission(dict(self.arrival, alias=result['alias'], task=task, pool_id=pool_id))
        self.arrival = None
        result['strategy'] = self.contract['strategy']
        return result


def make_dispatcher(contract):
    return Dispatcher(contract)


def validate_packet(packet):
    changed = []
    for source in [*packet['inputs'], *packet['sources'].values()]:
        try:
            if receipt(source['path']) != source:
                changed.append(source['path'])
        except OSError:
            changed.append(source['path'])
    for job in packet['jobs']:
        if digest(job['identity']) != job['id'] or digest(job['policy_contract']) != job['identity']['policy_contract_sha256']:
            changed.append(job['label'])
        if job['nominal_plan'] != nominal_plan(job['policy_contract']):
            changed.append(job['label'] + ':nominal_plan')
        if job.get('base_packet') and receipt(job['base_packet']['path']) != job['base_packet']:
            changed.append(job['label'] + ':base_packet')
        validate_contract(job['policy_contract'])
    return dict(valid=not changed, changed=changed, GPU_started=False)
