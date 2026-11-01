"""Cutoff-safe risk features and separate offline queue outcomes.

Public requests have exactly four fields. The feature path accepts only a
reader-visible observer projection, never completed runtime records. Full raw
timestamps enter a separate offline labeling function. This module does not
infer active GPU residual work or change the original stability/utility rule.
"""
from __future__ import annotations

import copy
import json
import math

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.recovery_risk import build_features
from gsdeploy.resources import identity

SCHEMA = 'gsdeploy.native_queue_risk_protocol.v9'
SCALES = (250_000_000, 1_000_000_000, 2_000_000_000)


def _ns(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer native timestamp required')
    return value


def protocol(plan):
    """Freeze interpretation of the plan's existing risk target before fitting."""
    if plan.get('split') != 'training' or not plan.get('training_only'):
        raise ValueError('Explicit training plan required')
    result = dict(schema=SCHEMA, split='training', training_only=True,
        execution_context_sha256=plan['execution_context_sha256'],
        target=copy.deepcopy(plan['risk_target']), scales_ns=list(SCALES),
        violation='Any arrival in [cutoff,cutoff+horizon) whose arrival-to-parent-reader receipt exceeds its original task latency reference',
        recovery='First instant at or after cutoff with no arrived request awaiting worker START, including subsequent declared arrivals; zero when already empty',
        queue_feature='Arrived requests without reader-visible START; full solo reference work only, no active residual estimate',
        hol_feature='Oldest visible waiting age behind an owner with visible START and no visible worker READY',
        envelope='Maximum public offered reference work in any half-open window of each scale in the remaining declared trace',
        censoring='Conservative completed-capture training adapter: require full original cohort STARTs, full future horizon and all target-cohort receipts; otherwise omit with explicit censored reason',
        latency_reference_ms=copy.deepcopy(plan['latency_reference_ms']),
        target_tuning_from_labels_allowed=False, stability_rule_changed=False,
        six_term_utility_changed=False, calibrated_probability=False)
    result['protocol_sha256'] = identity(result)
    return result


def public_trace(plan, phase_start_ns):
    return [dict(request_id=r['request_id'], alias=r['alias'], task=r['task'],
                 pool_id=r['pool_id'], arrival_ns=phase_start_ns+r['offset_ns'])
            for r in plan['trace']]


def visible_projection(records, now_ns):
    """Expose only already received START/READY facts; do not project phases/outcomes."""
    events = []
    for row in records:
        for event in row.get('execution_events', []):
            if 'event' not in event and event['received_time_ns'] <= now_ns:
                events.append(dict(request_id=row['request_id'], alias=row['model'],
                    kind='START', received_ns=event['received_time_ns']))
        ready = row.get('previous_handler_next_request_ready')
        if ready and ready['received_time_ns'] <= now_ns:
            events.append(dict(request_id=row['request_id'], alias=row['model'],
                kind='READY', received_ns=ready['received_time_ns']))
    return sorted(events, key=lambda e:(e['received_ns'],e['request_id'],e['kind']))


def _cost(row, cost_provider):
    # Field projection is deliberate: labels/outputs never reach a cost model.
    value = cost_provider({k:row[k] for k in ('alias','task','pool_id')})
    if value is None:
        raise UnknownEvidence('UNKNOWN_PUBLIC_INPUT_SOLO_REFERENCE_WORK')
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise ValueError('Nonnegative finite source-bound solo reference work required')
    return value


def features(trace, visible, *, now_ns, declared_at_ns, context, cost_provider,
             scales_ns=SCALES):
    """Current visible waiting/HOL plus *all* remaining declared arrivals."""
    fields = {'request_id','alias','task','pool_id','arrival_ns'}
    if (any(set(r) != fields or type(r['arrival_ns']) is not int or r['arrival_ns'] < 0 for r in trace)
            or len({r['request_id'] for r in trace}) != len(trace)):
        raise ValueError('Strict complete public arrival/input registry required')
    registry = {r['request_id']:r for r in trace}
    starts, ready = set(), set()
    seen = set()
    for event in visible:
        if (set(event) != {'request_id','alias','kind','received_ns'}
                or event['request_id'] not in registry
                or event['alias'] != registry[event['request_id']]['alias']
                or type(event['received_ns']) is not int or not 0 <= event['received_ns'] <= now_ns
                or event['kind'] not in ('START','READY')):
            raise ValueError('Only cutoff-visible native facts accepted')
        key = event['request_id'],event['kind']
        if key in seen:
            raise ValueError('Duplicate visible lifecycle fact')
        seen.add(key)
        (starts if event['kind']=='START' else ready).add(event['request_id'])
    if not ready <= starts:
        raise ValueError('Visible READY requires visible START')
    waiting = [r for r in trace if r['arrival_ns'] <= now_ns and r['request_id'] not in starts]
    active_owners = {registry[r]['alias'] for r in starts-ready}
    age = lambda r:(now_ns-r['arrival_ns'])/1e6
    future = sorted((r for r in trace if r['arrival_ns'] > now_ns), key=lambda r:r['arrival_ns'])
    # Compute maxima over the complete public suffix, not only the next burst.
    envelopes = []
    for scale in scales_ns:
        costs = [_cost(r,cost_provider) for r in future]
        best = 0.;right = 0;work = 0.
        for left,row in enumerate(future):
            while right < len(future) and future[right]['arrival_ns'] < row['arrival_ns']+scale:
                work += costs[right];right += 1
            best = max(best,work);work -= costs[left]
        envelopes.append(dict(window_ns=scale,max_offered_solo_work_ms=max(0.,best)))
    return build_features(now_ns=now_ns, execution_context_sha256=context,
        queue_work_ms=sum(_cost(r,cost_provider) for r in waiting),
        oldest_waiting_ms=max(map(age,waiting),default=0.),
        hol_blocked_ms=max((age(r) for r in waiting if r['alias'] in active_owners),default=0.),
        waiting_requests=len(waiting),visible_state_time_ns=now_ns,
        declared_at_ns=declared_at_ns,future_source='declared_schedule',
        future_work_envelopes=envelopes)


def offline_outcome(trace, records, *, now_ns, observation_end_ns, risk_protocol):
    """Offline only. Missing/censored future labels are never negative examples."""
    _ns(now_ns);_ns(observation_end_ns)
    horizon = risk_protocol['target']['horizon_ns'];end = now_ns+horizon
    rows = {r['request_id']:r for r in records}
    if len(rows) != len(records) or set(rows) != {r['request_id'] for r in trace}:
        raise ValueError('Full original offered cohort required for offline labels')
    if observation_end_ns < end:
        raise UnknownEvidence('CENSORED_RISK_FUTURE_HORIZON')
    intervals = []
    violation = False
    for public in trace:
        row = rows[public['request_id']]
        starts = [e for e in row.get('execution_events',[]) if 'event' not in e]
        if len(starts) != 1:
            raise UnknownEvidence('CENSORED_RISK_START_LABEL')
        arrival = _ns(public['arrival_ns']);start = _ns(starts[0]['perf_ns'])
        if not arrival <= start <= observation_end_ns:
            raise ValueError('Invalid offline arrival-to-worker START interval')
        if start > arrival:
            intervals.append((arrival,start))
        if now_ns <= arrival < end:
            receipt = row.get('response_reader_event_enqueue_time_ns')
            if receipt is None or receipt > observation_end_ns:
                raise UnknownEvidence('CENSORED_RISK_LATENCY_LABEL')
            _ns(receipt)
            if receipt < start:
                raise ValueError('Offline response precedes START')
            violation |= receipt-arrival > risk_protocol['latency_reference_ms'][public['task']]*1e6
    # Merge waiting intervals. Shared endpoints are processed atomically: an
    # arrival exactly when another STARTs can preserve a nonempty queue.
    merged = []
    for begin,finish in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],finish)
        else:
            merged.append([begin,finish])
    recovery = 0.
    for begin,finish in merged:
        if begin <= now_ns < finish:
            recovery = (finish-now_ns)/1e6;break
    return dict(window_start_ns=now_ns,window_end_ns=end,
        observed_until_ns=observation_end_ns,violation=bool(violation),recovery_ms=recovery)


def build_rows(plan, raw, cost_provider):
    """Training adapter; strict capture review must qualify raw before calling."""
    if (raw.get('split') != 'training' or raw.get('training_only') is not True
            or raw.get('status') != 'completed' or raw.get('kind') != 'measured'
            or raw.get('execution_context_sha256') != plan['execution_context_sha256']):
        raise ValueError('Completed exact-context measured training capture required')
    risk_protocol = protocol(plan);start = raw['phase_start_ns'];trace = public_trace(plan,start)
    rows, omitted = [], []
    for offset in plan['cutoff_offsets_ns']:
        now = start+offset
        try:
            feature = features(trace,visible_projection(raw['request_records'],now),now_ns=now,
                declared_at_ns=start,context=raw['execution_context_sha256'],cost_provider=cost_provider)
            outcome = offline_outcome(trace,raw['request_records'],now_ns=now,
                observation_end_ns=raw['observation_end_ns'],risk_protocol=risk_protocol)
        except UnknownEvidence as exc:
            omitted.append(dict(cutoff_offset_ns=offset,reason=str(exc)));continue
        rows.append(dict(id=raw['run_id']+':'+str(offset),run_id=raw['run_id'],
            group_id=raw['group_id'],split='training',kind='measured',
            execution_context_sha256=raw['execution_context_sha256'],decision_time_ns=now,
            features=feature,outcome=outcome))
    return dict(protocol=risk_protocol,rows=rows,omitted=omitted,
        scope='TRAINING_ONLY_OPTIONAL_RISK_ROWS_NOT_ACCURACY_OR_STABILITY_EVIDENCE')


def reviewed_rows(review_reference, cost_provider):
    """Recheck capture sources, supervision, full cohort and original metrics.

    The provider must reconstruct its public-input training reference costs
    from its own frozen sources. Costs are feature coordinates, not a proof
    of capacity/performance transfer to this collector context.
    """
    from .dag_busy_support_v9 import _reviewed
    plan,raw,fixture = _reviewed(review_reference,False)
    if fixture:
        raise ValueError('Measured risk rows require actual native training')
    result = build_rows(plan,raw,cost_provider)
    result['review'] = copy.deepcopy(review_reference)
    result['cost_provider_provenance'] = cost_provider.provenance()
    return result


class ReconstructedReferenceWork:
    """Frozen public feature coordinates, reconstructed from original training.

    These activity sums are useful covariates. Their use never establishes a
    new collector's physical service rate, SM throughput or tail accuracy.
    """
    def __init__(self, bundle_reference, plan_reference, solo_reference_configuration):
        from .multi_native_composite_v5 import checked
        from .multi_native_composite_v5 import MultiNativeProvider
        from .native_queue_busy_replay_v9 import queue_context_bridge
        from .component_generation_support_v7 import ComponentGenerationSupport
        from .solo_reference_work_v8 import SoloOrdinalReferenceWork
        bundle = checked(bundle_reference);plan = checked(plan_reference)
        if plan.get('split') != 'training' or plan.get('training_only') is not True:
            raise ValueError('Training public inputs required for reference work reconstruction')
        bridge = queue_context_bridge(bundle,plan['context'])
        composite = checked(bundle['composite'])
        configuration = checked(solo_reference_configuration)
        if configuration.get('schema') != 'gsdeploy.two_solo_public_feature_configuration.v9':
            raise ValueError('Explicit source-bound solo reference configuration required')
        adapter = MultiNativeProvider(composite,bundle['forecast_defaults']['aliases'],
            host_scenario=bundle['host_supply_scenario'])
        generation = ComponentGenerationSupport(checked(bundle['generation']['component_support']),adapter,
            allow_exact_content_generation_hypothesis=True)
        generation_support = generation.check(public_trace(plan,0))
        specs = {c['spec']['id']:c['spec'] for c in composite['components']}
        if set(configuration['components']) != set(specs):
            raise ValueError('Every original training component requires two solo captures')
        self.values = {};diagnostic = {}
        for cid,refs in configuration['components'].items():
            if refs['dag'] != specs[cid]['dag']:
                raise ValueError('Solo reference DAG must equal the original component DAG')
            work = SoloOrdinalReferenceWork(refs['dag'],refs['solo_reviews'])
            for row in work.groups:
                key = json.dumps([row['alias'],row['task'],row['pool_id']],separators=(',',':'))
                if key in self.values:raise ValueError('Duplicate exact solo reference cell')
                self.values[key] = row['reference_ns_sum']/1e6
            diagnostic[cid] = work.provenance()
        self.evidence = dict(bundle=copy.deepcopy(bundle_reference),plan=copy.deepcopy(plan_reference),
            solo_reference_configuration=copy.deepcopy(solo_reference_configuration),
            bridge=bridge,provider_diagnostic=diagnostic,generation_support=generation_support,
            purpose='FIXED_PUBLIC_TRAINING_REFERENCE_FEATURE_COORDINATES',
            coordinate='sum of ordinal median GPU kernel lifetimes from two real isolated captures; excludes host/copy/lifecycle',
            co_run_lifetimes_used=False,physical_capacity_transfer_claimed=False,active_gpu_residual_included=False)
        self.binding = identity(dict(values=self.values,evidence=self.evidence))

    def _verify(self):
        if self.binding != identity(dict(values=self.values,evidence=self.evidence)):
            raise ValueError('Frozen public reference work features changed')

    def __call__(self,row):
        self._verify()
        if set(row) != {'alias','task','pool_id'}:
            raise ValueError('Public input identity only')
        key = json.dumps([row['alias'],row['task'],row['pool_id']],separators=(',',':'))
        if key not in self.values:
            raise UnknownEvidence('UNKNOWN_EXACT_PUBLIC_REFERENCE_WORK')
        return self.values[key]

    def provenance(self):
        self._verify()
        return dict(model_sha256=self.binding,**copy.deepcopy(self.evidence))
