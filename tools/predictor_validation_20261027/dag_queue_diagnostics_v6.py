"""Incremental queue explanations for the actual CUDA DAG forecast path.

Activity work sums fitted isolated GPU/copy/memset durations. It is neither
request wall time nor physical GPU demand. Resource-weighted work stays unknown
without an explicitly supplied normalized-demand model. Nothing here changes
routes, execution rates, utility, admission, or the original stability rule.
"""
from __future__ import annotations

from collections import Counter
import heapq
import math

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.queue_diagnostics import workload_envelopes

ACTIVITY = ('gpu', 'copy', 'memset')
RESOURCES = ('sm', 'dram', 'l2')


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative ' + name + ' required')
    return value


class DagQueueDiagnostics:
    """One profile scan, then O(active nodes + resident owners) per event.

    ``duration`` can also be installed as the engine's duration provider, so
    diagnostic accounting and execution use exactly the same cached numbers.
    Resource demands, if available, must be supplied separately as a mapping
    from each GPU node to normalized sm/dram/l2 demand; absence is not zero.
    """

    def __init__(self, aliases, plans, duration_provider, *, now_ns=0,
                 resource_demand_provider=None):
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError('Integer initial diagnostic time required')
        self.aliases, self.plans = aliases, plans
        self.duration_provider = duration_provider
        self.resource_demand_provider = resource_demand_provider
        self.now_ns = self.initial_ns = now_ns
        self.cache, self.costs, self.registered = {}, {}, set()
        self.waiting, self.oldest = set(), []
        self.remaining = Counter({kind: 0. for kind in ACTIVITY})
        self.remaining_resources = dict.fromkeys(RESOURCES, 0.)
        self.max_resources = dict.fromkeys(RESOURCES, 0.)
        self.max_work = self.max_waiting = self.max_age = 0.
        self.max_by_kind = dict.fromkeys(sorted(ACTIVITY), 0.)
        self.max_waiting_by_owner = dict.fromkeys(aliases, 0)
        self.fifo_head_blocked_ns = dict.fromkeys(aliases, 0)
        self.fifo_request_wait_ns = dict.fromkeys(aliases, 0)
        self.blocking_causes = {a: Counter() for a in aliases}
        self.episode_start = None
        self.episodes, self.unknown = [], set()

    @staticmethod
    def _key(row):
        return row['alias'], row['task'], row['pool_id']

    def _profile(self, row):
        key = self._key(row)
        if key in self.cache:
            return self.cache[key]
        if (row['alias'] not in self.aliases
                or row['task'] not in self.aliases[row['alias']]['supported_tasks']
                or key not in self.plans):
            raise UnknownEvidence('UNKNOWN_DAG_QUEUE_WORK_PROFILE')
        nodes = self.plans[key]['nodes']
        if len({n['id'] for n in nodes}) != len(nodes):
            raise ValueError('Unique DAG node identities required')
        durations, kinds, resources = {}, {}, {}
        work = Counter({kind: 0. for kind in ACTIVITY})
        total_resources = dict.fromkeys(RESOURCES, 0.)
        for node in nodes:
            if node['kind'] not in ACTIVITY:
                continue
            value = _nonnegative(self.duration_provider(node, row), 'fitted activity duration')
            if value == 0:
                raise UnknownEvidence('POSITIVE_DAG_QUEUE_WORK_DURATION_REQUIRED')
            durations[node['id']] = value
            kinds[node['id']] = node['kind']
            work[node['kind']] += value
            if node['kind'] != 'gpu':
                continue
            demand = (self.resource_demand_provider(node, row)
                      if self.resource_demand_provider is not None else None)
            if demand is not None and set(demand) != set(RESOURCES):
                raise ValueError('Exact normalized SM/DRAM/L2 demand schema required')
            resources[node['id']] = {}
            for name in RESOURCES:
                fraction = demand[name] if demand is not None else None
                if fraction is None:
                    total_resources[name] = None
                else:
                    _nonnegative(fraction, 'normalized resource demand')
                    if fraction > 1:
                        raise ValueError('Normalized resource demand exceeds one')
                    if total_resources[name] is not None:
                        total_resources[name] += value * fraction
                resources[node['id']][name] = fraction
        result = dict(durations=durations, kinds=kinds, resources=resources,
                      full_work=work, full_resources=total_resources)
        self.cache[key] = result
        return result

    def duration(self, node, row):
        profile = self._profile(row)
        if profile['kinds'].get(node['id']) != node['kind']:
            raise ValueError('Activity node differs from bound queue profile')
        return profile['durations'][node['id']]

    def register(self, row):
        rid = row['request_id']
        if rid in self.registered:
            raise ValueError('Duplicate DAG queue request registration')
        if type(row['arrival_ns']) is not int or row['arrival_ns'] < self.now_ns:
            raise ValueError('DAG queue arrival precedes diagnostic start')
        profile = self._profile(row)
        self.registered.add(rid)
        self.waiting.add(rid)
        heapq.heappush(self.oldest, (row['arrival_ns'], rid))
        self.remaining.update(profile['full_work'])
        self.costs[rid] = sum(profile['full_work'].values()) / 1e6
        for name, value in profile['full_resources'].items():
            if value is None:
                self.remaining_resources[name] = None
                self.unknown.add('NORMALIZED_RESOURCE_DEMAND_UNAVAILABLE:' + name)
            elif self.remaining_resources[name] is not None:
                self.remaining_resources[name] += value

    def mark_started(self, row):
        rid = row['request_id']
        if rid not in self.waiting:
            raise ValueError('DAG START must consume a registered waiting request')
        self.waiting.remove(rid)

    def observe(self, clock, delta, pending, running, active, rates):
        """Observe the interval after arrivals/starts, before work consumption."""
        if (type(clock) is not int or type(delta) is not int
                or clock < self.now_ns or delta < 0):
            raise ValueError('Monotone integer DAG queue interval required')
        count = sum(len(rows) for rows in pending.values())
        if count != len(self.waiting):
            raise ValueError('Actual owner FIFO and diagnostic waiting set differ')
        self.max_waiting = max(self.max_waiting, count)
        while self.oldest and self.oldest[0][1] not in self.waiting:
            heapq.heappop(self.oldest)
        if self.oldest:
            self.max_age = max(self.max_age, (clock + delta - self.oldest[0][0]) / 1e6)
        self.max_work = max(self.max_work, sum(self.remaining.values()) / 1e6)
        for kind in ACTIVITY:
            self.max_by_kind[kind] = max(self.max_by_kind[kind], self.remaining[kind] / 1e6)
        for name, value in self.remaining_resources.items():
            self.max_resources[name] = (None if value is None else
                max(self.max_resources[name], value / 1e6))
        if count and self.episode_start is None:
            self.episode_start = clock
        elif not count and self.episode_start is not None:
            self.episodes.append(dict(backlog_start_ns=self.episode_start,
                queue_clear_ns=clock, recovery_ms=(clock - self.episode_start) / 1e6))
            self.episode_start = None
        for owner, rows in pending.items():
            size = len(rows)
            self.max_waiting_by_owner[owner] = max(self.max_waiting_by_owner[owner], size)
            self.fifo_request_wait_ns[owner] += size * delta
            if size:
                self.fifo_head_blocked_ns[owner] += delta
                cause = 'handler_active' if owner in running else 'handoff_or_worker_not_ready'
                self.blocking_causes[owner][cause] += delta
        for (owner, node_id), node in active.items():
            if (owner, node_id) not in rates:
                raise ValueError('Every active DAG activity requires its actual simulation rate')
            rate = _nonnegative(rates[owner, node_id], 'DAG progress rate')
            consumed = min(_nonnegative(node['remaining_ns'], 'remaining activity work'), delta * rate)
            self.remaining[node['kind']] = max(0., self.remaining[node['kind']] - consumed)
            if node['kind'] == 'gpu':
                profile = self._profile(running[owner]['row'])
                for name, fraction in profile['resources'][node_id].items():
                    if fraction is not None and self.remaining_resources[name] is not None:
                        self.remaining_resources[name] = max(0., self.remaining_resources[name] - consumed * fraction)
        self.now_ns = clock + delta

    def result(self, cohort):
        # The whole offered cohort is retained. A shortened simulation cannot
        # turn unregistered future work into a falsely complete envelope.
        costs = {rid: self.costs.get(rid) for rid in cohort}
        return dict(max_waiting_requests=int(self.max_waiting),
            registered_request_count=len(self.registered),
            used_template_count=len(self.cache),
            registered_request_activity_node_instances=sum(
                len(self.cache[self._key(cohort[rid])]['durations']) for rid in self.registered),
            unique_cached_activity_nodes=sum(len(p['durations']) for p in self.cache.values()),
            max_oldest_waiting_age_ms=self.max_age,
            max_remaining_activity_work_ms=self.max_work,
            max_remaining_activity_work_by_kind_ms=self.max_by_kind.copy(),
            remaining_activity_work_ms=sum(self.remaining.values()) / 1e6,
            max_normalized_resource_work_ms=self.max_resources.copy(),
            max_waiting_by_owner=self.max_waiting_by_owner.copy(),
            owner_fifo_head_blocked_ms={a: n / 1e6 for a, n in self.fifo_head_blocked_ns.items()},
            owner_fifo_request_wait_ms={a: n / 1e6 for a, n in self.fifo_request_wait_ns.items()},
            owner_fifo_blocking_causes_ms={a: {k: n / 1e6 for k, n in counts.items()}
                for a, counts in self.blocking_causes.items()},
            global_admission_hol_applicable=False,
            queue_recovery_episodes=list(self.episodes),
            unrecovered_backlog_since_ns=self.episode_start,
            initial_backlog_left_censored=bool(
                self.episode_start == self.initial_ns or self.episodes
                and self.episodes[0]['backlog_start_ns'] == self.initial_ns),
            offered_work_envelopes=workload_envelopes(list(cohort.values()), costs),
            unknown_reasons=sorted(self.unknown),
            work_basis='Sum of fitted isolated GPU/copy/memset activity work; excludes unidentified host work; not request wall time or measured resource capacity',
            fifo_scope='Fixed-route owner FIFO, including handler execution and worker handoff; no inferred global admission cap or bypass action',
            resource_demand_inferred_from_occupancy=False,
            visible_state_correction_applied=False,
            physical_accuracy_validated=False, stability_rule_changed=False,
            six_term_utility_changed=False, calibrated_long_term_risk=False)
