"""v16 execution with full-public-cohort queue explanations as a sidecar.

No work, release, phase, scheduler, route, resource, score or stability changes.
Public future input profiles supply offered-work envelopes without treating
not-yet-arrived requests as current waiting or remaining work.
"""
from __future__ import annotations

import ast
import copy
import heapq
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from . import dag_queue_diagnostics_v6, learned_availability_forecast_v16
from .dag_frozen_model_v10 import SOURCE_CACHE
from .dag_queue_diagnostics_v6 import DagQueueDiagnostics, RESOURCES
from .learned_availability_forecast_v16 import LearnedAvailabilityForecast as V16Forecast
from .native_collect import receipt

SCHEMA = 'gsdeploy.full_public_queue_diagnostics_sidecar.v19'


def implementation_sources():
    """Conservative static closure of local Python imports, including optional ones.

    Files are read, not imported or executed. External package implementation
    identity remains the upstream environment contract; this list claims only
    the local Python closure used to compose this instrumentation and v16.
    """
    root = Path(__file__).resolve().parents[2]
    pending = [Path(__file__).resolve(), Path(dag_queue_diagnostics_v6.__file__).resolve(),
               Path(learned_availability_forecast_v16.__file__).resolve()]
    found = set()

    def add(module):
        target = root.joinpath(*module)
        for candidate in (target.with_suffix('.py'), target/'__init__.py'):
            if candidate.is_file() and candidate not in found:
                pending.append(candidate)

    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        relative = path.relative_to(root)
        package = list(relative.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split('.')
                    if parts[0] in ('gsdeploy', 'tools'):
                        add(parts)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package)-node.level+1]
                    parts = base + (node.module.split('.') if node.module else [])
                else:
                    parts = node.module.split('.') if node.module else []
                if parts and parts[0] in ('gsdeploy', 'tools'):
                    add(parts)
                    for alias in node.names:
                        if alias.name != '*':
                            add(parts+[alias.name])
    return [receipt(p) for p in sorted(found)]


class FullPublicDagQueueDiagnostics(DagQueueDiagnostics):
    def __init__(self, aliases, plans, duration_provider, *, cohort, now_ns):
        super().__init__(aliases, plans, duration_provider, now_ns=now_ns)
        self.public_cohort = {}
        self.public_unknown = {}
        self.resource_unknown = set()
        self.zero_duration_state_snapshots = 0
        for original in cohort:
            row = copy.deepcopy(original)
            rid = row['request_id']
            if (rid in self.public_cohort or type(row['arrival_ns']) is not int
                    or row['arrival_ns'] < now_ns):
                raise ValueError('Complete distinct public cohort at original now required')
            self.public_cohort[rid] = row
            try:
                profile = self._profile(row)
                self.costs[rid] = sum(profile['full_work'].values())/1e6
                for name, value in profile['full_resources'].items():
                    if value is None:
                        self.resource_unknown.add(name)
            except UnknownEvidence as exc:
                self.costs[rid] = None
                self.public_unknown[rid] = str(exc)
                self.resource_unknown.update(RESOURCES)
        # No register, waiting membership, remaining work or service result is
        # created by the public prepass. Only profile/cost evidence is cached.

    def register(self, row):
        rid = row['request_id']
        if self.public_cohort.get(rid) != row:
            raise ValueError('Arrived request differs from complete declared public cohort')
        if rid not in self.public_unknown:
            return super().register(row)
        # An unsupported future template must not fail execution before the
        # original engine attempts to start it (it may still be in owner FIFO).
        if rid in self.registered or row['arrival_ns'] < self.now_ns:
            raise ValueError('Duplicate or past unknown-profile arrival')
        self.registered.add(rid)
        self.waiting.add(rid)
        heapq.heappush(self.oldest, (row['arrival_ns'], rid))

    def observe(self, clock, delta, pending, running, active, rates):
        if delta == 0:
            # v8 reports the inclusive drain boundary with no scheduler rates.
            # A zero-time snapshot consumes no work; no rate is imputed.
            self.zero_duration_state_snapshots += 1
            return super().observe(clock, delta, pending, running, {}, {})
        return super().observe(clock, delta, pending, running, active, rates)

    def result(self, cohort):
        if cohort != self.public_cohort:
            raise ValueError('Queue sidecar must retain the exact full original public cohort')
        unknown_registered = self.registered.intersection(self.public_unknown)
        # The base result counts node instances by looking up each registered
        # profile. A detached accounting view avoids inventing an empty graph
        # for unsupported inputs or masking the original engine UNKNOWN.
        view = copy.copy(self)
        view.registered = self.registered-unknown_registered
        result = DagQueueDiagnostics.result(view, cohort)
        result['registered_request_count'] = len(self.registered)
        if unknown_registered:
            result['known_registered_remaining_activity_work_ms'] = result['remaining_activity_work_ms']
            result['remaining_activity_work_ms'] = None
            result['max_remaining_activity_work_ms'] = None
            result['max_remaining_activity_work_by_kind_ms'] = dict.fromkeys(result['max_remaining_activity_work_by_kind_ms'])
            result['registered_request_activity_node_instances'] = None
        for name in self.resource_unknown:
            result['max_normalized_resource_work_ms'][name] = None
        result['unknown_reasons'] = sorted(set(result['unknown_reasons']) | {
            'NORMALIZED_RESOURCE_DEMAND_UNAVAILABLE:'+name for name in self.resource_unknown} | {
            'PUBLIC_OFFERED_PROFILE_UNKNOWN:'+rid+':'+reason
            for rid, reason in self.public_unknown.items()})
        result.update(schema=SCHEMA, full_public_request_count=len(self.public_cohort),
            future_profile_costs_precomputed=True,
            not_yet_registered_request_ids=sorted(set(cohort)-self.registered),
            unknown_public_profile_by_request=self.public_unknown.copy(),
            public_profile_prepass_registers_or_adds_backlog=False,
            future_arrival_or_service_outcomes_used=False,
            zero_duration_state_snapshots_without_rate_imputation=self.zero_duration_state_snapshots,
            offered_envelope_basis='Complete declared inputs/arrivals and frozen reference work, including future unregistered requests',
            current_backlog_basis='Registered arrivals only; future offered work is not current waiting/remaining work')
        return result


class QueueDiagnosticEngine:
    """Bind actual forecast rows/now where v16 first receives those values."""
    def __init__(self, engine, compiled, sources):
        if (engine.aliases is not compiled.graph.aliases
                or engine.plans is not compiled.graph.plans
                or engine.duration_provider is not compiled.duration
                or engine.release_provider is not compiled.release
                or engine.phase_provider is not compiled.phase
                or engine.queue_diagnostics is not None):
            raise ValueError('Unchanged v16 compiled engine without existing diagnostics required')
        self._engine, self._compiled = engine, compiled
        self.sources = [dict(ref) for ref in sources]
        self.sidecar = None

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def forecast(self, requests, now_ns, deadline_ns, *, drain_deadline_ns=None, **kwargs):
        if self.sidecar is not None or self._engine.queue_diagnostics is not None:
            raise ValueError('One bound public forecast per fresh diagnostic engine')
        for ref in self.sources:
            SOURCE_CACHE.verify(ref)
        self._compiled.assert_sources_unchanged()
        rows = copy.deepcopy(requests)
        diagnostic = FullPublicDagQueueDiagnostics(self._engine.aliases, self._engine.plans,
            self._compiled.duration, cohort=rows, now_ns=now_ns)
        self._engine.queue_diagnostics = diagnostic
        error = None
        try:
            return self._engine.forecast(requests, now_ns, deadline_ns,
                drain_deadline_ns=drain_deadline_ns, **kwargs)
        except UnknownEvidence as exc:
            error = str(exc)
            raise
        finally:
            self.sidecar = dict(schema=SCHEMA, now_ns=now_ns, deadline_ns=deadline_ns,
                drain_deadline_ns=drain_deadline_ns, full_public_cohort=rows,
                diagnostics=diagnostic.result({r['request_id']:r for r in rows}),
                original_forecast_unknown_reason=error, implementation_sources=self.sources,
                compiled_original_graph=self._compiled.diagnostic(),
                service_scheduler_routes_scores_or_rho_changed=False,
                independent_queue_accuracy_validated=False, physical_gain_validated=False,
                GPU_started=False)
            self._compiled.assert_sources_unchanged()
            for ref in self.sources:
                SOURCE_CACHE.verify(ref)


class LearnedAvailabilityForecast(V16Forecast):
    def engine(self, *, max_events):
        engine = super().engine(max_events=max_events)
        wrapped = QueueDiagnosticEngine(engine, self.compiled,
            [*implementation_sources(), *(dict(ref) for ref in self.compiled.sources)])
        self._latest_queue_engine = wrapped
        return wrapped

    def execute(self, *args, **kwargs):
        self._latest_queue_engine = None
        result = super().execute(*args, **kwargs)
        engine = self._latest_queue_engine
        if engine is None or engine.sidecar is None:
            raise ValueError('Original full public forecast did not execute diagnostic entry')
        result['queue_diagnostic_sidecar'] = engine.sidecar
        result['queue_diagnostic_instrumentation_version'] = 19
        return result
