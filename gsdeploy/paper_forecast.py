"""Connect ordered co-resident profiles and event FIFO to the common evaluator.

This is an explicit experimental prediction mode. It reuses FullHorizonModel,
ProfileProvider and LifecycleProvider; it never turns a missing co-run context
into a standalone-service multiplier. Solver eligibility describes the declared
mathematical model, and does not grant native accuracy or deployment approval.
"""

from collections import OrderedDict
import copy
from functools import lru_cache
import hashlib
import json
import math
import random
import weakref

import numpy as np

from .forecast_errors import UnknownEvidence
from .forecasting.profile_factory import execution_profile_provider
from .forecasting.lifecycle import LifecycleProvider
from .forecasting.simulator import FullHorizonModel
from .routing import Router
from .trace import expand_routes, make_trace, trace_digest
from .search_session import freeze
from . import normalization as norm

MODE = "paper_profiled_fifo_v1"
SCHEMA = "gsdeploy.paper_profiled_fifo.v1"
_snapshots = weakref.WeakValueDictionary()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _positive(value, name, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or not zero and value == 0):
        raise ValueError("Invalid " + name)


def _measurement_owner_ordinals(engine, entry, aliases, routes, rates, seed, arrival_process):
    """Advance the post-prime descriptor through the causal warmup cohort.

    Runtime creates a fresh router for each window. Inputs, output token counts
    and actual warmup completion times do not determine these owner counts.
    This does not validate transferring the training priming history.
    """
    counts = dict.fromkeys(aliases, 0)
    if getattr(engine.service_provider, 'native_order_law', None) is None:
        return counts
    protocol = entry.get('episode_protocol')
    if (not isinstance(protocol, dict) or protocol.get('arrival_process') != arrival_process
            or protocol.get('priming_policy') != 'largest_three_per_task'):
        raise UnknownEvidence('NATIVE_ORDER_REQUIRES_MATCHED_POST_PRIME_WARMUP_PROTOCOL')
    warmup = protocol.get('warmup_seconds')
    if type(warmup) not in (int, float) or not math.isfinite(warmup) or warmup < 0:
        raise UnknownEvidence('NATIVE_ORDER_REQUIRES_EXPLICIT_WARMUP_DURATION')
    router = Router(routes)
    for _, task in make_trace(rates, warmup, seed, arrival_process):
        counts[router.choose(task)] += 1
    return counts


def validate_spec(data):
    """Validate actual executable mechanisms, without asserting their accuracy."""
    if data.get("memory_mode") != MODE:
        raise ValueError("Paper FIFO requires its explicit whole-context memory mode")
    spec = data.get(MODE, {})
    required = {"schema", "allow_cpu_fixture", "input_order", "scenarios", "deployments"}
    if (set(spec) - required - {"max_events", "max_requests", "cache_size", "cache_max_bytes",
                              "fifo_cache_size", "fifo_cache_max_bytes"}
            or required - set(spec) or spec["schema"] != SCHEMA
            or type(spec["allow_cpu_fixture"]) is not bool):
        raise ValueError("Invalid paper FIFO profile binding")
    if not isinstance(spec["scenarios"], dict) or spec["scenarios"].get("1.0") is None:
        raise ValueError("An explicit nominal profile scenario is required")
    if set(spec["scenarios"]) - {"1.0", "1.25", "2.0"} or any(
            not isinstance(v, str) or not v for v in spec["scenarios"].values()):
        raise ValueError("Named profile scenarios required; no implicit stress scaling")
    for task, order in spec["input_order"].items():
        if (not isinstance(order, list) or not order or len(set(order)) != len(order)
                or any(not isinstance(s, str) or not s for s in order)):
            raise ValueError("Invalid public input order: " + task)
    quality = data.get("quality", {})
    if quality.get("mode") == "per_sample_replay" and any(
            quality.get("input_order", {}).get(t) != order
            for t, order in spec["input_order"].items()):
        raise ValueError("FIFO and per-sample quality must use the same input order")
    for name, default in (("max_events", 100000), ("max_requests", 20000),
                          ("cache_size", 128), ("cache_max_bytes", 16 * 1024 * 1024)):
        if type(spec.get(name, default)) is not int or spec.get(name, default) < 1:
            raise ValueError("Positive integer " + name + " required")
    for name,default in (("fifo_cache_size",128),("fifo_cache_max_bytes",16*1024*1024)):
        if type(spec.get(name,default)) is not int or spec.get(name,default)<0:
            raise ValueError('Nonnegative integer '+name+' required')
    if not isinstance(spec["deployments"], list) or not spec["deployments"]:
        raise ValueError("Explicit supported deployment profiles required")
    seen = set()
    for entry in spec["deployments"]:
        fields = {"deployment", "profiles", "lifecycle", "capacity_service_ms",
                  "capacity_evidence", "startup_joules"}
        if set(entry) - fields - {"resources", "startup_peak_memory_mib", "startup_energy_scope", "episode_protocol"} or fields - set(entry):
            raise ValueError("Explicit execution, lifecycle, capacity and startup evidence required")
        x = entry["deployment"]
        if not x or any(m not in data["models"] or type(n) is not int or n <= 0
                        or str(n) not in data["capacity_factors"][m] for m, n in x.items()):
            raise ValueError("Invalid supported deployment")
        key = canonical(x)
        if key in seen:
            raise ValueError("One execution configuration per deployment; use a separate experiment for another")
        seen.add(key)
        if not entry["capacity_evidence"]:
            raise ValueError("Original family-normalized rho/g(n) evidence required")
        for model, row in entry["capacity_service_ms"].items():
            if model not in x:
                raise ValueError("Capacity evidence for an undeployed model")
            for task, ms in row.items():
                if task not in data["models"][model]["supported_tasks"]:
                    raise ValueError("Unsupported capacity evidence task")
                _positive(ms, "capacity service")
        if entry["startup_joules"] is not None:
            _positive(entry["startup_joules"], "whole-deployment startup energy", zero=True)
        if entry.get("startup_peak_memory_mib") is not None:
            _positive(entry["startup_peak_memory_mib"], "startup raw peak memory", zero=True)
        context = entry["profiles"]["execution_context_sha256"]
        execution_profile_provider(entry["profiles"], context, allow_cpu_fixture=spec["allow_cpu_fixture"])
        LifecycleProvider(entry["lifecycle"], context, allow_cpu_fixture=spec["allow_cpu_fixture"])
        for profile in entry["profiles"]["profiles"]:
            model = profile["family"]
            if model not in x or profile["task"] not in data["models"][model]["supported_tasks"]:
                raise ValueError("Profile differs from supported deployment capabilities")
            if profile["execution"]["batch_size"] != 1:
                raise ValueError("Joint batches require a batch-aware owner contract; paper FIFO v1 is batch1")
    return spec


class PaperFifoSession:
    """Immutable inputs and bounded cache shared by all candidates in a search."""

    def __init__(self, data):
        self.data = freeze(copy.deepcopy(data))
        self.spec = validate_spec(self.data)
        self.identity = digest(self.data)
        self.entries = {}
        self.cache = OrderedDict()
        self.calls = self.hits = 0
        self.engine_calls = self.cache_bytes = 0
        self.fifo_cache = OrderedDict()
        self.fifo_cache_bytes = self.fifo_hits = 0
        for entry in self.spec["deployments"]:
            x = entry["deployment"]
            aliases = {a: {"family": m, "supported_tasks": self.data["models"][m]["supported_tasks"]}
                       for a, m in expand_routes(x, {})[0].items()}
            context = entry["profiles"]["execution_context_sha256"]
            provider = execution_profile_provider(entry["profiles"], context,
                                       allow_cpu_fixture=self.spec["allow_cpu_fixture"])
            lifecycle = LifecycleProvider(entry["lifecycle"], context,
                                           allow_cpu_fixture=self.spec["allow_cpu_fixture"])
            engine = FullHorizonModel(aliases, provider, lifecycle_provider=lifecycle,
                                     resources=(json.loads(canonical(entry["resources"]))
                                                if "resources" in entry else None),
                                     max_events=self.spec.get("max_events", 100000),
                                     max_requests=self.spec.get("max_requests", 20000))
            self.entries[canonical(x)] = entry, engine
        _snapshots[id(self.data)] = self

    def stats(self):
        return dict(event_engine_calls=self.engine_calls,uncached_scenario_queries=self.calls,
                    cache_hits=self.hits,fifo_cache_hits=self.fifo_hits,
                    score_cache_bytes=self.cache_bytes,fifo_cache_bytes=self.fifo_cache_bytes)

    def forecast_continuation(self,x,queues,visible_host_state,*,now_ns,deadline_ns,drain_deadline_ns,
                              future_arrivals=(),routing=None,handoff_state=(),scenario='mean',require_energy=True):
        """Resume this deployment's exact providers from coherent observed state.

        Past assignments remain fixed. Executing heads need an explicit current
        segment cursor; elapsed age is never converted to GPU work. This raw
        continuation has no invented whole-phase U: score_forecast requires the
        caller's completed history and observed energy/memory for that purpose.
        """
        from .forecasting.observed_state import forecast_snapshot_from_observed,validate_handoffs_against_observed
        from .resources import identity as resource_identity
        base=dict(status='UNKNOWN',U=None,native_eligible=False,independent_accuracy_qualified=False,
                  final_gain_eligible=False,scope='remaining fixed-deployment FIFO forecast; no full-phase utility without history')
        try:
            found=self.entries.get(canonical(x))
            if found is None:raise UnknownEvidence('UNSUPPORTED_DEPLOYMENT_EXECUTION_CONTEXT')
            entry,engine=found
            context=engine.service_provider.execution_context_sha256
            if getattr(engine.service_provider,'native_order_law',None) is not None:
                raise UnknownEvidence('NATIVE_ORDER_CONTINUATION_REQUIRES_VERIFIED_COMPLETE_OWNER_HISTORY')
            observed=forecast_snapshot_from_observed(visible_host_state,engine.aliases,context,now_ns,
                                                     self.spec['input_order'])
            expected_resource=resource_identity(engine.resources) if engine.resources is not None else None
            if visible_host_state.get('configuration_sha256')!=expected_resource:
                raise ValueError('Continuation observation resource identity differs from this deployment')
            validate_handoffs_against_observed(handoff_state,visible_host_state)
            gate=observed['admission']
            if engine.gated and gate is None:
                raise UnknownEvidence('OBSERVED_HOST_ADMISSION_STATE_REQUIRED')
            required=set()
            for rid,row in visible_host_state['requests'].items():
                if 'arrival_ns' not in row:
                    raise UnknownEvidence('INCOMPLETE_VISIBLE_ARRIVAL_REGISTRY')
                if (row.get('handler_end_ns') is None or row.get('worker_next_ready_ns') is None
                        or gate is not None and rid in gate['active']):
                    required.add(rid)
            current=copy.deepcopy(queues)
            listed=[r['request_id'] for rows in current.values() for r in rows]+[h['request_id'] for h in handoff_state]
            if len(listed)!=len(set(listed)) or set(listed)!=required:
                raise UnknownEvidence('INCOMPLETE_OR_FOREIGN_PENDING_COHORT')
            for alias,rows in current.items():
                if alias not in engine.aliases:raise ValueError('Unknown continuation alias')
                if any(a['arrival_ns']>b['arrival_ns'] for a,b in zip(rows,rows[1:])):
                    raise ValueError('Observed owner FIFO arrival order changed')
                for index,row in enumerate(rows):
                    fact=visible_host_state['requests'][row['request_id']]
                    if any(row.get(k)!=fact.get(k) for k in ('task','pool_id','arrival_ns')) or fact['alias']!=alias:
                        raise ValueError('Continuation queue differs from visible request identity')
                    if fact.get('handler_end_ns') is not None:
                        raise UnknownEvidence('POST_HANDLER_REQUEST_REQUIRES_HANDOFF_STATE')
                    started=fact.get('handler_start_ns')
                    if started is not None:
                        if index!=0 or 'forecast_cursor' not in row:
                            raise UnknownEvidence('CURRENT_EXECUTING_HEAD_SEGMENT_CURSOR_REQUIRED')
                        row['head_age_ms']=(now_ns-started)/1e6
                    elif 'forecast_cursor' in row or row.get('head_age_ms',0)>0:
                        raise ValueError('Unstarted request cannot have observed progress')
                    row['observed_pending']=True
            future_arrivals=tuple(future_arrivals)
            if routing is None and future_arrivals:raise ValueError('Explicit future routing required')
            for task,row in (routing or {}).items():
                if task not in self.spec['input_order']:raise ValueError('Unsupported continuation task')
                for model,q in row.items():
                    _positive(q,'continuation route rate',zero=True)
                    if q and (model not in x or task not in self.data['models'][model]['supported_tasks']):
                        raise ValueError('Continuation cannot route to an undeployed or unsupported model')
            if any(not any(q>0 for q in (routing or {}).get(r['task'],{}).values()) for r in future_arrivals):
                raise ValueError('Each future task needs a positive resident route')
            aliases,routes=expand_routes(x,routing or {})
            router=Router(routes)
            self.engine_calls+=1
            forecast=engine.forecast(current,now_ns,deadline_ns,scenario,future_arrivals=future_arrivals,
                route_future=lambda row,pending,clock:router.choose(row['task']),admission_state=gate,
                drain_deadline_ns=drain_deadline_ns,handoff_state=handoff_state,
                require_energy=require_energy,require_memory=True)
            return dict(base,status='FORECAST_COMPLETE',forecast=forecast,visible_state_sha256=observed['visible_state_sha256'],
                        execution_context_sha256=context)
        except UnknownEvidence as exc:
            return dict(base,reason=str(exc))

    def evaluate(self, x, routing, rates, budget, refs, weights, *, multiplier=1.,
                 horizon=120., seed=42, arrival_process="deterministic_periodic",
                 drain_seconds=60., energy_reference_watts=300.,
                 startup_amortization_seconds=None, trace=None, **unused):
        from .prediction import components
        from .quality import predict_quality

        if unused:
            raise TypeError("Unknown paper FIFO context: " + ",".join(unused))
        _positive(horizon, "horizon")
        _positive(drain_seconds, "drain", zero=True)
        _positive(budget, "memory budget")
        _positive(energy_reference_watts, "energy reference")
        _positive(multiplier, "scenario multiplier")
        if startup_amortization_seconds is not None:
            _positive(startup_amortization_seconds, "startup amortization")
        if not x or any(m not in self.data["models"] or type(n) is not int or n <= 0
                        or str(n) not in self.data["capacity_factors"][m] for m, n in x.items()):
            raise ValueError("Invalid deployment")
        if set(routing) != set(rates):
            raise ValueError("Routing task set does not match workload")
        for task, q in rates.items():
            _positive(q, "offered rate")
            for model, value in routing[task].items():
                _positive(value, "route rate", zero=True)
                if value and (model not in x or task not in self.data["models"][model]["supported_tasks"]):
                    raise ValueError("Unsupported executable route")
            if not math.isclose(sum(routing[task].values()), q, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("Routing does not conserve " + task)
        if norm.is_v2(self.data):
            norm.validate(self.data, rates, refs, horizon, budget)
        trace = tuple(make_trace(rates, horizon, seed, arrival_process) if trace is None else trace)
        if (not trace or any(t not in rates or not 0 <= a < horizon for a, t in trace)
                or list(trace) != sorted(trace)):
            raise ValueError("Complete ordered measurement trace required")
        key = canonical([x, routing, rates, budget, refs, weights, multiplier, horizon, seed,
                         arrival_process, drain_seconds, energy_reference_watts,
                         startup_amortization_seconds, trace])
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return copy.deepcopy(self.cache[key][0])
        self.calls += 1
        base = dict(U=None, diagnostic_partial_U=None, eligible=False, constraints_eligible=False,
                    performance_unknown=True, prediction_scope=MODE, native_eligible=False,
                    independent_accuracy_qualified=False, final_gain_eligible=False,
                    offered_requests=len(trace), trace_sha256=trace_digest(trace),
                    provider_sha256=self.identity, synthetic_inputs=self.spec["allow_cpu_fixture"])
        try:
            found = self.entries.get(canonical(x))
            if found is None:
                raise UnknownEvidence("UNSUPPORTED_DEPLOYMENT_EXECUTION_CONTEXT")
            entry, engine = found
            base["synthetic_inputs"] = not (engine.service_provider.physical_evidence
                                              and engine.lifecycle_provider.physical_evidence)
            scenario = self.spec["scenarios"].get(str(float(multiplier)))
            if scenario is None:
                raise UnknownEvidence("UNSUPPORTED_PROFILE_STRESS_SCENARIO")
            pools = {}
            for task in rates:
                if task not in self.spec["input_order"]:
                    raise UnknownEvidence("UNKNOWN_INPUT_POOL:" + task)
                pools[task] = list(self.spec["input_order"][task])
                random.Random(f"{seed}:samples:{task}").shuffle(pools[task])
            alias_models, routes = expand_routes(x, routing)
            router = Router(routes)
            counts = dict.fromkeys(rates, 0)
            rows = []
            native_order_counts=_measurement_owner_ordinals(
                engine,entry,alias_models,routes,rates,seed,arrival_process)
            initial_owner_ordinals=dict(native_order_counts)
            id_width = len(str(len(trace) - 1))
            for i, (at, task) in enumerate(trace):
                pool_id = pools[task][counts[task] % len(pools[task])]
                counts[task] += 1
                alias=router.choose(task)
                row=dict(request_id=str(i).zfill(id_width),task=task,pool_id=pool_id,
                         arrival_ns=round(at*1e9),alias=alias)
                if getattr(engine.service_provider,'native_order_law',None) is not None:
                    row['native_order']=native_order_counts[alias];native_order_counts[alias]+=1
                rows.append(row)
            queues = {a: [r for r in rows if r["alias"] == a and r["arrival_ns"] == 0]
                      for a in alias_models}
            future = [{k: v for k, v in r.items() if k != "alias"} for r in rows if r["arrival_ns"] > 0]
            assignment = {r["request_id"]: r["alias"] for r in rows}
            # Engine dynamics depend on exact assigned requests, not the
            # continuous weights that produced them. rho and quality below
            # MUST still be recomputed for each route (aggregate Q can vary).
            fifo_key=canonical([x,scenario,rows,horizon,drain_seconds,weights.get('E',0)>0])
            if fifo_key in self.fifo_cache:
                self.fifo_hits+=1
                self.fifo_cache.move_to_end(fifo_key)
                forecast=copy.deepcopy(self.fifo_cache[fifo_key][0])
            else:
                self.engine_calls += 1
                forecast = engine.forecast(queues, 0, round(horizon * 1e9), scenario,
                    future_arrivals=future,
                    route_future=lambda row, pending, now: assignment[row["request_id"]],
                    drain_deadline_ns=round((horizon + drain_seconds) * 1e9),
                    require_memory=True, require_energy=weights.get("E", 0) > 0,
                    require_episode_energy=norm.is_v2(self.data))
                size=len(fifo_key.encode())+len(canonical(forecast).encode())
                limit=self.spec.get('fifo_cache_max_bytes',16*1024*1024)
                count=self.spec.get('fifo_cache_size',128)
                if count and size<=limit:
                    self.fifo_cache[fifo_key]=copy.deepcopy(forecast),size
                    self.fifo_cache_bytes+=size
                    while len(self.fifo_cache)>count or self.fifo_cache_bytes>limit:
                        _,(_,removed)=self.fifo_cache.popitem(last=False)
                        self.fifo_cache_bytes-=removed
            rho = {}
            for m, n in x.items():
                demand = 0.
                for task, route in routing.items():
                    if route.get(m, 0) > 0:
                        ms = entry["capacity_service_ms"].get(m, {}).get(task)
                        if ms is None:
                            raise UnknownEvidence("UNKNOWN_ORIGINAL_RHO_SERVICE:" + m + ":" + task)
                        demand += route[m] * ms / 1000 * multiplier
                rho[m] = demand / self.data["capacity_factors"][m][str(n)]
            # Stress uses independently supplied phase laws for latency. The
            # original 1/1.25/2 rho stress convention is retained separately.
            mem = self.data["safe_factor"] * forecast["peak_memory_mib"]
            finishes = forecast["finishes"]
            stats = {}
            for task in rates:
                latencies = [(finishes[r["request_id"]]["finish_ns"] - r["arrival_ns"]) / 1e6
                             for r in rows if r["task"] == task and r["request_id"] in finishes]
                if len(latencies) != counts[task] or not latencies:
                    raise UnknownEvidence("DRAIN_CENSORED_OR_MISSING_TASK_LATENCY:" + task)
                stats[task] = dict(mean_ms=float(np.mean(latencies)),
                                   p95_ms=float(np.percentile(latencies, 95)), request_count=len(latencies))
            inwindow = sum(r["finish_ns"] <= round(horizon * 1e9) for r in finishes.values())
            Q, quality_evidence = (predict_quality(self.data, x, routing, rates, trace, seed)
                                   if weights.get("Q", 0) else (None, {"status": "not_required"}))
            startup = entry["startup_joules"]
            E = (None if startup is None or forecast["board_energy_joules"] is None else
                 forecast["board_energy_joules"] / (horizon * energy_reference_watts)
                 + startup / ((startup_amortization_seconds or horizon) * energy_reference_watts))
            T, M = 1-inwindow/len(rows), mem/budget
            episode = None
            if norm.is_v2(self.data):
                T = norm.deadline_loss([(r["task"],
                    (finishes[r["request_id"]]["finish_ns"]-r["arrival_ns"])/1e6, True)
                    for r in rows], norm.deadlines(self.data, refs))
                peak = entry.get("startup_peak_memory_mib")
                raw_peak = max(peak, forecast["peak_memory_mib"]) if peak is not None else None
                M = raw_peak/budget if raw_peak is not None else None
                if raw_peak is not None:
                    mem = self.data["safe_factor"]*raw_peak
                drain_j = forecast.get("through_drain_board_energy_joules")
                episode = dict(kind="synthetic_fixture" if base["synthetic_inputs"] else "predicted_episode",
                    episode_protocol=entry.get("episode_protocol"),
                    scope="whole_gpu_board", episode_coverage_valid=(startup is not None and drain_j is not None
                        and entry.get("startup_energy_scope") == "startup_load_prime_warmup"),
                    episode_joules=startup+drain_j if startup is not None and drain_j is not None else None)
                E = norm.episode_cost(self.data, episode)
            terms = components(stats, rates, refs, T, M, weights, E, Q, data=self.data)
            valid = mem <= budget + 1e-6 and max(rho.values(), default=0) <= .8 + 1e-8
            complete = terms["utility_complete"]
            result = {**base, **terms, "performance_unknown": not complete,
                      "eligible": valid and complete, "constraints_eligible": valid,
                      "reason": "OK" if valid and complete else
                                "MISSING_NORMALIZATION_OR_EPISODE_EVIDENCE" if not complete else "CONSTRAINT_FAILURE",
                      "rho": rho, "max_rho": max(rho.values(), default=0),
                      "operational_rho_08": max(rho.values(), default=0) <= .8 + 1e-8,
                      "safe_memory_mib": mem, "memory_prediction_scope": "whole_resident_context_peak",
                      "per_task": stats, "within_window_requests": inwindow,
                      "end_backlog_requests": len(rows) - inwindow, "incomplete_after_drain": 0,
                      "predicted_within_window_QPS": inwindow / horizon,
                      "route_conservation": True, "quality_capability": True,
                      "quality_evidence": quality_evidence,
                      "post_prime_owner_ordinals_at_measurement": initial_owner_ordinals,
                      "episode_energy_evidence": episode,
                      "performance_evidence": {"mode": MODE, "execution_context_sha256":
                          engine.service_provider.execution_context_sha256,
                          "capacity_evidence": entry["capacity_evidence"]},
                      "request_mapping": rows, "forecast": forecast,
                      "pooled_request_mean_ms": float(np.mean([
                          (r["finish_ns"] - r["arrival_ns"]) / 1e6 for r in finishes.values()])),
                      "pooled_request_P95_ms": float(np.percentile([
                          (r["finish_ns"] - r["arrival_ns"]) / 1e6 for r in finishes.values()], 95))}
        except UnknownEvidence as exc:
            result = {**base, "reason": str(exc)}
        size = len(key.encode()) + len(canonical(result).encode())
        byte_limit = self.spec.get("cache_max_bytes", 16 * 1024 * 1024)
        if size <= byte_limit:
            self.cache[key] = copy.deepcopy(result), size
            self.cache_bytes += size
        while len(self.cache) > self.spec.get("cache_size", 128) or self.cache_bytes > byte_limit:
            _, (_, removed_size) = self.cache.popitem(last=False)
            self.cache_bytes -= removed_size
        return result


@lru_cache(maxsize=2)
def _session(serialized):
    return PaperFifoSession(json.loads(serialized))


def session(data):
    frozen = _snapshots.get(id(data))
    if frozen is not None and frozen.data is data:
        return frozen
    # Content keys prevent a mutable catalog or changed quality/rho tables from
    # silently reusing a previous session. Only two bounded sessions are kept.
    return _session(canonical(data))


def predict(data, *args, **kwargs):
    return session(data).evaluate(*args, **kwargs)


def validate_plan(config, deployment, routing):
    from .system import DeploymentSystem
    p = predict(config.catalog, deployment, routing, config.workload, config.gpu_memory_mib,
                config.latency_reference_ms, config.weights, **DeploymentSystem(config).context())
    if not p.get("performance_unknown") and not p["constraints_eligible"]:
        raise ValueError("Plan exceeds memory or frozen rho<=0.8 constraints")
    return {"safe_memory_mib": p.get("safe_memory_mib"), "rho": p.get("rho"),
            "max_rho": p.get("max_rho"), "constraints_valid": p["constraints_eligible"],
            "memory_verified": p.get("safe_memory_mib") is not None,
            "capacity_verified": p.get("rho") is not None,
            "performance_evidence": p.get("performance_evidence", {"reason": p["reason"]}),
            "native_eligible": False}
