"""Complete-candidate adapters; no new trained predictor or duplicated FIFO."""

import copy
import math
from pathlib import Path

from ..energy import energy_key
from ..performance import profile_key
from ..prediction import predict_scenarios
from ..trace import expand_routes, trace_digest
from ..routing import Router
from .contracts import digest
from .. import normalization as norm


def global_cost_floor(oracle, problem):
    c = problem.config
    if norm.is_v2(c.catalog):
        floor = c.weights['Q']*norm.quality_floor(c.catalog, c.workload)
        return floor if math.isfinite(floor) and norm.ready(c.catalog) else None
    return oracle.cost_lower_bound


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ["normalization.py", "prediction.py", "performance.py", "trace.py", "routing.py", "quality.py",
             "energy.py", "calibration.py", "calibration_v2.py", "replay_transfer.py",
             "config.py", "coupled_mode.py", "surrogate_bounds.py", "milp.py",
             "search_session.py", "paper_forecast.py", "forecast_errors.py",
             "resources.py", "paper_training.py", "linear_solver.py", "paper_algorithms.py"]
    sources = {name: (root / name).read_text() for name in names}
    # The stage/FIFO mode is part of the oracle identity, including its
    # simulator, lifecycle and input/context contracts. Old service files
    # alone cannot identify the newly connected predictor implementation.
    sources.update({str(p.relative_to(root)): p.read_text()
                    for p in (root / "forecasting").rglob("*.py")})
    sources["continuous_adapter"] = Path(__file__).read_text()
    sources["continuous_contracts"] = Path(__file__).with_name("contracts.py").read_text()
    sources["continuous_bounds"] = Path(__file__).with_name("safe_bounds.py").read_text()
    for name in ("immutable.py", "validated_bounds.py", "paper_bounds.py", "candidate_bounds.py", "column_cache.py"):
        sources["continuous_" + name] = Path(__file__).with_name(name).read_text()
    return digest(sources)


class LegacyPredictorOracle:
    """Use the existing whole candidate scoring contract without modification.

    This adapter does not promote legacy proxies into measured GPU evidence.
    A missing resource, unsupported replay condition or placeholder stays unknown.
    """

    evidence_kind = "existing_predictor"
    supports_stress = True
    cost_lower_bound = 0.0  # Original U is negative nonnegative weighted losses.

    def __init__(self):
        self.version = source_identity()

    def check(self):
        if source_identity() != self.version:
            raise ValueError("ORACLE_SOURCE_CHANGED")

    def catalog_for(self, problem, deployment, routing, trace):
        return problem.config.catalog

    def state_costs(self, problem):
        c = problem.config
        d = c.catalog
        if norm.is_v2(d):
            return {}  # Legacy state floors omit potentially negative Q.
        if (d.get("performance_mode", "independent") == "independent"
                and d.get("memory_mode", "standalone_proxy") == "standalone_proxy"):
            return {(m, n): c.weights["M"] * d["safe_factor"]
                    * d["memory_table_mib"][m][str(n)] / c.gpu_memory_mib
                    for m, ns in problem.states.items() for n in ns}
        return {}

    def memory_states(self, problem):
        costs = self.state_costs(problem)
        if not costs:
            return {}
        c = problem.config
        return {(m, n): c.catalog["safe_factor"] * c.catalog["memory_table_mib"][m][str(n)]
                for m, ns in problem.states.items() for n in ns}

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        budget.consume("predictions", deadline)
        budget.counts["predictor_leaves"] += 3
        c = problem.config
        data = self.catalog_for(problem, deployment, routing, trace)
        return predict_scenarios(data, deployment, routing, c.workload,
            c.gpu_memory_mib, c.latency_reference_ms, c.weights,
            horizon=c.horizon_seconds, seed=trace.seed, trace=trace.arrivals,
            arrival_process=c.arrival_process, drain_seconds=c.drain_seconds,
            energy_reference_watts=c.energy_reference_watts,
            startup_amortization_seconds=c.startup_amortization_seconds)


class SyntheticColocationOracle(LegacyPredictorOracle):
    """Explicit CPU fixture profiles, evaluated by the original predictor.

    Parameters describe invented whole-residency slowdown and board power.
    No output from this class is physical evidence or native qualification.
    The existing FIFO, quality, rho, six losses and stress remain unchanged.
    """

    evidence_kind = "synthetic_fixture"

    def __init__(self, interference=0.08, load_interference=0.06,
                 idle_watts=15.0, worker_watts=3.0, busy_watts=45.0):
        super().__init__()
        self.parameters = dict(interference=interference,
            load_interference=load_interference, idle_watts=idle_watts,
            worker_watts=worker_watts, busy_watts=busy_watts)
        if any(not math.isfinite(v) or v < 0 for v in self.parameters.values()):
            raise ValueError("INVALID_SYNTHETIC_PARAMETERS")
        self._source = self.version
        self.version = digest(dict(source=self._source, fixture=self.parameters))

    def check(self):
        if digest(dict(source=source_identity(), fixture=self.parameters)) != self.version:
            raise ValueError("ORACLE_SOURCE_OR_FIXTURE_CHANGED")

    def state_costs(self, problem):
        # No standalone bound is transferred to this nonseparable fixture.
        return {}

    def memory_states(self, problem):
        return {}

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        result = super().predict(problem, deployment, routing, trace, budget, deadline)
        result["stress_evidence_complete"] = True
        return result

    def resource_values(self, problem, deployment, routing):
        c, p = problem.config, self.parameters
        data = c.catalog
        workers = sum(deployment.values())
        pressure = math.fsum(q * data["models"][m]["service_time_ms"][t] / 1000
            for t, row in routing.items() for m, q in row.items())
        slowdown = 1 + p["interference"] * (workers - 1) + p["load_interference"] * pressure
        service = {m: {t: s * slowdown for t, s in data["models"][m]["service_time_ms"].items()}
                   for m in deployment}
        memory = data["safe_factor"] * math.fsum(
            data["memory_table_mib"][m][str(n)] for m, n in deployment.items())
        return workers, pressure, slowdown, service, memory

    def catalog_for(self, problem, deployment, routing, trace):
        c, p = problem.config, self.parameters
        data = copy.deepcopy(c.catalog)
        workers, pressure, slowdown, service, memory = self.resource_values(problem, deployment, routing)
        data.update(performance_mode="profiled", memory_mode="profiled",
                    colocation_profiles={}, energy_profiles={})
        key = profile_key(data, deployment, routing, c.workload,
                          c.horizon_seconds, trace.seed, c.arrival_process)
        data["colocation_profiles"][key] = dict(kind="synthetic_fixture",
            scope="whole_gpu_colocation", coverage_valid=True,
            service_time_ms=service, safe_memory_mib=memory,
            provenance=dict(fixture="continuous_cpu_v1", parameters=p,
                            physical_evidence=False))
        for multiplier in (1.0, 1.25, 2.0):
            key = energy_key(deployment, routing, c.workload, c.horizon_seconds,
                trace.seed, c.arrival_process, multiplier, catalog=data)
            watts = p["idle_watts"] + p["worker_watts"] * workers + p["busy_watts"] * pressure * slowdown * multiplier
            data["energy_profiles"][key] = dict(kind="synthetic_fixture",
                scope="whole_gpu_board", coverage_valid=True,
                run_joules=watts * c.horizon_seconds,
                startup_joules=2.0 * workers,
                provenance="Invented CPU fixture, not measured power")
        return data


def assignment_digest(deployment, routing, trace):
    aliases, rows = expand_routes(deployment, routing)
    router = Router(rows)
    return digest([(i, t, router.choose(t)) for i, (_, t) in enumerate(trace.arrivals)])


def normalize_prediction(prediction, policy):
    """Three-state normalization: missing evidence cannot prove infeasibility."""
    p = prediction
    if policy == "robust":
        valid = p.get("stress_constraints_valid")
        value = p.get("scenario_U_low")
    else:
        valid, value = p.get("constraints_eligible"), p.get("U")
    unknown = (p.get("performance_unknown") or p.get("memory_unknown")
               or p.get("evidence_unknown") or p.get("constraint_unknown")
               or p.get("reason") == "MISSING_TASK_OBSERVATIONS")
    if policy == "robust" and p.get("stress_supported") is False:
        unknown = True
    if (policy == "robust" and value is None and p.get("constraints_eligible")
            and not p.get("stress_evidence_complete")):
        # The old aggregate API does not expose every stress failure reason.
        # Missing stress support must not become a proof of physical failure.
        unknown = True
    if unknown or valid is None:
        return "UNKNOWN", None
    if valid is False:
        return "INFEASIBLE", None
    if value is None or type(value) not in (float, int) or not math.isfinite(value):
        return "UNKNOWN", None
    return "FEASIBLE", float(value)
