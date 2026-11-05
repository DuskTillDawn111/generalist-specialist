"""Opt-in frozen problem and isolated digest memoization, never score caching."""

from pathlib import Path

from ..config import SystemConfig
from ..search_session import SearchSession, freeze
from .contracts import ContinuousProblem, digest
from .predictor_adapter import LegacyPredictorOracle


class ImmutableContinuousProblem(ContinuousProblem):
    """Detach inputs once; normal nested and top-level mutations are rejected.

    The ordinary mutable ContinuousProblem retains full identity checks. This
    path uses the repository's frozen JSON nodes; it does not trust mutable
    object identities or cache prediction values across different routes.
    """

    def __init__(self, config, traces=None, replica_options=None):
        super().__init__(config, traces, replica_options)
        self.config = SystemConfig(**freeze(self.config.as_dict()))
        self.states = freeze({m: list(ns) for m, ns in self.states.items()})
        self.check()
        self._sealed = True

    def __setattr__(self, key, value):
        if getattr(self, "_sealed", False):
            raise TypeError("Continuous problem inputs are immutable")
        super().__setattr__(key, value)

    def __delattr__(self, key):
        raise TypeError("Continuous problem inputs are immutable")

    def check(self):
        if not getattr(self, "_sealed", False):
            super().check()


class SessionPredictorOracle(LegacyPredictorOracle):
    """Same predictor in a private SearchSession, bound to one frozen problem."""

    def __init__(self, problem):
        if type(problem) is not ImmutableContinuousProblem:
            raise TypeError("Session oracle requires an immutable continuous problem")
        super().__init__()
        self.problem = problem
        self.session = SearchSession(Path(__file__).resolve().parents[2], problem.config.as_dict())
        self._config, self._input = self.session.config, self.session.input
        self._source = self.version
        self.version = digest(dict(source=self._source, immutable_problem=problem.identity,
                                   adapter="private_digest_session_v1"))

    def check(self):
        from .predictor_adapter import source_identity
        if self.session.closed:
            raise ValueError("ORACLE_SESSION_CLOSED")
        if self.session.config is not self._config or self.session.input is not self._input:
            raise ValueError("ORACLE_SESSION_CHANGED")
        expected = digest(dict(source=source_identity(), immutable_problem=self.problem.identity,
                               adapter="private_digest_session_v1"))
        if expected != self.version:
            raise ValueError("ORACLE_SOURCE_CHANGED")

    def predict(self, problem, deployment, routing, trace, budget, deadline=None):
        if problem is not self.problem:
            raise ValueError("ORACLE_PROBLEM_MISMATCH")
        self.check()
        budget.consume("predictions", deadline)
        budget.counts["predictor_leaves"] += 3
        c = self._config
        return self.session.module("prediction").predict_scenarios(
            c.catalog, deployment, routing, c.workload, c.gpu_memory_mib,
            c.latency_reference_ms, c.weights, horizon=c.horizon_seconds,
            seed=trace.seed, trace=trace.arrivals, arrival_process=c.arrival_process,
            drain_seconds=c.drain_seconds, energy_reference_watts=c.energy_reference_watts,
            startup_amortization_seconds=c.startup_amortization_seconds)

    def close(self):
        self.session.close()

    def engine_stats(self):
        if self._config.catalog.get('performance_mode')=='paper_profiled_fifo_v1':
            return self.session.module('paper_forecast').session(self._config.catalog).stats()
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
