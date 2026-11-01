"""Full future-cohort rollout adapter for the existing causal routing policy.

Only public workload distributions feed future scenarios. Each current-route
branch shares the same suffix and continuation policy. This is an explicit
one-action rollout (replanned at real arrivals), not an oracle that recursively
executes future actual policy decisions or proves physical optimality.
"""

from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel, score_forecast
from tools.queue_coupled_utility_20261023.policy import CoupledUtilityPolicy


class FullHorizonPolicy(CoupledUtilityPolicy):
    def __init__(self, *args, future_workload, memory_budget_mib, resources,
                 arrival_scenarios=3, drain_seconds=60., memory_provider=None,
                 continuation_factory=None, max_forecast_events=100000,
                 max_forecast_requests=20000, memory_safe_factor=1., lifecycle_provider=None,
                 batch_configurations=None, **kwargs):
        if type(arrival_scenarios) is not int or arrival_scenarios < 1:
            raise ValueError("Positive integer arrival scenario count required")
        from gsdeploy.forecasting.workload import number
        number(drain_seconds, "bounded drain seconds", positive=True)
        number(memory_budget_mib, "memory budget", positive=True)
        number(memory_safe_factor, "memory safety factor", positive=True)
        if memory_safe_factor < 1:
            raise ValueError("Memory safety factor must be >=1")
        if kwargs.get("memory_delta_penalty_upper_norm") is not None:
            raise ValueError("Full-horizon memory is scored directly; remove old proxy penalty")
        super().__init__(*args, **kwargs)
        if self.workload != future_workload.rates:
            raise ValueError("Future and scoring offered rates differ")
        self.future_workload = future_workload
        self.memory_budget_mib, self.drain_seconds = memory_budget_mib, drain_seconds
        self.memory_safe_factor = memory_safe_factor
        self.continuation_factory = continuation_factory
        self.model = FullHorizonModel(self.aliases, self.model.service_provider, self.model.power_provider,
                                     memory_provider=memory_provider, resources=resources,
                                     lifecycle_provider=lifecycle_provider,
                                     batch_configurations=batch_configurations,
                                     max_events=max_forecast_events, max_requests=max_forecast_requests)
        scenarios = self.scenarios
        self._scenario_keys = {f"{service}@arrival-{i}": (service, i)
                               for service in scenarios for i in range(arrival_scenarios)}
        self.scenarios = tuple(self._scenario_keys)
        self._suffixes, self._decision = {}, None
        self.forecast_state = {}

    def _queues(self, snapshot, now):
        snapshot = copy.deepcopy(snapshot)
        state = snapshot.pop("_forecast_state", {})
        if set(state) - {"admission", "observed_peak_memory_mib", "observed_board_energy_joules",
                         "observed_last_arrivals_ns", "visible_host_state", "handoff_state"}:
            raise ValueError("Unknown forecast execution state fields")
        if "visible_host_state" in state:
            from gsdeploy.forecasting.observed_state import forecast_snapshot_from_observed, validate_handoffs_against_observed
            observed = forecast_snapshot_from_observed(state["visible_host_state"], self.aliases,
                self.model.service_provider.execution_context_sha256, now, self.future_workload.rates)
            validate_handoffs_against_observed(state.get("handoff_state", ()), state["visible_host_state"])
            for key in ("admission", "observed_last_arrivals_ns"):
                if key in state and state[key] != observed[key]:
                    raise ValueError("Forecast snapshot conflicts with visible state")
                state[key] = observed[key]
        self.forecast_state = state
        if set(snapshot) != set(self.aliases):
            raise ValueError("Full resident snapshot required")
        queues, seen = {}, set(self.completed)
        for alias, worker in snapshot.items():
            rows = []
            for index, source in enumerate(worker.get("outstanding", [])):
                rid = source["request_id"]
                arrival = source.get("arrival_ns", self.admitted.get(rid, {}).get("arrival_ns", source.get("enqueued_ns")))
                if rid in seen or type(arrival) is not int or arrival > now:
                    raise ValueError("Duplicate or invalid visible pending arrival")
                seen.add(rid)
                row = {"request_id": rid, "task": source["task"], "pool_id": source["pool_id"],
                       "alias": alias, "arrival_ns": arrival, "observed_pending": True}
                if "forecast_cursor" in source:
                    if index != 0:
                        raise ValueError("Only an executing FIFO head may have segment progress")
                    row["forecast_cursor"] = copy.deepcopy(source["forecast_cursor"])
                for key in ("batch_ready_ns", "batch_ready_evidence"):
                    if key in source:
                        row[key] = source[key]
                if index == 0:
                    start = source.get("handler_start_ns", source.get("actual_submit_ns"))
                    if start is not None:
                        if type(start) is not int or not arrival <= start <= now:
                            raise ValueError("Invalid visible start/submit time")
                        row["head_age_ms"] = (now - start) / 1e6
                    elif source.get("status") == "queued" or "forecast_cursor" in source:
                        row["head_age_ms"] = 0.
                    elif arrival < now:
                        # Do not silently restart an old, unobserved head.
                        row["head_age_ms"] = (now - arrival) / 1e6
                rows.append(row)
            queues[alias] = rows
        return queues

    def _forecast(self, queues, now_ns, deadline_ns, scenario):
        if not self.phase_start_ns <= now_ns <= deadline_ns:
            raise UnknownEvidence("OUTSIDE_MEASUREMENT_ARRIVAL_WINDOW")
        service, index = self._scenario_keys[scenario]
        key = (self._decision, now_ns, deadline_ns, index)
        if key not in self._suffixes:
            self._suffixes[key] = self.future_workload.suffix(
                self.phase_start_ns, now_ns, deadline_ns, scenario_index=index, decision_id=self._decision,
                observed_last_arrivals_ns=self.forecast_state.get("observed_last_arrivals_ns")
                if self.future_workload.process == "gamma_renewal" else None)
        suffix = self._suffixes[key]
        if self.continuation_factory is None:
            def route(row, pending, now):
                choices = [a for a, spec in self.aliases.items() if row["task"] in spec["supported_tasks"]]
                specialists = [a for a in choices if self.aliases[a]["family"] == row["task"]]
                return min(specialists or choices, key=lambda a: (len(pending[a]), a))
        else:
            route = self.continuation_factory()
        forecast = self.model.forecast(
            queues, now_ns, deadline_ns, service, future_arrivals=suffix["arrivals"], route_future=route,
            admission_state=self.forecast_state.get("admission"),
            handoff_state=self.forecast_state.get("handoff_state", ()),
            drain_deadline_ns=deadline_ns + round(self.drain_seconds * 1e9), trace_request_id=self._decision)
        forecast.update(future_scenario=suffix, forecast_now_ns=now_ns,
                        continuation_scope="declared continuation rollout; future decisions are replanned online")
        return forecast

    def _terms(self, forecast, current_rows):
        expected_current = {r["request_id"] for rows in current_rows.values() for r in rows}
        expected_current.update(h["request_id"] for h in self.forecast_state.get("handoff_state", ())
                                if h["parent_receipt_ns"] is None)
        expected_future = {r["request_id"] for r in forecast["future_scenario"]["arrivals"]}
        if set(forecast["cohort"]) != expected_current | expected_future:
            raise ValueError("Full future candidate cohort mismatch")
        now = forecast["forecast_now_ns"]
        metric = score_forecast(
            forecast, self.completed.values(), self.aliases, self.workload, self.refs, self.quality_loss,
            phase_start_ns=self.phase_start_ns, now_ns=now, deadline_ns=self.deadline_ns,
            memory_budget_mib=self.memory_budget_mib, energy_reference_watts=self.energy_reference_watts,
            memory_safe_factor=self.memory_safe_factor,
            weights=self.weights,
            observed_peak_memory_mib=self.forecast_state.get("observed_peak_memory_mib"),
            observed_board_energy_joules=self.forecast_state.get("observed_board_energy_joules"))
        if not metric["six_term_comparison_complete"]:
            raise UnknownEvidence("INCOMPLETE_FULL_HORIZON_SCORE:" + ",".join(metric["unknown_reasons"]))
        if metric["memory_budget_satisfied"] is not True:
            raise UnknownEvidence("PREDICTED_MEMORY_BUDGET_EXCEEDED")
        metric.update(future_cohort_sha256=forecast["future_scenario"]["cohort_sha256"],
                      future_arrival_count=forecast["future_arrival_count"],
                      arrival_source=forecast["future_scenario"]["source"],
                      simulated_transition_count=forecast["transition_count"],
                      competition_trace_scope=forecast["trace_scope"],
                      continuation_scope=forecast["continuation_scope"])
        metric["queue_diagnostics"] = forecast["queue_diagnostics"]
        metric["lifecycle_scope"] = forecast["lifecycle_scope"]
        return metric

    def _own_service(self, forecast, current, alias, transition):
        return forecast["finishes"][current["request_id"]]["predicted_remaining_service_ms"]

    def choose(self, task, pool_id, now_ns, pending_snapshot):
        self._decision = self.current["request_id"] if self.current else f"coupled-current:{self.sequence + 1}"
        self._suffixes = {}
        result = super().choose(task, pool_id, now_ns, pending_snapshot)
        selected = next((r for r in result["audit"] if r["alias"] == result["alias"] and r["score"] is not None), None)
        result.update(
            trigger="FULL_FUTURE_ROLLOUT_UTILITY_SPILL" if result["actual_spill_selected"] else result["trigger"],
            conditional_proxy_terms="Q/T/L/P/M and remaining board E over identical full future scenarios",
            unknown_terms=["Physical counterfactual truth", "Formal forecast error bounds",
                           "Runtime decision overhead unless separately profiled",
                           "Absolute past/startup energy when not supplied"],
            objective_scope="Six-term full future-cohort rollout; empirical per-task P95; calibrated contexts required",
            memory_comparison_scope="Catalog safety factor times maximum of observed historical and predicted owned-worker peaks; no zero-fill",
            answers_used=False,
            future_arrival_scenarios_used=True,
            actual_future_trace_used=False,
            declared_future_plan_used=self.future_workload.process == "declared_schedule",
            arrival_scenario_count=len({index for _, index in self._scenario_keys.values()}),
            confidence_bounds_calibrated=False,
            continuation_scope="same declared continuation in each branch; one-action rollout, not recursive future-policy oracle")
        if selected:
            result.update(
                predicted_U_by_scenario={s: m["full_U"] for s, m in selected["scenarios"].items()},
                predicted_M_norm_by_scenario={s: m["terms"]["M"] for s, m in selected["scenarios"].items()},
                six_term_comparison_complete=True)
        else:
            result["six_term_comparison_complete"] = False
        result.pop("answers_or_future_arrivals_used", None)
        return result
