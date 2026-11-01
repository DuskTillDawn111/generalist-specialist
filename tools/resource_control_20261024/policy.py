"""Six-term policy adapter with resource-specific evidence separation.

Future workload scenarios and execution-consistent event replay are used by
default. Resource control and kernel profiling still require hardware evidence;
no physical benefit is asserted by this CPU adapter.
"""

from __future__ import annotations

import copy

from gsdeploy.resource_study import objective_contract, validate_packet
from tools.queue_coupled_utility_20261023.policy import CoupledUtilityPolicy
from tools.queue_coupled_utility_20261023.model import UnknownEvidence
from .forecast_policy import FullHorizonPolicy


class ResourceBoundProvider:
    """An unbound or wrong-resource predictor stays UNKNOWN, never scaled."""

    def __init__(self, provider, execution_context_sha256):
        self.provider = provider
        self.context = execution_context_sha256

    def _call(self, method, *args):
        if self.provider is None or getattr(self.provider, "execution_context_sha256", None) != self.context:
            return {"known": False, "context_supported": False, "tail_supported": False,
                    "service_ms": None, "remaining_ms": None, "watts": None,
                    "reason": "RESOURCE_EXECUTION_CONTEXT_NOT_BOUND"}
        function = getattr(self.provider, method, None)
        if function is None:
            return {"known": False, "context_supported": False, "tail_supported": False,
                    "peak_memory_mib": None, "reason": "RESOURCE_PROVIDER_METHOD_UNSUPPORTED:" + method}
        return function(*args)

    def __getattr__(self, method):
        # Presence of plan selects segment replay. Legacy providers retain
        # their explicit whole-handler proxy; no kernel phases are fabricated.
        if method != "plan" or self.provider is None or not hasattr(self.provider, "plan"):
            raise AttributeError(method)
        def plan(*args):
            if getattr(self.provider, "execution_context_sha256", None) != self.context:
                raise UnknownEvidence("RESOURCE_EXECUTION_CONTEXT_NOT_BOUND")
            return self.provider.plan(*args)
        return plan

    def service(self, *args):
        return self._call("service", *args)

    def residual(self, *args):
        return self._call("residual", *args)

    def power(self, *args):
        return self._call("power", *args)

    def memory(self, *args):
        return self._call("memory", *args)


class Dispatcher:
    """Maintain fixed background duties and paired target spill/no-spill policy."""

    def __init__(self, packet, job_id, *, service_provider=None, power_provider=None, quality_loss=None,
                 memory_provider=None, future_workload=None, forecast_options=None,
                 legacy_observed_prefix=False):
        check = validate_packet(packet)
        if not check["valid"]:
            raise ValueError(check["issues"])
        jobs = [job for job in packet["jobs"] if job["id"] == job_id]
        if len(jobs) != 1:
            raise ValueError("Unique job identity required")
        self.job = copy.deepcopy(jobs[0])
        self.system = packet["effective_system"]
        objective = objective_contract(packet["objective"])
        if self.job["weights"] != objective["weights"]:
            raise ValueError("Policy weights differ from six-term objective")
        policy = self.job["candidate"]["routing_policy"]
        self.target = policy["target_task"]
        self.aliases = {a: {"family": m, "supported_tasks": self.system["catalog"]["models"][m]["supported_tasks"]}
                        for a, m in packet["aliases"].items()}
        context = self.job["execution_context_sha256"]
        from gsdeploy.routing import Router
        from gsdeploy.trace import expand_routes
        _, routes = expand_routes(self.job["nominal_no_spill_plan"]["deployment"],
                                   self.job["nominal_no_spill_plan"]["routing"])
        self.router = Router(routes)
        forecast_kwargs = {}
        if not legacy_observed_prefix:
            from pathlib import Path
            import json
            from gsdeploy.forecasting import FutureWorkload
            from gsdeploy.runtime.workload import read_jsonl
            if future_workload is None:
                design_path = Path(packet["design"]["path"])
                samples_path = Path(json.loads(design_path.read_text())["samples"])
                if not samples_path.is_absolute():
                    samples_path = design_path.parent / samples_path
                pools = {task: [] for task in self.system["workload"]}
                for row in read_jsonl(samples_path):
                    task = row["task_type"]
                    if task in pools and row["pool_id"] not in pools[task]:
                        pools[task].append(row["pool_id"])
                future_workload = FutureWorkload(self.system["workload"], pools,
                    process=self.system["arrival_process"], namespace="forecast:" + packet["scope_sha256"])
            forecast_kwargs = dict(future_workload=future_workload,
                memory_budget_mib=self.system["gpu_memory_mib"], resources=self.job["candidate"]["resource_configuration"],
                memory_safe_factor=self.system["catalog"]["safe_factor"],
                drain_seconds=self.system["drain_seconds"],
                memory_provider=ResourceBoundProvider(memory_provider if memory_provider is not None else service_provider, context),
                continuation_factory=lambda: copy.deepcopy(self.router).choose,
                **(forecast_options or {}))
        elif future_workload is not None or forecast_options is not None:
            raise ValueError("Legacy observed-prefix mode cannot accept future forecast settings")
        policy_class = CoupledUtilityPolicy if legacy_observed_prefix else FullHorizonPolicy
        self.policy = policy_class(
            self.aliases, ResourceBoundProvider(service_provider, context), quality_loss or {},
            objective["weights"], self.system["latency_reference_ms"], self.system["workload"],
            power_provider=ResourceBoundProvider(power_provider if power_provider is not None else service_provider, context),
            allow_spill=policy["allow_spill"], energy_reference_watts=self.system["energy_reference_watts"],
            **policy["parameters"], **forecast_kwargs)
        if legacy_observed_prefix and self.job["candidate"]["resource_configuration"]["mode"] == "request_admission":
            raise ValueError("Observed-prefix model cannot simulate host admission")
        self.arrival = None

    def start_phase(self, start_ns, deadline_ns):
        self.policy.start_phase(start_ns, deadline_ns)

    def set_arrival(self, request_id, scheduled_ns):
        if self.arrival is not None:
            raise ValueError("Previous arrival has not been dispatched")
        self.arrival = {"request_id": request_id, "arrival_ns": scheduled_ns}
        self.policy.set_arrival(request_id, scheduled_ns)

    def observe_completed(self, records, now_ns):
        if any(r["completion_time_ns"] > now_ns for r in records):
            raise ValueError("Future completion feedback")
        self.policy.observe_completed(records)

    def choose(self, task, pool_id, now_ns, snapshot):
        if self.arrival is None or self.arrival["arrival_ns"] > now_ns:
            raise ValueError("A visible current arrival is required")
        if task == self.target:
            result = self.policy.choose(task, pool_id, now_ns, snapshot)
        else:
            alias = self.router.choose(task)
            self.policy.current = None
            result = {"alias": alias, "task": task, "status": "FROZEN_SWRR_BACKGROUND",
                      "actual_spill_selected": False, "pool_id": pool_id, "now_ns": now_ns}
        self.policy.observe_admission({**self.arrival, "alias": result["alias"], "task": task, "pool_id": pool_id})
        self.arrival = None
        result.update(execution_context_sha256=self.job["execution_context_sha256"],
                      resource_configuration_sha256=identity_resources(self.job),
                      physical_six_term_improvement_certified=False)
        return result


def identity_resources(job):
    from gsdeploy.resources import identity
    return identity(job["candidate"]["resource_configuration"])
