"""CPU-only forecast, profile fitting and held-out assessment commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..forecast_errors import UnknownEvidence
from ..resources import identity
from . import FutureWorkload, FullHorizonModel, ProfileProvider, fit_progress, score_forecast
from .metrics import assess_prediction_pairs


def forecast_request(request, *, allow_cpu_fixture=False, allow_analytical_hypothesis=False,
                     allow_profiled_baselines=False):
    if request.get("schema") != "gsdeploy.full_horizon_forecast_request.v1":
        raise ValueError("Explicit full horizon forecast request schema required")
    aliases = request["aliases"]
    context = request["execution_context_sha256"]
    provider_kind = request.get("provider_kind", "calibrated_segment_v1")
    auxiliary = (ProfileProvider(request["profiles"], context, allow_cpu_fixture=allow_cpu_fixture)
                 if "profiles" in request else None)
    if provider_kind == "kernel_resource_v2":
        from .kernel_provider import KernelResourceProvider
        provider = KernelResourceProvider(request["kernel_resource_profiles"], context,
            allow_cpu_fixture=allow_cpu_fixture, allow_analytical_hypothesis=allow_analytical_hypothesis,
            allow_profiled_baselines=allow_profiled_baselines,
            power_provider=auxiliary, memory_provider=auxiliary)
    elif provider_kind == "calibrated_segment_v1" and auxiliary is not None:
        provider = auxiliary
    else:
        raise ValueError("Unknown or incomplete service provider")
    from .lifecycle import LifecycleProvider
    lifecycle = (LifecycleProvider(request["lifecycle_profiles"], context, allow_cpu_fixture=allow_cpu_fixture)
                 if "lifecycle_profiles" in request else None)
    contract = request.get("execution_contract", "strict_lifecycle_v2" if lifecycle is not None else "legacy_instant_handoff")
    if contract not in {"legacy_instant_handoff", "strict_lifecycle_v2"}:
        raise ValueError("Unknown lifecycle execution contract")
    if contract == "strict_lifecycle_v2" and lifecycle is None:
        raise ValueError("Strict lifecycle requires explicit lifecycle profiles")
    if contract == "legacy_instant_handoff" and lifecycle is not None:
        raise ValueError("Legacy handoff cannot consume strict lifecycle profiles")
    from ..runtime.batching import BatchConfiguration
    batch_configs = {alias: BatchConfiguration(**{**config, "length_buckets": tuple(config.get("length_buckets", ()))})
                     for alias, config in request.get("batch_configurations", {}).items()}
    if batch_configs:
        from .batch_profiles import BatchedProfileProvider
        provider = BatchedProfileProvider(provider, request["batch_input_metadata"], execution_context_sha256=context)
    workload = FutureWorkload(**request["future_workload"])
    now, deadline, start = (request[k] for k in ("now_ns", "deadline_ns", "phase_start_ns"))
    observed = None
    if "visible_host_state" in request:
        from .observed_state import forecast_snapshot_from_observed, validate_handoffs_against_observed
        observed = forecast_snapshot_from_observed(request["visible_host_state"], aliases, context, now, workload.rates)
        validate_handoffs_against_observed(request.get("handoff_state", ()), request["visible_host_state"])
        for old, new in (("admission_state", "admission"), ("observed_last_arrivals_ns", "observed_last_arrivals_ns")):
            if old in request and request[old] != observed[new]:
                raise ValueError("Explicit forecast state conflicts with visible observations")
    suffix = workload.suffix(start, now, deadline, scenario_index=request.get("arrival_scenario_index", 0),
                             observed_last_arrivals_ns=(observed["observed_last_arrivals_ns"] if observed is not None
                                                       else request.get("observed_last_arrivals_ns"))
                             if workload.process == "gamma_renewal" else None)
    from ..routing import Router
    router = Router(request["continuation_routing_rps"])
    forecast = FullHorizonModel(aliases, provider, resources=request["resources"], lifecycle_provider=lifecycle,
                                batch_configurations=batch_configs).forecast(
        request["queues"], now, deadline, request.get("service_scenario", "mean"),
        future_arrivals=suffix["arrivals"], route_future=lambda row, queues, clock: router.choose(row["task"]),
        admission_state=observed["admission"] if observed is not None else request.get("admission_state"),
        drain_deadline_ns=request["drain_deadline_ns"],
        handoff_state=request.get("handoff_state", ()),
        require_energy=request.get("require_energy", True), require_memory=request.get("require_memory", False))
    metric = score_forecast(forecast, request.get("completed", []), aliases, workload.rates,
        request["latency_reference_ms"], request["quality_loss"], phase_start_ns=start,
        now_ns=now, deadline_ns=deadline, memory_budget_mib=request["memory_budget_mib"],
        energy_reference_watts=request["energy_reference_watts"],
        memory_safe_factor=request["memory_safe_factor"],
        observed_peak_memory_mib=request.get("observed_peak_memory_mib"),
        observed_board_energy_joules=request.get("observed_board_energy_joules"))
    recovery_risk = None
    if "risk_request" in request:
        from .recovery_risk import RecoveryRiskProvider
        risk_request = request["risk_request"]
        if set(risk_request) != {"model", "features"}:
            raise ValueError("Optional risk accepts a frozen model and public features only")
        risk_provider = RecoveryRiskProvider(risk_request["model"], context, allow_cpu_fixture=allow_cpu_fixture)
        try:
            recovery_risk = risk_provider.predict(risk_request["features"], now)
        except UnknownEvidence as exc:
            recovery_risk = {"status": "UNKNOWN", "reason": str(exc), "utility_changed": False}
    return {"schema": "gsdeploy.full_horizon_forecast_result.v1", "request_sha256": identity(request),
            "GPU_started": False,
            "physical_evidence": provider.physical_evidence and (lifecycle is None or lifecycle.physical_evidence),
            "provider_kind": provider_kind, "execution_contract": contract,
            "analytical_hypothesis_enabled": allow_analytical_hypothesis,
            "profiled_baselines_enabled": allow_profiled_baselines,
            "visible_state_sha256": observed["visible_state_sha256"] if observed is not None else None,
            "physical_optimality_certified": False, "future_scenario": suffix,
            "forecast": forecast, "metrics": metric, "recovery_risk": recovery_risk}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("forecast", "fit", "assess"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "forecast":
            command.add_argument("--allow-cpu-fixture", action="store_true")
            command.add_argument("--allow-analytical-hypothesis", action="store_true")
            command.add_argument("--allow-profiled-baselines", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing forecast artifact")
    request = json.loads(args.input.read_text())
    try:
        if args.command == "forecast":
            result = forecast_request(request, allow_cpu_fixture=args.allow_cpu_fixture,
                                      allow_analytical_hypothesis=args.allow_analytical_hypothesis,
                                      allow_profiled_baselines=args.allow_profiled_baselines)
        elif args.command == "fit":
            result = fit_progress(request["profiles"], request["observations"])
        else:
            result = assess_prediction_pairs(request["pairs"])
    except UnknownEvidence as exc:
        result = {"status": "UNKNOWN", "reason": str(exc), "GPU_started": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": identity(result), "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
