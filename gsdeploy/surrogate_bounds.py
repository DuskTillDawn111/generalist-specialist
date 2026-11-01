"""Separable mathematical bounds for explicitly validated frozen predictors.

These are not physical GPU bounds. Unknown versions, failed validation and
changed measurement contexts receive no transferred service or energy bound.
"""

import math

from .calibration import _baseline


class SurrogateBounds:
    def __init__(self, data, spec, services, horizon, reference, amortization,
                 minimum_services=None, pressure_services=None):
        self.data, self.spec, self.services = data, spec, services
        self.minimum_services, self.pressure_services = minimum_services, pressure_services
        self.reference = reference
        self.amortization = amortization or horizon
        self.memory_multiplier = spec["memory_multiplier"]
        self.intercept, self.busy, self.worker = spec["power_coefficients"]
        self.input_services = None
        if minimum_services is not None:
            from .calibration_v2 import input_service_ms

            # A path is usable only when its whole declared input pool was
            # validated by replay_path_statistics. Missing paths remain zero
            # relaxations; never remove them from the solver's candidate domain.
            self.input_services = {}
            for model, task, count in minimum_services:
                for sample in spec["input_order"][task]:
                    self.input_services[model, task, count, sample] = input_service_ms(
                        spec, model, task, count, sample)

    def family_service(self, model, task, count):
        if self.minimum_services is not None:
            return self.minimum_services.get((model, task, count), 0.0)
        return self.services.get((model, task, count), 0.0) * self.spec["service_multiplier"]

    def pressure_service(self, model, task, count):
        if self.pressure_services is not None:
            return self.pressure_services.get((model, task, count), 0.0)
        return self.family_service(model, task, count)

    def task_input_ids(self, trace, seed):
        """Route-independent per-task sample sequence, using the frozen replay."""
        if self.input_services is None:
            return None
        from .calibration_v2 import replay_input_ids

        result = {task: [] for task in self.spec["input_order"]}
        for (_, task), sample in zip(trace, replay_input_ids(self.data, trace, seed)):
            result[task].append(sample)
        return result

    def request_service(self, model, task, count, sample):
        """Per-worker ms for this exact input; zero only for an unknown path."""
        if self.input_services is None:
            return self.family_service(model, task, count)
        return self.input_services.get((model, task, count, sample), 0.0)

    def state_energy(self, count):
        return count * (self.worker + self.spec["startup_joules_per_worker"]
                        / self.amortization) / self.reference

    def route_energy(self, task, deployment, row):
        return self.busy * math.fsum(
            q * self.services.get((m, task, deployment[m]), 0.0) / 1000
            for m, q in row.items() if q > 0
        ) / self.reference

    @property
    def constant_energy(self):
        return self.intercept / self.reference


def surrogate_bounds(data, rates, *, horizon=120, energy_reference_watts=300,
                     startup_amortization_seconds=None, **context):
    """Return bounds only after the common calibrated-predictor scope gate.

    Known path/count states are reusable without repeated identity hashes.
    Missing states contribute zero service/busy cost only to the relaxation;
    the complete predictor still reports their performance as unknown.
    """
    mode = data.get("performance_mode")
    if mode not in {"calibrated_surrogate", "calibrated_replay_v2", "replay_transfer_v1"}:
        return None

    try:
        if mode == "calibrated_surrogate":
            from .calibration import validated_spec
            spec = validated_spec(data, rates)
        elif mode == "calibrated_replay_v2":
            from .calibration_v2 import validated_replay_spec, replay_path_statistics
            spec = validated_replay_spec(data, rates)
        else:
            from .replay_transfer import validated_transfer_spec
            from .calibration_v2 import replay_path_statistics
            spec = validated_transfer_spec(data, rates)
        if data.get("memory_mode") != mode:
            return None
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0
                   for v in (horizon, energy_reference_watts,
                             startup_amortization_seconds or horizon)):
            return None
        services = {}
        minima, pressure = ({}, {}) if mode in {"calibrated_replay_v2", "replay_transfer_v1"} else (None, None)
        for m, model in data["models"].items():
            for t in rates:
                if t not in model["supported_tasks"]:
                    continue
                for n in data["capacity_factors"][m]:
                    if int(n) <= 0:
                        continue
                    try:
                        services[m, t, int(n)] = _baseline(data, m, t, int(n))
                    except (KeyError, TypeError, ValueError):
                        pass
                    if minima is not None:
                        try:
                            stats = replay_path_statistics(spec, m, t, int(n))
                            low, mean = stats["minimum_ms"], stats["mean_ms"]
                            if not (all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                        for v in (low, mean)) and low <= mean):
                                continue
                            minima[m, t, int(n)], pressure[m, t, int(n)] = low, mean
                        except (KeyError, TypeError, ValueError):
                            pass
        return SurrogateBounds(data, spec, services, horizon,
                               energy_reference_watts, startup_amortization_seconds,
                               minima, pressure)
    except (ImportError, KeyError, TypeError, ValueError):
        return None
