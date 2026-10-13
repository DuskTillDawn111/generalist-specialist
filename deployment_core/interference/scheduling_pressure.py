"""CPU-only temporal sharing proxy. Co-path pressure, no spatial simulation."""
import json
import math
from pathlib import Path

class SchedulingPressureEstimator:
    def __init__(self, features, fit, variant=None):
        self.paths = features['paths']
        self.capacity = features['capacity_factors']
        self.fit = fit
        self.variant = variant or fit['selected_model']
        if self.variant not in ('I0', 'I1', 'I2'):
            raise ValueError('Unknown model')
        self.by_pair = {(v['model'], v['task']): v for v in self.paths.values()}

    @classmethod
    def load(cls, directory, variant=None):
        directory = Path(directory)
        return cls(json.loads((directory/'path_features.json').read_text()),
                   json.loads((directory/'fitted_model.json').read_text()), variant)

    def pressure(self, model, task):
        if (model, task) not in self.by_pair:
            raise ValueError('Unadmitted co-path')
        if self.variant == 'I0':
            return 0.0, 'NO_INTERFERENCE_SELECTED'
        if self.variant == 'I1':
            values = self.fit['delta_model_ms_per_kernel']
            return (values[model], 'CALIBRATED_CO_MODEL') if model in values else (0.0, 'UNCALIBRATED_ZERO_FALLBACK')
        values = self.fit['delta_path_ms_per_kernel']
        key = model+'__'+task
        return (values[key], 'CALIBRATED_CO_PATH') if key in values else (0.0, 'UNCALIBRATED_ZERO_FALLBACK')

    def pair(self, target, co_path):
        p = self.by_pair[tuple(target)]
        if target[0] == co_path[0]:
            raise ValueError('Same-model replicas are handled by empirical g')
        delta, confidence = self.pressure(*co_path)
        extra = p['kernel_count'] * delta
        time_ms = p['standalone_ms'] + extra
        if not math.isfinite(time_ms) or time_ms <= 0:
            raise ValueError('Nonpositive predicted service time')
        return {'effective_service_time_ms': time_ms, 'predicted_slowdown': extra/p['standalone_ms'],
                'additional_delay_ms': extra, 'pressure_ms_per_kernel': delta, 'confidence': confidence}

    def evaluate(self, deployment, routing, workload):
        """x: counts; r[k][m]: req/s; lambda[k]: req/s. Never modifies r."""
        x = deployment.counts if hasattr(deployment, 'counts') else dict(deployment)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in x.values()):
            raise ValueError('Deployment counts must be nonnegative integers')
        resident = sorted(m for m, n in x.items() if n)
        for m in resident:
            if m not in self.capacity or str(x[m]) not in self.capacity[m]:
                raise ValueError('Deployment outside frozen replica evidence')
        if any(not math.isfinite(v) or v < 0 for v in workload.values()):
            raise ValueError('Invalid arrival rate')
        for k, rates in routing.items():
            if k not in workload:
                raise ValueError('Routing task absent from workload')
            for m, rate in rates.items():
                if not math.isfinite(rate) or rate < 0 or m not in resident or (m, k) not in self.by_pair:
                    raise ValueError('Routing outside admitted deployment support')
        for k, arrival in workload.items():
            if not math.isclose(math.fsum(routing.get(k, {}).values()), arrival, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError('Routing does not conserve task arrival rate')
        demand = {m: math.fsum(rates.get(m, 0)*self.by_pair[m, k]['standalone_ms']/1000
                             for k, rates in routing.items() if (m, k) in self.by_pair) for m in resident}
        rho = {m: demand[m]/self.capacity[m][str(x[m])] for m in resident}
        mix = {m: {k: routing.get(k, {}).get(m, 0)*p['standalone_ms']/1000/demand[m] if demand[m] else 0.0
                   for (model, k), p in self.by_pair.items() if model == m} for m in resident}
        effective, slowdown, contributions, confidence = {}, {}, {}, {}
        for (m, k), p in self.by_pair.items():
            if m not in resident or k not in workload:
                continue
            per_co, flags = {}, []
            for j in resident:
                if j == m or not demand[j]:
                    continue
                pressure = 0.0
                for task, fraction in mix[j].items():
                    if not fraction:
                        continue
                    value, status = self.pressure(j, task)
                    pressure += fraction*value
                    flags.append(status)
                per_co[j] = min(rho[j], 1.0)*p['kernel_count']*pressure
            extra = math.fsum(per_co.values()); value = p['standalone_ms']+extra
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Nonpositive predicted service time')
            effective.setdefault(k, {})[m] = value
            slowdown.setdefault(k, {})[m] = extra/p['standalone_ms']
            contributions.setdefault(k, {})[m] = per_co
            confidence.setdefault(k, {})[m] = {'coefficient_support': sorted(set(flags)),
                'target_in_pair_validation': [m, k] in self.fit.get('validation_targets', []),
                'kernel_count_sample_size': len(p['kernel_count_samples']),
                'busy_probability_validated': False, 'multiple_co_model_aggregation_validated': False,
                'busy_co_models': len(per_co), 'scheduler_quantum_known': False,
                'negative_co_pressure': any(self.pressure(j, l)[0] < 0 for j in per_co for l, a in mix[j].items() if a),
                'no_hard_pruning': True}
        effective_rho = {m: math.fsum(rates.get(m, 0)*effective[k][m]/1000 for k, rates in routing.items() if m in effective.get(k, {}))/self.capacity[m][str(x[m])] for m in resident}
        return {'effective_service_time': effective, 'predicted_slowdown': slowdown,
                'task_mix': mix, 'base_utilization': rho, 'effective_utilization': effective_rho,
                'pairwise_contributions': contributions, 'confidence': confidence,
                'overload_flags': {'standalone': {m: rho[m] >= 1 for m in resident},
                                   'effective': {m: effective_rho[m] >= 1 for m in resident}},
                'SELF_INTERFERENCE_HANDLED_BY_G': True, 'selected_model': self.variant}
