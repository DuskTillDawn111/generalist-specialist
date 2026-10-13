"""Explicit planning-only feature projection and training-fold preprocessing."""
import math
import numpy as np
from deployment_core.workload import TASKS


def planning_vector(row, extractor, level, resource_keys):
    """Recompute from planning inputs/offline profiles, never measured row fields."""
    f = extractor.extract(row['deployment'], row['workload'], row['routing'])
    model, task = row['target_model'], row['task']
    m = f['per_model'][model]
    s0 = extractor.catalog[model].service_time_ms.get(task)
    if s0 is None or not math.isfinite(s0) or s0 <= 0:
        raise ValueError(f'Missing/invalid standalone service time: {task}/{model}')
    v = {'u': m['standalone_service_demand']}
    v.update({'q/' + k: m['task_demand_mix'][k] for k in TASKS})
    if level == 'C':
        profiles = extractor.resource_profiles.get((task, model), {}).get('values', {})
        v.update({'C/' + k: m['colocation_pressure'].get(k, {}).get('value') for k in resource_keys})
        v.update({'offline/' + k: profiles.get(k) for k in resource_keys})
        v['path/standalone_service_ms'] = s0
        v.update({'model/' + k: m[k] for k in ('instances', 'model_memory_mib', 'resident_memory_mib')})
        v.update({'deployment/' + k: f['deployment_features'][k]
                  for k in ('total_model_instances', 'total_resident_memory_mib')})
    elif level != 'B':
        raise ValueError(level)
    return v, float(s0), f


class FoldPreprocessor:
    """Train-only medians, missing indicators, constant removal, and scaling.

    All-missing training dimensions are dropped (never invented or fit on test).
    Missing indicators are available for every retained raw dimension. If an
    indicator is constant in training it is dropped and recorded, like any other
    constant feature. Held-out missingness is additionally reported in the audit.
    """
    def fit(self, vectors, names):
        self.names = list(names)
        a = self._array(vectors)
        self.available = np.isfinite(a).any(axis=0)
        self.medians = np.array([np.nanmedian(a[:, j]) for j in np.where(self.available)[0]])
        z = self._impute(a)
        self.keep = np.ptp(z, axis=0) > 1e-12
        self.means = z[:, self.keep].mean(axis=0)
        self.scales = z[:, self.keep].std(axis=0)
        self.scales[self.scales == 0] = 1.
        self.train_min = np.min(z, axis=0)
        self.train_max = np.max(z, axis=0)
        return self

    def _array(self, vectors):
        a = np.array([[v.get(k) if v.get(k) is not None else np.nan for k in self.names] for v in vectors], dtype=float)
        if np.isinf(a).any():
            raise ValueError('Infinite planning feature')
        return a

    def _impute(self, a):
        a = a[:, self.available]
        missing = ~np.isfinite(a)
        return np.column_stack([np.where(missing, self.medians, a), missing.astype(float)])

    def transform(self, vectors):
        z = self._impute(self._array(vectors))
        result = (z[:, self.keep] - self.means) / self.scales
        # An intercept-only fit remains possible in a constant synthetic fold.
        return result if result.shape[1] else np.zeros((len(vectors), 1))

    def audit(self, test_vectors):
        names = [k for k, keep in zip(self.names, self.available) if keep]
        expanded = names + ['missing/' + k for k in names]
        raw = self._array(test_vectors)
        z = self._impute(raw)
        return dict(raw_feature_order=self.names,
                    all_missing_training_features=[k for k, keep in zip(self.names, self.available) if not keep],
                    training_medians=dict(zip(names, self.medians.tolist())),
                    dropped_constant_features=[k for k, keep in zip(expanded, self.keep) if not keep],
                    transformed_feature_order=[k for k, keep in zip(expanded, self.keep) if keep],
                    training_means=self.means.tolist(), training_scales=self.scales.tolist(),
                    training_min=dict(zip(expanded, self.train_min.tolist())),
                    training_max=dict(zip(expanded, self.train_max.tolist())),
                    heldout_missing_values=int((~np.isfinite(raw)).sum()),
                    heldout_outside_training_range_counts={k: int(((z[:, j] < self.train_min[j] - 1e-10) |
                        (z[:, j] > self.train_max[j] + 1e-10)).sum()) for j, k in enumerate(expanded)})
