"""LP primary ranking and lexicographic runtime comparison, no weighted score."""
from typing import Protocol
import math

from .runtime import STABILITY


class PerformanceCoarseScorer(Protocol):
    def predict(self, deployment, workload, routing) -> float | None:
        """Optional predicted mean latency; lower is better. None = unavailable."""
        ...


def optional_prediction(scorer, deployment, workload, routing):
    if scorer is None:
        return None, 'unavailable'
    try:
        value = scorer.predict(dict(deployment), workload, dict(routing))
        if value is None or not math.isfinite(value) or value < 0:
            return None, 'unavailable/nonfinite'
        return float(value), 'available'
    except Exception as exc:
        # An optional predictor cannot make a feasible search fail or reject a state.
        return None, f'unavailable: {type(exc).__name__}: {exc}'


def rank_candidates(candidates):
    """Exact delta_z primary; scorer only orders exact ties where both have predictions.

    Missing-prediction slots retain their deterministic model order. This avoids
    interpreting unavailable predictions as either good or bad predictions.
    """
    ordered = sorted(candidates, key=lambda c: (-c.delta_z, c.mutation.model))
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end].delta_z == ordered[start].delta_z:
            end += 1
        slots = [i for i in range(start, end) if ordered[i].prediction is not None]
        predicted = sorted((ordered[i] for i in slots), key=lambda c: (c.prediction, c.mutation.model))
        for i, candidate in zip(slots, predicted):
            ordered[i] = candidate
        start = end
    return ordered


class RuntimeObjective:
    def __init__(self, mean_tolerance_ms=1.):
        if not math.isfinite(mean_tolerance_ms) or mean_tolerance_ms < 0:
            raise ValueError('Invalid mean tolerance')
        self.mean_tolerance_ms = mean_tolerance_ms

    def compare(self, candidate, current, candidate_memory, current_memory):
        """Return (strictly better, reason). Equal non-steady states do not improve."""
        a, b = STABILITY[candidate.stability_state_v2], STABILITY[current.stability_state_v2]
        if a != b:
            return a > b, 'stability_improved' if a > b else 'stability_worse'
        if candidate.stability_state_v2 != 'steady':
            return False, 'same_nonsteady_stability'
        delta = candidate.weighted_mean_e2e - current.weighted_mean_e2e
        if delta != 0 and abs(delta) >= self.mean_tolerance_ms:
            return delta < 0, 'mean_improved' if delta < 0 else 'mean_worse'
        if candidate.weighted_p95_e2e != current.weighted_p95_e2e:
            better = candidate.weighted_p95_e2e < current.weighted_p95_e2e
            return better, 'p95_improved' if better else 'p95_worse'
        if candidate_memory != current_memory:
            better = candidate_memory < current_memory
            return better, 'memory_improved' if better else 'memory_worse'
        return False, 'objective_tie'
