"""Module 3B opt-in noise gate; Module 3A defaults remain unchanged."""
import math
from .scoring import RuntimeObjective


class NoiseAwareRuntimeObjective(RuntimeObjective):
    def __init__(self, mean_tolerance_ms=1., min_relative_improvement=.05):
        super().__init__(mean_tolerance_ms)
        if not math.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement <= 1:
            raise ValueError('min_relative_improvement must be in [0,1]')
        self.min_relative_improvement = min_relative_improvement

    def compare(self, candidate, current, candidate_memory, current_memory):
        better, reason = super().compare(candidate, current, candidate_memory, current_memory)
        if better and reason in ('mean_improved', 'p95_improved'):
            baseline = current.weighted_mean_e2e
            gain = (baseline - candidate.weighted_mean_e2e) / baseline if baseline > 0 else 0.
            if gain < self.min_relative_improvement:
                return False, 'latency_gain_below_min_relative_improvement'
        return better, reason
