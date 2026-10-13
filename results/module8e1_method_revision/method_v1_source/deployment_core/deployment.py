"""Integer count deployments; tuple-name construction retained as a binary boundary."""

from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class Deployment:
    counts: dict[str, int]

    def __init__(self, models):
        from .replicas import normalize_counts
        if isinstance(models, dict):
            counts = normalize_counts(models)
        else:
            names = tuple(models)
            if (not names or any(not isinstance(name, str) or not name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError('Binary deployment must contain distinct model names')
            counts = normalize_counts({m: 1 for m in names})
        object.__setattr__(self, 'counts', counts)

    @property
    def models(self):
        """Legacy presence view; use counts for resource or capacity arithmetic."""
        return tuple(m for m, n in self.counts.items() if n)

    def __hash__(self):
        # Preserve legacy set/dict use, with replica counts included in identity.
        return hash(tuple(self.counts.items()))


@dataclass
class RoutingResult:
    # Contains all active, supported, deployed pairs, including zero-rate pairs.
    # All absent pairs are defined to have rate zero.
    routing_rps: dict[tuple[str, str], float]
    utilization: dict[str, float]


@dataclass
class FeasibleDeployment:
    deployment: Deployment
    memory_mib: float
    routing: RoutingResult
