"""MILP feasible seed initialization; legacy exhaustive APIs retained for validation."""

from .models import Hardware, ModelProfile, load_model_catalog
from .workload import Workload
from .deployment import Deployment, FeasibleDeployment, RoutingResult
from .milp_seeder import MilpDeploymentSeeder, SeedResult, SeedSolverError

__all__ = [
    "Hardware", "ModelProfile", "load_model_catalog", "Workload", "Deployment",
    "FeasibleDeployment", "RoutingResult", "generate_feasible_deployments",
    "screen_deployments", "MilpDeploymentSeeder", "SeedResult", "SeedSolverError",
]


def __getattr__(name):
    # Preserve legacy API for tests/baselines without importing enumeration on the seed path.
    if name in {"generate_feasible_deployments", "screen_deployments"}:
        from . import candidate_generator
        return getattr(candidate_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
