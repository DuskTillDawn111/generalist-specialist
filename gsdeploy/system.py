"""One public workflow: validate inputs, optimize, predict, execute, evaluate."""

from .config import SystemConfig
from .optimizer import UtilityOptimizer
from .prediction import predict_scenarios


class DeploymentSystem:
    def __init__(self, config: SystemConfig):
        self.config = config

    def plan(self, deployment=None, route_grid=None):
        c = self.config
        grid = c.route_grid if route_grid is None else route_grid
        if type(grid) is not int or grid < 1:
            raise ValueError("route_grid must be a positive integer")
        if deployment is not None:
            for model, count in deployment.items():
                if (
                    type(count) is not int
                    or count <= 0
                    or model not in c.catalog["models"]
                    or str(count) not in c.catalog["capacity_factors"][model]
                ):
                    raise ValueError(f"Invalid fixed deployment: {model}/{count}")
        solver = UtilityOptimizer(
            c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights
        )
        result = solver.solve(deployment=deployment, K=grid)
        if result["winner"]:
            result["constraints"] = c.validate_plan(
                result["winner"]["deployment"], result["winner"]["routing"]
            )
        return result

    def predict(self, deployment, routing):
        c = self.config
        c.validate_plan(deployment, routing)
        return predict_scenarios(
            c.catalog,
            deployment,
            routing,
            c.workload,
            c.gpu_memory_mib,
            c.latency_reference_ms,
            c.weights,
        )

    def run(self, plan, samples, **options):
        from .runtime.runner import run

        winner = plan.get("winner")
        if winner is None:
            raise ValueError("Cannot execute a plan without a winner")
        self.config.validate_plan(winner["deployment"], winner["routing"])
        return run(self.config, winner, samples, **options)
