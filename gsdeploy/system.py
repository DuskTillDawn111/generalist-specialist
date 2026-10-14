"""One public workflow: validate inputs, optimize, predict, execute, evaluate."""

from .config import SystemConfig
from dataclasses import replace
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
        c = replace(c, route_grid=grid)
        solver = UtilityOptimizer(
            c.catalog,
            c.workload,
            c.gpu_memory_mib,
            c.latency_reference_ms,
            c.weights,
            search_max_states=c.search_max_states,
            search_policy=c.search_policy,
            **self.context(),
        )
        result = solver.solve(deployment=deployment, K=grid)
        if result["winner"]:
            result["constraints"] = c.validate_plan(
                result["winner"]["deployment"], result["winner"]["routing"]
            )
        result["problem"] = c.as_dict()
        return result

    def context(self):
        c = self.config
        return dict(
            horizon=c.horizon_seconds,
            seed=c.random_seed,
            arrival_process=c.arrival_process,
            drain_seconds=c.drain_seconds,
            energy_reference_watts=c.energy_reference_watts,
            startup_amortization_seconds=c.startup_amortization_seconds,
        )

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
            **self.context(),
        )

    def run(self, plan, samples, **options):
        from .runtime.runner import run

        winner = plan.get("winner")
        if winner is None:
            raise ValueError("Cannot execute a plan without a winner")
        self.config.validate_plan(winner["deployment"], winner["routing"])
        changes = {}
        for option, field in {
            "duration": "horizon_seconds",
            "seed": "random_seed",
            "arrival_process": "arrival_process",
            "drain_timeout": "drain_seconds",
        }.items():
            value = options.pop(option, None)
            if value is not None:
                changes[field] = value
        source_problem = plan.get("problem")
        inherited = {
            field: source_problem[field]
            for field in (
                "horizon_seconds",
                "random_seed",
                "arrival_process",
                "drain_seconds",
                "route_grid",
                "search_policy",
                "search_max_states",
            )
            if source_problem and field in source_problem
        }
        effective = replace(self.config, **{**inherited, **changes})
        matched = DeploymentSystem(effective).predict(
            winner["deployment"], winner["routing"]
        )
        same_problem = source_problem == effective.as_dict()
        raw = run(
            effective,
            winner,
            samples,
            duration=effective.horizon_seconds,
            seed=effective.random_seed,
            arrival_process=effective.arrival_process,
            drain_timeout=effective.drain_seconds,
            **options,
        )
        raw["planning_contract"] = {
            "same_problem_as_search": same_problem,
            "optimality_transferred": same_problem
            and plan.get("optimality_certified", False),
            "matched_prediction": matched,
            "reason": (
                "SAME_PROBLEM"
                if same_problem
                else "RUN_OPTIONS_CHANGED_PROBLEM_REPREDICTED"
            ),
        }
        return raw
