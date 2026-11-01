"""One public workflow: validate inputs, optimize, predict, execute, evaluate."""

from .config import SystemConfig
from dataclasses import replace
import copy
from .optimizer import UtilityOptimizer
from .prediction import predict_scenarios


class DeploymentSystem:
    def __init__(self, config: SystemConfig):
        self.config = config

    def plan(self, deployment=None, route_grid=None, *, exclude_deployments=None,
             exclude_plans=None, no_split=False, replica_options=None):
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
        if c.solver == "milp":
            from .milp import JointMilpOptimizer

            optimizer = JointMilpOptimizer
        else:
            optimizer = UtilityOptimizer
        solver = optimizer(
            c.catalog,
            c.workload,
            c.gpu_memory_mib,
            c.latency_reference_ms,
            c.weights,
            search_max_states=c.search_max_states,
            search_policy=c.search_policy,
            route_pair_denominators_by_task=c.route_pair_denominators_by_task,
            **(
                dict(
                    milp_max_iterations=c.milp_max_iterations,
                    milp_time_limit_seconds=c.milp_time_limit_seconds,
                    milp_max_columns=c.milp_max_columns,
                    milp_absolute_gap=c.milp_absolute_gap,
                    milp_capacity_witness=c.milp_capacity_witness,
                )
                if c.solver == "milp"
                else dict(
                    milp_capacity_witness=c.milp_capacity_witness,
                    dfs_time_limit_seconds=c.milp_time_limit_seconds,
                )
            ),
            **self.context(),
        )
        exclusions = {}
        if exclude_deployments:
            exclusions["exclude_deployments"] = exclude_deployments
        if exclude_plans:
            exclusions["exclude_plans"] = exclude_plans
        if no_split:
            exclusions["no_split"] = True
        if replica_options is not None:
            exclusions["replica_options"] = replica_options
        result = solver.solve(deployment=deployment, K=grid, **exclusions)
        result["search_restrictions"] = {
            "fixed_deployment": deployment,
            "exclude_deployments": exclude_deployments or [],
            "exclude_plans": exclude_plans or [],
            "no_split": bool(no_split),
            "replica_options": copy.deepcopy(replica_options),
        }
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
                "route_pair_denominators_by_task",
                "search_policy",
                "search_max_states",
                "solver",
                "milp_max_iterations",
                "milp_time_limit_seconds",
                "milp_max_columns",
                "milp_absolute_gap",
                "milp_capacity_witness",
            )
            if source_problem and field in source_problem
        }
        effective = replace(self.config, **{**inherited, **changes})
        validation = effective.validate_plan(winner["deployment"], winner["routing"])
        if not validation["constraints_valid"]:
            raise ValueError("Cannot execute a plan with unknown colocation capacity")
        explicit_resources = options.get("resource_configuration") is not None
        matched = ({"U": None, "eligible": False,
                    "reason": "RESOURCE_SPECIFIC_PREDICTOR_NOT_VALIDATED"}
                   if explicit_resources else DeploymentSystem(effective).predict(
                       winner["deployment"], winner["routing"]))
        same_problem = source_problem == effective.as_dict() and not explicit_resources
        restrictions = copy.deepcopy(plan.get("search_restrictions", {}))
        restricted = any(restrictions.get(key) for key in (
            "fixed_deployment", "exclude_deployments", "exclude_plans", "no_split", "replica_options"
        ))
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
            "search_restrictions": restrictions,
            "certificate_scope": "restricted_frozen_predictor_domain" if restricted
            else "frozen_predictor_domain",
            "unrestricted_optimality_transferred": same_problem
            and plan.get("optimality_certified", False) and not restricted,
            "physical_optimality_certified": False,
            "matched_prediction": matched,
            "reason": (
                "SAME_PROBLEM"
                if same_problem
                else ("RESOURCE_CONFIGURATION_CHANGED_PREDICTOR_UNKNOWN" if explicit_resources
                      else "RUN_OPTIONS_CHANGED_PROBLEM_REPREDICTED")
            ),
        }
        return raw
