"""Opt-in lower-bound-guided DE with version-consistent parent/trial choices.

The original exact-inner DERoutingOptimizer is unchanged. This controller
explicitly optimizes feasible lower estimates when refinement is incomplete.
"""

import copy
import math

import numpy as np

from .contracts import BudgetExhausted, digest
from .deployment_pool import DeploymentCandidatePool
from .progressive import ProgressiveFixedRoutingSolver


class C1LiteOptimizer:
    def __init__(self, evaluator, *, pool=None, seed=11, policy="budget", population_size=6,
                 endpoint_limit=4, face_limit=2, inner_seconds=1., inner_predictions=16,
                 inner_milp_calls=32, pool_evaluations=8, pool_max_size=16,
                 exploration_period=4, use_bounds=True, exact_inner=False,
                 mutation=.7, crossover=.7):
        if (type(seed) is not int or type(population_size) is not int or population_size < 4
                or type(endpoint_limit) is not int or endpoint_limit < 0
                or type(face_limit) is not int or face_limit < 0
                or not math.isfinite(mutation) or not 0 < mutation <= 2
                or not math.isfinite(crossover) or not 0 <= crossover <= 1):
            raise ValueError("INVALID_PROGRESSIVE_DE_SETTINGS")
        self.e, self.seed = evaluator, seed
        self.pool = pool if pool is not None else DeploymentCandidatePool(evaluator.problem, pool_max_size)
        self.inner = ProgressiveFixedRoutingSolver(evaluator, self.pool, policy=policy,
            inner_seconds=inner_seconds, inner_predictions=inner_predictions,
            inner_milp_calls=inner_milp_calls, pool_evaluations=pool_evaluations,
            exploration_period=exploration_period, use_bounds=use_bounds)
        self.population_size, self.endpoint_limit, self.face_limit = population_size, endpoint_limit, face_limit
        self.exact_inner, self.mutation, self.crossover = exact_inner, mutation, crossover

    def solve(self):
        e, inner, domain = self.e, self.inner, self.e.domain
        rng = np.random.default_rng(self.seed)
        points, choices, convergence = [], [], []
        endpoint_count, face_count, generation = 0, 0, 0
        status = "SEARCH_COMPLETED"

        def evaluate(shares, **kwargs):
            row = inner.evaluate(shares, exact=self.exact_inner, **kwargs)
            convergence.append(dict(**e.budget.report(), pool_version=self.pool.version,
                best_feasible_utility=inner.best["best_feasible_utility"] if inner.best else None,
                candidate_status=row["status"], evaluation_kind=row["evaluation_kind"]))
            if row["status"] == "SOLVER_ERROR":
                raise BudgetExhausted("INNER_SOLVER_ERROR")
            if self.exact_inner and not row["certified"]:
                raise BudgetExhausted("INNER_EVALUATION_INCOMPLETE")
            return row

        def refresh(row):
            if row["certified"] or row["pool_version"] == self.pool.version:
                return row
            return evaluate(row["routing_shares"], pool_only=True)

        def value(row):
            # None is an unresolved status. Only actual feasible incumbents
            # participate in numeric ranking; unknown is never a numeric target.
            return row["best_feasible_utility"]

        try:
            seen = set()
            for shares in domain.endpoints():
                if endpoint_count >= self.endpoint_limit:
                    break
                row = evaluate(shares)
                points.append((domain.encode(row["routing_shares"]), row))
                seen.add(digest(row["routing_shares"]))
                endpoint_count += 1
            for face in domain.faces():
                if face_count >= self.face_limit:
                    break
                if not any(len(v) > 1 for v in face.values()):
                    continue
                shares = domain.decode([.5]*sum(len(v)-1 for v in face.values()), face)
                if digest(shares) in seen:
                    continue
                row = evaluate(shares)
                points.append((domain.encode(row["routing_shares"]), row))
                seen.add(digest(shares)); face_count += 1
            dim = domain.dimension
            if dim == 0:
                if not points:
                    evaluate(domain.decode([]))
            else:
                while len(points) < self.population_size:
                    vector = rng.random(dim)
                    points.append((vector, evaluate(domain.decode(vector))))
                # Keep a bounded population without losing the global ledger
                # winner, even if endpoint initialization supplied extra points.
                points = points[:self.population_size]
                while True:
                    generation += 1
                    for i in range(len(points)):
                        e.budget.check("outer")
                        candidates = [j for j in range(len(points)) if j != i]
                        a, b, c = rng.choice(candidates, 3, replace=False)
                        parent_vector, parent = points[i]
                        mutant = np.clip(np.asarray(points[a][0]) + self.mutation*(
                            np.asarray(points[b][0])-np.asarray(points[c][0])), 0., 1.)
                        mask = rng.random(dim) < self.crossover
                        mask[rng.integers(dim)] = True
                        vector = np.where(mask, mutant, parent_vector)
                        trial = evaluate(domain.decode(vector))
                        # Inner discovery changes the feasible lower estimate
                        # for old points. Refresh every population member before
                        # comparing, at the same current pool version.
                        points = [(v, refresh(row)) for v, row in points]
                        trial = refresh(trial)
                        parent = points[i][1]
                        refined = False
                        pu, tu = value(parent), value(trial)
                        separated = (tu is not None and parent["valid_upper_bound"] is not None
                                     and tu > parent["valid_upper_bound"]) or (
                                     pu is not None and trial["valid_upper_bound"] is not None
                                     and pu > trial["valid_upper_bound"])
                        if not separated and generation % self.inner.trigger.period == 0:
                            target = trial if not trial["certified"] else parent
                            if not target["certified"]:
                                updated = evaluate(target["routing_shares"], force=True)
                                if target is trial:
                                    trial = updated
                                else:
                                    points[i] = points[i][0], updated
                                points = [(v, refresh(row)) for v, row in points]
                                trial = refresh(trial); parent = points[i][1]; refined = True
                        pu, tu = value(parent), value(trial)
                        replace = tu is not None and (pu is None or tu > pu)
                        choice = dict(generation=generation, parent_lb=pu, trial_lb=tu,
                            parent_ub=parent["valid_upper_bound"], trial_ub=trial["valid_upper_bound"],
                            parent_certified=parent["certified"], trial_certified=trial["certified"],
                            pool_version=self.pool.version, parent_version=parent["pool_version"],
                            trial_version=trial["pool_version"], refined=refined, replaced=replace,
                            heuristic=not (parent["certified"] and trial["certified"]),
                            unresolved_unknown=(pu is None or tu is None))
                        choices.append(choice)
                        if replace:
                            points[i] = vector, trial
        except BudgetExhausted as exc:
            status = str(exc)
        return dict(algorithm="C1-Lite", status=status, winner=copy.deepcopy(inner.best),
            evaluation_semantics="exact_inner" if self.exact_inner else "lower_bound_guided",
            continuous_optimality=False, optimality_certified=False, native_qualified=False,
            evidence_kind=e.oracle.evidence_kind, seed=self.seed,
            inner_policy=self.inner.trigger.policy, inner_sessions=inner.inner_sessions,
            heuristic_selections=sum(r["heuristic"] for r in choices), selections=choices,
            pool=self.pool.report(), history=copy.deepcopy(inner.history), convergence=convergence,
            activation_coverage=dict(endpoint_evaluations=endpoint_count,
                endpoint_total=math.prod(len(v) for v in domain.edges.values()),
                face_evaluations=face_count, total_faces=domain.face_count,
                endpoint_limit=self.endpoint_limit, face_limit=self.face_limit),
            budget=e.budget.report(), problem_sha256=e.problem.identity,
            oracle_version=e.oracle.version, hybrid_gate_passed=False)
