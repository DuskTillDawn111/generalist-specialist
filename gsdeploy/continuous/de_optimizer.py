"""Budgeted differential evolution over closed task simplices."""

import copy
import math
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from .contracts import BudgetExhausted, digest


class DERoutingOptimizer:
    def __init__(self, evaluator, seed=42, strategy="best1bin", popsize=5,
                 maxiter=100, init="sobol", face_limit=128, initial_shares=None):
        if init not in {"sobol", "latinhypercube"}:
            raise ValueError("INITIALIZATION_MUST_BE_SOBOL_OR_LATINHYPERCUBE")
        if (type(seed) is not int or type(popsize) is not int or popsize < 1
                or type(maxiter) is not int or maxiter < 0
                or type(face_limit) is not int or face_limit < 1):
            raise ValueError("INVALID_DE_CONFIGURATION")
        self.evaluator, self.seed = evaluator, seed
        self.strategy, self.popsize, self.maxiter = strategy, popsize, maxiter
        self.init, self.face_limit = init, face_limit
        self.initial_shares = (evaluator.domain.validate(initial_shares)
                               if initial_shares is not None else None)

    def solve(self):
        e, domain = self.evaluator, self.evaluator.domain
        best, status, faces_seen, endpoint_count = None, "SEARCH_COMPLETED", set(), 0
        unique_routes = set()
        convergence = []
        uncompleted = []

        def evaluate(shares):
            nonlocal best
            key = digest(domain.validate(shares))
            if key in unique_routes:
                # Exact same point need not consume another outer budget slot.
                row = next(r for r in reversed(e.history) if digest(r["routing_shares"]) == key)
            else:
                row = e.evaluate(shares)
                unique_routes.add(key)
            if row["best_feasible_utility"] is not None and (
                    best is None or row["best_feasible_utility"] > best["best_feasible_utility"]):
                best = copy.deepcopy(row)
            convergence.append(dict(**e.budget.report(),
                best_feasible_utility=best["best_feasible_utility"] if best else None,
                candidate_status=row["status"]))
            if not row["inner_search_completed"]:
                uncompleted.append(row["status"])
                # The partial incumbent is retained, but is never handed back
                # to SciPy as an exact V(r) observation.
                raise BudgetExhausted("INNER_EVALUATION_INCOMPLETE")
            if not row["certified"]:
                raise BudgetExhausted("INCOMPLETE_PREDICTOR_EVIDENCE")
            return -row["best_feasible_utility"] if row["feasible"] else math.inf

        start = time.perf_counter()
        prediction_before = e.budget.times["prediction"]
        milp_before = e.budget.times["milp"]
        try:
            if self.initial_shares is not None:
                evaluate(self.initial_shares)
            # Endpoints come first and include all feasible pure-model covers.
            for shares in domain.endpoints():
                evaluate(shares)
                endpoint_count += 1
            faces = []
            for face in domain.faces():
                if len(faces) >= self.face_limit:
                    break
                faces.append(face)
                center = domain.decode([0.5] * sum(len(v)-1 for v in face.values()), face)
                evaluate(center)
                faces_seen.add(digest(face))
            dim = domain.dimension
            if dim:
                size = max(5, self.popsize * dim)
                sampler = (qmc.Sobol(dim, scramble=True, seed=self.seed) if self.init == "sobol"
                           else qmc.LatinHypercube(dim, seed=self.seed))
                if self.init == "sobol":
                    initial = sampler.random_base2(math.ceil(math.log2(size)))
                else:
                    initial = sampler.random(size)
                # Do one global DE generation stream, then bounded face runs.
                def run(face, population, seed):
                    face_dim = sum(len(v)-1 for v in face.values())
                    if not face_dim:
                        return
                    differential_evolution(lambda v: evaluate(domain.decode(v, face)),
                        [(0.0, 1.0)] * face_dim, strategy=self.strategy, seed=seed,
                        popsize=self.popsize, maxiter=self.maxiter, polish=False,
                        workers=1, updating="immediate", tol=0, atol=0, init=population,
                        **({"x0": domain.encode(self.initial_shares, face)}
                           if self.initial_shares is not None and face == domain.edges else {}))
                run(domain.edges, initial, self.seed)
                for i, face in enumerate(faces):
                    if face != domain.edges:
                        run(face, self.init, self.seed+i+1)
        except BudgetExhausted as exc:
            status = str(exc)
        wall = time.perf_counter() - start
        e.budget.times["outer"] += max(0.0, wall - (e.budget.times["prediction"]-prediction_before)
                                        - (e.budget.times["milp"]-milp_before))
        return dict(algorithm="de_" + e.inner, winner=best, status=status,
            optimality_certified=False, continuous_optimality=False,
            seed=self.seed, evidence_kind=e.oracle.evidence_kind,
            activation_coverage=dict(total_faces=domain.face_count,
                seeded_faces=len(faces_seen), endpoint_evaluations=endpoint_count,
                all_faces_seeded=len(faces_seen) == domain.face_count,
                interior_search="global DE followed by face DE if budget remains"),
            exact_inner_comparison=not uncompleted and status != "INCOMPLETE_PREDICTOR_EVIDENCE",
            convergence=convergence, budget=e.budget.report(),
            native_qualified=False)
