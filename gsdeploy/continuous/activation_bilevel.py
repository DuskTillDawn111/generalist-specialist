"""Activation-aware BO + mesh-adaptive local polls + progressive MILP.

Partial inner values remain lower bounds, never GP regression targets.
The best feasible joint plan survives every scheduling/budget decision.
"""

import copy
import math
import time

import numpy as np

from .activation import active_models
from .activation_search import ActivationSpaces
from .bayesian_search import CertifiedRouteGP
from .contracts import BudgetExhausted, digest
from .deployment_pool import DeploymentCandidatePool
from .mesh_search import MeshAdaptivePoll
from .progressive import ProgressiveFixedRoutingSolver


class ActivationAwareBilevelOptimizer:
    def __init__(self, evaluator, *, seed=11, pool=None, max_spaces=32,
                 initial_endpoints=4, bo_candidates_per_space=8, promotion_period=4,
                 poll_evaluations=8, inner_seconds=1., inner_predictions=8,
                 inner_milp_calls=16, pool_evaluations=8, pool_max_size=16,
                 policy="budget", use_bo=True, use_mads=True, exact_inner=False,
                 joint_seed_k=None, joint_seed_count=1, joint_seed_fraction=.2):
        for value in (bo_candidates_per_space, promotion_period, poll_evaluations, joint_seed_count):
            if type(value) is not int or value < 1:
                raise ValueError("INVALID_BILEVEL_COUNT")
        if (type(seed) is not int or type(initial_endpoints) is not int or initial_endpoints < 0
                or any(type(b) is not bool for b in (use_bo, use_mads, exact_inner))
                or not (use_bo or use_mads) or not math.isfinite(joint_seed_fraction)
                or not 0 < joint_seed_fraction < 1
                or joint_seed_k is not None and (type(joint_seed_k) is not int or joint_seed_k < 1)):
            raise ValueError("INVALID_BILEVEL_SETTINGS")
        self.e, self.seed = evaluator, seed
        self.pool = pool if pool is not None else DeploymentCandidatePool(evaluator.problem, pool_max_size)
        self.inner = ProgressiveFixedRoutingSolver(evaluator, self.pool, policy=policy,
            inner_seconds=inner_seconds, inner_predictions=inner_predictions,
            inner_milp_calls=inner_milp_calls, pool_evaluations=pool_evaluations)
        self.spaces = ActivationSpaces(evaluator.domain, evaluator.problem.config.catalog, max_spaces)
        self.gp = CertifiedRouteGP()
        self.rng = np.random.default_rng(seed)
        self.initial_endpoints, self.bo_candidates = initial_endpoints, bo_candidates_per_space
        self.promotion_period, self.poll_evaluations = promotion_period, poll_evaluations
        self.use_bo, self.use_mads, self.exact_inner = use_bo, use_mads, exact_inner
        self.joint_k, self.joint_count, self.joint_fraction = joint_seed_k, joint_seed_count, joint_seed_fraction
        self.rows, self.events, self.convergence, self.polls = {}, [], [], {}
        self.joint_result, self.joint_best = None, None
        self.iteration = 0
        self.rewards = dict(global_search=1., local_search=1.)

    def _winner(self):
        choices = [r for r in (self.inner.best, self.joint_best) if r is not None]
        return max(choices, key=lambda r: r["best_feasible_utility"]) if choices else None

    def _evaluate(self, shares, source, **kwargs):
        # Progressive inner solves restore their temporary prediction quotas
        # before returning. Check here, outside that scope, so a global limit
        # stops new route work while a local quota only yields to the scheduler.
        # Existing inner ledgers and the incumbent remain available on exit.
        self.e.budget.check("predictions")
        before = self._winner()
        before_counts = self.e.budget.counts.copy()
        start = time.perf_counter()
        row = self.inner.evaluate(shares, exact=self.exact_inner, **kwargs)
        row = copy.deepcopy(row)
        row.update(source=source, iteration=self.iteration)
        self.rows[digest(row["routing_shares"])] = row
        self.spaces.observe(row)
        best = self._winner()
        self.events.append(dict(source=source, iteration=self.iteration,
            routing_shares=row["routing_shares"], activation=list(active_models(row["routing_qps"])),
            lb=row["best_feasible_utility"], ub=row["valid_upper_bound"], certified=row["certified"],
            status=row["status"], pool_version=row["pool_version"], seconds=time.perf_counter()-start))
        self.convergence.append(dict(**self.e.budget.report(),
            best_feasible_utility=best["best_feasible_utility"] if best else None,
            source=source, pool_version=self.pool.version))
        gain = (max(0., best["best_feasible_utility"]-before["best_feasible_utility"])
                if best is not None and before is not None else float(best is not None))
        stage = "local_search" if source.startswith("mads") else "global_search"
        # Use measured work counts rather than noisy sub-second wall times so
        # a count-budget run remains reproducible with the same seed.
        work = (1.+self.e.budget.counts["predictions"]-before_counts["predictions"]+
                .1*(self.e.budget.counts["milp"]-before_counts["milp"]))
        reward = gain/work
        self.rewards[stage] = .8*self.rewards[stage]+.2*reward
        if row["status"] == "SOLVER_ERROR":
            raise BudgetExhausted("INNER_SOLVER_ERROR")
        if self.exact_inner and not row["certified"]:
            self.e.budget.check("predictions")
            raise BudgetExhausted("INNER_EVALUATION_INCOMPLETE")
        return row

    def _joint_initialize(self):
        if self.joint_k is None:
            return []
        from .immutable import SessionPredictorOracle
        from .multi_candidate import JointDeploymentCandidates
        if not isinstance(self.e.oracle, SessionPredictorOracle) or len(self.e.problem.train_traces) != 1:
            raise ValueError("JOINT_INITIALIZER_REQUIRES_SESSION_AND_ONE_TRAIN_TRACE")
        budget = self.e.budget
        deadline, limits = budget.deadline, budget.limits.copy()
        extractor = JointDeploymentCandidates(self.e.problem, self.e.oracle, budget,
                                              cache=self.e.prediction_cache)
        try:
            budget.check()
            budget.deadline = min(deadline, time.perf_counter()+
                                  self.joint_fraction*(deadline-time.perf_counter()))
            for name in ("predictions", "milp"):
                remaining = limits[name]-budget.counts[name]
                budget.limits[name] = budget.counts[name]+max(1, int(self.joint_fraction*remaining)) if remaining else limits[name]
            candidates = extractor.generate(self.joint_count, K=self.joint_k)
        except BudgetExhausted:
            candidates = []
        finally:
            budget.deadline, budget.limits = deadline, limits
        # A valid initializer incumbent must survive even if later outer
        # evaluation has no budget. Raw prediction is explicitly identified.
        if extractor.best is not None:
            b = extractor.best
            shares = self.e.domain.validate({t: {m: q/self.e.problem.config.workload[t]
                 for m, q in row.items()} for t, row in b["routing"].items()})
            self.joint_best = dict(deployment=copy.deepcopy(b["deployment"]), routing_shares=shares,
                routing_qps=copy.deepcopy(b["routing"]), best_feasible_utility=b["utility"],
                prediction=copy.deepcopy(b["prediction"]), prediction_format="predict_scenarios",
                source="joint_initializer")
            self.pool.add(b["deployment"], source="joint_initializer", routing=b["routing"], utility=b["utility"])
        for c in candidates:
            self.pool.add(c["deployment"], source="joint_initializer", routing=c["grid_routing"], utility=c["grid_utility"])
        self.joint_result = dict(K=self.joint_k, candidates=candidates, solves=extractor.solves,
                                 shared_global_budget=True, optional=True)
        return [self.e.domain.validate({t: {m: q/self.e.problem.config.workload[t] for m, q in row.items()}
                for t, row in c["grid_routing"].items()}) for c in candidates]

    def _refresh(self, row):
        current = self.rows[digest(row["routing_shares"])]
        if not current["certified"] and current["pool_version"] != self.pool.version:
            return self._evaluate(current["routing_shares"], "pool_refresh", pool_only=True)
        return current

    def _promote(self):
        best = self._winner()
        incumbent = best["best_feasible_utility"] if best else None
        partial = [r for r in self.rows.values() if not r["certified"] and
                   not r["master_domain_exhausted"] and (incumbent is None or
                   r["valid_upper_bound"] is None or r["valid_upper_bound"] > incumbent)]
        if not partial:
            return False
        # Older evaluations get a turn even if their lower bound is weak.
        target = min(partial, key=lambda r: (r["iteration"], -(r["valid_upper_bound"]
                     if r["valid_upper_bound"] is not None else math.inf)))
        self._evaluate(target["routing_shares"], "inner_promotion", force=True)
        return True

    def _global_step(self):
        self.gp.fit(self.rows.values(), self.spaces.features)
        keys = list(self.spaces.spaces)
        # A persistent exploration quota covers new activation spaces even
        # when GP EI concentrates on a previously strong combination.
        unseen = [k for k in keys if self.spaces.spaces[k]["visits"] == 0]
        if unseen:
            key = unseen[0]
            shares = self.spaces.center(key)
            if digest(shares) not in self.rows:
                self.spaces.expand(key)
                return self._evaluate(shares, "activation_exploration")
        candidates = {}
        for key in keys:
            self.e.budget.check()
            face = self.spaces.spaces[key]["face"]
            center = self.spaces.center(key)
            for _ in range(self.bo_candidates):
                point = self.spaces.random(key, self.rng)
                candidates[digest(point)] = point
            # Explicit task-edge zeros, including boundaries that leave a
            # model active through a different task.
            vector = self.e.domain.encode(center, face)
            for j in range(len(vector)):
                for edge in (0., 1.):
                    trial = list(vector); trial[j] = edge
                    point = self.e.domain.decode(trial, face)
                    candidates[digest(point)] = point
        candidates = [p for k, p in candidates.items() if k not in self.rows]
        if not candidates:
            return None
        features = np.asarray([self.spaces.features(p) for p in candidates])
        if self.use_bo and self.gp.factor is not None and self.iteration % 4:
            index = int(np.argmax(self.gp.expected_improvement(features)))
            source = "bo_expected_improvement"
        else:
            index = int(self.rng.integers(len(candidates)))
            source = "global_exploration"
        row = self._evaluate(candidates[index], source)
        key = tuple(active_models(row["routing_qps"]))
        if key in self.spaces.spaces:
            self.spaces.expand(key)
        return row

    @staticmethod
    def _comparison(parent, trial):
        """Return local decision and whether it was resolved by evidence."""
        pu, tu = parent["best_feasible_utility"], trial["best_feasible_utility"]
        if tu is not None and parent["valid_upper_bound"] is not None and tu > parent["valid_upper_bound"]:
            return "success", False
        if pu is not None and trial["valid_upper_bound"] is not None and trial["valid_upper_bound"] <= pu:
            return "failure", False
        if parent["certified"] and trial["certified"]:
            if tu is not None and (pu is None or tu > pu):
                return "success", False
            return "failure", False
        if tu is not None and (pu is None or tu > pu):
            return "success", True
        return "inconclusive", True

    def _local_step(self):
        feasible = [(k, v) for k, v in self.spaces.spaces.items() if v["best"] is not None
                    and any(len(row) > 1 for row in v["face"].values())]
        if not feasible:
            return False
        # Round-robin among promising spaces prevents one local sequence
        # from monopolizing all poll effort.
        feasible.sort(key=lambda item: (-item[1]["best"]["best_feasible_utility"], item[0]))
        key, space = feasible[(self.iteration//2) % len(feasible)]
        parent = self._refresh(space["best"])
        face = space["face"]
        center_key = digest(parent["routing_shares"])
        dimension = sum(len(row)-1 for row in face.values())
        if key not in self.polls:
            self.polls[key] = dict(poll=MeshAdaptivePoll(dimension, seed=int(self.rng.integers(2**31))),
                                   center=None, pending=[], unresolved=False)
        state = self.polls[key]
        if state["center"] != center_key:
            state.update(center=center_key, pending=[], unresolved=False)
        if not state["pending"]:
            vector = self.e.domain.encode(parent["routing_shares"], face)
            state["pending"] = state["poll"].candidates(vector)
        for _ in range(min(self.poll_evaluations, len(state["pending"]))):
            vector = state["pending"].pop(0)
            trial = self._evaluate(self.e.domain.decode(vector, face), "mads_poll")
            parent, trial = self._refresh(parent), self._refresh(trial)
            outcome, heuristic = self._comparison(parent, trial)
            if heuristic and not trial["certified"] and not trial["master_domain_exhausted"]:
                trial = self._evaluate(trial["routing_shares"], "mads_inner_refinement", force=True)
                parent, trial = self._refresh(parent), self._refresh(trial)
                outcome, heuristic = self._comparison(parent, trial)
            self.events.append(dict(source="mads_comparison", parent_lb=parent["best_feasible_utility"],
                trial_lb=trial["best_feasible_utility"], parent_ub=parent["valid_upper_bound"],
                trial_ub=trial["valid_upper_bound"], outcome=outcome, heuristic=heuristic,
                parent_version=parent["pool_version"], trial_version=trial["pool_version"],
                pool_version=self.pool.version))
            if outcome == "success":
                # The trial may activate fewer models at a boundary. Keep it
                # as the local center in this containing closed space too.
                space["best"] = trial
                state["pending"] = []
                state["unresolved"] = False
                state["center"] = digest(trial["routing_shares"])
                state["poll"].update("inconclusive" if heuristic else "success")
                return True
            state["unresolved"] |= outcome == "inconclusive"
        if not state["pending"]:
            state["poll"].update("inconclusive" if state["unresolved"] else "failure")
            state["unresolved"] = False
        return True

    def solve(self):
        status = "SEARCH_COMPLETED"
        try:
            self.e.check()
            self.e.budget.check("predictions")
            self.spaces.initialize()
            initial = self._joint_initialize()
            for index, shares in enumerate(self.e.domain.endpoints()):
                if index >= self.initial_endpoints:
                    break
                initial.append(shares)
            initial += [self.spaces.center(k) for k in list(self.spaces.spaces)]
            for shares in initial:
                shares = self.e.domain.validate(shares)
                if digest(shares) not in self.rows:
                    self._evaluate(shares, "initialization", force=True)
            if self.e.domain.dimension == 0:
                while self._promote():
                    self.e.budget.check()
                status = "SINGLE_ROUTE_COMPLETE" if all(r["certified"] for r in self.rows.values()) else "INCOMPLETE_EVIDENCE"
            else:
                while True:
                    # An inner solver may return a useful partial result after
                    # consuming the last prediction. Preserve it, then stop
                    # before fitting/proposing another search candidate.
                    self.e.budget.check("predictions")
                    self.e.budget.check("outer")
                    self.iteration += 1
                    if self.iteration % self.promotion_period == 0 and self._promote():
                        continue
                    reward_sum = sum(self.rewards.values())
                    global_probability = min(.75, max(.25, self.rewards["global_search"]/reward_sum if reward_sum else .5))
                    local = self.use_mads and (not self.use_bo or self.rng.random() > global_probability)
                    if local and self._local_step():
                        continue
                    if self._global_step() is None:
                        if not self._promote():
                            status = "NO_NEW_PROPOSAL"; break
        except BudgetExhausted as exc:
            status = str(exc)
        self.gp.fit(self.rows.values(), self.spaces.features)
        return dict(algorithm="ActivationAwareBilevel-BO-MADS", status=status,
            winner=copy.deepcopy(self._winner()), budget=self.e.budget.report(), seed=self.seed,
            continuous_optimality=False, optimality_certified=False, native_qualified=False,
            evaluation_semantics="exact_inner" if self.exact_inner else "bounds_and_feasible_lower_bounds",
            bo_enabled=self.use_bo, mads_enabled=self.use_mads, surrogate=self.gp.report(),
            activation_spaces=self.spaces.report(), pool=self.pool.report(),
            inner_sessions=self.inner.inner_sessions, history=copy.deepcopy(list(self.rows.values())),
            events=copy.deepcopy(self.events), convergence=copy.deepcopy(self.convergence),
            polls=[dict(models=list(k), **v["poll"].report()) for k, v in self.polls.items()],
            joint_initializer=self.joint_result, scheduler_rewards=self.rewards,
            problem_sha256=self.e.problem.identity, objective_id=self.e.problem.objective_id,
            oracle_version=self.e.oracle.version, evidence_kind=self.e.oracle.evidence_kind)
