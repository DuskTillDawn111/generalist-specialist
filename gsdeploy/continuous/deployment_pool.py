"""Bounded deployment structures shared across routes, never shared scores."""

import copy
import math

from .activation import active_models


def deployment_key(problem, deployment):
    if not isinstance(deployment, dict) or not deployment:
        raise ValueError("NONEMPTY_DEPLOYMENT_REQUIRED")
    for model, count in deployment.items():
        if (model not in problem.states or type(count) is not int
                or count not in problem.states[model]):
            raise ValueError("INVALID_POOL_REPLICA_STATE")
    key = tuple(sorted((m, n) for m, n in deployment.items() if n))
    if not key:
        raise ValueError("NONEMPTY_DEPLOYMENT_REQUIRED")
    return key


class DeploymentCandidatePool:
    def __init__(self, problem, max_size=16, selection_policy="utility", diversity=True):
        if (type(max_size) is not int or max_size < 1
                or selection_policy not in {"utility", "recent"} or type(diversity) is not bool):
            raise ValueError("INVALID_POOL_SETTINGS")
        self.problem = problem
        self.identity = problem.identity
        self.max_size, self.policy, self.diversity = max_size, selection_policy, diversity
        self.entries, self.events = {}, []
        self.version, self.sequence = 0, 0
        self.incumbent = None

    def check(self):
        self.problem.check()
        if self.problem.identity != self.identity:
            raise ValueError("POOL_PROBLEM_CHANGED")

    def add(self, deployment, *, source, routing=None, utility=None):
        self.check()
        key = deployment_key(self.problem, deployment)
        if utility is not None and not math.isfinite(utility):
            raise ValueError("NONFINITE_POOL_UTILITY")
        self.sequence += 1
        if utility is not None and (self.incumbent is None or utility > self.incumbent[1]):
            self.incumbent = key, utility
        if key not in self.entries:
            self.entries[key] = dict(deployment=dict(key), source=source,
                discovered_at=self.sequence, evaluations=0, best_utility=None,
                first_routing=copy.deepcopy(routing), best_routing=None,
                last_improved=None, problem_sha256=self.identity,
                covered_tasks=sorted({t for m, _ in key for t in
                    self.problem.config.catalog["models"][m]["supported_tasks"]}),
                memory_status="REQUIRES_ROUTE_CONTEXT_PREDICTION",
                structural_state_valid=True)
            self.version += 1
            self.events.append(dict(action="ADD", deployment=dict(key), source=source,
                                    version=self.version, sequence=self.sequence))
        entry = self.entries[key]
        if utility is not None:
            entry["evaluations"] += 1
            if entry["best_utility"] is None or utility > entry["best_utility"]:
                entry.update(best_utility=utility, best_routing=copy.deepcopy(routing),
                             last_improved=self.sequence)
        if len(self.entries) > self.max_size:
            protected = self.incumbent[0] if self.incumbent else None
            victims = [k for k in self.entries if k != protected]
            def priority(k):
                row = self.entries[k]
                activation = tuple(m for m, _ in k)
                duplicates = sum(tuple(m for m, _ in other) == activation for other in self.entries)
                value = row["best_utility"]
                return (-(duplicates if self.diversity else 0),
                        row["discovered_at"] if self.policy == "recent" else
                        value if value is not None else -math.inf,
                        row["discovered_at"], k)
            victim = min(victims, key=priority)
            del self.entries[victim]
            self.version += 1
            self.events.append(dict(action="EVICT", deployment=dict(victim), version=self.version,
                                    reason="HEURISTIC_POOL_CAPACITY", safe_prune=False))
        return key

    def compatible(self, routing):
        self.check()
        active = set(active_models(routing))
        rows = [r for k, r in self.entries.items() if active <= {m for m, _ in k}]
        rows.sort(key=lambda r: (-(r["best_utility"] if r["best_utility"] is not None else -math.inf),
                                 r["discovered_at"]))
        return [copy.deepcopy(r["deployment"]) for r in rows]

    def report(self):
        return dict(version=self.version, size=len(self.entries), max_size=self.max_size,
                    entries=copy.deepcopy(list(self.entries.values())), events=copy.deepcopy(self.events),
                    eviction_is_safe_pruning=False)
