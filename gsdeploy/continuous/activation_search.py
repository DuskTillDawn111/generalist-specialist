"""Budgeted model-activation spaces, separate from deployment residency."""

import numpy as np

from .activation import active_models


class ActivationSpaces:
    """Discover covering model subsets lazily; do not expand a routing grid."""

    def __init__(self, domain, catalog, max_spaces=32):
        if type(max_spaces) is not int or max_spaces < 1:
            raise ValueError("INVALID_ACTIVATION_SPACE_LIMIT")
        self.domain, self.catalog, self.limit = domain, catalog, max_spaces
        self.models = tuple(sorted({m for row in domain.edges.values() for m in row}))
        self.spaces, self.events, self.dropped = {}, [], 0

    def add(self, models, source):
        key = tuple(sorted(set(models)))
        if not key or set(key)-set(self.models):
            return None
        face = {t: tuple(m for m in row if m in key) for t, row in self.domain.edges.items()}
        if any(not row for row in face.values()):
            return None
        if key in self.spaces:
            return key
        if len(self.spaces) >= self.limit:
            self.dropped += 1
            return None
        self.spaces[key] = dict(face=face, visits=0, best=None, source=source)
        self.events.append(dict(models=list(key), source=source))
        return key

    def initialize(self):
        # A full space guarantees task coverage even if the space cap is one.
        self.add(self.models, "full_cover")
        for m in self.models:
            self.add((m,), "single_model_cover")
        specialists = {m for m in self.models if self.catalog["models"][m].get("role") == "specialist"}
        self.add(specialists, "specialist_cover")

    def observe(self, row):
        key = self.add(active_models(row["routing_qps"]), "observed_exact_activation")
        if key is not None:
            item = self.spaces[key]
            item["visits"] += 1
            if row["best_feasible_utility"] is not None and (item["best"] is None or
                    row["best_feasible_utility"] > item["best"]["best_feasible_utility"]):
                item["best"] = row
        return key

    def expand(self, key):
        # One-model additions/removals and swaps allow activation transitions.
        present = set(key)
        for m in self.models:
            self.add(present ^ {m}, "activation_toggle")
        for old in key:
            for new in self.models:
                if new not in present:
                    self.add((present-{old}) | {new}, "activation_swap")

    def center(self, key):
        return self.domain.validate({t: {m: 1./len(row) for m in row}
                                     for t, row in self.spaces[key]["face"].items()})

    def random(self, key, rng):
        # Dirichlet samples remain inside the chosen task simplices.
        return self.domain.validate({t: dict(zip(row, map(float, rng.dirichlet(np.ones(len(row))))))
                                     for t, row in self.spaces[key]["face"].items()})

    def features(self, shares):
        routing = self.domain.to_qps(shares)
        active = set(active_models(routing))
        return np.asarray([shares[t][m] for t, row in self.domain.edges.items() for m in row]+
                          [float(m in active) for m in self.models])

    def report(self):
        return dict(discovered=len(self.spaces), max_spaces=self.limit, capacity_drops=self.dropped,
            exhaustive=False, model_subset_count=2**len(self.models)-1,
            spaces=[dict(models=list(k), visits=v["visits"], source=v["source"],
                         dimension=sum(len(row)-1 for row in v["face"].values()))
                    for k, v in self.spaces.items()], events=self.events)
