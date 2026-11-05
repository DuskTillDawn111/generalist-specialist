"""Closed simplices, exact endpoints and task-specific support faces."""

import itertools
import math


class ContinuousRouteDomain:
    def __init__(self, problem):
        self.rates = dict(sorted(problem.config.workload.items()))
        self.edges = {t: tuple(m for m in problem.states
            if any(problem.states[m]) and t in problem.config.catalog["models"][m]["supported_tasks"])
            for t in self.rates}
        if any(not row for row in self.edges.values()):
            raise ValueError("UNCOVERED_TASK")
        self.dimension = sum(len(row) - 1 for row in self.edges.values())

    def validate(self, shares):
        if set(shares) != set(self.edges):
            raise ValueError("ROUTING_TASK_MISMATCH")
        result = {}
        for t, models in self.edges.items():
            row = shares[t]
            if set(row) - set(models):
                raise ValueError("UNSUPPORTED_ROUTE")
            if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 or v > 1
                   for v in row.values()):
                raise ValueError("INVALID_SHARE")
            if not math.isclose(math.fsum(row.values()), 1, rel_tol=0, abs_tol=2e-15):
                raise ValueError("SHARES_DO_NOT_SUM_TO_ONE")
            result[t] = {m: float(row.get(m, 0)) for m in models}
        return result

    def decode(self, vector, face=None):
        edges = self.edges if face is None else face
        if set(edges) != set(self.edges) or any(
                not edges[t] or set(edges[t]) - set(self.edges[t]) for t in edges):
            raise ValueError("INVALID_SUPPORT_FACE")
        if len(vector) != sum(len(v) - 1 for v in edges.values()):
            raise ValueError("VECTOR_DIMENSION_MISMATCH")
        cursor, shares = 0, {}
        for t in self.edges:
            models, remainder, row = edges[t], 1.0, {}
            for m in models[:-1]:
                u = float(vector[cursor])
                cursor += 1
                if not math.isfinite(u) or not 0 <= u <= 1:
                    raise ValueError("INVALID_CONTINUOUS_COORDINATE")
                value = remainder * u
                if remainder > 0 and u > 0 and value == 0:
                    raise ValueError("SHARE_UNDERFLOW")
                row[m] = value
                remainder -= value
            row[models[-1]] = remainder
            shares[t] = row
        return self.validate(shares)

    def to_qps(self, shares):
        shares = self.validate(shares)
        rows = {}
        for t, row in shares.items():
            rows[t] = {}
            for m, value in row.items():
                q = self.rates[t] * value
                if value > 0 and q == 0:
                    raise ValueError("QPS_UNDERFLOW")
                if q > 0:
                    rows[t][m] = q
        return rows

    def encode(self, shares, face=None):
        """Inverse stick coordinates for an exact legal warm-start route."""
        shares = self.validate(shares)
        edges = self.edges if face is None else face
        if set(edges) != set(self.edges) or any(
                not edges[t] or set(edges[t])-set(self.edges[t]) for t in edges):
            raise ValueError("INVALID_SUPPORT_FACE")
        vector = []
        for task, models in edges.items():
            if any(v > 0 and m not in models for m, v in shares[task].items()):
                raise ValueError("WARM_START_OUTSIDE_FACE")
            remainder = 1.0
            for model in models[:-1]:
                value = shares[task][model]
                coordinate = value/remainder if remainder > 0 else 0.0
                # Arithmetic roundoff may leave a sub-ulp excess only at a
                # simplex vertex; validate() already checked the original row.
                vector.append(min(1.0, max(0.0, coordinate)))
                remainder -= value
        return vector

    def endpoints(self):
        for selected in itertools.product(*self.edges.values()):
            yield {t: {m: 1.0} for t, m in zip(self.edges, selected)}

    def faces(self):
        options = [tuple(comb for n in range(1, len(models) + 1)
                         for comb in itertools.combinations(models, n))
                   for models in self.edges.values()]
        for selected in itertools.product(*options):
            yield dict(zip(self.edges, selected))

    @property
    def face_count(self):
        return math.prod(2 ** len(models) - 1 for models in self.edges.values())
