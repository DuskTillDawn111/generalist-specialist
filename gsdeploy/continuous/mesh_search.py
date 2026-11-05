"""Randomized positive-spanning mesh polls; bounds are rejected, not clipped.

The mesh refines faster than the poll frame. Integer rounded orthogonal bases
give varying normalized directions, unlike fixed coordinate pattern search.
No MADS convergence theorem is claimed for budgeted/inexact bilevel values.
"""

import numpy as np


class MeshAdaptivePoll:
    def __init__(self, dimension, seed=11, level=1, max_level=20):
        if (type(dimension) is not int or dimension < 0 or type(level) is not int
                or type(max_level) is not int or not 0 <= level <= max_level <= 24):
            raise ValueError("INVALID_MESH_SETTINGS")
        self.dimension, self.level, self.max_level = dimension, level, max_level
        self.rng = np.random.default_rng(seed)
        self.history = []

    @property
    def mesh(self):
        return 4.**(-self.level)

    @property
    def frame(self):
        return 2.**(-self.level)

    def basis(self):
        d = self.dimension
        if not d:
            return np.zeros((0, 0), dtype=int)
        q, _ = np.linalg.qr(self.rng.normal(size=(d, d)))
        scale = 2**self.level
        b = np.rint(q*scale).astype(int)
        if np.linalg.matrix_rank(b) != d:
            b = np.eye(d, dtype=int)*scale
        return b

    def candidates(self, center):
        x = np.asarray(center, dtype=float)
        if len(x) != self.dimension or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("INVALID_POLL_CENTER")
        basis = self.basis()
        # Coordinate directions additionally cover feasible corner directions.
        directions = np.concatenate((basis.T, -basis.T,
                                     np.eye(self.dimension, dtype=int)*2**self.level,
                                     -np.eye(self.dimension, dtype=int)*2**self.level))
        seen, result = set(), []
        for direction in directions:
            point = x+self.mesh*direction
            key = tuple(point)
            if key not in seen and np.all((point >= 0) & (point <= 1)) and np.any(direction):
                seen.add(key); result.append(point)
        self.rng.shuffle(result)
        return result

    def update(self, outcome):
        if outcome not in {"success", "failure", "inconclusive"}:
            raise ValueError("INVALID_POLL_OUTCOME")
        self.history.append(dict(level=self.level, mesh=self.mesh, frame=self.frame, outcome=outcome))
        if outcome == "success":
            self.level = max(0, self.level-1)
        elif outcome == "failure":
            self.level = min(self.max_level, self.level+1)

    def report(self):
        return dict(level=self.level, mesh=self.mesh, frame=self.frame, polls=self.history,
                    varying_positive_spanning_integer_basis=True,
                    continuous_stationarity_certified=False)
