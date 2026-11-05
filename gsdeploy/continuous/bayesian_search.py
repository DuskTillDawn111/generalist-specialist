"""A small GP/EI search model trained only on certified fixed-route values."""

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr


class CertifiedRouteGP:
    def __init__(self, length_scale=.4, jitter=1e-8, max_training=128):
        if (not math.isfinite(length_scale) or length_scale <= 0 or
                not math.isfinite(jitter) or jitter <= 0 or
                type(max_training) is not int or max_training < 1):
            raise ValueError("INVALID_GP_SETTINGS")
        self.length_scale, self.jitter, self.max_training = length_scale, jitter, max_training
        self.count, self.excluded_partial, self.failure = 0, 0, None

    def _kernel(self, a, b):
        return np.exp(-.5*np.sum((a[:, None, :]-b[None, :, :])**2, axis=2)/self.length_scale**2)

    def fit(self, observations, features):
        observations = list(observations)
        eligible = [r for r in observations if r["certified"] and
                    r["best_feasible_utility"] is not None and r["status"] != "SOLVER_ERROR"]
        self.excluded_partial = sum(not r["certified"] for r in observations)
        # Bounded refits retain the best exact value and recent exact values.
        if len(eligible) > self.max_training:
            best = max(eligible, key=lambda r: r["best_feasible_utility"])
            eligible = ([best]+[r for r in eligible if r is not best][-(self.max_training-1):]
                        if self.max_training > 1 else [best])
        self.count, self.failure = len(eligible), None
        self.factor = None
        if not eligible:
            return
        self.x = np.asarray([features(r["routing_shares"]) for r in eligible])
        y = np.asarray([r["best_feasible_utility"] for r in eligible])
        self.mean, self.scale = float(y.mean()), max(float(y.std()), 1e-3)
        self.y = (y-self.mean)/self.scale
        self.best = float(self.y.max())
        k = self._kernel(self.x, self.x)
        try:
            self.factor = cho_factor(k+self.jitter*np.eye(self.count), lower=True)
            self.alpha = cho_solve(self.factor, self.y)
        except np.linalg.LinAlgError:
            self.failure = "GP_FACTORIZATION_FAILED_USE_EXPLORATION"

    def expected_improvement(self, x):
        x = np.asarray(x)
        if self.factor is None:
            return np.ones(len(x))
        k = self._kernel(x, self.x)
        mean = k@self.alpha
        variance = np.maximum(0., 1.-np.sum(k*cho_solve(self.factor, k.T).T, axis=1))
        sigma = np.sqrt(np.maximum(variance, 1e-16))
        delta = mean-self.best
        z = delta/sigma
        return delta*ndtr(z)+sigma*np.exp(-.5*z*z)/math.sqrt(2*math.pi)

    def report(self):
        return dict(training_observations=self.count, excluded_partial=self.excluded_partial,
                    target="certified_fixed_route_V_only", interval_regression=False,
                    length_scale=self.length_scale, failure=self.failure)
