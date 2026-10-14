"""Independent small-space oracles and the audited failure counterexamples."""

import copy
import itertools
import math
import unittest
from dataclasses import replace
from gsdeploy import SystemConfig, DeploymentSystem
from gsdeploy.energy import energy_key, integrate_power
from gsdeploy.prediction import components
from gsdeploy.trace import make_trace


def config(rates=None, services=None, weights=None, horizon=10):
    rates = rates or {"text": 1.0, "image": 1.0}
    services = services or {"g": {t: 2.0 for t in rates}, "s": {t: 4.0 for t in rates}}
    catalog = {
        "safe_factor": 1.0,
        "models": {
            m: {
                "role": "generalist" if m == "g" else "specialist",
                "supported_tasks": list(s),
                "service_time_ms": s,
            }
            for m, s in services.items()
        },
        "capacity_factors": {m: {"0": 0.0, "1": 1.0} for m in services},
        "memory_table_mib": {m: {"0": 0.0, "1": 1.0} for m in services},
    }
    return SystemConfig(
        catalog,
        rates,
        100.0,
        {t: 10.0 for t in rates},
        weights or {"Q": 0.0, "T": 0.25, "L": 0.3, "P": 0.3, "M": 0.15},
        horizon_seconds=horizon,
    )


def oracle_score(c, x, r):
    # Independent implementations: seeded periodic arrivals, deficit router,
    # serial FIFO clocks, and linearly interpolated empirical quantiles.
    import random

    trace = []
    for t, q in sorted(c.workload.items()):
        phase = ((random.Random(f"{c.random_seed}:phase:{t}").random() + 1e-12) % 1) / q
        for i in range(math.ceil(q * c.horizon_seconds) + 1):
            a = phase + i / q
            if a < c.horizon_seconds:
                trace.append((a, t))
    trace.sort()
    clocks = {f"{m}#{i}": 0.0 for m, n in x.items() for i in range(n)}
    w = {
        t: {
            f"{m}#{i}": q / x[m]
            for m, q in sorted(row.items())
            if q > 0
            for i in range(x[m])
        }
        for t, row in r.items()
    }
    deficits = {t: dict.fromkeys(row, 0.0) for t, row in w.items()}
    latencies = {t: [] for t in c.workload}
    within = 0
    for a, t in trace:
        for alias, q in w[t].items():
            deficits[t][alias] += q
        alias = max(w[t], key=lambda k: deficits[t][k])
        deficits[t][alias] -= sum(w[t].values())
        m = alias.rsplit("#", 1)[0]
        svc = (
            c.catalog["models"][m]["service_time_ms"][t]
            / 1000
            * x[m]
            / c.catalog["capacity_factors"][m][str(x[m])]
        )
        finish = max(a, clocks[alias]) + svc
        clocks[alias] = finish
        within += finish <= c.horizon_seconds
        latencies[t].append((finish - a) * 1000)

    def percentile(values):
        a = sorted(values)
        p = 0.95 * (len(a) - 1)
        i = int(p)
        return a[i] if i + 1 == len(a) else a[i] + (a[i + 1] - a[i]) * (p - i)

    total = sum(c.workload.values())
    L = sum(
        c.workload[t] / total * sum(v) / len(v) / c.latency_reference_ms[t]
        for t, v in latencies.items()
    )
    P = sum(
        c.workload[t] / total * percentile(v) / c.latency_reference_ms[t]
        for t, v in latencies.items()
    )
    M = (
        c.catalog["safe_factor"]
        * sum(c.catalog["memory_table_mib"][m][str(n)] for m, n in x.items())
        / c.gpu_memory_mib
    )
    return -(
        c.weights["T"] * (1 - within / len(trace))
        + c.weights["L"] * L
        + c.weights["P"] * P
        + c.weights["M"] * M
    )


class SearchContractTests(unittest.TestCase):
    def test_joint_replicas_and_split_matches_independent_exhaustive_oracle(self):
        c = config()
        c.catalog["capacity_factors"]["g"]["2"] = 1.6
        c.catalog["memory_table_mib"]["g"]["2"] = 2.0
        scores = []
        for ng, ns in itertools.product(range(3), range(2)):
            x = {m: n for m, n in [("g", ng), ("s", ns)] if n}
            if not x:
                continue
            for choices in itertools.product((0.0, 0.5, 1.0), repeat=2):
                r = {
                    t: {m: q * f for m, f in [("g", f), ("s", 1 - f)] if f}
                    for (t, q), f in zip(c.workload.items(), choices)
                }
                try:
                    c.validate_plan(x, r)
                except ValueError:
                    continue
                scores.append(oracle_score(c, x, r))
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertTrue(result["optimality_certified"])
        self.assertAlmostEqual(result["winner"]["score"], max(scores), places=10)

    def test_joint_search_finds_required_replica_and_split(self):
        c = config(
            {"text": 1.6, "image": 0.5},
            {"g": {"text": 1000.0, "image": 1000.0}, "s": {"text": 1000.0}},
            horizon=20,
        )
        c.catalog["capacity_factors"]["g"]["2"] = 1.8
        c.catalog["memory_table_mib"]["g"]["2"] = 2.0
        self.assertIsNone(DeploymentSystem(c).plan(route_grid=1)["winner"])
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertEqual(result["winner"]["deployment"], {"g": 2, "s": 1})
        self.assertEqual(set(result["winner"]["routing"]["text"]), {"g", "s"})
        self.assertTrue(result["constraints"]["constraints_valid"])

    def test_sparse_split_conserves_cohort_and_throughput(self):
        c = config(
            {"text": 0.02}, {"g": {"text": 2.0}, "s": {"text": 2.0}}, horizon=120
        )
        system = DeploymentSystem(c)
        predictions = [
            system.predict(
                {"g": 1, "s": 1}, {"text": {"g": 0.02 * f, "s": 0.02 * (1 - f)}}
            )
            for f in (0.0, 0.5, 0.75, 1.0)
        ]
        self.assertTrue(all(p["offered_requests"] == 3 for p in predictions))
        self.assertEqual(len({p["trace_sha256"] for p in predictions}), 1)
        self.assertTrue(all(p["T_norm"] == 0 for p in predictions))
        self.assertLess(
            max(p["U"] for p in predictions) - min(p["U"] for p in predictions), 1e-10
        )

    def test_task_order_is_irrelevant(self):
        c = config(
            {"text": 8.0, "image": 8.0},
            {"g": {"text": 50.0, "image": 50.0}, "s": {"text": 53.0}},
            horizon=120,
        )
        c2 = replace(c, workload=dict(reversed(list(c.workload.items()))))
        a, b = (DeploymentSystem(item).plan(route_grid=2) for item in (c, c2))
        self.assertEqual(a["winner"]["deployment"], b["winner"]["deployment"])
        self.assertEqual(a["winner"]["routing"], b["winner"]["routing"])
        self.assertAlmostEqual(a["winner"]["score"], b["winner"]["score"], places=12)
        self.assertAlmostEqual(
            a["winner"]["score"],
            oracle_score(c, a["winner"]["deployment"], a["winner"]["routing"]),
            places=10,
        )

    def test_nested_grids_and_non_grid_capacity_boundary(self):
        c = config(
            {"text": 1.6}, {"g": {"text": 1000.0}, "s": {"text": 1000.0}}, horizon=20
        )
        for K in (2, 3, 4, 5, 6):
            result = DeploymentSystem(c).plan(route_grid=K)
            self.assertEqual(result["winner"]["deployment"], {"g": 1, "s": 1})
            self.assertAlmostEqual(result["winner"]["routing"]["text"]["g"], 0.8)
        c = replace(c, workload={"text": 1.3})
        c.catalog["capacity_factors"]["s"]["1"] = 0.7
        result = DeploymentSystem(c).plan(route_grid=2)
        self.assertIsNotNone(result["winner"])
        self.assertAlmostEqual(sum(result["winner"]["routing"]["text"].values()), 1.3)
        self.assertLessEqual(max(result["constraints"]["rho"].values()), 0.8 + 1e-8)

    def test_branch_bound_prunes_and_budget_is_honest(self):
        c = config()
        result = DeploymentSystem(c).plan(deployment={"g": 1, "s": 1}, route_grid=10)
        self.assertGreater(result["metrics"]["bound_pruned"], 0)
        self.assertLess(result["metrics"]["full_prediction_calls"], 121)
        limited = DeploymentSystem(replace(c, search_max_states=1)).plan(route_grid=10)
        self.assertFalse(limited["search_completed"])
        self.assertFalse(limited["optimality_certified"])
        self.assertIn("BUDGET_LIMIT", limited["status"])

    def test_robust_policy_rejects_nominal_overload(self):
        c = config(
            {"text": 8.0, "image": 8.0},
            {"g": {"text": 50.0, "image": 50.0}},
            horizon=10,
        )
        self.assertIsNotNone(DeploymentSystem(c).plan()["winner"])
        self.assertIsNone(
            DeploymentSystem(replace(c, search_policy="robust")).plan()["winner"]
        )

    def test_five_terms_missing_energy_and_whole_plan_profile(self):
        c = config(weights={"Q": 0.0, **{k: 0.2 for k in "TLPME"}})
        system = DeploymentSystem(c)
        missing = system.plan()
        self.assertEqual(missing["status"], "DIAGNOSTIC_ONLY_NO_ENERGY")
        self.assertIsNone(missing["winner"]["prediction"]["U"])
        self.assertFalse(missing["optimality_certified"])
        x = {"g": 1}
        r = {t: {"g": q} for t, q in c.workload.items()}
        key = energy_key(
            x,
            r,
            c.workload,
            c.horizon_seconds,
            c.random_seed,
            c.arrival_process,
            catalog=c.catalog,
        )
        c.catalog["energy_profiles"] = {
            key: {
                "kind": "synthetic_fixture",
                "scope": "whole_gpu_board",
                "coverage_valid": True,
                "run_joules": 600.0,
                "startup_joules": 300.0,
            }
        }
        p = DeploymentSystem(c).predict(x, r)
        self.assertAlmostEqual(p["E_norm"], 0.3)
        self.assertAlmostEqual(p["U"], -0.2 * sum(p[f"{k}_norm"] for k in "TLPME"))
        self.assertIsNone(DeploymentSystem(c).predict({"g": 1, "s": 1}, r)["U"])
        stress = DeploymentSystem(c).predict(x, r)
        self.assertIsNone(stress["scenario_U_low"])

    def test_nonadditive_energy_changes_joint_deployment(self):
        c = config(weights={"Q": 0.0, **{k: 0.2 for k in "TLPME"}})
        profiles = {}
        for x in [{"g": 1}, {"s": 1}, {"g": 1, "s": 1}]:
            for assignment in itertools.product(sorted(x), repeat=len(c.workload)):
                r = {t: {m: q} for (t, q), m in zip(c.workload.items(), assignment)}
                key = energy_key(
                    x,
                    r,
                    c.workload,
                    c.horizon_seconds,
                    c.random_seed,
                    c.arrival_process,
                    catalog=c.catalog,
                )
                profiles[key] = {
                    "kind": "synthetic_fixture",
                    "scope": "whole_gpu_board",
                    "coverage_valid": True,
                    "startup_joules": 0.0,
                    "run_joules": 1.0 if len(x) == 2 else (9000.0 if "g" in x else 3.0),
                }
        c.catalog["energy_profiles"] = profiles
        result = DeploymentSystem(c).plan()
        self.assertTrue(result["optimality_certified"])
        self.assertEqual(result["winner"]["deployment"], {"g": 1, "s": 1})
        self.assertEqual(
            {next(iter(row)) for row in result["winner"]["routing"].values()}, {"g"}
        )
        # The fixture intentionally makes an idle extra model reduce whole-board
        # energy. Local per-model Pareto pruning would incorrectly delete it.
        self.assertIsNotNone(result["winner"]["prediction"]["U"])

    def test_power_integration_requires_coverage_and_accepts_no_negative_energy(self):
        samples = [
            {"time_ns": t * 10**9, "watts": w}
            for t, w in [(0, 100.0), (1, 200.0), (2, 100.0)]
        ]
        self.assertAlmostEqual(integrate_power(samples, 500000000, 1500000000), 175.0)
        self.assertIsNone(integrate_power(samples, -1, 10**9))
        self.assertIsNone(integrate_power(samples, 0, 3 * 10**9))
        self.assertIsNone(
            integrate_power(
                [samples[0], samples[-1]], 0, 2 * 10**9, max_gap_seconds=1.0
            )
        )

    def test_poisson_cohort_is_route_independent_and_low_observations_explicit(self):
        c = replace(config(), arrival_process="poisson")
        system = DeploymentSystem(c)
        a = system.predict({"g": 1}, {t: {"g": q} for t, q in c.workload.items()})
        b = system.predict(
            {"g": 1, "s": 1},
            {t: {"g": q / 2, "s": q / 2} for t, q in c.workload.items()},
        )
        self.assertEqual(a["trace_sha256"], b["trace_sha256"])
        self.assertEqual(
            a["offered_requests"], len(make_trace(c.workload, 10, 42, "poisson"))
        )
        sparse = config({"text": 1e-8}, {"g": {"text": 2.0}})
        self.assertEqual(
            DeploymentSystem(sparse).plan()["status"], "INSUFFICIENT_COHORT"
        )
