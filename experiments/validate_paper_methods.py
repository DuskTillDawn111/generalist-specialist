"""Small, CPU-only integration demonstration; never a GPU benefit benchmark."""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.io import save
from gsdeploy.paper_forecast import _session, session
from gsdeploy.paper_algorithms import J3OInspiredOptimizer
from gsdeploy.system import DeploymentSystem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    c = SystemConfig.load(ROOT / "configs/paper_fifo_cpu_fixture.json")
    results = []
    started = time.perf_counter()
    for algorithm in ("milp", "dfs", "j3o_inspired"):
        _session.cache_clear()  # Each demonstration starts with an empty cache.
        s = session(c.catalog)
        at = time.perf_counter()
        if algorithm == "j3o_inspired":
            result = J3OInspiredOptimizer(c, max_evaluations=24, time_limit_seconds=3.,
                                         seed_seconds=1., max_rounds=1).solve()
        else:
            result = DeploymentSystem(replace(c, solver=algorithm)).plan()
        elapsed = time.perf_counter() - at
        winner = result["winner"]
        save(output / (algorithm + ".json"), result)
        results.append(dict(algorithm=algorithm, elapsed_seconds=elapsed,
            status=result["status"], U=winner["score"] if winner else None,
            deployment=winner["deployment"] if winner else None,
            routing=winner["routing"] if winner else None,
            scenario_evaluation_requests=s.calls, event_engine_calls=s.engine_calls,
            cache_hits=s.hits, cache_bytes=s.cache_bytes,
            milp_calls=result.get("milp_calls", result.get("metrics", {}).get("milp_calls", 0))))
    comparisons = []
    for g_rate in (0., 1., 2.):
        p = session(c.catalog).evaluate({"g": 1, "s": 1},
            {"text": {"g": g_rate, "s": 2. - g_rate}}, c.workload, c.gpu_memory_mib,
            c.latency_reference_ms, c.weights, trace=((0., "text"), (0., "text")),
            **DeploymentSystem(c).context())
        comparisons.append(dict(g_qps=g_rate, U=p["U"], safe_memory_mib=p["safe_memory_mib"],
            per_task=p["per_task"], mapping=p["request_mapping"],
            finishes=p["forecast"]["finishes"], lifecycle=p["forecast"]["lifecycle_events"]))
    files = ["gsdeploy/paper_forecast.py", "gsdeploy/paper_algorithms.py", "gsdeploy/config.py",
             "gsdeploy/coupled_mode.py", "gsdeploy/prediction.py", "gsdeploy/performance.py",
             "gsdeploy/optimizer.py", "gsdeploy/milp.py", "gsdeploy/system.py",
             "tests/test_paper_integration.py", "configs/paper_fifo_cpu_fixture.json",
             "experiments/paper_methods.py", "experiments/validate_paper_methods.py"]
    source_manifest = [{"path": f, "sha256": hashlib.sha256((ROOT / f).read_bytes()).hexdigest()} for f in files]
    save(output / "source_manifest.json", source_manifest)
    save(output / "controlled_fifo.json", comparisons)
    report = dict(schema="gsdeploy.paper_integration_smoke.v1", CPU_TEST_ONLY=True,
        GPU_started=False, native_measurements_generated=False,
        physical_accuracy_or_gain_claim=False, mathematical_fixture_domain_only=True,
        domain=dict(tasks=1, models=2, replica_options=[0, 1], supported_deployments=3,
                    nominal_requests=2, route_grid=2, complete_grid_candidates=5),
        timing_scope="tiny software integration only; no scalability or speedup inference",
        call_definitions=dict(scenario_evaluation_requests="uncached nominal or explicit stress query; unsupported queries included",
                              event_engine_calls="actual FullHorizonModel.forecast invocations",
                              milp_calls="HiGHS mixed-integer master solves; excludes continuous LP proposals"),
        results=results, elapsed_seconds=time.perf_counter() - started)
    save(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if all(r["U"] is not None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
