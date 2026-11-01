"""Run the portable CPU regression suite used by CI, without local artifacts."""

import argparse
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]

# Explicit selection keeps evidence-dependent research tests available without
# making a clean checkout depend on untracked artifacts or model environments.
MODULES = (
    "test_evidence",
    "test_milp",
    "test_milp_witness",
    "test_replicas",
    "test_runtime",
    "test_observed_state_v2",
    "test_search_contract",
    "test_system",
    "test_quality_utility",
    "test_forecast_quality_totals_v2",
    "test_calibrated_surrogate",
    "test_calibrated_replay_v2",
    "test_calibrated_search_bounds_v6",
    "test_static_dfs_v6",
    "test_static_milp_v6",
    "test_resource_control",
    "test_resource_study",
    "test_study",
    "test_study_reporting",
    "test_static_study_workflow",
    "test_compact_protocol",
    "test_compact_protocol_repairs",
    "test_compact_admission",
    "test_compact_memory_barrier",
    "test_compact_experiments",
    "test_compact_gpu_queue",
    "test_search_session",
    "test_system_pool_gate",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List selected test modules")
    args = parser.parse_args()
    names = [f"tests.{name}" for name in MODULES]
    if args.list:
        print("\n".join(names))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
