"""Independent identity regressions from the six-term workflow review."""
import copy
import unittest

from gsdeploy.energy import energy_key
from gsdeploy.performance import profile_key
from tests.test_static_dfs_v6 import six_term_config


class StaticMeasurementIdentityReviewTests(unittest.TestCase):
    def setUp(self):
        self.config = six_term_config()
        self.catalog = self.config.catalog
        self.catalog["quality"]["input_order"] = {"text": ["input-a", "input-b"]}
        for model in ("g", "s"):
            self.catalog["quality"]["paths"][model]["text"]["protocol_sha256"] = model + "-protocol"

    def key(self, catalog):
        c = self.config
        return energy_key({"s": 1}, {"text": {"s": 1.0}}, c.workload,
                          c.horizon_seconds, c.random_seed, c.arrival_process, catalog=catalog)

    def test_input_and_checkpoint_protocol_changes_invalidate_measurement(self):
        baseline = self.key(self.catalog)
        mutations = (
            lambda c: c["quality"]["identity"].update(input_pool_sha256="different pool"),
            lambda c: c["quality"]["identity"].update(reference_sha256="different reference"),
            lambda c: c["quality"]["input_order"]["text"].reverse(),
            lambda c: c["quality"]["paths"]["s"]["text"].update(checkpoint_sha256="different weights"),
            lambda c: c["quality"]["paths"]["s"]["text"].update(protocol_sha256="different native decoding"),
            lambda c: c.update(measurement_context={"warmup_seconds": 31}),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.catalog)
            mutate(changed)
            self.assertNotEqual(baseline, self.key(changed))

    def test_unused_models_states_and_quality_do_not_change_same_measurement(self):
        baseline = self.key(self.catalog)
        changed = copy.deepcopy(self.catalog)
        for field in ("models", "capacity_factors", "memory_table_mib"):
            changed[field].pop("g")
        changed["quality"]["paths"].pop("g")
        self.assertEqual(baseline, self.key(changed))
        changed["capacity_factors"]["s"]["2"] = 99
        changed["memory_table_mib"]["s"]["2"] = 99
        changed["quality"]["identity"]["source"] = "relocated evidence"
        changed["quality"]["paths"]["s"]["text"]["value"] = .5
        self.assertEqual(baseline, self.key(changed))

    def test_profile_lookup_and_energy_bind_the_same_service_contract(self):
        self.catalog["performance_mode"] = "profiled"
        x, route = {"s": 1}, {"text": {"s": 1.0}}
        c = self.config
        profile_id = profile_key(self.catalog, x, route, c.workload,
                                 c.horizon_seconds, c.random_seed, c.arrival_process)
        self.catalog["colocation_profiles"] = {profile_id: {
            "kind": "synthetic_fixture", "coverage_valid": True,
            "scope": "whole_gpu_colocation", "service_time_ms": {"s": {"text": 4.0}}}}
        baseline = self.key(self.catalog)
        self.catalog["colocation_profiles"][profile_id]["service_time_ms"]["s"]["text"] = 5.0
        self.assertNotEqual(baseline, self.key(self.catalog))

    def test_common_runtime_checkpoint_map_matches_per_job_binding(self):
        self.catalog["measurement_context"] = {
            "gpu_uuid": "gpu-fixed", "runtime_checkpoint_sha256": {"g": "g-weights", "s": "s-weights"}}
        baseline = self.key(self.catalog)
        changed = copy.deepcopy(self.catalog)
        changed["measurement_context"]["runtime_checkpoint_sha256"].pop("g")
        self.assertEqual(baseline, self.key(changed))
        changed["measurement_context"]["runtime_checkpoint_sha256"]["s"] = "different-weights"
        self.assertNotEqual(baseline, self.key(changed))
        changed = copy.deepcopy(self.catalog)
        changed["measurement_context"]["gpu_uuid"] = "different-gpu"
        self.assertNotEqual(baseline, self.key(changed))

    def test_unrouted_but_primed_resident_task_protocol_is_bound(self):
        c = self.config
        x, routing = {"g": 1, "s": 1}, {"text": {"s": 1.0}}
        def key(catalog):
            return energy_key(x, routing, c.workload, c.horizon_seconds,
                              c.random_seed, c.arrival_process, catalog=catalog)
        baseline = key(self.catalog)
        changed = copy.deepcopy(self.catalog)
        changed["quality"]["paths"]["g"]["text"]["protocol_sha256"] = "different priming generation"
        self.assertNotEqual(baseline, key(changed))


if __name__ == "__main__":
    unittest.main()
