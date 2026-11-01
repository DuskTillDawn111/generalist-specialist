"""CPU resource contracts: validation, isolated child environments and provenance."""

import copy
import json
import os
from pathlib import Path
import queue
import sys
import unittest
from unittest.mock import patch

from gsdeploy.energy import energy_key
from gsdeploy.resources import (RESOURCE_SCHEMA, identity, validate_resources,
                                prepare_registry, worker_environment)
from gsdeploy.runtime.transport import Worker
from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig


def resources(mode="shared", aliases=("s#0", "g#0")):
    value = {"schema": RESOURCE_SCHEMA, "mode": mode, "device": "0",
             "allocations": {a: {} for a in aliases}}
    if mode == "mps_cap":
        value["mps_session"] = {"pipe_directory": "/tmp/gs-resource-test/pipe",
                                "log_directory": "/tmp/gs-resource-test/log"}
        value["allocations"] = {a: {"active_thread_percentage": 80} for a in aliases}
    if mode in {"mps_static_sm", "green_context"}:
        value["sm_capacity"] = 8
        value["allocations"] = {a: {"sm_ids": [i * 2, i * 2 + 1]} for i, a in enumerate(aliases)}
    if mode == "request_admission":
        value["allocations"] = {a: {"max_inflight": 1} for a in aliases}
        value["groups"] = {"all": {"aliases": list(aliases), "max_active": 1}}
    return value


class ResourceConfigurationTests(unittest.TestCase):
    def test_all_modes_validate_without_gpu(self):
        with patch("subprocess.Popen", side_effect=AssertionError("No process allowed")), \
                patch("subprocess.run", side_effect=AssertionError("No GPU probe allowed")):
            for mode in ("shared", "mps_cap", "mps_static_sm", "green_context", "request_admission"):
                document = resources(mode)
                self.assertEqual(validate_resources(document, ("s#0", "g#0"))["mode"], mode)

    def test_every_resident_including_idle_must_be_covered(self):
        for change in (lambda d: d["allocations"].pop("g#0"),
                       lambda d: d["allocations"].update(idle={})):
            document = resources(); change(document)
            with self.assertRaisesRegex(ValueError, "exactly"):
                validate_resources(document, ("s#0", "g#0"))

    def test_percentage_is_not_an_exclusive_additive_budget(self):
        # 80+80 is valid as two ceilings; it is not an allocation of 160% SMs.
        document = resources("mps_cap")
        self.assertEqual(validate_resources(document, document["allocations"]), document)
        for value in (True, 0, -1, 101, float("nan"), float("inf"), "30"):
            with self.subTest(value=value):
                changed = copy.deepcopy(document)
                changed["allocations"]["g#0"]["active_thread_percentage"] = value
                with self.assertRaises(ValueError):
                    validate_resources(changed, document["allocations"])

    def test_sm_overlap_capacity_and_duplicates_rejected(self):
        for ids in ([0, 2], [8], [True], [1.5], [], [4, 4], [-1]):
            document = resources("mps_static_sm")
            document["allocations"]["g#0"]["sm_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_resources(document, document["allocations"])

    def test_unknown_keys_and_modes_do_not_silently_pass(self):
        for key, value in (("mode", "mig"), ("schema", "v0"), ("device", "0,1"),
                           ("device", "-1"), ("device", " 0 "), ("unknown", True)):
            document = resources(); document[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_resources(document, document["allocations"])

    def test_mode_specific_fields_rejected(self):
        for document in (dict(resources(), sm_capacity=8),
                         dict(resources(), mps_session={}),
                         dict(resources("mps_cap"), mps_session={"pipe_directory": "relative", "log_directory": "/log"})):
            with self.assertRaises(ValueError):
                validate_resources(document, document["allocations"])

    def test_resource_identity_and_energy_key_separate_modes(self):
        a, b = resources(), resources("mps_cap")
        self.assertNotEqual(identity(a), identity(b))
        args = ({"s": 1}, {"text": {"s": 1}}, {"text": 1}, 10, 1, "poisson")
        old = energy_key(*args)
        self.assertEqual(old, energy_key(*args, resource_configuration=None))
        self.assertNotEqual(old, energy_key(*args, resource_configuration=a))
        self.assertNotEqual(energy_key(*args, resource_configuration=a),
                            energy_key(*args, resource_configuration=b))


class WorkerBindingTests(unittest.TestCase):
    def registry(self):
        return {a: {"command": [sys.executable, "-c", "pass"], "env": {"OTHER": "kept"}}
                for a in ("s#0", "g#0")}

    def test_binding_does_not_mutate_original_and_has_no_effect_claim(self):
        original = self.registry(); before = copy.deepcopy(original)
        bound = prepare_registry(original, resources("mps_cap"))
        self.assertEqual(original, before)
        self.assertEqual(bound["s#0"]["env"]["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "80")
        self.assertFalse(bound["s#0"]["resource_binding"]["effective_isolation_verified"])
        with self.assertRaisesRegex(ValueError, "blocked"):
            worker_environment(bound["s#0"], {})

    def test_all_mps_controls_removed_from_explicit_shared_child(self):
        bound = prepare_registry(self.registry(), resources())
        parent = {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "5", "CUDA_MPS_PIPE_DIRECTORY": "/hidden",
                  "CUDA_MPS_FUTURE_CONTROL": "also-hidden",
                  "KEEP": "yes", "CUDA_VISIBLE_DEVICES": "4"}
        original = dict(parent)
        env = worker_environment(bound["s#0"], parent)
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", env)
        self.assertNotIn("CUDA_MPS_PIPE_DIRECTORY", env)
        self.assertNotIn("CUDA_MPS_FUTURE_CONTROL", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["KEEP"], "yes")
        self.assertEqual(parent, original)

    def test_legacy_environment_preserved_without_explicit_resources(self):
        env = worker_environment({"env": {"X": "child"}}, {"X": "parent", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"})
        self.assertEqual(env["X"], "child")
        self.assertEqual(env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"], "40")

    def test_device_and_existing_mps_configuration_conflicts(self):
        for env in ({"CUDA_VISIBLE_DEVICES": "1"}, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "30"}):
            registry = self.registry(); registry["s#0"]["env"].update(env)
            with self.assertRaises(ValueError):
                prepare_registry(registry, resources())

    def test_changed_prepared_environment_is_rejected(self):
        bound = prepare_registry(self.registry(), resources())
        bound["s#0"]["env"]["CUDA_VISIBLE_DEVICES"] = "1"
        with self.assertRaisesRegex(ValueError, "changed"):
            worker_environment(bound["s#0"], {})

    def test_clearing_blocker_does_not_enable_controlled_launch(self):
        bound = prepare_registry(self.registry(), resources("mps_cap"))
        bound["s#0"]["resource_binding"]["launch_blockers"] = []
        with self.assertRaisesRegex(ValueError, "verified platform adapter"):
            worker_environment(bound["s#0"], {})

    def test_worker_rejects_controls_before_popen(self):
        bound = prepare_registry(self.registry(), resources("mps_cap"))
        with patch("subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "blocked"):
                Worker("s#0", bound["s#0"], queue.Queue()).start()
            popen.assert_not_called()

    def test_whole_deployment_rejects_invalid_binding_before_first_worker(self):
        registry = prepare_registry(self.registry(), resources())
        registry["g#0"] = prepare_registry(self.registry(), resources("mps_cap"))["g#0"]
        config = MeasurementConfig(deployment={a: 1 for a in registry}, workload_rps={"text": 1},
                                   routing_rps={"text": {"s#0": 1}})
        runtime = DeploymentRuntime(config, registry)
        with patch("gsdeploy.runtime.deployment.ObservedWorker.start") as start:
            with self.assertRaisesRegex(ValueError, "blocked"):
                runtime.start()
            start.assert_not_called()

    def test_real_cpu_subprocess_observes_sanitized_environment(self):
        registry = self.registry()
        registry["s#0"]["command"] = [sys.executable, "-c",
            "import os,json; print('READY|'+json.dumps({'cap':os.getenv('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'),'device':os.getenv('CUDA_VISIBLE_DEVICES')}))"]
        bound = prepare_registry(registry, resources()); events = queue.Queue()
        worker = Worker("s#0", bound["s#0"], events)
        try:
            with patch.dict(os.environ, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "5"}):
                worker.start()
            name, kind, data = events.get(timeout=5)
            self.assertEqual((name, kind), ("s#0", "READY"))
            self.assertEqual(data, {"cap": None, "device": "0"})
            self.assertEqual(worker.process.wait(timeout=5), 0)
        finally:
            if worker.process:
                if worker.process.poll() is None:
                    worker.process.kill(); worker.process.wait()
                worker.process.stdin.close(); worker.process.stdout.close()


if __name__ == "__main__":
    unittest.main()
