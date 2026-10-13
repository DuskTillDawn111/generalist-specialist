"""Offline checks of Module 4B's evidence audit and mismatch reporting."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deployment_core.models import load_model_catalog
from deployment_core.workload import Workload
from deployment_search.campaign_runtime import CampaignRuntimeEvaluator
from deployment_search.measurement_runtime import aggregate_repetitions
from deployment_search.result import jsonable
from deployment_search.routing_real_pilot import (CONFIG, OBJECTIVE, decision_audit,
                                                  enrich_trace, ROOT)
from deployment_search.routing_search import GuidedRoutingSearch
from deployment_search.runtime_cache import file_hash


class RoutingRealPilotTests(unittest.TestCase):
    def test_runtime_correction_and_tampered_decision_audit(self):
        catalog = load_model_catalog()
        doc = json.loads((ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json').read_text())
        workload = Workload(next(w['arrival_rates_rps'] for w in doc['workloads'] if w['name'] == 'W1_balanced'))
        deployment = {m: int(m != 'distilbert') for m in catalog}
        with tempfile.TemporaryDirectory() as directory:
            identity_adapter = CampaignRuntimeEvaluator(directory, catalog, {}, {}, 'offline')
            records = []

            class OfflineEvidence:
                def evaluate(self, x, w, r):
                    # Deliberately artificial surface for the reporting test;
                    # never included in any pilot GPU evidence or report.
                    mean = 100. + 100. * r['ocr', 'qwen'] - 500. * r['image', 'qwen']
                    if len(records) >= 3:
                        mean = 200.
                    raw = Path(directory) / f'raw_{len(records)}.json'
                    raw.write_text('{}\n')
                    reps = [dict(stability_state_v2='steady', weighted_mean_e2e=mean,
                        weighted_p95_e2e=mean * 2, throughput=2.35, completion_ratio=1.,
                        per_task_mean={k: mean for k in w.active_tasks},
                        per_task_p95={k: mean * 2 for k in w.active_tasks},
                        worker_latency={k: {'mean_worker_latency_ms': mean} for k in w.active_tasks},
                        queue_metrics={}, raw_result_path=str(raw), raw_sha256=file_hash(raw),
                        run_status='completed') for _ in range(3)]
                    identity = identity_adapter.identity(x, w, r)
                    from deployment_search.runtime_cache import digest
                    value = aggregate_repetitions(reps, digest(identity), raw)
                    records.append(jsonable(dict(deployment=x, routing=r, full_identity=identity, runtime=value)))
                    return value

            result = GuidedRoutingSearch(OfflineEvidence(), predictor=None,
                runtime_objective=OBJECTIVE).search(deployment, workload, catalog, CONFIG)
            trace = enrich_trace(result, workload, catalog)
            self.assertTrue(trace['analytical_runtime_mismatch'])
            self.assertTrue(trace['runtime_feedback_correction'])
            first = sorted((c for c in trace['iterations'][0]['candidates'] if c['selected_top_b']), key=lambda c: c['rank'])
            self.assertEqual(first[0]['task'], 'ocr')
            self.assertEqual(first[0]['status'], 'ROLLBACK')
            self.assertEqual(first[1]['task'], 'image')
            self.assertEqual(first[1]['status'], 'ACCEPTED')
            audit = decision_audit(trace, records, {'run_attempts': []}, workload, catalog, 0)
            self.assertTrue(audit['passed'], audit['errors'])
            first[0]['status'] = 'ACCEPTED'
            first[1]['status'] = 'ROLLBACK'
            audit = decision_audit(trace, records, {'run_attempts': []}, workload, catalog, 0)
            self.assertFalse(audit['passed'])
            self.assertIn('accept/rollback differs from frozen RuntimeObjective', audit['errors'])

    def test_runtime_gate_and_stability_priority(self):
        from deployment_search.runtime import RuntimeEvaluation
        def value(mean, state='steady'):
            return RuntimeEvaluation(state, mean, mean * 2, 1., 1.)
        self.assertFalse(OBJECTIVE.compare(value(96.), value(100.), 0, 0)[0])
        self.assertTrue(OBJECTIVE.compare(value(95.), value(100.), 0, 0)[0])
        self.assertTrue(OBJECTIVE.compare(value(120.), value(100., 'recoverable_congested'), 0, 0)[0])
        self.assertFalse(OBJECTIVE.compare(value(50., 'unstable'), value(100.), 0, 0)[0])

    def test_recording_immutable_workload_is_serializable(self):
        from deployment_search.routing_real_pilot import RecordingEvaluator
        from deployment_search.runtime import RuntimeEvaluation
        workload = Workload({'text': 1.})
        deployment = {'qwen': 1}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = out / 'source_manifest.json'
            manifest.write_text('{}\n')
            adapter = SimpleNamespace(evaluate=lambda *args: RuntimeEvaluation('steady', 100., 200., 1., 1.),
                identity=lambda *args: {}, cache=SimpleNamespace(path=manifest))
            with patch('deployment_search.routing_real_pilot.OUT', out):
                RecordingEvaluator(adapter, deployment, workload).evaluate(deployment, workload, {('text', 'qwen'): 1.})
            record = json.loads((out / 'runtime_evaluations.json').read_text())[0]
            self.assertEqual(record['workload'], dict(workload.arrival_rate))


if __name__ == '__main__':
    unittest.main()
