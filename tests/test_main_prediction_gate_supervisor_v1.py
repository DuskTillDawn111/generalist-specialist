"""CPU tests of main-gate launch refusal and owned-process budget enforcement."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import main_prediction_gate_supervisor_v1 as m


def fixture():
    trace = [dict(request_id=str(i)) for i in range(57)]
    plan = dict(trace=trace, run_id='fresh', queue_sample_offsets_ns=[0, 1])
    predictions = dict(cells=[dict(candidate_id=c, run_id='fresh',
        requests=[dict(request_id=r['request_id'], service_ms=1., queue_ms=0., e2e_ms=1.) for r in trace],
        terms={k:1. for k in 'QTLPME'}, queue_depth_samples=[dict(offset_ns=i, depth=0) for i in (0, 1)])
        for c in m.CANDIDATES])
    return plan, predictions


class SupervisorTests(unittest.TestCase):
    def test_exact_seal_review_and_frozen_supervisor_closure_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan, predictions = fixture()
            for name, value in [('plan', plan), ('predictions', predictions), ('protocol', {}), ('source', {})]:
                m.save(root/(name+'.json'), value)
            source = m.receipt(root/'source.json')
            prep = dict(plan=m.receipt(root/'plan.json'))
            m.save(root/'preparation.json', prep)
            review = dict(schema='gsdeploy.projected_handler_artifact_binding_independent_review.v48',
                          status='PASS', issues=[], preparation=m.receipt(root/'preparation.json'),
                          complete_four_forecasts_reprojected=True)
            m.save(root/'review.json', review)
            production = dict(schema='gsdeploy.projected_handler_artifact_binding_campaign.v48',
                status='ARTIFACT_BOUND_PREDICTIONS_SEALED_REQUIRES_POSTSEAL_GPU_VALIDATION',
                protocol=m.receipt(root/'protocol.json'), predictions=m.receipt(root/'predictions.json'),
                preparation=m.receipt(root/'preparation.json'), independent_review=m.receipt(root/'review.json'),
                original_v5_precollection_bindings_passed=True, all_GPU_must_start_after_seal=True, GPU_started=False)
            m.save(root/'production.json', production)
            arguments = [root/(k+'.json') for k in ('plan', 'protocol', 'predictions', 'production')]
            protocol = dict(sources=[source])
            with patch.object(m, '_bindings', return_value=(protocol, predictions, plan)), \
                    patch.object(m, '_preserved_plan'), patch.object(m, 'implementation_sources', return_value=[source]):
                self.assertEqual(m.prerequisites(*arguments), (plan, [source]))
                protocol['sources'] = []
                with self.assertRaisesRegex(ValueError, 'closure'):
                    m.prerequisites(*arguments)
                protocol['sources'] = [source]
                production['status'] = 'UNKNOWN'
                (root/'production.json').write_text(json.dumps(production))
                with self.assertRaisesRegex(ValueError, 'seal receipt'):
                    m.prerequisites(*arguments)

    def test_complete_all_four_and_any_unknown_refused(self):
        plan, predictions = fixture()
        m.complete_cells(predictions, plan)
        for mutation in ('missing_candidate', 'missing_request', 'unknown_service', 'unknown_term', 'missing_queue', 'nan'):
            value = copy.deepcopy(predictions)
            if mutation == 'missing_candidate': value['cells'].pop()
            elif mutation == 'missing_request': value['cells'][0]['requests'].pop()
            elif mutation == 'unknown_service': value['cells'][1]['requests'][0]['service_ms'] = None
            elif mutation == 'unknown_term': value['cells'][2]['terms']['M'] = None
            elif mutation == 'missing_queue': value['cells'][3]['queue_depth_samples'].pop()
            else: value['cells'][0]['terms']['Q'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                m.complete_cells(value, plan)

    def test_actual_owned_group_timeout_and_disk_margin_cleanup(self):
        for field, value, expected in [('total_timeout_seconds', -1., 'TIMEOUT'),
                                       ('free_disk_reserve_bytes', 2**80, 'DISK_MARGIN'),
                                       ('incremental_storage_scenario_bytes', -1, 'STORAGE_SCENARIO_EXCEEDED')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                budget = dict(m.BUDGET, cleanup_grace_seconds=.1, watchdog_interval_seconds=.01)
                budget[field] = value
                try:
                    # Keep the failure cause deterministic on small /tmp mounts.
                    with patch.object(m.shutil, 'disk_usage', return_value=type('Disk', (), dict(free=20*2**30))()):
                        result = m.wait_bounded(process, directory, budget)
                    self.assertEqual(result['status'], expected)
                    self.assertEqual(result['cleanup']['live_survivors'], [])
                    self.assertIsNotNone(process.poll())
                finally:
                    m.cleanup_group(process, .1)

    def test_manifest_budget_and_low_disk_reject_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = {}
            for key in ('plan', 'protocol', 'predictions', 'production'):
                m.save(root/(key+'.json'), {})
                refs[key] = m.receipt(root/(key+'.json'))
            manifest = dict(schema=m.SCHEMA, status='CPU_PREPARED_REQUIRES_EXPLICIT_EXECUTE',
                            budget=m.BUDGET, candidates=list(m.CANDIDATES), sources=[], run_id='fresh',
                            preflight_free_bytes_required=18*2**30, disk_abort_floor_bytes=13*2**30, **refs)
            m.save(root/'manifest.json', manifest)
            with patch.object(m, 'prerequisites', return_value=(dict(run_id='fresh'), [])), \
                    patch.object(m.shutil, 'disk_usage', return_value=type('Disk', (), dict(free=18*2**30-1))()), \
                    patch.object(m.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, '18GiB'):
                    m.execute(root/'manifest.json', m.CANDIDATES[0], root/'candidate')
                launch.assert_not_called()
                self.assertFalse((root/'candidate').exists())
            manifest['budget'] = dict(m.BUDGET, total_timeout_seconds=9999.)
            m.save(root/'bad.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Exact prepared'):
                m.execute(root/'bad.json', m.CANDIDATES[0], root/'bad-candidate')

    def test_nonzero_child_preserved_and_source_drift_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); refs = {}
            for key in ('plan', 'protocol', 'predictions', 'production'):
                m.save(root/(key+'.json'), {}); refs[key] = m.receipt(root/(key+'.json'))
            manifest = dict(schema=m.SCHEMA, status='CPU_PREPARED_REQUIRES_EXPLICIT_EXECUTE',
                            budget=m.BUDGET, candidates=list(m.CANDIDATES), sources=[], run_id='fresh',
                            preflight_free_bytes_required=18*2**30, disk_abort_floor_bytes=13*2**30, **refs)
            m.save(root/'manifest.json', manifest)
            def child(*args, **kwargs):
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(args[0][3], 'execute')
                m.save(root/'candidate/collection-failed-evidence.json', dict(retained=True))
                return subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'],
                                        stdout=kwargs['stdout'], stderr=kwargs['stderr'], start_new_session=True)
            actual_popen = subprocess.Popen
            def safe_child(*args, **kwargs):
                with patch.object(m.subprocess, 'Popen', actual_popen):
                    return child(*args, **kwargs)
            with patch.object(m, 'prerequisites', side_effect=[(dict(run_id='fresh'), []), ValueError('source drift')]), \
                    patch.object(m.shutil, 'disk_usage', return_value=type('Disk', (), dict(free=20*2**30))()), \
                    patch.object(m.subprocess, 'Popen', side_effect=safe_child):
                result = m.execute(root/'manifest.json', m.CANDIDATES[0], root/'candidate')
            self.assertEqual(result['watchdog']['return_code'], 7)
            self.assertEqual(result['status'], 'FAILED_NO_RETRY')
            self.assertEqual(result['final_source_audit'], 'FAIL')
            self.assertTrue((root/'candidate/collection-failed-evidence.json').is_file())
            self.assertTrue((root/'candidate/execution.json').is_file())
            with self.assertRaises(FileExistsError), patch.object(m, 'prerequisites', return_value=(dict(run_id='fresh'), [])):
                m.execute(root/'manifest.json', m.CANDIDATES[0], root/'candidate')


if __name__ == '__main__':
    unittest.main()
