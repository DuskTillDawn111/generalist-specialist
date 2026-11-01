"""Gate seals derived from real CPU simulator execution, not hand-filled timing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecasting.prediction_gate import DATA_SCHEMA, freeze_gate_protocol
from tools.predictor_validation_20261027 import forecast_gate_v2 as producer
from tests.test_initial_handoff_forecast_v2 import cli_request, life, MS
from tests.test_full_horizon_forecast import resources


class ForecastGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        request = cli_request()
        request['resources'] = resources(False)
        request['execution_contract'] = 'strict_lifecycle_v2'
        request['lifecycle_profiles'] = life(False).document
        for key in producer.MODEL_FIELDS_FORBIDDEN:
            request.pop(key, None)
        self.bundle = dict(schema=producer.BUNDLE_SCHEMA, kind='cpu_fixture',
                           execution_context_sha256='fixture', forecast_defaults=request,
                           startup_amortization_seconds=3600., startup_prediction=dict(
                               basis='frozen_training_prediction', evidence='cpu-training',
                               energy_joules=3600., peak_memory_mib=100.))
        self.public = dict(schema=producer.INPUT_SCHEMA, forecast_inputs=dict(
            now_ns=0, phase_start_ns=0, deadline_ns=50*MS, drain_deadline_ns=100*MS,
            future_workload=dict(process='declared_schedule', rates={'text': 1.},
                                 input_pools={'text': ['x']}, declared_arrivals=[]),
            queues={'s': [dict(request_id=rid, task='text', pool_id='x', alias='s', arrival_ns=0)
                          for rid in ('one', 'two')]}))
        self.model = self.put('model.json', self.bundle)
        self.inputs = self.put('inputs.json', self.public)
        self.training = self.put('training.json', dict(schema=DATA_SCHEMA, split='training', records=[
            dict(run_id='training', request_id='train', pool_id='train-only', task='text')]))

    def put(self, name, value):
        path = self.base/name
        path.write_text(json.dumps(value))
        return path

    def setup_gate(self, *, source_paths=None):
        candidates = [dict(id=cid, model_id='cpu', execution_context_sha256='fixture') for cid in ('base', 'spill')]
        protocol = freeze_gate_protocol(model_paths={'cpu': self.model},
            source_paths=source_paths or [*producer.prediction_source_paths(), self.inputs],
            training_data_paths=[self.training], candidates=candidates, baseline_candidate_id='base',
            validation_domain=[dict(run_id='heldout', requests=[dict(request_id=rid, pool_id='x', task='text')
                                                              for rid in ('one', 'two')])],
            thresholds={}, created_unix_ns=1, weights={k: 1/6 for k in 'QTLPME'},
            queue_sample_offsets_ns=[0, 5*MS, 11*MS])
        protocol_path = self.put('protocol.json', protocol)
        jobs = [dict(run_id='heldout', candidate_id=c['id'], public_inputs_path=str(self.inputs)) for c in candidates]
        return protocol_path, jobs

    def test_actual_simulator_handler_and_receipt_boundaries_with_startup_cost(self):
        protocol, jobs = self.setup_gate()
        result = producer.run_and_seal(protocol, jobs, self.base/'output', allow_cpu_fixture=True)
        sealed = json.loads(Path(result['predictions']['path']).read_text())
        cell = sealed['cells'][0]
        self.assertEqual([(r['service_ms'], r['queue_ms'], r['e2e_ms']) for r in cell['requests']],
                         [(10., 0., 14.), (10., 12., 26.)])
        self.assertEqual([s['depth'] for s in cell['queue_depth_samples']], [1, 1, 1])
        forecast = json.loads((self.base/'output/forecast-0000.json').read_text())
        self.assertAlmostEqual(cell['terms']['E'], forecast['metrics']['terms']['E']+.01)
        self.assertFalse(result['GPU_started'])
        self.assertFalse(result['accuracy_established'])
        with self.assertRaises(FileExistsError):
            producer.run_and_seal(protocol, jobs, self.base/'output', allow_cpu_fixture=True)

    def test_all_jobs_and_frozen_actual_predictor_sources_required(self):
        protocol, jobs = self.setup_gate()
        with self.assertRaisesRegex(ValueError, 'Every predeclared'):
            producer.run_and_seal(protocol, jobs[:1], self.base/'partial', allow_cpu_fixture=True)
        protocol, jobs = self.setup_gate(source_paths=[Path(producer.__file__), self.inputs])
        with self.assertRaisesRegex(ValueError, 'source closure'):
            producer.run_and_seal(protocol, jobs, self.base/'unfrozen', allow_cpu_fixture=True)

    def test_unknown_service_keeps_all_requests_and_unknown_terms(self):
        for profile in self.bundle['forecast_defaults']['profiles']['profiles']:
            profile['pool_id'] = 'different'
        self.put('model.json', self.bundle)
        protocol, jobs = self.setup_gate()
        report = producer.run_and_seal(protocol, jobs, self.base/'unknown', allow_cpu_fixture=True)
        cells = json.loads(Path(report['predictions']['path']).read_text())['cells']
        self.assertEqual(len(cells), 2)
        self.assertEqual(len(cells[0]['requests']), 2)
        self.assertIsNone(cells[0]['requests'][0]['service_ms'])
        self.assertIsNone(cells[0]['terms'])
        self.assertIsNone(cells[0]['queue_depth_samples'][0]['depth'])

    def test_missing_startup_energy_does_not_become_zero(self):
        self.bundle['startup_prediction'].pop('energy_joules')
        self.put('model.json', self.bundle)
        protocol, jobs = self.setup_gate()
        report = producer.run_and_seal(protocol, jobs, self.base/'missing-startup', allow_cpu_fixture=True)
        cell = json.loads(Path(report['predictions']['path']).read_text())['cells'][0]
        self.assertIsNone(cell['terms']['E'])
        self.assertIsNotNone(cell['terms']['M'])

    def test_no_future_outputs_or_nonprospective_history_in_public_file(self):
        changed = copy.deepcopy(self.public)
        changed['forecast_inputs']['queues']['s'][0]['answer'] = 'hidden-validation-answer'
        self.put('inputs.json', changed)
        protocol, jobs = self.setup_gate()
        with self.assertRaisesRegex(ValueError, 'public request identity'):
            producer.run_and_seal(protocol, jobs, self.base/'bad-input', allow_cpu_fixture=True)


if __name__ == '__main__':
    unittest.main()
