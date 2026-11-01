"""Independent producer boundary checks using actual CPU simulator fixtures."""
import copy
import json
from pathlib import Path
import unittest

from gsdeploy.forecasting.prediction_gate import freeze_gate_protocol
from tests import test_forecast_gate_v2 as fixtures
from tools.predictor_validation_20261027 import forecast_gate_v2 as producer


class IndependentForecastGateReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ForecastGateTests('test_actual_simulator_handler_and_receipt_boundaries_with_startup_cost')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def paired_gate(self, altered):
        f = self.fixture
        base = f.put('base.json', f.bundle)
        spill = f.put('spill.json', altered)
        candidates = [dict(id=cid, model_id=cid, execution_context_sha256='fixture') for cid in ('base','spill')]
        protocol = freeze_gate_protocol(model_paths=dict(base=base,spill=spill),
            source_paths=[*producer.prediction_source_paths(), f.inputs],
            training_data_paths=[f.training], candidates=candidates, baseline_candidate_id='base',
            validation_domain=[dict(run_id='heldout', requests=[dict(request_id=rid,pool_id='x',task='text')
                                                               for rid in ('one','two')])],
            thresholds={}, created_unix_ns=1, weights={k:1/6 for k in 'QTLPME'})
        jobs = [dict(run_id='heldout',candidate_id=cid,public_inputs_path=str(f.inputs)) for cid in ('base','spill')]
        return f.put('protocol.json',protocol), jobs

    def test_paired_candidates_cannot_change_each_objective_normalizer(self):
        variants = [('latency_reference_ms',{'text':999}), ('memory_budget_mib',999),
                    ('energy_reference_watts',999), ('memory_safe_factor',2),
                    ('startup_amortization_seconds',999)]
        for index,(field,value) in enumerate(variants):
            with self.subTest(field=field):
                altered = copy.deepcopy(self.fixture.bundle)
                container = altered if field == 'startup_amortization_seconds' else altered['forecast_defaults']
                container[field] = value
                protocol,jobs = self.paired_gate(altered)
                with self.assertRaisesRegex(ValueError,'objective normalizers'):
                    producer.run_and_seal(protocol,jobs,self.fixture.base/f'normalizer-{index}',allow_cpu_fixture=True)

    def test_paired_workload_cannot_shift_request_arrivals(self):
        f = self.fixture
        other = copy.deepcopy(f.public)
        other['forecast_inputs']['queues']['s'].pop()
        other['forecast_inputs']['future_workload']['declared_arrivals'] = [
            dict(request_id='two',pool_id='x',task='text',offset_ns=1)]
        other_path = f.put('other-inputs.json',other)
        protocol,jobs = f.setup_gate(source_paths=[*producer.prediction_source_paths(),f.inputs,other_path])
        jobs[1]['public_inputs_path'] = str(other_path)
        with self.assertRaisesRegex(ValueError,'workload/arrival cohort differs'):
            producer.run_and_seal(protocol,jobs,f.base/'shifted-arrival',allow_cpu_fixture=True)

    def test_successful_seal_retains_full_cohort_and_integer_derived_timings(self):
        f = self.fixture
        protocol,jobs = f.setup_gate()
        report = producer.run_and_seal(protocol,jobs,f.base/'independent-success',allow_cpu_fixture=True)
        seal = json.loads(Path(report['predictions']['path']).read_text())
        self.assertEqual(len(seal['cells']),2)
        for cell in seal['cells']:
            self.assertEqual({row['request_id'] for row in cell['requests']},{'one','two'})
            for row in cell['requests']:
                self.assertGreaterEqual(row['e2e_ms'],row['service_ms']+row['queue_ms'])
        self.assertFalse(report['accuracy_established'])
        self.assertFalse(report['physical_gain_certified'])


if __name__ == '__main__':
    unittest.main()
