"""CPU checks for joint residual scenarios, not evidence of GPU calibration."""
import copy
import tempfile
import unittest
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import FullHorizonModel
from tests.test_background_statistical import fixture
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider
from tools.predictor_validation_20261027.mechanism_audit import receipt
from tools.predictor_validation_20261027.residual_scenarios import (
    fit_vectors, JointScenarioProvider, adapt_reconstructed_vectors)


class JointResidualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        source = Path(self.temp.name) / 'training.json'
        source.write_text('training fixture')
        self.sources = [receipt(source)]
        self.base = BackgroundStatisticalProvider(fixture(), 'v7-fixture')
        self.aliases = {'s': {'family': 'text', 'supported_tasks': ['text']},
                        'other': {'family': 'text', 'supported_tasks': ['text']}}
        self.rows = [{'trace_id': 'training', 'request_id': str(i), 'split': 'training',
                     'family': 'text', 'task': 'text', 'pool_id': 'input', 'known': True,
                     'work_by_phase': {'prefill': 2., 'decode': 2.},
                     'reconstruction_basis': 'CPU_FIXTURE'} for i in range(3)]

    def fit(self, rows=None):
        return fit_vectors(self.rows if rows is None else rows,
            base_model_sha256=self.base.document['model_sha256'],
            execution_context_sha256='v7-fixture', sources=self.sources)

    def provider(self, rows=None, index=0):
        return JointScenarioProvider(self.base, self.fit(rows), index)

    def test_joint_vector_and_alias_independent_common_uniform(self):
        rows = copy.deepcopy(self.rows)
        for i, row in enumerate(rows):
            row['work_by_phase'] = {'prefill': i + 1., 'decode': 10. * (i + 1)}
        provider = self.provider(rows)
        row = {'request_id': 'new', 'alias': 's', 'task': 'text', 'pool_id': 'input'}
        first = provider._vector(row, self.aliases)
        other = provider._vector({**row, 'alias': 'other'}, self.aliases)
        self.assertEqual(first, other)
        self.assertEqual(first['decode'], 10 * first['prefill'])
        self.assertEqual(first, self.provider(rows)._vector(row, self.aliases))

    def test_full_event_rollout_scales_once(self):
        queues = {'s': [{'request_id': 'new', 'task': 'text', 'pool_id': 'input', 'arrival_ns': 0}]}
        aliases = {'s': self.aliases['s']}
        def finish(provider):
            result = FullHorizonModel(aliases, provider).forecast(queues, 0, 1_000_000_000,
                require_energy=False, require_memory=False)
            return result['finishes']['new']['finish_ns']
        self.assertEqual(finish(self.provider()), 2 * finish(self.base))

    def test_validation_and_duplicate_vectors_rejected(self):
        for update in ({'split': 'validation'}, {'not_for_training': True}):
            rows = copy.deepcopy(self.rows)
            rows[0].update(update)
            with self.assertRaises(ValueError):
                self.fit(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.fit(self.rows + self.rows[:1])

    def test_unknown_and_partial_vectors_never_imputed(self):
        row = {'request_id': 'new', 'alias': 's', 'task': 'text', 'pool_id': 'input'}
        rows = copy.deepcopy(self.rows)
        rows[0]['known'] = False
        with self.assertRaisesRegex(UnknownEvidence, 'INSUFFICIENT'):
            self.provider(rows).plan(row, self.aliases, 'mean')
        for sample in rows:
            sample['known'] = True
            sample['work_by_phase'].pop('decode')
        with self.assertRaisesRegex(UnknownEvidence, 'INCOMPLETE'):
            self.provider(rows).plan(row, self.aliases, 'mean')

    def test_input_populations_remain_separate(self):
        row = {'request_id': 'new', 'alias': 's', 'task': 'text', 'pool_id': 'different'}
        with self.assertRaises(UnknownEvidence):
            self.provider()._vector(row, self.aliases)

    def test_source_model_and_context_tampering_rejected(self):
        document = self.fit()
        document['populations'].clear()
        with self.assertRaisesRegex(ValueError, 'Changed'):
            JointScenarioProvider(self.base, document, 0)
        Path(self.sources[0]['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Changed residual training source'):
            self.fit()

    def test_scenario_index_must_be_nonnegative_integer(self):
        for index in (-1, .5, True):
            with self.assertRaises(ValueError):
                self.provider(index=index)

    def test_reconstruction_adaptation_preserves_validation_role(self):
        original = {**self.rows[0], 'split': 'validation', 'not_for_training': True,
                    'complete_known_vector': True}
        original.pop('known')
        rows = adapt_reconstructed_vectors({
            'schema': 'gsdeploy.frozen_model_observed_joint_phase_work.v1',
            'requests': [original]}, reconstruction_basis={'fold': 'held-out-training-run'})
        self.assertTrue(rows[0]['known'])
        self.assertEqual(rows[0]['split'], 'validation')
        with self.assertRaises(ValueError):
            self.fit(rows)

    def test_generic_provider_rejected_to_prevent_double_scaling(self):
        with self.assertRaises(TypeError):
            JointScenarioProvider(object(), self.fit(), 0)


if __name__ == '__main__':
    unittest.main()
