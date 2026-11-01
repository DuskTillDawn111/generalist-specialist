"""Independent partial-return and source-change boundaries for v20."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests import test_learned_availability_forecast_v19_independent_review as fixture
from tools.predictor_validation_20261027 import availability_forecast_v11 as original
from tools.predictor_validation_20261027 import learned_availability_forecast_v20 as retention
from tools.predictor_validation_20261027.native_collect import receipt


class RetentionIndependentTests(unittest.TestCase):
    BASE = fixture.IndependentV19Tests.BASE
    engine = fixture.IndependentV19Tests.engine
    original_execute = fixture.IndependentV19Tests.original_execute
    forecast = fixture.IndependentV19Tests.forecast

    def setUp(self):
        fixture.IndependentV19Tests.setUp(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)/'wrapper-source.py'
        self.source.write_text('# original independent source\n')
        self.source_ref = receipt(self.source)

    def outer_context(self, instance, cutoff_ms):
        def execute(model):
            return self.original_execute(model.engine(max_events=10000), cutoff_ms)
        stack = __import__('contextlib').ExitStack()
        stack.enter_context(mock.patch.object(retention.V19Forecast, 'engine', return_value=self.engine(True)))
        stack.enter_context(mock.patch.object(retention.V19Forecast, 'execute', autospec=True, side_effect=execute))
        stack.enter_context(mock.patch.object(retention, 'implementation_sources', return_value=[self.source_ref]))
        return stack

    def test_actual_outer_retains_nonempty_partial_return_without_changing_original_UNKNOWN(self):
        expected = self.original_execute(self.engine(True), 20)
        instance = retention.LearnedAvailabilityForecast.__new__(retention.LearnedAvailabilityForecast)
        with self.outer_context(instance, 20):
            result = instance.execute()
        diagnostic = result.pop('unscored_forecast_diagnostic')
        self.assertEqual(result, expected)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertNotIn('metrics', result)
        self.assertIs(diagnostic['forecast'], instance._latest_retention_engine.last_returned_forecast)
        self.assertEqual(len(diagnostic['forecast']['cohort']), 5)
        self.assertEqual(len(diagnostic['forecast']['finishes']), 2)
        self.assertAlmostEqual(diagnostic['forecast']['board_energy_joules'], 2.29)
        self.assertFalse(diagnostic['score_recomputed'])
        self.assertFalse(diagnostic['complete_prediction_or_gate_evidence'])

    def test_post_forecast_source_verification_refuses_mutation(self):
        returned = {'actual_return': object()}

        def forecast():
            self.source.write_text('# changed while engine ran\n')
            return returned

        engine = retention.RetainReturnedForecast(SimpleNamespace(forecast=forecast), [self.source_ref])
        with self.assertRaisesRegex(ValueError, 'source.*changed'):
            engine.forecast()

    def test_original_score_source_mutation_refuses_both_UNKNOWN_and_complete_outer_returns(self):
        score = original.score_complete_forecast
        for cutoff in (20, 300):
            self.source.write_text('# original independent source\n')
            instance = retention.LearnedAvailabilityForecast.__new__(retention.LearnedAvailabilityForecast)

            def changing_score(*args, **kwargs):
                try:
                    return score(*args, **kwargs)
                finally:
                    self.source.write_text('# changed during original score\n')

            with self.subTest(cutoff=cutoff), self.outer_context(instance, cutoff), \
                 mock.patch.object(original, 'score_complete_forecast', side_effect=changing_score):
                with self.assertRaisesRegex(ValueError, 'source.*changed'):
                    instance.execute()

    def test_preentry_source_failure_resets_stale_return_without_calling_engine(self):
        underlying = SimpleNamespace(forecast=mock.Mock(return_value={'actual_return': 1}))
        engine = retention.RetainReturnedForecast(underlying, [self.source_ref])
        self.assertEqual(engine.forecast(), {'actual_return': 1})
        before = self.source.stat()
        self.source.write_text('# changed before second entry\n')
        stat = Path.stat

        def unchanged_metadata(path, *args, **kwargs):
            return before if path == self.source else stat(path, *args, **kwargs)

        # Equal metadata after an equal-length write was independently observed
        # on this filesystem. Hold that condition fixed for a stable regression.
        with mock.patch.object(Path, 'stat', new=unchanged_metadata):
            with self.assertRaisesRegex(ValueError, 'source.*changed'):
                engine.forecast()
        self.assertIsNone(engine.last_returned_forecast)
        self.assertEqual(underlying.forecast.call_count, 1)

    def test_actual_source_closure_includes_retention_and_original_score_paths(self):
        sources = retention.implementation_sources()
        by_name = {Path(ref['path']).name: ref for ref in sources}
        self.assertTrue({'learned_availability_forecast_v20.py', 'learned_availability_forecast_v19.py',
            'learned_availability_forecast_v16.py', 'availability_forecast_v11.py',
            'prospective_flow_contract_v11.py', 'metrics.py'} <= set(by_name))
        self.assertEqual(by_name['learned_availability_forecast_v20.py'], receipt(retention.__file__))

    def test_outer_retention_guard_also_binds_inner_compiled_training_sources(self):
        instance = retention.LearnedAvailabilityForecast.__new__(retention.LearnedAvailabilityForecast)
        inner = self.engine(True)
        with mock.patch.object(retention.V19Forecast, 'engine', return_value=inner):
            engine = instance.engine(max_events=10000)
        for reference in inner.sources:
            self.assertIn(reference, engine.sources)


if __name__ == '__main__':
    unittest.main()
