"""Exercise original DAG and six-term rejection through v20 retention."""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_learned_availability_forecast_v19_independent_review as fixtures
from tools.predictor_validation_20261027.learned_availability_forecast_v20 import (
    LearnedAvailabilityForecast, RetainReturnedForecast, V19Forecast,
    attach_unscored_diagnostic)


class RetentionTests(unittest.TestCase):
    setUp = fixtures.IndependentV19Tests.setUp
    engine = fixtures.IndependentV19Tests.engine
    original_execute = fixtures.IndependentV19Tests.original_execute
    forecast = fixtures.IndependentV19Tests.forecast
    BASE = fixtures.IndependentV19Tests.BASE

    def test_actual_score_rejection_retains_exact_original_forecast(self):
        engine = RetainReturnedForecast(self.engine(True))
        result = self.original_execute(engine, .5)
        before = copy.deepcopy(result)
        attach_unscored_diagnostic(result, engine)
        diagnostic = result.pop('unscored_forecast_diagnostic')
        self.assertEqual(result, before)
        self.assertIs(diagnostic['forecast'], engine.last_returned_forecast)
        self.assertEqual(len(diagnostic['forecast']['cohort']), 5)
        self.assertEqual(diagnostic['forecast']['finishes'], {})
        self.assertEqual(result['reason'], 'COMPLETE_DECLARED_FLOW_DAG_AND_PARENT_RECEIPTS_REQUIRED')
        self.assertNotIn('metrics', result)
        self.assertFalse(diagnostic['complete_prediction_or_gate_evidence'])

    def test_complete_original_metrics_forecast_unchanged(self):
        expected = self.original_execute(self.engine(True), 300)
        engine = RetainReturnedForecast(self.engine(True))
        actual = attach_unscored_diagnostic(self.original_execute(engine, 300), engine)
        self.assertEqual(actual, expected)
        self.assertIs(actual['forecast'], engine.last_returned_forecast)
        self.assertIsNotNone(actual['metrics']['full_U'])
        self.assertNotIn('unscored_forecast_diagnostic', actual)

    def test_engine_unknown_never_invents_return_and_clears_stale_result(self):
        engine = RetainReturnedForecast(self.engine(True))
        self.forecast(engine)
        self.assertIsNotNone(engine.last_returned_forecast)
        # Use an engine that can be called repeatedly, unlike one-shot v19.
        underlying = SimpleNamespace(forecast=mock.Mock(side_effect=UnknownEvidence('NO_RESOURCE')))
        engine._engine = underlying
        with self.assertRaisesRegex(UnknownEvidence, 'NO_RESOURCE'):
            engine.forecast([])
        self.assertIsNone(engine.last_returned_forecast)
        result = dict(status='UNKNOWN', reason='NO_RESOURCE')
        self.assertEqual(attach_unscored_diagnostic(result, engine), result)
        self.assertNotIn('unscored_forecast_diagnostic', result)

    def test_actual_v20_outer_entry_uses_original_result_and_engine(self):
        instance = LearnedAvailabilityForecast.__new__(LearnedAvailabilityForecast)
        base = self.engine(True)
        with mock.patch.object(V19Forecast, 'engine', return_value=base), \
             mock.patch.object(V19Forecast, 'execute', autospec=True,
                               side_effect=lambda model: self.original_execute(model.engine(max_events=10000), .5)):
            result = instance.execute()
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIs(result['unscored_forecast_diagnostic']['forecast'],
                      instance._latest_retention_engine.last_returned_forecast)
        self.assertIs(instance._latest_retention_engine._engine, base)


if __name__ == '__main__':
    unittest.main()
