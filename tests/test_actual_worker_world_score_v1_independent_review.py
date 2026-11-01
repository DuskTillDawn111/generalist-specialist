"""Independent endpoint/resource counterexamples for the integration wrapper."""
import copy
import unittest

from tests import test_actual_worker_twoFIFO_reference_v4 as queue_fixtures
from tests import test_actual_worker_world_score_v1 as score_fixtures
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v4 import simulate
from tools.predictor_validation_20261027.actual_worker_world_score_v1 import score_world


class DifferentiatedResources(score_fixtures.ExplicitCPUResources):
    def power(self,active,*,resident_owners):
        return dict(known=True,watts=300 if active else 100)

    def memory(self,active,*,resident_owners):
        return dict(known=True,peak_memory_mib=900 if active else 400)


class ActualWorldScoreIndependentTests(unittest.TestCase):
    def test_extremely_late_parent_processing_cannot_replace_reader_latency(self):
        fixture=score_fixtures.ActualWorldScoringTests();args=fixture.inputs()
        baseline=fixture.call(args,DifferentiatedResources())
        kwargs=queue_fixtures.ActualWorkerTwoFIFOTests().inputs()
        for marks in kwargs['protocol_slacks'].values():
            for slack in marks.values():slack['collector_processing_slack_ns']=10**9
        delayed=simulate(**kwargs)
        self.assertEqual(delayed['RESULT_processed'],{})
        self.assertEqual(delayed['RESULT_receipts'],args[0]['RESULT_receipts'])
        altered=(delayed,*args[1:])
        result=fixture.call(altered,DifferentiatedResources())
        self.assertEqual(result['original_score']['terms'],baseline['original_score']['terms'])
        self.assertEqual(result['original_score']['terms']['L'],24)
        self.assertAlmostEqual(result['original_score']['terms']['P'],28.5)

    def test_measurement_only_energy_startup_UNKNOWN_and_future_peak_safe_once(self):
        fixture=score_fixtures.ActualWorldScoringTests();args=fixture.inputs()
        result=fixture.call(args,DifferentiatedResources())
        # ns [0,1) idle100W; [1,2) active300W. Drain is excluded.
        # ((4e-7 J)/(2e-9 s) + 3600J/3600s) / 10W = 20.1.
        self.assertAlmostEqual(result['original_score']['terms']['E'],20.1)
        # max(startup600, future900) * 1.2 / 1000, once.
        self.assertAlmostEqual(result['original_score']['terms']['M'],1.08)
        unknown=fixture.call(args,DifferentiatedResources(startup=None))
        self.assertIsNone(unknown['original_score']['terms']['E'])
        self.assertIsNone(unknown['original_score']['full_U'])
        self.assertIsNotNone(unknown['resource_prediction']['board_energy_joules'])
        self.assertFalse(unknown['independent_prediction_accuracy_validated'])


if __name__=='__main__':unittest.main()
