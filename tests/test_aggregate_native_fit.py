import copy
import unittest

from experiments.fit_activation_aggregate_native import fit_progress
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import NativeContextLaw


class AggregateNativeFitTests(unittest.TestCase):
    def setUp(self):
        self.template = dict(execution_context_sha256='context', profiles=[
            dict(id=f, family=f, task=f, segments=[dict(id='native', kind='gpu', duration_ms={'mean': 100.})])
            for f in ('a', 'b')])
        self.target = dict(profile_id='a', segment_id='native')
        self.peer = dict(profile_id='b', segment_id='native')
        self.solo = {'a#0': self.target}
        self.pair = dict(self.solo, **{'b#0': self.peer})
        self.observations = []
        for run in ('r1', 'r2'):
            for name, intervals in (
                    ('solo', [(100., self.solo)]),
                    ('pair', [(200., self.pair)]),
                    ('mixed', [(50., self.solo), (100., self.pair)])):
                self.observations.append(dict(id=run+name, run_id=run, split='training',
                    execution_context_sha256='context', target_alias='a#0', profile_id='a', segment_id='native',
                    intervals=[dict(duration_ms=d, members=m) for d, m in intervals]))

    def law(self, observations):
        spec = fit_progress(self.template, observations)
        law = NativeContextLaw(spec, {p['id']: p for p in self.template['profiles']}, 'context')
        return spec, law

    def test_recovers_progress_from_changing_overlap(self):
        _, law = self.law(self.observations)
        self.assertAlmostEqual(law.progress('a#0', self.target, self.solo), 1., delta=.02)
        self.assertAlmostEqual(law.progress('a#0', self.target, self.pair), .5, delta=.02)

    def test_local_outlier_retained_without_claiming_qualification(self):
        rows = copy.deepcopy(self.observations)
        extra = copy.deepcopy(rows[0])
        extra['id'] = 'outlier'
        extra['intervals'][0]['duration_ms'] = 400.
        rows.append(extra)
        spec, _ = self.law(rows)
        diagnostic = spec['fit_diagnostics'][0]
        self.assertEqual(diagnostic['observations'], 7)
        self.assertGreater(diagnostic['maximum_work_error'], .2)
        self.assertTrue(diagnostic['fitted'])
        self.assertFalse(diagnostic['physical_accuracy_qualified'])

    def test_unrepeated_activity_is_not_granted_support(self):
        rows = copy.deepcopy(self.observations)
        extra = copy.deepcopy(rows[1])
        extra['id'] = 'one-off-three-owner'
        extra['intervals'][0]['members']['b#1'] = self.peer
        rows.append(extra)
        spec, law = self.law(rows)
        self.assertEqual(spec['fit_diagnostics'][0]['unrepeated_activity_vertices'], 1)
        with self.assertRaises(UnknownEvidence):
            law.progress('a#0', self.target, extra['intervals'][0]['members'])

    def test_validation_cannot_enter_training(self):
        rows = copy.deepcopy(self.observations)
        rows[0]['split'] = 'development_validation'
        with self.assertRaises(ValueError):
            fit_progress(self.template, rows)


if __name__ == '__main__':
    unittest.main()
