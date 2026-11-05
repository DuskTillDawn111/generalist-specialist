import unittest

from experiments.continue_activation_minimal_cupti_probe import unresolved_trim_repeats


def report(off, on):
    return dict(schema='gsdeploy.host_trim_runtime_diagnostic.v1', issues=[], output_changes=[],
        bands=[dict(mode='off', raw_constant_minimax=off), dict(mode='on', raw_constant_minimax=on)])


class MinimalCuptiFollowupTest(unittest.TestCase):
    def test_only_unresolved_trim_repeat_schedules_diagnosis(self):
        self.assertEqual(unresolved_trim_repeats(report(.5, .2)), [])
        self.assertEqual(unresolved_trim_repeats(report(.01, .25)),
                         [dict(mode='on', raw_constant_minimax=.25)])

    def test_output_or_capture_problems_require_inspection_first(self):
        for field in ('issues', 'output_changes'):
            data = report(.5, .5)
            data[field] = ['problem']
            with self.subTest(field=field), self.assertRaises(ValueError):
                unresolved_trim_repeats(data)

    def test_incomplete_arm_does_not_trigger_another_gpu_capture(self):
        data = report(.5, .5)
        data['bands'].pop()
        with self.assertRaises(ValueError):
            unresolved_trim_repeats(data)


if __name__ == '__main__':
    unittest.main()
