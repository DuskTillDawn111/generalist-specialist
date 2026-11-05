import copy
import unittest

from experiments.summarize_activation_host_trim_comparison import describe


def report():
    names = ('control-round1', 'trim-round1', 'trim-round2', 'control-round2')
    paths = {}
    for path in range(8):
        rows = []
        for repeat in (1, 2):
            for item in range(40):
                old, new = (1., 2.) if item == 0 else (100., 90.)
                rows.append(dict(round=repeat, key=[path, item], control_ms=old,
                                 trim_ms=new, trim_over_control=new / old))
        paths[str(path)] = dict(observations=rows)
    return dict(schema='gsdeploy.host_trim_runtime_diagnostic.v1',
        sources=dict.fromkeys(names, 'fixture'), paired_latency=paths, bands=[],
        trim_receipts=[dict(capture=name, alias=str(worker), rss_reduction_mib=10., procedure_ms=1.)
                       for name in names for worker in range(5)],
        episodes=[], issues=[], output_changes=[])


class HostTrimSummaryTest(unittest.TestCase):
    def test_aggregate_ratio_is_not_mean_paired_ratio_and_failures_survive(self):
        data = report()
        data['issues'] = ['fixture output failure']
        data['output_changes'] = [{'key': 'changed'}]
        result = describe(data)
        pooled = result['paths']['0']['pooled']
        self.assertAlmostEqual(pooled['mean_change_fraction'], 3512 / 3901 - 1)
        self.assertNotAlmostEqual(pooled['mean_change_fraction'], pooled['paired_ratio']['mean'] - 1)
        self.assertAlmostEqual(pooled['p95_change_fraction'], -.1)
        self.assertEqual(result['issues'], data['issues'])
        self.assertEqual(result['output_changes'], data['output_changes'])
        self.assertFalse(result['physical_predictor_qualified'])

    def test_missing_round_partial_cohort_and_duplicate_receipt_are_rejected(self):
        data = report()
        cases = []
        missing = copy.deepcopy(data)
        del missing['sources']['trim-round2']
        cases.append(missing)
        partial = copy.deepcopy(data)
        partial['paired_latency']['0']['observations'].pop()
        cases.append(partial)
        duplicate = copy.deepcopy(data)
        duplicate['trim_receipts'][1]['alias'] = duplicate['trim_receipts'][0]['alias']
        cases.append(duplicate)
        for case in cases:
            with self.subTest(case=len(case['sources'])):
                with self.assertRaises(ValueError):
                    describe(case)


if __name__ == '__main__':
    unittest.main()
