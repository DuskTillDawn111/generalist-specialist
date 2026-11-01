"""Native selection gates retain unknown and do not force spill."""
import unittest

from tools.queue_coupled_utility_20261023.study import analyze


class NativeStudyTests(unittest.TestCase):
    def fixture(self):
        jobs = []
        raw = {}
        for seed in (303, 304):
            for role in ('coupled_spill', 'matched_no_spill'):
                label = f'{role}-seed{seed}'
                spill = role == 'coupled_spill'
                jobs.append(dict(label=label, role=role, seed=seed,
                                 nominal_plan={'deployment': {'text': 2, 'generalist_2': 1, 'ocr': 1}},
                                 policy_contract={'coupled': {'allow_spill': spill}}))
                raw[label] = dict(status='completed', replica_to_family={'s': 'text', 'g': 'generalist_2'},
                                  request_records=[dict(task='text', model='g' if spill else 's')],
                                  evaluation=dict(utility=dict(U=-.5 if spill else -.6, eligible=True,
                                                               **{t + '_norm': .1 for t in 'QTLPME'}),
                                                  stability={'stability_state_v2': 'steady'}, dynamic_audit={'verified': True}))
        return dict(jobs=jobs, seeds=[303, 304]), raw

    def test_gate_requires_both_pairs_and_actual_spill(self):
        packet, raw = self.fixture()
        report = analyze(packet, raw)
        self.assertTrue(report['confirmation_gate_passed'])
        self.assertEqual(report['observations'][0]['terms']['Q'], .1)
        raw['coupled_spill-seed304']['request_records'][0]['model'] = 's'
        self.assertFalse(analyze(packet, raw)['confirmation_gate_passed'])

    def test_unknown_or_nonpositive_pair_cannot_be_replaced_by_good_mean(self):
        packet, raw = self.fixture()
        raw['coupled_spill-seed304']['evaluation']['utility']['U'] = None
        report = analyze(packet, raw)
        self.assertFalse(report['confirmation_gate_passed'])
        self.assertIsNone(report['paired'][1]['delta_U'])
        raw['coupled_spill-seed304']['evaluation']['utility']['U'] = -.7
        self.assertFalse(analyze(packet, raw)['confirmation_gate_passed'])


if __name__ == '__main__':
    unittest.main()
