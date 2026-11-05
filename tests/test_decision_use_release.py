import copy
from pathlib import Path
import tempfile
import unittest

from experiments.continue_decision_confirmation import require_pure_specialist
from experiments.run_scoped_decision_search import check_release
from experiments.scoped_routing_objective import file_receipt


class DecisionUseReleaseTests(unittest.TestCase):
    def release(self):
        return dict(status='FROZEN_FOR_BOUNDED_SEARCH_WITH_KNOWN_LIMITATION',
            decision_search_authorized=True, strict_latency_gate_pass=False,
            no_refit=True, no_additional_predictor_collection=True,
            mandatory_confirmation_controls=['pure_specialists'], max_candidates=3, max_gpu_episodes=6,
            search_methods=['Joint-K2', 'Activation-BO-MADS'], search_seeds=[11, 29, 47],
            search_budget_per_method_seed=dict(seconds=300, predictions=64, routes=32, master_calls=128),
            source_receipts=[])

    def test_original_failure_preserved_and_budget_not_expanded(self):
        release = self.release()
        check_release(release)
        for key, value in [('strict_latency_gate_pass', True), ('no_refit', False),
                           ('max_candidates', 4), ('max_gpu_episodes', 8)]:
            bad = copy.deepcopy(release)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_release(bad)

    def test_changed_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'evidence.json'
            p.write_text('{}')
            release = self.release()
            release['source_receipts'] = [file_receipt(p)]
            check_release(release)
            p.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                check_release(release)

    def test_mixed_or_incomplete_endpoint_cannot_replace_pure_specialist_control(self):
        good = dict(candidate_id='baseline', routing={t: {t: .1} for t in ('asr', 'image', 'ocr', 'text')})
        self.assertEqual(require_pure_specialist([good]), ['baseline'])
        bad = copy.deepcopy(good)
        bad['routing']['text'] = {'generalist_2': .1}
        for panel in ([], [bad], [dict(candidate_id='empty', routing={})]):
            with self.assertRaises(ValueError):
                require_pure_specialist(panel)


if __name__ == '__main__':
    unittest.main()
