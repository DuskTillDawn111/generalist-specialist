"""Pure prospective coverage checks; no runtime results or GPU access."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027.flow_suite_audit import check_structure
from tools.predictor_validation_20261027.profiling import digest


class SuiteAuditStructureTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(phase='predictor_validation', seed=1,
            trace=[dict(request_id='r', task='text', pool_id='p', offset_ns=0)],
            candidates=[dict(id='a', routes={'r': 's'})])
        metric = dict(six_term_comparison_complete=True, cost=1., cohort_size=1,
            cohort_ids=['r'], incomplete_ids=[])
        self.sealed = dict(cohort_sha256=digest(self.plan['trace']), information_cutoff_ns=1,
            created_ns=2, candidates=[dict(candidate_id='a', unknown_reason=None, metrics=metric,
            comparison_U=-1., requests=[dict(request_id='r', task='text', pool_id='p', alias='s',
                service_ms=1., wait_ms=2., e2e_ms=3., finish_ns=3_000_000)])])
        self.suite = dict(created_ns=4, jobs=[dict(candidate_id='a', output='/prospective/runs/a')],
            sources={'plan': {'path': 'plan', 'sha256': 'fixture'}}, no_retries=True, no_seed_replacement=True)
        self.qualification = dict(target_plan=self.suite['sources']['plan'], target_seed=1, created_ns=3)
        for target in ('condition', '_generation'):
            mock = patch('tools.predictor_validation_20261027.flow_suite_audit.' + target, return_value={})
            mock.start(); self.addCleanup(mock.stop)

    def check(self):
        return check_structure(self.plan, self.sealed, self.suite, self.qualification, '/prospective/suite.json')

    def test_complete_identity_and_predeclared_chronology(self):
        self.assertEqual(self.check()['complete_request_predictions'], 1)

    def test_empty_or_duplicate_request_predictions_cannot_vacuously_pass(self):
        original = copy.deepcopy(self.sealed)
        for rows in ([], self.sealed['candidates'][0]['requests'] * 2):
            self.sealed = copy.deepcopy(original); self.sealed['candidates'][0]['requests'] = rows
            with self.assertRaisesRegex(ValueError, 'complete unique offered cohort'):
                self.check()

    def test_wrong_alias_task_pool_or_time_arithmetic_rejected(self):
        for key, value in [('alias', 'other'), ('task', 'asr'), ('pool_id', 'unseen'), ('finish_ns', 4_000_000)]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.sealed)
                self.sealed['candidates'][0]['requests'][0][key] = value
                with self.assertRaises(ValueError): self.check()
                self.sealed = original

    def test_missing_jobs_retry_or_mismatched_qualification_rejected(self):
        self.suite['jobs'] = []
        with self.assertRaisesRegex(ValueError, 'one suite job'): self.check()
        self.suite['jobs'] = [dict(candidate_id='a', output='/prospective/runs/a')]
        self.suite['no_retries'] = False
        with self.assertRaisesRegex(ValueError, 'no retries'): self.check()
        self.suite['no_retries'] = True; self.qualification['target_seed'] = 2
        with self.assertRaisesRegex(ValueError, 'different target'): self.check()
        self.qualification['target_seed'] = 1; self.qualification['created_ns'] = 5
        with self.assertRaisesRegex(ValueError, 'precede'): self.check()


if __name__ == '__main__': unittest.main()
