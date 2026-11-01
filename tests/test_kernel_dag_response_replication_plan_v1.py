"""CPU checks for the concrete full-owner replication matrix."""
from collections import Counter
from itertools import combinations
import unittest

from tools.predictor_validation_20261027 import kernel_dag_response_replication_plan_v1 as m


class ReplicationPlanTests(unittest.TestCase):
    def fixture(self):
        return {owner:dict(pool_id='exact-'+owner,task='ocr' if owner=='ocr#0' else 'text',exact_input_sha256=str(i)*64)
                for i,owner in enumerate(('text#0','text#1','generalist_2#0','ocr#0'))}

    def test_all_singles_pairs_and_all_four_are_distinct_complete_jobs(self):
        bindings=self.fixture();jobs=m.jobs_for(bindings)
        cohorts={frozenset(job['inputs']) for job in jobs}
        expected={frozenset(c) for width in (1,2,4) for c in combinations(bindings,width)}
        self.assertEqual(cohorts,expected);self.assertEqual(len(jobs),11)
        self.assertEqual(len({job['id'] for job in jobs}),11)
        self.assertEqual(Counter(len(job['inputs']) for job in jobs),{1:4,2:6,4:1})

    def test_twenty_exact_work_requests_and_five_observations_per_owner(self):
        bindings=self.fixture();jobs=m.jobs_for(bindings)
        self.assertEqual(sum(len(job['inputs']) for job in jobs),20)
        self.assertEqual(Counter(owner for job in jobs for owner in job['inputs']),dict.fromkeys(bindings,5))
        for job in jobs:
            self.assertEqual(job['start_offsets_ms'],dict.fromkeys(job['inputs'],0))
            self.assertEqual(job['split'],'training')
            for owner,pool in job['inputs'].items():
                self.assertEqual(pool,bindings[owner]['pool_id']);self.assertEqual(job['tasks'][owner],bindings[owner]['task'])

    def test_incomplete_owner_bindings_are_rejected(self):
        bindings=self.fixture();bindings.pop('ocr#0')
        with self.assertRaises(ValueError):m.jobs_for(bindings)
        bindings=self.fixture();bindings['ocr#0'].pop('task')
        with self.assertRaises(ValueError):m.jobs_for(bindings)


if __name__=='__main__':unittest.main()
