import itertools
import unittest

from experiments.prepare_activation_aggregate_training import TASKS, make_jobs


class AggregateTrainingDesignTests(unittest.TestCase):
    def setUp(self):
        self.aliases = {f + '#0': f for f in ('generalist_2', *TASKS)}
        self.template = dict(profiles=[
            dict(family=f, task=t, pool_id=f'{t}:{i}',
                 segments=[dict(duration_ms={'mean': 100. * (i + 1)})])
            for f in self.aliases.values()
            for t in (TASKS if f == 'generalist_2' else (f,)) for i in range(10)])

    def test_covers_every_legal_activity_roster_with_one_request_per_owner(self):
        jobs = make_jobs(self.template, self.aliases, 123)
        actual = set()
        for job in jobs:
            if job.get('design', {}).get('kind') != 'crossed_roster':
                continue
            rows = job['requests']
            self.assertEqual(len(rows), len({r['alias'] for r in rows}))
            actual.add(tuple(sorted((r['alias'], r['task']) for r in rows)))
        expected = set()
        for size in range(2, 6):
            for owners in itertools.combinations(sorted(self.aliases), size):
                for tasks in itertools.product(*(TASKS if self.aliases[a] == 'generalist_2'
                                                 else (self.aliases[a],) for a in owners)):
                    expected.add(tuple(zip(owners, tasks)))
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 71)

    def test_routing_variants_share_offered_cohort_and_hold_window(self):
        jobs = make_jobs(self.template, self.aliases, 123)
        for load in (.4, .8):
            selected = [j for j in jobs if j['id'].startswith(f'arrivals:{load}:')]
            self.assertEqual(len(selected), 3)
            cohorts = [[(r['offset_ms'], r['task'], r['pool_id']) for r in j['requests']]
                       for j in selected]
            self.assertEqual(cohorts[0], cohorts[1])
            self.assertEqual(cohorts[1], cohorts[2])
            self.assertTrue(cohorts[0])
            self.assertTrue(all(j['duration_ms'] == 45000 for j in selected))
            self.assertTrue(all(0 <= r['offset_ms'] < 45000 for j in selected for r in j['requests']))

    def test_reproducible_design_and_distinct_independent_order(self):
        first = make_jobs(self.template, self.aliases, 123)
        self.assertEqual(first, make_jobs(self.template, self.aliases, 123))
        second = make_jobs(self.template, self.aliases, 124)
        self.assertNotEqual([j['id'] for j in first], [j['id'] for j in second])
        self.assertEqual(len(first), len({j['id'] for j in first}))
        for job in first:
            for row in job['requests']:
                family = self.aliases[row['alias']]
                self.assertTrue(family == 'generalist_2' or row['task'] == family)
                self.assertEqual(set(row), {'alias', 'task', 'pool_id', 'offset_ms'})


if __name__ == '__main__':
    unittest.main()
