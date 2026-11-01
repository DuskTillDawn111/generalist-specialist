import copy
import unittest

from tools.predictor_validation_20261027.natural_resources_v5 import SCHEMA
from tools.predictor_validation_20261027.profiling import RESIDENTS, digest
from tools.predictor_validation_20261027.resource_training_campaign_v5 import ALIASES, design, universe
from tools.predictor_validation_20261027.resource_training_collection_review_v5 import coverage


def fixture():
    states = universe()
    known = []
    for state in states:
        tasks = state['owner_tasks']
        ntext = sum(a.startswith('text#') for a in tasks)
        gtask = tasks.get('generalist_2#0')
        ocr = 'ocr#0' in tasks
        if (not ocr and (gtask is None or ntext == 0 or gtask == 'text')) or (ocr and len(tasks) == 1):
            known.append(state)
    primers = [dict(task=task, pool_id=task + '-' + str(i), input_size=10 + i)
               for task in ('text', 'image', 'asr', 'ocr') for i in range(2)]
    return dict(schema=SCHEMA, states={s['key']: dict(known=True) for s in known}), primers


class ResourceTrainingDesignTests(unittest.TestCase):
    def test_exact_24_state_universe_and_15_missing_are_direct_not_synthesized(self):
        model, primers = fixture()
        result = design(model, primers)
        self.assertEqual(len(result['universe']), 24)
        self.assertEqual(len({s['key'] for s in result['universe']}), 24)
        self.assertEqual((len(result['covered']), len(result['missing']), len(result['jobs'])), (9, 15, 30))
        self.assertEqual(sum(len(j['inputs']) for j in result['jobs']), 84)
        for job in result['jobs']:
            self.assertEqual(set(job['inputs']), set(job['tasks']))
            self.assertEqual(set(job['start_offsets_ms'].values()), {0})
            self.assertTrue(set(job['inputs']) <= set(RESIDENTS))
            self.assertEqual(job['split'], 'training')
            self.assertEqual(job['id'], digest({k: v for k, v in job.items() if k != 'id'}))
            self.assertNotIn(job['resource_target_state_key'], model['states'])

    def test_each_missing_state_has_two_strata_and_original_primers_unchanged(self):
        model, primers = fixture()
        before = copy.deepcopy(primers)
        result = design(model, primers)
        for state in result['missing']:
            jobs = [j for j in result['jobs'] if j['resource_target_state_key'] == state['key']]
            self.assertEqual({j['input_stratum'] for j in jobs}, {'short', 'long'})
            self.assertEqual(len(jobs), 2)
        self.assertEqual(primers, before)
        self.assertEqual(len(ALIASES), 4)

    def test_unsupported_existing_state_is_still_targeted_and_aliases_not_added(self):
        model, primers = fixture()
        one = next(k for k, v in model['states'].items() if k != digest([]))
        model['states'][one]['known'] = False
        result = design(model, primers)
        self.assertEqual(len(result['missing']), 16)
        self.assertIn(one, {s['key'] for s in result['missing']})
        self.assertEqual({a for j in result['jobs'] for a in j['inputs']}, set(RESIDENTS))

    def test_missing_primers_and_idle_without_real_evidence_are_rejected(self):
        model, primers = fixture()
        with self.assertRaisesRegex(ValueError, 'short/long'):
            design(model, primers[:-1])
        model['states'].pop(digest([]))
        with self.assertRaisesRegex(ValueError, 'Idle'):
            design(model, primers)

    def test_strict_adapter_checks_all_generic_jobs_without_six_job_assumption(self):
        model, primers = fixture()
        plan = dict(jobs=design(model, primers)['jobs'])
        rows, batches = [], []
        for job in plan['jobs']:
            request_ids = []
            for alias, pool in job['inputs'].items():
                request_id = 'request' + str(len(rows))
                rows.append(dict(profile_job_id=job['id'], request_id=request_id, model=alias,
                                 pool_id=pool, task=job['tasks'][alias], split='training'))
                request_ids.append(request_id)
            batches.append(dict(job_id=job['id'], request_ids=request_ids))
        raw = dict(request_records=rows, batches=batches)
        self.assertEqual(coverage(plan, raw, plan), [])
        broken = copy.deepcopy(raw)
        broken['request_records'].pop()
        self.assertIn('EXACT_TRAINING_REQUEST_COVERAGE_FAILED', coverage(plan, broken, plan))
        broken = copy.deepcopy(raw)
        broken['request_records'][0]['split'] = 'validation'
        self.assertIn('EXACT_TRAINING_REQUEST_COVERAGE_FAILED', coverage(plan, broken, plan))
        broken = copy.deepcopy(raw)
        broken['request_records'][1]['request_id'] = broken['request_records'][0]['request_id']
        self.assertIn('INVALID_REQUEST_IDS', coverage(plan, broken, plan))


if __name__ == '__main__':
    unittest.main()
