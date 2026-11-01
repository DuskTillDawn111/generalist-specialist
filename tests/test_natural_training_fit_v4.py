"""CPU synthetic files test split integrity, independent support and units."""
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_startup_prediction_fit_v4 import raw_fixture
from tools.predictor_validation_20261027 import natural_training_fit_v4 as tool


class NaturalTrainingV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = self.write('context.json', {'execution': {'admission': 'default_shared_no_permit'}})
        self.catalog = self.write('catalog.json', {'quality': {'references': {'text': {
            'pool': {'task': 'text', 'answers': ['answer']},
            'single-only': {'task': 'text', 'answers': ['answer']}}}}})
        self.reviews = [self.capture(i) for i in range(2)]
        self.clock = self.write_clock()

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def capture(self, index, trials=1, pool='pool'):
        jobs, rows = [], []
        for trial in range(trials):
            job = dict(id='j' + str(trial), split='training', inputs={'text#0': pool}, tasks={'text#0': 'text'})
            jobs.append(job)
            rid = 'r' + str(trial)
            rows.append(dict(request_id=rid, profile_job_id=job['id'], model='text#0', task='text',
                             pool_id=pool, split='training', status='completed',
                             validation_view='TRAINING_ONLY_NOT_VALIDATION', completion_time_ns=12_000_000,
                             response_reader_event_enqueue_time_ns=12_000_000,
                             worker_response=dict(task='text', prediction='answer' if index else 'wrong',
                                                  success=True, end_perf_ns=10_000_000),
                             previous_handler_next_request_ready=dict(request_id=rid, pid=123,
                                 perf_ns=11_000_000, received_time_ns=12_000_000, clock='host_perf_counter_ns',
                                 boundary='previous_handler_next_request_ready', scope='after_response_flush_before_next_read')))
        raw = raw_fixture()
        raw.update(execution_context_sha256='ctx', request_records=rows, fixture_capture=index)
        raw_path = self.write(f'raw{index}.json', raw)
        plan = self.write(f'plan{index}.json', dict(domain='natural', split='training', jobs=jobs,
                                                  execution_context_sha256='ctx'))
        return self.write(f'review{index}.json', dict(schema='gsdeploy.startup_training_collection_strict_review.v4',
            status='PASS', raw=tool.receipt(raw_path), context=tool.receipt(self.context),
            plan=tool.receipt(plan), owners={}, evaluator_sources=[tool.receipt(__file__)]))

    def write_clock(self):
        refs = [tool.receipt(self.context)]
        refs.extend(json.loads(path.read_text())['raw'] for path in self.reviews)
        return self.write('clock.json', dict(schema=tool.CLOCK_SCHEMA, sources=refs,
            clock_contract=dict(handler='host_perf_counter_ns', runtime='host_perf_counter_ns',
                worker_next_ready='host_perf_counter_ns', same_host_verified=True, source='bound by fit')))

    def run_fit(self, reviews=None, name='fit'):
        return tool.fit(self.reviews if reviews is None else reviews, self.clock, self.catalog, self.root / name)

    def test_fit_uses_joint_boundaries_and_original_quality_totals(self):
        result = self.run_fit()
        self.assertEqual(result['status'], 'FITTED')
        model = json.loads(Path(result['model']['path']).read_text())
        self.assertEqual(model['profiles'][0]['scenarios']['mean'], dict(
            parent_receipt_ms=2., worker_next_ready_ms=1., permit_release_ms=None))
        quality = json.loads(Path(result['quality_loss']['path']).read_text())
        self.assertEqual(quality['text']['text']['by_input']['pool']['numerator'], .5)
        self.assertFalse(result['physical_accuracy_validated'])

    def test_many_trials_in_one_capture_do_not_make_independent_support(self):
        self.reviews = [self.capture(0, trials=4)]
        self.clock = self.write_clock()
        result = self.run_fit()
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['profile_support'][0]['independent_capture_count'], 1)
        self.assertNotIn('model', result)
        self.assertNotIn('quality_loss', result)

    def test_single_capture_profile_is_omitted_without_losing_supported_profiles(self):
        self.reviews.append(self.capture(2, pool='single-only'))
        self.clock = self.write_clock()
        result = self.run_fit()
        self.assertEqual(result['status'], 'FITTED_PARTIAL')
        model = json.loads(Path(result['model']['path']).read_text())
        self.assertEqual([p['pool_id'] for p in model['profiles']], ['pool'])
        self.assertEqual(len(result['profile_support']), 2)

    def test_duplicate_raw_and_unfrozen_clock_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, 'independent capture'):
            self.run_fit([self.reviews[0], self.reviews[0]])
        (self.root / 'raw0.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'clock evidence'):
            self.run_fit(name='drift')

    def test_validation_and_changed_cohort_cannot_enter_training(self):
        plan_path = self.root / 'plan0.json'
        plan = json.loads(plan_path.read_text())
        plan['jobs'][0]['split'] = 'validation'
        plan_path.write_text(json.dumps(plan))
        review = json.loads(self.reviews[0].read_text())
        review['plan'] = tool.receipt(plan_path)
        self.reviews[0].write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, 'exclusively training'):
            self.run_fit()

    def test_missing_ready_is_unknown_not_zero(self):
        raw_path = self.root / 'raw0.json'
        raw = json.loads(raw_path.read_text())
        raw['request_records'][0].pop('previous_handler_next_request_ready')
        raw_path.write_text(json.dumps(raw))
        review = json.loads(self.reviews[0].read_text())
        review['raw'] = tool.receipt(raw_path)
        self.reviews[0].write_text(json.dumps(review))
        self.clock = self.write_clock()
        result = self.run_fit()
        self.assertEqual(result['status'], 'FITTED_PARTIAL')
        model = json.loads(Path(result['model']['path']).read_text())
        self.assertIsNone(model['profiles'][0]['scenarios']['mean']['worker_next_ready_ms'])


if __name__ == '__main__':
    unittest.main()
