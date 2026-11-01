import copy
import unittest

from tools.predictor_validation_20261027.profile_training_evidence_review_v1 import inspect_cohort
from tools.predictor_validation_20261027.profiling import RESIDENTS


def fixture():
    jobs, rows, batches = [], [], []
    for index in range(26):
        jid = 'job:' + str(index)
        start = 1000 + index * 100
        job = dict(id=jid, inputs={a: 'ocr-input' if a == 'ocr#0' else 'text-input' for a in RESIDENTS},
            tasks={a: 'ocr' if a == 'ocr#0' else 'text' for a in RESIDENTS},
            start_offsets_ms=dict.fromkeys(RESIDENTS, 0), split='training', validation_view='training')
        jobs.append(job)
        ids = []
        for alias in RESIDENTS:
            rid = jid + ':' + alias
            ids.append(rid)
            rows.append(dict(request_id=rid, model=alias, pool_id=job['inputs'][alias], task=job['tasks'][alias],
                profile_job_id=jid, split='training', validation_view='training', status='completed',
                scheduled_time_ns=start, response_reader_event_enqueue_time_ns=start+50,
                completion_time_ns=start+50, worker_response={'success': True}))
        batches.append(dict(job_id=jid, start_ns=start, end_ns=start+60, request_ids=ids))
    priming = [dict(request_id='prime:'+a, model=a, task='ocr' if a == 'ocr#0' else 'text',
        pool_id='ocr-input' if a == 'ocr#0' else 'text-input', status='completed',
        worker_response={'success': True}) for a in RESIDENTS]
    plan = dict(split='training', jobs=jobs, inputs=[dict(task='text', pool_id='text-input', input_size=1),
        dict(task='ocr', pool_id='ocr-input', input_size=1)])
    for row in [*rows, *priming]:
        start = row.setdefault('scheduled_time_ns', 100)
        end = row.setdefault('completion_time_ns', start+50)
        row.update(enqueued_time_ns=start, actual_submit_time_ns=start+1,
            submit_write_completed_time_ns=start+3, observed_worker_start_time_ns=start+2,
            response_reader_event_enqueue_time_ns=end, response_collector_dequeued_time_ns=end+1,
            response_collector_processing_time_ns=end+2)
        row['worker_response'].update(start_perf_ns=start+2, end_perf_ns=end-1)
    raw = dict(status='completed', errors=[], runtime_errors=[], telemetry_errors=[], request_records=rows,
        priming_request_records=priming, batches=batches,
        ready={a:dict(pid=100+i) for i,a in enumerate(RESIDENTS)},
        shutdown={a:dict(exit_code=0, forced=False, bye_received=True) for a in RESIDENTS})
    return plan, raw


class TrainingEvidenceTests(unittest.TestCase):
    def test_complete_declared_cohort(self):
        plan, raw = fixture()
        before = copy.deepcopy((plan, raw))
        result = inspect_cohort(plan, raw)
        self.assertEqual(result['jobs'], 26)
        self.assertEqual(result['measured_requests'], 104)
        self.assertEqual(result['priming_requests'], 4)
        self.assertEqual((plan, raw), before)

    def test_missing_or_duplicate_request_rejected(self):
        for mode in ('missing', 'duplicate'):
            plan, raw = fixture()
            if mode == 'missing':raw['request_records'].pop()
            else:raw['request_records'].append(copy.deepcopy(raw['request_records'][0]))
            with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_changed_input_route_task_or_role_rejected(self):
        for key, value in [('pool_id', 'other'), ('model', 'other'), ('task', 'asr'),
                           ('split', 'validation'), ('validation_view', 'unseen_input')]:
            with self.subTest(key=key):
                plan, raw = fixture()
                raw['request_records'][0][key] = value
                with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_exact_arrival_parent_receipt_and_job_boundary(self):
        for key, value in [('scheduled_time_ns', 1001), ('completion_time_ns', 1049),
                           ('response_reader_event_enqueue_time_ns', 2000), ('status', 'failed')]:
            plan, raw = fixture()
            raw['request_records'][0][key] = value
            with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_order_and_batch_identity_rejected(self):
        plan, raw = fixture()
        raw['batches'].reverse()
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)
        plan, raw = fixture()
        raw['batches'][0]['request_ids'].pop()
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_priming_never_dropped_or_relabelled_measured(self):
        plan, raw = fixture()
        raw['priming_request_records'].pop()
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)
        plan, raw = fixture()
        raw['priming_request_records'][0]['request_id'] = raw['request_records'][0]['request_id']
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_shutdown_failure_and_telemetry_error_rejected(self):
        plan, raw = fixture()
        raw['telemetry_errors'].append('unknown_pid')
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_completed_transport_does_not_imply_successful_work(self):
        for key in ('request_records', 'priming_request_records'):
            for value in (False, None, 1):
                plan, raw = fixture()
                raw[key][0]['worker_response']['success'] = value
                with self.assertRaises(ValueError):inspect_cohort(plan, raw)
        plan, raw = fixture()
        raw['shutdown']['text#0']['forced'] = True
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)

    def test_other_roles_and_partial_pilot_rejected(self):
        plan, raw = fixture()
        plan['split'] = 'smoke'
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)
        plan, raw = fixture()
        plan['jobs'].pop()
        with self.assertRaises(ValueError):inspect_cohort(plan, raw)


if __name__ == '__main__':
    unittest.main()
