"""Campaign partition/clock proof tests; no CUDA import or model execution."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.kernel_profile_campaign_v2 import (
    TASKS, anchors_from_records, choose_inputs, make_jobs,
)


class KernelCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def selection(self):
        rows = []
        for task in TASKS:
            for index in range(8):
                sample = {'context': 'c'*(index+1), 'question': str(index)}
                if task != 'text':
                    path = Path(self.tmp.name)/f'{task}-{index}'
                    path.write_bytes((task+str(index)).encode()*(index+1))
                    sample = {'input': str(path)}
                rows.append(dict(task_type=task, pool_id=f'{task}{index}', sample=sample))
        return choose_inputs(rows)

    def test_partitions_are_content_disjoint_and_span_short_long(self):
        selection = self.selection()
        for task in TASKS:
            training = selection['training'][task]
            validation = selection['validation'][task]
            self.assertEqual(len(training), 4)
            self.assertFalse({r['exact_input_sha256'] for r in training} & {r['exact_input_sha256'] for r in validation})
            self.assertLess(training[0]['input_size'], validation[-1]['input_size'])

    def test_actual_deployment_pairs_offsets_and_independent_seeds(self):
        selection = self.selection()
        training = make_jobs(selection, 'training', 101)
        validation = make_jobs(selection, 'validation', 202)
        self.assertTrue({'S0', 'S1', 'SS', 'SG', 'S1G', 'SSG', 'SSGO', 'SSGO_asr', 'SSGO_image'} <= {j['context'] for j in training})
        self.assertEqual({j['repeat'] for j in training}, {0, 1})
        self.assertEqual({j['validation_view'] for j in validation}, {'known_input_independent_trial', 'unseen_input'})
        self.assertFalse({j['id'] for j in training} & {j['id'] for j in validation})
        self.assertEqual(training, make_jobs(selection, 'training', 101))
        jobs = [j for j in training if j['context'] == 'SG']
        self.assertTrue(any(j['start_offsets_ms']['text#0'] > j['start_offsets_ms']['generalist_2#0'] for j in jobs))
        self.assertTrue(any(j['start_offsets_ms']['text#0'] < j['start_offsets_ms']['generalist_2#0'] for j in jobs))

    def test_duplicate_payloads_do_not_manufacture_holdout(self):
        with self.assertRaisesRegex(ValueError, 'unique inputs'):
            choose_inputs([dict(task_type='text', pool_id=str(i), sample={'context': 'same', 'question': 'same'}) for i in range(8)])

    def test_worker_anchors_pair_actual_events_and_reject_missing_end(self):
        row = {'worker_response': {'start_perf_ns': 1}, 'execution_events': [
            dict(request_id='r', kind='gpu', segment_id='native', event='start', perf_ns=10),
            dict(request_id='r', kind='gpu', segment_id='native', event='end', perf_ns=20)]}
        self.assertEqual(anchors_from_records([row], 123), [dict(pid=123, request_id='r', kind='gpu', segment_id='native', start_ns=10, end_ns=20)])
        row['execution_events'].pop()
        with self.assertRaisesRegex(ValueError, 'Unclosed'):
            anchors_from_records([row], 123)


if __name__ == '__main__':
    unittest.main()
