"""Independent complete-cohort and frozen source checks; CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_kernel_dag_response_replication_producer_v1 import ReplicationProducerTests
from tools.predictor_validation_20261027 import kernel_dag_response_replication_producer_v1 as producer
from tools.predictor_validation_20261027 import kernel_dag_repeated_instance_alignment_v1 as adapter
from tools.predictor_validation_20261027.native_collect import receipt


class IndependentReplicationTests(unittest.TestCase):
    def test_actual_cpu04_has_same_complete_context_and_all_four_in_one_job(self):
        path = Path('artifacts/predictor-kernel-dag-response-replication-producer-v1-20261029-cpu04-prepared/campaign.json')
        campaign = producer.audit(path)
        contexts = [Path(entry['context']['path']).read_bytes() for entry in campaign['runs']]
        self.assertEqual(len(contexts), 3); self.assertTrue(all(c == contexts[0] for c in contexts))
        self.assertEqual(len({entry['plan']['sha256'] for entry in campaign['runs']}), 3)
        for entry in campaign['runs']:
            plan = producer.read(entry['plan'])
            self.assertEqual(sorted(len(job['inputs']) for job in plan['jobs']), [1]*4+[2]*6+[4])
            self.assertEqual(sum(len(job['inputs']) for job in plan['jobs']), 20)
            self.assertFalse((Path(entry['run'])/'response-replication-claim.json').exists())

    def test_persistent_static_or_node_order_change_prevents_any_partial_fit_export(self):
        fixture = ReplicationProducerTests()
        design = dict(schema=adapter.SCHEMA+'.design', training_only=True,
                      validation_labels_read=False, captures=[{}, {}, {}])
        for mode in ('static', 'order'):
            members = fixture.repeated_fixture()
            common = members[2]['observed'][4]['common']
            if mode == 'static': common['nodes'][0]['static'] = {'registers_per_thread': 999}
            else: common['nodes'].append(dict(id='extra-copy', kind='cpu'))
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    patch.object(adapter, 'read', return_value=design), \
                    patch.object(adapter, 'load_member', side_effect=members), \
                    patch.object(adapter, 'implementation_sources', return_value=[]), \
                    patch.object(adapter, 'verify'), patch.object(adapter, 'receipt', return_value={'path': 'CPU-FIXTURE'}):
                result = adapter.align('CPU-FIXTURE', Path(directory)/'output')
                self.assertEqual(result['offered_instances'], 60)
                self.assertEqual(result['status'], 'UNKNOWN_FULL_INSTANCE_COHORT_RETAINED')
                self.assertEqual(result['artifacts'], {})
                self.assertFalse((Path(directory)/'output/captures.json').exists())

    def test_second_capture_plan_change_restore_during_first_claim_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); entries = []
            for index in range(3):
                run = root/str(index); run.mkdir()
                for name in ('plan.json', 'context.json'): (run/name).write_text('{}')
                entries.append(dict(capture_id=str(index), run=str(run),
                    plan=receipt(run/'plan.json'), context=receipt(run/'context.json')))
            campaign = dict(runs=entries); path = root/'campaign.json'; path.write_text(json.dumps(campaign))
            after = root/'after.json'; after.write_text('{}')
            original_write = producer.write
            def write(target, value):
                result = original_write(target, value)
                if Path(target).name == 'response-replication-claim.json':
                    watched = Path(entries[1]['context']['path']); content = watched.read_bytes()
                    watched.write_bytes(content+b' '); watched.write_bytes(content)
                return result
            with patch.object(producer, 'audit', return_value=campaign), \
                    patch.object(producer, 'require_matching_completion'), patch.object(producer, 'write', side_effect=write), \
                    patch.object(producer.collector, 'execute', return_value=dict(status='COLLECTED', GPU_started=False)):
                with self.assertRaises(ValueError): producer.execute(path, '0', receipt(after))
            self.assertTrue((root/'0/response-replication-claim.json').exists())
            self.assertFalse((root/'0/response-replication-result.json').exists())


if __name__ == '__main__': unittest.main()
