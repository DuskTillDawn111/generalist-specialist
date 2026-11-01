"""Independent wait/identity/no-retry tests; no processes or fits are launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_matching_downstream_driver_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_matching_downstream_driver_v1 as m


class IndependentMatchingDownstream(unittest.TestCase):
    def test_source_guard_precedes_even_already_complete_final_consumption(self):
        with tempfile.TemporaryDirectory() as temp:
            final=Path(temp)/'final.json';final.write_text('{"status":"complete"}')
            with patch.object(m,'writer_identity',side_effect=AssertionError('unneeded live writer')):
                with self.assertRaisesRegex(RuntimeError,'source changed'):
                    m.wait_for_final(final,writer_pid=1,queue_claim='unused',manifest={},
                        check_sources=lambda:(_ for _ in ()).throw(RuntimeError('source changed')))

    def test_partial_or_absent_final_with_dead_writer_never_sleeps(self):
        with tempfile.TemporaryDirectory() as temp:
            final=Path(temp)/'report.json'
            for partial in (False,True):
                if partial:final.write_text('{')
                with patch.object(m,'writer_identity',side_effect=FileNotFoundError('original PID gone')),\
                     patch.object(m.time,'sleep',side_effect=AssertionError('dead writer waited')):
                    with self.assertRaises(FileNotFoundError):
                        m.wait_for_final(final,writer_pid=1,queue_claim='missing',manifest={},check_sources=lambda:None)

    def test_same_PID_changed_starttime_or_command_stops_without_stage(self):
        original=dict(pid=123,start_ticks='1',command=['recorded'])
        with tempfile.TemporaryDirectory() as temp:
            for key,value in [('start_ticks','2'),('command',['reused_pid'])]:
                with patch.object(m,'writer_identity',return_value=dict(original,**{key:value})),\
                     patch.object(m.time,'sleep',side_effect=AssertionError('foreign writer waited')):
                    with self.assertRaisesRegex(ValueError,'identity changed'):
                        m.wait_for_final(Path(temp)/'missing',writer_pid=123,queue_claim='stub',manifest={},
                            check_sources=lambda:None,expected_identity=original)

    def test_real_full_final_gate_rejects_duplicate_attempt_and_wrong_executor(self):
        manifest=dict(executor_manifest=dict(path='/synthetic/executor',sha256='a'*64),
            expected_attempt_ids=['a'+str(i) for i in range(15)])
        final=dict(schema=m.response.executor.CONT_SCHEMA+'.run',manifest=manifest['executor_manifest'],
            status=m.response.intake.FINAL_STATUS,no_retries=True,numerical_fit_run=False,
            validation_labels_read=False,fit_eligible=False,automatic_fit=False,
            completed_attempts=[dict(attempt_id=x) for x in manifest['expected_attempt_ids']])
        self.assertTrue(m.response._complete_final(manifest,final))
        for kind in ('duplicate','missing','foreign','labels','automatic'):
            changed=json.loads(json.dumps(final))
            if kind=='duplicate':changed['completed_attempts'][0]=changed['completed_attempts'][1]
            elif kind=='missing':changed['completed_attempts'].pop()
            elif kind=='foreign':changed['manifest']['sha256']='b'*64
            elif kind=='labels':changed['validation_labels_read']=True
            else:changed['automatic_fit']=True
            self.assertFalse(m.response._complete_final(manifest,changed),kind)

    def test_queue_claim_change_and_restore_stops_before_second_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture=fixtures.MatchingDownstreamTests()
            fit,gen,res,queue,manifest,design=fixture.inputs(root);calls=[]
            def stage(actions,out,completed):
                calls.append(actions[0]['name']);completed.append(dict(name=actions[0]['name']))
                original=queue.read_bytes();queue.write_bytes(b'changed');queue.write_bytes(original)
            with patch.object(m.response,'audit',return_value=manifest),\
                 patch.object(m.generated,'audit',return_value=design),\
                 patch.object(m.resource,'audit',return_value=design),\
                 patch.object(m.response,'_complete_final',return_value=True):
                result=m.checked(m.run(fit,gen,res,queue,root/'out',writer_pid=1,run_stages=stage))
        self.assertEqual(calls,['fit']);self.assertEqual(result['completed'],[{'name':'fit'}])
        self.assertEqual(result['status'],'STOPPED_MATCHING_DOWNSTREAM_NO_RETRY')


if __name__=='__main__':unittest.main()
