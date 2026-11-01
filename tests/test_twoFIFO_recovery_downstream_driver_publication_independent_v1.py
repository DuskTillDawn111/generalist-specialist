"""Partial final publication waits only for the matching original writer."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_recovery_downstream_driver_independent_v1 as fixtures
from tools.predictor_validation_20261027 import twoFIFO_recovery_downstream_driver_v1 as m


class PublicationIndependentTests(unittest.TestCase):
    def test_partial_and_invalid_utf8_publish_complete_under_same_writer(self):
        for partial in (b'{"status":',b'\xff\xfe\x00'):
            with self.subTest(partial=partial),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'final.json';path.write_bytes(partial)
                original=dict(pid=123,start_ticks='11');checks=[];sleeps=[]
                document=dict(status='COMPLETE',fixture=True)
                def publish(seconds):sleeps.append(seconds);path.write_text(json.dumps(document))
                with patch.object(m,'controller_identity',return_value=original),patch.object(m.time,'sleep',side_effect=publish):
                    ref,value=m.wait_for_final(path,controller_pid=123,recovery_manifest='fixture',poll_seconds=.01,
                        check_sources=lambda:checks.append('checked'),expected_identity=original)
                self.assertEqual(value,document);self.assertEqual(ref,m.receipt(path))
                self.assertEqual(checks,['checked','checked']);self.assertEqual(sleeps,[.01])

    def test_changed_writer_identity_does_not_wait_or_accept_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'final.json';path.write_text('{')
            with patch.object(m,'controller_identity',return_value={'start_ticks':'new'}),patch.object(m.time,'sleep') as sleep:
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    m.wait_for_final(path,controller_pid=123,recovery_manifest='fixture',poll_seconds=.01,
                        check_sources=lambda:None,expected_identity={'start_ticks':'old'})
                sleep.assert_not_called()

    def test_partial_without_live_writer_records_stop_before_any_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,_=fixtures.DownstreamIndependentTests().fixture(root)
            Path(manifest['expected_continuation_report_path']).write_text('{')
            with stack,patch.object(m,'controller_identity',side_effect=FileNotFoundError('writer exited')),patch.object(m,'execute_stages') as execute,patch.object(m.time,'sleep') as sleep:
                result=m.checked(m.run(*paths,root/'driver',controller_pid=123))
                execute.assert_not_called();sleep.assert_not_called()
            self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
            self.assertIn('FileNotFoundError',result['error'])
            self.assertEqual(result['completed'],[])

    def test_pid_change_after_partial_publication_stops_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,_=fixtures.DownstreamIndependentTests().fixture(root)
            final=Path(manifest['expected_continuation_report_path']);final.unlink();old={'start_ticks':'old'}
            def partial(seconds):final.write_text('{')
            with stack,patch.object(m,'controller_identity',side_effect=[old,old,{'start_ticks':'new'}]),patch.object(m.time,'sleep',side_effect=partial) as sleep,patch.object(m,'execute_stages') as execute:
                result=m.checked(m.run(*paths,root/'driver',controller_pid=123,poll_seconds=.01))
                execute.assert_not_called();self.assertEqual(sleep.call_count,1)
            self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
            self.assertIn('identity changed',result['error'])

    def test_complete_stopped_json_after_partial_does_not_extend_wait_or_start_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,final=fixtures.DownstreamIndependentTests().fixture(root)
            path=Path(manifest['expected_continuation_report_path']);path.write_text('{')
            final['status']='STOPPED_NO_RETRY'
            def stopped(seconds):path.write_text(json.dumps(final))
            with stack,patch.object(m,'controller_identity',return_value={'stable':True}),patch.object(m.time,'sleep',side_effect=stopped) as sleep,patch.object(m,'execute_stages') as execute:
                result=m.checked(m.run(*paths,root/'driver',controller_pid=123,poll_seconds=.01))
                execute.assert_not_called();self.assertEqual(sleep.call_count,1)
            self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
            self.assertIn('incomplete/stopped',result['error'])

    def test_complete_bytes_changed_between_read_and_receipt_wait_for_same_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'final.json';path.write_text('{"version":1}')
            real=m.receipt;seen=[];checks=[]
            def changed(p):
                if Path(p)==path and not seen:
                    seen.append(True);path.write_text('{"version":2}')
                return real(p)
            with patch.object(m,'receipt',side_effect=changed),patch.object(m,'controller_identity',return_value={'stable':True}),patch.object(m.time,'sleep') as sleep:
                ref,value=m.wait_for_final(path,controller_pid=123,recovery_manifest='fixture',poll_seconds=.01,
                    check_sources=lambda:checks.append(True),expected_identity={'stable':True})
                self.assertEqual(sleep.call_count,1)
            self.assertEqual(value,{'version':2});self.assertEqual(ref,real(path));self.assertEqual(len(checks),2)

    def test_source_or_design_drift_is_not_hidden_by_partial_publication(self):
        for kind in ('source','design'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stack,paths,manifest,_=fixtures.DownstreamIndependentTests().fixture(root)
                Path(manifest['expected_continuation_report_path']).write_text('{')
                real=m.receipt;driver_reads=[]
                def receipt(p):
                    ref=real(p)
                    if Path(p).resolve()==Path(m.__file__).resolve():
                        driver_reads.append(True)
                        if len(driver_reads)>1:
                            if kind=='source':return {**ref,'sha256':'f'*64}
                            paths[1].write_text('{"changed":true}')
                    return ref
                with stack,patch.object(m,'receipt',side_effect=receipt),patch.object(m,'execute_stages') as execute,patch.object(m,'controller_identity') as identity,patch.object(m.time,'sleep') as sleep:
                    result=m.checked(m.run(*paths,root/'driver',controller_pid=123,poll_seconds=.01))
                    execute.assert_not_called();sleep.assert_not_called();identity.assert_not_called()
                self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
                self.assertIn('changed',result['error'])

    def test_published_final_change_before_stage_launch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,final=fixtures.DownstreamIndependentTests().fixture(root)
            path=Path(manifest['expected_continuation_report_path']);ref=m.receipt(path)
            def changed(*args,**kwargs):path.write_text('{"changed":true}');return ref,final
            with stack,patch.object(m,'wait_for_final',side_effect=changed),patch.object(m,'execute_stages') as execute:
                result=m.checked(m.run(*paths,root/'driver',controller_pid=123))
                execute.assert_not_called()
            self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
            self.assertIn('changed after publication',result['error'])

    def test_complete_stable_final_needs_no_live_writer_and_uses_same_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'final.json';payload=b'{"status":"published"}\n';path.write_bytes(payload)
            with patch.object(m,'controller_identity',side_effect=AssertionError('completed writer may exit')),patch.object(m.time,'sleep',side_effect=AssertionError('stable complete file must not wait')):
                ref,value=m.wait_for_final(path,controller_pid=123,recovery_manifest='fixture',poll_seconds=.01,
                    check_sources=lambda:None)
            self.assertEqual(value,{'status':'published'})
            self.assertEqual(ref['sha256'],hashlib.sha256(payload).hexdigest())

    def test_full_run_waits_through_partial_then_launches_each_cpu_stage_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);helper=fixtures.DownstreamIndependentTests()
            stack,paths,manifest,final=helper.fixture(root)
            path=Path(manifest['expected_continuation_report_path']);path.unlink()
            output=root/'driver';seen=[];helper.inject(stack,helper.runner(output,seen));sleeps=[]
            def publish(seconds):
                self.assertEqual(seen,[]);self.assertEqual(list(output.glob('*-started.json')),[])
                sleeps.append(seconds)
                path.write_text('{' if len(sleeps)==1 else json.dumps(final))
            with stack,patch.object(m,'controller_identity',return_value={'stable':True}),patch.object(m.time,'sleep',side_effect=publish):
                result=m.checked(m.run(*paths,output,controller_pid=123,poll_seconds=.01))
            self.assertEqual(len(sleeps),2);self.assertEqual(len(seen),4)
            self.assertEqual(len(result['completed']),4)
            self.assertEqual(result['status'],'COMPLETE_FOUR_CPU_TRAINING_DIAGNOSTIC_STAGES_NOT_QUALIFICATION')


if __name__=='__main__':unittest.main()
