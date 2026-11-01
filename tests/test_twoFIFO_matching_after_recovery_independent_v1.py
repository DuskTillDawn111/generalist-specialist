"""Independent CPU checks: actual wait/rank-free metadata gate and source guard.

All native auditors and the matching subprocess are explicit CPU doubles.
The wait, final metadata check, file publication and inotify guard remain real.
"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_matching_after_recovery_v1 as author_fixtures
from tests import test_twoFIFO_recovery_phase12_independent_v1 as recovery_fixtures
from tools.predictor_validation_20261027 import twoFIFO_matching_after_recovery_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_recovery_downstream_driver_v1 as waiting


class IndependentAfterRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        manifest, review, fit, final_path, matching, baseline = author_fixtures.MatchingAfterRecoveryTests().setup_case(root)
        recovery_root = root/'metadata'; recovery_root.mkdir()
        metadata, recovered, final, _ = recovery_fixtures.IndependentRecoveryPhase12Tests().fixture(recovery_root)
        metadata['expected_continuation_report_path'] = str(final_path)
        final_path.write_text(json.dumps(final))
        own_source = root/'driver-source.py'; own_source.write_text('# frozen CPU fixture\n')
        target = root/'matching-run'; output = root/'driver'; calls=[]
        def process(command, **kwargs):
            calls.append((command, kwargs))
            target.mkdir()
            (target/'report.json').write_text(json.dumps(dict(
                schema=m.matching.CONT_SCHEMA+'.run', manifest=m.receipt(manifest),
                status='COMPLETE_ALL_FIFTEEN_MATCHING_NATIVE_ATTEMPTS_REVIEWED_NO_FIT',
                completed_attempts=[{}]*15, numerical_fit_run=False)))
            return SimpleNamespace(returncode=0)
        stack=ExitStack()
        stack.enter_context(patch.object(m.matching,'audit',return_value=matching))
        stack.enter_context(patch.object(m.baseline,'audit',return_value=metadata))
        stack.enter_context(patch.object(m.baseline.recovery,'audit',return_value=recovered))
        stack.enter_context(patch.object(m,'implementation_sources',return_value=[m.receipt(own_source)]))
        identity=dict(pid=123,start_ticks='stable-original-writer')
        stack.enter_context(patch.object(m,'controller_identity',return_value=identity))
        stack.enter_context(patch.object(waiting,'controller_identity',return_value=identity))
        args=(fit,manifest,review,output,target)
        return SimpleNamespace(**locals())

    def run_fixture(self, f, process=None):
        ref=m.run(*f.args,controller_pid=123,poll_seconds=.01,run_command=process or f.process)
        return json.loads(Path(ref['path']).read_text())

    def test_waits_partial_publication_without_matching_claims_and_launches_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory)); f.final_path.unlink(); sleeps=[]
            def publish(seconds):
                self.assertEqual(f.calls,[])
                self.assertFalse(f.target.exists())
                self.assertTrue(all(not Path(a['claim']).exists() and not Path(a['collection']).exists()
                                    for a in f.matching['attempts']))
                self.assertFalse((f.output/'matching-started.json').exists())
                sleeps.append(seconds)
                f.final_path.write_text('{' if len(sleeps)==1 else json.dumps(f.final))
            with f.stack,patch.object(waiting.time,'sleep',side_effect=publish):
                result=self.run_fixture(f)
            self.assertEqual(len(sleeps),2);self.assertEqual(len(f.calls),1)
            self.assertTrue(result['status'].startswith('COMPLETE_MATCHING_CAPTURE'))
            self.assertNotIn('env',f.calls[0][1]);self.assertFalse(result['numerical_fit_run'])

    def test_changed_or_dead_writer_without_published_final_cannot_launch(self):
        for dead in (False,True):
            with self.subTest(dead=dead),tempfile.TemporaryDirectory() as directory:
                f=self.fixture(Path(directory));f.final_path.unlink()
                side=FileNotFoundError('writer exited') if dead else None
                with f.stack,patch.object(waiting,'controller_identity',return_value={'start_ticks':'reused'},side_effect=side),patch.object(waiting.time,'sleep') as sleep:
                    result=self.run_fixture(f)
                self.assertEqual(f.calls,[]);sleep.assert_not_called()
                self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_actual_exact12_gate_rejects_stopped_foreign_missing_or_reordered_final(self):
        changes=[{'status':'STOPPED_NO_RETRY'},{'recovery_manifest':{'path':'foreign','sha256':'f'*64}},
                 {'numerical_fit_run':True},{'validation_labels_read':True}]
        for mode in ('missing','reordered','tail'):
            changes.append({'mode':mode})
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                f=self.fixture(Path(directory)); final=copy.deepcopy(f.final);change=dict(change)
                mode=change.pop('mode',None)
                if mode=='missing':final['completed_attempts'].pop()
                if mode=='reordered':final['completed_attempts'].reverse()
                if mode=='tail':final['completed_tail'].pop()
                final.update(change);f.final_path.write_text(json.dumps(final))
                with f.stack:result=self.run_fixture(f)
                self.assertEqual(f.calls,[]);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_racing_claim_collection_or_output_while_waiting_prevents_launch(self):
        for kind in ('claim','collection','output'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                f=self.fixture(Path(directory));f.final_path.unlink()
                def race(seconds):
                    if kind=='claim':Path(f.matching['attempts'][14]['claim']).write_text('another process')
                    elif kind=='collection':Path(f.matching['attempts'][14]['collection']).mkdir()
                    else:f.target.mkdir()
                    f.final_path.write_text(json.dumps(f.final))
                with f.stack,patch.object(waiting.time,'sleep',side_effect=race):result=self.run_fixture(f)
                self.assertEqual(f.calls,[]);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_future_plan_own_source_and_review_change_restore_during_wait_are_rejected(self):
        for kind in ('plan','own-source','review'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                f=self.fixture(Path(directory));f.final_path.unlink()
                path={'plan':Path(f.matching['attempts'][14]['plan']['path']),
                      'own-source':f.own_source,'review':f.review}[kind]
                def mutate(seconds):
                    old=path.read_bytes();path.write_bytes(b'changed');path.write_bytes(old)
                    f.final_path.write_text(json.dumps(f.final))
                with f.stack,patch.object(waiting.time,'sleep',side_effect=mutate):result=self.run_fixture(f)
                self.assertEqual(f.calls,[]);self.assertEqual(result['status'],'STOPPED_NO_RETRY')
                self.assertIn('source',result['error'].lower())

    def test_final_changed_after_metadata_gate_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory)); original=m.baseline.complete_final
            def changed(*args):
                valid=original(*args);f.final_path.write_text('{}');return valid
            with f.stack,patch.object(m.baseline,'complete_final',side_effect=changed):result=self.run_fixture(f)
            self.assertEqual(f.calls,[]);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_final_drift_during_child_stops_without_second_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory))
            def changed(*args,**kwargs):
                done=f.process(*args,**kwargs);f.final_path.write_text('{}');return done
            with f.stack:result=self.run_fixture(f,changed)
            self.assertEqual(len(f.calls),1);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_source_guard_is_live_before_second_full_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));audits=[]
            def audit(*args):
                audits.append(True)
                if len(audits)==2:
                    path=Path(f.matching['attempts'][0]['plan']['path']);old=path.read_bytes()
                    path.write_bytes(b'changed');path.write_bytes(old)
                return f.matching
            with f.stack,patch.object(m.matching,'audit',side_effect=audit):result=self.run_fixture(f)
            self.assertEqual(f.calls,[]);self.assertEqual(result['status'],'STOPPED_NO_RETRY')

    def test_reusing_driver_output_cannot_invoke_matching_again(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory))
            with f.stack:
                self.run_fixture(f)
                with self.assertRaises(FileExistsError):self.run_fixture(f)
            self.assertEqual(len(f.calls),1)

    def test_invalid_poll_interval_rejected_before_audit_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory))
            for value in (0,-1,61,float('nan'),float('inf'),True):
                with self.subTest(value=value),f.stack,patch.object(m.matching,'audit') as audit:
                    with self.assertRaises(ValueError):m.run(*f.args,controller_pid=123,poll_seconds=value,run_command=f.process)
                    audit.assert_not_called()
            self.assertFalse(f.output.exists());self.assertEqual(f.calls,[])


if __name__=='__main__':unittest.main()
