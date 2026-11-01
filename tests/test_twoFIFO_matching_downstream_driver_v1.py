"""CPU orchestration checks; synthetic native auditors are explicit stubs."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_matching_downstream_driver_v1 as m
from tools.predictor_validation_20261027.native_collect import receipt


class MatchingDownstreamTests(unittest.TestCase):
    def put(self,root,name,value):
        path=root/name;path.write_text(json.dumps(value));return path

    def inputs(self,root):
        fit=self.put(root,'fit.json',{});forecast=self.put(root,'generated.json',{})
        resources=self.put(root,'resource.json',{});executor=self.put(root,'executor.json',{})
        queue=self.put(root,'queue.json',dict(schema=m.queued.SCHEMA,
            matching_manifest=receipt(executor),matching_output=str(root/'matching')))
        final=root/'matching'/'report.json';final.parent.mkdir()
        self.put(final.parent,final.name,{'status':'synthetic-exact15'})
        manifest=dict(independent_source_review_complete=True,sources=[],executor_manifest=receipt(executor),
            expected_continuation_report_path=str(final))
        design=dict(fit_manifest=receipt(fit),sources=[])
        return fit,forecast,resources,queue,manifest,design

    def run_case(self,root,*,complete=True,failure_at=None,mutate=False):
        fit,forecast,resources,queue,manifest,design=self.inputs(root);calls=[]
        def stage(actions,output,completed):
            action=actions[0];calls.append(action['name'])
            if failure_at==action['name']:raise RuntimeError('stage held; no retry')
            completed.append(dict(name=action['name'],status=action['statuses'][0]))
            if mutate:
                old=fit.read_bytes();fit.write_bytes(b'changed');fit.write_bytes(old)
        with patch.object(m.response,'audit',return_value=manifest), \
             patch.object(m.generated,'audit',return_value=design), \
             patch.object(m.resource,'audit',return_value=design), \
             patch.object(m.response,'_complete_final',return_value=complete):
            ref=m.run(fit,forecast,resources,queue,root/'output',writer_pid=1,run_stages=stage)
        return json.loads(Path(ref['path']).read_text()),calls

    def test_complete_runs_four_matching_commands_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_case(Path(tmp))
        self.assertEqual(calls,['fit','generated','resources','rescore'])
        self.assertEqual(result['status'],'COMPLETE_MATCHING_FOUR_CPU_TRAINING_STAGES_NOT_PREDICTOR_QUALIFICATION')
        self.assertFalse(result['GPU_started']);self.assertFalse(result['independent_prediction_accuracy_validated'])

    def test_complete_ineligible_fit_can_continue_unknown_diagnostics(self):
        actions=m.stages('fit','gen','resources','/tmp/synthetic-matching-only')
        self.assertIn('COMPLETE_MATCHING_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN',actions[1]['statuses'])
        self.assertTrue(all('twoFIFO_matching_' in action['command'][2] for action in actions))
        self.assertIn('--fit',actions[0]['command']);self.assertIn('--fit',actions[2]['command'])

    def test_foreign_or_incomplete_final_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_case(Path(tmp),complete=False)
        self.assertEqual(calls,[]);self.assertEqual(result['completed'],[])
        self.assertEqual(result['status'],'STOPPED_MATCHING_DOWNSTREAM_NO_RETRY')

    def test_stage_failure_preserves_successful_prefix_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_case(Path(tmp),failure_at='resources')
        self.assertEqual(calls,['fit','generated','resources'])
        self.assertEqual([s['name'] for s in result['completed']],['fit','generated'])
        self.assertFalse(result['automatic_retry'])

    def test_source_change_and_restore_stops_before_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls=self.run_case(Path(tmp),mutate=True)
        self.assertEqual(calls,['fit']);self.assertEqual(result['status'],'STOPPED_MATCHING_DOWNSTREAM_NO_RETRY')

    def test_partial_publication_requires_same_live_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'final.json';p.write_text('{')
            identity=dict(pid=42,start_ticks='1',command=['synthetic']);checks=[]
            def publish(_):p.write_text('{"status":"done"}')
            with patch.object(m,'writer_identity',return_value=identity),patch.object(m.time,'sleep',side_effect=publish):
                ref,doc=m.wait_for_final(p,writer_pid=42,queue_claim='stub',manifest={},
                    expected_identity=identity,poll_seconds=.01,check_sources=lambda:checks.append(1))
            self.assertEqual(doc,{'status':'done'});self.assertEqual(ref,receipt(p));self.assertEqual(len(checks),2)
            p.write_text('{')
            with patch.object(m,'writer_identity',return_value=dict(identity,start_ticks='2')):
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    m.wait_for_final(p,writer_pid=42,queue_claim='stub',manifest={},
                        expected_identity=identity,poll_seconds=.01,check_sources=lambda:None)

    def test_matching_stages_use_CPU_environment_and_reject_zero_exit_HOLD(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);action=m.stages('fit','gen','resource',root)[0];seen=[]
            def process(command,**kwargs):
                seen.append(kwargs['env']);p=Path(action['result']);p.parent.mkdir()
                p.write_text('{"status":"HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL"}')
                return SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(ValueError,'held/stopped'):
                m.execute_stages([action],root,run_command=process)
            self.assertEqual(seen[0]['CUDA_VISIBLE_DEVICES'],'')
            self.assertTrue((root/'fit-started.json').exists());self.assertFalse((root/'fit-complete.json').exists())

    def test_writer_identity_checks_real_argument_binding_and_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc=root/'proc'/'42';proc.mkdir(parents=True);(proc/'cwd').symlink_to(root)
            fit=self.put(root,'baseline.json',{});exe=self.put(root,'exe.json',{});review=self.put(root,'review.json',{})
            queue=root/'queued';queue.mkdir();target=root/'matching'
            claim=self.put(queue,'claim.json',dict(schema=m.queued.SCHEMA,source=receipt(m.queued.__file__),
                fit_manifest=receipt(fit),matching_manifest=receipt(exe),independent_review=receipt(review),matching_output=str(target)))
            command=['python','-m','tools.predictor_validation_20261027.twoFIFO_matching_after_recovery_v1',
                '--fit-manifest','baseline.json','--matching-manifest','exe.json','--independent-review','review.json',
                '--output','queued','--matching-output','matching','--controller-pid','640208','--execute']
            (proc/'cmdline').write_bytes('\0'.join(command).encode()+b'\0')
            (proc/'stat').write_text('42 (process name) '+' '.join(['S']+['0']*18+['123']))
            manifest=dict(executor_manifest=receipt(exe),expected_continuation_report_path=str(target/'report.json'))
            self.assertEqual(m.writer_identity(42,claim,manifest,proc_root=root/'proc')['start_ticks'],'123')
            command[12]='foreign';(proc/'cmdline').write_bytes('\0'.join(command).encode()+b'\0')
            with self.assertRaisesRegex(ValueError,'identity differs'):
                m.writer_identity(42,claim,manifest,proc_root=root/'proc')


if __name__=='__main__':unittest.main()
