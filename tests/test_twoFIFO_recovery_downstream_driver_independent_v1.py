"""Isolated CPU driver review. No native intake, subprocess, or fit is run."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_recovery_phase12_independent_v1 as phase_fixtures
from tools.predictor_validation_20261027 import twoFIFO_recovery_downstream_driver_v1 as m


class DownstreamIndependentTests(unittest.TestCase):
    def fixture(self, root):
        manifest,recovered,final,_=phase_fixtures.IndependentRecoveryPhase12Tests().fixture(root)
        paths=[]
        for name in ('fit-manifest','generated-design','resource-design'):
            p=root/(name+'.json');p.write_text(json.dumps({'fixture':name}));paths.append(p)
        design=dict(fit_manifest=m.receipt(paths[0]))
        stack=ExitStack()
        stack.enter_context(patch.object(m.response,'audit',return_value=manifest))
        stack.enter_context(patch.object(m.response.recovery,'audit',return_value=recovered))
        stack.enter_context(patch.object(m.generated,'audit',return_value=design))
        stack.enter_context(patch.object(m.resource,'audit',return_value=design))
        stack.enter_context(patch.object(m.response.numeric,'fit_bundle',side_effect=AssertionError('real fit forbidden')))
        return stack,paths,manifest,final

    def runner(self, output, seen, *, fail_at=None, fail_kind='exit'):
        actions=m.stages('unused-fit','unused-gen','unused-res',output)
        def run(command,**kw):
            seen.append(command)
            self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertEqual(kw['env']['OPENBLAS_NUM_THREADS'],'1')
            self.assertFalse(kw['check'])
            i=len(seen)-1;action=actions[i]
            self.assertTrue((output/(action['name']+'-started.json')).exists())
            if i==fail_at and fail_kind=='spawn':raise OSError('synthetic spawn failure')
            if i==fail_at and fail_kind=='exit':return SimpleNamespace(returncode=7)
            target=Path(action['result']);target.parent.mkdir()
            if i==fail_at and fail_kind=='missing':return SimpleNamespace(returncode=0)
            target.write_text('{' if i==fail_at and fail_kind=='json' else json.dumps(dict(
                status='STOPPED_SYNTHETIC_NO_RETRY' if i==fail_at else action['statuses'][0])))
            return SimpleNamespace(returncode=0)
        return run

    def inject(self,stack,runner):
        original=m.execute_stages
        stack.enter_context(patch.object(m,'execute_stages',side_effect=lambda a,o,**kw:
            original(a,o,run_command=runner,**kw)))

    def test_waiting_controller_to_exact_final_runs_four_cpu_stages_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,final=self.fixture(root)
            final_path=Path(manifest['expected_continuation_report_path']);final_path.unlink()
            output=root/'driver';seen=[];self.inject(stack,self.runner(output,seen))
            identity=dict(pid=123,command=['synthetic-reviewed-controller'],start_ticks='111')
            def finish(seconds):
                self.assertLessEqual(seconds,60);final_path.write_text(json.dumps(final))
            with stack,patch.object(m,'controller_identity',return_value=identity),patch.object(m.time,'sleep',side_effect=finish):
                ref=m.run(*paths,output,controller_pid=123,poll_seconds=.01)
                with self.assertRaises(FileExistsError):m.run(*paths,output,controller_pid=123)
            report=m.checked(ref)
            self.assertEqual(len(seen),4);self.assertEqual([r['name'] for r in report['completed']],['fit','generated','resources','rescore'])
            self.assertEqual(report['status'],'COMPLETE_FOUR_CPU_TRAINING_DIAGNOSTIC_STAGES_NOT_QUALIFICATION')
            self.assertFalse(report['GPU_started']);self.assertFalse(report['automatic_retry'])
            self.assertFalse(report['independent_prediction_accuracy_validated'])
            for action in m.checked(m.receipt(output/'claim.json'))['actions']:
                self.assertTrue((output/(action['name']+'-complete.json')).exists())

    def test_real_complete_final_gate_rejects_stopped_foreign_or_short_tail(self):
        for kind in ('stopped','foreign','short-tail'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stack,paths,manifest,final=self.fixture(root)
                if kind=='stopped':final['status']='STOPPED_NO_RETRY'
                elif kind=='foreign':final['schema']='old.executor.run'
                else:final['completed_tail'].pop()
                Path(manifest['expected_continuation_report_path']).write_text(json.dumps(final))
                seen=[];self.inject(stack,self.runner(root/'driver',seen))
                with stack:ref=m.run(*paths,root/'driver',controller_pid=123)
                self.assertEqual(seen,[]);self.assertEqual(m.checked(ref)['status'],'STOPPED_DOWNSTREAM_NO_RETRY')

    def test_exit_zero_hold_missing_or_invalid_report_preserves_completed_prefix(self):
        for kind in ('hold','missing','json','exit','spawn'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stack,paths,_,_=self.fixture(root);seen=[];output=root/'driver'
                self.inject(stack,self.runner(output,seen,fail_at=1,fail_kind=kind))
                with stack:ref=m.run(*paths,output,controller_pid=123)
                report=m.checked(ref)
                self.assertEqual(len(seen),2);self.assertEqual([r['name'] for r in report['completed']],['fit'])
                self.assertEqual(report['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
                self.assertTrue((output/'fit-complete.json').exists())
                self.assertTrue((output/'generated-started.json').exists())
                self.assertFalse((output/'generated-complete.json').exists())
                self.assertFalse((output/'resources-started.json').exists())

    def test_existing_output_or_started_marker_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actions=m.stages('a','b','c',root);seen=[]
            Path(actions[0]['result']).parent.mkdir();sentinel=Path(actions[0]['result']);sentinel.write_text('unchanged')
            with self.assertRaisesRegex(ValueError,'Never reuse'):
                m.execute_stages(actions,root,run_command=self.runner(root,seen))
            self.assertEqual(sentinel.read_text(),'unchanged');self.assertEqual(seen,[])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actions=m.stages('a','b','c',root);seen=[]
            runner=self.runner(root,seen,fail_at=0,fail_kind='spawn')
            with self.assertRaises(OSError):m.execute_stages(actions,root,run_command=runner)
            with self.assertRaises(FileExistsError):m.execute_stages(actions,root,run_command=runner)
            self.assertEqual(len(seen),1)

    def test_waiting_pid_identity_change_stops_before_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,_=self.fixture(root)
            Path(manifest['expected_continuation_report_path']).unlink();seen=[];self.inject(stack,self.runner(root/'driver',seen))
            with stack,patch.object(m,'controller_identity',side_effect=[{'start_ticks':'1'},{'start_ticks':'2'}]),patch.object(m.time,'sleep',side_effect=AssertionError('must not wait')):
                ref=m.run(*paths,root/'driver',controller_pid=123)
            self.assertEqual(seen,[]);self.assertIn('identity changed',m.checked(ref)['error'])

    def test_waiting_input_and_driver_source_changes_stop_without_fit(self):
        for kind in ('design','source'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stack,paths,manifest,_=self.fixture(root)
                Path(manifest['expected_continuation_report_path']).unlink();seen=[];self.inject(stack,self.runner(root/'driver',seen))
                real_receipt=m.receipt;state={'mutate':False}
                def receipt(p):
                    r=real_receipt(p)
                    if kind=='source' and state['mutate'] and Path(p).resolve()==Path(m.__file__).resolve():r={**r,'sha256':'f'*64}
                    return r
                def sleep(seconds):
                    state['mutate']=True
                    if kind=='design':paths[1].write_text('{"changed":true}')
                with stack,patch.object(m,'controller_identity',return_value={'stable':True}),patch.object(m,'receipt',side_effect=receipt),patch.object(m.time,'sleep',side_effect=sleep):
                    ref=m.run(*paths,root/'driver',controller_pid=123,poll_seconds=.01)
                self.assertEqual(seen,[]);self.assertEqual(m.checked(ref)['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
                self.assertIn('changed',m.checked(ref)['error'])

    def test_missing_controller_is_recorded_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stack,paths,manifest,_=self.fixture(root)
            Path(manifest['expected_continuation_report_path']).unlink();seen=[];self.inject(stack,self.runner(root/'driver',seen))
            with stack,patch.object(m,'controller_identity',side_effect=FileNotFoundError('controller exited')):
                ref=m.run(*paths,root/'driver',controller_pid=123)
            self.assertEqual(seen,[]);self.assertIn('FileNotFoundError',m.checked(ref)['error'])

    def test_invalid_pid_and_poll_parameters_fail_before_output_claim(self):
        for pid,poll in [(0,30),(-1,30),(True,30),(1,0),(1,61),(1,float('nan'))]:
            with self.subTest(pid=pid,poll=poll),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);output=root/'out'
                with self.assertRaises(ValueError):m.run('a','b','c',output,controller_pid=pid,poll_seconds=poll)
                self.assertFalse(output.exists())

    def test_each_failed_stage_retains_exact_completed_prefix_and_no_later_start(self):
        names=['fit','generated','resources','rescore']
        for index in (0,2,3):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);stack,paths,_,_=self.fixture(root);seen=[];output=root/'driver'
                self.inject(stack,self.runner(output,seen,fail_at=index))
                with stack:ref=m.run(*paths,output,controller_pid=123)
                report=m.checked(ref)
                self.assertEqual([r['name'] for r in report['completed']],names[:index])
                self.assertEqual(len(seen),index+1)
                for name in names[index+1:]:self.assertFalse((output/(name+'-started.json')).exists())

    def test_proc_identity_parser_requires_exact_controller_manifest_and_starttime(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);proc=root/'proc'/'123';proc.mkdir(parents=True)
            manifest=root/'manifest.json';manifest.write_text('{}');(proc/'cwd').symlink_to(root)
            command=['python','-m','tools.predictor_validation_20261027.twoFIFO_prelaunch_recovery_v1',
                'run','--manifest','manifest.json','--execute']
            def write_command(c):(proc/'cmdline').write_bytes('\0'.join(c).encode()+b'\0')
            write_command(command)
            (proc/'stat').write_text('123 (worker name with spaces) '+' '.join(['S']+['0']*18+['12345']))
            with patch.object(m,'Path',side_effect=lambda p:root/'proc' if p=='/proc' else Path(p)):
                original=m.controller_identity(123,manifest)
                self.assertEqual(original['start_ticks'],'12345')
                (proc/'stat').write_text('123 (worker) '+' '.join(['S']+['0']*18+['99999']))
                self.assertNotEqual(m.controller_identity(123,manifest),original)
                write_command([*command[:-1],'--other'])
                with self.assertRaisesRegex(ValueError,'Exact running'):m.controller_identity(123,manifest)
                write_command([*command[:5],'foreign.json','--execute'])
                with self.assertRaisesRegex(ValueError,'foreign recovery manifest'):m.controller_identity(123,manifest)


if __name__=='__main__':unittest.main()
