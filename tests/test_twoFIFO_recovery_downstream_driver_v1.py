import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_recovery_downstream_driver_v1 as m


class DownstreamDriverTests(unittest.TestCase):
    def actions(self, root):
        return [dict(name='step'+str(i),command=['cpu-step',str(i)],result=str(root/str(i)/'report.json'),
            statuses=['PASS']) for i in range(2)]

    def test_command_order_and_outputs_match_the_predeclared_four_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            actions=m.stages('fit-manifest','generated-design','resource-design',directory)
            self.assertEqual([a['name'] for a in actions],['fit','generated','resources','rescore'])
            self.assertIn('--fit',actions[0]['command']);self.assertIn('--fit',actions[2]['command'])
            self.assertNotIn('--fit',actions[1]['command'])
            self.assertIn('model.json',actions[2]['result'])
            self.assertTrue(all('_child' not in a['command'] and '--execute' not in a['command'] for a in actions))

    def test_cpu_only_stage_success_is_persisted_without_overwriting_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actions=self.actions(root);seen=[]
            def run(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                seen.append(command)
                target=Path(actions[len(seen)-1]['result']);target.parent.mkdir()
                target.write_text(json.dumps(dict(status='PASS')))
                return SimpleNamespace(returncode=0)
            result=m.execute_stages(actions,root,run_command=run)
            self.assertEqual(len(result),2);self.assertEqual(len(seen),2)
            self.assertTrue((root/'step0-complete.json').is_file())
            with self.assertRaisesRegex(ValueError,'Never reuse'):
                m.execute_stages(actions,root,run_command=run)
            self.assertEqual(len(seen),2)

    def test_zero_exit_with_hold_status_stops_and_never_launches_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actions=self.actions(root);seen=[]
            def run(command, **kwargs):
                seen.append(command);target=Path(actions[0]['result']);target.parent.mkdir()
                target.write_text(json.dumps(dict(status='STOPPED_NO_RETRY')))
                return SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(ValueError,'held/stopped'):
                m.execute_stages(actions,root,run_command=run)
            self.assertEqual(len(seen),1)
            self.assertFalse((root/'step0-complete.json').exists())

    def test_failed_later_stage_keeps_completed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actions=self.actions(root);seen=[];completed=[]
            def run(command, **kwargs):
                seen.append(command)
                if len(seen)==2:return SimpleNamespace(returncode=1)
                target=Path(actions[0]['result']);target.parent.mkdir()
                target.write_text(json.dumps(dict(status='PASS')))
                return SimpleNamespace(returncode=0)
            with self.assertRaisesRegex(ValueError,'failed'):
                m.execute_stages(actions,root,run_command=run,completed=completed)
            self.assertEqual(len(completed),1);self.assertEqual(len(seen),2)

    def test_stopped_capture_final_cannot_start_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);final=root/'final.json';final.write_text('{"status":"STOPPED_NO_RETRY"}')
            fit=root/'fit-manifest.json';fit.write_text('{}')
            gen=root/'generated-design.json';gen.write_text('{}')
            resource=root/'resource-design.json';resource.write_text('{}')
            manifest=dict(expected_continuation_report_path=str(final))
            design=dict(fit_manifest=m.receipt(fit))
            with patch.object(m.response,'audit',return_value=manifest), \
                 patch.object(m.generated,'audit',return_value=design), \
                 patch.object(m.resource,'audit',return_value=design), \
                 patch.object(m.response,'complete_final',return_value=False), \
                 patch.object(m,'execute_stages') as execute:
                ref=m.run(fit,gen,resource,root/'run',controller_pid=1)
            execute.assert_not_called()
            result=m.checked(ref)
            self.assertEqual(result['status'],'STOPPED_DOWNSTREAM_NO_RETRY')
            self.assertEqual(result['completed'],[])
            self.assertFalse(result['GPU_started'])


if __name__=='__main__':unittest.main()
