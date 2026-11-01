"""One-attempt continuation never retries failed fits or starts a full model."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from tools.predictor_validation_20261027 import h600_training_watch as watch
from tools.predictor_validation_20261027.native_flow import receipt


class H600WatchTests(unittest.TestCase):
    def test_completed_marker_uses_frozen_collector_contract_without_phase_field(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);campaign=root/'campaign.json';campaign.write_text('{}')
            started=root/'started-main.json';started.write_text(json.dumps(dict(campaign=receipt(campaign),phase='main',created_ns=10)))
            completed=root/'completed-main.json';completed.write_text(json.dumps(dict(campaign=receipt(campaign),jobs=10,completed_ns=11)))
            suite=root/'suite.json';suite.write_text(json.dumps({'jobs':[]}))
            config=dict(completion_marker=str(completed),startup_marker=receipt(started),campaign=receipt(campaign),primary_suite=receipt(suite))
            with self.assertRaisesRegex(ValueError,'Exactly eight'):watch.audit_ready(config)
            completed.write_text(json.dumps(dict(campaign=receipt(campaign),jobs=9,completed_ns=11)))
            with self.assertRaisesRegex(ValueError,'Completion marker'):watch.audit_ready(config)

    def test_fit_commands_are_only_two_fold_and_resources_and_peer_stops_on_failure(self):
        first=Mock();first.poll.return_value=1;second=Mock();second.poll.return_value=None
        config=dict(training_protocol={'path':'train.json'},resource_protocol={'path':'resource.json'},
            service_output='cv',resource_output='resources')
        with patch.object(watch.subprocess,'Popen',side_effect=[first,second]) as popen:
            with self.assertRaisesRegex(RuntimeError,'no retry'):watch.run_fits(config)
        self.assertEqual(popen.call_count,2);second.terminate.assert_called_once()
        for call in popen.call_args_list:
            self.assertNotIn('--full',call.args[0]);self.assertNotIn('seal',call.args[0])

    def fixture(self,root):
        marker=root/'completed-main.json';marker.write_text('{}');jobs=[]
        for index in range(8):
            folder=root/str(index);folder.mkdir()
            for name in ('raw.json','scored-raw.json'):(folder/name).write_text('{}')
            jobs.append(dict(output=str(folder)))
        suite=root/'suite.json';suite.write_text(json.dumps({'jobs':jobs}))
        quality=root/'quality.json';quality.write_text('{}');threshold=root/'threshold.json';threshold.write_text('{}')
        return dict(completion_marker=str(marker),primary_suite=receipt(suite),
            service_output=str(root/'cv'),resource_output=str(root/'resources'),report_output=str(root/'report'),
            training_protocol={'path':'train.json'},resource_protocol={'path':'resources.json'},
            quality=receipt(quality),threshold_addendum=receipt(threshold),source_receipts=[])

    def test_failure_stops_without_report_and_preserves_exclusive_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=self.fixture(root);out=root/'watch'
            with patch.dict(os.environ,{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}),\
                    patch.object(watch,'configuration',return_value=config),patch.object(watch,'audit_ready',return_value=[]),\
                    patch.object(watch,'run_fits',side_effect=RuntimeError('fixed fit failed')) as fits,\
                    patch.object(watch.subprocess,'run') as report:
                with self.assertRaisesRegex(RuntimeError,'fixed fit failed'):watch.run(out)
                self.assertEqual(fits.call_count,1);report.assert_not_called()
                with self.assertRaises(FileExistsError):watch.run(out)
            self.assertEqual(json.loads((out/'failed-CPU.json').read_text())['status'],'FAILED_STOPPED_NO_RETRY')

    def test_success_writes_completion_and_report_uses_matching_resource_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=self.fixture(root);out=root/'watch'
            def report(command,**kwargs):
                report_root=Path(config['report_output']);report_root.mkdir()
                (report_root/'report.json').write_text(json.dumps(dict(requested=9600,known=9599,
                    original_threshold_assessment={'original_training_cv_thresholds_passed':False})))
                self.assertIn('--resource-manifest',command);self.assertNotIn('--full',command)
            with patch.dict(os.environ,{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}),\
                    patch.object(watch,'configuration',return_value=config),patch.object(watch,'audit_ready',return_value=[]),\
                    patch.object(watch,'run_fits') as fits,patch.object(watch.subprocess,'run',side_effect=report) as reporter:
                watch.run(out);fits.assert_called_once();reporter.assert_called_once()
            completion=json.loads((out/'completed-CPU.json').read_text())
            self.assertEqual(completion['known'],9599);self.assertFalse(completion['full_fit_performed'])
            self.assertFalse(completion['original_threshold_assessment']['original_training_cv_thresholds_passed'])


if __name__=='__main__':unittest.main()
