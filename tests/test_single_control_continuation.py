"""CPU-only scheduling gates: no queues, terminal real data or GPU needed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('single_continuation',ROOT/'tools/spillover_context_20261022/single_control_continuation.py')
f=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(f)


class Continuation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'report.json'
        self.ids=[f'j{i}' for i in range(6)]
        self.report=dict(observations=[dict(job_id=j,audit_status='VERIFIED_TERMINAL_EVIDENCE',raw_status='completed',
            memory_pass=True,energy_pass=True,shutdown_pass=True,utility=dict(eligible=False,U=None)) for j in self.ids])
        self.path.write_text(json.dumps(self.report))
        self.progress=dict(status='COMPLETE',comparison=f.base.receipt(self.path),stages=[dict(exit_code=0) for _ in range(4)])

    def test_wait_statuses(self):
        for status in ['STARTING','WAITING_FINE_TERMINALS','RUNNING_STRONG_GPU']:
            self.assertFalse(f.predecessor_status(dict(status=status)))
        self.assertTrue(f.predecessor_status(self.progress))

    def test_stop_and_unknown_never_continue(self):
        for status in ['STOPPED_NO_RETRY',None,'COMPLETE_BUT_BAD']:
            with self.assertRaises(RuntimeError):f.predecessor_status(dict(status=status))

    def test_complete_valid_six_does_not_require_utility_or_eligibility(self):
        f.predecessor_audit(self.progress,self.report,self.ids,self.path)

    def test_pending_cannot_use_even_good_report(self):
        progress={**self.progress,'status':'RUNNING_STRONG_GPU'}
        with self.assertRaises(RuntimeError):f.predecessor_audit(progress,self.report,self.ids,self.path)

    def test_changed_receipt_stops(self):
        self.path.write_text('{}')
        with self.assertRaises(ValueError):f.predecessor_audit(self.progress,self.report,self.ids,self.path)

    def test_incomplete_duplicate_and_failed_audits_stop(self):
        variants=[]
        for field in ['memory_pass','energy_pass','shutdown_pass']:
            report=copy.deepcopy(self.report);report['observations'][2][field]=False;variants.append(report)
        report=copy.deepcopy(self.report);report['observations'][2]['job_id']='j0';variants.append(report)
        report=copy.deepcopy(self.report);report['observations'].pop();variants.append(report)
        for report in variants:
            with self.assertRaises(RuntimeError):f.predecessor_audit(self.progress,report,self.ids,self.path)

    def test_failed_predecessor_stage_stops(self):
        self.progress['stages'][1]['exit_code']=1
        with self.assertRaises(RuntimeError):f.predecessor_audit(self.progress,self.report,self.ids,self.path)

    def test_wait_does_not_read_results_before_complete(self):
        progress=Path(self.temp.name)/'progress.json';progress.write_text(json.dumps(dict(status='RUNNING_STRONG_GPU')))
        prior_path=Path(self.temp.name)/'request.json';prior_path.write_text('{}')
        execution=Path(self.temp.name)/'new-execution';execution.mkdir()
        request=dict(execution=str(execution),maximum_seconds=60,predecessor=dict(path=str(prior_path)),
            predecessor_progress=str(progress),wait_seconds=5)
        supervisor=f.Supervisor(request)
        with patch.object(f.base,'freeze_valid'),patch.object(f.time,'sleep',side_effect=RuntimeError('test wait boundary')):
            with self.assertRaisesRegex(RuntimeError,'test wait boundary'):supervisor.wait_predecessor()
        # No source queue/report location is even declared, so an eager read fails this test.


if __name__=='__main__':unittest.main()
