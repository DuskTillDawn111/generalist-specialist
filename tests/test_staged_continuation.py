"""Continuation never launches strong GPU before exact terminal/audit gates."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_staged_test',ROOT/'tools/spillover_context_20261022/staged_continuation.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class StagedContinuationTests(unittest.TestCase):
    def fixture(self,directory,status='RUNNING',count=0):
        root=Path(directory);packetpath=root/'packet.json';packet={'jobs':[{'id':str(i)} for i in range(4)]}
        packetpath.write_text(json.dumps(packet));binding=stage.receipt(packetpath);runner={'path':'runner','sha256':'frozen'}
        queue={'packet':binding,'source':runner,'status':status,'jobs':[],'current':{'raw':'MUST_NOT_READ_LIVE'}}
        for i in range(count):
            raw=root/str(i)/'run.json';raw.parent.mkdir();raw.write_text('not parsed by supervisor')
            queue['jobs'].append({'job_id':str(i),'status':'completed','exit_code':0,'raw':stage.receipt(raw)})
        queuepath=root/'queue.json';queuepath.write_text(json.dumps(queue))
        request=dict(fine_packet=binding,fine_queue=str(queuepath),runner=runner,execution=str(root/'execution'),
            maximum_seconds=60,wait_seconds=5,outputs={'fine_audit':str(root/'audit')},
            stages=[{'name':'fine_audit'},{'name':'strong_gpu'}])
        return request,packet,queue

    def run_fixture(self,request,sleep=None,audit=None):
        supervisor=stage.Supervisor(request);launched=[]
        def run(item):
            launched.append(item['name'])
            if item['name']=='fine_audit':
                path=Path(request['outputs']['fine_audit']);path.mkdir();(path/'report.json').write_text(json.dumps(audit))
        with patch.object(stage,'freeze_valid'),patch.object(supervisor,'run_stage',side_effect=run),patch.object(stage.time,'sleep',side_effect=sleep):
            result=supervisor.execute()
        return result,launched,supervisor.state

    def test_running_or_terminal_missing_jobs_cannot_launch_gpu(self):
        for status,count in [('RUNNING',3),(stage.TERMINAL,3)]:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                request,_,_=self.fixture(tmp,status,count)
                code,launched,state=self.run_fixture(request,sleep=TimeoutError('bounded test wait'))
                self.assertEqual(code,1);self.assertEqual(launched,[]);self.assertEqual(state['status'],'STOPPED_NO_RETRY')

    def test_stopped_queue_never_audits_or_launches_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            request,_,_=self.fixture(tmp,'STOPPED_FOR_DIAGNOSIS_NO_RETRY',4)
            _,launched,state=self.run_fixture(request)
            self.assertEqual(launched,[]);self.assertIn('Queue stopped',state['error'])

    def test_audit_failure_or_resource_failure_prevents_gpu(self):
        for failure in ['status','memory_pass','energy_pass','shutdown_pass']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                request,_,_=self.fixture(tmp,stage.TERMINAL,4)
                jobs=[dict(job_id=str(i),audit_status='VERIFIED_TERMINAL_EVIDENCE',raw_status='completed',
                    memory_pass=True,energy_pass=True,shutdown_pass=True) for i in range(4)]
                report={'status':'ALL_FOUR_TERMINALS_AUDITED','jobs':jobs}
                if failure=='status':report['status']='FAILED'
                else:jobs[0][failure]=False
                _,launched,state=self.run_fixture(request,audit=report)
                self.assertEqual(launched,['fine_audit']);self.assertEqual(state['status'],'STOPPED_NO_RETRY')

    def test_ineligible_or_negative_utility_is_not_a_scheduling_gate(self):
        jobs=[dict(job_id=str(i),audit_status='VERIFIED_TERMINAL_EVIDENCE',raw_status='completed',
            memory_pass=True,energy_pass=True,shutdown_pass=True,utility={'eligible':False,'U':None}) for i in range(4)]
        stage.audit_gate({'status':'ALL_FOUR_TERMINALS_AUDITED','jobs':jobs},[str(i) for i in range(4)],'ALL_FOUR_TERMINALS_AUDITED')

    def test_changed_terminal_receipt_and_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            request,packet,queue=self.fixture(tmp,stage.TERMINAL,4)
            queue['source']={'changed':True}
            with self.assertRaisesRegex(ValueError,'runner binding'):
                stage.queue_gate(queue,packet,request['fine_packet'],4,request['runner'])
            queue['source']=request['runner'];Path(queue['jobs'][0]['raw']['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'receipt changed'):
                stage.queue_gate(queue,packet,request['fine_packet'],4,request['runner'])

    def test_signal_cleanup_terminates_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            request,_,_=self.fixture(tmp);supervisor=stage.Supervisor(request)
            child=Mock(pid=12345);child.poll.return_value=None;supervisor.process=child
            with patch.object(stage.os,'killpg') as kill:
                supervisor.stop_child()
            kill.assert_called_once_with(12345,signal.SIGTERM);child.wait.assert_called_once_with(timeout=40)


if __name__=='__main__':unittest.main()
