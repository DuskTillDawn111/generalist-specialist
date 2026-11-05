"""Heap reclamation must remain inside priming lifecycle, before GC snapshots."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path=Path(__file__).resolve().parents[1]/'tools/predictor_validation_20261027/postprime_trim_scoped_worker.py'
spec=importlib.util.spec_from_file_location('postprime_trim_test',path);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class PostprimeTrimTests(unittest.TestCase):
    def test_trim_once_after_all_supported_priming_and_before_snapshot_boundary(self):
        events=[];calls=[];complete=SimpleNamespace(emit=lambda kind,payload:events.append((kind,payload)))
        def procedure(mode):
            calls.append(mode);return dict(procedure_end_ns=900)
        worker.install(complete,'on',procedure,clock=lambda:1000)
        complete.emit('READY',dict(tasks=['text','image','ocr','asr']))
        for i in range(11):complete.emit('WORKER_READY',dict(request_id='priming:'+str(i),perf_ns=100))
        self.assertEqual(calls,[])
        complete.emit('WORKER_READY',dict(request_id='priming:11',perf_ns=100))
        self.assertEqual(calls,['on']);self.assertEqual(events[-1][1]['perf_ns'],1000)
        self.assertEqual(events[-1][1]['postprime_trim_receipt']['procedure_end_ns'],900)
        complete.emit('PHASE',dict(event='start',segment_id='host_setup',request_id='training:0'))
        complete.emit('WORKER_READY',dict(request_id='training:0',perf_ns=1100));self.assertEqual(calls,['on'])

    def test_formal_request_before_complete_priming_rejected(self):
        complete=SimpleNamespace(emit=lambda *args:None)
        worker.install(complete,'off',lambda mode:dict(procedure_end_ns=1),clock=lambda:2)
        complete.emit('READY',dict(tasks=['asr']))
        complete.emit('WORKER_READY',dict(request_id='priming:0',perf_ns=1))
        with self.assertRaises(ValueError):complete.emit('PHASE',dict(event='start',segment_id='host_setup',request_id='training:0'))


if __name__=='__main__':unittest.main()
