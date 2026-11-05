"""Observer accounting fixtures; these do not establish physical overhead."""
from types import SimpleNamespace
import unittest

from tools.predictor_validation_20261027 import semantic_observed_worker as worker


class FakeEvent:
    def __init__(self,cuda):self.cuda=cuda;self.at=None
    def record(self,stream):self.at=self.cuda.clock;self.cuda.clock+=1
    def query(self):return self.at is not None
    def elapsed_time(self,other):return float(other.at-self.at)


class FakeCUDA:
    def __init__(self):
        self.clock=0;self.stack=[]
        self.nvtx=SimpleNamespace(range_push=self.stack.append,range_pop=self.stack.pop)
    def Event(self,**kwargs):return FakeEvent(self)
    def current_stream(self):return SimpleNamespace(cuda_stream=7,device=SimpleNamespace(index=0))
    def synchronize(self):raise AssertionError('Per-forward synchronization is forbidden')


class SemanticObserverTests(unittest.TestCase):
    def tearDown(self):worker.CURRENT=None

    def test_async_events_preserve_output_and_finish_without_extra_sync(self):
        cuda=FakeCUDA();recorder=worker.DetailRecorder(SimpleNamespace(cuda=cuda),'request')
        result=recorder.call(lambda x:x*2,'ocr_detection',(3,),{})
        self.assertEqual(result,6);recorder.finish()
        self.assertEqual(recorder.records[0]['device_elapsed_ms'],1.)
        self.assertEqual(recorder.records[0]['stream_id'],7)
        self.assertEqual(cuda.stack,[])

    def test_language_classification_uses_only_current_inputs_and_prior_calls(self):
        recorder=worker.DetailRecorder(SimpleNamespace(cuda=FakeCUDA()),'request')
        for n in (40,1,1):
            recorder.call(lambda **kw:None,'language',(),{'input_ids':SimpleNamespace(shape=(1,n))})
        self.assertEqual([r['role'] for r in recorder.records],
                         ['language_multi_token','language_decode','language_decode'])
        self.assertEqual([r['ordinal'] for r in recorder.records],[0,1,2])

    def test_failure_closes_nvtx_and_keeps_unfinished_event_explicit(self):
        cuda=FakeCUDA();recorder=worker.DetailRecorder(SimpleNamespace(cuda=cuda),'request')
        def fail():raise ValueError('inference failed')
        with self.assertRaises(ValueError):recorder.call(fail,'vision_encoder',(),{})
        self.assertEqual(cuda.stack,[]);self.assertIn('host_end_ns',recorder.records[0])

    def test_attached_ledger_receives_device_results_at_existing_final_sync(self):
        recorder=worker.DetailRecorder(SimpleNamespace(cuda=FakeCUDA()),'request');worker.CURRENT=recorder
        owner=SimpleNamespace(forward=lambda x:x+1);worker.observe_method(owner,'forward','image_features')
        self.assertEqual(owner.forward(4),5)
        result=worker.attach_result({'prediction':'unchanged'})
        self.assertNotIn('device_elapsed_ms',result['semantic_phase_events'][0])
        recorder.finish()
        self.assertTrue(result['semantic_phase_events'][0]['device_event_complete'])
        self.assertEqual(result['prediction'],'unchanged')


if __name__=='__main__':unittest.main()
