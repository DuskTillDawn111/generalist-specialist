"""CPU-only independent adversarial probe and real torch hook contract checks."""
import copy
import json
from types import SimpleNamespace
import unittest

from tests.test_device_progress_observer_v1 import Cuda
from tests.test_semantic_observer import FakeModule,FakeNvtx,TensorShape
from tools.predictor_validation_20261027 import device_progress_observer_v1 as v


class IndependentObserverReview(unittest.TestCase):
    def setUp(self):
        self.owner=FakeModule();self.cuda=Cuda();self.nvtx=FakeNvtx();self.ticks=1000
        self.executor=SimpleNamespace(family='chatqa',model=self.owner)

    def clock(self):self.ticks+=10;return self.ticks

    def observe(self,**kwargs):
        return v.observe_language_forwards(self.executor,'independent',torch=SimpleNamespace(cuda=self.cuda),nvtx=self.nvtx,clock=self.clock,**kwargs)

    def test_real_torch_cpu_module_always_call_and_hook_cleanup(self):
        import torch
        before_initialized=torch.cuda.is_initialized()
        class ActualModule(torch.nn.Module):
            def forward(self,input_ids=None,fail=False):
                if fail:raise RuntimeError('actual module failure')
                return input_ids
        owner=ActualModule();self.owner=owner;self.executor.model=owner
        before_forward=owner.forward.__func__;existing=[]
        handle=owner.register_forward_hook(lambda m,a,o:existing.append(o))
        tensor=torch.ones((1,3),device='cpu')
        with self.observe() as observer:
            self.assertIs(owner(tensor),tensor)
            with self.assertRaisesRegex(RuntimeError,'actual module failure'):owner(tensor,fail=True)
        self.assertIs(owner.forward.__func__,before_forward)
        self.assertEqual(len(observer.records),2);self.assertEqual(self.nvtx.active,[])
        self.assertEqual(len(owner._forward_pre_hooks),0);self.assertEqual(len(owner._forward_hooks),1)
        self.assertNotIn(owner,v._ACTIVE);handle.remove()
        self.assertEqual(torch.cuda.is_initialized(),before_initialized)

    def test_query_error_and_sink_error_do_not_change_forward_exception_or_result(self):
        def sink(_):raise RuntimeError('broken sink')
        with self.observe(sink=sink) as observer:
            output=self.owner(TensorShape(1,3));self.assertIs(output,self.owner.outputs[-1])
            end=observer.device_rows[0]['_end'];end.query=lambda:(_ for _ in ()).throw(RuntimeError('query unavailable'))
            observer.poll()
            self.assertEqual(observer.device_rows[0]['observation_error'],'RuntimeError')
            self.assertFalse(any(q['end_event_complete'] for q in observer.queries))
            with self.assertRaisesRegex(RuntimeError,'native forward failure'):self.owner(TensorShape(1,3),fail=True)
        self.assertEqual(self.owner.before+self.owner.after,[]);self.assertNotIn(self.owner,v._ACTIVE)

    def test_nested_document_mutation_cannot_mutate_probe_or_old_snapshot(self):
        with self.observe() as observer:
            self.owner(TensorShape(1,3));first=observer.document();saved=copy.deepcopy(first)
            first['device_progress']['records'][0]['begin_record_host_bracket_ns'][0]=-1
            first['device_progress']['nvtx_host_brackets'][0]['push_before_ns']=-1
            first['device_progress']['queries'][0]['end_event_complete']=True
            self.assertEqual(observer.document(),saved)
            self.cuda.complete=True;observer.poll()
            self.assertFalse(saved['device_progress']['queries'][0]['end_event_complete'])

    def test_internal_peer_stream_work_is_never_certified_as_all_work(self):
        # The observed caller stream may complete while a separately issued
        # peer stream remains busy; no claim about peer streams is permitted.
        peer=SimpleNamespace(cuda_stream=99,device=SimpleNamespace(index=0),complete=False)
        with self.observe() as observer:
            observer.before(self.owner,(TensorShape(1,3),),{})
            saved=self.cuda.stream;self.cuda.stream=peer;self.cuda.stream=saved
            self.cuda.complete=True;observer.after(self.owner,(),{},object())
            q=observer.queries[-1]
            self.assertTrue(q['end_event_complete']);self.assertFalse(q['all_request_GPU_work_complete'])
            self.assertEqual(q['evidence_scope'],'recorded_current_stream_only')
            self.assertFalse(observer.device_rows[0]['other_stream_dependencies_certified'])
            self.assertFalse(peer.complete)

    def test_partial_hook_registration_failure_releases_active_guard(self):
        def unsupported(*args,**kwargs):raise TypeError('always_call unsupported')
        self.owner.register_forward_hook=unsupported
        with self.assertRaisesRegex(TypeError,'always_call unsupported'):
            with self.observe():pass
        self.assertEqual(self.owner.before,[]);self.assertNotIn(self.owner,v._ACTIVE)

    def test_same_default_stream_pointer_on_different_device_is_stream_change(self):
        self.cuda.stream=SimpleNamespace(cuda_stream=0,device=SimpleNamespace(index=0));self.cuda.complete=True
        with self.observe() as observer:
            observer.before(self.owner,(TensorShape(1,3),),{})
            self.cuda.stream=SimpleNamespace(cuda_stream=0,device=SimpleNamespace(index=1))
            observer.after(self.owner,(),{},object())
            self.assertIn('observation_error',observer.device_rows[0])
            self.assertFalse(any(q['end_event_complete'] for q in observer.queries))

    def test_sink_mutation_cannot_rewrite_retained_completion_evidence(self):
        def sink(value):
            if value['event']=='observation':
                value['end_event_complete']=True;value['all_request_GPU_work_complete']=True
                value['host_query_begin_ns']=-1
        with self.observe(sink=sink) as observer:
            self.owner(TensorShape(1,3));q=observer.document()['device_progress']['queries'][-1]
            self.assertFalse(q['end_event_complete']);self.assertFalse(q['all_request_GPU_work_complete'])
            self.assertGreater(q['host_query_begin_ns'],0)

    def test_end_record_failure_and_elapsed_failure_remain_unknown(self):
        with self.observe() as observer:
            observer.before(self.owner,(TensorShape(1,3),),{})
            def fail_record(stream):raise RuntimeError('end record unavailable')
            observer.device_rows[0]['_end_pending'].record=fail_record
            observer.after(self.owner,(),{},object())
            self.assertEqual(observer.device_rows[0]['observation_error'],'RuntimeError')
            self.assertIsNone(observer.device_rows[0]['_end'])
            self.assertEqual(observer.queries,[])
            self.assertEqual(self.nvtx.active,[])
        with self.observe() as observer:
            self.owner(TensorShape(1,3));self.cuda.complete=True
            def fail_elapsed(other):raise RuntimeError('elapsed unavailable')
            observer.device_rows[0]['_begin'].elapsed_time=fail_elapsed
            observer.poll()
            self.assertFalse(any(q['end_event_complete'] for q in observer.queries))
            self.assertEqual(len(observer.pending),1)

    def test_extra_observations_pass_existing_transport_but_not_phase_anchor_pairing(self):
        from gsdeploy.runtime.transport import Worker
        from tools.predictor_validation_20261027.trace_export import anchors_from_worker_lines
        emitted=[]
        with self.observe(sink=emitted.append):self.owner(TensorShape(1,3))
        lines=['START|'+json.dumps(dict(request_id='independent',pid=123))]
        lines.extend('PHASE|'+json.dumps(row) for row in emitted)
        anchors=anchors_from_worker_lines(lines)
        self.assertEqual(len(anchors),1);self.assertEqual(anchors[0]['kind'],'gpu')
        queued=[];worker=Worker('owner',{},SimpleNamespace(put=queued.append))
        worker.process=SimpleNamespace(stdout=iter(lines));worker._reader()
        observations=[payload for owner,kind,payload in queued if payload.get('event')=='observation']
        self.assertEqual(len(observations),sum(row.get('event')=='observation' for row in emitted))
        self.assertGreater(len(observations),0)
        self.assertFalse(observations[0]['all_request_GPU_work_complete'])


if __name__=='__main__':unittest.main()
