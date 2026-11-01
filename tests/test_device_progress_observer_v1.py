"""Nonblocking stream completion, causal brackets and unchanged forward semantics."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from tests.test_semantic_observer import FakeModule, FakeNvtx, TensorShape
from tools.predictor_validation_20261027 import device_progress_observer_v1 as v


class Event:
    def __init__(self, cuda):
        self.cuda, self.recorded = cuda, None

    def record(self, stream):
        self.recorded = stream

    def query(self):
        return self.cuda.complete

    def elapsed_time(self, other):
        if not self.query() or not other.query():
            raise AssertionError('Unfinished event used for elapsed measurement')
        return 12.5


class Cuda:
    def __init__(self):
        self.complete = False
        self.stream = SimpleNamespace(cuda_stream=17, device=SimpleNamespace(index=0))
        self.events = []

    def current_stream(self):
        return self.stream

    def Event(self, *, enable_timing):
        assert enable_timing
        event = Event(self); self.events.append(event); return event

    def synchronize(self):
        raise AssertionError('Observer must never synchronize CUDA')


class DeviceObserverTests(unittest.TestCase):
    def setUp(self):
        self.owner, self.cuda, self.nvtx = FakeModule(), Cuda(), FakeNvtx()
        self.executor = SimpleNamespace(family='chatqa',model=self.owner)
        self.clock_ns = 1000

    def clock(self):
        self.clock_ns += 10
        return self.clock_ns

    def observe(self, **kwargs):
        return v.observe_language_forwards(self.executor,'req',torch=SimpleNamespace(cuda=self.cuda),
            nvtx=self.nvtx,clock=self.clock,**kwargs)

    def test_host_forward_end_does_not_certify_device_completion_or_remaining_fraction(self):
        emitted = []
        with self.observe(sink=emitted.append) as observer:
            original = self.owner(TensorShape(1,20))
            self.assertIs(original,self.owner.outputs[0])
            doc = observer.document()['device_progress']
            self.assertEqual(len(doc['pending_event_ids']),1)
            q = doc['queries'][-1]
            self.assertFalse(q['end_event_complete'])
            self.assertIsNone(q['in_flight_remaining_fraction'])
            self.assertIsNone(q['recorded_stream_elapsed_ms'])
            self.assertFalse(q['all_request_GPU_work_complete'])
            self.assertEqual(q['stream_id'],17)
            self.assertLess(q['host_query_begin_ns'],q['host_query_end_ns'])
            self.assertTrue(any(e['event']=='observation' for e in emitted))
        self.assertFalse(observer.document()['device_progress']['cuda_synchronization_added'])

    def test_delayed_nonblocking_query_retains_actual_bracket_and_stream_scope(self):
        with self.observe() as observer:
            self.owner(TensorShape(1,20))
            first = observer.document()
            self.cuda.complete = True
            observer.poll()
            final = observer.document()['device_progress']
            q = final['queries'][-1]
            self.assertTrue(q['end_event_complete'])
            self.assertEqual(q['recorded_stream_elapsed_ms'],12.5)
            self.assertIsNone(q['completion_time_ns'])
            self.assertIsNone(q['physical_SM_service_ns'])
            self.assertEqual(q['evidence_scope'],'recorded_current_stream_only')
            self.assertFalse(q['all_request_GPU_work_complete'])
            self.assertFalse(final['records'][0]['other_stream_dependencies_certified'])
            self.assertEqual(final['pending_event_ids'],[])
            self.assertEqual(final['records'][0]['completion_query_bracket_ns'],
                [q['host_query_begin_ns'],q['host_query_end_ns']])
            self.assertFalse(first['device_progress']['queries'][-1]['end_event_complete'])
            count = len(final['queries']); observer.poll()
            self.assertEqual(len(observer.queries),count)

    def test_dual_sided_nvtx_and_record_brackets_enclose_original_phase(self):
        with self.observe() as observer:
            self.owner(TensorShape(2,30))
            phase = observer.records[0]
            bracket = observer.bracketed_nvtx.records[0]
            item = observer.document()['device_progress']['records'][0]
            self.assertLess(bracket['push_before_ns'],bracket['push_after_ns'])
            self.assertLess(bracket['push_after_ns'],phase['start_ns'])
            self.assertLess(phase['start_ns'],item['begin_record_host_bracket_ns'][0])
            self.assertLess(item['end_record_host_bracket_ns'][1],phase['end_ns'])
            self.assertLess(phase['end_ns'],bracket['pop_before_ns'])
            self.assertLess(bracket['pop_before_ns'],bracket['pop_after_ns'])
            self.assertEqual(phase['input_sequence_length'],30)
            self.assertEqual(phase['input_batch_size'],2)

    def test_forward_failure_and_probe_failure_preserve_output_and_hook_cleanup(self):
        with self.assertRaisesRegex(RuntimeError,'native forward failure'):
            with self.observe() as observer:
                self.owner(TensorShape(1,4),fail=True)
        self.assertEqual(self.owner.before+self.owner.after,[])
        self.assertEqual(self.nvtx.active,[])
        self.assertFalse(observer.document()['device_progress']['records'][0]['forward_success_certified_by_probe'])
        def broken(**kwargs):raise RuntimeError('probe unavailable')
        self.cuda.Event = broken
        with self.observe() as observer:
            output = self.owner(TensorShape(1,4))
        self.assertIs(output,self.owner.outputs[-1])
        self.assertEqual(observer.device_rows,[])
        self.assertEqual(observer.errors[-1]['where'],'device_record_begin')

    def test_stream_change_and_nested_observer_cannot_certify_another_stream(self):
        with self.observe() as observer:
            with self.assertRaisesRegex(ValueError,'already has an active'):
                with self.observe():pass
            # Exercise the actual hook boundary, with an explicit stream swap
            # before always_call exit, without replacing the forward method.
            observer.before(self.owner,(TensorShape(1,3),),{})
            self.cuda.stream = SimpleNamespace(cuda_stream=99,device=SimpleNamespace(index=0))
            observer.after(self.owner,(),{},object())
            self.assertEqual(observer.device_rows[0]['observation_error'],'ValueError')
            self.assertFalse(observer.queries)
        self.assertEqual(self.owner.before+self.owner.after,[])

    def test_isolated_worker_only_changes_observer_and_queries_after_existing_final_sync(self):
        root = Path(v.__file__).parent
        expected = (root/'semantic_worker.py').read_text()
        expected = expected.replace('from tools.predictor_validation_20261027.semantic_observer import observe_language_forwards',
            'from tools.predictor_validation_20261027.device_progress_observer_v1 import observe_language_forwards')
        expected = expected.replace('observe_language_forwards(adapter.executor, rid, nvtx=',
            'observe_language_forwards(adapter.executor, rid, torch=torch, nvtx=')
        expected = expected.replace("observer.call('final_sync', 'sync', torch.cuda.synchronize)",
            "observer.call('final_sync', 'sync', torch.cuda.synchronize)\n                semantic.poll()\n                result['semantic_forward_observations'] = semantic.document()")
        actual = (root/'semantic_progress_worker_v1.py').read_text()
        trees = [ast.parse(s) for s in (expected,actual)]
        for tree in trees:tree.body = tree.body[1:]
        self.assertEqual(ast.dump(trees[0]),ast.dump(trees[1]))


if __name__ == '__main__':unittest.main()
