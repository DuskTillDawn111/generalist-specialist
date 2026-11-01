import unittest

from tools.predictor_validation_20261027.cuda_dependency_dag_v2 import build_dag


def node(name,stream,issue,duration=0,**extra):
    return dict(id=name,stream_key=[42,0,1,stream],issue_start_ns=issue,duration_ns=duration,**extra)


class CudaDependencyDagTests(unittest.TestCase):
    def test_event_wait_edges_use_identity_and_preserve_parallel_branch(self):
        a=node('a',7,1,10);b=node('b',13,2,20);c=node('c',13,6,5)
        event=node('e',7,3,event_identity=[42,1,99,1],raw_event_timestamp=0)
        wait=node('w',13,5,event_identity=[42,1,99,1])
        result=build_dag([a,b,c],[event],[wait])
        self.assertFalse(result['cyclic']);self.assertEqual(result['partial_dag_gpu_activity_critical_path_ns'],25)
        self.assertIn(dict(before='e',after='w',kind='event_record_to_wait'),result['edges'])
        self.assertNotIn(dict(before='a',after='b',kind='stream_fifo'),result['edges'])

    def test_reused_event_handle_requires_sync_instance(self):
        event=node('e',7,1,event_identity=[42,1,99,1])
        wait=node('w',13,2,event_identity=[42,1,99,2])
        result=build_dag([],[event],[wait])
        self.assertEqual(result['unresolved'][0]['reason'],'NO_EXACT_EVENT_RECORD_IDENTITY')

    def test_ambiguous_host_enqueue_overlap_not_forced_order(self):
        a=node('a',7,1,10,issue_thread=1,issue_end_ns=5)
        b=node('b',7,2,20,issue_thread=2,issue_end_ns=6)
        result=build_dag([a,b],[],[])
        self.assertFalse(result['edges']);self.assertTrue(result['unresolved'])

    def test_cycle_stays_explicit_unknown(self):
        event=node('e',7,3,event_identity=[42,1,99,1])
        wait=node('w',7,2,event_identity=[42,1,99,1])
        result=build_dag([],[event],[wait])
        self.assertTrue(result['cyclic']);self.assertIsNone(result['partial_dag_gpu_activity_critical_path_ns'])


if __name__=='__main__':unittest.main()
