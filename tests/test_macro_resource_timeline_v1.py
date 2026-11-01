import unittest
from dataclasses import replace
from fractions import Fraction as F

from tests.test_macro_response_queue_reference_v1 import request,plan,run,ns
from tools.predictor_validation_20261027.macro_resource_timeline_v1 import timeline


class ResourceTimelineTests(unittest.TestCase):
    def test_single_board_clock_idle_and_drain_retained_without_double_sum(self):
        rows=[request(),request('b','s1')]
        reference=run(rows,{'s0':plan(producer='s0'),'s1':plan(producer='s1')})
        result=timeline(reference,rows,resident_owners=('s0','s1','g2','ocr'))
        self.assertEqual(sum(ns(s['duration_ns']) for s in result['slices']),100)
        self.assertEqual(sum(ns(s['measurement_overlap_ns']) for s in result['slices']),80)
        self.assertEqual(len(result['slices'][0]['actual_START_to_READY_handlers']),2)
        self.assertEqual(result['slices'][-1]['actual_START_to_READY_handlers'],[])
        self.assertEqual(result['slices'][-1]['resident_owners'],['s0','s1','g2','ocr'])
        self.assertIsNone(result['board_energy_joules'])
        self.assertIsNone(result['peak_memory_mib'])

    def test_actual_handler_ends_before_late_READY_notification(self):
        rows=[request()]
        reference=run(rows,{'s0':plan(10,ready_delay=30)})
        result=timeline(reference,rows,resident_owners=('s0','g2','ocr'))
        busy=sum(ns(s['duration_ns']) for s in result['slices'] if s['actual_START_to_READY_handlers'])
        self.assertEqual(busy,10)
        self.assertFalse(result['old_resource_execution_identity_transferred'])

    def test_handoff_wait_is_not_actual_handler_work_and_cutoff_stays_exact(self):
        rows=[request()]
        p=replace(plan(),dispatch_to_worker_start_ns=F(5))
        reference=run(rows,{'s0':p},cutoff=8)
        result=timeline(reference,rows,resident_owners=('s0','g2'))
        self.assertEqual(sum(ns(s['duration_ns']) for s in result['slices']),8)
        self.assertEqual(ns(result['slices'][0]['duration_ns']),5)
        self.assertFalse(result['slices'][0]['actual_START_to_READY_handlers'])
        self.assertEqual(ns(result['slices'][-1]['duration_ns']),3)

    def test_missing_resident_or_cohort_identity_cannot_silently_reduce_memory(self):
        rows=[request()];reference=run(rows,{'s0':plan()})
        with self.assertRaises(ValueError):timeline(reference,rows,resident_owners=('g2',))
        with self.assertRaises(ValueError):timeline(reference,[],resident_owners=('s0',))
        with self.assertRaises(ValueError):timeline(reference,[request('foreign')],resident_owners=('s0',))


if __name__=='__main__':unittest.main()
