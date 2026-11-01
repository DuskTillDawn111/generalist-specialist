from fractions import Fraction
import unittest

from tools.predictor_validation_20261027 import worker_stdin_fifo_contract_v1 as module


def transport(write=1,line=0,worker=0,enqueue=0,pickup=0):
    return module.TransportDraw(enqueue,pickup,write,line,worker)


def handler():
    return dict(START_to_RESULT_mark_ns=8,START_to_READY_occurrence_ns=10,
        RESULT_mark_to_reader_ns=10,RESULT_reader_to_processing_ns=100,
        READY_occurrence_to_reader_ns=10,READY_reader_to_processing_ns=100)


class WorkerFIFOTests(unittest.TestCase):
    def requests(self):
        return [dict(request_id='r'+str(i),owner='owner',task='text',pool_id='input',arrival_ns=i) for i in range(2)]

    def world(self,draws=None,handlers=None):
        keys=[('owner','text','input',i) for i in range(2)]
        return module.compose_explicit_world(self.requests(),draws or {k:transport() for k in keys},
            handlers or {k:handler() for k in keys},initial_worker_read_release_ns={'owner':0},
            initial_sender_available_ns={'owner':0})

    def test_queued_next_starts_at_actual_READY_without_parent_processing_gate(self):
        result=self.world();first,second=result['rows']
        self.assertEqual(second['parent_submit_ns'],1)
        self.assertEqual(second['worker_START_ns'],first['READY_occurrence_ns'])
        self.assertLess(second['worker_START_ns'],first['READY_processed_ns'])
        self.assertLess(second['worker_START_ns'],first['RESULT_reader_ns'])
        self.assertFalse(result['READY_processing_gates_queued_worker_START'])

    def test_worker_can_start_before_parent_write_record(self):
        send=module.sender_fifo(0,0,transport(write=5,line=2,worker=1))
        start=module.worker_fifo_START(0,send)
        self.assertEqual(start,3);self.assertLess(start,send['parent_write_record_ns'])

    def test_sender_FIFO_blocking_is_separate_from_worker_FIFO(self):
        keys=[('owner','text','input',i) for i in range(2)]
        result=self.world(draws={keys[0]:transport(write=20,line=0),keys[1]:transport()})
        first,second=result['rows']
        self.assertEqual(second['parent_submit_ns'],first['parent_write_record_ns'])
        self.assertEqual(first['worker_START_ns'],0)
        self.assertEqual(second['worker_START_ns'],20)

    def test_identical_idle_lag_does_not_identify_busy_transfer(self):
        late_line=module.sender_fifo(0,0,transport(write=5,line=5,worker=0))
        late_worker=module.sender_fifo(0,0,transport(write=5,line=0,worker=5))
        self.assertEqual(module.worker_fifo_START(0,late_line),5)
        self.assertEqual(module.worker_fifo_START(0,late_worker),5)
        self.assertEqual(module.worker_fifo_START(10,late_line),10)
        self.assertEqual(module.worker_fifo_START(10,late_worker),15)

    def test_feasible_component_bounds_do_not_impose_write_gate(self):
        bounds=module.idle_donor_decomposition_bounds(10,15,13,5)
        self.assertEqual(bounds['stdin_line_available_ns_bounds'],[10,13])
        self.assertEqual(bounds['worker_release_to_START_extra_ns_bounds'],[0,3])
        self.assertFalse(bounds['uniquely_identified'])
        self.assertFalse(bounds['exact_worker_stdin_waiting_idle_observed'])

    def test_missing_negative_and_inconsistent_transport_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'missing components are UNKNOWN'):module.sender_fifo(0,0,None)
        with self.assertRaises(ValueError):module.sender_fifo(0,0,transport(line=2,write=1))
        with self.assertRaises(ValueError):module.sender_fifo(0,0,transport(worker=-1))
        with self.assertRaises(ValueError):module.sender_fifo(0.,0,transport())

    def test_stdout_FIFO_conflicts_are_UNKNOWN_not_clamped(self):
        keys=[('owner','text','input',i) for i in range(2)]
        one=handler();two=handler();one['READY_reader_to_processing_ns']=1000
        with self.assertRaisesRegex(ValueError,'CROSS_REQUEST_STDOUT_FIFO'):
            self.world(handlers={keys[0]:one,keys[1]:two})

    def test_parent_enqueue_order_and_exact_rational_time(self):
        keys=[('owner','text','input',i) for i in range(2)]
        with self.assertRaisesRegex(ValueError,'REORDER_PUBLIC_PARENT_ENQUEUE_FIFO'):
            self.world(draws={keys[0]:transport(enqueue=3),keys[1]:transport()})
        send=module.sender_fifo(Fraction(1,3),0,transport(write=Fraction(1,2),line=Fraction(1,4)))
        self.assertEqual(module.worker_fifo_START(0,send),Fraction(7,12))


if __name__=='__main__':unittest.main()
