from fractions import Fraction
import copy
import unittest

from tools.predictor_validation_20261027 import twoFIFO_transport_partial_identification_v2 as m


class PartialIdentificationTests(unittest.TestCase):
    def test_busy_source_and_busy_target_identify_continuation_and_start_without_line_choice(self):
        source=m.identify_source(100,110,150,140)
        self.assertEqual(source['identified_worker_continuation_extra_ns'],10)
        self.assertEqual(source['stdin_line_extra_ns_bounds'],[0,10])
        target=m.project_target(source,1000,1020)
        self.assertEqual(target['identified_START_ns'],1030)
        self.assertFalse(target['latent_point_selected']);self.assertFalse(target['latent_probability_law_identified'])

    def test_idle_to_idle_uses_identified_composite_not_invented_zero_line(self):
        source=m.identify_source(100,120,130,90)
        self.assertFalse(source['worker_continuation_extra_identified'])
        target=m.project_target(source,1000,990)
        self.assertEqual(target['identified_START_ns'],1030)
        self.assertEqual(m.project_target(source,1000,1040)['START_ns_bounds'],[1050,1070])

    def test_exact_envelope_over_all_breakpoints_and_integer_grid(self):
        # Independent enumeration also checks source-submit/READY/write boundary
        # equality and write-record timestamps later than observed START.
        for W in range(5):
            for H in range(5):
                for qsource in range(-2,H+1):
                    source=m.identify_source(10,10+W,10+H,10+qsource)
                    for qtarget in range(-2,7):
                        target=m.project_target(source,100,100+qtarget)
                        us=[Fraction(i,4) for i in range(4*min(W,H)+1)]
                        starts=[100+max(qtarget,u)+H-max(qsource,u) for u in us]
                        self.assertEqual(target['START_ns_bounds'],[min(starts),max(starts)])
                        same=m.project_target(source,10,10+qsource)
                        self.assertEqual(same['identified_START_ns'],10+H)

    def test_fractional_corners_and_invalid_chronology(self):
        source=m.identify_source(10,14,16,Fraction(23,2))
        target=m.project_target(source,20,Fraction(91,4))
        self.assertEqual(target['START_ns_bounds'],[26,Fraction(109,4)])
        for values in ((10,9,20,5),(10,20,9,5),(10,20,20,21)):
            with self.assertRaises(ValueError):m.identify_source(*values)
        with self.assertRaises(ValueError):m.identify_source(True,20,30,0)

    def test_offered_unknowns_and_priming_history_not_dropped(self):
        priming=dict(request_id='prime',model='text#0',actual_submit_time_ns=1,submit_write_completed_time_ns=2,
            observed_worker_start_time_ns=3,previous_handler_next_request_ready=dict(perf_ns=9))
        first=dict(request_id='first',model='text#0',case_id='case',actual_submit_time_ns=5,
            submit_write_completed_time_ns=6,observed_worker_start_time_ns=10,previous_handler_next_request_ready=dict(perf_ns=14))
        missing=dict(request_id='missing',model='text#0',status='not_started')
        plan=dict(execution_id='exec',priming=[{}],cases=[dict(arrivals=[dict(request_id='first',arrival_ns=0,arrival_index=0),dict(request_id='missing',arrival_ns=1,arrival_index=1)])])
        raw=dict(execution_id='exec',priming_request_records=[priming],request_records=[first,missing])
        report=m.observed_dominance(raw,plan)
        self.assertEqual(report['measured_offered'],2);self.assertEqual(report['priming_observed'],1)
        self.assertEqual(report['counts']['measured']['PREVIOUS_READY_DOMINATES_WRITE_RECORD'],1)
        self.assertEqual(report['counts']['measured']['unknown'],1)
        self.assertEqual(report['counts']['priming']['unknown'],1)

    def test_closed_source_rejects_negative_float_bool_extra_and_inconsistent_fields(self):
        base=m.identify_source(100,110,130,120)
        for key,value in (('submit_to_write_record_ns',-1),('observed_submit_to_START_ns',30.0),
                ('previous_READY_minus_submit_ns',True),('stdin_line_extra_ns_bounds',[0.0,10]),
                ('worker_continuation_extra_identified',False),('unobserved_extra_key',0)):
            bad=copy.deepcopy(base);bad[key]=value
            with self.assertRaises(ValueError):m.project_target(bad,1000,1020)


if __name__=='__main__':unittest.main()
