"""Touching phases are not overlap; tiny genuine intervals remain evidence."""
import unittest
from unittest.mock import patch
from experiments.analyze_activation_phase_attainment import interval_attainment, describe
from experiments.activation_phase_crossed_design import align
from tests.test_phase_crossed_design import template


class AttainmentTests(unittest.TestCase):
    def test_touching_and_missed_attempts_are_retained_as_nonattainment(self):
        for intervals,gap in [([(0,10),(10,20)],0),([(0,10),(12,20)],2)]:
            result=interval_attainment(intervals)
            self.assertFalse(result['attained']);self.assertEqual(result['overlap_ns'],0)
            self.assertEqual(result['nonoverlap_gap_ns'],gap)

    def test_one_nanosecond_common_intersection_counts_for_all_owners(self):
        result=interval_attainment([(0,10),(9,20),(5,15),(2,12)])
        self.assertTrue(result['attained']);self.assertEqual(result['overlap_ns'],1)
        with self.assertRaises(ValueError):interval_attainment([(0,10),(5,5)])

    def test_planned_alignment_does_not_hide_a_missed_attempt_or_dropped_request(self):
        t=template();text=next(p for p in t['profiles'] if p['family']=='text')
        image=next(p for p in t['profiles'] if p['family']=='image')
        condition=[dict(alias='text#0',family='text',task='text',role='tail'),
                   dict(alias='image#0',family='image',task='image',role='whole')]
        job=align(condition,[text,image],t,{'text#0':0,'image#0':0});job['id']='j'
        rows=[]
        for request,role,left,right in [(job['requests'][0],'tail',10_000_000,30_000_000),
                                         (job['requests'][1],'whole',40_000_000,50_000_000)]:
            rows.append(dict(profile_job_id='j',model=request['alias'],task=request['task'],pool_id=request['pool_id'],
                request_id=request['alias'],scheduled_time_ns=round(request['offset_ms']*1e6),
                fixture_bands=[dict(id=role,native_band_role=role,start_ns=left,end_ns=right)]))
        bundle=dict(source='fixture',plan=dict(jobs=[job]),raw=dict(run_id='one',batches=[dict(id='j',start_ns=0)],request_records=rows))
        with patch('experiments.analyze_activation_phase_attainment.validate_capture',return_value='fixture'),patch(
                'experiments.analyze_activation_phase_attainment.completion_band_layout',side_effect=lambda row,family:row['fixture_bands']):
            report=describe(bundle,t)
            self.assertEqual(report['attempted_conditions'],1);self.assertEqual(report['attained_conditions'],0)
            self.assertEqual(report['attempts'][0]['nonoverlap_gap_ns'],10_000_000)
            bundle['raw']['request_records'].pop()
            with self.assertRaisesRegex(ValueError,'observed denominator'):describe(bundle,t)


if __name__=='__main__':unittest.main()
