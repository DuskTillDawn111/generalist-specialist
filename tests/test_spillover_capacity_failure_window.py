"""Failure cutoff: preserve late native output without admitting it to evidence."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
f=load('failure_fixture',ROOT/'tests/test_spillover_capacity_sustained_analysis.py')
w=load('failure_window',ROOT/'tools/spillover_capacity_20261022/failure_window.py')


class DeadlineSlice(unittest.TestCase):
    def fixture(self):
        report=f.analyze(f.fixture());window=report['measurement_timing']['measurement_window'];window['start_unix_ns']=window['start_time_ns']
        window['end_time_ns']=window['start_time_ns']+2_500_000_000;report['raw_status']='failed'
        return report

    def test_late_completion_is_unknown_but_preserved(self):
        original=self.fixture();result=w.slice_job(original);alias=result['aliases']['generalist_2#0']
        self.assertEqual(alias['primary']['service_available_requests'],83)
        self.assertEqual(alias['primary_deadline_classes']['COMPLETED_AFTER_DEADLINE_NOT_QUALIFIED'],13)
        row=next(r for r in result['request_diagnostics'] if r['request_id']=='measurement:generalist_2#0:95')
        self.assertEqual(row['terminal_service_ms'],30);self.assertEqual(row['service_status'],'UNKNOWN');self.assertIsNone(row['service_ms'])
        self.assertEqual(row['terminal_quality']['status'],'AVAILABLE');self.assertEqual(row['quality']['status'],'UNKNOWN')
        self.assertEqual(original['request_diagnostics'][-1]['service_status'],'AVAILABLE')

    def test_crossing_handler_is_clipped_and_not_scored(self):
        result=w.slice_job(self.fixture());cutoff=result['deadline']['cutoff_unix_ns']
        self.assertTrue(all(i['end_ns']<=cutoff for i in result['observed_handler_intervals_clipped_to_deadline']))
        cross=next(i for i in result['observed_handler_intervals_clipped_to_deadline'] if i['request_id']=='measurement:generalist_2#0:83')
        self.assertEqual(cross['end_ns'],cutoff);self.assertEqual(cross['end_ns']-cross['start_ns'],10_000_000)
        row=next(r for r in result['request_diagnostics'] if r['request_id']==cross['request_id']);self.assertIsNone(row['activity'])

    def test_unreleased_primary_remains_distinct_from_late(self):
        report=self.fixture();row=next(r for r in report['request_diagnostics'] if r['request_id']=='measurement:generalist_2#0:95')
        row['service_status']='UNKNOWN';row['quality']={'status':'UNKNOWN'};row['original_record']['scheduled_time_ns']=None
        row['original_record']['completion_time_ns']=None
        result=w.slice_job(report)
        self.assertEqual(result['aliases']['generalist_2#0']['primary_deadline_classes']['UNRELEASED_PRIMARY_UNKNOWN'],1)
        self.assertEqual(result['aliases']['generalist_2#0']['primary']['planned_requests'],96)
        self.assertFalse(result['coverage']['passed'])

    def test_hard600_cutoff_overrides_late_window_end(self):
        report=self.fixture();window=report['measurement_timing']['measurement_window'];window['end_time_ns']=window['start_time_ns']+601_000_000_000
        result=w.slice_job(report);self.assertEqual(result['deadline']['duration_seconds'],600)
        self.assertEqual(result['status'],'INSUFFICIENT_DIAGNOSTIC_EVIDENCE');self.assertIsNone(result['U'])

    def test_same_input_repeat_pairs_exclude_late_values(self):
        joint=w.slice_job(self.fixture());baseline=f.analyze(f.fixture(state='B'));window=baseline['measurement_timing']['measurement_window'];window['start_unix_ns']=window['start_time_ns']
        baseline=w.slice_job(baseline);pairs=w.paired_rows(baseline,joint)
        self.assertEqual(len(pairs),96);self.assertEqual(sum(r['joint_over_baseline'] is not None for r in pairs),83)


if __name__=='__main__':unittest.main()
