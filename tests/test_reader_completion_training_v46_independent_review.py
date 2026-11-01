"""Independent loader accounting/label integrity checks using full69 CPU fixtures."""
from collections import Counter
import copy
import unittest

from tools.predictor_validation_20261027 import reader_completion_training_v46 as v


def dataset():
    raw=dict(request_records=[],priming_request_records=[]);risks=[];labels=[];counts=Counter()
    for index in range(69):
        cohort='priming' if index<12 else 'measured';request='r'+str(index);lo=1000+index*200;hi=lo+100
        worker=dict(request_id=request,start_perf_ns=lo-10,end_perf_ns=hi-20)
        ready=dict(request_id=request,perf_ns=hi-10,received_time_ns=hi+10)
        row=dict(request_id=request,model='text#0',task='text',pool_id='public',status='completed',
            scheduled_time_ns=lo-100,enqueued_time_ns=lo-90,actual_submit_time_ns=lo-80,
            submit_write_completed_time_ns=lo-70,execution_events=[dict(perf_ns=lo-5,received_time_ns=lo)],
            response_reader_event_enqueue_time_ns=hi,completion_time_ns=hi,worker_response=worker,
            previous_handler_next_request_ready=ready)
        raw['priming_request_records' if cohort=='priming' else 'request_records'].append(row)
        risks.append(dict(capture='capture',request_id=request,cohort=cohort,origin_reader_START_ns=lo,
            start_ns=lo,end_ns=hi,age_entry_ns=0,age_exit_ns=100,completion_observed=True,
            features=dict(alias='text#0',task='text',pool_id='public',attained_reader_handler_age_ns=0,
                active_handler_by_owner={o:o=='text#0' for o in v.hazard.OWNERS})))
        labels.append(dict(capture='capture',request_id=request,cohort=cohort,alias='text#0',task='text',pool_id='public',
            scheduled_time_ns=row['scheduled_time_ns'],enqueued_time_ns=row['enqueued_time_ns'],
            dispatch_begin_ns=row['actual_submit_time_ns'],dispatch_write_completed_ns=row['submit_write_completed_time_ns'],
            origin_reader_START_ns=lo,RESULT_reader_received_ns=hi,worker_handler_start_perf_ns=worker['start_perf_ns'],
            worker_handler_end_perf_ns=worker['end_perf_ns'],worker_next_ready_perf_ns=ready['perf_ns'],
            worker_next_ready_reader_received_ns=ready['received_time_ns'],worker_response_labels=copy.deepcopy(worker),
            worker_next_ready_record=copy.deepcopy(ready)))
        counts[cohort+'_requests']+=1;counts[cohort+'_risk_intervals']+=1;counts[cohort+'_exposure_ns']+=100;counts[cohort+'_completion_events']+=1
    return dict(capture='capture',counts=dict(counts)),raw,risks,labels


class IndependentCompletionLoaderTests(unittest.TestCase):
    def test_complete_original_57_plus12_reader_risk_is_accepted(self):
        report,raw,risks,labels=dataset()
        self.assertEqual(v.verify_rows(report,raw,risks,labels),report['counts'])

    def test_each_flat_worker_and_transport_label_must_match_raw(self):
        report,raw,risks,labels=dataset()
        for field in ('scheduled_time_ns','enqueued_time_ns','dispatch_begin_ns','dispatch_write_completed_ns',
                'worker_handler_start_perf_ns','worker_handler_end_perf_ns','worker_next_ready_perf_ns',
                'worker_next_ready_reader_received_ns'):
            changed=copy.deepcopy(labels);changed[0][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):v.verify_rows(report,raw,risks,changed)

    def test_reclassifying_priming_as_measured_cannot_hide_in_total69(self):
        report,raw,risks,labels=dataset();moved=raw['priming_request_records'].pop();raw['request_records'].append(moved)
        for row in [*risks,*labels]:
            if row['request_id']==moved['request_id']:row['cohort']='measured'
        order={row['request_id']:i for i,row in enumerate(raw['priming_request_records']+raw['request_records'])}
        risks.sort(key=lambda row:order[row['request_id']])
        for suffix,width in (('requests',1),('risk_intervals',1),('exposure_ns',100),('completion_events',1)):
            report['counts']['measured_'+suffix]+=width;report['counts']['priming_'+suffix]-=width
        with self.assertRaises(ValueError):v.verify_rows(report,raw,risks,labels)

    def test_worker_clock_origin_and_wrong_reader_peer_mask_are_rejected(self):
        report,raw,risks,labels=dataset()
        wrong=copy.deepcopy(risks);wrong[0]['features']['active_handler_by_owner']['ocr#0']=True
        with self.assertRaises(ValueError):v.verify_rows(report,raw,wrong,labels)
        wrong=copy.deepcopy(risks);wrong[0]['origin_reader_START_ns']-=10
        with self.assertRaises(ValueError):v.verify_rows(report,raw,wrong,labels)


if __name__=='__main__':unittest.main()
