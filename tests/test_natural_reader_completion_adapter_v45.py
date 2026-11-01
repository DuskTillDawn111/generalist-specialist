import copy
import unittest

from tools.predictor_validation_20261027 import natural_reader_completion_adapter_v45 as v
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import IntervalIndex


def record(owner='text#0',start=100,end=400,physical_start=1):
    return dict(model=owner,task='text',pool_id='public-input',request_id='request',
        execution_events=[dict(perf_ns=physical_start,received_time_ns=start)],
        response_reader_event_enqueue_time_ns=end,completion_time_ns=end,
        worker_response=dict(start_perf_ns=physical_start,end_perf_ns=end-50,generated_tokens=999),
        previous_handler_next_request_ready=dict(perf_ns=end-20))


def indexes(values):return {o:IntervalIndex(values.get(o,[])) for o in v.OWNERS}


class NaturalReaderCompletionTests(unittest.TestCase):
    def test_reader_boundaries_drive_half_open_masks_and_unique_completion(self):
        row=record();ix=indexes({'text#0':[(100,400)],'ocr#0':[(150,250)]})
        pieces=list(v.split_request(row,'measured','capture',ix,[100,150,250,400]))
        self.assertEqual([(p['start_ns'],p['end_ns']) for p in pieces],[(100,150),(150,250),(250,400)])
        self.assertEqual([p['completion_observed'] for p in pieces],[False,False,True])
        self.assertEqual(sum(p['end_ns']-p['start_ns'] for p in pieces),300)
        self.assertEqual([p['features']['active_handler_by_owner']['ocr#0'] for p in pieces],[False,True,False])
        self.assertEqual([p['age_entry_ns'] for p in pieces],[0,50,150])
        self.assertEqual(set(pieces[0]['features']),v.FEATURE_FIELDS)

    def test_physical_handler_and_ready_changes_do_not_change_reader_risk(self):
        row=record();changed=copy.deepcopy(row)
        changed['worker_response'].update(start_perf_ns=60,end_perf_ns=350,generated_tokens=1)
        changed['previous_handler_next_request_ready']['perf_ns']=360
        ix=indexes({'text#0':[(100,400)]})
        self.assertEqual(list(v.split_request(row,'measured','c',ix,[100,400])),
                         list(v.split_request(changed,'measured','c',ix,[100,400])))
        bad=copy.deepcopy(row);bad['execution_events']=[]
        with self.assertRaises(ValueError):v.reader_interval(bad)

    def test_priming_history_remains_active_during_first_measured_request(self):
        ix=indexes({'text#0':[(100,400)],'ocr#0':[(50,250)]})
        pieces=list(v.split_request(record(),'measured','capture',ix,[50,100,250,400]))
        self.assertTrue(pieces[0]['features']['active_handler_by_owner']['ocr#0'])
        self.assertFalse(pieces[-1]['features']['active_handler_by_owner']['ocr#0'])
        self.assertTrue(all(p['cohort']=='measured' for p in pieces))

    def test_future_peer_RESULT_mutation_cannot_change_earlier_prefix_features(self):
        early=indexes({'text#0':[(100,400)],'ocr#0':[(50,300)]})
        late=indexes({'text#0':[(100,400)],'ocr#0':[(50,600)]})
        a=list(v.split_request(record(),'measured','c',early,[50,100,200,300,400]))
        b=list(v.split_request(record(),'measured','c',late,[50,100,200,400,600]))
        self.assertEqual(a[0],b[0])
        for bad in ('future_RESULT_ns','generated_tokens','worker_ready_ns','physical_context'):
            value=dict(a[0]['features'],**{bad:1})
            with self.assertRaises(ValueError):v.validate_features(value)


if __name__=='__main__':unittest.main()
