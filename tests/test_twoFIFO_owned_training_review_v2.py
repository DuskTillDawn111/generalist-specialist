"""Contract counterexamples, including actual-source late-write scheduling."""
import copy
import unittest

from tests.test_original_sender_late_write_record_independent_v1 import source_counterexample
from tests.test_stage_host_partition_v1 import fixture as macro_fixture
from tools.predictor_validation_20261027.twoFIFO_owned_training_review_v2 import (
    inspect_request_clock, inspect_cohort, PLAN_SCHEMA, RAW_SCHEMA,
    MEASUREMENT_NS, DRAIN_NS, digest)
from tools.predictor_validation_20261027.twoFIFO_composable_macros_v2 import request_rows


def request(rid, owner, start, job, arrival_index=0):
    row=macro_fixture();row.update(request_id=rid,model=owner,profile_job_id=job,split='training',
        arrival_index=arrival_index,arrival_ns=0,scheduled_time_ns=start-20,enqueued_time_ns=start-10,
        actual_submit_time_ns=start-5,submit_write_completed_time_ns=start+1)
    delta=start-100
    for key in ('observed_worker_start_time_ns','completion_time_ns','response_reader_event_enqueue_time_ns',
                'response_collector_dequeued_time_ns','response_collector_processing_time_ns'):
        row[key]+=delta
    row['worker_response'].update(request_id=rid,start_perf_ns=start,end_perf_ns=start+100)
    for event in row['execution_events']+[row['previous_handler_next_request_ready']]:
        event['request_id']=rid
        for key in ('perf_ns','received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns'):
            event[key]+=delta
        if event.get('segment_id')=='decode_0001':event['semantic_phase']='decode'
    row['execution_events'].insert(0,dict(request_id=rid,perf_ns=start,received_time_ns=start+2,
        collector_dequeued_time_ns=start+3,collector_processing_time_ns=start+4))
    row['previous_handler_next_request_ready'].update(received_time_ns=start+116,
        collector_dequeued_time_ns=start+117,collector_processing_time_ns=start+118)
    return row


def fixture():
    priming=[request('prime'+str(i),'text#0',1000+i*1000,'priming') for i in range(18)]
    epoch=100_000
    rows=[request('first','text#0',epoch+20,'case',0),
          request('second','text#0',epoch+220,'case',1),
          request('peer','text#1',epoch+20,'case',2)]
    # Both same-owner requests arrive together; worker serialization is not a
    # global cap and cannot be inferred from public arrival tie alone.
    for row in rows:row['scheduled_time_ns']=epoch
    arrivals=[dict(request_id=r['request_id'],owner=r['model'],task=r['task'],pool_id=r['pool_id'],
                   arrival_ns=0,arrival_index=r['arrival_index']) for r in rows]
    case=dict(case_id='case',arrivals=arrivals,public_arrivals_sha256=digest(arrivals))
    common=dict(split='training',capture_id='block',attempt_id='shard',execution_id='execution')
    plan=dict(schema=PLAN_SCHEMA,cases=[case],priming=[dict(owner=r['model'],task=r['task'],pool_id=r['pool_id']) for r in priming],**common)
    raw=dict(schema=RAW_SCHEMA,backend='CPU_TEST_ONLY',request_records=rows,priming_request_records=priming,
        cases=[dict(case_id='case',public_arrivals_sha256=case['public_arrivals_sha256'],
            request_ids=[r['request_id'] for r in rows],epoch_ns=epoch,measurement_end_ns=epoch+MEASUREMENT_NS,
            fixed_drain_deadline_ns=epoch+MEASUREMENT_NS+DRAIN_NS,
            observation_finalized_ns=epoch+MEASUREMENT_NS+DRAIN_NS)],**common)
    return plan,raw


class OwnedTrainingReviewTests(unittest.TestCase):
    def test_original_source_RESULT_before_sender_record_is_accepted(self):
        row,errors=source_counterexample()
        self.assertFalse(errors)
        self.assertTrue(inspect_request_clock(row)['late_write_after_RESULT'])
        row['submit_write_completed_time_ns']=row['actual_submit_time_ns']-1
        with self.assertRaises(ValueError):inspect_request_clock(row)

    def test_same_owner_FIFO_does_not_limit_parallel_other_owner(self):
        plan,raw=fixture();result=inspect_cohort(plan,raw)
        self.assertEqual(result['offered_requests'],3);self.assertEqual(result['issues'],[])
        self.assertEqual(raw['request_records'][0]['observed_worker_start_time_ns'],
                         raw['request_records'][2]['observed_worker_start_time_ns'])
        for which in ('sender','worker'):
            altered=copy.deepcopy(raw)
            previous,current=altered['request_records'][:2]
            if which=='sender':current['actual_submit_time_ns']=previous['submit_write_completed_time_ns']-1
            else:previous['previous_handler_next_request_ready']['perf_ns']=current['observed_worker_start_time_ns']+1
            with self.subTest(which=which),self.assertRaises(ValueError):inspect_cohort(plan,altered)

    def test_fixed_window_and_offered_rows_cannot_be_removed_or_retimed(self):
        for which in ('delete','cutoff','arrival','owner'):
            plan,raw=fixture()
            if which=='delete':raw['request_records'].pop()
            elif which=='cutoff':raw['cases'][0]['fixed_drain_deadline_ns']+=1
            elif which=='arrival':raw['request_records'][0]['scheduled_time_ns']+=1
            else:raw['request_records'][0]['model']='generalist_2#0'
            with self.subTest(which=which),self.assertRaises(ValueError):inspect_cohort(plan,raw)

    def test_reversing_public_FIFO_with_self_consistent_runtime_clocks_is_rejected(self):
        plan,raw=fixture();first,second=raw['request_records'][:2]
        for key in ('enqueued_time_ns','actual_submit_time_ns','submit_write_completed_time_ns',
            'observed_worker_start_time_ns','completion_time_ns','response_reader_event_enqueue_time_ns',
            'response_collector_dequeued_time_ns','response_collector_processing_time_ns','worker_response',
            'previous_handler_next_request_ready','execution_events'):
            first[key],second[key]=second[key],first[key]
        # Preserve each row's request identities while reversing execution order.
        for row in (first,second):
            row['worker_response']['request_id']=row['request_id']
            for event in row['execution_events']+[row['previous_handler_next_request_ready']]:event['request_id']=row['request_id']
        with self.assertRaisesRegex(ValueError,'public per-owner'):inspect_cohort(plan,raw)

    def test_delayed_prior_READY_cannot_cross_next_START_in_parent_FIFO(self):
        plan,raw=fixture()
        for key in ('received_time_ns','collector_dequeued_time_ns','collector_processing_time_ns'):
            raw['request_records'][0]['previous_handler_next_request_ready'][key]+=1_000_000
        with self.assertRaisesRegex(ValueError,'producer READY'):inspect_cohort(plan,raw)

    def test_failed_unstarted_rows_stay_in_full_denominator_and_are_not_macros(self):
        plan,raw=fixture();raw['request_records'][1]['status']='incomplete'
        result=inspect_cohort(plan,raw)
        self.assertEqual(result['offered_requests'],3);self.assertEqual(result['completed_requests'],2)
        self.assertTrue(result['issues'])
        with self.assertRaises(ValueError):request_rows(raw,plan,'cpu-only')
        plan,raw=fixture()
        raw['cases'][0]['epoch_ns']=None
        for row in raw['request_records']:row.update(status='not_started',scheduled_time_ns=None)
        result=inspect_cohort(plan,raw)
        self.assertEqual(result['unstarted_requests'],3);self.assertEqual(result['offered_requests'],3)

    def test_macro_adapter_retains_priming_history_but_excludes_priming_labels(self):
        plan,raw=fixture()
        for row in raw['priming_request_records']:row.pop('profile_job_id')
        before=copy.deepcopy(raw)
        result,events=request_rows(raw,plan,'cpu-only')
        self.assertEqual(raw,before);self.assertEqual(len(result),3)
        self.assertEqual(len(events),sum(len(r['execution_events']) for r in raw['request_records']+raw['priming_request_records']))
        self.assertEqual(sum(len(r['macros']) for r in result),12)
        self.assertEqual(sum(r['worker_start_to_ready_ns'] for r in result),330)
        # Peer concurrent prefill is available on the global history path.
        self.assertTrue(any(piece['state']['peer_visible_macro_owner_count']>0
            for r in result for macro in r['macros'] for piece in macro['causal_covariate_path']))


if __name__=='__main__':unittest.main()
