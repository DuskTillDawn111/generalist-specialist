"""Actual reader visibility gates the real DAG engine, without future labels."""
import copy
import unittest

from tools.predictor_validation_20261027.cuda_dag_simulator_v7 import DagFullHorizonModel
from tools.predictor_validation_20261027.dag_visible_observer_v8 import (
    observe_cutoff, forecast_visible_suffix, public_dag_cohort,
)

CONTEXT = 'a' * 64
ALIASES = {'a': dict(family='text', supported_tasks=['text'])}


def duration(node, row):
    return node['duration_ns']


def release(node, row):
    return dict(not_before_offset_ns=0, after_predecessors_ns=0)


class Life:
    def delays(self, *args):
        return dict(known=True, parent_receipt_ms=.000001, worker_next_ready_ms=.000002)


class Resource:
    def power(self, *args):
        return dict(known=True, watts=20.)

    def memory(self, *args):
        return dict(known=True, peak_memory_mib=30.)


class Poison(dict):
    def get(self, *args):
        raise AssertionError('Future response body was read')

    def __getitem__(self, key):
        raise AssertionError('Future response body was read')


def native(rid, arrival, submit, start, end, receipt, ready, ready_receipt):
    return dict(request_id=rid, model='a', task='text', pool_id='x', scheduled_time_ns=arrival,
        actual_submit_time_ns=submit,
        execution_events=[dict(request_id=rid, perf_ns=start, pid=7, received_time_ns=start+1)],
        response_reader_event_enqueue_time_ns=receipt,
        worker_response=dict(end_perf_ns=end, generated_tokens=9999, prediction='hidden', quality=1),
        previous_handler_next_request_ready=dict(request_id=rid, pid=7, perf_ns=ready,
            received_time_ns=ready_receipt, clock='host_perf_counter_ns',
            boundary='previous_handler_next_request_ready', scope='after_response_flush_before_next_read'))


def fixture():
    rows=[dict(request_id='r0',alias='a',task='text',pool_id='x',arrival_ns=100),
          dict(request_id='r1',alias='a',task='text',pool_id='x',arrival_ns=200)]
    raw=dict(schema='gsdeploy.cpu_fixture.native.v8',execution_context_sha256=CONTEXT,
        phase_start_ns=100,ready={'a':dict(pid=7)},
        priming_request_records=[native('prime',10,11,12,20,21,22,23)],
        request_records=[native('r0',100,101,102,120,122,123,125),
                         native('r1',200,201,202,220,222,223,225)])
    raw['initial_request_inventory']=[{k:r[k] for k in ('request_id','model','task','pool_id')} for r in raw['priming_request_records']]
    return rows,raw


def snapshot(cutoff, raw=None, rows=None):
    default_rows,default_raw=fixture()
    return observe_cutoff(default_raw if raw is None else raw,
        default_rows if rows is None else rows, ALIASES, execution_context_sha256=CONTEXT,
        phase_start_ns=100,deadline_ns=300,drain_deadline_ns=400,cutoff_ns=cutoff,
        allow_cpu_fixture=True)


def factory(*, now_ns):
    plans={('a','text','x'):dict(nodes=[dict(id='gpu',kind='gpu',duration_ns=10),
        dict(id='end',kind='handler_end')],edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)}
    return DagFullHorizonModel(ALIASES,plans,duration_provider=duration,release_provider=release,
        lifecycle_provider=Life(),power_provider=Resource(),memory_provider=Resource())


class VisibleDagTests(unittest.TestCase):
    def test_idle_cutoff_calls_actual_engine_and_scores_original_prefix_plus_future(self):
        seen=[]
        def call(**kwargs):
            seen.append(kwargs);return factory(**kwargs)
        state=snapshot(150)
        result=forecast_visible_suffix(state,call,score_kwargs=dict(rates={'text':1},refs={'text':1},
            quality_loss={'text':{'text':.1}},memory_budget_mib=100,energy_reference_watts=20))
        self.assertEqual(result['status'],'IDLE_SUFFIX_FORECAST')
        self.assertEqual(seen,[{'now_ns':150}])
        self.assertEqual(set(result['full_public_cohort']),{'r0','r1'})
        self.assertEqual(result['completed_prefix'][0]['arrival_ns'],100)
        self.assertEqual(result['completed_prefix'][0]['completion_ns'],122)
        self.assertEqual(result['forecast']['cohort']['r1']['arrival_ns'],200)
        self.assertEqual(result['forecast']['finishes']['r1']['finish_ns'],211)
        self.assertEqual(result['metrics']['cohort_size'],2)
        self.assertAlmostEqual(result['metrics']['per_task']['text']['mean_ms'],.0000165)
        self.assertIsNone(result['metrics']['full_U'])
        self.assertIsNone(result['metrics']['terms']['M'])
        self.assertIsNone(result['metrics']['terms']['E'])
        self.assertIsNotNone(result['suffix_energy_diagnostic_joules'])
        self.assertIsNone(result['prefix_resources']['board_energy_joules'])
        self.assertFalse(result['rolling_risk_correction_complete'])

    def test_busy_and_both_delayed_boundaries_never_call_engine(self):
        for cutoff,status in [(101,'SUBMITTED_START_NOT_VISIBLE'),(110,'STARTED_RESIDUAL_UNKNOWN'),
                              (123,'HANDLER_END_READY_NOT_VISIBLE')]:
            with self.subTest(cutoff=cutoff):
                state=snapshot(cutoff)
                self.assertEqual(state['request_status']['r0'],status)
                result=forecast_visible_suffix(state,lambda **kw:self.fail('Busy engine called'))
                self.assertEqual(result['status'],'UNKNOWN')
                self.assertEqual(set(result['full_public_cohort']),{'r0','r1'})
                self.assertIsNone(state['owners']['a']['kernel_cursor'])

    def test_future_response_body_is_not_touched_and_future_outcome_changes_do_not_change_snapshot(self):
        rows,raw=fixture();raw['request_records'][1]['worker_response']=Poison()
        first=snapshot(110,raw,rows)
        raw['request_records'][0]['worker_response']=Poison()
        second=snapshot(110,raw,rows)
        self.assertEqual(first,second)
        text=str(first)
        self.assertNotIn('generated_tokens',text)
        self.assertNotIn('hidden',text)
        self.assertEqual(first['future_suffix'],[rows[1]])

    def test_phase_reader_time_controls_visibility_without_inventing_progress(self):
        rows,raw=fixture();events=raw['request_records'][0]['execution_events']
        events += [dict(request_id='r0',perf_ns=105,received_time_ns=115,event='start',segment_id='decode',kind='gpu'),
                   dict(request_id='r0',perf_ns=107,received_time_ns=116,event='end',segment_id='decode',kind='gpu')]
        early=snapshot(110,raw,rows);later=snapshot(115,raw,rows)
        self.assertEqual(early['observed_host_state']['requests']['r0']['phases'],{})
        phase=later['observed_host_state']['requests']['r0']['phases']['decode']
        self.assertEqual(phase['start_ns'],105);self.assertIsNone(phase['end_ns'])
        self.assertIsNone(later['observed_host_state']['requests']['r0']['kernel_cursor'])

    def test_arrived_but_unsent_and_unknown_initial_ready_are_not_idle(self):
        self.assertIn('ARRIVED_DISPATCH_UNOBSERVED:r0',snapshot(100)['unknown_reasons'])
        rows,raw=fixture();raw['priming_request_records']=[]
        with self.assertRaises(ValueError):snapshot(150,raw,rows)
        rows,raw=fixture();raw['priming_request_records'][0]['previous_handler_next_request_ready']['received_time_ns']=151
        self.assertIn('INITIAL_OWNER_IDLE_NOT_VERIFIED:a',snapshot(150,raw,rows)['unknown_reasons'])

    def test_missing_future_registry_and_last_priming_row_are_rejected(self):
        rows,raw=fixture();raw['request_records'].pop()
        with self.assertRaisesRegex(ValueError,'complete original public cohort'):
            snapshot(150,raw,rows)
        rows,raw=fixture()
        extra=native('prime:last',30,31,32,40,41,42,43)
        raw['priming_request_records'].append(extra)
        raw['initial_request_inventory'].append({k:extra[k] for k in ('request_id','model','task','pool_id')})
        self.assertEqual(snapshot(150,raw,rows)['owners']['a']['status'],'IDLE_READY_OBSERVED')
        raw['priming_request_records'].pop()
        with self.assertRaisesRegex(ValueError,'Complete planned priming/warmup'):
            snapshot(150,raw,rows)

    def test_owner_route_input_arrival_context_and_snapshot_mutation_rejected(self):
        for key,value in [('model','other'),('task','ocr'),('pool_id','other'),('scheduled_time_ns',101)]:
            rows,raw=fixture();raw['request_records'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):snapshot(150,raw,rows)
        rows,raw=fixture();raw['execution_context_sha256']='b'*64
        with self.assertRaises(ValueError):snapshot(150,raw,rows)
        state=snapshot(150);state['future_suffix']=[]
        with self.assertRaises(ValueError):forecast_visible_suffix(state,factory)

    def test_public_gate_schedule_routes_entire_future_once_before_cutoff(self):
        public=dict(phase_start_ns=100,now_ns=100,deadline_ns=300,drain_deadline_ns=400,
            queues={'a':[dict(request_id='r0',alias='a',task='text',pool_id='x',arrival_ns=100)]},
            future_workload=dict(rates={'text':1},input_pools={'text':[dict(pool_id='x',weight=1)]},
                process='declared_schedule',declared_arrivals=[dict(request_id='r1',task='text',pool_id='x',offset_ns=100)]))
        rows=public_dag_cohort(public,ALIASES,{'text':{'a':1}})
        self.assertEqual(set(rows),{'r0','r1'})
        self.assertEqual(rows['r1']['arrival_ns'],200)
        self.assertEqual(rows['r1']['alias'],'a')


if __name__=='__main__':unittest.main()
