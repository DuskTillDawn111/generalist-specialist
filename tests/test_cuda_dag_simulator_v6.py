"""Numerical progression, FIFO and diagnostic contracts for the v6 event axis."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cuda_dag_simulator_v4 import DagFullHorizonModel as OldEngine
from tools.predictor_validation_20261027.cuda_dag_simulator_v6 import DagFullHorizonModel
from tests import test_cuda_dag_simulator_v2 as original
from tests.test_cuda_dag_simulator_v2 import ALIASES, plan, requests


class PreservedDagContracts(original.DagSimulatorTests):
    def model(self, **options):
        return DagFullHorizonModel(ALIASES, {(a,'task','input'):plan() for a in ALIASES}, **options)


class QueueDiagnosticIntegrationTests(unittest.TestCase):
    def test_real_event_path_reports_full_work_fifo_and_recovery(self):
        from tools.predictor_validation_20261027.dag_queue_diagnostics_v6 import DagQueueDiagnostics
        aliases={'owner':dict(family='text',supported_tasks=['text'])}
        plans={('owner','text',pool):dict(nodes=[dict(id='gpu',kind='gpu',duration_ns=ns)],
                                       edges=[],unresolved=[],cyclic=False)
               for pool,ns in [('long',100_000_000),('short',50_000_000)]}
        rows=[dict(request_id='first',alias='owner',task='text',pool_id='long',arrival_ns=0),
              dict(request_id='second',alias='owner',task='text',pool_id='short',arrival_ns=25_000_000)]
        diagnostic=DagQueueDiagnostics(aliases,plans,lambda node,row:node['duration_ns'])
        result=DagFullHorizonModel(aliases,plans,duration_provider=diagnostic.duration,
            queue_diagnostics=diagnostic,max_events=30).forecast(rows,0,200_000_000)
        q=result['queue_diagnostics']
        self.assertEqual(q['max_remaining_activity_work_ms'],125.)
        self.assertEqual(q['remaining_activity_work_ms'],0.)
        self.assertEqual(q['owner_fifo_head_blocked_ms'],{'owner':75.})
        self.assertEqual(q['owner_fifo_request_wait_ms'],{'owner':75.})
        self.assertEqual(q['max_oldest_waiting_age_ms'],75.)
        self.assertEqual(q['queue_recovery_episodes'],[dict(backlog_start_ns=25_000_000,
                          queue_clear_ns=100_000_000,recovery_ms=75.)])
        self.assertEqual(result['queue_wait_ns']['second'],75_000_000)
        self.assertEqual(result['finishes']['second']['finish_ns'],150_000_000)
        self.assertIsNone(q['max_normalized_resource_work_ms']['sm'])
        self.assertFalse(q['global_admission_hol_applicable'])


class IntegerEventClockTests(unittest.TestCase):
    def model(self, duration, engine=DagFullHorizonModel, **options):
        nodes=[dict(id='x',kind='gpu',duration_ns=duration,release_ns=0)]
        p=dict(nodes=nodes,edges=[],unresolved=[],cyclic=False)
        return engine(ALIASES,{(a,'task','input'):p for a in ALIASES},
                      allow_profiled_replay=True,max_events=50,**options)

    def test_old_large_absolute_clock_counterexample_stays_unknown(self):
        at=30_000_000_000;rows=[dict(requests()[0],arrival_ns=at)]
        with self.assertRaisesRegex(UnknownEvidence,'EVENT_BUDGET'):
            self.model(10000.0000018,engine=OldEngine).forecast(rows,at,at+20_000)

    def test_integer_relative_completion_resolves_counterexample_without_budget_change(self):
        for at in (0,30_000_000_000,100_000_000_000):
            with self.subTest(at=at):
                rows=[dict(requests()[0],arrival_ns=at)]
                result=self.model(10000.0000018).forecast(rows,at,at+20_000)
                self.assertEqual(result['finishes']['a']['finish_ns']-at,10001)
                self.assertLess(result['events'],10)
                self.assertEqual(result['event_time_resolution_ns'],1)
                self.assertTrue(all(type(t['time_ns']) is int and type(t['until_ns']) is int
                                    for t in result['resource_transitions']))

    def test_sub_nanosecond_service_has_positive_progress(self):
        at=100_000_000_000;row=dict(requests()[0],arrival_ns=at)
        result=self.model(1.8e-6).forecast([row],at,at+100)
        self.assertEqual(result['finishes']['a']['finish_ns'],at+1)

    def test_rate_change_at_arrival_uses_integer_elapsed_consumption(self):
        at=30_000_000_000
        rows=[dict(requests()[0],arrival_ns=at),dict(requests()[1],arrival_ns=at+5)]
        result=self.model(10.25,allow_context_time_slice_hypothesis=True).forecast(rows,at,at+100)
        # a consumes 5 alone, then 5.5 across 11 shared ticks; b has 4.75 left.
        self.assertEqual(result['finishes']['a']['finish_ns'],at+16)
        self.assertEqual(result['finishes']['b']['finish_ns'],at+21)

    def test_resources_use_actual_integer_interval_and_measurement_only_energy(self):
        class Resources:
            def power(self,*args):return dict(known=True,watts=2.)
            def memory(self,*args):return dict(known=True,peak_memory_mib=42.)
        at=30_000_000_000;row=dict(requests()[0],arrival_ns=at)
        r=Resources()
        result=self.model(10.25,power_provider=r,memory_provider=r,require_energy=True,
                          require_memory=True).forecast([row],at,at+5,drain_deadline_ns=at+100)
        self.assertEqual(result['finishes']['a']['finish_ns'],at+11)
        self.assertEqual(result['board_energy_joules'],2*5/1e9)
        self.assertEqual(result['peak_memory_mib'],42.)
        self.assertEqual(result['memory_window_ns'],[at,at+11])

    def test_multistream_dependency_host_delay_and_lifecycle_stay_on_same_axis(self):
        class Lifecycle:
            def delays(self,*args):return dict(known=True,parent_receipt_ms=.000005,
                                              worker_next_ready_ms=.000003)
        at=30_000_000_000
        p=dict(nodes=[dict(id='x',kind='gpu'),dict(id='y',kind='gpu'),dict(id='end',kind='handler_end')],
               edges=[dict(before='x',after='end'),dict(before='y',after='end')],
               unresolved=[],cyclic=False)
        rows=[dict(requests()[0],arrival_ns=at),dict(requests()[0],request_id='next',arrival_ns=at+1)]
        result=DagFullHorizonModel(ALIASES,{('a','task','input'):p},max_events=40,
            duration_provider=lambda node,row:10.25 if node['id']=='x' else 20.25,
            release_provider=lambda node,row:dict(not_before_offset_ns=0,
                after_predecessors_ns=2 if node['id']=='end' else 0),
            lifecycle_provider=Lifecycle()).forecast(rows,at,at+100)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)
        self.assertEqual(result['finishes']['a']['handler_end_ns'],at+23)
        self.assertEqual(result['finishes']['a']['finish_ns'],at+28)
        self.assertEqual(result['starts']['next'],at+26)
        self.assertEqual(result['finishes']['next']['finish_ns'],at+54)

    def test_budget_and_drain_never_invent_finish(self):
        at=30_000_000_000;row=dict(requests()[0],arrival_ns=at)
        result=self.model(10.25).forecast([row],at,at+5,drain_deadline_ns=at+6)
        self.assertNotIn('a',result['finishes'])
        model=self.model(10.25);model.max_events=1
        with self.assertRaisesRegex(UnknownEvidence,'EVENT_BUDGET'):
            model.forecast([row],at,at+100)

    def test_diagnostic_hooks_follow_arrival_start_and_interval_order(self):
        class Diagnostics:
            def __init__(self):self.calls=[]
            def register(self,row):self.calls.append(('register',row['request_id']))
            def mark_started(self,row):self.calls.append(('start',row['request_id'],row['start_ns']))
            def observe(self,clock,delta,pending,running,active,rates):
                self.calls.append(('observe',clock,delta,len(running),len(active)))
            def result(self,cohort):return dict(request_ids=sorted(cohort),calls=self.calls)
        at=30_000_000_000;d=Diagnostics();row=dict(requests()[0],arrival_ns=at)
        result=self.model(10.25,queue_diagnostics=d).forecast([row],at,at+100)
        q=result['queue_diagnostics']
        self.assertEqual(q['calls'][0],('register','a'))
        self.assertEqual(q['calls'][1],('start','a',at))
        self.assertEqual(q['calls'][2],('observe',at,11,1,1))
        self.assertEqual(q['max_waiting_requests'],0)
        self.assertEqual(q['request_ids'],['a'])


if __name__=='__main__':unittest.main()
