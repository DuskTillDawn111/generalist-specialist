import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cuda_dag_simulator_v2 import DagFullHorizonModel


ALIASES={a:dict(family=a,supported_tasks=['task']) for a in ('a','b')}


def plan():
    return dict(nodes=[dict(id='x',kind='gpu',duration_ns=10,release_ns=0),
                       dict(id='y',kind='gpu',duration_ns=20,release_ns=0)],edges=[],unresolved=[],cyclic=False)


def requests():
    return [dict(request_id=a,alias=a,task='task',pool_id='input',arrival_ns=0) for a in ALIASES]


class DagSimulatorTests(unittest.TestCase):
    def model(self,**options):
        return DagFullHorizonModel(ALIASES,{(a,'task','input'):plan() for a in ALIASES},**options)

    def test_parallel_owner_streams_not_summed(self):
        result=self.model(allow_profiled_replay=True).forecast(requests()[:1],0,100)
        self.assertEqual(result['finishes']['a']['finish_ns'],20)
        self.assertEqual(result['max_simultaneous_nodes_per_owner']['a'],2)
        self.assertEqual(result['completed_dag_nodes'],2)

    def test_cross_owner_time_slice_and_future_fifo(self):
        rows=requests();rows.append(dict(rows[0],request_id='a2',arrival_ns=5))
        result=self.model(allow_profiled_replay=True,allow_context_time_slice_hypothesis=True).forecast(rows,0,100)
        self.assertEqual(result['finishes']['a']['finish_ns'],40)
        self.assertEqual(result['finishes']['b']['finish_ns'],40)
        self.assertEqual(result['finishes']['a2']['finish_ns'],60)
        self.assertEqual(result['queue_wait_ns']['a2'],35)
        self.assertEqual(result['queue_diagnostics']['max_waiting_requests'],1)

    def test_dependency_release_and_copy_remain_explicit(self):
        p=plan();p['nodes'].append(dict(id='copy',kind='copy',duration_ns=3,release_ns=25))
        p['edges']=[dict(before='x',after='copy')]
        model=DagFullHorizonModel(ALIASES,{('a','task','input'):p},allow_profiled_replay=True)
        with self.assertRaises(UnknownEvidence):model.forecast(requests()[:1],0,100)
        model.allow_independent_copy_hypothesis=True
        result=model.forecast(requests()[:1],0,100)
        self.assertEqual(result['finishes']['a']['finish_ns'],28)

    def test_fitted_duration_injection_keeps_observed_targets(self):
        p=plan()
        model=DagFullHorizonModel(ALIASES,{('a','task','input'):p},duration_provider=lambda n,r:n['duration_ns']*2)
        result=model.forecast(requests()[:1],0,100)
        self.assertEqual(result['finishes']['a']['finish_ns'],40)
        self.assertEqual(p['nodes'][0]['duration_ns'],10)
        self.assertEqual(result['duration_source'],'injected_fitted_provider')

    def test_lifecycle_ready_can_precede_parent_receipt_without_losing_fifo(self):
        class Lifecycle:
            def delays(self,*args):return dict(known=True,parent_receipt_ms=.000005,worker_next_ready_ms=.000003)
        rows=requests()[:1];rows.append(dict(rows[0],request_id='a2',arrival_ns=1))
        result=self.model(allow_profiled_replay=True,lifecycle_provider=Lifecycle()).forecast(rows,0,100)
        self.assertEqual(result['finishes']['a']['handler_end_ns'],20)
        self.assertEqual(result['finishes']['a']['finish_ns'],25)
        self.assertEqual(result['starts']['a2'],23)
        self.assertEqual(result['finishes']['a2']['finish_ns'],48)

    def test_missing_evidence_and_deadlock_are_not_synthetic_finish(self):
        with self.assertRaises(UnknownEvidence):self.model().forecast(requests()[:1],0,100)
        with self.assertRaises(UnknownEvidence):self.model(allow_profiled_replay=True).forecast(requests(),0,100)
        p=plan();p['edges']=[dict(before='x',after='y'),dict(before='y',after='x')]
        with self.assertRaises(UnknownEvidence):
            DagFullHorizonModel(ALIASES,{('a','task','input'):p},allow_profiled_replay=True).forecast(requests()[:1],0,100)


if __name__=='__main__':unittest.main()
