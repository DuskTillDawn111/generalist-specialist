import math
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cuda_dag_simulator_v6 import DagFullHorizonModel
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import (
    DagPhaseFullHorizonModel,LegacyEqualContextDiagnosticScheduler,
    SchedulingDecision,UnidentifiedPhaseScheduler)


class TestScheduler:
    def __init__(self,decide):self.callback=decide;self.views=[]
    def decide(self,view):self.views.append(view);return self.callback(view)
    def provenance(self):return {'kind':'CPU_FIXTURE_ONLY','physical_accuracy_validated':False}


def inputs():
    aliases={k:dict(family=k,supported_tasks=['text']) for k in ('a','b')}
    plans={(k,'text','x'):dict(nodes=[dict(id='gpu',kind='gpu',duration_ns=d,release_ns=0)],edges=[]) for k,d in [('a',10),('b',5)]}
    rows=[dict(request_id=k,alias=k,task='text',pool_id='x',arrival_ns=0) for k in aliases]
    return aliases,plans,rows


def run(scheduler,*,rows=None,plans=None):
    aliases,p,r=inputs()
    return DagPhaseFullHorizonModel(aliases,plans or p,scheduler=scheduler,
        allow_profiled_replay=True,allow_independent_copy_hypothesis=True,
        allow_no_implicit_default_stream_hypothesis=True,
        phase_provider=lambda n,row:'prefill' if row['alias']=='a' else 'decode',
        max_events=100).forecast(rows or r,0,100,drain_deadline_ns=200)


class PhaseSchedulerTests(unittest.TestCase):
    def test_unidentified_real_scheduler_rejects_even_single_gpu_node(self):
        a,p,r=inputs()
        with self.assertRaisesRegex(UnknownEvidence,'PARAMETERS_UNIDENTIFIED'):
            run(UnidentifiedPhaseScheduler(),rows=r[:1])

    def test_explicit_legacy_adapter_preserves_original_dag_numerics(self):
        aliases,plans,rows=inputs()
        plans['a','text','x']['nodes'].append(dict(id='copy',kind='copy',duration_ns=3,release_ns=0))
        original=DagFullHorizonModel(aliases,plans,allow_profiled_replay=True,
            allow_independent_copy_hypothesis=True,allow_context_time_slice_hypothesis=True,
            allow_no_implicit_default_stream_hypothesis=True).forecast(rows,0,100,drain_deadline_ns=200)
        new=run(LegacyEqualContextDiagnosticScheduler(allow_legacy_equal_context_diagnostic=True),plans=plans)
        # Same-clock closure changes dispatch placement, while preserving the
        # budgeted work and observable timing/resources in this fixture.
        for key in ('cohort','finishes','starts','queue_wait_ns','completed_dag_nodes','queue_diagnostics','resource_transitions'):
            self.assertEqual(new[key],original[key],key)
        self.assertEqual(new['events'],original['events'])
        with self.assertRaises(ValueError):LegacyEqualContextDiagnosticScheduler()

    def test_ownership_boundaries_and_arrival_conserve_remaining_reference_work(self):
        # Synthetic ownership intervals prove mechanics only; no real quantum is fitted.
        def decide(view):
            t=view.clock_ns
            owner='a' if t<7 or t>=12 else 'b'
            boundary=7 if t<7 else 12 if t<12 else None
            return SchedulingDecision({n.key:float(n.key[0]==owner) for n in view.active_gpu_nodes},boundary)
        scheduler=TestScheduler(decide);a,p,r=inputs();r[1]['arrival_ns']=4
        result=run(scheduler,rows=r)
        self.assertEqual(result['finishes']['a']['handler_end_ns'],15)
        self.assertEqual(result['finishes']['b']['handler_end_ns'],12)
        clocks={v.clock_ns for v in scheduler.views}
        self.assertTrue({0,4,7,12,15}<=clocks)
        view7=next(v for v in scheduler.views if v.clock_ns==7)
        self.assertEqual({n.key[0]:n.remaining_reference_work_ns for n in view7.active_gpu_nodes},{'a':3.,'b':5.})
        self.assertEqual({n.phase for n in view7.active_gpu_nodes},{'prefill','decode'})

    def test_completion_and_host_release_preempt_later_scheduler_boundary(self):
        a,p,r=inputs();p['b','text','x']['nodes'][0]['release_ns']=6
        scheduler=TestScheduler(lambda view:SchedulingDecision({n.key:1. for n in view.active_gpu_nodes},20 if view.clock_ns<20 else None))
        result=run(scheduler,plans=p)
        self.assertEqual(result['finishes']['a']['handler_end_ns'],10)
        self.assertEqual(result['finishes']['b']['handler_end_ns'],11)
        self.assertTrue({0,6,10,11} <= {v.clock_ns for v in scheduler.views})

    def test_zero_rates_require_future_integer_transition(self):
        for transition in (None,0,-1,1.5,True):
            with self.subTest(transition=transition),self.assertRaises(UnknownEvidence):
                run(TestScheduler(lambda v:SchedulingDecision({n.key:0 for n in v.active_gpu_nodes},transition)))

    def test_rate_keys_and_finite_zero_one_contract(self):
        callbacks=[lambda v:{},lambda v:{('foreign','gpu'):1}]
        callbacks += [lambda v,x=x:{n.key:x for n in v.active_gpu_nodes} for x in (-1,1.1,math.inf,math.nan,True)]
        for callback in callbacks:
            with self.subTest(callback=callback),self.assertRaises(UnknownEvidence):
                run(TestScheduler(lambda v:SchedulingDecision(callback(v),1)))

    def test_checkpoint_and_pause_are_explicitly_rejected(self):
        aliases,plans,rows=inputs();model=DagPhaseFullHorizonModel(aliases,plans)
        for extra in ({'checkpoint':{}},{'pause_after_event_count':1}):
            with self.subTest(extra=extra),self.assertRaisesRegex(UnknownEvidence,'CHECKPOINT_STATE_NOT_IMPLEMENTED'):
                model.forecast(rows,0,100,**extra)


if __name__=='__main__':unittest.main()
