import copy
import unittest

from tests import test_actual_worker_twoFIFO_reference_v4 as fixtures
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v4 import simulate
from tools.predictor_validation_20261027.actual_worker_world_score_v1 import score_world
from tools.predictor_validation_20261027.native_collect import context_identity


class ExplicitCPUResources:
    def __init__(self,startup=3600,memory_unknown=False):
        self.context=dict(cpu_fixture='no_physical_evidence',execution_sources=[])
        self.execution_context_sha256=context_identity(self.context)
        self.startup_j=startup;self.memory_unknown=memory_unknown

    def power(self,active,*,resident_owners):
        return dict(known=True,watts=100)

    def memory(self,active,*,resident_owners):
        return dict(known=not self.memory_unknown,peak_memory_mib=None if self.memory_unknown else 500 if active else 400)

    def startup(self):
        return dict(joules=self.startup_j,premeasurement_owned_sampled_peak_mib=600)


class ActualWorldScoringTests(unittest.TestCase):
    def inputs(self,**changes):
        kwargs=fixtures.ActualWorkerTwoFIFOTests().inputs(**changes);world=simulate(**kwargs)
        score=dict(aliases={'owner':dict(family='text',supported_tasks=['text'])},
            workload_rps={'text':1e9},latency_reference_ms={'text':1e-6},
            window_seconds=2e-9,startup_amortization_seconds=3600,memory_budget_mib=1000,
            memory_safe_factor=1.2,energy_reference_watts=10,weights={k:1/6 for k in 'QTLPME'})
        return world,kwargs['requests'],score,{'text':{'text':.25}},dict(
            window_start_ns=0,window_end_ns=2,drain_end_ns=150)

    def call(self,args,provider=None):
        world,rows,plan,quality,options=args
        if provider is not None:
            options=dict(options,resource_provider=provider,target_context=provider.context,
                resource_source_evidence=['Explicit synthetic CPU world; not GPU evidence'])
        return score_world(world,rows,plan,quality,**options)

    def test_real_two_FIFO_queue_to_original_reader_score_and_board_clock(self):
        args=self.inputs();before=copy.deepcopy(args);out=self.call(args,ExplicitCPUResources())
        score=out['original_score'];terms=score['terms']
        self.assertAlmostEqual(terms['L'],24);self.assertAlmostEqual(terms['M'],.72)
        self.assertAlmostEqual(terms['E'],10.1)
        self.assertEqual(score['cohort_size'],2);self.assertEqual(score['incomplete_ids'],[])
        self.assertEqual(out['resource_timeline']['total_wall_ns']['numerator'],150)
        self.assertEqual(out['resource_timeline']['total_measurement_overlap_ns']['numerator'],2)
        self.assertEqual(args,before)
        self.assertFalse(out['main_queue_provider_eligible']);self.assertFalse(out['final_gain_eligible'])
        self.assertFalse(out['original_score_bridge']['original_dispatch_qualified'])

    def test_missing_resource_support_keeps_whole_utility_unknown(self):
        for provider,missing in [(None,['M','E']),(ExplicitCPUResources(startup=None),['E']),
                (ExplicitCPUResources(memory_unknown=True),['M'])]:
            score=self.call(self.inputs(),provider)['original_score']
            for term in missing:self.assertIsNone(score['terms'][term])
            self.assertIsNone(score['full_U'])

    def test_full_cohort_and_original_draining_scope_cannot_change_at_integration(self):
        args=self.inputs();args[1].pop()
        with self.assertRaises(ValueError):self.call(args)
        args=self.inputs();args[-1]['drain_end_ns']=151
        with self.assertRaises(ValueError):self.call(args)
        with self.assertRaises(ValueError):self.call(self.inputs(cutoff=5))

    def test_incomplete_at_original_drain_preserves_all_latency_denominators(self):
        args=self.inputs();world,rows,plan,quality,options=args
        kwargs=fixtures.ActualWorkerTwoFIFOTests().inputs();kwargs.update(drain_end_ns=5,cutoff_ns=5)
        world=simulate(**kwargs);options['drain_end_ns']=5
        score=self.call((world,rows,plan,quality,options),ExplicitCPUResources())['original_score']
        self.assertEqual(score['cohort_size'],2);self.assertEqual(set(score['incomplete_ids']),{'r0','r1'})
        self.assertIsNone(score['terms']['L']);self.assertIsNone(score['terms']['P'])
        self.assertIsNone(score['full_U'])

    def test_resource_provider_requires_explicit_execution_and_evidence(self):
        world,rows,plan,quality,options=self.inputs()
        with self.assertRaises(ValueError):
            score_world(world,rows,plan,quality,resource_provider=ExplicitCPUResources(),**options)
        with self.assertRaises(ValueError):
            score_world(world,rows,plan,quality,target_context={},**options)


if __name__=='__main__':unittest.main()
