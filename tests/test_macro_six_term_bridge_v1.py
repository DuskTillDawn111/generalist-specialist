import unittest
from copy import deepcopy
from fractions import Fraction as F

from tests.test_macro_response_queue_reference_v1 import plan,request
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import simulate
from tools.predictor_validation_20261027.macro_six_term_bridge_v1 import whole_score


class SixTermBridgeTests(unittest.TestCase):
    def inputs(self,drain=3_000_000_000,work=2_000_000_000):
        rows=[request(),request('b',arrival=500_000_000)]
        reference=simulate(rows,{('s0','text','known'):plan(work)},
            {('prefill',0,0):F(1)},window_start_ns=0,window_end_ns=1_000_000_000,drain_end_ns=drain)
        scoring=dict(aliases={'s0':dict(family='text',supported_tasks=['text'])},
            workload_rps={'text':2.},latency_reference_ms={'text':1000.},
            window_seconds=1.,startup_amortization_seconds=3600,
            memory_budget_mib=100.,memory_safe_factor=1.2,energy_reference_watts=10.,
            weights={k:1/6 for k in 'QTLPME'})
        quality={'text':{'text':.25}}
        resources=dict(original_window_ns=[0,1_000_000_000],original_drain_end_ns=drain,
            resident_owners=['s0'],startup_energy_joules=3600.,
            startup_through_phase_start_owned_peak_mib=50.,phase_start_through_drain_owned_peak_mib=40.,
            measurement_board_energy_joules=20.,measurement_energy_includes_drain=False,
            memory_safety_factor_already_applied=False,startup_energy_already_amortized=False,
            source_evidence=['explicit CPU resource world'])
        return reference,rows,scoring,quality,resources

    def test_original_scorer_keeps_drain_censored_requests_and_unknown_latency(self):
        ref,rows,scoring,q,res=self.inputs()
        result=whole_score(ref,rows,scoring,q,res)
        self.assertEqual(result['score']['cohort_size'],2)
        self.assertEqual(result['score']['incomplete_ids'],['b'])
        self.assertIsNone(result['score']['terms']['L'])
        self.assertIsNone(result['score']['terms']['P'])
        self.assertEqual(result['score']['terms']['T'],1.)
        self.assertIsNone(result['score']['full_U'])

    def test_original_six_weights_memory_factor_and_startup_amortization_once(self):
        ref,rows,scoring,q,res=self.inputs(work=400_000_000)
        result=whole_score(ref,rows,scoring,q,res);terms=result['score']['terms']
        self.assertEqual(terms['Q'],.25)
        self.assertEqual(terms['T'],0.)
        self.assertAlmostEqual(terms['L'],.4)
        self.assertAlmostEqual(terms['P'],.4)
        self.assertAlmostEqual(terms['M'],.6)
        self.assertAlmostEqual(terms['E'],2.1) # (20 J + 3600 J * 1 s / 3600 s) / 10 J
        self.assertAlmostEqual(result['score']['full_U'],-sum(terms.values())/6)
        self.assertFalse(result['final_gain_eligible'])

    def test_missing_resources_remain_unknown_and_inputs_are_not_mutated(self):
        args=self.inputs(work=400_000_000);before=deepcopy(args)
        result=whole_score(*args[:4])
        self.assertIsNone(result['score']['terms']['M']);self.assertIsNone(result['score']['terms']['E'])
        self.assertEqual(args,before)

    def test_scope_changes_or_duplicate_safety_startup_are_rejected(self):
        args=self.inputs(work=400_000_000)
        for field,value in [('measurement_energy_includes_drain',True),
                ('memory_safety_factor_already_applied',True),('startup_energy_already_amortized',True),
                ('original_drain_end_ns',4_000_000_000),('resident_owners',[])]:
            values=list(deepcopy(args));values[-1][field]=value
            with self.assertRaises(ValueError):whole_score(*values)
        values=list(deepcopy(args));values[2]['startup_amortization_seconds']=60
        with self.assertRaises(ValueError):whole_score(*values)


if __name__=='__main__':unittest.main()
