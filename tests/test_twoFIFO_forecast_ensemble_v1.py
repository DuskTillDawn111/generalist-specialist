import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from tests.test_actual_worker_world_forecast_v4 import inputs
from tests.test_actual_worker_twoFIFO_reference_v5_2 import paired_inputs
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as m
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C
from tests.test_actual_worker_twoFIFO_reference_v4 import decode


def arguments():
    args=inputs(paired_inputs());args.pop('cutoff_ns',None)
    request=args.pop('requests');draw={k:args.pop(k) for k in m.INPUT_FIELDS}
    return request,[dict(draw_id='draw0',block_id='training-block-1',weight=F(1),inputs=draw)],args


class EnsembleTests(unittest.TestCase):
    def test_real_generated_complete_cohort_has_known_timing_unknown_resource_utility(self):
        rows,draws,args=arguments();out=decode(m.evaluate(rows,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],0)
        self.assertEqual(out['six_terms']['P']['known_probability_mass'],1)
        self.assertAlmostEqual(out['six_terms']['P']['expected_value'],28.5)
        self.assertEqual(out['expected_world_complete_cohort_P95_ns']['expected_value'],28.5)
        self.assertEqual(out['utility']['unknown_probability_mass'],1)
        self.assertIsNone(out['utility']['expected_value'])
        self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds']==[0,0] for r in out['fixed_drain_risk']))
        self.assertFalse(out['main_queue_provider_eligible'])

    def test_nonidentified_world_keeps_half_probability_and_all_requests(self):
        rows,draws,args=arguments();draws[0]['weight']=F(1,2)
        unknown=copy.deepcopy(draws[0]);unknown.update(draw_id='unknown',weight=F(1,2))
        unknown['inputs']['transport_draws'][('owner','text','input',0)]=C(0,0,5,10,9)
        draws.append(unknown);before=copy.deepcopy((rows,draws,args))
        out=decode(m.evaluate(rows,draws,**args));self.assertEqual((rows,draws,args),before)
        self.assertEqual(out['world_unknown_probability_mass'],F(1,2))
        self.assertEqual(out['six_terms']['P']['known_probability_mass'],F(1,2))
        self.assertIsNone(out['six_terms']['P']['expected_value'])
        self.assertAlmostEqual(out['six_terms']['P']['conditional_on_known_mean'],28.5)
        self.assertEqual(out['draws'][1]['all_public_request_ids_retained'],['r0','r1'])
        self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds']==[0,F(1,2)] for r in out['fixed_drain_risk']))
        self.assertFalse(out['unknown_worlds_renormalized'])

    def test_all_unknown_cannot_bypass_public_or_original_score_contract(self):
        rows,draws,args=arguments();draws[0]['inputs']['transport_draws'][('owner','text','input',0)]=C(0,0,5,10,9)
        for mutation in ('weights','window','arrival','offeredweight'):
            a=copy.deepcopy(args);d=copy.deepcopy(draws);r=copy.deepcopy(rows)
            if mutation=='weights':a['scoring_plan']['weights']['P']=2
            elif mutation=='window':a['drain_end_ns']=1
            elif mutation=='arrival':r[0]['arrival_ns']=-1
            else:d[0]['weight']=F(1,2)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.evaluate(r,d,**a)

    def test_known_world_incomplete_at_fixed_drain_is_miss_not_unknown_or_completed_P95(self):
        rows,draws,args=arguments();args['drain_end_ns']=5
        out=decode(m.evaluate(rows,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],0)
        self.assertIsNone(out['six_terms']['P']['expected_value'])
        self.assertEqual(out['expected_world_complete_cohort_P95_ns']['unknown_probability_mass'],1)
        self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds']==[1,1] for r in out['fixed_drain_risk']))
        self.assertEqual(out['offered_requests'],2)

    def test_programming_errors_propagate_not_disguised_as_scientific_unknown(self):
        rows,draws,args=arguments()
        with patch.object(m,'forecast',side_effect=ValueError('invalid contract')):
            with self.assertRaisesRegex(ValueError,'invalid contract'):m.evaluate(rows,draws,**args)

    def test_paired_delta_and_sign_bounds_do_not_renormalize_unknown_utility(self):
        rows,draws,args=arguments();base=m.evaluate(rows,draws,**args);candidate=copy.deepcopy(base)
        # Synthetic known utilities isolate comparison math from resource fitting.
        base['draws'][0]['forecast']['scoring']['original_score']['full_U']=-2
        candidate['draws'][0]['forecast']['scoring']['original_score']['full_U']=-1
        out=decode(m.compare(base,candidate));self.assertEqual(out['expected_delta_U'],1)
        self.assertEqual(out['positive_delta_probability_bounds'],[1,1])
        candidate['draws'][0]['forecast']['scoring']['original_score']['full_U']=None
        out=decode(m.compare(base,candidate));self.assertIsNone(out['expected_delta_U'])
        self.assertEqual(out['positive_delta_probability_bounds'],[0,1]);self.assertIsNone(out['candidate_better'])
        candidate['draws'][0]['draw_id']='unpaired'
        with self.assertRaises(ValueError):m.compare(base,candidate)

    def test_different_score_scale_or_quality_cannot_be_compared(self):
        rows,draws,args=arguments();base=m.evaluate(rows,draws,**args)
        for field in ('latency_reference_ms','memory_budget_mib','energy_reference_watts','quality'):
            changed=copy.deepcopy(args)
            if field=='quality':changed['quality']={}
            elif field=='latency_reference_ms':
                changed['scoring_plan'][field]={k:v*2 for k,v in changed['scoring_plan'][field].items()}
            else:changed['scoring_plan'][field]*=2
            with self.subTest(field=field):
                candidate=m.evaluate(rows,draws,**changed)
                with self.assertRaisesRegex(ValueError,'normalization'):m.compare(base,candidate)

    def test_missing_compilation_evidence_retains_draw_not_empty_plan(self):
        rows,draws,args=arguments();draws[0]['weight']=F(1,3)
        draws.append(dict(draw_id='missing',block_id='training-block-2',weight=F(2,3),
            inputs=m.UnknownForecastInputs('public input lacks training donor',{'pool_id':'unseen'})))
        out=decode(m.evaluate(rows,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],F(2,3))
        self.assertEqual(out['six_terms']['P']['known_probability_mass'],F(1,3))
        self.assertEqual(out['draws'][1]['all_public_request_ids_retained'],['r0','r1'])
        self.assertEqual(out['draws'][1]['evidence'],{'pool_id':'unseen'})
        with self.assertRaisesRegex(ValueError,'shared state'):
            m.evaluate(rows,[draws[1]|{'weight':F(1)}],global_admission_cap=1,**args)
        args['scoring_plan']['weights']['P']=2
        with self.assertRaises(ValueError):m.evaluate(rows,[draws[1]|{'weight':F(1)}],**args)

    def test_serialized_compare_rechecks_probability_and_denominator(self):
        rows,draws,args=arguments();base=m.evaluate(rows,draws,**args)
        for kind in ('mass','offered','drawoffered','status'):
            bad=copy.deepcopy(base)
            if kind=='mass':bad['draws'][0]['weight']=2
            elif kind=='offered':bad['offered_requests']=1
            elif kind=='drawoffered':bad['draws'][0]['offered_requests']=1
            else:bad['draws'][0]['status']='UNKNOWN_WORLD'
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.compare(bad,copy.deepcopy(bad))


if __name__=='__main__':unittest.main()
