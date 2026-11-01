"""Independent CPU-only support, probability and integration boundary checks."""
import copy
import math
import unittest

from tools.predictor_validation_20261027 import projected_handler_context_v42 as v


def state(age=0, active=('text#0',), previous=None):
    return dict(held_owner='text#0', previous_held_owner=previous, context_age_ns=age,
                handler_active={o:o in active for o in v.OWNERS})


def candidate():
    width=len(v.mark_names()); coefficients=[[0.]*4 for _ in range(width)]
    coefficients[0]=[math.log(p) for p in (.1,.2,.6,.1)]
    signature=v.signature(state())
    mark=dict(feature_names=v.mark_names(),coefficients=coefficients,center=[0.]*width,
        scale=[1.]*width,feature_minimum=[0.]*width,feature_maximum=[100.]*width,
        label_counts={o:1 for o in v.OWNERS},
        state_age_ranges_ns={v.digest([signature,None]):[0,10000]})
    hazard=dict(columns=['intercept','age_bin:2'],coefficients=[math.log(2.),math.log(3.)],
        fixed_age_edges_ns=list(v.EDGES_NS),training_vocabulary=list(v.hazard_tokens(state())),
        state_age_support_ns={signature:[[0,5000],[7000,10000]]})
    model=dict(schema=v.SCHEMA,parameters=copy.deepcopy(v.PARAMETERS),training_only=True,
        fitted=dict(mark=mark,hazard=hazard),sources=[])
    model['model_sha256']=v.digest(model)
    return v.ObservableContextCandidate(model)


class IndependentProjectedHandlerTests(unittest.TestCase):
    def test_fixed_state_integral_splits_age_bins_and_does_not_bridge_support_gap(self):
        c=candidate(); result=c.save_hazard(state(1000),7000)
        self.assertEqual(result['restricted_horizon_ns'],4000)
        expected=2.*1000/1e6+6.*3000/1e6
        self.assertAlmostEqual(result['restricted_candidate_CDF'],-math.expm1(-expected))
        self.assertIn('HORIZON_OUTSIDE_OBSERVED_STATE_RISK',result['OOD_reasons'])
        self.assertIsNone(result['conditional_CDF'])
        self.assertEqual(c.save_hazard(state(5000),0)['restricted_horizon_ns'],0)
        self.assertIn('AGE_OUTSIDE_OBSERVED_STATE_RISK',c.save_hazard(state(5000),0)['OOD_reasons'])
        self.assertNotIn('AGE_OUTSIDE_OBSERVED_STATE_RISK',c.save_hazard(state(7000),0)['OOD_reasons'])

    def test_physical_distribution_retains_held_mass_and_all_four_classes(self):
        result=candidate().next_owner(state(100))
        self.assertEqual(set(result['physical_owner_probabilities']),set(v.OWNERS))
        self.assertAlmostEqual(sum(result['physical_owner_probabilities'].values()),1.)
        self.assertAlmostEqual(result['physical_owner_probabilities']['text#0'],.6)
        self.assertEqual(result['most_likely_physical_owner'],'text#0')
        self.assertFalse(result['physical_context_equals_active_service'])
        self.assertIn('UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION',result['OOD_reasons'])

    def test_history_changes_mark_design_but_not_hazard_and_masks_have_exact_support(self):
        first=state(100); second=state(100,previous='ocr#0')
        self.assertNotEqual(v.mark_vector(first),v.mark_vector(second))
        self.assertEqual(v.hazard_tokens(first),v.hazard_tokens(second))
        changed=state(100,active=('text#0','text#1'))
        self.assertIn('UNSEEN_HOLDER_WORKER_HANDLER_TOKENS',candidate().save_hazard(changed)['OOD_reasons'])
        self.assertIn('UNSEEN_HELD_PREVIOUS_WORKER_HANDLER_STATE',candidate().next_owner(changed)['OOD_reasons'])

    def test_future_end_residual_ready_and_native_phase_cannot_enter_feature_contract(self):
        for key in ('future_END_ns','projection_residual_ns','ready','native_phase'):
            value=dict(state(),**{key:100})
            with self.assertRaises(ValueError):v.mark_vector(value)
            with self.assertRaises(ValueError):v.hazard_tokens(value)
        for bad in (None,0,1):
            value=state();value['handler_active']['ocr#0']=bad
            with self.assertRaises(ValueError):v.validate_features(value)


if __name__=='__main__':unittest.main()
