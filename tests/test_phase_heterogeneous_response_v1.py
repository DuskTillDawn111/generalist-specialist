import unittest

from tests.test_protocol_timing_slack_v3 import timing_vector
from tools.predictor_validation_20261027 import phase_heterogeneous_response_training_v1 as model
from tools.predictor_validation_20261027 import phase_heterogeneous_protocol_candidate_v1 as candidate
from tools.predictor_validation_20261027 import phase_heterogeneous_protocol_reference_v1 as reference
from tools.predictor_validation_20261027 import joint_request_protocol_transport_candidate_v3 as protocol
from tools.predictor_validation_20261027 import macro_protocol_timing_reference_v3 as original_reference


class PhaseHeterogeneousResponseTests(unittest.TestCase):
    def test_own_family_phase_feature_is_local_and_same_remains_semantic(self):
        state={model.COUNT:3,model.SAME:1}
        features=model.DESIGNS['own_family_phase_count_and_same']
        vector=model.vector('text#1','prefill',state,features)
        self.assertEqual(sum(vector),4)
        self.assertEqual(vector[features.index('count@text:prefill')],3)
        self.assertEqual(vector[features.index('same@text:prefill')],1)
        self.assertEqual(vector[features.index('count@generalist_2:prefill')],0)
        self.assertEqual(vector,model.vector('text#0','prefill',state,features))

    def test_entry_OCR_and_ready_tail_remain_explicit_in_design(self):
        for owner,phases in [('text#0',('entry','prefill','decode','ready_tail')),
                             ('generalist_2#0',('entry','prefill','decode','ready_tail')),
                             ('ocr#0',('entry','ocr_inference','ready_tail'))]:
            for phase in phases:
                self.assertIn(model.category(owner,phase),model.CATEGORIES)
        with self.assertRaises(ValueError):model.category('unseen#0','prefill')
        with self.assertRaises(ValueError):model.vector('ocr#0','entry',{model.COUNT:1,model.SAME:2},())

    def test_free_coefficients_do_not_force_slowdown(self):
        fit=dict(features=list(model.DESIGNS['global_phase_aware']),coefficients=[-.2,0])
        self.assertGreater(candidate.rate('ocr#0','ocr_inference',{model.COUNT:1,model.SAME:0},fit),1)

    def test_progress_uses_piecewise_own_family_phase_only(self):
        fit=dict(features=['count@ocr:ocr_inference'],coefficients=[.5])
        macro=dict(owner='ocr#0',semantic_phase='ocr_inference',left_ns=10,right_ns=30,
            causal_covariate_path=[dict(left_ns=10,right_ns=20,state={model.COUNT:0,model.SAME:0}),
                                  dict(left_ns=20,right_ns=30,state={model.COUNT:2,model.SAME:0})])
        expected=10+5*candidate.rate('ocr#0','ocr_inference',{model.COUNT:2,model.SAME:0},fit)
        self.assertEqual(candidate.progress(macro,25,fit),expected)
        with self.assertRaises(ValueError):candidate.progress(macro,31,fit)

    def test_owner_aware_reference_reduces_exactly_to_original_global_law(self):
        # The same protocol vector and globally equal rates must generate the
        # same paths, not merely the same final rounded latency.
        vector=timing_vector();vector['public_key'][0]='text#0'
        plan,slacks=protocol.to_reference(vector,public_dispatch_ordinal=0)
        key=tuple(vector['public_key']);public=[dict(request_id='fresh',owner=key[0],task=key[1],pool_id=key[2],arrival_ns=0)]
        fit=dict(features=[model.COUNT,model.SAME],coefficients=[.2,-.1])
        kwargs=dict(window_start_ns=0,window_end_ns=1,drain_end_ns=100,
                    producer_fifo=(plan.ready_producer_id,),protocol_slacks={plan:slacks})
        from tools.predictor_validation_20261027.joint_request_response_transport_candidate_v2 import rate_table
        a=original_reference.simulate(public,{key:plan},rate_table(fit['coefficients']),**kwargs)
        b=reference.simulate(public,{key:plan},candidate.rate_table(fit),**kwargs)
        for field in ('starts','RESULT_receipts','RESULT_processed','trace','unfinished_request_ids'):
            self.assertEqual(a[field],b[field])
        self.assertIsNone(b['global_admission_cap'])

    def test_unknown_draw_keeps_full_request_ensemble_unknown(self):
        from tools.predictor_validation_20261027.phase_heterogeneous_generated_audit_v1 import score_worlds
        rows=[dict(heldout_capture='a',evaluation_request_id='r',status='SIMULATED',
            predicted_ns=10,actual_ns=12,owner='text#0') for _ in range(4)]
        rows[2]['status']='UNKNOWN';rows[2]['predicted_ns']=None
        score=score_worlds(rows)
        self.assertEqual((score['offered_requests'],score['complete_ensembles'],score['unknown_ensembles']),(1,0,1))
        self.assertIsNone(score['complete_subset_metrics'])
        self.assertFalse(score['whole_offered_population_metrics_qualified'])


if __name__=='__main__':unittest.main()
