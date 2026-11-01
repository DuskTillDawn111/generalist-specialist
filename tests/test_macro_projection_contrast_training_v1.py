import copy
import math
import unittest

import numpy as np

from tests.test_macro_joint_response_fit_v1 import document
from tests.test_static_language_work_v1 import binding, begin
from tools.predictor_validation_20261027 import macro_projection_contrast_training_v1 as module
from tools.predictor_validation_20261027.static_language_work_v1 import features_from_visible_begin


class ProjectionContrastTests(unittest.TestCase):
    def fixture(self):
        macro=dict(owner='own',request_id='r',macro_id='prefill_0000',kind='language_forward_macro')
        peer=dict(owner='peer',request_id='p',phase_id='prefill_0000')
        piece=dict(left_ns=100,state={module.COUNT:1,'peer_macros':[peer]})
        own=dict(partial_decoder_projection_intensity=10.,available_at_processing_ns=100)
        other=dict(partial_decoder_projection_intensity=1.,available_at_processing_ns=50)
        return macro,piece,{('own','r','prefill_0000'):own,('peer','p','prefill_0000'):other}

    def test_pair_definition_denominators_similarity_alias(self):
        macro,piece,begins=self.fixture();state=module.pair_state(macro,piece,begins)
        self.assertEqual((state['peer_denominator_N'],state['known_pair_denominator_K'],state['unknown_pair_denominator_U']),(1,1,0))
        self.assertAlmostEqual(state[module.CONTRAST],math.log(10)/(1+math.log(10)))
        self.assertAlmostEqual(state[module.CONTRAST]+state['known_partial_decoder_projection_similarity_sum'],1.)
        self.assertIsNotNone(state['all_peer_contrast_sum'])

    def test_unknown_own_with_peers_and_without_peers_not_resource_zero(self):
        macro,piece,begins=self.fixture();macro['kind']='host_entry_response'
        state=module.pair_state(macro,piece,begins)
        self.assertFalse(state['own_supported']);self.assertEqual(state['unknown_pair_denominator_U'],1)
        self.assertIsNone(state['all_peer_contrast_sum']);self.assertEqual(state[module.CONTRAST],0)
        piece['state']={module.COUNT:0,'peer_macros':[]};state=module.pair_state(macro,piece,begins)
        self.assertEqual(state['peer_denominator_N'],0);self.assertFalse(state['own_supported'])
        self.assertEqual(state[module.OWN_UNKNOWN],1)

    def test_unprocessed_BEGIN_and_unknown_peer_remain_unknown(self):
        macro,piece,begins=self.fixture();piece['left_ns']=99
        self.assertEqual(module.pair_state(macro,piece,begins)['known_pair_denominator_K'],0)
        piece['left_ns']=100;piece['state']['peer_macros'][0]['phase_id']='native_ocr'
        state=module.pair_state(macro,piece,begins)
        self.assertTrue(state['own_supported']);self.assertEqual(state['unknown_pair_denominator_U'],1)
        self.assertIsNone(state['pairs'][0]['partial_decoder_projection_contrast'])
        self.assertIsNone(state['all_peer_mean_contrast'])

    def test_future_BEGIN_mutation_does_not_change_processed_prefix(self):
        macro,piece,begins=self.fixture()
        begins[('peer','p','prefill_0000')]['available_at_processing_ns']=101
        before=module.pair_state(macro,piece,begins)
        begins[('peer','p','prefill_0000')]['partial_decoder_projection_intensity']=10000.
        self.assertEqual(module.pair_state(macro,piece,begins),before)

    def test_phase_shape_batch_dynamic_partial_scope(self):
        event=begin();event.update(owner='own',request_ordinal=0)
        x=features_from_visible_begin(binding(),event,parent_now_ns=20)
        first=module.descriptor(x,event,'capture')
        event.update(input_batch_size=1,input_sequence_length=1,semantic_phase='decode',segment_id='decode_0001')
        x=features_from_visible_begin(binding(),event,parent_now_ns=20)
        second=module.descriptor(x,event,'capture')
        self.assertNotEqual(first['partial_decoder_projection_intensity'],second['partial_decoder_projection_intensity'])
        self.assertEqual(second['attention_KV_and_quantized_runtime_costs'],'UNKNOWN_EXCLUDED_FROM_PARTIAL_PROXY')
        self.assertNotEqual(first['phase'],second['phase'])

    def records(self,doc):
        records=[]
        for request in doc['requests']:
            macro=request['macros'][0];piece=macro['causal_covariate_path'][0]
            n=piece['state'][module.COUNT]
            records.append(dict(capture_sha256='capture',request_id=macro['request_id'],macro_id=macro['macro_id'],
                path=[dict(duration_ns=macro['response_wall_ns'],**{module.COUNT:n,module.PHASE:0},
                    projection_state={module.CONTRAST:n/2,module.UNKNOWN:0,module.OWN_UNKNOWN:0})]))
        return records

    def test_new_projection_feature_is_actually_fitted_with_joint_nuisance(self):
        doc=document(.3);rows=module.prepared([doc],self.records(doc),(module.CONTRAST,))
        fit,beta,mu,var=module.fit(rows,(module.CONTRAST,))
        self.assertAlmostEqual(beta[0],.6,places=5)
        self.assertEqual(fit['nuisance_adjusted_derivative_rank'],1)
        self.assertTrue(fit['numerical_qualification'])

    def test_no_row_drop_and_nuisance_absorbs_constant_structure(self):
        doc=document();records=self.records(doc)
        for request,record in zip(doc['requests'],records):
            request['macros'][0]['kind']='ocr_macro'
            record['path'][0]['projection_state'][module.OWN_UNKNOWN]=1
        rows=module.prepared([doc],records,(module.CONTRAST,module.OWN_UNKNOWN))
        self.assertEqual(len(rows),6)
        for row in rows:row['group']=(row['matrix'][0,0],)
        derivative=np.asarray([-r['matrix'][0] for r in rows])
        centered=derivative.copy()
        for group in {r['group'] for r in rows}:
            indices=[i for i,r in enumerate(rows) if r['group']==group]
            centered[indices]-=np.mean(derivative[indices],axis=0)
        self.assertEqual(np.linalg.matrix_rank(centered),0)
        with self.assertRaisesRegex(ValueError,'No unsupported macro may be dropped'):
            module.prepared([doc],records[:-1],())


if __name__=='__main__':unittest.main()
