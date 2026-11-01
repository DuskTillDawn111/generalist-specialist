"""CPU fixtures for public shape support, not measured model evidence."""
import copy
import unittest

from tools.predictor_validation_20261027.chatqa_public_shapes_v3 import (
    hull_support,learn_module_rule,project_features,public_content_sha,query_support)


def evidence(length,positions=None):
    positions=length if positions is None else positions
    return (dict(module='layer.linear',implementation='torch.nn.modules.linear.Linear',
        in_features=4,out_features=2,direct_tensors=[dict(name='weight',dtype='torch.float32',logical_bytes=32)],
        input=dict(shape=[1,positions,4],dtype='torch.float32',element_size_bytes=4,logical_bytes=positions*16),
        output=dict(shape=[1,positions,2],dtype='torch.float32',element_size_bytes=4,logical_bytes=positions*8),
        dense_gemm_flops=positions*16),
        dict(semantic_phase='prefill',input_batch_size=1,input_sequence_length=length))


class PublicShapeTests(unittest.TestCase):
    def test_two_actual_lengths_establish_affine_relation(self):
        rule=learn_module_rule([evidence(10),evidence(20)])
        self.assertEqual(project_features(rule,15)['input_tensor_bytes'],240)
        self.assertEqual(project_features(rule,15)['flops'],240)

    def test_last_position_head_stays_fixed(self):
        rule=learn_module_rule([evidence(10,1),evidence(20,1)])
        self.assertEqual(rule['positions_rule'],'last_position_only')
        self.assertEqual(project_features(rule,100)['input_tensor_bytes'],16)

    def test_one_length_does_not_establish_shape_rule(self):
        with self.assertRaisesRegex(ValueError,'two distinct'):learn_module_rule([evidence(10),evidence(10)])

    def test_conflicting_position_relation_unknown(self):
        with self.assertRaisesRegex(ValueError,'Consistent'):learn_module_rule([evidence(10),evidence(20,1)])

    def test_bytes_mismatch_rejected(self):
        one=evidence(10);one[0]['input']['logical_bytes']+=1
        with self.assertRaisesRegex(ValueError,'byte metadata'):learn_module_rule([one,evidence(20)])

    def test_flops_mismatch_rejected(self):
        one=evidence(10);one[0]['dense_gemm_flops']+=1
        with self.assertRaisesRegex(ValueError,'GEMM'):learn_module_rule([one,evidence(20)])

    def test_batch_extrapolation_disallowed(self):
        rule=learn_module_rule([evidence(10),evidence(20)])
        with self.assertRaises(ValueError):project_features(rule,15,batch_size=2)

    def test_line_hull_requires_joint_affine_relation(self):
        rule=learn_module_rule([evidence(10),evidence(20)]);points=[list(project_features(rule,n).values()) for n in (10,20)]
        self.assertTrue(hull_support(project_features(rule,15),points)['known'])
        invalid=project_features(rule,15);invalid['flops']=200
        self.assertEqual(hull_support(invalid,points)['reason'],'OUTSIDE_TRAINING_AFFINE_SUBSPACE')

    def test_no_length_extrapolation(self):
        rule=learn_module_rule([evidence(10),evidence(20)]);points=[list(project_features(rule,n).values()) for n in (10,20)]
        self.assertEqual(hull_support(project_features(rule,25),points)['reason'],'OUTSIDE_TRAINING_CONVEX_HULL')

    def test_renaming_input_does_not_make_heldout_content(self):
        sample=dict(id='old',task='text',dataset='fixture',question='why?',context='same text',answers=['answer'])
        renamed=copy.deepcopy(sample);renamed.update(id='new',answers=['different reference'])
        self.assertEqual(public_content_sha(sample),public_content_sha(renamed))
        renamed['context']='different public context'
        self.assertNotEqual(public_content_sha(sample),public_content_sha(renamed))

    def test_pool_and_content_must_match_same_training_work(self):
        a=dict(task='text',dataset='fixture',question='q',context='first')
        b=dict(a,context='second');sa,sb=public_content_sha(a),public_content_sha(b)
        rows=[dict(pool_id='a',public_content_sha256=sa),dict(pool_id='b',public_content_sha256=sb)]
        result=query_support(a,'b',{sa:a,sb:b},rows,{sa:dict(input_tokens=10)},
            {},{},{},{'10':dict(status='EXACT_OBSERVED_TRAINING_SHAPE_SIGNATURE_ORDER')})
        self.assertFalse(result['exact_training_workload_template_available'])
        self.assertFalse(result['independent_heldout_input'])
        renamed=query_support(a,'new',{sa:a},rows,{sa:dict(input_tokens=10)},
            {},{},{},{'10':dict(status='EXACT_OBSERVED_TRAINING_SHAPE_SIGNATURE_ORDER')})
        self.assertFalse(renamed['independent_heldout_input'])
        self.assertFalse(renamed['exact_training_workload_template_available'])


if __name__=='__main__':unittest.main()
