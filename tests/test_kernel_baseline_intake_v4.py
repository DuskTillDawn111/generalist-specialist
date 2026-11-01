import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.kernel_baseline_intake_v4 import explicit_configuration,isolation_receipt
from tools.predictor_validation_20261027.native_collect import context_identity


class StartupAwareIntakeTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(path='fixture',sha256='a'*64)
        self.context=dict(execution=dict(batch_size=1),worker_bindings={'generalist_2':dict(execution=dict(dtype='float16'))},
                          execution_sources=[dict(original_path='fixture',frozen=self.source)],sources=[self.source])
        self.record=dict(id='affine0',module='model.q',implementation='auto_gptq.nn_modules.qlinear.qlinear_exllama.QuantLinear',
            direct_tensors=[dict(name='qweight',dtype='torch.int32',logical_bytes=16)],
            input=dict(dtype='torch.float16',shape=[3,100,8]),output=dict(dtype='torch.float16',shape=[3,100,8]))
        self.semantic=dict(id='prefill0',input_batch_size=3)
        self.signature=dict(name='observed',grid=[1,1,1],block=[32,1,1])

    def configuration(self,task='image',beams=3):
        return explicit_configuration(self.record,self.semantic,'generalist_2',0,self.signature,
            self.source,self.source,self.source,self.context,task=task,effective_generation=dict(num_beams=beams))

    def test_image_semantic_batch_three_does_not_rewrite_request_context(self):
        before=copy.deepcopy(self.context);identity=context_identity(self.context)
        result=self.configuration()
        self.assertEqual(result['batch_size'],3)
        self.assertEqual(result['explicit_batch_semantics']['user_request_batch_size'],1)
        self.assertEqual(result['explicit_batch_semantics']['effective_num_beams'],3)
        self.assertFalse(result['explicit_batch_semantics']['user_requests_in_joint_batch'])
        self.assertEqual(self.context,before);self.assertEqual(context_identity(self.context),identity)

    def test_shape_or_beam_name_cannot_invent_semantic_batch(self):
        self.semantic.pop('input_batch_size')
        with self.assertRaisesRegex(ValueError,'semantic forward batch'):self.configuration()

    def test_inconsistent_beam_and_unknown_expansion_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unsupported semantic batch'):self.configuration(beams=2)
        with self.assertRaisesRegex(ValueError,'Unsupported semantic batch'):self.configuration(task='asr')
        self.record['input']['shape'][0]=1
        with self.assertRaisesRegex(ValueError,'leading dimension'):self.configuration()

    def test_execution_key_preserves_semantic_batch_definition(self):
        beam_key=self.configuration()['execution_key']
        self.semantic['input_batch_size']=1
        self.record['input']['shape'][0]=1;self.record['output']['shape'][0]=1
        no_beam_key=self.configuration(beams=1)['execution_key']
        self.assertNotEqual(beam_key,no_beam_key)

    def isolation_fixture(self):
        job=dict(id='j',inputs={'generalist_2#0':'i'},tasks={'generalist_2#0':'image'},
            context='G2-solo-image',split='training',start_offsets_ms={'generalist_2#0':0})
        row=dict(profile_job_id='j',request_id='r',model='generalist_2#0',task='image',pool_id='i',
            split='training',status='completed',worker_response=dict(start_perf_ns=10,end_perf_ns=20))
        return dict(jobs=[job]),dict(request_records=[row],board_samples=[dict(foreign_pids=[])])

    def test_modality_solo_label_is_accepted_with_actual_isolation(self):
        plan,raw=self.isolation_fixture()
        with tempfile.TemporaryDirectory() as temp:
            ref=isolation_receipt(dict(raw=self.source,plan=self.source),plan,raw,Path(temp))
            self.assertTrue(json.loads(Path(ref['path']).read_text())['actual_request_intervals_disjoint'])

    def test_solo_label_does_not_override_priming_overlap(self):
        plan,raw=self.isolation_fixture()
        raw['priming_request_records']=[dict(request_id='prime',model='text#0',worker_response=dict(start_perf_ns=15,end_perf_ns=25))]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Concurrent actual'):
                isolation_receipt(dict(raw=self.source,plan=self.source),plan,raw,Path(temp))

    def test_wrong_task_or_second_alias_cannot_be_solo(self):
        plan,raw=self.isolation_fixture();plan['jobs'][0]['context']='G2-solo-asr'
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'one-alias'):
                isolation_receipt(dict(raw=self.source,plan=self.source),plan,raw,Path(temp))
        plan,raw=self.isolation_fixture();plan['jobs'][0]['inputs']['text#0']='t'
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'one-alias'):
                isolation_receipt(dict(raw=self.source,plan=self.source),plan,raw,Path(temp))


if __name__=='__main__':unittest.main()
