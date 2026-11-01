import copy
from types import SimpleNamespace
import unittest

from gsdeploy.backends.compact import CompactAdapter
from gsdeploy.backends.compact_megrez import complete_megrez_inputs, native_output_budget
from gsdeploy.backends.compact_native import observe, resolve_dtype
from tests.test_compact_protocol import fixtures


class MegrezNativeTests(unittest.TestCase):
    def test_native_deterministic_preset_retains_official_three_beams(self):
        adapter,_,_=fixtures()
        policy=copy.deepcopy(adapter.policy)
        model='Infinigence/Megrez-3B-Omni'
        policy['native_decoding_presets']={model:{task:'megrez_deterministic_chat_v1' for task in ['text','image','ocr','asr']}}
        changed=CompactAdapter(policy,adapter.classmap)
        for task in ['text','image','ocr','asr']:
            observed=changed.generation_policy(task,model)
            self.assertEqual(observed['num_beams'],3)
            self.assertEqual(observed['repetition_penalty'],1.2)
            self.assertFalse(observed['do_sample'])

    def test_requested_dtype_is_never_silently_replaced(self):
        owner = SimpleNamespace(float16=object(), bfloat16=object(), float32=object())
        for name in ['float16','bfloat16','float32']:
            self.assertIs(resolve_dtype(owner,name),getattr(owner,name))
        with self.assertRaisesRegex(ValueError,'Unsupported explicit'):
            resolve_dtype(owner,'float64')

    def test_four_tasks_keep_complete_instruction_and_no_references(self):
        adapter, study, samples = fixtures()
        candidate = {'model':'Infinigence/Megrez-3B-Omni', 'revision':'2a15ac85df31645708eb0bdf769aa1f92eb44a31',
                     'slot':'generalist_3', 'precision':'native BF16, unvalidated'}
        self.assertEqual(adapter.supported_tasks(candidate),('text','image','ocr','asr'))
        for sample in samples.values():
            first = adapter.build(candidate, sample)
            changed = {**sample, 'label':'SECRET', 'answers':['SECRET']}
            self.assertEqual(first, adapter.build(candidate, changed))
            self.assertEqual(first['instruction'], adapter.build(study['generalists'][0],sample)['instruction'])
            self.assertEqual(first['native']['messages'][0]['content']['text'],first['instruction'])
            self.assertFalse(first['input_truncation'])

    def test_context_exhaustion_is_never_an_eos_completion(self):
        with self.assertRaisesRegex(ValueError,'reservation'):
            native_output_budget(1000,4096,8192,'strict_output_reservation_v1')
        self.assertEqual(native_output_budget(1000,4096,8192,'full_input_native_context_v1'),3096)
        for length in [4096,5000]:
            with self.assertRaisesRegex(ValueError,'truncation prohibited'):
                native_output_budget(length,4096,8192,'full_input_native_context_v1')
        self.assertEqual(observe([7,7,7],0,[2],3)['finish_reason'],'length')
        self.assertEqual(observe([7,2],0,[2],3)['finish_reason'],'eos')

    def test_hidden_truncation_removed_and_processors_restored_after_failure(self):
        calls=[]
        class Tokenizer:
            def __call__(self,text,**kwargs):
                calls.append((text,kwargs))
                return text
        tokenizer=Tokenizer()
        extractor=SimpleNamespace()
        class Processor:
            def __call__(self,messages,apply_data_collator):
                self.assert_unbatched=not apply_data_collator
                value=self.tokenizer(messages[0],max_length=None,truncation=True)
                if messages[0]=='error':
                    raise RuntimeError('native failure')
                return {'input_ids':SimpleNamespace(shape=(5000,)), 'text':value}
            def data_collator(self,examples,max_length=4096):
                self.full_collator_length=max_length
                return {**examples[0],'input_ids':SimpleNamespace(shape=(1,max_length))}
        processor=Processor()
        processor.tokenizer,processor.audio_feature_extractor=tokenizer,extractor
        self.assertEqual(complete_megrez_inputs(processor,['full prompt'],' prefix')['text'],'full prompt prefix')
        self.assertTrue(processor.assert_unbatched)
        self.assertEqual(processor.full_collator_length,5000)
        self.assertFalse(calls[-1][1]['truncation'])
        with self.assertRaises(RuntimeError):
            complete_megrez_inputs(processor,['error'])
        self.assertIs(processor.tokenizer,tokenizer)
        self.assertIs(processor.audio_feature_extractor,extractor)

    def test_unrecognized_context_protocol_is_rejected(self):
        adapter,_,samples=fixtures()
        policy=copy.deepcopy(adapter.policy)
        policy['megrez_context_budget']='silently_crop_input'
        changed=CompactAdapter(policy,adapter.classmap)
        with self.assertRaisesRegex(ValueError,'context budget'):
            changed.build({'model':'Infinigence/Megrez-3B-Omni','revision':'a'*40,'precision':'BF16'},samples['text'])
