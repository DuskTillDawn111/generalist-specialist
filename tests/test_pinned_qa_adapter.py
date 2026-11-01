import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.backends.compact_adapters import sha256, verified_qa_adapter
from tests.test_compact_protocol import fixtures


class PinnedQAAdapterTests(unittest.TestCase):
    def create(self, directory):
        root = Path(directory)
        (root/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','task_type':'CAUSAL_LM','bias':'none'}))
        (root/'adapter_model.safetensors').write_bytes(b'local synthetic test bytes, not model weights')
        candidate = {'model':'nvidia/Llama3-ChatQA-1.5-8B', 'revision':'3b98162e3f97550d62aeeb19ea50208f968c678a',
                     'slot':'text','precision':'NF4 + pinned QA adapter'}
        manifest = {'schema':'gsdeploy.local_qa_adapter.v1','model':candidate['model'], 'revision':candidate['revision'],
                    'path':str(root),'files':[{'name':name,'size':(root/name).stat().st_size,'sha256':sha256(root/name)}
                          for name in ['adapter_config.json','adapter_model.safetensors']]}
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest))
        candidate['adapter_sha256'] = sha256(path)
        profile = {'peft_adapter':{'manifest':str(path),'sha256':sha256(path)}}
        return candidate,profile,path

    def test_bound_manifest_and_each_file_are_verified_without_framework_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate,profile,path = self.create(directory)
            self.assertEqual(verified_qa_adapter(profile,candidate)['path'], directory)
            (Path(directory)/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'adapter file'):
                verified_qa_adapter(profile,candidate)

    def test_unbound_derived_identity_and_wrong_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate,profile,path = self.create(directory)
            with self.assertRaisesRegex(ValueError,'missing'):
                verified_qa_adapter({},candidate)
            with self.assertRaisesRegex(ValueError,'identities'):
                verified_qa_adapter(profile,{**candidate,'adapter_sha256':'0'*64})
            with self.assertRaisesRegex(ValueError,'base model'):
                verified_qa_adapter(profile,{**candidate,'revision':'a'*40})
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'manifest'):
                verified_qa_adapter(profile,candidate)

    def test_derived_request_identity_cannot_inherit_base_model_quality(self):
        adapter,_,samples = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            candidate,_,_ = self.create(directory)
            first = adapter.build(candidate,samples['text'])
            second = adapter.build({**candidate,'adapter_sha256':'a'*64},samples['text'])
            self.assertNotEqual(first,second)
            self.assertEqual(first['adapter_sha256'],candidate['adapter_sha256'])

    def test_real_peft_saved_keys_match_an_explicitly_named_active_adapter(self):
        from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
        from transformers import LlamaConfig,LlamaForCausalLM
        config = LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                            num_attention_heads=4,num_key_value_heads=2)
        options = dict(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,target_modules=['q_proj','v_proj'])
        original = get_peft_model(LlamaForCausalLM(config),LoraConfig(**options))
        named = get_peft_model(LlamaForCausalLM(config),LoraConfig(**options),adapter_name='qa_train_only')
        named.set_adapter('qa_train_only')
        loaded = get_peft_model_state_dict(named,adapter_name='qa_train_only')
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory,safe_serialization=True,save_embedding_layers=False)
            with (Path(directory)/'adapter_model.safetensors').open('rb') as handle:
                size = int.from_bytes(handle.read(8),'little')
                header = json.loads(handle.read(size))
            self.assertEqual(set(loaded),set(header)-{'__metadata__'})
            self.assertTrue(all(list(value.shape)==header[name]['shape'] for name,value in loaded.items()))
        self.assertIn('qa_train_only',named.active_adapters)


if __name__ == '__main__':
    unittest.main()
