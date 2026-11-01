import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from gsdeploy.backends.compact import CompactAdapter
from tests.test_compact_protocol import fixtures
from tests import test_compact_native as native_fixtures


class OCR2NativeTests(unittest.TestCase):
    def test_larger_native_omni_candidate_keeps_all_four_underlying_task_inputs(self):
        adapter,study,samples=fixtures()
        original=study['generalists'][0]
        candidate={'slot':'generalist_3','model':'Qwen/Qwen2.5-Omni-7B',
                   'revision':'ae9e1690543ffd5c0221dc27f79834d0294cba00','precision':'NF4 language/BF16 towers; unvalidated'}
        self.assertEqual(adapter.supported_tasks(candidate),('text','image','ocr','asr'))
        for sample in samples.values():
            before=adapter.build(original,sample)
            after=adapter.build(candidate,sample)
            self.assertEqual(after['instruction'],before['instruction'])
            self.assertEqual(after['media'],before['media'])
            self.assertEqual(after['native'],before['native'])
            self.assertNotEqual(after['revision'],before['revision'])

    def candidate(self):
        return {'slot':'ocr','model':'deepseek-ai/DeepSeek-OCR-2',
                'revision':'aaa02f3811945a91062062994c5c4a3f4c0af2b0','precision':'native BF16'}

    def adapter(self):
        original,_,samples=fixtures()
        policy=copy.deepcopy(original.policy)
        policy['tasks']['ocr']['deepseek2_image_preset']='native_dynamic_v1'
        return CompactAdapter(policy,original.classmap),samples

    def test_distinct_native_visual_preset_and_ocr_only_routing_are_required(self):
        original,_,samples=fixtures()
        with self.assertRaisesRegex(ValueError,'explicit pinned native'):
            original.build(self.candidate(),samples['ocr'])
        adapter,_=self.adapter()
        request=adapter.build(self.candidate(),samples['ocr'])
        self.assertEqual(adapter.supported_tasks(self.candidate()),('ocr',))
        self.assertEqual({k:request['native']['options'][k] for k in ['base_size','image_size','crop_mode']},
                         {'base_size':1024,'image_size':768,'crop_mode':True})
        self.assertEqual(request['native']['options']['prompt'],'<image>\nFree OCR.')
        self.assertTrue(request['native']['options']['eval_mode'])
        with self.assertRaises(ValueError):
            adapter.build(self.candidate(),samples['image'])

    def test_actual_verified_image_bytes_survive_original_path_changes(self):
        from PIL import Image
        native_fixtures.NativeBindingTests.setUpClass()
        helper=native_fixtures.NativeBindingTests()
        adapter,samples=self.adapter()
        with helper.no_cuda(),tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.png'
            Image.new('RGB',(3,3),'red').save(path)
            original=path.read_bytes()
            expected=hashlib.sha256(original).hexdigest()
            sample={**samples['ocr'],'input':str(path),'input_sha256':expected}
            executor,_,model=helper.binding(self.candidate(),sample,directory)
            executor.adapter=adapter
            executor.image_loader=None
            infer=model.infer
            def check(tokenizer,**options):
                Image.new('RGB',(3,3),'blue').save(path)
                actual=Path(options['image_file']).read_bytes()
                self.assertEqual(actual,original)
                self.assertNotEqual(options['image_file'],str(path))
                return infer(tokenizer,**options)
            model.infer=check
            result=executor.infer(sample)
            self.assertTrue(result['success'],result)
            self.assertEqual(result['native_input_binding']['sha256'],expected)
            self.assertEqual(result['media_binding']['sha256'],expected)
            with self.assertRaisesRegex(ValueError,'Image bytes changed'):
                executor.infer(sample)


if __name__=='__main__':
    unittest.main()
