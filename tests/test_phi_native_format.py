import copy
import unittest

from gsdeploy.backends.compact import CompactAdapter
from tests.test_compact_protocol import fixtures


class PhiNativeFormatTests(unittest.TestCase):
    def test_compact_format_keeps_full_content_and_uses_the_pinned_role_layout(self):
        original,study,samples = fixtures()
        candidate = study['generalists'][2]
        policy = copy.deepcopy(original.policy)
        policy['phi_prompt_format'] = 'native_compact_v1'
        adapter = CompactAdapter(policy,original.classmap)
        for task,sample in samples.items():
            request = adapter.build(candidate,sample)
            tag = '<|audio_1|>' if task == 'asr' else '<|image_1|>' if task != 'text' else ''
            expected = '<|user|>'+tag+request['instruction']+'<|end|><|assistant|>'
            self.assertEqual(request['native']['prompt'],expected)
            before = original.build(candidate,sample)
            self.assertEqual(request['media'],before['media'])
            self.assertEqual(request['generation'],before['generation'])
            self.assertEqual(request['instruction'],before['instruction'])
            self.assertNotEqual(request['native']['prompt'],before['native']['prompt'])
        policy['phi_prompt_format'] = 'unsupported'
        with self.assertRaisesRegex(ValueError,'Phi prompt format'):
            CompactAdapter(policy,original.classmap)


if __name__ == '__main__':
    unittest.main()
