import unittest
from types import SimpleNamespace
from contextlib import contextmanager
from gsdeploy.backends.compact_memory import actual_generation_scratch_rows,exllama_request_scratch
from gsdeploy.backends.compact_native import capture_generation

class ActualScratchTests(unittest.TestCase):
 def test_native_full_prefill_and_beams_bound_rows_without_changing_input(self):
  inputs=SimpleNamespace(shape=(1,1703,4096))
  kwargs={'inputs_embeds':inputs,'num_beams':3}
  self.assertEqual(actual_generation_scratch_rows(kwargs,16384),5109)
  self.assertIs(kwargs['inputs_embeds'],inputs)
  with self.assertRaisesRegex(ValueError,'no truncation'):actual_generation_scratch_rows(kwargs,1000)
  with self.assertRaises(ValueError):actual_generation_scratch_rows({'inputs_embeds':SimpleNamespace(shape=(2,10,3))},100)
  with self.assertRaises(ValueError):actual_generation_scratch_rows({'inputs_embeds':inputs,'num_beams':True},16384)
  with self.assertRaisesRegex(ValueError,'cached decoding'):actual_generation_scratch_rows({'input_ids':SimpleNamespace(shape=(1,5))},100)
  self.assertEqual(actual_generation_scratch_rows({'input_ids':SimpleNamespace(shape=(1,5)),'use_cache':True},100),5)
 def test_scratch_resizing_surrounds_actual_generate_and_releases_on_exception(self):
  import torch
  events=[];audit={'requests':0,'releases':0};fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
  owner=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
  def generate(**kwargs):
   events.append('generate');return torch.tensor([[7,2]])
  owner.generate=generate
  def around(kwargs):
   return exllama_request_scratch(owner,True,actual_generation_scratch_rows(kwargs,16384),fake,audit=audit,resize=lambda model,rows:events.append(rows))
  generation={'num_beams':3,'do_sample':False,'max_new_tokens':10}
  with capture_generation(owner,SimpleNamespace(eos_token_id=2),generation,prefix_override=0,around=around) as calls:
   result=owner.generate(inputs_embeds=torch.ones((1,13,4)))
  self.assertEqual(result.tolist(),[[7,2]])
  self.assertEqual(calls[0]['observation']['finish_reason'],'eos')
  self.assertLess(events.index(39),events.index('generate'));self.assertGreater(events.index(1),events.index('generate'))
  self.assertEqual(audit['requests'],1);self.assertEqual(audit['releases'],1)
  owner.generate=lambda **kwargs:(_ for _ in ()).throw(RuntimeError('native failure'))
  with self.assertRaises(RuntimeError):
   with capture_generation(owner,None,generation,prefix_override=0,around=around):owner.generate(inputs_embeds=torch.ones((1,13,4)))
  self.assertEqual(audit['requests'],2);self.assertEqual(audit['releases'],2)
