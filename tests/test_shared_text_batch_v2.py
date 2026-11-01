"""Real batch-call/mask/output contract with CPU doubles; never imports torch."""
from contextlib import nullcontext
import copy
import importlib.util
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types
import unittest

import numpy as np

from gsdeploy.backends.shared_text_batch import SharedTextBatchExecutor, load_retained


MODEL = 'nvidia/Llama3-ChatQA-1.5-8B'
ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    bos_token_id = 1
    unk_token_id = 99
    bos_token = '<bos>'
    def convert_tokens_to_ids(self, token):
        return {'<|end_of_text|>':8, '<|eot_id|>':9}.get(token, 99)
    def convert_ids_to_tokens(self, token):
        return {8:'<|end_of_text|>', 9:'<|eot_id|>'}.get(token, '?')
    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return ' '.join(str(t) for t in ids if t not in (1, 8, 9))


class FakeAdapter:
    def __init__(self):
        self.policy = {'do_sample':False, 'num_beams':1, 'max_new_tokens':4, 'top_p':.91}
        self.finishes = []
    def generation_policy(self, task, model):
        return copy.deepcopy(self.policy)
    def build(self, candidate, sample):
        return dict(sample_id=sample['id'], task=sample['task'], model=candidate['model'],
            revision=candidate['revision'], precision='unchanged', protocol_sha256='same',
            input_truncation=False, media=None, generation=copy.deepcopy(self.policy),
            native=dict(text=sample['question'], terminator_tokens=['<|end_of_text|>', '<|eot_id|>']))
    def finish(self, request, sample, raw, *, evaluate_quality, **observation):
        self.finishes.append((copy.deepcopy(request), copy.deepcopy(sample), evaluate_quality, observation))
        # Same fields for both independent and batch paths; fake quality only.
        result = dict(task='text', sample_id=sample['id'], raw_output=raw, prediction=raw,
            success=observation['finish_reason']=='eos', quality_eligible=observation['finish_reason']=='eos',
            protocol_sha256=request['protocol_sha256'], **observation)
        if evaluate_quality:
            result['score'] = {'correct':raw == sample.get('label')}
        return result


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(is_encoder_decoder=False, model_type='llama')
        self.generation_config = SimpleNamespace(do_sample=False, num_beams=1,
            num_return_sequences=1, pad_token_id=None, use_cache=True)
        self.calls = []
        self.fail = False
        self.output_transform = lambda rows:rows
    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.copy(), attention_mask.copy(), copy.deepcopy(kwargs)))
        if self.fail:
            raise RuntimeError('original generation failure')
        budget, pad = kwargs['max_new_tokens'], kwargs['pad_token_id']
        outputs = []
        for row, mask in zip(input_ids, attention_mask):
            tokens = row[mask.astype(bool)].tolist()
            # Independent masks, prompts and completion lengths per row.
            last = tokens[-1]
            completion = [21, 9] if last == 2 else [22, 23, 8] if last == 3 else [24]*budget
            completion = completion[:budget]
            outputs.append(row.tolist()+completion)
        max_width = max(map(len, outputs))
        outputs = [row+[pad]*(max_width-len(row)) for row in outputs]
        return SimpleNamespace(sequences=np.array(self.output_transform(outputs), dtype=np.int64))


class FakeExecutor:
    family = 'chatqa'
    memory_options = {}
    memory_audit = {'requested':{}}
    def __init__(self):
        self.candidate = dict(model=MODEL, revision='a'*40)
        self.model, self.tokenizer, self.adapter = FakeModel(), FakeTokenizer(), FakeAdapter()
        self.preparations = []
    def _context_limit(self):
        return 100
    def prepare(self, request):
        ids = [int(t) for t in request['native']['text'].split()]
        self.preparations.append(copy.deepcopy(request))
        return dict(inputs=dict(input_ids=np.array([ids], dtype=np.int64),
                                attention_mask=np.ones((1,len(ids)), dtype=np.int64)), input_tokens=len(ids))


def sample(rid, ids):
    return dict(id=rid, task='text', question=' '.join(map(str,ids)), context='full context',
                label='21', answers=['21'])


def backend(executor=None):
    executor = executor or FakeExecutor()
    owner = SharedTextBatchExecutor(executor,
        tensor_factory=lambda rows:np.array(rows,dtype=np.int64), inference_context=nullcontext,
        reclaim_cache=lambda:{'reclaimed':False})
    return owner


class SharedTextBatchV2Tests(unittest.TestCase):
    def test_single_real_generate_call_leftpad_mask_prompt_slice_mixed_eos(self):
        owner = backend()
        samples = [sample('short',[1,2]), sample('long',[1,40,41,3]), sample('budget',[1,4])]
        execution = owner.run_batch(samples, evaluate_quality=True)
        self.assertEqual(len(owner.executor.model.calls),1)
        ids, mask, kwargs = owner.executor.model.calls[0]
        np.testing.assert_array_equal(ids,[[8,8,1,2],[1,40,41,3],[8,8,1,4]])
        np.testing.assert_array_equal(mask,[[0,0,1,1],[1,1,1,1],[0,0,1,1]])
        self.assertNotIn('past_key_values',kwargs)
        self.assertNotIn('position_ids',kwargs)
        self.assertEqual(kwargs,dict(do_sample=False,num_beams=1,max_new_tokens=4,top_p=.91,
                                     eos_token_id=[8,9],pad_token_id=8))
        self.assertEqual([r['raw_output'] for r in execution.results],['21','22 23','24 24 24 24'])
        self.assertEqual([r['generated_tokens'] for r in execution.results],[2,3,4])
        self.assertEqual([r['terminal_token_id'] for r in execution.results],[9,8,24])
        self.assertEqual([r['finish_reason'] for r in execution.results],['eos','eos','length'])
        self.assertEqual([r['input_tokens'] for r in execution.results],[2,4,2])
        self.assertTrue(all(call[2] for call in owner.executor.adapter.finishes))
        self.assertEqual(execution.metrics['padding_tokens'],4)
        self.assertEqual(execution.metrics['generate_calls'],1)
        self.assertFalse(execution.metrics['row_early_delivery'])

    def test_batch_output_schema_matches_independent_single_calls(self):
        samples = [sample('a',[1,2]),sample('b',[1,10,3])]
        actual = backend().run_batch(samples, evaluate_quality=True)
        for s,result in zip(samples,actual.results):
            independent = backend().run_batch([s], evaluate_quality=True).results[0]
            for key in ('raw_output','prediction','success','quality_eligible','sample_id','task',
                        'protocol_sha256','finish_reason','generated_tokens','terminal_token_id',
                        'eos_token_ids','score','reference','answers','input_tokens','effective_generation'):
                self.assertEqual(result[key],independent[key])

    def test_prepare_exposes_length_exact_policy_grouping_and_owner(self):
        owner = backend()
        a,b = owner.prepare(sample('a',[1,2])),owner.prepare(sample('b',[1,3,3]))
        self.assertEqual(a.input_tokens,2)
        self.assertEqual(a.compatibility_key,b.compatibility_key)
        with self.assertRaisesRegex(ValueError,'another model owner'):
            backend().execute_prepared([a])
        a.sample['question']='1 77'
        with self.assertRaisesRegex(ValueError,'Prepared input'):
            owner.execute_prepared([a])
        self.assertFalse(owner.executor.model.calls)

    def test_policy_and_inherited_config_changes_refused_before_generate(self):
        owner = backend(); item=owner.prepare(sample('a',[1,2]))
        owner.executor.adapter.policy['max_new_tokens']=3
        with self.assertRaisesRegex(ValueError,'generation policy changed'):
            owner.execute_prepared([item])
        owner=backend(); item=owner.prepare(sample('a',[1,2]))
        owner.executor.model.generation_config.pad_token_id=7
        with self.assertRaisesRegex(ValueError,'generation config changed'):
            owner.execute_prepared([item])

    def test_no_silent_policy_override_for_sampling_beams_or_history_controls(self):
        bad = [('do_sample',True),('num_beams',3),('num_return_sequences',2),
               ('repetition_penalty',1.2),('no_repeat_ngram_size',3),
               ('max_new_tokens',0),('max_length',10),('min_length',2),('stopping_criteria',[])]
        for key,value in bad:
            owner=backend();owner.executor.adapter.policy[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                owner.run_batch([sample('a',[1,2])])
            self.assertFalse(owner.executor.model.calls)

    def test_inherited_history_or_absolute_stopping_controls_refused(self):
        for key,value in [('repetition_penalty',1.2),('no_repeat_ngram_size',2),
                          ('min_length',4),('stop_strings',['stop']),('forced_eos_token_id',9),
                          ('cache_implementation','static')]:
            ex=FakeExecutor();setattr(ex.model.generation_config,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):
                backend(ex).run_batch([sample('a',[1,2])])

    def test_eos_padding_and_prompt_corruption_are_not_hidden(self):
        transforms = [lambda rows:[row+[77] for row in rows],
                      lambda rows:[[77]+row[1:] for row in rows],
                      lambda rows:[row[:2]+[55] for row in rows],
                      lambda rows:rows+rows]
        for transform in transforms:
            owner=backend();owner.executor.model.output_transform=transform
            with self.subTest(transform=transform),self.assertRaises(ValueError):
                owner.run_batch([sample('a',[1,2])])

    def test_batch_state_fresh_for_each_call_and_original_error_not_substituted(self):
        owner=backend();owner.executor.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'original generation failure'):
            owner.run_batch([sample('a',[1,2])])
        self.assertEqual(len(owner.executor.model.calls),1)
        owner.executor.model.fail=False
        owner.run_batch([sample('b',[1,3])])
        self.assertEqual(len(owner.executor.model.calls),2)
        self.assertTrue(all('past_key_values' not in kwargs for _,_,kwargs in owner.executor.model.calls))

    def test_owner_lock_rejects_concurrent_entry(self):
        owner=backend();owner._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'already active'):
                owner.prepare(sample('a',[1,2]))
        finally:
            owner._lock.release()

    def test_context_overflow_bos_mask_task_and_memory_hooks_refused(self):
        owner=backend();owner.executor._context_limit=lambda:5
        with self.assertRaisesRegex(ValueError,'budget exceeds context'):
            owner.run_batch([sample('a',[1,40,2])])
        for s in (sample('a',[77,2]),dict(sample('a',[1,2]),task='image')):
            with self.assertRaises(ValueError):backend().run_batch([s])
        ex=FakeExecutor();ex.memory_options={'last_position_logits':True}
        with self.assertRaisesRegex(ValueError,'memory hooks'):
            backend(ex)

    def test_no_implicit_model_load_and_help_cpu_safe(self):
        with self.assertRaisesRegex(RuntimeError,'execute=True'):
            load_retained('/missing','/missing')
        result = subprocess.run([sys.executable,str(ROOT/'scripts/shared_text_batch_worker.py'),'--help'],
                                text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--execute',result.stdout)
        result = subprocess.run([sys.executable,'-c',
            'import sys; import gsdeploy.backends.shared_text_batch; assert "torch" not in sys.modules'],
            text=True,capture_output=True,cwd=ROOT)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_wire_one_batch_result_identity_and_error_stops_no_retry(self):
        spec=importlib.util.spec_from_file_location('shared_batch_worker',ROOT/'scripts/shared_text_batch_worker.py')
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        events=[]
        message=dict(op='batch',batch_id='B',requests=[dict(request_id='r1',sample=sample('a',[1,2]),submit_ns=1),
                                                      dict(request_id='r2',sample=sample('b',[1,3]))])
        worker.serve(backend(),[json.dumps(message),json.dumps(dict(op='shutdown'))],sink=lambda k,v:events.append((k,v)))
        self.assertEqual([k for k,v in events],['READY','START','BATCH_RESULT','BYE'])
        result=events[2][1]
        self.assertEqual(result['request_ids'],['r1','r2'])
        self.assertEqual([r['request_id'] for r in result['results']],['r1','r2'])
        self.assertEqual(result['results'][0]['end_ns'],result['results'][1]['end_ns'])
        owner=backend();owner.executor.model.fail=True;events=[]
        with self.assertRaisesRegex(RuntimeError,'original generation failure'):
            worker.serve(owner,[json.dumps(message),json.dumps(message)],sink=lambda k,v:events.append((k,v)))
        self.assertEqual([k for k,v in events],['READY','START','ERROR'])
        self.assertFalse(events[-1][1]['retry_permitted'])
        self.assertEqual(len(owner.executor.model.calls),1)

    def test_real_frozen_build_prepare_finish_schema_and_reference_isolation(self):
        source = ROOT/'artifacts/retained-replica-profiles-20261021/source-qa/gsdeploy'
        if not source.exists():
            self.skipTest('Frozen retained-source fixture not installed')
        name = '_gsdeploy_batch_cpu_schema_test'
        package = types.ModuleType(name);package.__path__=[str(source)];package.__package__=name
        sys.modules[name]=package
        native=importlib.import_module(name+'.backends.compact_native')
        policy=json.loads((ROOT/'configs/compact_protocol.json').read_text())
        policy['tasks']['text']['chatqa_prompt_format']='native_system_v2'
        policy['tasks']['text']['no_answer_normalization']='optional_brackets_terminal_punctuation_v2'
        classes={str(i):[f'n{i:08d}',f'class_{i}'] for i in range(1000)}
        adapter=native.CompactAdapter(policy,classes)
        candidate=dict(slot='text',model=MODEL,revision='a'*40,precision='fixture')
        tokenizer=FakeTokenizer()
        class Processor:
            def __call__(self,text,**options):
                self.last=(text,copy.deepcopy(options))
                assert text.startswith('<bos>System: ')
                # CPU tokenization double; real native prepare calls this and
                # verifies BOS/context exactly as serving does.
                ids=[1]+[30+(ord(c)%20) for c in text[5:]]+[2]
                return dict(input_ids=np.array([ids],dtype=np.int64),
                            attention_mask=np.ones((1,len(ids)),dtype=np.int64))
        model=FakeModel();model.config.max_position_embeddings=20000
        executor=native.NativeExecutor(adapter,candidate,model,Processor(),torch=SimpleNamespace(),
                                      device='cpu',dtype='float32',tokenizer=tokenizer)
        owner=backend(executor)
        s=dict(id='real',task='text',dataset=policy['tasks']['text']['dataset'],
               question='Which city?',context='The city is Paris.',label='SECRET',answers=['SECRET'])
        first=owner.prepare(s)
        changed=copy.deepcopy(s);changed.update(label='OTHER SECRET',answers=['OTHER SECRET'])
        second=owner.prepare(changed)
        self.assertEqual(first.request,second.request)
        self.assertEqual(first.input_ids,second.input_ids)
        self.assertNotIn('SECRET',json.dumps(first.request))
        execution=owner.execute_prepared([first,second],evaluate_quality=True)
        eos=dict(finish_reason='eos',generated_tokens=2,terminal_token_id=9,eos_token_ids=[8,9])
        expected=adapter.finish(first.request,s,'21',evaluate_quality=True,**eos)
        for key,value in expected.items():
            self.assertEqual(execution.results[0][key],value)
        self.assertEqual(execution.results[0]['effective_generation']['max_new_tokens'],policy['tasks']['text']['max_new_tokens'])
        self.assertEqual(len(model.calls),1)


if __name__ == '__main__':unittest.main()
