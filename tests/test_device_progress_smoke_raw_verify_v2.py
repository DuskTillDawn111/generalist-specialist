"""Explicit no-answer is a protocol outcome, never a fabricated output."""
import copy
import json
from pathlib import Path
import unittest

from tests.test_device_progress_smoke_raw_verify_v1 import fixture
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v2 as v
from tools.predictor_validation_20261027.smoke_verifier_v3 import load_cpu_adapter,validate_output

POLICY='optional_brackets_terminal_punctuation_v2'


class ProtocolOutputTests(unittest.TestCase):
    def no_answer(self,text='<No Answer>'):
        raw=fixture();row=raw['request_records'][0];row['task']='text'
        row['worker_response'].update(raw_output=text,prediction='')
        return raw,row

    def test_only_exact_enabled_text_noanswer_may_normalize_empty(self):
        for text in ('<No Answer>',' <No Answer>','no answer.','< no answer >!'):
            raw,row=self.no_answer(text);before=copy.deepcopy(raw)
            self.assertTrue(v.valid_prediction(row,POLICY))
            self.assertEqual(v.validate_raw(raw,{row['model']:POLICY})['issues'],[])
            self.assertEqual(raw,before)
        for text in ('','   ','some actual answer','<No Answer> extra'):
            _,row=self.no_answer(text);self.assertFalse(v.valid_prediction(row,POLICY))
        _,row=self.no_answer();self.assertFalse(v.valid_prediction(row,None))
        row['task']='ocr';self.assertFalse(v.valid_prediction(row,POLICY))
        for text in ('<No Answer>',' no answer> '):
            _,row=self.no_answer(text);self.assertTrue(v.valid_prediction(row,v.BASE_NO_ANSWER))
        for text in ('no answer.','< no answer >!','<No Answer> extra',''):
            _,row=self.no_answer(text);self.assertFalse(v.valid_prediction(row,v.BASE_NO_ANSWER))

    def test_empty_unqualified_or_failed_output_still_fails_probe_review(self):
        raw,row=self.no_answer()
        self.assertEqual(v.validate_raw(raw)['status'],'FAIL')
        row['worker_response']['success']=False
        self.assertEqual(v.validate_raw(raw,{row['model']:POLICY})['status'],'FAIL')
        raw,row=self.no_answer('');self.assertEqual(v.validate_raw(raw,{row['model']:POLICY})['status'],'FAIL')

    def test_exact_original_CPU_adapter_accepts_noanswer_smoke_rejects_blank(self):
        root=Path(__file__).resolve().parents[1]
        context=json.loads((root/'artifacts/predictor-device-progress-paired-v4-20261028-training02/smoke/natural/context.json').read_text())
        binding=context['worker_bindings']['text'];spec=json.loads(Path(binding['spec']).read_text());adapter=load_cpu_adapter(binding,spec)
        pool=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        item=next(json.loads(line) for line in Path(pool).read_text().splitlines() if json.loads(line)['pool_id']=='tydiqa-new-native-e0ff7b578f9c742a6697469a')
        candidate=spec.get('task_candidates',{}).get('text',spec['candidate']);request=adapter.build(candidate,item['sample'])
        self.assertEqual(adapter.policy['tasks']['text']['no_answer_normalization'],POLICY)
        raw=json.loads((root/'artifacts/predictor-device-progress-paired-v4-20261028-training02/smoke/natural/raw.json').read_text())
        for family in ('text','generalist_2'):
            binding=context['worker_bindings'][family];spec=json.loads(Path(binding['spec']).read_text());adapter=load_cpu_adapter(binding,spec)
            candidate=spec.get('task_candidates',{}).get('text',spec['candidate']);request=adapter.build(candidate,item['sample'])
            response=next(row['worker_response'] for row in raw['request_records'] if row['model']==family+'#0')
            observation={key:response[key] for key in ('finish_reason','generated_tokens','terminal_token_id','eos_token_ids')}
            valid=adapter.finish(request,item['sample'],' <No Answer>',evaluate_quality=False,**observation)
            self.assertTrue(valid['success']);self.assertEqual(valid['prediction'],'')
            row=copy.deepcopy(next(row for row in raw['request_records'] if row['model']==family+'#0'))
            row['worker_response']['raw_output']='   '
            self.assertEqual(validate_output(row,item,spec,adapter),['TASK_OUTPUT_STRUCTURE_INVALID:'+row['request_id']])


if __name__=='__main__':unittest.main()
