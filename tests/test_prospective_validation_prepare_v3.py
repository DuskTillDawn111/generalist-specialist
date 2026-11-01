import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.prospective_validation_prepare_v3 import (
    candidate_public,nonstationary_clock,public_trace,select_public)
from tools.predictor_validation_20261027.kernel_profile_campaign_v2 import input_identity
from gsdeploy.routing import Router


class ProspectiveValidationPrepareTests(unittest.TestCase):
    def test_input_id_and_content_both_excluded_without_reading_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for task in ('text','image','asr','ocr'):
                for n in range(3):
                    sample=dict(context='context'+str(n),question='question') if task=='text' else dict(input=str(root/(task+str(n))))
                    if task!='text':Path(sample['input']).write_bytes((task+str(n)).encode())
                    rows.append(dict(task_type=task,pool_id=task+str(n),sample=sample))
            # Renaming a training input cannot make its exact content held out.
            rows.append(dict(rows[0],pool_id='renamed-training-content'))
            training=[dict(task='text',pool_id='text0',exact_input_sha256=input_identity(rows[0]))]
            selected,manifest=select_public(rows,training,per_task=2)
            self.assertNotIn('text0',selected['text']);self.assertNotIn('renamed-training-content',selected['text'])
            self.assertEqual(len(manifest),8)
            self.assertNotIn(training[0]['exact_input_sha256'],[r['exact_input_sha256'] for r in manifest])

    def test_nonstationary_trace_preserves_total_public_intensity_and_all_tasks(self):
        pools={task:[task+'0',task+'1'] for task in ('text','image','asr','ocr')}
        trace=public_trace(dict(text=1.85,image=.05,asr=.05,ocr=.05),pools)
        self.assertEqual(nonstationary_clock(15),30);self.assertEqual(nonstationary_clock(75),60)
        self.assertEqual(nonstationary_clock(90),90)
        self.assertEqual(len(trace),181)
        counts=[sum(30*k*1e9<=row['offset_ns']<30*(k+1)*1e9 and row['task']=='text' for row in trace) for k in range(3)]
        self.assertEqual(counts,[28,111,28]);self.assertEqual({r['task'] for r in trace},set(pools))
        self.assertEqual(len({r['request_id'] for r in trace}),len(trace))

    def test_candidate_arrivals_equal_and_future_router_continues_after_initial(self):
        rates=dict(text=1.85,image=.05,asr=.05,ocr=.05)
        trace=public_trace(rates,{task:[task] for task in rates})
        routes=dict(text=dict(a=.5,b=.5),image=dict(g=1.),asr=dict(g=1.),ocr=dict(o=1.))
        public,actual=candidate_public(dict(routing_weights=routes),trace,rates,dict.fromkeys(('a','b','g','o')))
        router=Router(routes);inputs=public['forecast_inputs']
        initial=sorted((r for queue in inputs['queues'].values() for r in queue),key=lambda r:(r['arrival_ns'],r['task'],r['request_id']))
        for row in initial:self.assertEqual(router.choose(row['task']),row['alias'])
        for row in inputs['future_workload']['declared_arrivals']:
            self.assertEqual(router.choose(row['task']),actual[row['request_id']])
        self.assertEqual(sum(map(len,inputs['queues'].values()))+len(inputs['future_workload']['declared_arrivals']),len(trace))


if __name__=='__main__':unittest.main()
