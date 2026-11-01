import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import training_campaign_audit_watch as watch
from tools.predictor_validation_20261027.native_flow import receipt


class CampaignAuditWatcherTests(unittest.TestCase):
    def test_atomic_progress_update_preserves_other_fields_and_reuses_run_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            original={'other':{'frozen':[1,2]},'training_v14_h600':{'status':'MAIN_RUNNING','seeds':[1,2]}}
            path.write_text(json.dumps(original))
            entry=dict(run_id='a',measurement_requests=10,quality_count=10)
            watch._update(path,entry);watch._update(path,{**entry,'verified':True})
            value=json.loads(path.read_text())
            self.assertEqual(value['other'],original['other'])
            field=value['training_v14_h600'];self.assertEqual(field['status'],'MAIN_RUNNING')
            self.assertEqual(field['completed_main_runs'],1);self.assertEqual(field['audited_measurement_requests'],10)
            self.assertTrue(field['audited_runs'][0]['verified'])

    def test_pair_preserves_exact_trace_and_quantifies_paired_service_queue_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'run-audits').mkdir();source=root/'source.py';source.write_text('frozen')
            def write(path,value):path.write_text(json.dumps(value));return receipt(path)
            trace=[dict(request_id='a',task='text',pool_id='same',offset_ns=0)]
            plan=dict(seed=1,trace=trace,candidates=[dict(id='c',routes={'a':'x'})])
            jobs=[]
            for name,role in [('primary','primary_training'),('repeat','identical_trace_training_repeat')]:
                path=root/(name+'-plan.json');write(path,plan)
                raw_path=root/(name+'-raw.json')
                delta=0 if name=='primary' else 10_000_000
                write(raw_path,dict(request_records=[dict(request_id='a',model='x',task='text',pool_id='same',
                    scheduled_time_ns=0,observed_worker_start_time_ns=1_000_000+delta,
                    completion_time_ns=22_000_000+2*delta,
                    worker_response={'end_perf_ns':21_000_000+2*delta})]))
                write(root/'run-audits'/(name+'.json'),dict(status='PASS',raw=receipt(raw_path),source=receipt(source)))
                jobs.append(dict(run_id=name,campaign_role=role,candidate_id='c',plan=receipt(path)))
            campaign=root/'campaign.json';write(campaign,dict(main_jobs=jobs))
            with patch.object(watch,'verify_campaign',return_value={'main_jobs':jobs}):
                value=watch.repeat_pair(campaign,jobs[1],root/'pair.json')
                self.assertEqual(value['overall']['service_ms']['paired_repeat_minus_primary']['mean_ms'],10.)
                self.assertEqual(value['overall']['wait_ms']['paired_repeat_minus_primary']['mean_ms'],10.)
                self.assertEqual(value['request_count'],1)
                self.assertEqual(watch.repeat_pair(campaign,jobs[1],root/'pair.json'),value)
                changed=copy.deepcopy(plan);changed['trace'][0]['pool_id']='foreign'
                write(root/'repeat-plan.json',changed);jobs[1]['plan']=receipt(root/'repeat-plan.json')
                with self.assertRaisesRegex(ValueError,'exact same'):
                    watch.repeat_pair(campaign,jobs[1],root/'bad-pair.json')


if __name__=='__main__':unittest.main()
