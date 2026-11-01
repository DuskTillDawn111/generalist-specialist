import unittest
from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_host_diagnostic_v7 import jobs_for_capture,simulate_job


class TrainingComponentDiagnosticTests(unittest.TestCase):
    def fixture(self):
        aliases={'a':dict(family='text',supported_tasks=['text'])}
        p=dict(alias='a',family='text',task='text',pool_id='input',nodes=[dict(id='gpu',kind='gpu')],
               edges=[],unresolved=[],cyclic=False)
        plans={('a','text','input'):p}
        row=dict(request_id='train',model='a',task='text',pool_id='input',profile_job_id='job',
            worker_response=dict(start_perf_ns=1000,end_perf_ns=1100,observed_stages=[]))
        job=dict(id='job',inputs={'a':'input'},start_offsets_ms={'a':.000010})
        return aliases,plans,row,job

    def test_actual_dag_cpu_entry_preserves_offsets_and_uses_injected_work(self):
        aliases,plans,row,job=self.fixture()
        result=simulate_job(plans,aliases,[row],job,lambda n,r:0,lambda n,r:200,max_events=10)
        self.assertEqual(result['rows'][0]['public_start_offset_ms'],.000010)
        self.assertEqual(result['rows'][0]['predicted_handler_ms'],.0002)
        self.assertEqual(result['rows'][0]['observed_training_handler_ms'],.0001)
        self.assertEqual(result['rows'][0]['absolute_relative_error'],1.)
        self.assertEqual(result['queue_diagnostics']['registered_request_count'],1)
        self.assertLess(result['events'],10)

    def test_actual_unknown_duration_or_event_budget_never_invents_endpoint(self):
        aliases,plans,row,job=self.fixture()
        def missing(*args):raise UnknownEvidence('NO_NUMERIC')
        with self.assertRaisesRegex(UnknownEvidence,'NO_NUMERIC'):
            simulate_job(plans,aliases,[row],job,lambda n,r:0,missing)
        with self.assertRaisesRegex(UnknownEvidence,'BUDGET'):
            simulate_job(plans,aliases,[row],job,lambda n,r:0,lambda n,r:200,max_events=1)

    def test_training_job_pairing_rejects_missing_or_extra_owner(self):
        aliases,plans,row,job=self.fixture()
        self.assertEqual(len(list(jobs_for_capture({'request_records':[row]},{'jobs':[job]}))),1)
        job['inputs']['b']='input'
        with self.assertRaisesRegex(ValueError,'Full original'):
            list(jobs_for_capture({'request_records':[row]},{'jobs':[job]}))

if __name__=='__main__':unittest.main()
