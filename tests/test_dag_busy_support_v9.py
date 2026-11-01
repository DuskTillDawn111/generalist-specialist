"""Raw rederivation rejects self-resigned, caller-invented training values."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.resources import identity
from tools.predictor_validation_20261027.dag_checkpoint_v9 import receipt
from tools.predictor_validation_20261027.dag_busy_support_v9 import (
    fit_support,verify_support,load_support,extract_run,FIXTURE_REVIEW,JointLifecycleFromSupport,
)
from tools.predictor_validation_20261027.dag_busy_posterior_v8 import key,condition_busy_scenarios
from tools.predictor_validation_20261027.cuda_dag_simulator_v9 import DagFullHorizonModel
from tests.test_dag_busy_posterior_v8 import raw_fixture,visible,ALIASES,CONTEXT,Duration,Release,Resources,Life


class JointSupportFitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.reviews=[]
        for run_id,delta in (('A',0),('B',2)):
            rows,raw=raw_fixture();run=self.root/run_id;run.mkdir()
            plan=dict(schema='gsdeploy.cpu_fixture.queue_training_plan.v9',split='training',training_only=True,
                execution_context_sha256=CONTEXT,aliases=ALIASES,run_id=run_id,
                trace=[dict(request_id=row['request_id'],alias=row['alias'],task=row['task'],pool_id=row['pool_id'],
                            offset_ns=row['arrival_ns']-100) for row in rows])
            plan_path=run/'plan.json';plan_path.write_text(json.dumps(plan))
            raw.update(schema='gsdeploy.native_flow_collection.v1',split='training',training_only=True,
                kind='cpu_fixture',status='completed',run_id=run_id,plan=receipt(plan_path))
            for index,row in enumerate(raw['request_records']):
                shift=delta+(1 if delta and index else 0)
                row['split']='training';row['worker_response']={'end_perf_ns':110+index*100+shift}
                for event in row['execution_events']:
                    event['perf_ns']+=shift;event['received_time_ns']+=shift
                row['response_reader_event_enqueue_time_ns']+=shift
                row['previous_handler_next_request_ready']['perf_ns']+=shift
                row['previous_handler_next_request_ready']['received_time_ns']+=shift
            raw_path=run/'raw.json';raw_path.write_text(json.dumps(raw))
            review=dict(schema=FIXTURE_REVIEW,status='PASS',issues=[],
                        evidence=dict(plan=receipt(plan_path),raw=receipt(raw_path)))
            review_path=run/'review.json';review_path.write_text(json.dumps(review));self.reviews.append(receipt(review_path))

    def fit(self):return fit_support(self.reviews,allow_cpu_fixture=True)

    def test_whole_run_occurrences_and_leave_entire_run_out_radius_rederived(self):
        support=self.fit();k=key('a','text','x')
        self.assertTrue(support['fixture_only']);self.assertEqual(support['observation_tolerance_ns'],3)
        self.assertEqual(support['scenarios'][0]['request_start_delay_ns_by_input'][k],[0,0])
        self.assertEqual(support['scenarios'][1]['request_start_delay_ns_by_input'][k],[2,3])
        self.assertEqual(support['scenarios'][0]['prior_mass'],1)
        for e in support['leave_one_run_out_errors']:
            self.assertNotIn(e['heldout_run_id'],e['training_run_ids'])
        self.assertEqual(verify_support(support,allow_cpu_fixture=True),support)
        path=self.root/'support.json';path.write_text(json.dumps(support))
        self.assertEqual(load_support(path,allow_cpu_fixture=True),support)
        self.assertNotIn('prediction',json.dumps(support['training_timing_records']))

    def test_modified_emission_start_radius_prior_rejected_even_after_resigning(self):
        original=self.fit();k=key('a','text','x')
        for field in ('emission','start','radius','prior','definition','run_order'):
            support=copy.deepcopy(original)
            if field=='emission':support['scenarios'][0]['emissions'][k][0]['ready_reader_delay_ns']+=1
            if field=='start':support['scenarios'][0]['request_start_delay_ns_by_input'][k][0]+=1
            if field=='radius':support['observation_tolerance_ns']+=1
            if field=='prior':support['scenarios'][0]['prior_mass']=2
            if field=='definition':support['timing_radius_scope']='calibrated confidence guarantee'
            if field=='run_order':support['scenarios'][0]['emissions'][k].reverse()
            support.pop('model_sha256');support['model_sha256']=identity(support)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'recomputed'):
                verify_support(support,allow_cpu_fixture=True)

    def test_cpu_fixture_never_qualifies_measured_or_validation_data(self):
        with self.assertRaisesRegex(ValueError,'CPU fixture'):fit_support(self.reviews)
        with self.assertRaisesRegex(ValueError,'CPU fixture'):verify_support(self.fit())
        ref=self.reviews[0];review=json.loads(Path(ref['path']).read_text())
        raw_path=Path(review['evidence']['raw']['path']);raw=json.loads(raw_path.read_text());raw['split']='validation'
        raw_path.write_text(json.dumps(raw));review['evidence']['raw']=receipt(raw_path)
        Path(ref['path']).write_text(json.dumps(review));self.reviews[0]=receipt(ref['path'])
        with self.assertRaisesRegex(ValueError,'Validation'):self.fit()

    def test_duplicate_capture_and_insufficient_independent_runs(self):
        with self.assertRaisesRegex(ValueError,'Distinct independent'):
            fit_support([self.reviews[0],self.reviews[0]],allow_cpu_fixture=True)
        support=fit_support(self.reviews[:1],allow_cpu_fixture=True)
        self.assertTrue(support['fit_status'].startswith('UNKNOWN'))
        self.assertTrue(support['missing_leave_one_run_out_support'])

    def test_fifo_wait_is_removed_before_fitting_host_dispatch_lag(self):
        ref=self.reviews[0];review=json.loads(Path(ref['path']).read_text())
        plan_path=Path(review['evidence']['plan']['path']);plan=json.loads(plan_path.read_text())
        plan['trace'][1]['offset_ns']=0;plan_path.write_text(json.dumps(plan));review['evidence']['plan']=receipt(plan_path)
        raw_path=Path(review['evidence']['raw']['path']);raw=json.loads(raw_path.read_text())
        row=raw['request_records'][1];row['scheduled_time_ns']=100;row['actual_submit_time_ns']=100
        for event in row['execution_events']:
            event['perf_ns']-=83;event['received_time_ns']-=83
        row['worker_response']['end_perf_ns']-=83;row['response_reader_event_enqueue_time_ns']-=83
        row['previous_handler_next_request_ready']['perf_ns']-=83
        row['previous_handler_next_request_ready']['received_time_ns']-=83
        raw['plan']=review['evidence']['plan'];raw_path.write_text(json.dumps(raw));review['evidence']['raw']=receipt(raw_path)
        Path(ref['path']).write_text(json.dumps(review));ref=receipt(ref['path'])
        records=extract_run(ref,allow_cpu_fixture=True)['records']
        self.assertEqual(records[1]['owner_eligible_to_start_ns'],5)
        self.assertEqual(records[1]['observed_handler_wall_ns'],10)
        self.assertNotEqual(records[1]['owner_eligible_to_start_ns'],117-100)

    def test_rederived_fit_and_loader_drive_actual_joint_dag_scenarios(self):
        # Both independent training runs support the exact already observed
        # prefix. Their unseen future host supplies still differ.
        ref=self.reviews[1];review=json.loads(Path(ref['path']).read_text())
        path=Path(review['evidence']['raw']['path']);raw=json.loads(path.read_text());row=raw['request_records'][0]
        for event in row['execution_events']:
            event['perf_ns']-=2;event['received_time_ns']-=2
        row['worker_response']['end_perf_ns']-=2;row['response_reader_event_enqueue_time_ns']-=2
        row['previous_handler_next_request_ready']['perf_ns']-=2
        row['previous_handler_next_request_ready']['received_time_ns']-=2
        path.write_text(json.dumps(raw));review['evidence']['raw']=receipt(path)
        Path(ref['path']).write_text(json.dumps(review));self.reviews[1]=receipt(ref['path'])
        support=self.fit();state=visible()
        def factory(*,scenario_id,now_ns,request_start_provider):
            plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
                edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
            life=JointLifecycleFromSupport(support,scenario_id,state['full_public_cohort'],allow_cpu_fixture=True)
            return DagFullHorizonModel(ALIASES,{('a','text','x'):plan},duration_provider=Duration(10),
                release_provider=Release(),lifecycle_provider=life,power_provider=Resources(),memory_provider=Resources(),
                request_start_provider=request_start_provider,
                checkpoint_binding=dict(execution_context_sha256=CONTEXT,evidence=support['sources']))
        result=condition_busy_scenarios(state,support,factory,allow_cpu_fixture=True)
        self.assertEqual(result['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        self.assertEqual(len(result['candidates']),2)
        a,b=result['candidates']
        self.assertEqual(a['full_dag_forecast']['starts']['r0'],100)
        self.assertEqual(b['full_dag_forecast']['starts']['r0'],100)
        self.assertEqual(b['full_dag_forecast']['starts']['r1'],203)
        self.assertEqual(b['full_dag_forecast']['cohort']['r1']['arrival_ns'],200)
        self.assertTrue(a['latent_request_states']['r0']['end_happened_result_in_transit'])
        self.assertFalse(result['posterior_calibrated'])

    def test_supported_start_lag_conditions_to_exact_fact_instead_of_requiring_atom_equality(self):
        support=self.fit();state=visible()
        def factory(*,scenario_id,now_ns,request_start_provider):
            plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
                edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
            life=JointLifecycleFromSupport(support,scenario_id,state['full_public_cohort'],allow_cpu_fixture=True)
            return DagFullHorizonModel(ALIASES,{('a','text','x'):plan},duration_provider=Duration(10),
                release_provider=Release(),lifecycle_provider=life,request_start_provider=request_start_provider,
                checkpoint_binding=dict(execution_context_sha256=CONTEXT,evidence=support['sources']))
        result=condition_busy_scenarios(state,support,factory,allow_cpu_fixture=True)
        self.assertEqual(result['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        self.assertEqual([c['id'] for c in result['candidates']],['A','B'])
        self.assertEqual(result['rejected'],[])
        self.assertTrue(all(c['full_dag_forecast']['starts']['r0']==100 for c in result['candidates']))

    def test_timing_radius_cannot_relax_an_exact_visible_end_or_ready(self):
        support=self.fit();rows,raw=raw_fixture()
        raw['request_records'][0]['worker_response']={'end_perf_ns':110}
        state=visible(150,raw)
        def factory(*,scenario_id,now_ns,request_start_provider):
            plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
                edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
            life=JointLifecycleFromSupport(support,scenario_id,state['full_public_cohort'],allow_cpu_fixture=True)
            return DagFullHorizonModel(ALIASES,{('a','text','x'):plan},duration_provider=Duration(11),
                release_provider=Release(),lifecycle_provider=life,request_start_provider=request_start_provider,
                checkpoint_binding=dict(execution_context_sha256=CONTEXT,evidence=support['sources']))
        result=condition_busy_scenarios(state,support,factory,allow_cpu_fixture=True)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidates'],[])
        self.assertEqual([r['reason'] for r in result['rejected']],
            ['EXACT_VISIBLE_HOST_LIFECYCLE_BOUNDARY_MISMATCH']*2)


if __name__=='__main__':unittest.main()
