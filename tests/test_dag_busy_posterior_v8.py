"""Busy observations select actual joint DAG trajectories, without target labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.cuda_dag_simulator_v9 import DagFullHorizonModel
from tools.predictor_validation_20261027.dag_busy_posterior_v8 import condition_busy_scenarios, key, SCHEMA
from tools.predictor_validation_20261027.dag_checkpoint_v7 import receipt
from tools.predictor_validation_20261027.dag_visible_observer_v8 import observe_cutoff
from tests.test_dag_visible_observer_v8 import ALIASES, CONTEXT, native, Poison


class Duration:
    def __init__(self, ns):self.ns=ns
    def __call__(self, node, row):return self.ns


class Release:
    def __call__(self, node, row):return dict(not_before_offset_ns=0,after_predecessors_ns=0)


class Life:
    def __init__(self, receipt_ns=20):self.receipt_ns=receipt_ns
    def delays(self,*args):return dict(known=True,parent_receipt_ms=self.receipt_ns/1e6,worker_next_ready_ms=2/1e6)


class Resources:
    def power(self,*args):return dict(known=True,watts=10.)
    def memory(self,*args):return dict(known=True,peak_memory_mib=30.)


def raw_fixture():
    rows=[dict(request_id='r0',alias='a',task='text',pool_id='x',arrival_ns=100),
          dict(request_id='r1',alias='a',task='text',pool_id='x',arrival_ns=200)]
    raw=dict(schema='gsdeploy.cpu_fixture.native.v8',execution_context_sha256=CONTEXT,
        phase_start_ns=100,ready={'a':dict(pid=7)},
        priming_request_records=[native('prime',10,11,12,20,21,22,23)],
        request_records=[native('r0',100,100,100,110,130,112,131),
                         native('r1',200,200,200,210,230,212,231)])
    raw['initial_request_inventory']=[{k:r[k] for k in ('request_id','model','task','pool_id')} for r in raw['priming_request_records']]
    raw['request_records'][0]['execution_events'][0]['received_time_ns']=101
    raw['request_records'][0]['execution_events'] += [
        dict(request_id='r0',perf_ns=104,received_time_ns=105,event='start',segment_id='decode',kind='gpu'),
        dict(request_id='r0',perf_ns=106,received_time_ns=108,event='end',segment_id='decode',kind='gpu')]
    # Future body access must throw; neither observer nor conditioning touches it.
    raw['request_records'][0]['worker_response']=Poison()
    raw['request_records'][1]['worker_response']=Poison()
    return rows,raw


def visible(cutoff=115,raw=None):
    rows,base=raw_fixture()
    return observe_cutoff(base if raw is None else raw,rows,ALIASES,execution_context_sha256=CONTEXT,
        phase_start_ns=100,deadline_ns=300,drain_deadline_ns=400,cutoff_ns=cutoff,allow_cpu_fixture=True)


class BusyPosteriorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        sources=[]
        for run in ('A','B','C'):
            path=Path(self.temp.name)/(run+'.json')
            path.write_text(json.dumps(dict(split='training',kind='cpu_fixture',run_id=run,
                execution_context_sha256=CONTEXT,request_records=[dict(split='training')])));sources.append(receipt(path))
        profile=dict(start_reader_delay_ns=1,ready_reader_delay_ns=19,phases=[
            dict(segment_id='decode',kind='gpu',event='start',perf_offset_ns=4,reader_delay_ns=1),
            dict(segment_id='decode',kind='gpu',event='end',perf_offset_ns=6,reader_delay_ns=2)])
        self.support=dict(schema=SCHEMA,split='training',execution_context_sha256=CONTEXT,sources=sources,
            generation_work_scope='training_only_public_input_conditioned',
            joint_scope='whole_run_node_host_lifecycle_reader_emission',
            timing_radius_basis='cross_fitted_training_run_support',observation_tolerance_ns=0,
            minimum_independent_survivor_runs=2,scenarios=[dict(id=x,training_run_ids=[x],prior_mass=1,
                request_start_delay_ns_by_input={key('a','text','x'):0},
                emissions={key('a','text','x'):copy.deepcopy(profile)}) for x in ('A','B','C')])
        self.calls=[]

    def factory(self,*,scenario_id,now_ns,request_start_provider=None):
        self.calls.append((scenario_id,now_ns))
        plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
                  edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
        return DagFullHorizonModel(ALIASES,{('a','text','x'):plan},
            duration_provider=Duration({'A':10,'B':15,'C':10}[scenario_id]),release_provider=Release(),
            lifecycle_provider=Life(1 if scenario_id=='C' else 20),power_provider=Resources(),memory_provider=Resources(),
            checkpoint_binding=dict(execution_context_sha256=CONTEXT,evidence=self.support['sources']),
            request_start_provider=request_start_provider)

    def condition(self,state=None,**kwargs):
        return condition_busy_scenarios(visible() if state is None else state,self.support,self.factory,
            allow_cpu_fixture=True,**kwargs)

    def test_no_result_conditions_reader_survival_not_physical_end_and_runs_actual_dag(self):
        value=self.condition(score_kwargs=dict(rates={'text':1},refs={'text':1},quality_loss={'text':{'text':.1}},
            memory_budget_mib=100,energy_reference_watts=10))
        self.assertEqual(value['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        self.assertEqual([c['id'] for c in value['candidates']],['A','B'])
        self.assertEqual(value['rejected'],[dict(id='C',reason='VISIBLE_EVENT_PATTERN_OR_READER_SURVIVAL_MISMATCH')])
        a=value['candidates'][0];b=value['candidates'][1]
        self.assertTrue(a['latent_request_states']['r0']['end_happened_result_in_transit'])
        self.assertTrue(a['latent_request_states']['r0']['ready_happened_reader_in_transit'])
        self.assertFalse(a['latent_request_states']['r0']['handler_active_at_cutoff'])
        self.assertEqual(a['latent_request_states']['r0']['result_reader_remaining_ns'],15)
        self.assertEqual(b['latent_request_states']['r0']['result_reader_remaining_ns'],20)
        self.assertEqual(a['conditional_empirical_mass'],.5)
        self.assertLessEqual(a['dag_state']['left_clock_ns'],115)
        self.assertGreater(a['dag_state']['right_clock_ns'],115)
        self.assertEqual(set(a['full_dag_forecast']['cohort']),{'r0','r1'})
        self.assertEqual(a['full_dag_forecast']['cohort']['r1']['arrival_ns'],200)
        self.assertEqual(a['metrics']['cohort_size'],2)
        self.assertAlmostEqual(a['metrics']['per_task']['text']['mean_ms'],.00003)
        self.assertIsNone(a['metrics']['terms']['E']);self.assertIsNone(a['metrics']['terms']['M'])
        self.assertIsNone(a['metrics']['full_U']);self.assertFalse(value['posterior_calibrated'])
        self.assertFalse(value['elapsed_fraction_cursor_used']);self.assertFalse(value['whole_request_multiplier_used'])

    def test_no_training_or_single_independent_run_is_unknown_without_engine_call(self):
        first=condition_busy_scenarios(visible(),None,self.factory)
        self.assertEqual(first['status'],'UNKNOWN');self.assertTrue(first['additional_collection_plan'])
        self.support['sources']=self.support['sources'][:1]
        second=self.condition()
        self.assertEqual(second['status'],'UNKNOWN');self.assertEqual(self.calls,[])

    def test_phase_evidence_rejects_wrong_joint_emission_without_converting_to_cursor(self):
        for s in self.support['scenarios']:
            s['emissions'][key('a','text','x')]['phases'][0]['perf_offset_ns']=5
        result=self.condition(capture_latent_checkpoints=False)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidates'],[])
        self.assertTrue(all(x['reason']=='VISIBLE_BOUNDARY_OUTSIDE_TRAINING_SUPPORT_RADIUS' or
            x['reason']=='VISIBLE_EVENT_PATTERN_OR_READER_SURVIVAL_MISMATCH' for x in result['rejected']))

    def test_actual_prefix_screen_and_resume_preserve_complete_forecast(self):
        screened=self.condition(capture_latent_checkpoints=False)
        uninterrupted=self.condition(capture_latent_checkpoints=False,prefix_screen=False)
        self.assertEqual(screened,uninterrupted)

    def test_prefix_only_keeps_full_cohort_but_produces_no_full_utility_or_posterior(self):
        result=self.condition(capture_latent_checkpoints=False,prefix_screen_only=True)
        self.assertEqual(result['status'],'ACTUAL_DAG_PREFIX_SCREEN_ONLY_NOT_FORECAST')
        self.assertEqual([r['id'] for r in result['prefix_screened']],['A','B'])
        self.assertTrue(all(r['complete_original_cohort']==2 for r in result['prefix_screened']))
        self.assertEqual(result['prefix_screened'][0]['atomic_clock_ns'],130)
        self.assertFalse(result['full_horizon_forecast_complete'])
        self.assertFalse(result['conditional_mass_available'])
        self.assertIsNone(result['full_U']);self.assertEqual(result['candidates'],[])

    def test_rejected_prefix_never_consumes_future_request_service(self):
        for scenario in self.support['scenarios']:
            scenario['emissions'][key('a','text','x')]['phases'][0]['perf_offset_ns']=5
        original=self.factory
        class RejectFutureDuration:
            def __call__(self,node,row):
                if row['request_id']=='r1':
                    raise AssertionError('Rejected prefix unnecessarily rolled out future service')
                return 10
        def factory(**kwargs):
            engine=original(**kwargs);engine.duration_provider=RejectFutureDuration();return engine
        result=condition_busy_scenarios(visible(),self.support,factory,allow_cpu_fixture=True,
            capture_latent_checkpoints=False)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidates'],[])

    def test_chronology_failure_rejects_each_scenario_without_clipping_phase_or_batch_crash(self):
        for scenario in self.support['scenarios']:
            scenario['emissions'][key('a','text','x')]['phases'][1]['perf_offset_ns']=20
        result=self.condition(capture_latent_checkpoints=False)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidates'],[])
        self.assertEqual(len(result['rejected']),3)
        self.assertTrue(all(r['reason']=='DAG_READER_JOINT_CHRONOLOGY_INCONSISTENT' for r in result['rejected']))
        self.assertTrue(all(s['emissions'][key('a','text','x')]['phases'][1]['perf_offset_ns']==20
                            for s in self.support['scenarios']))

    def test_future_target_body_replacement_does_not_change_conditioning(self):
        first=self.condition(capture_latent_checkpoints=False)
        rows,raw=raw_fixture();raw['request_records'][0]['worker_response']={'end_perf_ns':99999,'quality':0,'generated_tokens':1}
        raw['request_records'][1]['worker_response']={'end_perf_ns':222222,'quality':1,'generated_tokens':1000000}
        second=self.condition(visible(raw=raw),capture_latent_checkpoints=False)
        self.assertEqual(first,second)

    def test_heldout_or_mutated_training_sources_and_reweighted_objective_rejected(self):
        path=Path(self.support['sources'][0]['path']);doc=json.loads(path.read_text());doc['split']='validation'
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'Changed joint'):self.condition()
        self.support['sources'][0]=receipt(path)
        with self.assertRaisesRegex(ValueError,'Validation'):self.condition()

    def test_reweighted_original_scoring_window_rejected(self):
        with self.assertRaisesRegex(ValueError,'Original scoring'):
            self.condition(score_kwargs=dict(weights={k:1/5 if k!='P' else 0 for k in 'QTLPME'}),
                           capture_latent_checkpoints=False)

    def test_visible_start_binds_real_host_delay_while_future_uses_training_supply(self):
        rows,raw=raw_fixture();first=raw['request_records'][0]
        first['actual_submit_time_ns']=104
        for event in first['execution_events']:
            event['perf_ns']+=8;event['received_time_ns']+=8
        first['response_reader_event_enqueue_time_ns']=138
        result=self.condition(visible(125,raw),capture_latent_checkpoints=True)
        self.assertEqual(result['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        for candidate in result['candidates']:
            forecast=candidate['full_dag_forecast']
            self.assertEqual(forecast['starts']['r0'],108)
            self.assertEqual(forecast['queue_wait_ns']['r0'],8)
            self.assertEqual(forecast['cohort']['r0']['arrival_ns'],100)
            self.assertEqual(forecast['cohort']['r1']['arrival_ns'],200)
            self.assertEqual(forecast['starts']['r1'],200)

    def test_missing_future_start_supply_stays_unknown_without_engine_call(self):
        for scenario in self.support['scenarios']:
            scenario['request_start_delay_ns_by_input']={}
        result=self.condition()
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(self.calls,[])
        self.assertEqual(len(result['rejected']),3)
        self.assertTrue(all(x['reason']=='TRAINED_FUTURE_PUBLIC_INPUT_REQUEST_START_SUPPLY_REQUIRED' for x in result['rejected']))

    def test_completed_observed_prefix_replaces_predicted_prefix_before_full_cohort_score(self):
        rows,raw=raw_fixture();raw['request_records'][0]['worker_response']={'end_perf_ns':110}
        self.support['scenarios'][0]['training_run_ids']=['A','B']
        self.support['scenarios']=[self.support['scenarios'][0],self.support['scenarios'][2]]
        result=self.condition(visible(150,raw),score_kwargs=dict(rates={'text':1},refs={'text':1},
            quality_loss={'text':{'text':.1}},memory_budget_mib=100,energy_reference_watts=10))
        self.assertEqual(result['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        a=result['candidates'][0]
        self.assertEqual(set(a['scoring_forecast']['cohort']),{'r1'})
        self.assertEqual(a['metrics']['cohort_size'],2)
        self.assertAlmostEqual(a['metrics']['per_task']['text']['mean_ms'],.00003)
        self.assertTrue(all(r['time_ns']>=150 for r in a['scoring_forecast']['resource_transitions']))
        self.assertIsNone(a['metrics']['terms']['E']);self.assertIsNone(a['metrics']['terms']['M'])


if __name__=='__main__':unittest.main()
