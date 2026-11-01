"""Actual v10 busy inference executes the trained scheduler and original cohort."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.dag_checkpoint_v7 import receipt
from tools.predictor_validation_20261027.dag_busy_posterior_v8 import SCHEMA,key
from tools.predictor_validation_20261027.dag_visible_busy_v10 import condition_visible_phase_scenarios
from tools.predictor_validation_20261027.dag_phase_engine_v10 import DagPhaseCheckpointModel
from tools.predictor_validation_20261027.trained_dag_availability_v8 import TrainedDagAvailabilityScheduler
from tests.test_dag_busy_posterior_v8 import visible,ALIASES,CONTEXT,Duration,Release,Life,Resources
from tests.test_dag_phase_checkpoint_v10 import Phase


class PhaseBusyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        sources=[]
        for run in 'ABC':
            path=Path(self.temp.name)/(run+'.json')
            path.write_text(json.dumps(dict(split='training',kind='cpu_fixture',run_id=run,
                execution_context_sha256=CONTEXT,request_records=[dict(split='training')])));sources.append(receipt(path))
        profile=dict(start_reader_delay_ns=1,ready_reader_delay_ns=19,phases=[
            dict(segment_id='decode',kind='gpu',event='start',perf_offset_ns=4,reader_delay_ns=1),
            dict(segment_id='decode',kind='gpu',event='end',perf_offset_ns=6,reader_delay_ns=2)])
        self.support=dict(schema=SCHEMA,split='training',execution_context_sha256=CONTEXT,sources=sources,
            generation_work_scope='training_only_public_input_conditioned',joint_scope='whole_run_node_host_lifecycle_reader_emission',
            timing_radius_basis='cross_fitted_training_run_support',observation_tolerance_ns=0,
            minimum_independent_survivor_runs=2,scenarios=[dict(id=x,training_run_ids=[x],prior_mass=1,
                request_start_delay_ns_by_input={key('a','text','x'):0},emissions={key('a','text','x'):copy.deepcopy(profile)}) for x in 'ABC'])

    def factory(self,*,scenario_id,now_ns,request_start_provider):
        plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
            edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
        solo=dict(frontier=[['a','text','x',['prefill']]],dispatch_wait_ns=0)
        policy=dict(schema='gsdeploy.trained_DAG_availability_policy.v8',training_only=True,
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',cells=[],uncontended_cells=[solo],initial_uncontended_cells=[solo])
        return DagPhaseCheckpointModel(ALIASES,{('a','text','x'):plan},
            duration_provider=Duration(15 if scenario_id=='B' else 10),release_provider=Release(),phase_provider=Phase(),
            lifecycle_provider=Life(1 if scenario_id=='C' else 20),power_provider=Resources(),memory_provider=Resources(),
            scheduler=TrainedDagAvailabilityScheduler(policy,allow_unvalidated_training_candidate=True),
            checkpoint_binding=dict(execution_context_sha256=CONTEXT,evidence=self.support['sources']),
            request_start_provider=request_start_provider)

    def condition(self,**kwargs):
        return condition_visible_phase_scenarios(visible(),self.support,self.factory,allow_cpu_fixture=True,**kwargs)

    def test_actual_phase_dag_full_suffix_and_busy_reader_survival(self):
        result=self.condition(score_kwargs=dict(rates={'text':1},refs={'text':1},quality_loss={'text':{'text':.1}},
            memory_budget_mib=100,energy_reference_watts=10))
        self.assertEqual(result['status'],'EMPIRICAL_CONDITIONAL_DAG_SCENARIOS_NOT_CALIBRATED')
        self.assertEqual([c['id'] for c in result['candidates']],['A','B'])
        self.assertEqual(result['engine_kind'],'independent_v10_trained_availability_DAG')
        self.assertFalse(result['equal_context_fallback_used'])
        for candidate in result['candidates']:
            self.assertEqual(set(candidate['full_dag_forecast']['cohort']),{'r0','r1'})
            self.assertEqual(candidate['full_dag_forecast']['cohort']['r1']['arrival_ns'],200)
            self.assertTrue(candidate['full_dag_forecast']['scheduler_checkpoint_supported'])
            self.assertIsNone(candidate['metrics']['full_U'])

    def test_actual_prefix_resume_equals_uninterrupted_new_scheduler_forecast(self):
        screened=self.condition(capture_latent_checkpoints=False)
        uninterrupted=self.condition(capture_latent_checkpoints=False,prefix_screen=False)
        self.assertEqual(screened,uninterrupted)

    def test_unsupported_phase_policy_rejects_each_whole_scenario_without_fallback(self):
        original=self.factory
        def factory(**kwargs):
            engine=original(**kwargs)
            class UnsupportedPhase:
                def __call__(self,node,row):return 'untrained'
            engine.phase_provider=UnsupportedPhase();return engine
        result=condition_visible_phase_scenarios(visible(),self.support,factory,allow_cpu_fixture=True,
            capture_latent_checkpoints=False)
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(result['candidates'],[])
        self.assertEqual(len(result['rejected']),3)
        self.assertTrue(all(r['stage']=='TRAINED_PHASE_DAG_UNSUPPORTED' for r in result['rejected']))
        self.assertFalse(result['equal_context_fallback_used'])


if __name__=='__main__':unittest.main()
