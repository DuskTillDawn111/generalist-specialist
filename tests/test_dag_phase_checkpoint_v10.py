"""Continuation counterexamples for absolute trained scheduler state."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.dag_checkpoint_v7 import digest,receipt
from tools.predictor_validation_20261027.dag_phase_checkpoint_v10 import decode,encode
from tools.predictor_validation_20261027.dag_phase_engine_v10 import DagPhaseCheckpointModel
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import DagPhaseFullHorizonModel
from tools.predictor_validation_20261027.trained_dag_availability_v8 import TrainedDagAvailabilityScheduler
from tools.predictor_validation_20261027.dag_frozen_model_v10 import FrozenNodeTable
from gsdeploy.forecast_errors import UnknownEvidence


class Work:
    def __call__(self,node,row):return 8 if row['alias']=='a' else 5


class Phase:
    def __call__(self,node,row):return 'prefill'


def policy():
    frontiers={owners:[[owner,'text','x',['prefill']] for owner in owners] for owners in ('a','b','ab')}
    cells=[dict(current_owner=None,reason='initial',frontier=frontiers['ab'],next_owner='a',grant_budget_ns=7,handoff_ns=2)]
    for owner in 'ab':
        other='b' if owner=='a' else 'a'
        cells.extend([
            dict(current_owner=owner,reason='contention_entry',frontier=frontiers['ab'],next_owner=owner,grant_budget_ns=7,handoff_ns=0),
            dict(current_owner=owner,reason='grant_expired',frontier=frontiers['ab'],next_owner=other,grant_budget_ns=5,handoff_ns=2),
            dict(current_owner=owner,reason='yield',frontier=frontiers[other],next_owner=other,grant_budget_ns=5,handoff_ns=1)])
    solo=[dict(frontier=frontiers[owner],dispatch_wait_ns=1) for owner in 'ab']
    return dict(schema='gsdeploy.trained_DAG_availability_policy.v8',training_only=True,
        reference_work_domain='SOLO_ONLY_REFERENCE_NS',cells=cells,uncontended_cells=solo,
        initial_uncontended_cells=[dict(frontier=frontiers[owner],dispatch_wait_ns=2) for owner in 'ab'])


class PhaseCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        source=Path(self.temp.name)/'model.json';source.write_text(json.dumps(policy()))
        self.evidence=receipt(source)
        self.aliases={owner:dict(family=owner,supported_tasks=['text']) for owner in 'ab'}
        plan=dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
            edges=[dict(before='gpu',after='end')],unresolved=[],cyclic=False)
        self.plans={(owner,'text','x'):copy.deepcopy(plan) for owner in 'ab'}
        self.rows=[dict(request_id=rid,alias=owner,task='text',pool_id='x',arrival_ns=arrival)
            for rid,owner,arrival in (('a0','a',0),('b0','b',4),('a1','a',0))]

    def engine(self,*,max_events=1000,policy_value=None):
        scheduler=TrainedDagAvailabilityScheduler(policy() if policy_value is None else policy_value,
            allow_unvalidated_training_candidate=True)
        return DagPhaseCheckpointModel(self.aliases,self.plans,duration_provider=Work(),phase_provider=Phase(),
            scheduler=scheduler,max_events=max_events,checkpoint_binding=dict(execution_context_sha256='a'*64,evidence=[self.evidence]))

    def full(self,engine=None,**kwargs):
        return (self.engine() if engine is None else engine).forecast(self.rows,0,100,drain_deadline_ns=150,**kwargs)

    def test_complete_numeric_result_matches_frozen_runtime_v8_engine(self):
        original=DagPhaseFullHorizonModel(self.aliases,self.plans,duration_provider=Work(),phase_provider=Phase(),
            scheduler=TrainedDagAvailabilityScheduler(policy(),allow_unvalidated_training_candidate=True))
        expected=original.forecast(self.rows,0,100,drain_deadline_ns=150)
        actual=self.full()
        for name in expected:
            if name not in ('scheduler_provenance','scheduler_checkpoint_supported'):
                self.assertEqual(actual[name],expected[name],name)

    def test_every_atomic_pause_resumes_identical_complete_trajectory(self):
        expected=self.full()
        for consumed in range(1,expected['events']):
            paused=self.full(pause_after_event_count=consumed)
            if 'checkpoint' not in paused:continue
            actual=self.full(checkpoint=paused['checkpoint'])
            self.assertEqual(actual,expected,consumed)

    def test_checkpoint_contains_compact_progress_and_separate_absolute_scheduler_state(self):
        cp=self.full(pause_at_or_after_ns=5)['checkpoint']
        state=decode(cp['state']);scheduler=decode(cp['scheduler_state'])
        self.assertTrue(all(set(r)=={'row','start_ns','done'} for r in state['running'].values()))
        self.assertEqual(scheduler['grant_end_ns'],11)
        self.assertLess(scheduler['last_clock_ns'],state['clock'])
        self.assertEqual(self.full(checkpoint=cp),self.full())

    def test_resigned_scheduler_field_or_history_forgery_is_rejected(self):
        original=self.full(pause_at_or_after_ns=5)['checkpoint']
        for name in ('current_owner','mode','grant_end_ns','handoff_end_ns','last_clock_ns','transitions'):
            cp=copy.deepcopy(original);state=decode(cp['scheduler_state'])
            if name=='current_owner':state[name]='b'
            elif name=='mode':state[name]='uncontended'
            elif name=='transitions':state[name]=tuple(state[name][:-1])
            else:state[name]=(state[name] or 0)+1
            cp['scheduler_state']=encode(state);cp['checkpoint_sha256']=digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'original-prefix replay'):
                self.full(checkpoint=cp)

    def test_policy_budget_or_source_change_cannot_continue_original_checkpoint(self):
        cp=self.full(pause_at_or_after_ns=5)['checkpoint']
        changed=policy();changed['cells'][0]['grant_budget_ns']+=1
        for engine in (self.engine(max_events=999),self.engine(policy_value=changed)):
            with self.assertRaisesRegex(ValueError,'execution contract'):self.full(engine,checkpoint=cp)
        path=Path(self.evidence['path']);path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'source changed'):self.full(checkpoint=cp)

    def test_frozen_plan_and_numeric_provider_reject_mutation(self):
        engine=self.engine()
        with self.assertRaises(TypeError):engine.plans['a','text','x']['nodes'][0]['kind']='copy'
        with self.assertRaises(AttributeError):engine._frozen_model.plans={}
        table=FrozenNodeTable({('a','text','x','gpu'):8},sources=[self.evidence],domain='SOLO_ONLY_REFERENCE_NS')
        with self.assertRaises(TypeError):table._values['a','text','x','gpu']=20
        with self.assertRaises(AttributeError):table.table_sha256='changed'
        with self.assertRaises(TypeError):engine.scheduler.policy['training_only']=False
        with self.assertRaises(AttributeError):engine.scheduler.policy={}
        for obj,names in ((engine._frozen_model,('_sealed','plans','plans_sha256')),
                (table,('_sealed','_values','table_sha256')),
                (engine.scheduler,('_static_sealed','policy','cells','policy_sha256'))):
            for name in names:
                with self.subTest(object=type(obj).__name__,field=name),self.assertRaises(AttributeError):
                    delattr(obj,name)

    def test_compact_checkpoint_does_not_duplicate_large_static_plan_payload(self):
        self.plans['a','text','x']['immutable_profile_payload']='x'*1_000_000
        cp=self.full(pause_at_or_after_ns=5)['checkpoint']
        self.assertLess(len(json.dumps(cp)),50_000)
        self.assertNotIn('immutable_profile_payload',json.dumps(cp))
        self.assertEqual(self.full(checkpoint=cp),self.full())

    def test_resume_consumes_original_absolute_event_budget(self):
        engine=self.engine(max_events=8)
        cp=self.full(engine,pause_after_event_count=2)['checkpoint']
        with self.assertRaisesRegex(UnknownEvidence,'EVENT_BUDGET_EXCEEDED'):
            self.full(self.engine(max_events=8),checkpoint=cp)
        with self.assertRaisesRegex(UnknownEvidence,'EVENT_BUDGET_EXCEEDED'):
            self.full(self.engine(max_events=8))

    def test_queued_start_supply_is_relative_to_preceding_explicit_ready(self):
        class Supply:
            def __call__(self,row):return row['arrival_ns']+2
            def from_owner_ready(self,row,*,owner_ready_ns):return max(row['arrival_ns'],owner_ready_ns)+2
        engine=self.engine();engine.request_start_provider=Supply()
        result=self.full(engine)
        self.assertEqual(result['starts']['a0'],2)
        self.assertEqual(result['starts']['a1'],result['finishes']['a0']['worker_next_ready_ns']+2)
        self.assertEqual(result['cohort']['a1']['arrival_ns'],0)
        self.assertEqual(result['cohort']['b0']['arrival_ns'],4)


if __name__=='__main__':unittest.main()
