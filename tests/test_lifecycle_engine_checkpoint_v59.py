"""Executable v49/v24 lifecycle continuation. Explicit CPU synthetic evidence."""
import copy
import json
import unittest
from unittest.mock import patch
from tests import test_lifecycle_scheduler_checkpoint_v1 as component_fixture
from tools.predictor_validation_20261027 import lifecycle_engine_checkpoint_v59 as cp
from tools.predictor_validation_20261027 import lifecycle_scheduler_checkpoint_v1 as scp
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleDagPhaseModel
from tools.predictor_validation_20261027.owner_lifecycle_engine_v59 import ResumableLifecycleDagPhaseModel
from tools.predictor_validation_20261027.dag_queue_diagnostics_v6 import DagQueueDiagnostics
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import receipt
from tools.predictor_validation_20261027.profiling import digest


def duration(node,row):return node['duration_ns']
def release(node,row):return dict(node['supply']) if 'supply' in node else 0
def phase(node,row):return 'decode'
class Lifecycle:
    def delays(self,row,aliases,scenario):
        return dict(known=True,parent_receipt_ms=.03,worker_next_ready_ms=.05)
class Resources:
    def power(self,handlers,aliases,scenario):return dict(known=True,watts=10.+5.*len(handlers))
    def memory(self,handlers,aliases,scenario):return dict(known=True,peak_memory_mib=20.+2.*len(handlers))


class EngineCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.component=component_fixture.SchedulerCheckpointTests()
        self.component.setUp();self.addCleanup(self.component.doCleanups)
        self.owners=sorted(self.component.scheduler().owners)
        self.aliases={owner:dict(family='f',supported_tasks=['text']) for owner in self.owners}
        # Four resident owners retained; only the two dedicated workers are offered here.
        # text#1 needs a real SAVE/handoff before its queued reference work can progress.
        self.plans={(owner,'text','short'):dict(nodes=[
            dict(id='g',kind='gpu',duration_ns=10000 if owner=='text#0' else 80000,
                 supply=dict(not_before_offset_ns=1000,after_predecessors_ns=0)),
            dict(id='tail',kind='handler_end',supply=dict(not_before_offset_ns=0,after_predecessors_ns=7000))],
            edges=[dict(before='g',after='tail')],cyclic=False,unresolved=False,
            default_stream_semantics='CPU_FIXTURE_EXPLICIT') for owner in self.owners}
        self.rows=[dict(alias='text#0',task='text',pool_id='short',request_id='a0',arrival_ns=0),
            dict(alias='text#0',task='text',pool_id='short',request_id='a1',arrival_ns=5000),
            dict(alias='text#1',task='text',pool_id='short',request_id='b0',arrival_ns=1000),
            dict(alias='text#1',task='text',pool_id='short',request_id='b1',arrival_ns=20000)]
        self.deadline=1800000;self.drain=1850000

    def engine(self,*,seeded=False,diagnostics=True,old=False):
        scheduler=self.component.scheduler(seeded=seeded)
        diag=DagQueueDiagnostics(self.aliases,self.plans,duration) if diagnostics else None
        cls=LifecycleDagPhaseModel if old else ResumableLifecycleDagPhaseModel
        return cls(self.aliases,self.plans,scheduler=scheduler,duration_provider=duration,
            release_provider=release,phase_provider=phase,lifecycle_provider=Lifecycle(),
            power_provider=Resources(),memory_provider=Resources(),require_energy=True,require_memory=True,
            queue_diagnostics=diag,checkpoint_binding=dict(execution_context_sha256='b'*64,
                evidence=[receipt(__file__)],provider_state_policy='DETERMINISTIC_STATELESS_NO_MUTABLE_RNG'))

    def run_engine(self,engine,**kw):
        return engine.forecast(self.rows,0,self.deadline,drain_deadline_ns=self.drain,**kw)

    def checkpoint(self,cutoff,*,seeded=False):
        return self.run_engine(self.engine(seeded=seeded),pause_at_ns=cutoff)['checkpoint']

    def reseal(self,checkpoint,state):
        checkpoint['state']=scp.encode(state)
        checkpoint['six_term_prefix']=scp.encode(cp.prefix_accounting(state,checkpoint['contract']))
        checkpoint['checkpoint_sha256']=digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})

    def test_uninterrupted_matches_frozen_v21_numerical_results(self):
        baseline=self.run_engine(self.engine(old=True))
        actual=self.run_engine(self.engine())
        self.assertTrue(actual.pop('scheduler_checkpoint_supported'))
        baseline.pop('scheduler_checkpoint_supported')
        self.assertEqual(actual,baseline)
        self.assertEqual(set(actual['finishes']),{'a0','a1','b0','b1'})

    def test_exact_arbitrary_cutoff_and_JSON_multi_resume_are_identical(self):
        for seeded in (False,True):
            with self.subTest(seeded=seeded):
                cutoffs=(0,500,1000,5001,12000,17001,35000,70000,200000,350000,
                         1377296,1400000,1419295,1500000)
                if seeded:
                    from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleSchedulingView
                    from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
                    probe=self.component.scheduler(seeded=True)
                    view=LifecycleSchedulingView(0,(),tuple((owner,OwnerLifecycle(False,False,False)) for owner in self.owners),(),())
                    first=probe.decide(view).next_transition_ns
                    self.deadline=min(200000,max(10,first//4));self.drain=self.deadline+1
                    self.rows=[r for r in self.rows if r['arrival_ns']<self.deadline]
                    cutoffs=tuple(sorted(set([0,1,self.deadline//3,self.deadline//2,self.deadline-1])))
                baseline_engine=self.engine(seeded=seeded)
                baseline=self.run_engine(baseline_engine)
                checkpoint=None
                for cutoff in cutoffs:
                    engine=self.engine(seeded=seeded)
                    paused=self.run_engine(engine,checkpoint=checkpoint,pause_at_ns=cutoff)
                    self.assertEqual(paused['status'],'PAUSED')
                    self.assertEqual(paused['cutoff_ns'],cutoff)
                    checkpoint=json.loads(json.dumps(paused['checkpoint'],allow_nan=False))
                    state=scp.decode(checkpoint['state'])
                    self.assertEqual(set(state['cohort']),{r['request_id'] for r in self.rows})
                    self.assertFalse(checkpoint['physical_progress_observed'])
                    self.assertFalse(checkpoint['native_observation_assimilation_supported'])
                resumed_engine=self.engine(seeded=seeded)
                actual=self.run_engine(resumed_engine,checkpoint=checkpoint)
                self.assertEqual(actual,baseline)
                ready={owner:0 for owner in self.owners}
                self.assertEqual(scp._state(resumed_engine.scheduler,ready),scp._state(baseline_engine.scheduler,ready))

    def test_future_arrivals_FIFO_host_release_receipt_READY_prefix_preserved(self):
        checkpoint=self.checkpoint(500)
        state=scp.decode(checkpoint['state'])
        self.assertEqual(len(state['arrivals']),3)
        self.assertEqual(state['ready'][0][0],1000)
        checkpoint=self.run_engine(self.engine(),checkpoint=checkpoint,pause_at_ns=35000)['checkpoint']
        state=scp.decode(checkpoint['state'])
        self.assertIn('a0',state['receipt_pending'])
        self.assertEqual(state['receipt_pending']['a0']['handler_end_ns'],18000)
        self.assertEqual(state['receipt_pending']['a0']['finish_ns'],48000)
        self.assertEqual(state['owner_last_ready']['text#0'],68000)
        self.assertEqual([r['request_id'] for r in state['pending']['text#0']],['a1'])
        prefix=scp.decode(checkpoint['six_term_prefix'])
        self.assertEqual(prefix['received'],{})
        self.assertGreater(prefix['measurement_energy_joules'],0)
        self.assertEqual(prefix['original_measurement_window_ns'],[0,self.deadline])

    def test_import_is_pure_no_decide_observe_budget_or_future_result_replay(self):
        checkpoint=self.checkpoint(12000)
        engine=self.engine();before=scp._state(engine.scheduler,{owner:0 for owner in self.owners})
        contract=cp.checkpoint_contract(engine,self.rows,0,self.deadline,self.drain)
        with (patch.object(scp.LearnedLifecycleScheduler,'decide',side_effect=AssertionError('decide')),
             patch.object(scp.LearnedLifecycleOracle,'observe_query',side_effect=AssertionError('observe')),
             patch.object(scp.CumulativeLifecycleHazard,'_budget',side_effect=AssertionError('redraw'))):
            restored=cp.decode_checkpoint(engine,checkpoint,contract)
        self.assertEqual(before,scp._state(engine.scheduler,{owner:0 for owner in self.owners}))
        self.assertFalse(engine._forecast_invoked)
        self.assertEqual(restored['clock'],12000)

    def test_transactional_rejection_of_rehashed_state_contradictions(self):
        original=self.checkpoint(35000)
        def bad_ready(s):s['owner_last_ready']['text#0']=35000
        def bad_energy(s):s['energy']+=1
        def bad_release(s):s['advance']['boundary_ns']+=10
        def bad_remaining(s):next(iter(s['active'].values()))['remaining_ns']+=1
        def bad_future(s):s['finishes']['b1']=dict(s['cohort']['b1'],finish_ns=99999999)
        for mutation in (bad_ready,bad_energy,bad_release,bad_remaining,bad_future):
            with self.subTest(mutation=mutation.__name__):
                checkpoint=copy.deepcopy(original);state=scp.decode(checkpoint['state']);mutation(state)
                self.reseal(checkpoint,state);engine=self.engine()
                before=scp._state(engine.scheduler,{owner:0 for owner in self.owners})
                diagnostic=cp.diagnostic_state(engine.queue_diagnostics)
                with self.assertRaises((ValueError,KeyError)):
                    self.run_engine(engine,checkpoint=checkpoint)
                self.assertEqual(before,scp._state(engine.scheduler,{owner:0 for owner in self.owners}))
                self.assertEqual(diagnostic,cp.diagnostic_state(engine.queue_diagnostics))
                self.assertFalse(engine._forecast_invoked)
                self.run_engine(engine) # failed import did not consume destination

    def test_original_six_term_scorer_identical_and_prefix_not_double_counted(self):
        from gsdeploy.forecasting.metrics import score_forecast,percentile
        baseline=self.run_engine(self.engine())
        checkpoint=self.checkpoint(1400000)
        restored=self.run_engine(self.engine(),checkpoint=json.loads(json.dumps(checkpoint)))
        kwargs=dict(phase_start_ns=0,now_ns=0,deadline_ns=self.deadline,
            memory_budget_mib=100.,energy_reference_watts=10.,
            observed_peak_memory_mib=30.,observed_board_energy_joules=0.,memory_safe_factor=1.1)
        def score(result):
            return score_forecast(result,[],self.aliases,{'text':10.},{'text':1.},
                {'f':{'text':.2}},**kwargs)
        expected=score(baseline);actual=score(restored)
        self.assertEqual(actual,expected)
        self.assertTrue(actual['absolute_utility_complete'])
        self.assertEqual(set(actual['terms']),set('QTLPME'))
        prefix=scp.decode(checkpoint['six_term_prefix'])
        cutoff=prefix['cutoff_ns'];received=prefix['received']
        suffix=dict(restored,
            cohort={rid:row for rid,row in restored['cohort'].items() if rid not in received},
            finishes={rid:row for rid,row in restored['finishes'].items() if rid not in received},
            board_energy_joules=restored['board_energy_joules']-prefix['measurement_energy_joules'])
        history=[dict(row,completion_ns=row['finish_ns']) for row in received.values()]
        split_kwargs=dict(kwargs,now_ns=cutoff,
            observed_peak_memory_mib=max(kwargs['observed_peak_memory_mib'],prefix['memory_peak_through_cutoff_mib']),
            observed_board_energy_joules=prefix['measurement_energy_joules'])
        split=score_forecast(suffix,history,self.aliases,{'text':10.},{'text':1.},
            {'f':{'text':.2}},**split_kwargs)
        self.assertEqual(split['terms'],expected['terms'])
        self.assertEqual(split['full_U'],expected['full_U'])
        latencies=[(r['finish_ns']-r['arrival_ns'])/1e6 for r in restored['finishes'].values()]
        self.assertEqual(actual['terms']['P'],percentile(latencies))
        self.assertEqual(actual['terms']['M'],.33)
        # Unknown historical energy keeps the absolute score unknown.
        unknown=score_forecast(suffix,history,self.aliases,{'text':10.},{'text':1.},
            {'f':{'text':.2}},**dict(split_kwargs,observed_board_energy_joules=None))
        self.assertIsNone(unknown['full_U'])

    def test_pending_handoff_engine_clocks_and_budget_survive_without_redraw(self):
        paused=self.checkpoint(1400000);state=scp.decode(paused['state'])
        scheduler=scp.decode(paused['scheduler']['state'])
        self.assertIsNotNone(scheduler['scheduler']['pending_switch'])
        self.assertEqual(scheduler['scheduler']['context_events'],
            [(1419295,'RESTORE_START','text#1'),(1419295,'SERVICE_READY','text#1')])
        budget=scheduler['oracle']['hazard']['hazard_budget']
        before=scheduler['oracle']['hazard']['state']['visible_update_count']
        target=self.engine()
        next_pause=self.run_engine(target,checkpoint=paused,pause_at_ns=1400001)['checkpoint']
        after=scp.decode(next_pause['scheduler']['state'])
        self.assertEqual(after['scheduler'],scheduler['scheduler'])
        self.assertEqual(after['oracle']['hazard']['hazard_budget'],budget)
        self.assertEqual(after['oracle']['hazard']['state']['visible_update_count'],before)
        self.assertEqual(state['advance']['boundary_ns'],1419295)

    def test_scheduler_resealed_bad_budget_rejected_by_main_transaction(self):
        checkpoint=self.checkpoint(12000)
        state=scp.decode(checkpoint['scheduler']['state']);hazard=state['oracle']['hazard']
        hazard['state']['hazard_budget']+=1;hazard['hazard_budget']+=1
        hazard['state']['state_sha256']=digest({k:v for k,v in hazard['state'].items() if k!='state_sha256'})
        checkpoint['scheduler']['state']=scp.encode(state)
        checkpoint['scheduler']['checkpoint_sha256']=digest({k:v for k,v in checkpoint['scheduler'].items() if k!='checkpoint_sha256'})
        checkpoint['checkpoint_sha256']=digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})
        target=self.engine();before=scp._state(target.scheduler,{o:0 for o in self.owners})
        before_diag=cp.diagnostic_state(target.queue_diagnostics)
        with self.assertRaisesRegex(ValueError,'budget'):self.run_engine(target,checkpoint=checkpoint)
        self.assertEqual(before,scp._state(target.scheduler,{o:0 for o in self.owners}))
        self.assertEqual(before_diag,cp.diagnostic_state(target.queue_diagnostics))
        self.assertFalse(target._forecast_invoked)

    def test_atomic_commit_exception_rolls_back_all_components(self):
        target=self.engine();checkpoint=self.checkpoint(12000)
        contract=cp.checkpoint_contract(target,self.rows,0,self.deadline,self.drain)
        state=cp.decode_checkpoint(target,checkpoint,contract)
        before=scp._state(target.scheduler,{o:0 for o in self.owners})
        diag=cp.diagnostic_state(target.queue_diagnostics)
        original=type(target.queue_diagnostics).__setattr__
        attempts=[]
        def fail_first(obj,key,value):
            if key=='__dict__' and not attempts:
                attempts.append(key);raise RuntimeError('Injected commit failure')
            return original(obj,key,value)
        with patch.object(type(target.queue_diagnostics),'__setattr__',fail_first):
            with self.assertRaisesRegex(RuntimeError,'commit failure'):cp.commit_checkpoint(target,state)
        self.assertEqual(before,scp._state(target.scheduler,{o:0 for o in self.owners}))
        self.assertEqual(diag,cp.diagnostic_state(target.queue_diagnostics))
        self.assertFalse(target._forecast_invoked)

    def test_source_window_input_and_unknown_fields_are_explicitly_rejected(self):
        checkpoint=self.checkpoint(500)
        target=self.engine();target.untracked_rng=123
        with self.assertRaisesRegex(ValueError,'Untracked'):self.run_engine(target,checkpoint=checkpoint)
        del target.untracked_rng
        changed=copy.deepcopy(self.rows);changed[-1]['arrival_ns']+=1
        with self.assertRaisesRegex(ValueError,'execution contract'):
            target.forecast(changed,0,self.deadline,drain_deadline_ns=self.drain,checkpoint=checkpoint)
        with self.assertRaisesRegex(ValueError,'execution contract'):
            target.forecast(self.rows,0,self.deadline+1,drain_deadline_ns=self.drain,checkpoint=checkpoint)
        self.assertFalse(target._forecast_invoked)
        self.component.fixture.source.write_text('Changed immutable evidence')
        with self.assertRaisesRegex(ValueError,'source changed'):
            self.run_engine(target,checkpoint=checkpoint)
        self.assertFalse(target._forecast_invoked)

    def test_explicit_unsupported_native_sidecar_random_provider_and_legacy_cutoff(self):
        engine=self.engine();engine.checkpoint_binding.pop('provider_state_policy')
        with self.assertRaisesRegex(ValueError,'stateless'):self.run_engine(engine)
        engine=self.engine();engine.queue_diagnostics=object()
        with self.assertRaisesRegex(ValueError,'sidecars'):self.run_engine(engine)
        with self.assertRaisesRegex(ValueError,'exact pause'):self.run_engine(self.engine(),pause_at_or_after_ns=1000)
        with self.assertRaisesRegex(ValueError,'measurement window'):self.run_engine(self.engine(),pause_at_ns=self.deadline)
        with self.assertRaisesRegex(ValueError,'checkpoint state is not implemented'):
            self.run_engine(self.engine(old=True),checkpoint=self.checkpoint(500))


if __name__=='__main__':unittest.main()
