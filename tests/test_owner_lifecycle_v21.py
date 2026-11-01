import copy
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_compiled_static_providers_v16 import CompilationTests, Lifecycle, Resources, Scheduler
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode, SchedulingDecision
from tools.predictor_validation_20261027.dag_frozen_model_v10 import plain
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleDagPhaseModel, LifecycleSchedulingView
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import (
    RESIDENTS, SCHEMA, OwnerLifecycle, engine_states, features, feature_names,
    probabilities, state_features, state_signature, LifecycleSchedulerBridge)
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import receipt
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import IntervalIndex,states_at
from tools.predictor_validation_20261027.owner_lifecycle_hazard_v21 import lifecycle_features
from tools.predictor_validation_20261027.owner_lifecycle_hazard_v24_fit import (
    SCHEMA as HAZARD_SCHEMA,EDGES_NS)
from tools.predictor_validation_20261027.owner_lifecycle_hazard_query_v24 import LifecycleSaveHazard


class MarkTests(unittest.TestCase):
    def setUp(self):
        self.pools = {'short':'text'}
        self.query = dict(held_owner='text#1',context_age_ns=10,
            pending_frontier=[['text#0','text','short',['prefill']]],
            previous_held_owner=None, public_workload=None,
            pending_age_ns_by_owner=None,pending_reference_work_ns_by_owner=None)
        self.states = {owner:OwnerLifecycle(False,False,False) for owner in RESIDENTS}
        self.states['text#0'] = OwnerLifecycle(True,True,True)
        names = feature_names(self.pools)
        coefficients = np.zeros((len(names),len(RESIDENTS)))
        coefficients[0,0] = 10.  # Raw physical G2 probability intentionally large.
        self.model = dict(schema=SCHEMA,pool_tasks=self.pools,feature_names=names,
            lifecycle_support_signatures=[state_signature(self.query,self.states)],
            fitted=dict(center=[0.]*len(names),scale=[1.]*len(names),
                coefficients=coefficients.tolist(),feature_minimum=[-100.]*len(names),
                feature_maximum=[100.]*len(names)))

    def test_unoffered_idle_owner_not_active_service_but_physical_class_retained(self):
        value = probabilities(self.model,self.query,self.states)
        self.assertGreater(value['physical_context_owner_probabilities']['generalist_2#0'],.99)
        self.assertEqual(value['active_service_target'],'text#0')
        self.assertEqual(value['active_service_owner_probabilities']['generalist_2#0'],0.)
        self.assertEqual(set(value['physical_context_owner_probabilities']),set(RESIDENTS))

    def test_handler_without_ready_kernel_is_eligible(self):
        self.states['text#0'] = OwnerLifecycle(True,None,False)
        self.query['pending_frontier'] = []
        value = probabilities(self.model,self.query,self.states)
        self.assertEqual(value['active_service_target'],'text#0')
        self.assertIn('HELD_FRONTIER_HISTORY_LIFECYCLE_COMBINATION_UNSEEN',value['ood_flags'])

    def test_unknown_work_is_unknown_and_all_idle_does_not_fabricate_initial_holder(self):
        self.states['generalist_2#0'] = OwnerLifecycle(None,None,False)
        value = probabilities(self.model,self.query,self.states)
        self.assertEqual(value['status'],'UNKNOWN_OWNER_WORK_DOMAIN')
        self.assertIsNone(value['active_service_target'])
        states = {owner:OwnerLifecycle(False,False,False) for owner in RESIDENTS}
        value = probabilities(self.model,dict(self.query,pending_frontier=[]),states)
        self.assertEqual(value['status'],'NO_VISIBLE_ACTIVE_SERVICE_TARGET')
        self.assertFalse(value['initial_residual_context_modelled'])
        self.assertFalse(value['background_context_modelled'])

    def test_future_public_arrivals_do_not_change_features(self):
        before = features(self.query,self.states,self.pools)
        changed = dict(self.query,public_workload=[['text#0','text','short',9999999]])
        self.assertEqual(before,features(changed,self.states,self.pools))
        with self.assertRaises(ValueError):
            features(dict(self.query,next_owner='text#0'),self.states,self.pools)

    def test_typed_tristate_and_inconsistent_issue_state_rejected(self):
        with self.assertRaises(ValueError):OwnerLifecycle(1,False,False)
        with self.assertRaises(ValueError):OwnerLifecycle(True,False,True)
        with self.assertRaises(ValueError):state_features({})

    def test_frontier_lifecycle_contradiction_rejected(self):
        self.states['text#0']=OwnerLifecycle(True,None,False)
        with self.assertRaisesRegex(ValueError,'disagree'):
            probabilities(self.model,self.query,self.states)

    def test_future_handler_END_value_cannot_change_projected_prefix_state(self):
        def index(future_end):
            empty=lambda:{owner:IntervalIndex([]) for owner in RESIDENTS}
            value={key:empty() for key in ('phase','certain','possible','issued')}
            for key in ('certain','possible'):
                value[key]['text#0']=IntervalIndex([(10,future_end)])
            return value
        # A future END=101 or END=1000 both have the same emitted prefix at100.
        first=states_at(index(101),100,[])
        second=states_at(index(1000),100,[])
        self.assertEqual(first,second)
        self.assertTrue(first['text#0'].handler_active)
        self.assertFalse(states_at(index(101),101,[])['text#0'].handler_active)

    def test_engine_does_not_claim_unreleased_DAG_is_issued(self):
        states = engine_states(RESIDENTS,{'text#0':{}},{})
        self.assertTrue(states['text#0'].handler_active)
        self.assertIsNone(states['text#0'].issued_outstanding)
        self.assertFalse(states['text#0'].gpu_ready)
        self.assertTrue(states['text#0'].active_service_eligible)
        self.assertFalse(states['generalist_2#0'].active_service_eligible)

    def test_empty_frontier_hazard_has_same_tristate_features_as_mark(self):
        self.states['text#0']=OwnerLifecycle(True,None,False)
        names=lifecycle_features('text#1',[],self.states)
        self.assertIn('frontier:empty',names)
        self.assertIn('lifecycle:text#0:handler_active:true',names)
        self.assertIn('lifecycle:text#0:issued_outstanding:unknown',names)
        self.assertIn('lifecycle:generalist_2#0:handler_active:false',names)

    def test_hazard_query_uses_lifecycle_and_empty_frontier_without_hard_zero(self):
        states={owner:OwnerLifecycle(False,False,False) for owner in RESIDENTS}
        states['text#0']=OwnerLifecycle(True,None,False)
        from tools.predictor_validation_20261027.owner_lifecycle_hazard_v24_fit import lifecycle_features as fs
        vocabulary=set(fs('text#0',[],states))
        changed=dict(states)
        changed['generalist_2#0']=OwnerLifecycle(True,None,False)
        vocabulary.update(fs('text#0',[],changed))
        columns=['intercept',*sorted(vocabulary),*[f'age_bin:{i}' for i in range(len(EDGES_NS))]]
        coefficients=[0.]*len(columns)
        coefficients[columns.index('competition:present')]=1.
        hazard=LifecycleSaveHazard(dict(schema=HAZARD_SCHEMA,fixed_age_edges_ns=list(EDGES_NS),
            columns=columns,coefficients=coefficients,training_vocabulary=sorted(vocabulary),
            age_support_ns=[[0,100000000]]))
        solo=hazard.query('text#0',[],states,100,100)
        contender=hazard.query('text#0',[],changed,100,100)
        self.assertGreater(solo['hazard_per_ms'],0.)
        self.assertGreater(contender['hazard_per_ms'],solo['hazard_per_ms'])
        self.assertIn('ZERO_OBSERVED_EVENTS_DOES_NOT_CERTIFY_ZERO_HAZARD',solo['OOD_reasons'])
        self.assertAlmostEqual(contender['hazard_per_ms'],hazard.query('text#0',[],changed,200)['hazard_per_ms'])
        self.assertFalse(solo['empty_frontier_zero_hazard_assumed'])
        with self.assertRaisesRegex(UnknownEvidence,'FRONTIER_MISMATCH'):
            hazard.query('text#0',[['text#0','text','short',['prefill']]],states,100)

    def test_bridge_observes_current_view_before_controller_and_freezes_partial_state(self):
        class Mark:
            model={'model_sha256':'fixture'}
            def observe_view(self,view):self.clock=view.clock_ns
        mark=Mark()
        class Controller:
            oracle=type('Oracle',(),{'next_owner':mark})()
            def decide(self,view):
                if mark.clock!=view.clock_ns:raise AssertionError('stale lifecycle')
                return SchedulingDecision({node.key:1. for node in view.active_gpu_nodes})
            def provenance(self):return {}
        bridge=LifecycleSchedulerBridge(Controller(),mark)
        node=ActiveGpuNode(('text#0','g'),3.,'prefill','r','text','short')
        view=LifecycleSchedulingView(7,(node,),tuple(sorted(self.states.items())),(('text#0','r',1),),())
        bridge.decide(view)
        frozen=bridge.freeze_observation_state()
        self.assertFalse(frozen['is_executable_checkpoint'])
        frozen['owner_lifecycle']['text#0']['handler_active']=False
        self.assertTrue(dict(bridge.last_view.owner_lifecycle)['text#0'].handler_active)
        with self.assertRaises(FrozenInstanceError):view.clock_ns=9


class EngineTests(unittest.TestCase):
    setUp = CompilationTests.setUp
    old_engine = CompilationTests.engine

    def provider(self,values):
        def call(node,row):
            original=next(n for n in self.plans[row['alias'],row['task'],row['pool_id']]['nodes'] if n['id']==node['id'])
            self.assertEqual(plain(node),plain(original))
            return copy.deepcopy(values[row['alias'],row['task'],row['pool_id'],node['id']])
        return call

    def engine(self,scheduler=None,**kwargs):
        return LifecycleDagPhaseModel(self.aliases,self.plans,
            duration_provider=self.provider(self.work),release_provider=self.provider(self.releases),
            phase_provider=self.provider(self.phases),scheduler=scheduler or Scheduler(),
            lifecycle_provider=Lifecycle(),power_provider=Resources(),memory_provider=Resources(),
            require_energy=True,require_memory=True,
            allow_independent_copy_hypothesis=True,allow_no_implicit_default_stream_hypothesis=True,**kwargs)

    def rows(self):
        return [dict(alias=owner,task='text',pool_id='short',request_id=f'{owner}-{arrival}',arrival_ns=arrival)
            for arrival in (0,4,9,17) for owner in self.aliases]

    def test_lifecycle_hook_preserves_all_numerical_results(self):
        for drain in (40,1000):
            old=self.old_engine().forecast(self.rows(),0,30,drain_deadline_ns=drain)
            new=self.engine().forecast(self.rows(),0,30,drain_deadline_ns=drain)
            self.assertTrue(new.pop('owner_lifecycle_snapshot_supported'))
            self.assertFalse(new.pop('issued_dependency_queue_complete'))
            self.assertEqual(old,new)

    def test_handler_and_queue_visible_during_empty_GPU_frontier(self):
        seen=[]
        class Observe(Scheduler):
            def decide(self,view):
                seen.append(view)
                return super().decide(view)
        self.engine(Observe()).forecast(self.rows(),0,30,drain_deadline_ns=1000)
        first=seen[0]
        self.assertEqual(first.clock_ns,0)
        self.assertFalse(first.active_gpu_nodes)
        self.assertTrue(all(state.handler_active for _,state in first.owner_lifecycle))
        self.assertTrue(all(state.issued_outstanding is None for _,state in first.owner_lifecycle))
        self.assertTrue(any(ids for view in seen for _,ids in view.pending_requests))
        self.assertEqual({rid for _,rid,_ in first.handler_requests},{'a-0','b-0'})

    def test_future_owner_not_treated_as_having_handler(self):
        seen=[]
        class Observe(Scheduler):
            def decide(self,view):seen.append(view);return super().decide(view)
        rows=[dict(alias='a',task='text',pool_id='short',request_id='a-now',arrival_ns=0),
              dict(alias='b',task='text',pool_id='short',request_id='b-later',arrival_ns=20)]
        self.engine(Observe()).forecast(rows,0,30,drain_deadline_ns=1000)
        for view in seen:
            if view.clock_ns<20:
                self.assertFalse(dict(view.owner_lifecycle)['b'].active_service_eligible)
                self.assertFalse(dict(view.pending_requests)['b'])

    def test_request_start_delay_and_nonreusable_engine(self):
        rows=self.rows()[:1]
        engine=self.engine(request_start_provider=lambda row:row['arrival_ns']+10)
        value=engine.forecast(rows,0,30,drain_deadline_ns=1000)
        self.assertEqual(value['starts']['a-0'],10)
        with self.assertRaises(ValueError):engine.forecast(rows,0,30)
        with self.assertRaises(ValueError):self.engine().forecast(rows,0,30,checkpoint={})

    def test_direct_byte_hash_detects_same_size_source_change_even_with_mtime_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'provider.json'
            path.write_text('one')
            stat=path.stat()
            engine=self.engine(source_receipts=[receipt(path)])
            path.write_text('two')
            os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaisesRegex(UnknownEvidence,'SOURCE_BYTES_CHANGED'):
                engine.forecast(self.rows(),0,30,drain_deadline_ns=1000)


if __name__=='__main__':unittest.main()
