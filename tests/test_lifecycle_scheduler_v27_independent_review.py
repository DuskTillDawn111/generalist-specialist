"""Actual engine-clock regression for lifecycle-only hazard advancement."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tests import test_owner_lifecycle_v21 as mark_tests
from tests import test_learned_lifecycle_forecast_v25 as factory_tests

from tools.predictor_validation_20261027 import learned_lifecycle_forecast_v27 as factory
from tools.predictor_validation_20261027 import owner_lifecycle_mark_v21 as mark_module
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.held_context_scheduler_v14 import InitialContextDecision
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources
from tools.predictor_validation_20261027.learned_lifecycle_scheduler_v27 import LearnedLifecycleOracle,LearnedLifecycleScheduler
from tools.predictor_validation_20261027.owner_lifecycle_engine_v21 import LifecycleDagPhaseModel
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import RESIDENTS,receipt
from tools.predictor_validation_20261027.profiling import digest


class Rate:
    def query(self,holder,frontier,states,age,horizon_ns=0):
        return dict(hazard_per_ms=10. if states['text#1'].handler_active else 1.,OOD_reasons=[])


class Mark:
    def __init__(self):self.views=[];self.calls=[]
    def observe_view(self,view):self.views.append(view)
    def predict_given_save_now(self,**query):
        self.calls.append(query)
        return dict(most_likely_next_owner='text#1',ood_flags=[])


class SchedulerIndependentTests(unittest.TestCase):
    def test_real_DAG_engine_visits_earlier_SAVE_with_unchanged_GPU_frontier(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'training';source.write_text('CPU fixture')
            controller=CumulativeLifecycleHazard(Rate(),age_edges_ns=(0,100_000,1_000_000),
                sources=[receipt(source)],candidate_identity='a'*64,allow_unvalidated_approximation=True)
            mark=Mark()
            oracle=LearnedLifecycleOracle(initial_prior=None,save_controller=controller,next_owner=mark,
                handoff=SimpleNamespace(query=lambda **q:dict(save_to_restore_gap_ns=2,service_ready_delay_after_restore_ns=0,ood_flags=[])),
                public_setup={},public_workload=[],source_identity='a'*64,initial_quantile=.5,
                initial_readiness_rule='lower_bound_hypothesis',handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
                allow_initial_pooling=True,allow_observer_transfer=True,allow_flagged_statistical_extrapolation=True)
            oracle.initialize=lambda query:InitialContextDecision('text#0',0,0,'a'*64)
            scheduler=LearnedLifecycleScheduler(RESIDENTS,oracle=oracle)
            aliases={o:dict(supported_tasks=['text']) for o in RESIDENTS}
            plans={(o,'text','known'):dict(nodes=[dict(id='g',kind='gpu'),dict(id='end',kind='handler_end')],
                edges=[dict(before='g',after='end')],default_stream_semantics='EXPLICIT') for o in RESIDENTS}
            engine=LifecycleDagPhaseModel(aliases,plans,scheduler=scheduler,
                duration_provider=lambda *args:2_000_000,
                release_provider=lambda node,row:dict(not_before_offset_ns=500_000 if row['alias']=='text#1' else 0,after_predecessors_ns=0),
                phase_provider=lambda *args:'decode')
            rows=[dict(request_id='first',alias='text#0',task='text',pool_id='known',arrival_ns=0),
                  dict(request_id='host-arrival',alias='text#1',task='text',pool_id='known',arrival_ns=100_000)]
            result=engine.forecast(rows,0,1_000_000,drain_deadline_ns=200_000)
            saves=[x for x in scheduler.transitions if x['kind']=='SAVE_END']
            self.assertEqual([x['clock_ns'] for x in saves],[159_315])
            self.assertEqual(result['starts'],{'first':0,'host-arrival':100_000})
            self.assertEqual(result['finishes'],{})
            self.assertEqual(len(mark.calls),1)
            self.assertEqual(mark.calls[0]['context_age_ns'],159_315)
            initial=next(v for v in mark.views if v.clock_ns==0)
            changed=next(v for v in mark.views if v.clock_ns==100_000)
            self.assertEqual([n.key for n in initial.active_gpu_nodes],[n.key for n in changed.active_gpu_nodes])
            self.assertTrue(dict(changed.owner_lifecycle)['text#1'].handler_active)
            self.assertFalse(dict(changed.owner_lifecycle)['text#1'].gpu_ready)

    def test_static_closure_covers_every_parent_initializer_of_bound_module(self):
        refs=implementation_sources(factory.__file__)
        paths={Path(x['path']).resolve() for x in refs};root=Path(factory.__file__).resolve().parents[2]
        parents=set()
        for path in paths:
            for parent in path.parents:
                if parent==root:break
                initializer=parent/'__init__.py'
                if initializer.is_file():parents.add(initializer)
        self.assertLessEqual(parents,paths)
        for name in ('learned_availability_forecast_v20.py','idle_owner_selection_audit_v20.py'):
            self.assertIn(Path(factory.__file__).resolve().parent/name,paths)
        self.assertTrue(all(receipt(x['path'])==x for x in refs))

    def test_actual_factory_rejects_stale_training_identity_before_engine_construction(self):
        # Replace only the expensive inherited provider initialization. The new
        # factory's model reads, real mark validation and receipt merge all run.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'training';source.write_text('fixture')
            base=mark_tests.MarkTests();base.setUp()
            mark_model=dict(base.model,training_only=True,implementation=receipt(mark_module.__file__),sources=[receipt(source)])
            mark_model['model_sha256']=digest(mark_model)
            stale=dict(mark_model['implementation'],sha256='0'*64)
            def save(name,value):
                path=root/name;path.write_text(json.dumps(value));return receipt(path)
            stats=save('stats.json',dict(missing_marked_SAVE_labels=[],original_marked_SAVE_labels=1,
                original_marked_SAVE_labels_preserved=1,source_receipts=[stale]))
            hazard=save('hazard.json',dict(schema='gsdeploy.full_context_lifecycle_SAVE_hazard.v24',
                fixed_age_edges_ns=list(factory.EDGES_NS),columns=['intercept'],coefficients=[0.],
                source_statistics=stats,training_only=True,training_vocabulary=[],age_support_ns=[[0,1000]]))
            config=dict(schema=factory.SCHEMA,base_v15_configuration={},lifecycle_mark_model=save('mark.json',mark_model),
                lifecycle_hazard_model=hazard,options=dict(factory.OPTIONS_V25))
            def inherited(model,*args,**kwargs):model.compiled=SimpleNamespace(sources=())
            with patch.object(factory.V16Forecast,'__init__',inherited):
                with self.assertRaisesRegex(ValueError,'Conflicting implementation or training source'):
                    factory.LearnedLifecycleForecast(config,allow_unvalidated_training_hypothesis=True)

    def test_v27_actual_factory_engine_preserves_complete_small_cohort(self):
        fixture=factory_tests.FactoryEngineTests('test_actual_factory_engine_receives_settled_lifecycle_and_future_work')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.model.__class__=factory.LearnedLifecycleForecast
        engine,result=fixture.run_engine(fixture.rows(),20_000_000)
        self.assertIsInstance(engine.scheduler,LearnedLifecycleScheduler)
        self.assertEqual(set(result['cohort']),{'0','1'})
        self.assertEqual(set(result['finishes']),{'0','1'})
        self.assertEqual(result['finishes']['0']['finish_ns'],3_000_000)
        self.assertEqual(result['finishes']['1']['finish_ns'],8_000_000)
        self.assertEqual(result['queue_diagnostics']['full_public_request_count'],2)
        self.assertIs(engine.last_returned_forecast,result)


if __name__=='__main__':unittest.main()
