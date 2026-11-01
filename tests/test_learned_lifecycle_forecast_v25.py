"""Run actual v25 factory engine on CPU DAGs without expensive model fitting."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.compiled_static_providers_v16 import CompiledStaticProviders
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.held_context_scheduler_v14 import InitialContextDecision
from tools.predictor_validation_20261027.learned_lifecycle_forecast_v25 import LearnedLifecycleForecast
from tools.predictor_validation_20261027.learned_lifecycle_oracle_v25 import LearnedLifecycleOracle
from tools.predictor_validation_20261027.learned_availability_forecast_v20 import attach_unscored_diagnostic
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import RESIDENTS


class SlowHazard:
    def query(self,*args):
        return dict(hazard_per_ms=1e-9,OOD_reasons=['SYNTHETIC_SLOW_RATE'])


class ObserveOnlyMark:
    def __init__(self):self.views=[]
    def observe_view(self,view):self.views.append(view)


class FactoryEngineTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        source=Path(tmp.name)/'toy-training';source.write_text('synthetic')
        self.ref=receipt(source)
        aliases={owner:dict(family='text',supported_tasks=['text']) for owner in RESIDENTS}
        plans={(owner,'text','known'):dict(nodes=[dict(id='gpu',kind='gpu'),dict(id='end',kind='handler_end')],
            edges=[dict(before='gpu',after='end')],default_stream_semantics='EXPLICIT') for owner in RESIDENTS}
        compiled=CompiledStaticProviders(aliases,plans,duration=lambda *a:2_000_000,
            release=lambda *a:dict(not_before_offset_ns=0,after_predecessors_ns=0),
            phase=lambda *a:'decode',sources=[self.ref])
        model=LearnedLifecycleForecast.__new__(LearnedLifecycleForecast)
        model.aliases=aliases;model.compiled=compiled
        model._execution_binding={},[['text#0','text','known',0]]
        model.initial_prior=None;model.handoff=None;model.mark=ObserveOnlyMark()
        model._new_mark=ObserveOnlyMark
        model.lifecycle_identity='a'*64;model._lifecycle_sources=(self.ref,)
        model.save_controller=CumulativeLifecycleHazard(SlowHazard(),age_edges_ns=(0,1_000_000),
            sources=[self.ref],candidate_identity='b'*64,allow_unvalidated_approximation=True)
        model.lifecycle=SimpleNamespace(delays=lambda *a:dict(known=True,parent_receipt_ms=1.,worker_next_ready_ms=2.))
        model.resources=SimpleNamespace(power=lambda *a:dict(known=True,watts=100.),
            memory=lambda *a:dict(known=True,peak_memory_mib=200.))
        self.model=model

    def run_engine(self,rows,drain):
        engine=self.model.engine(max_events=10000)
        with patch.object(LearnedLifecycleOracle,'initialize',return_value=InitialContextDecision('text#0',0,0,'a'*64)):
            result=engine.forecast(rows,0,10_000_000,drain_deadline_ns=drain)
        return engine,result

    def rows(self):
        return [dict(request_id=str(i),alias='text#0',task='text',pool_id='known',arrival_ns=t)
                for i,t in enumerate((0,5_000_000))]

    def test_actual_factory_engine_receives_settled_lifecycle_and_future_work(self):
        engine,result=self.run_engine(self.rows(),20_000_000)
        self.assertEqual(set(result['finishes']),{'0','1'})
        self.assertEqual(result['finishes']['0']['finish_ns'],3_000_000)
        self.assertEqual(result['finishes']['1']['finish_ns'],8_000_000)
        self.assertEqual(result['queue_diagnostics']['full_public_request_count'],2)
        self.assertEqual(result['queue_diagnostics']['remaining_activity_work_ms'],0.)
        first=self.model.mark.views[0]
        self.assertEqual(dict(first.owner_lifecycle)['text#0'].handler_active,True)
        self.assertFalse(dict(first.owner_lifecycle)['generalist_2#0'].active_service_eligible)
        self.assertIs(engine.last_returned_forecast,result)
        self.assertIsNotNone(engine.sidecar)

    def test_partial_actual_engine_is_retained_only_as_diagnostic(self):
        engine,result=self.run_engine(self.rows(),1_000_000)
        self.assertEqual(len(result['cohort']),2)
        self.assertEqual(result['finishes'],{})
        refused=dict(status='UNKNOWN',reason='COMPLETE_DECLARED_FLOW_DAG_AND_PARENT_RECEIPTS_REQUIRED')
        attach_unscored_diagnostic(refused,engine)
        self.assertIs(refused['unscored_forecast_diagnostic']['forecast'],result)
        self.assertEqual(refused['status'],'UNKNOWN')
        self.assertNotIn('metrics',refused)
        self.assertEqual(result['queue_diagnostics']['not_yet_registered_request_ids'],['1'])


if __name__=='__main__':unittest.main()
