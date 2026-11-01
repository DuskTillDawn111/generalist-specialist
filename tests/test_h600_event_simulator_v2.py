"""Selected completion events avoid sub-ULP fixed points at long horizons."""
import math
import unittest
from tests import test_full_horizon_forecast as fixtures
from tools.predictor_validation_20261027.h600_event_simulator_v2 import FullHorizonModel
from tools.predictor_validation_20261027 import h600_event_predict_v2
from tools.predictor_validation_20261027 import flow_crossfit_report
from tests import test_flow_validation


class EventV2Tests(unittest.TestCase):
    def model(self,source=None,**kwargs):return FullHorizonModel(fixtures.aliases(),source or fixtures.provider(),**kwargs)

    def test_actual_minimal_float_reproduction_has_selected_rounded_completion(self):
        clock=571791700004.1077;rate=.4716724721000537;fraction=1.148153794261475e-10
        duration=fraction*rate*1e6;boundary=clock+duration
        self.assertGreater(fraction,1e-10);self.assertLess(duration,math.ulp(clock)/2)
        self.assertEqual(boundary,clock);self.assertEqual(fraction-(boundary-clock)/(rate*1e6),fraction)
        self.assertTrue(clock+duration<=boundary)

    def long_fixture(self,gated=False):
        source=fixtures.provider(s_ms=.4716724721000537,g_ms=.4716724721000537,speed=lambda alias,active:1.)
        # First stage's duration places the finalize at the actual problematic clock.
        class TwoStage:
            def plan(self,row,aliases,scenario):
                return {'profile_id':row['alias'],'segments':[
                    {'id':'prefill','kind':'host','duration_ms':571791.2283316356,'demand':{}},
                    {'id':'finalize','kind':'host','duration_ms':.4716724721000537,'demand':{}}]}
            def service(self,row,active,aliases,scenario):
                rate=571791.2283316356 if row['segment_id']=='prefill' else .4716724721000537
                return {'known':True,'context_supported':True,'service_ms':rate}
            def power(self,*args):return {'known':True,'watts':100.}
            def memory(self,*args):return {'known':True,'peak_memory_mib':200.}
        return self.model(TwoStage(),resources=fixtures.resources(gated),max_events=20)

    def test_long_600s_finalize_progresses_without_budget_change_or_fraction_inflation(self):
        result=self.long_fixture().forecast({'s':[fixtures.request('r')]},0,600*fixtures.SECOND)
        self.assertEqual(result['incomplete_ids'],[]);self.assertLess(result['transition_count'],5)
        self.assertEqual(result['finishes']['r']['finish_ns'],round(571791700004.1077))

    def test_ordinary_forecast_and_timing_resource_accounting_equal_frozen_engine(self):
        source=fixtures.provider(ordered=True)
        args=dict(future_arrivals=[fixtures.request('g',arrival_ns=25*fixtures.MS)],route_future=lambda *args:'g')
        before=fixtures.FullHorizonModel(fixtures.aliases(),source).forecast({'s':[fixtures.request('s')]},0,fixtures.SECOND,**args)
        after=self.model(source).forecast({'s':[fixtures.request('s')]},0,fixtures.SECOND,**args)
        self.assertEqual(after,before)

    def test_arrival_window_drain_and_two_alias_completion_ties(self):
        source=fixtures.provider(s_ms=100.,g_ms=100.,speed=lambda alias,active:1.)
        result=self.model(source).forecast({'s':[fixtures.request('s')],'g':[fixtures.request('g')]},
            0,100*fixtures.MS,drain_deadline_ns=100*fixtures.MS,
            future_arrivals=[fixtures.request('next',arrival_ns=50*fixtures.MS)],route_future=lambda *args:'s')
        self.assertEqual(result['finishes']['s']['finish_ns'],100*fixtures.MS)
        self.assertEqual(result['finishes']['g']['finish_ns'],100*fixtures.MS)
        self.assertEqual(result['incomplete_ids'],['next'])
        result=self.model(source).forecast({'s':[fixtures.request('s')]},0,200*fixtures.MS,
            drain_deadline_ns=200*fixtures.MS,future_arrivals=[fixtures.request('next',arrival_ns=100*fixtures.MS)],
            route_future=lambda *args:'s')
        self.assertEqual(result['finishes']['next']['start_ns'],100*fixtures.MS)
        self.assertEqual(result['finishes']['next']['finish_ns'],200*fixtures.MS)

    def test_admission_permit_releases_at_long_clock_completion_and_does_not_deadlock(self):
        result=self.long_fixture(gated=True).forecast({'s':[fixtures.request('s')],'g':[fixtures.request('g')]},
            0,1200*fixtures.SECOND,drain_deadline_ns=1201*fixtures.SECOND)
        self.assertEqual(result['incomplete_ids'],[]);self.assertEqual(set(result['finishes']),{'s','g'})
        self.assertEqual(result['finishes']['s']['start_ns'],result['finishes']['g']['finish_ns'])

    def test_predict_helper_schema_equivalence_and_explicit_numerics_receipt(self):
        f=test_flow_validation.FlowValidationTests('test_full_arrivals_parent_receipt_and_wait_service_separated')
        f.setUp();self.addCleanup(f.doCleanups)
        model=flow_crossfit_report.read(f.model_path);resource=flow_crossfit_report.read(f.resources_path)
        quality=flow_crossfit_report.read(f.quality_path)
        old=flow_crossfit_report.predict(f.plan,f.plan['candidates'][0],model,resource,quality)
        new=h600_event_predict_v2.predict(f.plan,f.plan['candidates'][0],model,resource,quality)
        for key in old:self.assertEqual(old[key],new[key])
        self.assertEqual(new['forecast_numerics'],'selected_completion_event_v2')
        self.assertEqual(new['original_event_budget'],100000)
        self.assertTrue(new['numerics_source']['path'].endswith('h600_event_simulator_v2.py'))


if __name__=='__main__':unittest.main()
