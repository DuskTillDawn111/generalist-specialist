"""Independent v11 adapter control path; training constructors/hardware mocked.

Real public binding, complete-score arithmetic and gate projection execute.
These fixtures test contracts, not evidence provenance or physical accuracy.
"""
import copy
import types
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_prospective_flow_contract_v11 as contract_fixtures
from tools.predictor_validation_20261027 import availability_forecast_v11 as tool
from tools.predictor_validation_20261027.profiling import digest
from tools.predictor_validation_20261027.prospective_flow_contract_v11 import gate_cell


class AvailabilityAdapterIndependentTests(unittest.TestCase):
    def setUp(self):
        fixture=contract_fixtures.ProspectiveFlowContractTests();fixture.setUp();self.fixture=fixture
        self.plan=copy.deepcopy(fixture.plan);self.plan['context']=dict(path='context',sha256='ctx')
        self.plan['execution_context_sha256']='context';self.rehash()
        self.plan_ref=dict(path='plan',sha256='plan');self.startup_ref=dict(path='startup',sha256='startup')
        self.adapter=tool.AvailabilityForecast.__new__(tool.AvailabilityForecast)
        self.adapter.configuration={k:dict(path=k,sha256=k) for k in ('parameters','availability_model','component_bundle')}
        self.adapter.aliases=copy.deepcopy(fixture.aliases);self.adapter.defaults=copy.deepcopy(fixture.defaults)
        self.adapter.bundle=dict(weights=copy.deepcopy(self.plan['weights']))
        self.adapter.source_native=dict(projection_sha256='native')
        self.adapter.source_target_context_ref=dict(path='old-context',sha256='old')
        self.checked_rows=[]
        def generation(rows):self.checked_rows.append(copy.deepcopy(rows));return dict(rows=copy.deepcopy(rows),future_actual_output_used=False)
        self.adapter.generation=types.SimpleNamespace(check=generation)
        self.adapter.lifecycle=types.SimpleNamespace(quality={'text':{'text':.2}})
        self.engine=types.SimpleNamespace(forecast=lambda *args,**kwargs:copy.deepcopy(fixture.forecast),
            scheduler=types.SimpleNamespace(provenance=lambda:dict(numeric_hypothesis=True)))
        self.engine_rows=[]
        def engine_factory(**kwargs):
            def forecast(rows,now,deadline,**options):
                self.engine_rows.append((copy.deepcopy(rows),now,deadline,copy.deepcopy(options)))
                if hasattr(self,'engine_error'):raise self.engine_error
                return copy.deepcopy(fixture.forecast)
            self.engine.forecast=forecast;return self.engine
        self.adapter.engine=engine_factory

    def rehash(self):
        self.plan.pop('plan_sha256',None);self.plan['plan_sha256']=digest(self.plan)

    def execute(self,native='native'):
        def checked(ref):return self.plan if ref['path']=='plan' else {}
        with patch.object(tool,'checked',side_effect=checked), \
                patch.object(tool,'native_execution_projection',return_value=dict(original_full_context_sha256='context',projection_sha256=native)), \
                patch.object(tool,'startup_prediction',return_value=copy.deepcopy(self.fixture.startup)):
            return self.adapter.execute(self.plan_ref,'shared',self.fixture.public,self.fixture.routes,self.startup_ref)

    def test_full_repeated_future_suffix_original_routes_and_six_terms_retained(self):
        result=self.execute()
        self.assertEqual(result['status'],'COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST')
        self.assertEqual(result['full_public_cohort'],self.fixture.rows)
        self.assertEqual(self.engine_rows,[(self.fixture.rows,0,1_000_000_000,dict(drain_deadline_ns=3_000_000_000))])
        self.assertEqual(self.checked_rows,[self.fixture.rows])
        self.assertEqual(set(result['metrics']['terms']),set('QTLPME'))
        cell=gate_cell(result,self.fixture.rows,self.plan['candidates'][0],'fresh',[0,100_000_000,110_000_000],phase_start_ns=0)
        self.assertEqual(cell['terms'],result['metrics']['terms'])
        self.assertEqual([s['depth'] for s in cell['queue_depth_samples']],[0,2,1])
        self.assertAlmostEqual(cell['terms']['E'],25/50+360/(50*3600))
        self.assertAlmostEqual(cell['terms']['M'],1.2*70/100)

    def test_changed_original_reference_budget_and_power_rejected(self):
        for field,value in [('latency_reference_ms',{'text':2000.}),('memory_budget_mib',200.),('energy_reference_watts',100.)]:
            with self.subTest(field=field):
                previous=copy.deepcopy(self.plan[field]);self.plan[field]=value;self.rehash()
                with self.assertRaises(ValueError):self.execute()
                self.plan[field]=previous;self.rehash()

    def test_unknown_engine_preserves_declared_whole_cohort_without_numeric_subset(self):
        self.engine_error=UnknownEvidence('UNSUPPORTED_EXACT_POLICY_STATE')
        result=self.execute()
        self.assertEqual(result['status'],'UNKNOWN');self.assertIn('UNSUPPORTED',result['reason'])
        self.assertEqual(result['full_public_cohort'],self.fixture.rows)
        self.assertNotIn('forecast',result);self.assertNotIn('metrics',result)
        self.assertIn('startup_prediction',result)

    def test_changed_projection_and_foreign_routes_are_rejected_before_forecast(self):
        with self.assertRaisesRegex(ValueError,'native execution'):self.execute(native='other')
        self.fixture.routes[self.fixture.rows[-1]['request_id']]='foreign'
        with self.assertRaisesRegex(ValueError,'routes'):self.execute()
        self.assertEqual(self.engine_rows,[])

    def test_drain_censored_full_forecast_remains_unknown_and_keeps_all_rows(self):
        self.fixture.forecast['finishes'].pop(self.fixture.rows[-1]['request_id'])
        result=self.execute();self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['full_public_cohort'],self.fixture.rows)
        self.assertNotIn('metrics',result)


if __name__=='__main__':unittest.main()
