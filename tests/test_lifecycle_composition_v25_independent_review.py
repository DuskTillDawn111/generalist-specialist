"""Independent analytic and real small-engine checks for the v24/v25 bridge."""
import math
from pathlib import Path
import tempfile
import unittest

from tests import test_learned_lifecycle_forecast_v25 as factory_tests
from tests import test_owner_lifecycle_v21 as mark_tests
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027 import owner_lifecycle_mark_v21 as mark
from tools.predictor_validation_20261027.profiling import digest


class Rate:
    def query(self,holder,frontier,states,age,horizon_ns=0):
        # A new handler can change hazard with no GPU frontier change.
        r=4. if states['text#1'].handler_active else 1.
        if age>=200_000:r*=2
        return dict(hazard_per_ms=r,OOD_reasons=[])


class IndependentIntegralTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        path=Path(self.temp.name)/'training';path.write_text('fixture')
        self.source=mark.receipt(path)
        self.low={owner:mark.OwnerLifecycle(False,False,False) for owner in mark.RESIDENTS}
        self.low['text#0']=mark.OwnerLifecycle(True,None,False)
        self.high=dict(self.low);self.high['text#1']=mark.OwnerLifecycle(True,None,False)

    def controller(self,**kw):
        return CumulativeLifecycleHazard(Rate(),age_edges_ns=(0,200_000,1_000_000),
            sources=[self.source],candidate_identity='a'*64,allow_unvalidated_approximation=True,**kw)

    def test_lifecycle_age_and_empty_intervals_match_manual_integral(self):
        c=self.controller()
        state=c.start_restore('text#0',0,[],'r',self.low)
        after=c.update(state['state'],100_000,[],self.high)
        # H=.1 then .4 until age200000, and rate8 thereafter.
        crossing=200_000+math.ceil((math.log(2)-.5)/8*1e6)
        self.assertAlmostEqual(after['consumed_hazard'],.1)
        self.assertEqual(after['absolute_save_candidate_ns'],crossing)
        boundary=c.update(after['state'],200_000,[],self.high)
        self.assertAlmostEqual(boundary['consumed_hazard'],.5)
        terminal=c.update(boundary['state'],crossing,[],self.low)
        self.assertEqual(terminal['status'],'SAVE_BUDGET_REACHED')
        self.assertEqual(terminal['state']['lifecycle'],boundary['state']['lifecycle'])
        self.assertEqual(terminal['origin_restore_ns'],0)
        self.assertEqual(terminal['hazard_budget'],state['hazard_budget'])

    def test_left_truncation_retains_age_and_conditions_on_survival(self):
        c=self.controller()
        value=c.start_restore('text#0',-1_000_000,[],'old',self.low,observation_ns=0)
        self.assertEqual(value['restore_age_ns'],1_000_000)
        self.assertEqual(value['consumed_hazard'],0.)
        self.assertEqual(value['absolute_save_candidate_ns'],math.ceil(math.log(2)/2*1e6))

    def test_seeded_budget_is_one_draw_despite_repeated_updates(self):
        c=self.controller(budget_policy='SEEDED_UNIFORM',seed=123)
        calls=[];original=c._budget
        def budget(key):calls.append(key);return original(key)
        c._budget=budget
        value=c.start_restore('text#0',0,[],'draw',self.low)
        initial=value['hazard_budget']
        for t in range(0,1001,10):
            value=c.update(value['state'],t,[],self.low if t%20 else self.high)
            self.assertEqual(value['hazard_budget'],initial)
        self.assertEqual(calls,['draw'])


class FreshMarkTests(unittest.TestCase):
    def test_two_actual_factory_engines_start_with_distinct_real_marks(self):
        fixture=factory_tests.FactoryEngineTests('test_actual_factory_engine_receives_settled_lifecycle_and_future_work')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        base=mark_tests.MarkTests();base.setUp()
        model=dict(base.model,training_only=True,implementation=mark.receipt(mark.__file__),sources=[fixture.ref])
        model['model_sha256']=digest(model)
        # Remove the author's observe-only stub so the real clock guard runs.
        del fixture.model._new_mark
        fixture.model._mark_model=model
        _,first=fixture.run_engine(fixture.rows(),20_000_000)
        first_mark=fixture.model.mark
        self.assertGreater(first_mark.current_clock_ns,0)
        _,second=fixture.run_engine(fixture.rows(),20_000_000)
        self.assertIsNot(first_mark,fixture.model.mark)
        self.assertEqual(first,second)
        self.assertIsNot(first_mark.model,fixture.model.mark.model)


if __name__=='__main__':unittest.main()
