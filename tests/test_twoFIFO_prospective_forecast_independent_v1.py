import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest

from tests import test_twoFIFO_prospective_forecast_v1 as author
from tools.predictor_validation_20261027 import twoFIFO_prospective_forecast_v1 as m


class IndependentProspectiveTests(unittest.TestCase):
    def test_unequal_world_mass_retained_without_conditional_utility(self):
        plan,objective,draws,quality=author.numeric_fixture()
        worlds=copy.deepcopy(draws[:2])
        worlds[0].update(weight=Fraction(1,3),inputs=m.ensemble.UnknownForecastInputs('missing-first'))
        worlds[1]['weight']=Fraction(2,3)
        result=m.decode(m.forecast_compiled(plan,objective,request_id_map=author.fresh(plan),
            forecast_origin_ns=100,settled_seed=author.seed(),quality=quality,
            draws_by_candidate={c['id']:copy.deepcopy(worlds) for c in plan['candidates']}))
        for candidate in result['candidates'].values():
            self.assertEqual(candidate['world_unknown_probability_mass'],Fraction(1,3))
            self.assertEqual([d['weight'] for d in candidate['draws']],[Fraction(1,3),Fraction(2,3)])
            self.assertIsNone(candidate['utility']['expected_value'])
            self.assertEqual(len(candidate['public_requests']),2)
            self.assertEqual(len(candidate['draws'][0]['all_public_request_ids_retained']),2)
            self.assertEqual(candidate['draws'][1]['forecast']['world']['offered_requests'],2)
        self.assertEqual(result['paired_candidate_ranking']['winner_status'],'UNKNOWN')

    def test_original181_full_law_keeps12_worlds_and_absolute_future_calendar(self):
        plan=json.loads(Path('artifacts/predictor-gpu-20261027/flow-validation-v13-ocr-audited/plan.json').read_text())
        objective=json.loads(Path('configs/resource_spillover_objective_20261024.json').read_text())
        before=copy.deepcopy(plan);epoch=1_000_000_001
        result=m.decode(m.forecast_donor_law(plan,objective,request_id_map=author.fresh(plan),
            forecast_origin_ns=epoch,quality={},joint_law=dict(schema=m.generated.joint.SCHEMA,
                blocks=['block-a','block-b','block-c'],prospective_template_transfer_qualified=False)))
        self.assertEqual(plan,before)
        for cid, prediction in result['candidates'].items():
            self.assertEqual(len(prediction['draws']),12)
            self.assertEqual(sum(d['weight'] for d in prediction['draws']),1)
            self.assertEqual(prediction['world_unknown_probability_mass'],1)
            calendar=result['public_calendars'][cid]
            self.assertEqual(calendar['original_drain_end_ns'],epoch+210_000_000_000)
            self.assertEqual([r['arrival_ns'] for r in calendar['public_requests']],
                [epoch+r['offset_ns'] for r in plan['trace']])
            self.assertEqual(len(calendar['public_requests']),181)
            self.assertEqual(len(result['queue_envelopes'][cid]['draws']),12)
        self.assertFalse(result['native_source_transfer_authenticated'])
        self.assertFalse(result['predictions_sealed'])

    def test_fresh_ids_cannot_reuse_settled_prior_identity(self):
        plan,objective,draws,quality=author.numeric_fixture()
        state=author.seed()
        identity=author.fresh(plan)
        identity['old1']=state['prior_request_ids'][0]
        result=m.decode(m.forecast_compiled(plan,objective,request_id_map=identity,
            forecast_origin_ns=100,settled_seed=state,quality=quality,
            draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']}))
        self.assertTrue(all(c['world_unknown_probability_mass']==1 for c in result['candidates'].values()))


if __name__=='__main__':unittest.main()
