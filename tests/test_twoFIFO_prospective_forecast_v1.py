import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_prospective_contract_v1 import fixture
from tests.test_twoFIFO_settled_history_initial_v1 import seed
from tests.test_actual_worker_world_forecast_v4 import inputs
from tests.test_actual_worker_twoFIFO_reference_v5 import inputs as engine_inputs
from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints as C
from tools.predictor_validation_20261027 import twoFIFO_prospective_forecast_v1 as m


def fresh(plan):
    return {r['request_id']:'fresh:'+r['request_id'] for r in plan['trace']}


def numeric_fixture():
    plan, objective, _, _ = fixture()
    args = inputs(engine_inputs())
    plan.update(aliases=dict(owner=dict(family='text',supported_tasks=['text']),
        idle=dict(family='text',supported_tasks=['text'])),workload_rps={'text':1.85},
        latency_reference_ms={'text':1000},baseline_candidate_id='baseline',
        trace=[dict(request_id='old0',offset_ns=0,task='text',pool_id='input'),
               dict(request_id='old1',offset_ns=80_000_000_000,task='text',pool_id='input')],
        candidates=[dict(id=cid,execution_context_sha256='old-context',
            resource_configuration=dict(mode='shared'),routes={'old0':'owner','old1':'owner'})
            for cid in ('baseline','candidate')])
    joint = {k:args[k] for k in m.ensemble.INPUT_FIELDS}
    joint['transport_draws'] = {key:C(0,0,1,1,-5) for key in joint['transport_draws']}
    draws = [dict(draw_id=d['draw_id'],block_id=d['block_id'],weight=d['weight'],inputs=copy.deepcopy(joint))
             for d in m.generated.catalog(['a','b'])]
    return plan, objective, draws, args['quality']


class ProspectiveForecastTests(unittest.TestCase):
    def test_original181_all_tasks_routes_rates_and_full90plus120_horizon(self):
        plan = json.loads(Path('artifacts/predictor-gpu-20261027/flow-validation-v13-ocr-audited/plan.json').read_text())
        objective = json.loads(Path('configs/resource_spillover_objective_20261024.json').read_text())
        original = copy.deepcopy(plan); identity = fresh(plan)
        out = m.decode(m.forecast_donor_law(plan, objective, request_id_map=identity,
            forecast_origin_ns=100, donor_blocks=['a','b'], quality={}))
        self.assertEqual(plan, original)
        self.assertEqual(out['original_scoring_contract']['workload_rps'], plan['workload_rps'])
        for candidate in plan['candidates']:
            cid = candidate['id']; calendar = out['public_calendars'][cid]
            self.assertEqual(calendar['original_window'], [100,90_000_000_100])
            self.assertEqual(calendar['original_drain_end_ns'],210_000_000_100)
            self.assertEqual(len(calendar['public_requests']),181)
            self.assertEqual({r['task'] for r in calendar['public_requests']},{'text','image','asr','ocr'})
            for src, row in zip(plan['trace'],calendar['public_requests']):
                self.assertEqual(row,dict(request_id=identity[src['request_id']],
                    task=src['task'],pool_id=src['pool_id'],arrival_ns=100+src['offset_ns'],
                    owner=candidate['routes'][src['request_id']]))
            prediction = out['candidates'][cid]
            self.assertEqual(len(prediction['draws']),8)
            self.assertEqual(prediction['world_unknown_probability_mass'],1)
            self.assertTrue(all(len(d['all_public_request_ids_retained'])==181 for d in prediction['draws']))
        self.assertEqual(out['paired_candidate_ranking']['winner_status'],'UNKNOWN')
        self.assertEqual(out['gate_readiness'],'HOLD')
        for field in ('GPU_started','model_fit','validation_labels_read','predictions_sealed','final_gain_eligible'):
            self.assertFalse(out[field])

    def test_real_engine_generates_late_future_arrival_and_keeps_original_score(self):
        plan, objective, draws, quality = numeric_fixture()
        out = m.decode(m.forecast_compiled(plan,objective,request_id_map=fresh(plan),forecast_origin_ns=100,
            draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']},
            quality=quality,settled_seed=seed()))
        for prediction in out['candidates'].values():
            self.assertEqual(prediction['world_unknown_probability_mass'],0)
            self.assertEqual(prediction['six_terms']['P']['known_probability_mass'],1)
            self.assertEqual(prediction['utility']['unknown_probability_mass'],1)
            for draw in prediction['draws']:
                forecast = draw['forecast']; world = forecast['world']
                self.assertEqual(world['offered_requests'],2)
                self.assertGreater(world['RESULT_receipts']['fresh:old1'],80_000_000_100)
                self.assertEqual(forecast['scoring']['original_score']['cohort_size'],2)
                self.assertEqual(len(forecast['queue_diagnostics']['full_public_arrival_envelopes'])>0,True)
        self.assertEqual(out['paired_candidate_ranking']['winner_status'],'UNKNOWN')

    def test_missing_state_retains_eight_draws_and_partial_unknown_mass(self):
        plan,objective,draws,quality = numeric_fixture()
        draws[0]['inputs'] = m.ensemble.UnknownForecastInputs('missing donor')
        kwargs=dict(request_id_map=fresh(plan),forecast_origin_ns=100,
            draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']},quality=quality)
        known = m.decode(m.forecast_compiled(plan,objective,settled_seed=seed(),**kwargs))
        unknown = m.decode(m.forecast_compiled(plan,objective,**kwargs))
        for cid in known['candidates']:
            self.assertEqual(known['candidates'][cid]['world_unknown_probability_mass'],F(1,8))
            self.assertEqual(unknown['candidates'][cid]['world_unknown_probability_mass'],1)
            self.assertEqual(len(unknown['candidates'][cid]['draws']),8)
            self.assertEqual(unknown['candidates'][cid]['draws'][0]['reason'],'missing donor')

    def test_full_three_block_law_retains_twelve_and_cannot_be_qualified_by_boolean(self):
        plan,objective,_,_ = fixture()
        law=dict(schema=m.generated.joint.SCHEMA,blocks=['a','b','c'],
            prospective_template_transfer_qualified=False)
        kwargs=dict(request_id_map=fresh(plan),forecast_origin_ns=100,quality={})
        result=m.decode(m.forecast_donor_law(plan,objective,joint_law=law,**kwargs))
        self.assertTrue(all(len(v['draws'])==12 for v in result['candidates'].values()))
        self.assertTrue(all(d['weight']==F(1,12) for v in result['candidates'].values() for d in v['draws']))
        law['prospective_template_transfer_qualified']=True
        with self.assertRaisesRegex(ValueError,'No authenticated'):
            m.forecast_donor_law(plan,objective,joint_law=law,**kwargs)

    def test_invalid_public_contract_or_draw_catalog_raises_even_without_state(self):
        plan,objective,draws,quality = numeric_fixture()
        kwargs=dict(request_id_map=fresh(plan),forecast_origin_ns=100,
            draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']},quality=quality)
        for kind in ('missing_map','duplicate_map','old_map','candidate','mass','unpaired','weights','route','cap'):
            p=copy.deepcopy(plan); k=copy.deepcopy(kwargs)
            if kind=='missing_map':k['request_id_map'].pop('old1')
            elif kind=='duplicate_map':k['request_id_map']['old1']=k['request_id_map']['old0']
            elif kind=='old_map':k['request_id_map']['old1']='old1'
            elif kind=='candidate':k['draws_by_candidate'].pop('candidate')
            elif kind=='mass':k['draws_by_candidate']['baseline'][0]['weight']=F(1,9)
            elif kind=='unpaired':k['draws_by_candidate']['baseline'][0]['draw_id']='other'
            elif kind=='weights':p['weights']['P']=2
            elif kind=='route':p['candidates'][0]['routes'].pop('old1')
            else:p['candidates'][0]['global_admission_cap']=2
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.forecast_compiled(p,objective,**k)

    def test_programming_error_is_not_unknown_and_resources_cannot_change_boundary(self):
        plan,objective,draws,quality=numeric_fixture()
        kwargs=dict(request_id_map=fresh(plan),forecast_origin_ns=100,settled_seed=seed(),quality=quality,
            draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']})
        with patch.object(m.ensemble,'forecast',side_effect=ValueError('invalid macro')):
            with self.assertRaisesRegex(ValueError,'invalid macro'):m.forecast_compiled(plan,objective,**kwargs)
        with self.assertRaisesRegex(ValueError,'resource interface'):
            m.forecast_compiled(plan,objective,resources_by_candidate={'baseline':{'drain_end_ns':1}},**kwargs)

    def test_seed_for_wrong_origin_or_resident_roster_is_unknown(self):
        plan,objective,draws,quality=numeric_fixture()
        for mutation in ('origin','roster'):
            state=seed()
            if mutation=='origin':state['forecast_origin_ns']=99
            else:plan['aliases']['third']=dict(family='text',supported_tasks=['text'])
            result=m.decode(m.forecast_compiled(plan,objective,request_id_map=fresh(plan),forecast_origin_ns=100,
                settled_seed=state,quality=quality,
                draws_by_candidate={c['id']:copy.deepcopy(draws) for c in plan['candidates']}))
            self.assertTrue(all(v['world_unknown_probability_mass']==1 for v in result['candidates'].values()))

    def test_public_cpu_receipt_replays_and_rejects_forged_evidence_claim(self):
        plan,objective,_,_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            m.public.save(root/'plan.json',plan);m.public.save(root/'objective.json',objective)
            result=m.prepare_cpu(root/'plan.json',root/'objective.json',donor_blocks=['a','b'],output=root/'out')
            report_path=root/'out/report.json'
            self.assertEqual(m.audit_cpu(report_path),result)
            result['predictions_sealed']=True
            report_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'cannot claim'):m.audit_cpu(report_path)


if __name__=='__main__':unittest.main()
