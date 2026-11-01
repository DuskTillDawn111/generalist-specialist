import copy
import math
from pathlib import Path
import tempfile
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import projected_state_save_hazard_v57 as v
from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle
from tools.predictor_validation_20261027.whole57_lifecycle_training_adapter_v30 import transition_records


def stats(t=10**9,n=0,support=None):
    value=v.empty();value.update(rows=max(1,n) if t else 0,exposure_ns=t,events=n,
        age_support_intervals_ns=([[0,t]] if support is None and t else support or []))
    spans=value['age_support_intervals_ns'];value.update(age_min_ns=spans[0][0] if spans else None,age_max_ns=spans[-1][1] if spans else None)
    return value


def model():
    result=dict(schema=v.SCHEMA,parameters=v.PARAMETERS,training_only=True,validation_labels_used=False,sources=[],
        fitted={key:v.posterior(stats()) for key in v.KEYS})
    result['model_sha256']=v.digest(result);return result


def rehash(value):value['model_sha256']=v.digest({k:x for k,x in value.items() if k!='model_sha256'});return value


def states(bits='0000'):
    return {o:OwnerLifecycle(bit=='1',None,False) for o,bit in zip(v.OWNERS,bits)}


def row(left,right,event=False,qualified=True,bits='0000',owner='text#0',capture='a'):
    return dict(capture=capture,held_owner=owner,driver_context=[1,2,3,4],restore_start_ns=0,
        start_ns=left,end_ns=right,age_entry_ns=left,age_exit_ns=right,
        exposure_qualified=qualified,physical_SAVE_END_observed=event,
        shared_left_features=dict(held_owner=owner,previous_held_owner=None,context_age_ns=left,
            handler_active={o:bit=='1' for o,bit in zip(v.OWNERS,bits)}))


class StateHazardTests(unittest.TestCase):
    def test_all_risk_positive_zero_unresolved_and_capture_separation(self):
        rows=[row(0,10),row(10,20,True),row(20,30,True,False),
            row(30,40,True,bits='0010'),row(40,60,bits='0010',capture='b')]
        counts=v.aggregate(rows,['a','b'])
        self.assertEqual(len(counts['a']),64)
        idle=v.pool(counts,['a'],'text#0:0000')
        self.assertEqual(tuple(idle[k] for k in v.COUNT_FIELDS),(2,20,1,1,10,1))
        self.assertEqual(idle['age_support_intervals_ns'],[[0,20]])
        self.assertEqual(v.pool(counts,['a','b'],'text#0:0010')['exposure_ns'],30)
        self.assertEqual(v.pool(counts,['a','b'],'text#0:0010')['events'],1)
        self.assertEqual(v.posterior(v.pool(counts,['a','b'],'text#1:0010'))['status'],'UNKNOWN_NO_EXPOSURE')
        with self.assertRaisesRegex(ValueError,'duplicated'):v.aggregate([rows[0],rows[0]],['a'])

    def test_fixed_posterior_units_positive_zero_and_predictive_log(self):
        zero=v.posterior(stats(2*10**9));positive=v.posterior(stats(2*10**9,3))
        self.assertEqual(zero['hazard_per_ms'],.00025);self.assertEqual(positive['hazard_per_second'],1.75)
        pred=v.predictive(positive,stats(10**9,2));shape=3.5;rate=2.;held=1.;n=2
        expected=math.lgamma(n+shape)-math.lgamma(shape)-math.lgamma(n+1)+shape*math.log(rate/(rate+held))+n*math.log(held/(rate+held))
        self.assertAlmostEqual(pred['log_predictive_probability'],expected,places=13)
        self.assertEqual(v.predictive(v.posterior(stats(0)),stats())['status'],'UNKNOWN_NO_TRAINING_EXPOSURE')
        self.assertEqual(v.predictive(positive,stats(0))['status'],'NO_HELDOUT_EXPOSURE')

    def test_every_holder_mask_is_a_distinct_runtime_stratum(self):
        saved=model()
        for index,key in enumerate(v.KEYS):saved['fitted'][key]=v.posterior(stats((index+1)*10**9,index%3))
        query=v.ProjectedStateSaveHazard(rehash(saved))
        for key,value in query.model['fitted'].items():
            owner,bits=key.split(':');actual=query.query(owner,[('future','ignored')],states(bits),0,10**9)
            self.assertEqual(actual['state_key'],key);self.assertEqual(actual['hazard_per_ms'],value['hazard_per_ms'])
            self.assertAlmostEqual(actual['restricted_candidate_CDF'],-math.expm1(-value['hazard_per_second']))
            self.assertFalse(actual['budget_redrawn']);self.assertFalse(actual['restore_age_reset'])

    def test_no_exposure_and_unknown_handler_remain_unknown(self):
        saved=model();saved['fitted']['ocr#0:0000']=v.posterior(stats(0));query=v.ProjectedStateSaveHazard(rehash(saved))
        with self.assertRaisesRegex(UnknownEvidence,'NO_HOLDER_MASK_EXPOSURE:ocr#0:0000'):query.query('ocr#0',[],states(),0)
        unknown=states();unknown['text#1']=OwnerLifecycle(None,True,True)
        with self.assertRaisesRegex(UnknownEvidence,'UNKNOWN_HANDLER_MASK'):query.query('text#0',[],unknown,0)
        # Issued/ready are deliberately excluded, while typed lifecycle validation remains.
        ready=states();ready['text#0']=OwnerLifecycle(False,True,True)
        self.assertEqual(query.query('text#0',[],ready,0),query.query('text#0',['unrelated_frontier'],states(),0))
        with self.assertRaises(ValueError):query.query('text#0',[],states(),True)

    def test_age_union_gaps_not_minmax_and_model_defensive_copy(self):
        saved=model();saved['fitted']['text#0:0000']=v.posterior(stats(200,0,[[0,100],[200,300]]))
        query=v.ProjectedStateSaveHazard(rehash(saved));saved['fitted']['text#0:0000']['hazard_per_ms']=0
        self.assertGreater(query.query('text#0',[],states(),150)['hazard_per_ms'],0)
        self.assertIn('PROJECTED_STATE_AGE_OUTSIDE_OBSERVED_EXPOSURE_UNION',query.query('text#0',[],states(),150)['OOD_reasons'])
        self.assertIn('PROJECTED_STATE_AGE_OUTSIDE_OBSERVED_EXPOSURE_UNION',query.query('text#0',[],states(),100)['OOD_reasons'])
        self.assertIn('PROJECTED_STATE_HORIZON_OUTSIDE_OBSERVED_EXPOSURE_UNION',query.query('text#0',[],states(),50,200)['OOD_reasons'])
        self.assertNotIn('PROJECTED_STATE_HORIZON_OUTSIDE_OBSERVED_EXPOSURE_UNION',query.query('text#0',[],states(),210,50)['OOD_reasons'])

    def test_tampered_posterior_sources_and_exact_original_manifest(self):
        saved=model();saved['fitted']['text#0:0000']['hazard_per_ms']=0
        with self.assertRaisesRegex(ValueError,'Exact unpooled'):v.ProjectedStateSaveHazard(rehash(saved))
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.write_text('first');saved=model();saved['sources']=[v.receipt(source)];rehash(saved)
            source.write_text('second')
            with self.assertRaisesRegex(ValueError,'source drift'):v.ProjectedStateSaveHazard(saved)
            manifest=Path(directory)/'manifest.json';manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Exact frozen original'):v.train(manifest,Path(directory)/'fit')

    def test_cumulative_old_state_rate_and_budget_restore_age_never_reset(self):
        saved=model();saved['fitted']['text#0:0010']=v.posterior(stats(10**9,1));query=v.ProjectedStateSaveHazard(rehash(saved))
        controller=CumulativeLifecycleHazard(query,age_edges_ns=[0],sources=[v.receipt(v.__file__)],
            candidate_identity=query.model['model_sha256'],allow_unvalidated_approximation=True)
        initial=controller.start_restore('text#0',0,[],'fixed-restore',states())
        active=controller.update(initial['state'],200_000_000,[],states('0010'))
        self.assertAlmostEqual(active['consumed_hazard'],.1)
        idle=controller.update(active['state'],400_000_000,[],states())
        self.assertAlmostEqual(idle['consumed_hazard'],.4)
        for result in (active,idle):
            self.assertEqual(result['hazard_budget'],math.log(2));self.assertFalse(result['budget_redrawn'])
            self.assertEqual(result['origin_restore_ns'],0);self.assertEqual(result['state']['restore_key'],'fixed-restore')
        self.assertEqual(idle['restore_age_ns'],400_000_000)
        self.assertAlmostEqual(idle['absolute_save_candidate_ns'],400_000_000+(math.log(2)-.4)/.5*1e9,delta=2)
        with self.assertRaisesRegex(ValueError,'Monotone'):controller.update(idle['state'],300_000_000,[],states())

    def test_native_clean_same_owner_different_context_not_definitionally_removed(self):
        clean,unresolved=transition_records([(0,10,'text#0',(1,2,3,4)),(12,20,'text#0',(1,2,5,4))],[],0,30,'capture',[])
        self.assertEqual(len(clean),1);self.assertTrue(clean[0]['same_owner']);self.assertFalse(clean[0]['same_driver_context']);self.assertEqual(unresolved,[])
        clean,unresolved=transition_records([(0,10,'text#0',(1,2,3,4)),(12,20,'text#0',(1,2,5,4))],[dict(timestamp=11)],0,30,'capture',[])
        self.assertEqual(clean,[]);self.assertEqual(len(unresolved),1)


if __name__=='__main__':unittest.main()
