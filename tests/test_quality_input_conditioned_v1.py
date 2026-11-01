import copy
import unittest

from tools.predictor_validation_20261027 import quality_input_conditioned_v1 as module
from tools.predictor_validation_20261027.profiling import digest


class InputQualityTests(unittest.TestCase):
    def setUp(self):
        self.catalog={'quality':{'references':{
            'ocr':{'short':{'task':'ocr','label':'a'},'long':{'task':'ocr','label':'abcdefghi'}},
            'asr':{'short':{'task':'asr','label':'one'},'long':{'task':'asr','label':'one two three four five six seven eight nine'}}}}}
        records=[]
        for task in ('ocr','asr'):
            for pool in ('short','long'):
                reference=self.catalog['quality']['references'][task][pool]
                records.append(dict(request_id=f'{task}-{pool}',split='training',status='completed',
                    model='a',task=task,pool_id=pool,worker_response=dict(task=task,success=True,
                        prediction='x' if pool=='short' else reference['label'])))
        self.raw=dict(kind='measured',backend='native',status='completed',request_records=records)
        self.observations=module.score_rows(self.raw,self.catalog,{'a':'f'})
        self.model=module.fit_observations(self.observations)
        self.plan=dict(aliases={'a':{'family':'f'}},workload_rps={'ocr':1.,'asr':3.},trace=[])
        for task in ('ocr','asr'):
            for i,pool in enumerate(('short','short','long')):
                self.plan['trace'].append(dict(request_id=f'{task}-{i}',task=task,pool_id=pool))
        self.candidate=dict(routes={r['request_id']:'a' for r in self.plan['trace']})

    def test_real_native_asr_ocr_unequal_reference_units_ratio_of_totals(self):
        prediction=module.predict(self.plan,self.candidate,self.model,self.catalog)
        for task in ('ocr','asr'):
            self.assertEqual(self.model['conditional']['f'][task]['long']['reference_denominator'],9.)
            self.assertEqual(prediction['per_task'][task]['expected_numerator'],2.)
            self.assertEqual(prediction['per_task'][task]['reference_denominator'],11.)
            self.assertAlmostEqual(prediction['per_task'][task]['loss'],2/11)
            self.assertNotAlmostEqual(prediction['per_task'][task]['loss'],2/3) # mean request CER/WER
        self.assertAlmostEqual(prediction['Q'],2/11)
        pooled=module.aggregate_prediction(self.plan,self.candidate,self.model['pooled_loss_by_family_task'])
        self.assertAlmostEqual(pooled['Q'],.1)

    def test_across_family_pooling_uses_reference_not_request_weights(self):
        observations=[dict(family='shortmodel',task='asr',pool_id='short',numerator=1.,denominator=1.),
            dict(family='longmodel',task='asr',pool_id='long',numerator=0.,denominator=9.)]
        model=module.fit_observations(observations)
        plan=dict(aliases={'a':{'family':'shortmodel'},'b':{'family':'longmodel'}},workload_rps={'asr':1.},trace=[
            dict(request_id='1',task='asr',pool_id='short'),dict(request_id='2',task='asr',pool_id='long')])
        candidate=dict(routes={'1':'a','2':'b'})
        self.assertAlmostEqual(module.predict(plan,candidate,model,self.catalog)['Q'],.1)
        self.assertAlmostEqual(module.aggregate_prediction(plan,candidate,model['pooled_loss_by_family_task'])['Q'],.5)

    def test_rates_weight_tasks_after_native_reference_aggregation(self):
        plan=copy.deepcopy(self.plan);plan['trace']=[r for r in plan['trace'] if r['task']=='asr' or r['pool_id']=='long']
        candidate=dict(routes={r['request_id']:'a' for r in plan['trace']})
        prediction=module.predict(plan,candidate,self.model,self.catalog)
        self.assertEqual(prediction['per_task']['ocr']['loss'],0.)
        self.assertAlmostEqual(prediction['Q'],.75*2/11)

    def test_missing_positive_input_keeps_whole_task_and_Q_unknown(self):
        observations=[r for r in self.observations if not (r['task']=='asr' and r['pool_id']=='short')]
        prediction=module.predict(self.plan,self.candidate,module.fit_observations(observations),self.catalog)
        self.assertIsNone(prediction['Q']);self.assertIsNone(prediction['per_task']['asr']['expected_numerator'])
        self.assertEqual(len(prediction['per_task']['asr']['missing']),2)
        self.assertEqual(prediction['per_task']['ocr']['loss'],2/11)

    def test_repeated_outputs_estimate_error_count_without_changing_reference(self):
        observations=copy.deepcopy(self.observations)
        extra=copy.deepcopy(observations[0]);extra['numerator']=0.;observations.append(extra)
        model=module.fit_observations(observations)
        self.assertEqual(model['conditional']['f']['ocr']['short']['expected_numerator'],.5)
        self.assertEqual(model['conditional']['f']['ocr']['short']['observation_count'],2)
        prediction=module.predict(self.plan,self.candidate,model,self.catalog)
        self.assertEqual(prediction['per_task']['ocr']['loss'],1/11)

    def test_native_edit_errors_can_exceed_reference_length(self):
        raw=copy.deepcopy(self.raw);raw['request_records']=[raw['request_records'][2]]
        raw['request_records'][0]['worker_response']['prediction']='two three four five'
        rows=module.score_rows(raw,self.catalog,{'a':'f'})
        self.assertEqual(rows[0]['numerator'],4.)
        model=module.fit_observations(rows)
        self.assertEqual(model['conditional']['f']['asr']['short']['expected_numerator'],4.)

    def test_changed_reference_or_conditional_denominator_rejected(self):
        model=copy.deepcopy(self.model);model['conditional']['f']['asr']['short']['reference_denominator']=2
        model.pop('model_sha256');model['model_sha256']=digest(model)
        with self.assertRaisesRegex(ValueError,'denominator changed'):
            module.predict(self.plan,self.candidate,model,self.catalog)
        rows=copy.deepcopy(self.observations);row=copy.deepcopy(rows[0]);row['denominator']+=1;rows.append(row)
        with self.assertRaisesRegex(ValueError,'deterministic denominator'):module.fit_observations(rows)

    def test_validation_incomplete_and_duplicate_rows_rejected(self):
        for change in ({'split':'validation'},{'status':'error'},{'request_id':self.raw['request_records'][1]['request_id']}):
            with self.subTest(change=change):
                raw=copy.deepcopy(self.raw);raw['request_records'][0].update(change)
                with self.assertRaisesRegex(ValueError,'training quality rows'):
                    module.score_rows(raw,self.catalog,{'a':'f'})

    def test_public_predictor_cannot_consume_target_outputs_and_rejects_bad_routes(self):
        self.assertNotIn('worker_response',str(module.predict(self.plan,self.candidate,self.model,self.catalog)))
        candidate=copy.deepcopy(self.candidate);candidate['routes'].pop(next(iter(candidate['routes'])))
        with self.assertRaisesRegex(ValueError,'full trace/routes'):
            module.predict(self.plan,candidate,self.model,self.catalog)


if __name__=='__main__':unittest.main()
