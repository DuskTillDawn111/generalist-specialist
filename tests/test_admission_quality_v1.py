"""Independent native CPU fixtures and original Q arithmetic; no GPU."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import test_admission_training_v1 as fixtures
from tools.predictor_validation_20261027 import admission_quality_v1 as m


class AdmissionQualityTests(unittest.TestCase):
    fixture=fixtures.AdmissionTrainingTests.fixture

    def setUp(self):
        fixtures.AdmissionTrainingTests.setUp(self)
        self.protocol=self.root/'quality-protocol.json'
        m.prepare(self.path,self.protocol)

    def annotate(self,job):
        raw,path=self.fixture(job)
        for row in raw['request_records']:row['worker_response']['prediction']=''
        fixtures.write(path,raw)
        plan=m.read(job['plan']['path']);catalog=m.read(plan['catalog']['path'])
        evidence=dict(kind='measured_per_request',errors=[],
            totals_by_request=m.audit.quality_totals(raw['request_records'],catalog),
            source_receipts=[plan['catalog'],m.receipt(m.capacity.training.native_flow.__file__),
                m.receipt(m.audit.score_result.__code__.co_filename),m.receipt(m.audit.sample_totals.__code__.co_filename)])
        scored={**raw,'native_raw':m.receipt(path),'quality_evidence':evidence}
        out=path.with_name('scored-raw.json');fixtures.write(out,scored)
        return path,out

    def test_real_eight_native_training_mappings_and_independent_quality_recompute(self):
        before={}
        for job in self.campaign['jobs']:
            raw,scored=self.annotate(job);before[raw]=raw.read_bytes();before[scored]=scored.read_bytes()
        with patch.object(m.capacity.training.flow_profiles,'map_flow_collection',side_effect=AssertionError('No shared mapping')):
            bundle=m.load_runs(self.protocol)
        self.assertEqual(len(bundle['runs']),8)
        self.assertEqual(sum(len(r['quality_rows']) for r in bundle['runs']),9600)
        self.assertTrue(all(r['quality_audit']['independently_recomputed'] for r in bundle['runs']))
        for path,content in before.items():self.assertEqual(path.read_bytes(),content)
        out=self.root/'fit';value=m.fit(self.protocol,out)
        self.assertEqual(len(value['folds']),2)
        self.assertEqual(len(list(out.glob('*-quality.json'))),12)
        cv=m.read(value['quality_cv']['path']);self.assertEqual(len(cv['rows']),8)
        self.assertTrue(all(c['seed'] not in c['training_seeds'] for c in cv['rows']))
        self.assertEqual(m.verify_manifest(out/'manifest.json',recompute=False),value)

    def test_prospective_protocol_source_drift_or_validation_wrapper_rejected(self):
        value=m.read(self.protocol)
        self.assertFalse(value['training_outcomes_read']);self.assertFalse(value['GPU_started'])
        value['policy']={**value['policy'],'pooling':'shared'}
        value['protocol_sha256']=m.digest({k:v for k,v in value.items() if k!='protocol_sha256'})
        fixtures.write(self.protocol,value)
        with self.assertRaisesRegex(ValueError,'Changed prospective'):m.verify_protocol(self.protocol)
        with self.assertRaisesRegex(ValueError,'wrappers'):
            m.capacity.training.map_training({**self.job,'split':'validation'},self.path)

    def test_missing_training_data_never_fitted_or_output_created(self):
        out=self.root/'fit'
        with self.assertRaisesRegex(ValueError,'All eight'):m.fit(self.protocol,out)
        self.assertFalse(out.exists())

    def minimal_run(self,seed=628101,cid=None):
        cid=cid or self.job['candidate_id'];plan=copy.deepcopy(self.plan)
        candidate=next(c for c in plan['candidates'] if c['id']==cid)
        identity=plan['controller_binding']['controller_sha256']
        # Synthetic scoring-unit fixture, explicitly no native artifact claim.
        rows=[dict(request_id='short',family='ocr',task='ocr',pool_id='short',numerator=1.,denominator=1.),
            dict(request_id='long',family='ocr',task='ocr',pool_id='long',numerator=0.,denominator=9.)]
        for row in rows:row.update(split='training',controller_sha256=identity,candidate_id=cid,seed=seed)
        return dict(seed=seed,candidate_id=cid,controller_sha256=identity,
            condition=m.capacity.condition(plan,candidate),generation=m.capacity._generation(plan,m.capacity.condition(plan,candidate)),
            plan=plan,plan_source=self.job['plan'],raw_source=self.job['plan'],scored_source=self.job['plan'],
            quality_rows=rows,quality_audit={'cpu_fixture':True})

    def test_ratio_of_sums_not_mean_cer_and_whole_seed_isolation(self):
        run=self.minimal_run();cid=run['candidate_id']
        model=m.merge_route([run],cid,m.receipt(self.protocol),628102)
        self.assertEqual(model['loss_by_family_task']['ocr']['ocr'],.1)
        self.assertEqual(model['totals_by_family_task']['ocr']['ocr']['unique_input_count'],2)
        self.assertEqual(model['training_seeds'],[628101])
        with self.assertRaisesRegex(ValueError,'heldout seed'):m.merge_route([run],cid,m.receipt(self.protocol),628101)
        wrong=copy.deepcopy(run);wrong['controller_sha256']='shared'
        with self.assertRaisesRegex(ValueError,'conditions'):m.merge_route([run,wrong],cid,m.receipt(self.protocol))
        run['quality_rows'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Only exact training'):m.merge_route([run],cid,m.receipt(self.protocol))

    def prediction_model(self):
        run=self.minimal_run();model=m.merge_route([run],run['candidate_id'],m.receipt(self.protocol))
        model['loss_by_family_task']={'text':{'text':.2},'generalist_2':{'text':.4,'image':.6,'asr':.8},'ocr':{'ocr':1.}}
        model['model_sha256']=m.digest({k:v for k,v in model.items() if k!='model_sha256'})
        candidate=next(c for c in run['plan']['candidates'] if c['id']==run['candidate_id'])
        return model,run['plan'],candidate

    def test_original_rateweighted_Q_and_missing_no_fallback(self):
        model,plan,candidate=self.prediction_model()
        result=m.predict(model,plan,candidate)
        expected=m.arithmetic.aggregate_prediction(plan,candidate,model['loss_by_family_task'])
        self.assertEqual(result['Q'],expected['Q'])
        self.assertAlmostEqual(result['Q'],sum(plan['workload_rps'][t]/2.*q for t,q in result['per_task'].items()))
        self.assertNotAlmostEqual(result['Q'],sum(result['per_task'].values())/4)
        del model['loss_by_family_task']['generalist_2']['image']
        model['model_sha256']=m.digest({k:v for k,v in model.items() if k!='model_sha256'})
        missing=m.predict(model,plan,candidate)
        self.assertIsNone(missing['Q']);self.assertEqual(missing['status'],'UNKNOWN')
        self.assertIsNone(missing['per_task']['image'])

    def test_wrong_route_controller_source_or_target_law_rejected(self):
        model,plan,candidate=self.prediction_model()
        other=next(c for c in plan['candidates'] if c['id']!=candidate['id'])
        with self.assertRaisesRegex(ValueError,'exact admission'):m.predict(model,plan,other)
        changed=copy.deepcopy(model);changed['controller_binding']['controller_sha256']='shared'
        changed['model_sha256']=m.digest({k:v for k,v in changed.items() if k!='model_sha256'})
        with self.assertRaisesRegex(ValueError,'exact admission'):m.predict(changed,plan,candidate)
        model['source_receipts'].append({'path':str(self.root/'foreign'),'sha256':'changed'})
        model['model_sha256']=m.digest({k:v for k,v in model.items() if k!='model_sha256'})
        with self.assertRaises(FileNotFoundError):m.predict(model,plan,candidate)

    def test_changed_scored_annotation_and_raw_content_rejected(self):
        # Use real strict mapping of this native fixture; isolate the other seven
        # acquisition slots only at the loader boundary for failure injection.
        raw_path,scored_path=self.annotate(self.job)
        mapped=m.capacity.training.map_training(self.job,self.path)
        raw=m.read(raw_path);plan=mapped['plan'];candidate=mapped['candidate']
        run=dict(seed=self.job['seed'],candidate_id=self.job['candidate_id'],plan=plan,plan_source=self.job['plan'],
            raw_source=m.receipt(raw_path),startup_begin_ns=raw['startup_begin_ns'],
            controller_sha256=self.job['controller_sha256'])
        with patch.object(m.capacity,'load_campaign',return_value={'runs':[run],'source_receipts':[]}):
            scored=m.read(scored_path);scored['quality_evidence']['totals_by_request'][raw['request_records'][0]['request_id']]['numerator']+=1
            fixtures.write(scored_path,scored)
            with self.assertRaisesRegex(ValueError,'recomputed'):m.load_runs(self.protocol)
            scored['request_records'][0]['split']='validation';fixtures.write(scored_path,scored)
            with self.assertRaisesRegex(ValueError,'preserve'):m.load_runs(self.protocol)


if __name__=='__main__':unittest.main()
