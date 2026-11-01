import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027 import twoFIFO_native_quality_labels_v1 as m


class NativeQualityTests(unittest.TestCase):
    def fixture(self, root):
        pool = root / 'inputs.jsonl'
        refs, entries, arrivals, records = {}, [], [], []
        for i, text in enumerate(('a', 'b'*9)):
            media = root / str(i)
            media.write_bytes(bytes([i]))
            sample = dict(id='sample'+str(i), task='ocr', input=str(media), input_sha256=m.receipt(media)['sha256'])
            pool_id = 'pool'+str(i)
            refs[pool_id] = {**sample, 'label':text}
            entries.append(dict(task_type='ocr', pool_id=pool_id, sample=sample))
            request_id = 'r'+str(i)
            arrivals.append(dict(request_id=request_id, owner='ocr#0', task='ocr', pool_id=pool_id, arrival_ns=i))
            records.append(dict(request_id=request_id, model='ocr#0', task='ocr', pool_id=pool_id,
                sample_id=sample['id'], scheduled_time_ns=100+i, completion_time_ns=200+i, status='completed',
                worker_response=dict(task='ocr', request_id=request_id, prediction='' if i==0 else text, success=True)))
        pool.write_text(''.join(json.dumps(r)+'\n' for r in entries))
        references = {'ocr':refs}
        catalog = root / 'catalog.json'
        catalog.write_text(json.dumps(dict(quality=dict(schema_version=1, references=references,
            identity=dict(reference_sha256=m.quality_digest(references))))))
        case = dict(case_id='case', arrivals=arrivals)
        plan = dict(capture_id='block', attempt_id='attempt', execution_id='exec',
                    prepared_pool=m.receipt(pool), cases=[case])
        raw = dict(schema=m.labels.executor.RAW_SCHEMA, backend='native', status='completed', split='training',
            **{k:plan[k] for k in ('capture_id','attempt_id','execution_id')},
            cases=[dict(case_id='case', observation_start_ns=100)], request_records=records)
        return raw, plan, case, {'workload_rps':{'ocr':1.}}, m.receipt(catalog)

    def test_measured_cer_sums_units_and_missing_positive_task_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,plan,case,score,ref=self.fixture(Path(directory))
            value=m.measure_case(raw,plan,case,100,score,ref)
            self.assertEqual(value['measured_Q'],.1)  # One error / ten reference chars, not mean(1,0).
            self.assertEqual(value['per_task']['ocr']['scored'],2)
            score['workload_rps']['image']=.01
            value=m.measure_case(raw,plan,case,100,score,ref)
            self.assertIsNone(value['measured_Q'])
            self.assertIn('NO_COHORT_OBSERVATION_FOR_TASK:image',value['unknown_reasons'])
            self.assertEqual(value['offered_requests'],2)

    def test_unfinished_or_late_native_outputs_are_not_training_quality_fallbacks(self):
        for kind in ('unfinished','late'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                raw,plan,case,score,ref=self.fixture(Path(directory))
                if kind=='unfinished':raw['request_records'][0]['status']='pending'
                else:raw['request_records'][0]['completion_time_ns']=100+71*10**9
                value=m.measure_case(raw,plan,case,100,score,ref)
                self.assertIsNone(value['measured_Q'])
                self.assertEqual(value['per_task']['ocr']['missing'],['r0'])
                self.assertEqual(value['offered_requests'],2)
                self.assertFalse(value['training_quality_fallback_used'])

    def test_input_reference_bytes_route_and_fingerprint_are_authenticated(self):
        for kind in ('payload','reference','fingerprint','route','sample','response'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);raw,plan,case,score,ref=self.fixture(root)
                if kind=='payload':(root/'0').write_bytes(b'changed')
                elif kind in ('reference','fingerprint'):
                    doc=m.checked(ref)
                    doc['quality']['references']['ocr']['pool0']['input_sha256']='f'*64
                    if kind=='reference':doc['quality']['identity']['reference_sha256']=m.quality_digest(doc['quality']['references'])
                    Path(ref['path']).write_text(json.dumps(doc));ref=m.receipt(ref['path'])
                elif kind=='route':raw['request_records'][0]['model']='foreign'
                elif kind=='sample':raw['request_records'][0]['sample_id']='foreign'
                else:raw['request_records'][0]['worker_response']['request_id']='foreign'
                with self.assertRaises(ValueError):m.measure_case(raw,plan,case,100,score,ref)

    def test_actual_Q_can_complete_original_score_without_changing_other_five_terms(self):
        score_plan=dict(workload_rps={'ocr':1.},latency_reference_ms={'ocr':10.},weights={k:1/6 for k in 'QTLPME'})
        original=dict(terms=dict(Q=None,T=.1,L=.2,P=.3,M=.4,E=.5),unknown_reasons=['UNKNOWN_AGGREGATE_QUALITY:ocr'],
            absolute_utility_complete=False,per_task={'ocr':dict(mean_ms=2.,p95_ms=3.)})
        observed=dict(schema=m.labels.SCHEMA,attempt_id='attempt',case_id='case',measured_original_score=original,
            measured_resources=dict(startup_energy_joules=10.,measurement_board_energy_joules=20.))
        quality=dict(schema=m.SCHEMA,attempt_id='attempt',case_id='case',measured_Q=.6,unknown_reasons=[],per_task={},
            scoring_contract=copy.deepcopy(score_plan))
        result=m.with_measured_quality(observed,quality,score_plan)
        self.assertAlmostEqual(result['full_U'],-.35)
        self.assertTrue(result['absolute_utility_complete'])
        self.assertTrue(all(result['terms'][k]==original['terms'][k] for k in 'TLPME'))
        self.assertIsNone(original['terms']['Q'])
        observed['measured_resources']['startup_energy_joules']=None
        self.assertIsNone(m.with_measured_quality(observed,quality,score_plan)['full_U'])

    def test_contract_binding_and_preserved_term_recomputation(self):
        score_plan=dict(workload_rps={'ocr':1.},latency_reference_ms={'ocr':10.},weights={k:1/6 for k in 'QTLPME'})
        original=dict(terms=dict(Q=None,T=.1,L=.2,P=.3,M=.4,E=.5),unknown_reasons=[],
            per_task={'ocr':dict(mean_ms=2.,p95_ms=3.)})
        observed=dict(schema=m.labels.SCHEMA,attempt_id='a',case_id='c',measured_original_score=original,
            measured_resources=dict(startup_energy_joules=10.,measurement_board_energy_joules=20.))
        quality=dict(schema=m.SCHEMA,attempt_id='a',case_id='c',measured_Q=.6,unknown_reasons=[],per_task={},
            scoring_contract=copy.deepcopy(score_plan))
        changed=copy.deepcopy(score_plan);changed['workload_rps']['ocr']=2.
        with self.assertRaises(ValueError):m.with_measured_quality(observed,quality,changed)
        changed=copy.deepcopy(score_plan);changed['latency_reference_ms']['ocr']=100.
        quality['scoring_contract']=copy.deepcopy(changed)
        with self.assertRaises(ValueError):m.with_measured_quality(observed,quality,changed)

    def test_explicit_failed_output_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,plan,case,score,ref=self.fixture(Path(directory))
            raw['request_records'][0]['worker_response']['success']=False
            value=m.measure_case(raw,plan,case,100,score,ref)
            self.assertIsNone(value['measured_Q'])
            self.assertEqual(value['offered_requests'],2)
            self.assertEqual(value['requests'][0]['status'],'UNVERIFIED_QUALITY_OUTPUT')


if __name__=='__main__':unittest.main()
