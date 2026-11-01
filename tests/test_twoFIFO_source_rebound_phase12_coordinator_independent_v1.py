"""Independent CPU coordinator tests: numerical fit/evaluation are always mocked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.test_twoFIFO_source_rebound_phase12_adapter_independent_v1 import fixture
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_phase12_fit_v1 as c


def make_bundle():
    plans,_,docs=fixture()
    # Two complete macros per request prove 546 requests is not a macro-row count.
    for doc in docs:
        for request in doc['requests']:
            first=request['macros'][0];second=copy.deepcopy(first)
            delta=first['response_wall_ns'];second['macro_id']='ready_tail';second['semantic_phase']='ready_tail';second['kind']='ready_tail'
            for key in ('left_ns','right_ns'):second[key]+=delta
            for piece in second['causal_covariate_path']:
                for key in ('left_ns','right_ns'):piece[key]+=delta
                piece['state']['cutoff_processing_ns']=piece['left_ns']
            request['macros'].append(second);request['worker_start_to_ready_ns']+=delta
    rows=c.adapter.prepared_rows_new(docs,c.FEATURES)
    return dict(schema=c.adapter.SCHEMA,features=list(c.FEATURES),design=c.DESIGN,numerical_fit_run=False,
        validation_labels_read=False,GPU_started=False,denominator=dict(measured_requests=546,priming_requests_excluded_as_labels=216),
        full=dict(rows=rows),plans=plans,documents=docs,observed_support={'each_target_OCR_input_each_block_n0_and_npositive':True})


class IndependentCoordinatorTests(unittest.TestCase):
    def test_verified_callback_is_numeric_free_and_rechecks_source_after_callback(self):
        for mutate in (False,True):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);mp=root/'manifest';mp.write_text('{}');final=root/'final.json'
                manifest=dict(sources=[],expected_continuation_report_path=str(final),executor_manifest={'path':'executor','sha256':'a'*64})
                final.write_text(json.dumps(dict(schema=c.executor.CONT_SCHEMA+'.run',manifest=manifest['executor_manifest'],
                    status=c.intake.FINAL_STATUS,no_retries=True,numerical_fit_run=False,validation_labels_read=False,
                    fit_eligible=False,automatic_fit=False,completed_attempts=[{}]*12)))
                evidence=root/'evidence';evidence.write_text('original');evidence_ref=c.receipt(evidence)
                def ingest(*args,output):
                    output=Path(output);output.mkdir();path=output/'report.json'
                    path.write_text(json.dumps({'sources':[evidence_ref]}));return c.receipt(path)
                def callback(bundle,doc):
                    self.assertEqual(len(bundle['full']['rows']),1092)
                    if mutate:evidence.write_text('changed')
                    return doc
                with patch.object(c,'audit',return_value=manifest),patch.object(c.intake,'intake',side_effect=ingest), \
                     patch.object(c.adapter,'build_training_rows',return_value=make_bundle()),patch.object(c,'fit_bundle') as fit, \
                     patch.object(c.original,'fitted_rows') as numeric:
                    if mutate:
                        with self.assertRaises(ValueError):c.with_verified_bundle(mp,final,callback)
                    else:
                        result=c.with_verified_bundle(mp,final,callback)
                        self.assertEqual(result['sources'],[evidence_ref])
                    fit.assert_not_called();numeric.assert_not_called()

    def test_explicit_run_revalidates_intake_before_adapter_and_keeps_exclusive_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mp=root/'manifest';mp.write_text('{}');final=root/'final.json'
            manifest=dict(sources=[],expected_continuation_report_path=str(final),executor_manifest={'path':'executor','sha256':'a'*64})
            final.write_text(json.dumps(dict(schema=c.executor.CONT_SCHEMA+'.run',manifest=manifest['executor_manifest'],
                status=c.intake.FINAL_STATUS,no_retries=True,numerical_fit_run=False,validation_labels_read=False,
                fit_eligible=False,automatic_fit=False,completed_attempts=[{}]*12)))
            events=[]
            def ingest(manifest_path,final_path,*,output):
                events.append('intake');output=Path(output);output.mkdir()
                self.assertEqual(manifest_path,'executor');self.assertEqual(final_path,str(final))
                path=output/'report.json';path.write_text(json.dumps({'sources':[]}))
                return c.receipt(path)
            def build(path,*,workdir):
                events.append('adapter');self.assertTrue(Path(path).exists())
                return make_bundle()
            def fake_fit(bundle):
                events.append('mock_numeric');return {'unqualified_prediction_interface':'ALL_UNKNOWN_TEST'}
            with patch.object(c,'audit',return_value=manifest),patch.object(c.intake,'intake',side_effect=ingest), \
                 patch.object(c.adapter,'build_training_rows',side_effect=build),patch.object(c,'fit_bundle',side_effect=fake_fit):
                result=c.checked(c.run(mp,root/'out',fit_enabled=True))
                self.assertEqual(events,['intake','adapter','mock_numeric'])
                self.assertTrue((root/'phase12-fit-claim.json').is_file())
                self.assertEqual(result['status'],'COMPLETE_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS')
                with self.assertRaises(FileExistsError):c.run(mp,root/'retry',fit_enabled=True)

    def test_waiting_and_disabled_fit_never_open_intake_or_numeric(self):
        for state in ('missing','foreign','ready'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);mp=root/'manifest';mp.write_text('{}');final=root/'final.json'
                manifest=dict(sources=[],expected_continuation_report_path=str(final),executor_manifest={'path':'executor','sha256':'a'*64})
                if state!='missing':
                    doc=dict(schema=c.executor.CONT_SCHEMA+'.run',manifest=manifest['executor_manifest'],status=c.intake.FINAL_STATUS,
                        no_retries=True,numerical_fit_run=False,validation_labels_read=False,fit_eligible=False,automatic_fit=False,completed_attempts=[{}]*12)
                    if state=='foreign':doc['manifest']={}
                    final.write_text(json.dumps(doc))
                with patch.object(c,'audit',return_value=manifest),patch.object(c.intake,'intake') as ingest,patch.object(c,'fit_bundle') as fit:
                    result=c.checked(c.run(mp,root/'out',fit_enabled=state!='ready'))
                ingest.assert_not_called();fit.assert_not_called()
                self.assertEqual(result['status'],{'missing':'WAITING_FOR_EXACT12_CONTINUATION_FINAL','foreign':'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL','ready':'READY_EXPLICIT_FIT_REQUIRED'}[state])
                self.assertFalse(result['numerical_fit_run'])

    def test_failed_roots_keep_1092_macro_rows_for_546_requests_unknown(self):
        bundle=make_bundle();calls=[]
        def fail(rows):
            calls.append(rows)
            return {'numerical_qualification':False},None
        with patch.object(c.original,'fitted_rows',side_effect=fail),patch.object(c.original,'evaluate') as evaluate:
            result=c.fit_bundle(bundle)
        evaluate.assert_not_called()
        self.assertEqual(len(calls),4)
        self.assertEqual(result['measured_requests'],546)
        self.assertEqual(result['macro_rows'],1092)
        self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in result['whole_block_leaveout']),1092)
        self.assertFalse(result['response_law_compilation_eligible'])
        self.assertTrue(result['unqualified_prediction_interface'].startswith('ALL_UNKNOWN'))
        for fold in result['whole_block_leaveout']:
            self.assertEqual(len(fold['training_attempt_ids']),8);self.assertEqual(len(fold['heldout_attempt_ids']),4)
            self.assertTrue(set(fold['training_attempt_ids']).isdisjoint(fold['heldout_attempt_ids']))

    def test_supported_roots_cannot_override_missing_ocr_or_qualify_main_provider(self):
        bundle=make_bundle();bundle['observed_support']['each_target_OCR_input_each_block_n0_and_npositive']=False
        def evaluate(rows,*args):
            return dict(summary=dict(rows=len(rows),supported_rows=len(rows),unknown_rows=0,mean_nll=2.),
                records=[dict(capture_sha256=r['capture'],request_id=r['request_id'],macro_id=r['macro_id'],support='SUPPORTED') for r in rows])
        with patch.object(c.original,'fitted_rows',return_value=({'numerical_qualification':True},(0,0,0))),patch.object(c.original,'evaluate',side_effect=evaluate):
            result=c.fit_bundle(bundle)
        self.assertTrue(result['fitted_root_numerical_qualification'])
        self.assertFalse(result['response_law_compilation_eligible'])
        self.assertFalse(result['main_queue_provider_eligible']);self.assertFalse(result['final_gain_eligible'])
        self.assertFalse(result['independent_prediction_accuracy_validated'])
        self.assertTrue(all('raw_sha256' in r and 'block_id' in r for f in result['whole_block_leaveout'] for r in f['evaluation']['records']))

    def test_duplicate_evaluator_records_rejected(self):
        bundle=make_bundle()
        def evaluate(rows,*args):
            return dict(summary=dict(rows=len(rows),supported_rows=len(rows),unknown_rows=0),records=[dict(capture_sha256=rows[0]['capture'],request_id=rows[0]['request_id'],macro_id=rows[0]['macro_id']) for _ in rows])
        with patch.object(c.original,'fitted_rows',return_value=({'numerical_qualification':True},(0,0,0))),patch.object(c.original,'evaluate',side_effect=evaluate):
            with self.assertRaises(ValueError):c.fit_bundle(bundle)


if __name__=='__main__':unittest.main()
