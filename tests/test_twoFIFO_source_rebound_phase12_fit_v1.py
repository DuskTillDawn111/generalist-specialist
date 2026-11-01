"""CPU-only scheduling tests; fitted_rows/evaluate never execute an optimizer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_phase12_fit_v1 as m


def bundle_fixture():
    rows=[];plans=[];docs=[]
    for block in m.original.intake.campaign_v2.strict.CAPTURES:
        for shard in range(1,5):
            aid=f'{block}-shard-{shard}'
            plans.append(dict(schema=m.executor.original.SCHEMA+'.attempt',split='training',no_retries=True,
                capture_id=block,attempt_id=aid,execution_id=aid,shard_index=shard,shards_in_block=4))
            requests=[]
            for ordinal in range(46 if shard<=2 else 45):
                rid=f'{aid}:r{ordinal}';macros=[]
                for kind in ('entry','ready_tail'):
                    macros.append(dict(macro_id=kind))
                    rows.append(dict(capture=block,attempt_id=aid,request_id=rid,macro_id=kind,
                        raw_sha256='raw-'+aid,matrix=np.zeros((1,12)),group=(kind,)))
                requests.append(dict(request_id=rid,macros=macros))
            docs.append(dict(requests=requests))
    return dict(schema=m.adapter.SCHEMA,features=list(m.FEATURES),design=m.DESIGN,
        numerical_fit_run=False,validation_labels_read=False,GPU_started=False,
        denominator=dict(measured_requests=546,priming_requests_excluded_as_labels=216,training_replication_units=3),
        full=dict(rows=rows),plans=plans,documents=docs,
        observed_support=dict(each_target_OCR_input_each_block_n0_and_npositive=True))


def save(path, document):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document));return m.receipt(path)


def prepared(root):
    execution=save(root/'executor'/'manifest.json',dict(cpu_fixture=True))
    entries=[m.receipt(module.__file__) for module in (m.executor,m.intake,m.adapter)]+[m.receipt(m.__file__)]
    review=save(root/'independent.json',dict(schema='test.independent_review',status='PASS_CPU_FIXTURE',sources=entries))
    final=root/'executor'/'run01'/'report.json'
    with patch.object(m.executor,'audit',return_value={}),patch.object(m,'_closure',return_value=[m.receipt(m.__file__)]):
        ref=m.prepare(execution['path'],final,[review['path']],root/'fit-prepared')
    return ref,execution,final


def complete_final(execution):
    return dict(schema=m.executor.CONT_SCHEMA+'.run',manifest=execution,status=m.intake.FINAL_STATUS,
        no_retries=True,numerical_fit_run=False,validation_labels_read=False,fit_eligible=False,
        automatic_fit=False,completed_attempts=[dict(attempt_id=str(i)) for i in range(12)])


class SourceBoundPhase12Tests(unittest.TestCase):
    def source_patches(self):
        return patch.object(m.executor,'audit',return_value={}),patch.object(m,'_closure',return_value=[m.receipt(m.__file__)])

    def test_prepare_and_audit_bind_dynamic_executor_reviews_and_complete_closure(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,execution,final=prepared(root)
            a,b=self.source_patches()
            with a,b:manifest=m.audit(ref['path'])
            self.assertEqual(manifest['executor_manifest'],execution)
            self.assertEqual(manifest['expected_continuation_report_path'],str(final.resolve()))
            self.assertEqual(manifest['features'],list(m.FEATURES));self.assertFalse(manifest['automatic_fit'])
            self.assertEqual(manifest['expected_measured_requests'],546)
            self.assertIn('NOT_FIXED_TO_REQUEST_COUNT',manifest['macro_row_count'])

    def test_review_must_bind_current_entries_and_be_independent_pass(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,_,_=prepared(root)
            manifest=m.checked(ref);review_path=Path(manifest['independent_reviews'][0]['path'])
            for status,schema in [('WITHDRAWN','test.independent'),('PASS','test.author')]:
                review=save(review_path,dict(schema=schema,status=status,sources=[]))
                with self.subTest(status=status),self.assertRaises(ValueError):
                    m._review_bindings([review],manifest['reviewed_entry_sources'])

    def test_missing_final_never_reads_raw_or_enters_adapter_or_numeric(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,_,_=prepared(root);a,b=self.source_patches()
            with a,b,patch.object(m.intake,'intake',side_effect=AssertionError('raw forbidden')) as intake, \
                 patch.object(m.adapter,'build_training_rows',side_effect=AssertionError('adapter forbidden')) as adapter, \
                 patch.object(m.original,'fitted_rows',side_effect=AssertionError('numeric forbidden')) as fitter:
                out=m.checked(m.run(ref['path'],root/'waiting',fit_enabled=True))
            self.assertEqual(out['status'],'WAITING_FOR_EXACT12_CONTINUATION_FINAL')
            self.assertFalse(out['numerical_fit_run']);intake.assert_not_called();adapter.assert_not_called();fitter.assert_not_called()
            self.assertFalse((root/'fit-prepared'/'phase12-fit-claim.json').exists())

    def test_foreign_failed_partial_or_label_final_holds_before_raw(self):
        for mutation in ('foreign','failed','partial','labels','fit','retry'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as work:
                root=Path(work);ref,execution,final=prepared(root);doc=complete_final(execution)
                if mutation=='foreign':doc['manifest']={}
                elif mutation=='failed':doc['status']='STOPPED_NO_RETRY'
                elif mutation=='partial':doc['completed_attempts'].pop()
                elif mutation=='labels':doc['validation_labels_read']=True
                elif mutation=='fit':doc['numerical_fit_run']=True
                else:doc['no_retries']=False
                save(final,doc);a,b=self.source_patches()
                with a,b,patch.object(m.intake,'intake',side_effect=AssertionError('raw forbidden')):
                    out=m.checked(m.run(ref['path'],root/'hold',fit_enabled=True))
                self.assertEqual(out['status'],'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL')
                self.assertFalse(out['numerical_fit_run'])

    def test_complete_final_still_requires_explicit_fit(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,execution,final=prepared(root);save(final,complete_final(execution));a,b=self.source_patches()
            with a,b,patch.object(m.intake,'intake',side_effect=AssertionError('raw forbidden')):
                out=m.checked(m.run(ref['path'],root/'ready'))
            self.assertEqual(out['status'],'READY_EXPLICIT_FIT_REQUIRED');self.assertFalse(out['numerical_fit_run'])

    def test_full_and_three_whole_block_folds_keep_1092_macros_for_546_requests(self):
        bundle=bundle_fixture();calls=[]
        def fitted(rows):
            calls.append(list(rows));return dict(numerical_qualification=True,macro_rows=len(rows),features=list(m.FEATURES)),(None,None,None)
        def evaluated(rows,*parameters):
            return dict(summary=dict(rows=len(rows),supported_rows=len(rows),unknown_rows=0,mean_nll=2),
                records=[dict(capture_sha256=r['capture'],request_id=r['request_id'],macro_id=r['macro_id']) for r in rows])
        with patch.object(m.original,'fitted_rows',side_effect=fitted),patch.object(m.original,'evaluate',side_effect=evaluated), \
             patch.object(m.original.numeric,'fit',side_effect=AssertionError('optimizer forbidden')):
            result=m.fit_bundle(bundle)
        self.assertEqual(len(calls),4);self.assertEqual(len(calls[0]),1092)
        self.assertEqual(result['measured_requests'],546);self.assertEqual(result['macro_rows'],1092)
        self.assertIs(result['denominator'],bundle['denominator'])
        for index,fold in enumerate(result['whole_block_leaveout']):
            self.assertEqual(len(fold['training_attempt_ids']),8);self.assertEqual(len(fold['heldout_attempt_ids']),4)
            self.assertNotIn(fold['heldout_block_id'],{r['capture'] for r in calls[index+1]})
            self.assertTrue(all('raw_sha256' in r for r in fold['evaluation']['records']))
        self.assertTrue(result['response_law_compilation_eligible'])
        self.assertFalse(result['main_queue_provider_eligible']);self.assertFalse(result['USHER_physical_qualification'])

    def test_unqualified_root_and_unknown_support_preserve_full_denominator(self):
        bundle=bundle_fixture()
        with patch.object(m.original,'fitted_rows',return_value=(dict(numerical_qualification=False),None)), \
             patch.object(m.original,'evaluate') as evaluate:
            result=m.fit_bundle(bundle)
        evaluate.assert_not_called()
        self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in result['whole_block_leaveout']),1092)
        self.assertFalse(result['response_law_compilation_eligible']);self.assertTrue(result['unqualified_prediction_interface'].startswith('ALL_UNKNOWN'))

    def test_missing_ocr_cannot_be_overridden_by_qualified_roots(self):
        bundle=bundle_fixture();bundle['observed_support']['each_target_OCR_input_each_block_n0_and_npositive']=False
        def evaluate(rows,*parameters):
            return dict(summary=dict(rows=len(rows),supported_rows=len(rows),unknown_rows=0),
                records=[dict(capture_sha256=r['capture'],request_id=r['request_id'],macro_id=r['macro_id']) for r in rows])
        with patch.object(m.original,'fitted_rows',return_value=(dict(numerical_qualification=True),(0,0,0))), \
             patch.object(m.original,'evaluate',side_effect=evaluate):result=m.fit_bundle(bundle)
        self.assertTrue(result['fitted_root_numerical_qualification']);self.assertFalse(result['response_law_compilation_eligible'])

    def test_changed_features_duplicate_row_or_request_denominator_rejected_before_fitting(self):
        for kind in ('features','duplicate','requests','attempts'):
            bundle=bundle_fixture()
            if kind=='features':bundle['features'].pop()
            elif kind=='duplicate':bundle['full']['rows'][1]=bundle['full']['rows'][0]
            elif kind=='requests':bundle['denominator']['measured_requests']=545
            else:bundle['plans'][1]['execution_id']=bundle['plans'][0]['execution_id']
            with self.subTest(kind=kind),patch.object(m.original,'fitted_rows',side_effect=AssertionError('must reject first')):
                with self.assertRaises(ValueError):m.fit_bundle(bundle)

    def test_verified_callback_reuses_once_rederived_bundle_and_detects_source_drift(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,execution,final=prepared(root);save(final,complete_final(execution));a,b=self.source_patches()
            events=[];bundle=bundle_fixture()
            def ingest(executor_path,final_path,*,output):
                events.append('intake');return save(Path(output)/'report.json',dict(sources=[]))
            def build(path,*,workdir):events.append('adapter');return bundle
            def callback(actual,intake_doc):
                events.append('callback');self.assertIs(actual,bundle);self.assertEqual(intake_doc,dict(sources=[]));return 'generated'
            with a,b,patch.object(m.intake,'intake',side_effect=ingest),patch.object(m.adapter,'build_training_rows',side_effect=build), \
                 patch.object(m.original,'fitted_rows',side_effect=AssertionError('callback must not refit')):
                result=m.with_verified_bundle(ref['path'],str(final),callback)
            self.assertEqual(result,'generated');self.assertEqual(events,['intake','adapter','callback'])

    def test_explicit_fixture_run_claim_and_saved_audit_never_refit(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,execution,final=prepared(root);save(final,complete_final(execution));a,b=self.source_patches()
            bundle=bundle_fixture();events=[]
            def ingest(executor_path,final_path,*,output):
                events.append('intake');return save(Path(output)/'report.json',dict(sources=[]))
            def build(path,*,workdir):events.append('adapter');return bundle
            real_fit=m.fit_bundle
            def fake_fit(actual):
                events.append('numeric_fixture')
                with patch.object(m.original,'fitted_rows',return_value=(dict(numerical_qualification=False),None)):
                    return real_fit(actual)
            with a,b,patch.object(m.intake,'intake',side_effect=ingest),patch.object(m.adapter,'build_training_rows',side_effect=build), \
                 patch.object(m,'fit_bundle',side_effect=fake_fit),patch.object(m.original.numeric,'fit',side_effect=AssertionError('optimizer forbidden')):
                report_ref=m.run(ref['path'],root/'result',fit_enabled=True)
                result=m.checked(report_ref)
                self.assertEqual(result['status'],'COMPLETE_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS')
                self.assertEqual(events,['intake','adapter','numeric_fixture'])
                audit=m.audit_fit(report_ref)
                self.assertFalse(audit['numeric_recomputed']);self.assertEqual(events,['intake','adapter','numeric_fixture'])
                with self.assertRaises(FileExistsError):m.run(ref['path'],root/'retry',fit_enabled=True)
            self.assertTrue((root/'fit-prepared'/'phase12-fit-claim.json').is_file())

    def test_intake_rejection_writes_stopped_claim_and_never_fits(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);ref,execution,final=prepared(root);save(final,complete_final(execution));a,b=self.source_patches()
            with a,b,patch.object(m.intake,'intake',side_effect=ValueError('incomplete evidence')), \
                 patch.object(m.original,'fitted_rows',side_effect=AssertionError('numeric forbidden')):
                report=m.checked(m.run(ref['path'],root/'stopped',fit_enabled=True))
            self.assertEqual(report['status'],'STOPPED_NO_RETRY');self.assertFalse(report['numerical_fit_run'])
            self.assertTrue(report['unqualified_prediction_interface'].startswith('ALL_UNKNOWN'))


if __name__=='__main__':unittest.main()
