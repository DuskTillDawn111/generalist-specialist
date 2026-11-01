"""Independent CPU rebound law/runner tests; no fit, GPU, or validation inputs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests.test_twoFIFO_source_rebound_phase12_adapter_independent_v1 import fixture
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_generated_v1 as runner
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_joint_law_v1 as compiler


def data():
    plans,_,docs=fixture()
    binding={'path':'authenticated-fixture','sha256':'f'*64}
    raws=[]
    for plan,doc in zip(plans,docs):
        doc['source_rebound_manifest']=binding
        raws.append(dict(capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],
            backend='native',status='completed',priming_request_records=[{'request_id':f"{plan['attempt_id']}:p{i}",'model':'text#0','execution_events':[]} for i in range(18)],
            request_records=[{'request_id':r['request_id'],'model':r['owner'],'profile_job_id':'fixture'} for r in doc['requests']]))
    return plans,docs,raws


class IndependentReboundGeneratedTests(unittest.TestCase):
    def test_saved_fit_run_reuses_verified_intake_without_refitting_and_rejects_receipt_drift(self):
        for mutate in (False,True):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                def put(name,value):
                    path=root/name;path.write_text(json.dumps(value));return runner.receipt(path)
                design_ref=put('design.json',{});manifest_ref=put('fit-manifest.json',{})
                score_ref=put('score.json',{});quality_ref=put('quality.json',{});fit_ref=put('fit.json',{})
                final_ref=put('final.json',{});intake_doc={'sources':[],'fixture_exact12':True}
                design=dict(fit_manifest=manifest_ref,scoring_plan=score_ref,quality=quality_ref,sources=[])
                fit_report=dict(manifest=manifest_ref,continuation=final_ref,intake_evidence=intake_doc,fit={'fixture':True},sources=[])
                calls=[]
                def verified(mp,fp,callback):
                    calls.append('verified_intake');self.assertEqual(mp,manifest_ref['path']);self.assertEqual(fp,final_ref['path'])
                    return callback({'verified_bundle':True},intake_doc)
                def evaluate(bundle,fitted,score,quality,output):
                    calls.append('evaluate');Path(output).mkdir()
                    if mutate:Path(fit_ref['path']).write_text('changed')
                    return {'fixture_result':True}
                with patch.object(runner,'audit',return_value=design), \
                     patch.object(runner.response,'audit_fit',return_value={'report':fit_report,'receipt':fit_ref}), \
                     patch.object(runner.response,'with_verified_bundle',side_effect=verified), \
                     patch.object(runner,'evaluate_bundle',side_effect=evaluate),patch.object(runner.response,'fit_bundle') as fitter:
                    if mutate:
                        with self.assertRaises(ValueError):runner.run(design_ref['path'],fit_ref['path'],root/'output')
                    else:
                        result=runner.checked(runner.run(design_ref['path'],fit_ref['path'],root/'output'))
                        self.assertTrue(result['fixture_result'])
                        self.assertEqual(result['response_fit'],fit_ref)
                    fitter.assert_not_called();self.assertEqual(calls,['verified_intake','evaluate'])

    def test_foreign_saved_fit_manifest_rejected_before_verified_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);design=root/'design';design.write_text('{}')
            with patch.object(runner,'audit',return_value={'fit_manifest':{'path':'expected','sha256':'a'*64}}), \
                 patch.object(runner.response,'audit_fit',return_value={'report':{'manifest':{'path':'foreign','sha256':'b'*64}}}), \
                 patch.object(runner.response,'with_verified_bundle') as callback:
                with self.assertRaises(ValueError):runner.run(design,'fixture-fit',root/'out')
                callback.assert_not_called()

    def test_full_exact546_gate_preserves_new_schema_and_rejects_public_rebinding(self):
        plans,docs,raws=data();original=copy.deepcopy(docs)
        self.assertEqual(compiler.complete_documents(docs,plans,raws)['measured_requests'],546)
        self.assertEqual(docs,original)
        for mutation in ('old_schema','pool','missing_request','manifest','raw_identity'):
            ds,rs=copy.deepcopy(docs),copy.deepcopy(raws)
            if mutation=='old_schema':ds[0]['schema']=compiler.adapter.frozen.MACRO_SCHEMA
            elif mutation=='pool':ds[0]['requests'][0]['macros'][0]['pool_id']='foreign'
            elif mutation=='missing_request':ds[0]['requests'].pop()
            elif mutation=='manifest':ds[0]['source_rebound_manifest']={'path':'foreign','sha256':'0'*64}
            else:rs[0]['execution_id']='foreign'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):compiler.complete_documents(ds,plans,rs)

    def test_fold_compiler_excludes_all_heldout_sources_and_uses_that_fold_coefficients(self):
        plans,docs,raws=data();denominator=compiler.complete_documents(docs,plans,raws)
        blocks=sorted({p['capture_id'] for p in plans});heldout=blocks[1]
        qualified=dict(response_law_compilation_eligible=True,features=list(compiler.adapter.FEATURES),
            fitted_root_numerical_qualification=True,denominator=denominator,
            full_fit=dict(numerical_qualification=True,whole_blocks=blocks,coefficients=['full']),
            whole_block_leaveout=[dict(heldout_block_id=b,
                training_fit=dict(numerical_qualification=True,whole_blocks=[x for x in blocks if x!=b],coefficients=[b]),
                evaluation={'summary':{'unknown_rows':0}},
                training_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=b],
                heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==b]) for b in blocks])
        observed=[]
        def vector(request,row,fit,*args):
            observed.append((request['capture_id'],fit['coefficients']))
            return {'vector_id':'placeholder','public_key':[row['model'],'text','fixture']}
        with patch.object(compiler.joint,'source_messages',return_value=[]),patch.object(compiler.joint,'derive',return_value=[]), \
             patch.object(compiler.joint,'public_source_order',side_effect=lambda raw,plan:raw['priming_request_records']+raw['request_records']), \
             patch.object(compiler.joint,'transport_constraints',return_value={}),patch.object(compiler.joint,'request_vector',side_effect=vector):
            law=compiler.build_joint_law(docs,plans,raws,qualified,heldout_block_id=heldout)
        self.assertEqual(len(observed),364)
        self.assertEqual(law['donor_startup_attempts'],8)
        self.assertNotIn(heldout,law['blocks'])
        self.assertTrue(all(b!=heldout and coeff==[heldout] for b,coeff in observed))
        self.assertEqual(law['source_macro_schema'],compiler.adapter.MACRO_SCHEMA)
        self.assertFalse(law['source_documents_relabelled'])

    def test_actual_full150_unknown_preserves1200_worlds_and_saves_before_labels(self):
        plans,docs,raws=data();rows,denominator=compiler.adapter._rows(docs,plans)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reviews=[]
            for i,(plan,raw) in enumerate(zip(plans,raws)):
                cases=[];requests=[]
                for index,case in enumerate(plan['cases']):
                    epoch=10**12+index*100_000_000_000
                    cases.append(dict(case_id=case['case_id'],observation_start_ns=epoch))
                    requests.extend(dict(request_id=a['request_id'],status='completed',completion_time_ns=epoch+a['arrival_ns']+1000) for a in case['arrivals'])
                raw.update(cases=cases,case_histories=[],request_records=requests)
                pp=root/f'plan{i}.json';pp.write_text(json.dumps(plan))
                rp=root/f'raw{i}.json';rp.write_text(json.dumps(raw))
                rr=root/f'review{i}.json';rr.write_text(json.dumps({'inputs':{'plan.json':runner.receipt(pp)},'raw':runner.receipt(rp)}))
                reviews.append(runner.receipt(rr))
            bundle=dict(schema=compiler.adapter.SCHEMA,design=compiler.adapter.DESIGN,features=list(compiler.adapter.FEATURES),
                denominator=denominator,plans=plans,documents=docs,full={'rows':rows},review_receipts=reviews,
                numerical_fit_run=False,validation_labels_read=False,GPU_started=False)
            fit=dict(denominator=denominator,features=list(compiler.adapter.FEATURES),response_law_compilation_eligible=False)
            design=json.loads(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json').read_text())
            score=runner.checked(design['scoring_plan']);quality=runner.checked(design['quality'])
            output=root/'generated';native=runner.generated.native_labels;calls=[]
            def labels(raw,case,epoch,score,quality):
                preds=list(output.glob('*/case-*/prediction.json'));bounds=list(output.glob('*/case-*/queue-envelope.json'))
                self.assertEqual(len(preds),len(calls)+1);self.assertEqual(len(bounds),len(calls)+1)
                for p in (preds[-1],bounds[-1]):self.assertIsInstance(json.loads(p.read_text()),dict)
                calls.append(case['case_id']);return native(raw,case,epoch,score,quality)
            with patch.object(runner.generated,'native_labels',side_effect=labels),patch.object(runner.laws,'build_joint_law',side_effect=AssertionError('unqualified compile forbidden')):
                result=runner.evaluate_bundle(bundle,fit,score,quality,output)
            self.assertEqual(len(calls),150);self.assertEqual(result['summary']['offered_requests'],546)
            self.assertEqual(result['ensemble_worlds_unknown'],1200);self.assertEqual(result['ensemble_worlds_generated'],0)
            self.assertEqual(result['summary']['point_supported_requests'],0)
            for fold in result['whole_block_leaveout']:
                self.assertEqual(len(fold['donor_attempt_ids']),8);self.assertEqual(len(fold['heldout_attempt_ids']),4)
                for case in fold['cases']:
                    self.assertEqual(case['ensemble_summary']['worlds'],8)
                    self.assertTrue(case['durable_before_case_evaluation_label_extraction'])
            self.assertFalse(result['numerical_fit_run']);self.assertFalse(result['main_queue_provider_eligible'])


if __name__=='__main__':unittest.main()
