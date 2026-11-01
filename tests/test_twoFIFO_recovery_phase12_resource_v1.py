"""CPU fixtures only: new recovery producer and unchanged numerical interfaces."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_recovery_phase12_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_recovery_resource_v1 as r
from tests.test_twoFIFO_source_rebound_phase12_fit_v1 import bundle_fixture, save
from tests.test_twoFIFO_source_rebound_resource_v1 import inputs
from tests.test_twoFIFO_resource_fold_provider_v1 import timeline_and_binding


class RecoveryPhaseResourcesTests(unittest.TestCase):
    def fixture(self, root, *, resource_inputs=False):
        if resource_inputs:
            bundle, document, context = inputs(root/'data')
        else:
            bundle=bundle_fixture();context={}
            document=dict(manifest=save(root/'original.json',{'CPU_FIXTURE':True}),sources=[])
        original=document['manifest']
        stopped=save(root/'stopped.json',{'status':'STOPPED_NO_RETRY','CPU_FIXTURE':True})
        items=[dict(attempt_id=p['attempt_id']) for p in bundle['plans']]
        recovered=dict(proof=dict(original_manifest=original,stopped_report=stopped,
            prefix=items[:4],tail=items[4:]),run_output=str(root/'run02'),sources=[])
        recovery_ref=save(root/'recovery.json',recovered)
        stack=ExitStack()
        stack.enter_context(patch.object(m.recovery,'audit',return_value=recovered))
        stack.enter_context(patch.object(m,'_entries',side_effect=lambda:[m.receipt(m.__file__)]))
        stack.enter_context(patch.object(m,'implementation_sources',side_effect=lambda _:[m.receipt(m.__file__)]))
        stack.enter_context(patch.object(r,'implementation_sources',side_effect=lambda _:[m.receipt(r.__file__)]))
        review=save(root/'independent.json',dict(schema='CPU.independent',status='PASS_CPU',sources=[m.receipt(m.__file__)]))
        final=root/'run02'/'report.json'
        manifest_ref=m.prepare(recovery_ref['path'],final,[review['path']],root/'phase12')
        manifest=m.audit(manifest_ref['path'])
        document.update(schema=m.intake.SCHEMA,status=m.INTAKE_STATUS,recovery_manifest=recovery_ref,
            recovery_policy=m.recovery.POLICY,stopped_report=stopped,preserved_prefix_attempts=4,
            new_policy_tail_attempts=8,attempt_count=12,measured_requests=546,priming_requests=216,
            numerical_fit_run=False,validation_labels_read=False)
        return stack,bundle,document,context,recovered,manifest_ref,manifest,final

    def final(self,recovered,manifest):
        return dict(schema=m.recovery.SCHEMA+'.run',status=m.recovery.FINAL,policy=m.recovery.POLICY,
            recovery_manifest=manifest['recovery_manifest'],original_manifest=manifest['original_executor_manifest'],
            stopped_report=recovered['proof']['stopped_report'],prefix_attempts=recovered['proof']['prefix'],
            completed_attempts=recovered['proof']['prefix']+recovered['proof']['tail'],
            completed_tail=recovered['proof']['tail'],no_started_capture_retries=True,fit_eligible=False,
            numerical_fit_run=False,automatic_fit=False,validation_labels_read=False)

    def native_mocks(self,stack,bundle,document,events):
        def ingest(*args,output,**kwargs):
            events.append('new_intake');return save(Path(output)/'report.json',document)
        def rows(*args,**kwargs):
            events.append('rederived_native_macros');return bundle['plans'],bundle['documents'],bundle.get('review_receipts',[])
        stack.enter_context(patch.object(m.intake,'intake',side_effect=ingest))
        stack.enter_context(patch.object(m.adapter,'_rederive',side_effect=rows))
        stack.enter_context(patch.object(m.adapter,'_rows',return_value=(bundle['full']['rows'],bundle['denominator'])))
        stack.enter_context(patch.object(m.adapter.frozen,'observed_support',return_value=bundle['observed_support']))
        stack.enter_context(patch.object(m.numeric.original.numeric,'fit',side_effect=AssertionError('optimizer forbidden')))

    def test_prepare_audit_missing_defaultoff_and_foreign_final_never_touch_intake(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root)
            with s,patch.object(m.intake,'intake',side_effect=AssertionError('native forbidden')):
                self.assertEqual(m.audit(ref['path'])['upstream_intake_schema'],m.intake.SCHEMA)
                self.assertEqual(m.checked(m.run(ref['path'],root/'waiting',fit_enabled=True))['status'],
                    'WAITING_FOR_RECOVERY_EXACT12_FINAL')
                value=self.final(rec,manifest);save(final,value)
                self.assertEqual(m.checked(m.run(ref['path'],root/'ready'))['status'],'READY_EXPLICIT_RECOVERY_FIT_REQUIRED')
                value['schema']=m.numeric.executor.CONT_SCHEMA+'.run';save(final,value)
                self.assertEqual(m.checked(m.run(ref['path'],root/'hold',fit_enabled=True))['status'],
                    'HOLD_INCOMPLETE_OR_FOREIGN_RECOVERY_FINAL')
                self.assertFalse((root/'phase12'/'phase12-fit-claim.json').exists())

    def test_complete_role_rejects_changed_prefix_tail_policy_flags(self):
        with tempfile.TemporaryDirectory() as work:
            s,b,d,c,rec,ref,manifest,final=self.fixture(Path(work))
            with s:
                base=self.final(rec,manifest);self.assertTrue(m.complete_final(manifest,base))
                for kind in ('prefix','tail','policy','labels','fit','foreign'):
                    bad=copy.deepcopy(base)
                    if kind=='prefix':bad['prefix_attempts']=[]
                    elif kind=='tail':bad['completed_tail'][0]['attempt_id']='foreign'
                    elif kind=='policy':bad['policy']='old'
                    elif kind=='labels':bad['validation_labels_read']=True
                    elif kind=='fit':bad['numerical_fit_run']=True
                    else:bad['recovery_manifest']={}
                    with self.subTest(kind=kind):self.assertFalse(m.complete_final(manifest,bad))

    def test_real_callback_row_interface_and_old_intake_rejection(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root)
            with s:
                d['continuation']=save(final,self.final(rec,manifest));events=[]
                self.native_mocks(s,b,d,events)
                def callback(bundle,document):
                    events.append('callback')
                    self.assertEqual(bundle['schema'],m.adapter.SCHEMA)
                    self.assertEqual(bundle['producer_schema'],m.SCHEMA+'.rows')
                    self.assertEqual(bundle['upstream_intake_schema'],m.intake.SCHEMA)
                    self.assertEqual(document['schema'],m.intake.SCHEMA)
                    return len(bundle['full']['rows'])
                self.assertEqual(m.with_verified_bundle(ref['path'],final,callback),1092)
                self.assertEqual(events,['new_intake','rederived_native_macros','callback'])
                wrong=copy.deepcopy(d);wrong['schema']=m.numeric.intake.SCHEMA
                with self.assertRaisesRegex(ValueError,'new recovery intake'):m.check_intake(wrong,manifest)
                with self.assertRaisesRegex(ValueError,'new recovery intake'):m.row_bundle(wrong,root/'badrows')

    def test_source_drift_in_callback_is_rejected_on_exit(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root)
            with s:
                marker=root/'evidence.json';marker_ref=save(marker,{'immutable':True})
                d['sources']=[marker_ref];d['continuation']=save(final,self.final(rec,manifest))
                self.native_mocks(s,b,d,[])
                def corrupt(*args):save(marker,{'immutable':False})
                with self.assertRaisesRegex(ValueError,'Changed source/evidence'):
                    m.with_verified_bundle(ref['path'],final,corrupt)

    def test_explicit_fit_reuses_unchanged_math_and_saved_audit_no_refit(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root)
            with s:
                d['continuation']=save(final,self.final(rec,manifest));events=[];self.native_mocks(s,b,d,events)
                with patch.object(m.numeric.original,'fitted_rows',return_value=(dict(numerical_qualification=False),None)) as fitter:
                    result=m.run(ref['path'],root/'fit',fit_enabled=True)
                    self.assertEqual(fitter.call_count,4)
                    doc=m.checked(result);self.assertEqual(doc['fit']['macro_rows'],1092)
                    self.assertFalse(doc['fit']['response_law_compilation_eligible'])
                    self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in doc['fit']['whole_block_leaveout']),1092)
                    self.assertFalse(m.audit_fit(result)['numeric_recomputed']);self.assertEqual(fitter.call_count,4)
                    with self.assertRaises(FileExistsError):m.run(ref['path'],root/'retry',fit_enabled=True)

    def test_failed_intake_consumes_fit_claim_without_numeric_call(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root)
            with s:
                save(final,self.final(rec,manifest))
                with patch.object(m.intake,'intake',side_effect=ValueError('incomplete recovery')), \
                     patch.object(m.numeric,'fit_bundle',side_effect=AssertionError('numeric forbidden')):
                    result=m.checked(m.run(ref['path'],root/'failed',fit_enabled=True))
                self.assertEqual(result['status'],'STOPPED_RECOVERY_FIT_NO_RETRY')
                self.assertFalse(result['numerical_fit_run'])
                self.assertTrue((root/'phase12'/'phase12-fit-claim.json').is_file())

    def test_resource_real_tables_provider_rederivation_and_original_scopes(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);s,b,d,c,rec,ref,manifest,final=self.fixture(root,resource_inputs=True)
            with s:
                design=r.prepare(ref['path'],root/'resource-design')
                self.assertEqual(r.checked(r.run(design['path'],root/'waiting'))['status'],'WAITING_FOR_RECOVERY_EXACT12_FINAL')
                d['continuation']=save(final,self.final(rec,manifest));events=[];self.native_mocks(s,b,d,events)
                with patch.object(m.intake,'intake',side_effect=AssertionError('default off')):
                    self.assertEqual(r.checked(r.run(design['path'],root/'ready'))['status'],
                                     'READY_EXPLICIT_RECOVERY_RESOURCE_FIT_REQUIRED')
                model=r.run(design['path'],root/'resources',fit_enabled=True)
                target=next(p for p in b['plans'] if p['capture_id']=='training-block-3')
                provider=r.FoldResourceTrainingProvider(model['path'],target,c,heldout_block_id='training-block-3',
                    allow_training_resource_hypothesis=True)
                self.assertEqual(len(provider.document['training_attempt_ids']),8)
                self.assertNotIn('training-block-3',provider.document['training_block_ids'])
                board,binding=timeline_and_binding(provider)
                predicted=r.predict_timeline(board,provider,target_context=c,execution_binding=binding)
                self.assertAlmostEqual(predicted['board_energy_joules'],15.)
                self.assertFalse(predicted['startup_energy_already_amortized'])
                self.assertFalse(predicted['safe_factor_already_applied'])
                binding.pop('memory_initial_state')
                self.assertIsNone(r.predict_timeline(board,provider,target_context=c,execution_binding=binding)['peak_memory_mib'])
                with self.assertRaisesRegex(ValueError,'heldout target'):
                    r.FoldResourceTrainingProvider(model['path'],b['plans'][0],c,heldout_block_id='training-block-3',
                        allow_training_resource_hypothesis=True)
                changed=m.checked(model);changed['whole_block_leaveout'][0]['training_model']['startup']['joules']+=1
                bad=save(root/'bad-model.json',changed)
                with self.assertRaisesRegex(ValueError,'every fold must rederive'):
                    r.FoldResourceTrainingProvider(bad['path'],target,c,heldout_block_id='training-block-3',
                        allow_training_resource_hypothesis=True)


if __name__=='__main__':unittest.main()
