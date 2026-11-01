"""Synthetic exact12 resources: pure math is real, native intake is mocked."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_source_rebound_resource_v1 as m
from tests.test_twoFIFO_source_rebound_phase12_fit_v1 import bundle_fixture, save, complete_final
from tests.test_twoFIFO_resource_fold_provider_v1 import timeline_and_binding


def inputs(root):
    bundle = bundle_fixture()
    manifest_ref = save(root/'executor.json', {'CPU_FIXTURE': True})
    actual_manifest = m.checked(m.receipt('artifacts/predictor-twoFIFO-source-rebound-executor-v1-20261028-cpu11/manifest.json'))
    base = m.checked(actual_manifest['attempts'][0]['plan'])  # Public metadata only.
    context = m.checked(base['context'])
    context_ref = save(root/'context.json', context)
    refs, reviews = [], []
    for i, (plan, doc) in enumerate(zip(bundle['plans'], bundle['documents'])):
        identity = copy.deepcopy(plan)
        plan.clear(); plan.update(copy.deepcopy(base)); plan.update(identity); plan['context'] = context_ref
        plan_ref = save(root/(str(i)+'-plan.json'), plan)
        block = int(plan['capture_id'].split('-')[-1])
        rows = []
        for j, request in enumerate(doc['requests']):
            a = 1020+j*20
            rows.append(dict(request_id=request['request_id'], model=base['resident_owners'][0],
                task='text',pool_id=base['priming'][0]['pool_id'],observed_worker_start_time_ns=a,
                previous_handler_next_request_ready=dict(perf_ns=a+10)))
            for k, macro in enumerate(request['macros']):
                macro.update(left_ns=a+5*k,right_ns=a+5*(k+1),semantic_phase='entry' if k==0 else 'ready_tail')
        raw = dict(schema=m.response.executor.RAW_SCHEMA,backend='native',status='completed',split='training',
            CPU_FIXTURE_NOT_NATIVE=True, attempt_id=plan['attempt_id'],capture_id=plan['capture_id'],
            execution_id=plan['execution_id'],plan=plan_ref,context=context_ref,
            observation_start_ns=1000,observation_end_ns=2500,shutdown_memory_sampled=False,
            lifecycle_boundaries=[dict(boundary=n,time_ns=t) for n,t in [
                ('startup_begin',10),('priming_end',900),('measurement_and_drain_end',2500),
                ('monitoring_end',2600),('shutdown_begin',2700)]],
            board_samples=[dict(time_ns=t,query_duration_ns=0,watts=10.*block,owned_used_mib=100.*block,
                owned_aliases=base['resident_owners'],phase='CPU_FIXTURE',foreign_pids=[],
                unresolved_nvml_pids=[],unresolved_ownership_reasons=[]) for t in range(0,2601,2)],
            priming_request_records=[dict(model=p['owner'],task=p['task'],pool_id=p['pool_id'],status='completed')
                                     for p in base['priming']],request_records=rows,
            measurement_windows=[dict(case_id='CPU_CASE',start_ns=1000,end_ns=2000,fixed_drain_deadline_ns=2500)],
            cases=[dict(case_id='CPU_CASE')])
        raw_ref = save(root/(str(i)+'-raw.json'), raw)
        doc.update(schema=m.response.adapter.MACRO_SCHEMA,attempt_id=plan['attempt_id'],capture_id=plan['capture_id'],
            execution_id=plan['execution_id'],source_rebound_manifest=manifest_ref,sources=[raw_ref])
        review = dict(schema=m.response.intake.native_review.SCHEMA,
            status='PASS_COMPLETE_SOURCE_REBOUND_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION',
            training_feature_extraction_eligible=True,issues=[],validation_labels_read=False,
            source_rebound_manifest=manifest_ref,inputs={'plan.json':plan_ref},raw=raw_ref,
            sources=[raw_ref,plan_ref,context_ref])
        review_ref = save(root/(str(i)+'-review.json'), review)
        reviews.append(review_ref); refs.extend([raw_ref,plan_ref,context_ref,review_ref])
    bundle['review_receipts'] = reviews
    intake = dict(schema=m.response.intake.SCHEMA,
        status='COMPLETE_SOURCE_REBOUND_EXACT12_TRAINING_INTAKE_NOT_FIT',attempt_count=12,
        measured_requests=546,priming_requests=216,validation_labels_read=False,
        manifest=manifest_ref,sources=m.strict_references(refs))
    return bundle, intake, context


def prepared(root):
    executor = save(root/'executor.json', {'CPU_FIXTURE':True})
    final = root/'run01'/'report.json'
    manifest = dict(executor_manifest=executor,sources=[],expected_continuation_report_path=str(final))
    fit_manifest = save(root/'fit'/'manifest.json', manifest)
    with patch.object(m.response,'audit',return_value=manifest):
        design = m.prepare(fit_manifest['path'],root/'resource-design')
    return design, manifest, final


class SourceReboundResourcesTests(unittest.TestCase):
    def test_new_schema_raw_and_macros_reach_real_pure_table_math(self):
        with tempfile.TemporaryDirectory() as work:
            bundle,intake,_=inputs(Path(work))
            original=m.resource.table_from_verified;calls=[]
            def table(raw,doc,plan,ctx):
                self.assertEqual(raw['schema'],m.response.executor.RAW_SCHEMA)
                self.assertEqual(doc['schema'],m.response.adapter.MACRO_SCHEMA)
                calls.append(plan['attempt_id']);return original(raw,doc,plan,ctx)
            with patch.object(m.resource,'table_from_verified',side_effect=table), \
                 patch.object(m.resource,'load_complete',side_effect=AssertionError('old loader forbidden')), \
                 patch.object(m.response.original,'fitted_rows',side_effect=AssertionError('response fitter forbidden')):
                result=m.build_from_bundle(bundle,intake)
            self.assertEqual(len(calls),12);self.assertEqual(result['measured_requests'],546)
            self.assertEqual(result['history_only_priming_requests'],216)
            self.assertEqual(len(result['whole_block_leaveout']),3)
            for fold in result['whole_block_leaveout']:
                self.assertEqual(len(fold['training_model']['training_attempt_ids']),8)
                self.assertEqual(len(fold['heldout_attempt_ids']),4)
                self.assertNotIn(fold['heldout_block_id'],fold['training_model']['training_block_ids'])
            self.assertFalse(result['resource_model']['startup_energy_already_amortized'])
            self.assertFalse(result['resource_model']['safe_factor_already_applied'])

    def test_prepare_audit_and_missing_final_do_not_touch_labels(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);design,manifest,_=prepared(root)
            with patch.object(m.response,'audit',return_value=manifest), \
                 patch.object(m.response,'with_verified_bundle',side_effect=AssertionError('labels forbidden')):
                self.assertEqual(m.audit(design['path'])['design'],m.DESIGN)
                out=m.checked(m.run(design['path'],root/'waiting',fit_enabled=True))
            self.assertEqual(out['status'],'WAITING_FOR_EXACT12_CONTINUATION_FINAL')
            self.assertFalse(out['native_training_resource_labels_read'])

    def test_preparation_after_final_or_response_fit_claim_is_rejected(self):
        for kind in ('final','claim'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as work:
                root=Path(work);design,manifest,final=prepared(root)
                if kind=='final':save(final,complete_final(manifest['executor_manifest']))
                else:save(root/'fit'/'phase12-fit-claim.json',{'CPU':True})
                with patch.object(m.response,'audit',return_value=manifest),self.assertRaises(ValueError):
                    m.prepare(m.checked(design)['fit_manifest']['path'],root/'late')

    def test_partial_foreign_and_default_off_complete_never_open_native(self):
        for kind in ('partial','foreign','labels','complete'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as work:
                root=Path(work);design,manifest,final=prepared(root)
                value=complete_final(manifest['executor_manifest'])
                if kind=='partial':value['completed_attempts'].pop()
                elif kind=='foreign':value['manifest']={}
                elif kind=='labels':value['validation_labels_read']=True
                save(final,value)
                with patch.object(m.response,'audit',return_value=manifest), \
                     patch.object(m.response,'with_verified_bundle',side_effect=AssertionError('labels forbidden')):
                    out=m.checked(m.run(design['path'],root/'out'))
                self.assertEqual(out['status'],'READY_EXPLICIT_RESOURCE_FIT_REQUIRED' if kind=='complete'
                                 else 'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL')

    def test_bad_intake_and_changed_raw_schema_rejected_before_table_math(self):
        with tempfile.TemporaryDirectory() as work:
            bundle,intake,_=inputs(Path(work));bad=copy.deepcopy(intake);bad['measured_requests']=545
            with patch.object(m.resource,'table_from_verified',side_effect=AssertionError('must gate first')):
                with self.assertRaisesRegex(ValueError,'true exact12'):m.build_from_bundle(bundle,bad)
            review=m.checked(bundle['review_receipts'][0]);raw_path=Path(review['raw']['path'])
            raw=m.checked(review['raw']);raw['schema']='OLD_SCHEMA'
            raw_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'Changed'):m.build_from_bundle(bundle,intake)

    def test_true_coordinator_intake_adapter_resource_order_and_provider_rederive(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work);design,manifest,final=prepared(root)
            bundle,intake,context=inputs(root/'data');save(final,complete_final(manifest['executor_manifest']))
            calls=[]
            def ingest(*args,output):calls.append('intake');return save(Path(output)/'report.json',intake)
            def adapted(*args,**kwargs):calls.append('adapter');return bundle
            real=m.build_from_bundle
            def build(*args):calls.append('resources');return real(*args)
            with patch.object(m.response,'audit',return_value=manifest), \
                 patch.object(m.response.intake,'intake',side_effect=ingest), \
                 patch.object(m.response.adapter,'build_training_rows',side_effect=adapted), \
                 patch.object(m,'build_from_bundle',side_effect=build), \
                 patch.object(m.resource,'load_complete',side_effect=AssertionError('old loader forbidden')):
                model=m.run(design['path'],root/'model',fit_enabled=True)
                self.assertEqual(calls,['intake','adapter','resources'])
                target=next(p for p in bundle['plans'] if p['capture_id']=='training-block-3')
                provider=m.FoldResourceTrainingProvider(model['path'],target,context,
                    heldout_block_id='training-block-3',allow_training_resource_hypothesis=True)
                self.assertEqual(calls,['intake','adapter','resources']*2)
                self.assertEqual(len(provider.document['training_attempt_ids']),8)
                self.assertNotIn('training-block-3',provider.document['training_block_ids'])
                self.assertTrue(provider.fold_binding['all_three_folds_rederived'])
                # Pure timeline/physical matching and inherited lookup use the
                # same contract as actual_worker_world_score_v3.
                board,binding=timeline_and_binding(provider)
                prediction=m.predict_timeline(board,provider,target_context=context,execution_binding=binding)
                self.assertAlmostEqual(prediction['board_energy_joules'],15.)
                self.assertEqual(prediction['startup_energy_joules'],provider.startup()['joules'])
                self.assertFalse(prediction['safe_factor_already_applied'])
                no_history=copy.deepcopy(binding);no_history.pop('memory_initial_state')
                unknown=m.predict_timeline(board,provider,target_context=context,execution_binding=no_history)
                self.assertAlmostEqual(unknown['board_energy_joules'],15.)
                self.assertIsNone(unknown['peak_memory_mib'])
                self.assertFalse(unknown['memory_initial_history_supplied'])
                changed=copy.deepcopy(context);changed['threads']['OMP_NUM_THREADS']='999'
                with self.assertRaisesRegex(ValueError,'physical'):
                    m.FoldResourceTrainingProvider(model['path'],target,changed,
                        heldout_block_id='training-block-3',allow_training_resource_hypothesis=True)
                doc=m.checked(model);doc['whole_block_leaveout'][0]['training_model']['startup']['joules']+=1
                altered=save(root/'altered.json',doc)
                with self.assertRaisesRegex(ValueError,'must rederive'):
                    m.FoldResourceTrainingProvider(altered['path'],target,context,
                        heldout_block_id='training-block-3',allow_training_resource_hypothesis=True)

    def test_fold_heldout_poison_minimum_two_blocks_and_history_unknown(self):
        from tests.test_twoFIFO_resource_fold_provider_v1 import fixture
        plan,context,tables=fixture();full,folds=m._fits(tables)
        selected=folds[2]['training_model'];changed=copy.deepcopy(tables)
        for table in changed:
            if table['capture_id']=='training-block-3':
                table['startup_joules']=1e12;table['slices'][0]['power_training_joules']=1e12
        self.assertEqual(m._fits(changed)[1][2]['training_model'],selected)
        for table in changed:
            if table['capture_id']=='training-block-2':
                table['slices'][0].update(power_training_duration_ns=0,sampled_owned_peak_mib=None)
        selected=m._fits(changed)[1][2]['training_model']
        self.assertTrue(all(not s['known'] for s in selected['power_states'].values()))
        self.assertTrue(all(not s['known'] for s in selected['memory_history_states'].values()))
        provider=object.__new__(m.FoldResourceTrainingProvider);provider.document=full;provider.scenario='mean'
        self.assertFalse(provider.memory([],resident_owners=plan['resident_owners'],
            ordered_state_history_identity='UNKNOWN')['known'])


if __name__=='__main__':unittest.main()
