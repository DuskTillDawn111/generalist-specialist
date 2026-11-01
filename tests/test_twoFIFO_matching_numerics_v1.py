"""CPU synthetic matching-cohort adapters; no native fit or GPU launch."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_twoFIFO_training_intake_v4 import documents
from tools.predictor_validation_20261027 import twoFIFO_matching_phase12_adapter_v1 as adapter
from tools.predictor_validation_20261027 import twoFIFO_matching_phase12_v1 as phase
from tools.predictor_validation_20261027 import twoFIFO_matching_generated_v1 as generated
from tools.predictor_validation_20261027 import twoFIFO_matching_joint_law_v1 as joint
from tools.predictor_validation_20261027 import twoFIFO_matching_resource_v1 as resource
from tools.predictor_validation_20261027 import macro_response_queue_reference_v1 as rational

ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'artifacts/predictor-twoFIFO-matching-owned-producer-v1-20261029-cpu02-prepared/manifest.json'
SCORE_DESIGN=ROOT/'artifacts/predictor-twoFIFO-source-rebound-generated-v1-20261029-cpu01-prepared/design.json'


def put(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))
    return phase.receipt(path)


def matching_fixture():
    manifest=phase.checked(phase.receipt(PRODUCER))
    plans=[phase.checked(a['plan']) for a in manifest['attempts']]
    old=documents(plans);docs=copy.deepcopy(old)
    for doc in docs:
        doc.update(schema=adapter.MACRO_SCHEMA,fitted_model_qualified=False,
            matching_manifest=phase.receipt(PRODUCER))
        for request in doc['requests']:
            request['pool_id']=request['macros'][0]['pool_id']
    return plans,old,docs


def bundle_fixture(directory):
    plans,_,docs=matching_fixture()
    with patch.object(adapter,'_load_report',return_value={}),patch.object(adapter,'_rederive',return_value=(plans,docs,[])):
        return adapter.build_training_rows('CPU_SYNTHETIC_ONLY',workdir=directory)


def resource_tables():
    return [dict(schema=resource.TABLE_SCHEMA,resource_domain=copy.deepcopy(resource.DOMAIN),capture_id='block-'+str(b),
        attempt_id='block-'+str(b)+'-shard-'+str(s),physical_execution_identity='physical',
        physical_execution_payload={'CPU_SYNTHETIC_ONLY':True},priming_content_identity='prime',
        resident_owners=['text#0'],startup_joules=10.*b,premeasurement_owned_sampled_peak_mib=100.*b,
        slices=[dict(state_key='state',memory_state_key='history',power_training_joules=2.*b,
            power_training_duration_ns=1_000_000_000,sampled_owned_peak_mib=1000.*b)])
        for b in (1,2,3) for s in (1,2,3,4,5)]


def timeline(seconds=10,drain=60):
    return rational.encode(dict(schema='gsdeploy.macro_global_resource_timeline.v1',
        original_window=[F(0),F(seconds*10**9)],original_drain_end_ns=F((seconds+drain)*10**9),
        total_wall_ns=F((seconds+drain)*10**9),total_measurement_overlap_ns=F(seconds*10**9),
        slices=[dict(duration_ns=F(seconds*10**9),measurement_overlap_ns=F(seconds*10**9),
            resident_owners=list(phase.executor.OWNERS),actual_START_to_READY_handlers=[]),
            dict(duration_ns=F(drain*10**9),measurement_overlap_ns=F(0),
            resident_owners=list(phase.executor.OWNERS),actual_START_to_READY_handlers=[])]))


class MatchingNumericalContracts(unittest.TestCase):
    def test_rows_equal_original_math_and_retain_matching_identities(self):
        plans,old,docs=matching_fixture()
        expected=adapter.frozen.prepared_rows(old,adapter.FEATURES)
        actual,denominator=adapter._rows(docs,plans)
        self.assertEqual((len(actual),len(adapter.FEATURES)),(390,12))
        self.assertEqual(denominator,dict(measured_requests=390,priming_requests_excluded_as_labels=270,training_replication_units=3))
        for left,right in zip(expected,actual):
            self.assertEqual(set(left),set(right))
            for key in left:
                if isinstance(left[key],np.ndarray):np.testing.assert_array_equal(left[key],right[key])
                else:self.assertEqual(left[key],right[key])
        self.assertTrue(all(d['schema']==adapter.MACRO_SCHEMA for d in docs))

    def test_bundle_uses_three_whole_10_5_folds_without_optimizer(self):
        with tempfile.TemporaryDirectory() as root,patch.object(phase.original,'fitted_rows') as fitter:
            bundle=bundle_fixture(root);rows,blocks=phase._validate_bundle(bundle)
        fitter.assert_not_called();self.assertEqual((len(rows),len(blocks)),(390,3))
        for fold in bundle['whole_block_folds']:
            self.assertEqual((len(fold['training_attempt_ids']),len(fold['heldout_attempt_ids'])),(10,5))
            self.assertTrue(set(fold['training_attempt_ids']).isdisjoint(fold['heldout_attempt_ids']))
            self.assertFalse(fold['fit_performed'])

    def test_OCR_target_set_nonempty_and_missing_actual_exposure_UNKNOWN(self):
        plans,_,docs=matching_fixture()
        pools={a['pool_id'] for p in plans for c in p['cases'] for a in c['arrivals'] if a['owner']=='ocr#0'}
        self.assertTrue(pools)
        support=adapter.observed_support(docs,plans)
        self.assertEqual(len(support['OCR_exact_input_by_whole_block']),3*len(pools))
        for doc in docs:
            for request in doc['requests']:
                for macro in request['macros']:
                    if macro['owner']=='ocr#0':
                        for piece in macro['causal_covariate_path']:
                            piece['state'].update(peer_visible_macro_owner_count=0,peer_macros=[])
        support=adapter.observed_support(docs,plans)
        self.assertFalse(support['each_target_OCR_input_each_block_n0_and_npositive'])
        self.assertTrue(all(r['status'].startswith('UNKNOWN') for r in support['OCR_exact_input_by_whole_block']))
        empty=copy.deepcopy(plans)
        for plan in empty:
            for case in plan['cases']:case['arrivals']=[a for a in case['arrivals'] if a['owner']!='ocr#0']
        with self.assertRaisesRegex(ValueError,'nonempty OCR'):adapter.observed_support(docs,empty)

    def test_invalid_macro_path_schema_or_denominator_rejected(self):
        plans,_,docs=matching_fixture()
        for mutation in ('schema','path','request','feature'):
            ds=copy.deepcopy(docs)
            if mutation=='schema':ds[0]['schema']=adapter.frozen.MACRO_SCHEMA
            elif mutation=='path':ds[0]['requests'][0]['macros'][0]['causal_covariate_path'][0]['left_ns']+=1
            elif mutation=='request':ds[0]['requests'][0]['worker_start_to_ready_ns']+=1
            else:ds[0]['requests'][0]['macros'][0]['causal_covariate_path'][0]['state']['peer_visible_macro_owner_count']=99
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):adapter._rows(ds,plans)

    def test_mock_failed_roots_retain_all_390_heldout_macros_UNKNOWN(self):
        calls=[]
        def stub(rows):
            calls.append({r['capture'] for r in rows})
            return dict(numerical_qualification=False,whole_blocks=sorted(calls[-1])),None
        with tempfile.TemporaryDirectory() as root:
            bundle=bundle_fixture(root)
            with patch.object(phase.original,'fitted_rows',side_effect=stub),patch.object(phase.original,'evaluate') as evaluator:
                fitted=phase.fit_bundle(bundle)
        evaluator.assert_not_called();self.assertEqual(len(calls),4)
        self.assertEqual([len(c) for c in calls],[3,2,2,2])
        self.assertFalse(fitted['response_law_compilation_eligible'])
        self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in fitted['whole_block_leaveout']),390)
        self.assertTrue(all(len(f['training_attempt_ids'])==10 and len(f['heldout_attempt_ids'])==5 for f in fitted['whole_block_leaveout']))

    def test_mock_successful_evaluator_keeps_exact_heldout_identity_and_raw_provenance(self):
        def fit(rows):return dict(numerical_qualification=True,whole_blocks=sorted({r['capture'] for r in rows})),('MOCK',)
        def evaluate(rows,*args):
            self.assertEqual(args,('MOCK',))
            return dict(summary=dict(rows=len(rows),supported_rows=len(rows),unknown_rows=0,mean_nll=1.),
                records=[dict(capture_sha256=r['capture'],request_id=r['request_id'],macro_id=r['macro_id'],support='MOCK') for r in rows])
        with tempfile.TemporaryDirectory() as root:
            bundle=bundle_fixture(root);bundle['observed_support']['each_target_OCR_input_each_block_n0_and_npositive']=True
            with patch.object(phase.original,'fitted_rows',side_effect=fit),patch.object(phase.original,'evaluate',side_effect=evaluate):
                fitted=phase.fit_bundle(bundle)
        self.assertTrue(fitted['response_law_compilation_eligible'])
        records=[r for f in fitted['whole_block_leaveout'] for r in f['evaluation']['records']]
        self.assertEqual(len(records),390)
        self.assertTrue(all(r['raw_sha256'].startswith('raw-') and r['block_id']!=r['raw_sha256'] for r in records))

    def test_joint_compiler_requires_all_new_raw_roles_and_270_prime_IDs(self):
        plans,_,docs=matching_fixture();raws=[]
        for doc,plan in zip(docs,plans):
            raws.append(dict(schema=phase.executor.RAW_SCHEMA,backend='native',status='completed',split='training',
                **{k:plan[k] for k in ('capture_id','attempt_id','execution_id')},
                priming_request_records=[dict(request_id=r['request_id']) for r in plan['priming']],
                request_records=[dict(request_id=r['request_id']) for r in doc['requests']]))
        self.assertEqual(joint.complete_documents(docs,plans,raws)['measured_requests'],390)
        for mutation in ('oldschema','primeid','measuredorder','missing'):
            changed=copy.deepcopy(raws)
            if mutation=='oldschema':changed[0]['schema']='gsdeploy.original_twoFIFO_owned_training.v3.raw'
            elif mutation=='primeid':changed[0]['priming_request_records'][0]['request_id']='other'
            elif mutation=='measuredorder':changed[0]['request_records'].reverse()
            else:changed.pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):joint.complete_documents(docs,plans,changed)

    def test_all171_real_queue_forecasts_save_before_labels_and_keep1368_UNKNOWN_worlds(self):
        score_design=phase.checked(phase.receipt(SCORE_DESIGN))
        score,quality=phase.checked(score_design['scoring_plan']),phase.checked(score_design['quality'])
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);bundle=bundle_fixture(root);reviews=[]
            for i,plan in enumerate(bundle['plans']):
                cases=[];records=[]
                for j,case in enumerate(plan['cases']):
                    epoch=(100+j*100)*10**9;cases.append(dict(case_id=case['case_id'],observation_start_ns=epoch))
                    records.extend(dict(request_id=a['request_id'],status='completed',completion_time_ns=epoch+a['arrival_ns']+10**6) for a in case['arrivals'])
                raw=put(root/f'raw-{i}.json',dict(CPU_SYNTHETIC_ONLY=True,cases=cases,request_records=records,
                    case_histories=[dict(case_id=c['case_id'],prior_history_snapshot=None) for c in cases]))
                pp=put(root/f'plan-{i}.json',plan)
                reviews.append(put(root/f'review-{i}.json',dict(CPU_SYNTHETIC_ONLY=True,raw=raw,inputs={'plan.json':pp})))
            bundle['review_receipts']=reviews
            fitted=dict(denominator=bundle['denominator'],features=list(phase.FEATURES),response_law_compilation_eligible=False)
            original=generated.generated.native_labels;seen=[]
            def labels(raw,case,*args):
                matches=list((root/'predictions').glob('*/case-*/prediction.json'))
                bounds=list((root/'predictions').glob('*/case-*/queue-envelope.json'))
                seen.append(case['case_id']);self.assertEqual(len(matches),len(seen));self.assertEqual(len(bounds),len(seen))
                return original(raw,case,*args)
            with patch.object(generated.generated,'native_labels',side_effect=labels):
                report=generated.evaluate_bundle(bundle,fitted,score,quality,root/'predictions')
            self.assertEqual((report['forecasts_generated'],report['ensemble_worlds_unknown'],report['ensemble_worlds_generated']),(171,1368,0))
            self.assertEqual(report['summary']['offered_requests'],390)
            self.assertFalse(report['heldout_future_service_labels_used_for_generation'])
            for fold in report['whole_block_leaveout']:
                self.assertEqual((len(fold['donor_attempt_ids']),len(fold['heldout_attempt_ids'])),(10,5))
                for case in fold['cases']:
                    pred=phase.checked(case['prediction']);self.assertEqual(F(**pred['ensemble']['world_unknown_probability_mass']),1)

    def test_resource_15_tables_equal_block_means_and_all_10_5_folds(self):
        tables=resource_tables();full,folds=resource._fits(tables)
        self.assertEqual(full['power_states']['state']['mean'],4.)
        self.assertEqual(full['memory_history_states']['history']['mean'],2000.)
        self.assertEqual(full['startup']['joules'],20.)
        self.assertEqual(full['resource_domain'],resource.DOMAIN)
        for fold in folds:
            self.assertEqual((len(fold['training_model']['training_attempt_ids']),len(fold['heldout_attempt_ids'])),(10,5))
        with self.assertRaises(ValueError):resource._fits(tables[:-1])
        changed=copy.deepcopy(tables);changed[0]['physical_execution_identity']='other'
        with self.assertRaises(ValueError):resource.fit_tables(changed)

    def test_resource_physical_axes_exclude_prime_IDs_but_bind_contents_and_context(self):
        plans,_,_=matching_fixture();context=phase.checked(plans[0]['context'])
        identities=[resource.physical_execution_identity(p,context) for p in plans]
        self.assertEqual(len(set(identities)),1)
        self.assertEqual(len({r['request_id'] for p in plans for r in p['priming']}),270)
        changed=copy.deepcopy(context);changed['threads']['OMP_NUM_THREADS']='99'
        self.assertNotEqual(resource.physical_execution_identity(plans[0],changed),identities[0])
        changed_plan=copy.deepcopy(plans[0]);changed_plan['priming'].reverse()
        self.assertNotEqual(resource.physical_execution_identity(changed_plan,context),identities[0])

    def provider(self):
        plans,_,_=matching_fixture();plan=plans[0];context=phase.checked(plan['context'])
        provider=object.__new__(resource.FoldResourceTrainingProvider)
        provider.target_plan=plan;provider.target_context=context
        provider.physical_execution_identity=resource.physical_execution_identity(plan,context)
        provider.document=dict(priming_content_identity='prime',resource_domain=copy.deepcopy(resource.DOMAIN))
        provider.fold_binding={'CPU_SYNTHETIC_ONLY':True}
        provider.power=lambda *a,**k:dict(known=True,watts=20.)
        provider.memory=lambda *a,**k:dict(known=True,peak_memory_mib=300.)
        provider.startup=lambda:dict(joules=40.,premeasurement_owned_sampled_peak_mib=400.)
        return provider,context

    def predict(self,t,provider,context,initial=None):
        binding=dict(physical_execution_identity=resource.physical_execution_identity(provider.target_plan,context),
            resource_timeline_sha256=resource.digest(t),basis='caller_declared_explicit_reference_execution_binding')
        if initial is not None:binding['memory_initial_state']=initial
        return resource.predict_timeline(t,provider,target_context=context,execution_binding=binding)

    def test_supported_resource_math_energy_once_and_memory_history_never_reset(self):
        provider,context=self.provider();t=timeline()
        result=self.predict(t,provider,context,dict(basis='fresh_after_original_priming'))
        self.assertEqual((result['board_energy_joules'],result['peak_memory_mib'],result['startup_energy_joules']),(200.,400.,40.))
        self.assertFalse(result['startup_energy_already_amortized']);self.assertFalse(result['safe_factor_already_applied'])
        result=self.predict(t,provider,context)
        self.assertIsNone(result['peak_memory_mib'])
        self.assertEqual(F(**result['memory_unknown_observation_duration_ns']),70*10**9)
        self.assertEqual(result['board_energy_joules'],200.)

    def test_90_plus120_or_context_drift_all_resource_exposure_stays_UNKNOWN(self):
        provider,context=self.provider()
        for kind in ('horizon','context','domain'):
            t=timeline(90,120) if kind=='horizon' else timeline()
            changed=copy.deepcopy(context)
            if kind=='context':changed['threads']['OMP_NUM_THREADS']='99'
            if kind=='domain':provider.document['resource_domain']={}
            with patch.object(provider,'power') as power,patch.object(provider,'memory') as memory:
                result=self.predict(t,provider,changed)
            power.assert_not_called();memory.assert_not_called()
            self.assertIsNone(result['board_energy_joules']);self.assertIsNone(result['peak_memory_mib'])
            self.assertEqual(result['power_unknown_measurement_duration_ns'],t['total_measurement_overlap_ns'])
            self.assertEqual(result['memory_unknown_observation_duration_ns'],t['total_wall_ns'])
            self.assertFalse(result['horizon_transfer_qualified'])

    def test_resource_denominator_tamper_fails_before_lookup(self):
        provider,context=self.provider();t=timeline();t['slices'].pop()
        with patch.object(provider,'power') as power:
            with self.assertRaisesRegex(ValueError,'denominator'):self.predict(t,provider,context)
        power.assert_not_called()



    def test_two_macros_per_request_preserve780_rows_without_counting_extra_requests(self):
        plans,_,docs=matching_fixture()
        for doc in docs:
            for request in doc['requests']:
                first=request['macros'][0];second=copy.deepcopy(first);duration=first['response_wall_ns']
                second['macro_id']='ready_tail'
                for key in ('left_ns','right_ns'):second[key]+=duration
                for piece in second['causal_covariate_path']:
                    for key in ('left_ns','right_ns'):piece[key]+=duration
                    piece['state']['cutoff_processing_ns']+=duration
                request['macros'].append(second);request['worker_start_to_ready_ns']*=2
            doc['macros']*=2
        with tempfile.TemporaryDirectory() as temp,patch.object(adapter,'_load_report',return_value={}), \
            patch.object(adapter,'_rederive',return_value=(plans,docs,[])):
            bundle=adapter.build_training_rows('CPU_ONLY',workdir=temp)
        self.assertEqual(len(phase._validate_bundle(bundle)[0]),780)
        with patch.object(phase.original,'fitted_rows',return_value=(dict(numerical_qualification=False),None)):
            fitted=phase.fit_bundle(bundle)
        self.assertEqual((fitted['macro_rows'],fitted['measured_requests']),(780,390))
        self.assertEqual(sum(f['evaluation']['summary']['unknown_rows'] for f in fitted['whole_block_leaveout']),780)

    def test_resource_domain_window_checks_precede_native_table_algebra(self):
        plans,_,docs=matching_fixture();plan=plans[0];context=phase.checked(plan['context'])
        raw=dict(measurement_windows=[dict(case_id=c['case_id'],start_ns=0,end_ns=10**10,
            fixed_drain_deadline_ns=70*10**9) for c in plan['cases']])
        raw['measurement_windows'][0]['end_ns']+=1
        with patch.object(resource.legacy,'observed_table') as extract:
            with self.assertRaisesRegex(ValueError,'Actual measured resource windows'):
                resource.table_from_verified(raw,docs[0],plan,context)
        extract.assert_not_called()


class MatchingNumericalGateContracts(unittest.TestCase):
    def prepared(self,root,reviewed=False):
        producer=phase.checked(phase.receipt(PRODUCER))
        executor=put(root/'producer'/'manifest.json',{'CPU_SYNTHETIC_ONLY':True})
        reviews=[]
        if reviewed:
            entries=[phase.receipt(m.__file__) for m in (phase.executor,phase.intake,phase.adapter)]+[phase.receipt(phase.__file__)]
            reviews=[put(root/'review.json',dict(schema='CPU_TEST.independent',status='PASS_CPU_SYNTHETIC',sources=entries))['path']]
        final=root/'producer'/'run01'/'report.json'
        with patch.object(phase.executor,'audit',return_value=producer):
            ref=phase.prepare(executor['path'],final,reviews,root/'prepared')
        return ref,producer,final

    def final(self,manifest):
        return dict(schema=phase.executor.CONT_SCHEMA+'.run',manifest=manifest['executor_manifest'],
            status=phase.intake.FINAL_STATUS,no_retries=True,numerical_fit_run=False,validation_labels_read=False,
            fit_eligible=False,automatic_fit=False,completed_attempts=[dict(attempt_id=a) for a in manifest['expected_attempt_ids']])

    def test_missing_final_gates_every_native_reader_and_optimizer_even_fit_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,_=self.prepared(root)
            with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.intake,'intake') as intake, \
                patch.object(phase.adapter,'build_training_rows') as rows,patch.object(phase.original,'fitted_rows') as fit:
                manifest=phase.audit(ref['path']);report=phase.checked(phase.run(ref['path'],root/'waiting',fit_enabled=True))
            self.assertFalse(manifest['independent_source_review_complete'])
            self.assertEqual(report['status'],'WAITING_FOR_MATCHING_EXACT15_CONTINUATION_FINAL')
            self.assertFalse(report['numerical_fit_run']);intake.assert_not_called();rows.assert_not_called();fit.assert_not_called()
            self.assertFalse((root/'prepared'/'phase12-fit-claim.json').exists())

    def test_complete_final_requires_explicit_optin_and_independent_current_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,final=self.prepared(root);manifest=phase.checked(ref);put(final,self.final(manifest))
            with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.intake,'intake') as intake:
                ready=phase.checked(phase.run(ref['path'],root/'ready'))
                stopped=phase.checked(phase.run(ref['path'],root/'stopped',fit_enabled=True))
            self.assertEqual(ready['status'],'READY_EXPLICIT_FIT_REQUIRED')
            self.assertEqual(stopped['status'],'STOPPED_NO_RETRY')
            self.assertIn('independent review',stopped['error']);self.assertFalse(stopped['numerical_fit_run']);intake.assert_not_called()
            self.assertTrue((root/'prepared'/'phase12-fit-claim.json').exists())

    def test_duplicate_missing_foreign_or_label_final_holds_before_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,final=self.prepared(root);manifest=phase.checked(ref)
            for kind in ('duplicate','missing','foreign','label','fit'):
                doc=self.final(manifest)
                if kind=='duplicate':doc['completed_attempts'][1]=doc['completed_attempts'][0]
                elif kind=='missing':doc['completed_attempts'].pop()
                elif kind=='foreign':doc['manifest']={}
                elif kind=='label':doc['validation_labels_read']=True
                else:doc['numerical_fit_run']=True
                put(final,doc)
                with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.intake,'intake') as intake:
                    report=phase.checked(phase.run(ref['path'],root/kind,fit_enabled=True))
                self.assertEqual(report['status'],'HOLD_INCOMPLETE_OR_FOREIGN_CONTINUATION_FINAL');intake.assert_not_called()

    def test_source_drift_and_foreign_saved_fit_cannot_reach_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,_=self.prepared(root);manifest=phase.checked(ref)
            manifest['sources'].pop();bad=put(root/'bad.json',manifest)
            with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.intake,'intake') as intake:
                with self.assertRaises(ValueError):phase.run(bad['path'],root/'should-not-exist')
                foreign=put(root/'old-fit.json',dict(schema='gsdeploy.twoFIFO_source_bound_phase12_fit.v1',
                    status='COMPLETE_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS',numerical_fit_run=True))
                with self.assertRaises(ValueError):phase.audit_fit(foreign)
            intake.assert_not_called()

    def test_real_prepare_generated_resource_and_WAITING_never_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,_=self.prepared(root)
            score=phase.checked(phase.receipt(SCORE_DESIGN))
            with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.original,'fitted_rows') as fit, \
                patch.object(resource,'build_from_bundle') as resource_fit:
                gd=generated.prepare(ref['path'],score['scoring_plan']['path'],score['quality']['path'],root/'generated')
                rd=resource.prepare(ref['path'],root/'resources')
                self.assertEqual(generated.audit(gd['path'])['expected_cases'],171)
                self.assertEqual(resource.audit(rd['path'])['design']['resource_domain'],resource.DOMAIN)
                waiting=phase.checked(resource.run(rd['path'],root/'resource-waiting',fit_enabled=True))
            self.assertEqual(waiting['status'],'WAITING_FOR_MATCHING_EXACT15_CONTINUATION_FINAL')
            fit.assert_not_called();resource_fit.assert_not_called()

    def test_public_run_executes_mock_original_math_via_verified_intake_and_new_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,final=self.prepared(root,reviewed=True);manifest=phase.checked(ref)
            put(final,self.final(manifest));bundle=bundle_fixture(root/'rows');calls=[]
            def intake(*args,output):
                calls.append('NEW_INTAKE');return put(Path(output)/'report.json',dict(sources=[]))
            def build(*args,**kwargs):calls.append('NEW_ROWS');return bundle
            def mockfit(rows):calls.append('ORIGINAL_FITTED_ROWS_MOCK');return dict(numerical_qualification=False),None
            with patch.object(phase.executor,'audit',return_value=producer),patch.object(phase.intake,'intake',side_effect=intake), \
                patch.object(phase.adapter,'build_training_rows',side_effect=build),patch.object(phase.original,'fitted_rows',side_effect=mockfit):
                report_ref=phase.run(ref['path'],root/'result',fit_enabled=True)
                report=phase.checked(report_ref);audited=phase.audit_fit(report_ref)
            self.assertEqual(calls,['NEW_INTAKE','NEW_ROWS']+['ORIGINAL_FITTED_ROWS_MOCK']*4)
            self.assertEqual(report['status'],'COMPLETE_MATCHING_SOURCE_BOUND_PHASE12_TRAINING_DIAGNOSTICS')
            self.assertFalse(report['fit']['response_law_compilation_eligible']);self.assertFalse(audited['numeric_recomputed'])

    def test_resource_fit_claim_prevents_retry_after_native_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref,producer,final=self.prepared(root,reviewed=True)
            with patch.object(phase.executor,'audit',return_value=producer):
                design=resource.prepare(ref['path'],root/'resource')
                put(final,self.final(phase.checked(ref)))
                with patch.object(phase,'with_verified_bundle',side_effect=RuntimeError('CPU_TEST_NATIVE_FAILURE')) as reader:
                    with self.assertRaisesRegex(RuntimeError,'CPU_TEST_NATIVE_FAILURE'):
                        resource.run(design['path'],root/'failed',fit_enabled=True)
                    with self.assertRaises(FileExistsError):resource.run(design['path'],root/'retry',fit_enabled=True)
                self.assertEqual(reader.call_count,1)


class MatchingJointLawAlgebraContracts(unittest.TestCase):
    def test_full_and_folded_matching_law_execute_original_request_algebra_without_fit(self):
        plans,_,docs=matching_fixture();raws=[];groups=set()
        for plan,doc in zip(plans,docs):
            def row(owner,rid,task,pool,left,case=None):
                event=dict(perf_ns=left,received_time_ns=left,collector_dequeued_time_ns=left,collector_processing_time_ns=left)
                ready=dict(perf_ns=left+110,received_time_ns=left+110,collector_dequeued_time_ns=left+110,collector_processing_time_ns=left+110)
                return dict(model=owner,request_id=rid,task=task,pool_id=pool,profile_job_id=case,
                    execution_events=[event],previous_handler_next_request_ready=ready,
                    worker_response={'end_perf_ns':left+100},response_reader_event_enqueue_time_ns=left+100,
                    response_collector_dequeued_time_ns=left+100,response_collector_processing_time_ns=left+100,
                    actual_submit_time_ns=left-1,enqueued_time_ns=left-2,scheduled_time_ns=left-2,
                    submit_write_completed_time_ns=left,observed_worker_start_time_ns=left)
            prime=[row(p['owner'],p['request_id'],p['task'],p['pool_id'],1000+i*1000) for i,p in enumerate(plan['priming'])]
            arrivals=[(c['case_id'],a) for c in plan['cases'] for a in c['arrivals']];records=[]
            for i,(request,(case,a)) in enumerate(zip(doc['requests'],arrivals)):
                left=100000+i*1000;macro=request['macros'][0]
                macro.update(left_ns=left,right_ns=left+100,response_wall_ns=100)
                macro['causal_covariate_path']=[dict(left_ns=left,right_ns=left+100,duration_ns=100,
                    each_state_uses_only_its_processed_prefix=True,state=dict(cutoff_processing_ns=left,
                        peer_visible_macro_owner_count=0,peer_same_visible_macro_phase_owner_count=0,peer_macros=[]))]
                request['worker_start_to_ready_ns']=100
                groups.add((macro['owner'],macro['pool_id'],macro['task'],macro['macro_id'],macro['semantic_phase'],macro['kind'],None,None,None))
                records.append(row(a['owner'],a['request_id'],a['task'],a['pool_id'],left,case))
            raws.append(dict(schema=phase.executor.RAW_SCHEMA,backend='native',status='completed',split='training',
                CPU_SYNTHETIC_ONLY=True,**{k:plan[k] for k in ('capture_id','attempt_id','execution_id')},
                request_records=records,priming_request_records=prime))
        denominator=joint.complete_documents(docs,plans,raws);blocks=sorted({p['capture_id'] for p in plans})
        def fit(bs):return dict(numerical_qualification=True,whole_blocks=bs,coefficients=[0.]*12,
            features=list(phase.FEATURES),design=phase.DESIGN,
            baselines=[dict(group=list(group),log_reference_ms=0.) for group in groups])
        fitted=dict(response_law_compilation_eligible=True,features=list(phase.FEATURES),
            fitted_root_numerical_qualification=True,denominator=denominator,full_fit=fit(blocks),
            whole_block_leaveout=[dict(heldout_block_id=b,training_fit=fit([x for x in blocks if x!=b]),
                evaluation={'summary':{'unknown_rows':0}},
                training_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=b],
                heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==b]) for b in blocks])
        with patch.object(phase.original,'fitted_rows',side_effect=AssertionError('NO_FIT_ALLOWED')):
            full=joint.build_joint_law(docs,plans,raws,fitted)
            folded=joint.build_joint_law(docs,plans,raws,fitted,heldout_block_id=blocks[0])
            compiled=joint.joint.compile_draw_v4(folded['vectors'][0],0)
        self.assertEqual((full['measured_request_vectors'],folded['measured_request_vectors']),(390,260))
        self.assertEqual((full['donor_startup_attempts'],folded['donor_startup_attempts']),(15,10))
        self.assertEqual(full['source_macro_schema'],adapter.MACRO_SCHEMA)
        self.assertFalse(full['source_documents_relabelled'])
        self.assertEqual(folded['blocks'],blocks[1:]);self.assertEqual(len(compiled['plan'].macros),1)
        self.assertFalse(compiled['latent_line_or_continuation_point_selected'])
        self.assertTrue(all(v['training_block_id']!=blocks[0] for v in folded['vectors']))


if __name__=='__main__':unittest.main()
