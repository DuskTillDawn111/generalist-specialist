"""CPU counterexamples for full native rederivation and numerical HOLD gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import kernel_dag_response_numerical_pipeline_v1 as m


def exports(root,*,timing=9,proof=1):
    root.mkdir()
    eq=m.write(root/'template.json',dict(native_full_identity=proof))
    arts={name:m.write(root/name,dict(timing=timing,proof=eq))
          for name in ('catalog.json','captures.json','paths.json')}
    return dict(matched=[dict(equivalence=eq)],artifacts=arts,training_only=True,
                validation_labels_read=False,status='COMPLETE')


def prepared_fixture():
    rng=np.random.default_rng(42);rows=[];registry=[]
    for cap in range(3):
        for i in range(20):
            vector=rng.uniform(0,1,6);owner=i%4
            duration=float(np.exp(owner/5+vector@np.arange(6)/10+rng.normal(0,.2)))
            rows.append(dict(group=('owner'+str(owner),'template','node','context','profiler'),
                capture='cap'+str(cap),request_id=str(i),macro_id='node',
                durations=np.asarray([duration]),matrix=vector[None,:],
                ordered_durations=np.asarray([duration]),ordered_matrix=vector[None,:]))
            registry.append(dict(capture_id='cap'+str(cap),request_id=str(i)))
    return dict(schema=m.compiler.PATH_SCHEMA,features=list(m.compiler.features.FEATURES),
        rows=rows,offered_requests=60,offered_kernel_count=60,accepted_kernel_count=60,
        rejected_rows=[],full_offered_request_registry=registry)


class NumericalPipelineTests(unittest.TestCase):
    def test_temporary_proof_paths_require_full_content_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);published=exports(base/'old');fresh=exports(base/'fresh')
            m.compare_rebuild(published,fresh)

    def test_rehashed_timing_catalog_or_path_changes_are_rejected(self):
        for artifact in ('catalog.json','captures.json','paths.json'):
            with self.subTest(artifact=artifact),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);published=exports(base/'old');fresh=exports(base/'fresh')
                path=Path(published['artifacts'][artifact]['path'])
                value=json.loads(path.read_text());value['timing']=10;path.write_text(json.dumps(value))
                published['artifacts'][artifact]=m.receipt(path)
                with self.assertRaisesRegex(ValueError,'Fresh complete repeated export differs'):
                    m.compare_rebuild(published,fresh)

    def test_rehashed_equivalence_proof_cannot_authenticate_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);published=exports(base/'old',proof=2);fresh=exports(base/'fresh')
            with self.assertRaisesRegex(ValueError,'equivalence proof differs'):
                m.compare_rebuild(published,fresh)

    def test_full_report_role_and_denominator_changes_are_rejected(self):
        for field,value in [('validation_labels_read',True),('training_only',False),('matched',[])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp);published=exports(base/'old');fresh=exports(base/'fresh')
                published[field]=value
                with self.assertRaises(ValueError):m.compare_rebuild(published,fresh)

    def test_beta_zero_derivative_rank_is_not_a_structural_gate(self):
        prepared=prepared_fixture()
        # Every derivative at zero is zero, but full paths span all six columns.
        for i,row in enumerate(prepared['rows']):
            row['ordered_matrix']=row['matrix']=np.asarray([np.eye(6)[i%6],-np.eye(6)[i%6]])
            row['durations']=row['ordered_durations']=np.asarray([1.,1.])
        self.assertEqual(m.structural_summary(prepared)['path_contrast_rank'],6)
        derivatives=np.asarray([-(np.asarray([.5,.5])@row['matrix']) for row in prepared['rows']])
        self.assertEqual(m.likelihood.nuisance_certificate(prepared['rows'],derivatives)['projected_derivative_rank'],0)
        with patch.object(m.likelihood,'fit',return_value=({'numerical_qualification':False},None,None,None)) as fitting:
            m.compiler.fit_paths(prepared)
        fitting.assert_called_once()

    def test_rank_deficiency_retains_all_sixty_and_never_calls_optimizer(self):
        prepared=prepared_fixture()
        for row in prepared['rows']:row['ordered_matrix'][:]=row['matrix'][:]=0
        with patch.object(m.likelihood,'fit',side_effect=AssertionError('must HOLD')):
            info=m.compiler.fit_paths(prepared)
        self.assertEqual(info['reason'],'STRUCTURAL_KERNEL_COVARIATE_RANK_FAILURE')
        self.assertFalse(info['columns_dropped'])
        self.assertEqual(m.denominator(prepared)['offered_requests'],60)
        self.assertEqual(sum(r['kernels'] for r in m.denominator(prepared)['accepted_kernels_per_request']),60)

    def test_complete_cpu_synthetic_likelihood_preserves_original_solver_and_actual_root_rank(self):
        prepared=prepared_fixture();self.assertEqual(m.structural_summary(prepared)['path_contrast_rank'],6)
        info=m.compiler.fit_paths(prepared)
        self.assertEqual(info['macro_rows'],60)
        self.assertEqual([a['start'] for a in info['trust_exact_attempts']],m.protocol()['starts'])
        self.assertEqual(len(info['gradient_root_attempts']),3)
        self.assertEqual(info['nuisance_adjusted_features'],6)
        self.assertTrue(info['joint_response_jacobian']['includes_baseline_nuisance_means'])
        self.assertFalse(info['prospective_prediction_qualified'])

    def test_incomplete_kernel_cohort_holds_before_original_solver(self):
        prepared=prepared_fixture();prepared['rows'].pop();prepared['accepted_kernel_count']=59
        prepared['rejected_rows']=[dict(reason='UNSUPPORTED')]
        with patch.object(m.likelihood,'fit',side_effect=AssertionError('must HOLD')):
            self.assertEqual(m.compiler.fit_paths(prepared)['reason'],'INCOMPLETE_SUPPORTED_NODE_COHORT')

    def test_cpu_environment_required_before_any_source_or_label_access(self):
        with patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(m,'audit',side_effect=AssertionError('labels')):
            with self.assertRaisesRegex(ValueError,'CPU runner'):m.fit_sealed('no-plan','no-seal','no-output')

    def test_fixed_contract_has_no_feature_or_threshold_relaxation(self):
        p=m.protocol();self.assertEqual(len(p['features']),6)
        self.assertFalse(p['rows_dropped']);self.assertFalse(p['columns_dropped'])
        self.assertFalse(p['beta_zero_derivative_rank_is_hold_gate'])
        self.assertEqual(p['offered_requests'],60);self.assertEqual(p['response_sigma_floor'],None)

    def run_fixture(self,root,*,tamper=False,rank_zero=False):
        root=Path(root);plan_path=root/'plan.json';plan_path.write_text('{}')
        final_ref=m.write(root/'native.json',{});report_ref=m.write(root/'repeated.json',{})
        report=dict(design=final_ref,artifacts={},rejected=[])
        plan=dict(repeated_report_path=report_ref['path'])
        prepared=prepared_fixture()
        if rank_zero:
            for row in prepared['rows']:row['ordered_matrix'][:]=row['matrix'][:]=0
        sealed=m.data_record(m.receipt(plan_path),final_ref,report,prepared)
        sealed.update(schema=m.SCHEMA+'.seal',status='SEALED_COMPLETE_CPU_NUMERICAL_DATA_NO_FIT',repeated_alignment=report_ref)
        if tamper:sealed['denominator']['ordered_complete_kernel_registry_sha256']='f'*64
        seal_ref=m.write(root/'seal.json',sealed)
        with patch.dict(m.os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(m,'audit',return_value=plan),\
             patch.object(m,'native_publication',return_value=(final_ref,report,[final_ref,report_ref])),\
             patch.object(m,'authenticated_alignment') as native_rebuild,\
             patch.object(m,'compile_complete',return_value=prepared):
            result=m.fit_sealed(plan_path,seal_ref['path'],root/'fit')
        native_rebuild.assert_called_once()
        return result

    def test_rehashed_seal_denominator_cannot_bypass_fresh_compilation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.compiler,'fit_paths',side_effect=AssertionError('must reject')):
            result=self.run_fixture(tmp,tamper=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_EVIDENCE_PRESERVED')
            self.assertFalse(result['fit_started'])
            self.assertIn('differs from seal',result['error'])

    def test_actual_pipeline_rank_hold_preserves_claim_and_original_denominator(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.likelihood,'fit',side_effect=AssertionError('must HOLD')):
            result=self.run_fixture(tmp,rank_zero=True)
            self.assertEqual(result['status'],'UNKNOWN_NUMERICAL_QUALIFICATION_FAILED')
            self.assertFalse(result['fit_started']);self.assertEqual(result['denominator']['offered_requests'],60)
            self.assertTrue((Path(tmp)/'fit/claim.json').is_file())
            self.assertTrue((Path(tmp)/'fit/report.json').is_file())

    def test_optimizer_exception_preserves_started_evidence_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(m.compiler,'fit_paths',side_effect=ValueError('nonfinite likelihood')) as fitter:
            result=self.run_fixture(tmp)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_EVIDENCE_PRESERVED')
            self.assertTrue(result['fit_started']);fitter.assert_called_once()
            self.assertTrue((Path(tmp)/'fit/fit-start.json').is_file())

    def test_source_change_and_restore_after_qualified_solver_revokes_all_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fitting(prepared):
                seal=Path(tmp)/'seal.json';original=seal.read_bytes()
                seal.write_bytes(original+b' ');seal.write_bytes(original)
                return dict(numerical_qualification=True,coefficients=[1.]*6)
            with patch.object(m.compiler,'fit_paths',side_effect=fitting):
                result=self.run_fixture(tmp)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY_EVIDENCE_PRESERVED')
            self.assertFalse(result['numerical_qualification'])
            self.assertFalse(result['prospective_prediction_qualified'])
            self.assertFalse(result['native_sources_authenticated'])
            self.assertFalse(result['response_fit_authenticated'])
            self.assertNotIn('response_fit',result)
            diagnostic=result['unauthenticated_response_fit_diagnostic']
            self.assertTrue(diagnostic['result']['numerical_qualification'])
            self.assertFalse(diagnostic['authentication_passed'])
            self.assertFalse(diagnostic['usable_as_qualification'])


if __name__=='__main__':unittest.main()
