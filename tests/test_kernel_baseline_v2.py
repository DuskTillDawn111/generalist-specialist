"""Synthetic CPU contract checks, never target-model profiling evidence."""
import copy
import hashlib
import unittest

from gsdeploy.forecasting.kernel_baseline import (
    FEATURES, MANIFEST_SCHEMA, KernelBaselineEstimator, adapt_affine_feature_bindings, adapt_nsys_trace,
    adapt_raw_kernel_profiles, trace_kernel_order,
)


def source(label):
    return dict(path=label, sha256=hashlib.sha256(label.encode()).hexdigest())


def observation(i, batch=None, *, split='training', run=None, profiled=True):
    batch = i+1 if batch is None else batch
    return dict(id=f'{split}_{i}', run_id=run or f'{split}_run_{i}', split=split,
                evidence_kind='cpu_fixture', hardware_id='synthetic_device_uuid',
                operator='linear', kernel_id='synthetic_kernel', precision='fp16',
                execution_key='synthetic_backend_settings', execution_context_sha256='a'*64,
                feature_schema_id='synthetic_explicit_feature_definitions_v1',
                timing_basis='profiled_isolated' if profiled else 'unprofiled_isolated',
                profiler_id='synthetic_nsys_version_and_command' if profiled else None,
                profiler_perturbed=profiled,
                perturbation_flags=['cuda_nvtx_profiler'] if profiled else [],
                isolation_evidence=source('synthetic_isolation_attestation'),
                measurement_source=source(f'{split}_measurement_{i}'),
                features=dict(batch_size=batch, weight_bytes=100,
                              input_tensor_bytes=10*batch, output_tensor_bytes=20*batch,
                              flops=1000*batch),
                feature_evidence={name: [source('shared_training_and_validation_feature_graph')]
                                  for name in FEATURES},
                duration_ns=100+20*batch, intermediate_peak_bytes=None)


def estimator(rows=None, **kwargs):
    return KernelBaselineEstimator(rows or [observation(i) for i in range(4)],
                                   allow_cpu_fixture=True, **kwargs)


def manifest(row=None):
    row = observation(0) if row is None else row
    fields = ('run_id', 'split', 'evidence_kind', 'hardware_id',
              'execution_context_sha256', 'timing_basis', 'profiler_id',
              'profiler_perturbed', 'perturbation_flags', 'isolation_evidence',
              'measurement_source')
    config_fields = ('operator', 'kernel_id', 'precision', 'execution_key',
                     'feature_schema_id', 'features', 'feature_evidence')
    return dict(schema=MANIFEST_SCHEMA, **{name: copy.deepcopy(row[name]) for name in fields},
                configurations={'config': {name: copy.deepcopy(row[name]) for name in config_fields}})


def kernel(start=10, name='synthetic_kernel'):
    return dict(nsys_start_ns=start, nsys_end_ns=start+120, pid=100, stream=7,
                correlation_id=start, name=name)


def trace():
    k = kernel()
    return dict(schema='gsdeploy.nsys_observed_phases.v1', sources=[source('trace.sqlite')],
                nvtx_ranges=[dict(kernels=[k])], phases=[dict(kernels=[copy.deepcopy(k)])],
                unattributed_kernels=[kernel(200)])


class BaselineRegressionTests(unittest.TestCase):
    def test_profiled_training_is_supported_and_stays_profiled(self):
        fit = estimator()
        result = fit.predict(observation(9, batch=2))
        self.assertTrue(result['known'])
        self.assertAlmostEqual(result['targets']['duration_ns']['value'], 140)
        self.assertEqual(result['timing_basis'], 'profiled_isolated')
        self.assertTrue(result['profiler_perturbed'])
        self.assertFalse(result['natural_timing_validated'])
        self.assertFalse(result['deployment_qualified'])
        self.assertFalse(result['targets']['intermediate_peak_bytes']['known'])
        self.assertTrue(result['cpu_fixture'])

    def test_fixture_training_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'explicit opt-in'):
            KernelBaselineEstimator([observation(i) for i in range(4)])

    def test_natural_and_profiler_domains_never_mix(self):
        fit = estimator()
        result = fit.predict(observation(9, batch=2, profiled=False))
        self.assertEqual(result['reasons'], ['EXACT_TRAINING_DOMAIN_NOT_FOUND'])
        query = observation(9, batch=2)
        query['profiler_id'] = 'another_profiler'
        self.assertFalse(fit.predict(query)['known'])
        query['profiler_id'] = observation(0)['profiler_id']
        query['perturbation_flags'].append('another_instrument')
        self.assertFalse(fit.predict(query)['known'])

    def test_unprofiled_isolated_does_not_qualify_natural_timing(self):
        fit = estimator([observation(i, profiled=False) for i in range(4)])
        result = fit.predict(observation(9, batch=2, profiled=False))
        self.assertTrue(result['known'])
        self.assertFalse(result['natural_timing_validated'])

    def test_profiled_metadata_cannot_be_relabelled_unprofiled(self):
        rows = [observation(i) for i in range(4)]
        rows[0]['timing_basis'] = 'unprofiled_isolated'
        with self.assertRaisesRegex(ValueError, 'Unprofiled'):
            estimator(rows)
        rows[0]['timing_basis'] = 'profiled_isolated'
        rows[0]['profiler_id'] = None
        with self.assertRaisesRegex(ValueError, 'profiler identity'):
            estimator(rows)

    def test_collinear_coefficients_still_identify_in_hull_prediction(self):
        result = estimator().predict(observation(9, batch=2))
        diag = result['targets']['duration_ns']['diagnostics']
        self.assertEqual(diag['affine_design_rank'], 2)
        self.assertFalse(diag['coefficients_individually_identifiable'])
        self.assertFalse(diag['training_error_is_generalization_bound'])

    def test_in_feature_range_but_outside_affine_subspace_is_unknown(self):
        query = observation(9, batch=2)
        query['features']['input_tensor_bytes'] = 30  # independently in observed min/max
        result = estimator().predict(query)
        self.assertEqual(result['targets']['duration_ns']['reasons'],
                         ['OUTSIDE_TRAINING_AFFINE_SUBSPACE'])

    def test_affine_hull_extrapolation_is_unknown(self):
        result = estimator().predict(observation(9, batch=5))
        self.assertEqual(result['targets']['duration_ns']['reasons'],
                         ['OUTSIDE_TRAINING_CONVEX_HULL'])

    def test_inside_each_range_but_outside_joint_convex_hull_is_unknown(self):
        rows = [observation(i, batch=b) for i, b in enumerate((1, 4, 1))]
        for row, value in zip(rows, (10, 10, 40)):
            row['features']['input_tensor_bytes'] = value
            row['features']['output_tensor_bytes'] = 0
            row['features']['flops'] = 0
            row['duration_ns'] = 100+row['features']['batch_size']+value
        query = copy.deepcopy(rows[0])
        query['features'].update(batch_size=4, input_tensor_bytes=40)
        result = estimator(rows).predict(query)
        self.assertEqual(result['targets']['duration_ns']['reasons'],
                         ['OUTSIDE_TRAINING_CONVEX_HULL'])

    def test_peak_has_independent_coverage_and_evidence(self):
        rows = [observation(i) for i in range(4)]
        for row in rows:
            row['intermediate_peak_bytes'] = 1000+10*row['features']['batch_size']
            row['peak_measurement_source'] = source('peak_'+row['id'])
        fit = estimator(rows)
        self.assertAlmostEqual(fit.predict(observation(9, batch=2))['targets']
                               ['intermediate_peak_bytes']['value'], 1020)
        rows[0].pop('peak_measurement_source')
        with self.assertRaisesRegex(ValueError, 'peak and its source'):
            estimator(rows)

    def test_minimum_runs_and_bad_fit_are_explicit_unknowns(self):
        rows = [observation(i, run='one_run') for i in range(4)]
        self.assertFalse(estimator(rows).predict(observation(9, batch=2))['known'])
        rows = [observation(i, batch=2) for i in range(4)]
        rows[0]['duration_ns'] = 10000
        result = estimator(rows).predict(observation(9, batch=2))
        self.assertEqual(result['targets']['duration_ns']['reasons'],
                         ['TRAINING_RESIDUAL_EXCEEDS_LIMIT'])

    def test_validation_has_disjoint_measurements_and_never_changes_fit(self):
        fit = estimator()
        before = fit.training_sha256
        result = fit.evaluate([observation(100, batch=2, split='validation')])
        self.assertEqual(result['metrics']['duration_ns']['evaluated_count'], 1)
        self.assertAlmostEqual(result['metrics']['duration_ns']['max_relative_error'], 0)
        self.assertEqual(result['metrics']['intermediate_peak_bytes']['evaluated_count'], 0)
        self.assertEqual(fit.training_sha256, before)
        # Shared feature graph receipts are legitimate; raw measurements are not.
        row = observation(100, batch=2, split='validation')
        row['measurement_source'] = observation(0)['measurement_source']
        with self.assertRaisesRegex(ValueError, 'sources overlap'):
            fit.evaluate([row])
        row = observation(100, batch=2, split='validation')
        row['run_id'] = observation(0)['run_id']
        with self.assertRaisesRegex(ValueError, 'identities/runs overlap'):
            fit.evaluate([row])

    def test_missing_features_are_not_synthetic_zeros(self):
        rows = [observation(i) for i in range(4)]
        rows[0]['features'].pop('weight_bytes')
        with self.assertRaisesRegex(ValueError, 'weight_bytes'):
            estimator(rows)

    def test_feature_schema_is_part_of_exact_group(self):
        query = observation(9, batch=2)
        query['feature_schema_id'] = 'logical_instead_of_unique_storage'
        self.assertFalse(estimator().predict(query)['known'])


class BaselineAdapterTests(unittest.TestCase):
    def test_combined_observer_supplies_real_covariates_without_kernel_targets(self):
        document = dict(affine_operator_observations=dict(
            schema='gsdeploy.affine_operator_observations.v2', records=[dict(
                id='affine_0000000', operator_kind='linear_declared',
                direct_tensors=[dict(name='qweight', logical_bytes=64),
                                dict(name='alias_or_scale', logical_bytes=1024)],
                input=dict(logical_bytes=256), output=dict(logical_bytes=512),
                dense_gemm_flops=8192, start_ns=1, end_ns=1000000)]))
        binding = dict(kernel_id='synthetic_kernel', precision='fp16_input_int4_weight',
                       execution_key='real_backend_settings_receipt', batch_size=2,
                       batch_evidence=source('input_manifest'), weight_tensor_name='qweight',
                       operator_to_kernel_evidence=source('explicit_cupti_association'))
        result = adapt_affine_feature_bindings(document, {'affine_0000000': binding}, source('observer'))
        config = result['configurations']['affine_0000000']
        self.assertEqual(config['features'], dict(batch_size=2, weight_bytes=64,
                         input_tensor_bytes=256, output_tensor_bytes=512, flops=8192))
        self.assertNotIn('duration_ns', config)
        self.assertFalse(result['training_targets_created'])
        self.assertIn('not unique storage', config['feature_interpretation'])
        # Missing output on an exception cannot be completed using an estimate.
        document['affine_operator_observations']['records'][0]['output'] = None
        result = adapt_affine_feature_bindings(document, {'affine_0000000': binding}, source('observer'))
        self.assertEqual(result['configurations'], {})
        self.assertEqual(result['rejected_rows'][0]['reasons'], ['INCOMPLETE_INPUT_OUTPUT_METADATA'])

    def test_observer_batch_and_weight_attribution_are_explicit(self):
        document = dict(schema='gsdeploy.affine_operator_observations.v2', records=[dict(id='op')])
        result = adapt_affine_feature_bindings(document, {}, source('observer'))
        self.assertEqual(result['configurations'], {})
        self.assertEqual(result['rejected_rows'][0]['reasons'],
                         ['EXPLICIT_OPERATOR_CONFIGURATION_BINDING_REQUIRED'])

    def test_raw_adapter_uses_measured_duration_and_separate_peak(self):
        result = adapt_raw_kernel_profiles([dict(id='launch', config_id='config', duration_ns=173)], manifest())
        self.assertEqual(result['rows'][0]['duration_ns'], 173)
        self.assertIsNone(result['rows'][0]['intermediate_peak_bytes'])

    def test_raw_adapter_missing_binding_or_feature_stays_unknown(self):
        raw = [dict(id='launch', config_id='missing', duration_ns=173)]
        result = adapt_raw_kernel_profiles(raw, manifest())
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['rejected_rows'][0]['reasons'], ['MISSING_CONFIGURATION_BINDING'])
        doc = manifest()
        doc['configurations']['config']['features'].pop('flops')
        result = adapt_raw_kernel_profiles([dict(id='launch', config_id='config', duration_ns=173)], doc)
        self.assertIn('flops', result['rejected_rows'][0]['detail'])

    def test_trace_adapter_counts_canonical_launches_once_and_keeps_unattributed(self):
        self.assertEqual(len(trace_kernel_order(trace())), 2)
        doc = manifest()
        doc['measurement_source'] = source('trace.sqlite')
        doc['kernel_bindings'] = {str(i): dict(config_id='config', launch_identity={
            k: launch[k] for k in ('nsys_start_ns', 'nsys_end_ns', 'pid', 'stream', 'correlation_id')})
            for i, launch in enumerate(trace_kernel_order(trace()))}
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(result['rows'][0]['duration_ns'], 120)
        self.assertTrue(result['rows'][0]['id'].startswith('training_run_0:'))
        self.assertEqual(result['rows'][0]['timing_basis'], 'profiled_isolated')
        self.assertIsNone(result['rows'][0]['intermediate_peak_bytes'])
        doc['kernel_bindings']['0']['launch_identity']['stream'] = 999
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rejected_rows'][0]['reasons'],
                         ['EXACT_LAUNCH_CONFIGURATION_BINDING_REQUIRED'])

    def test_trace_source_hash_is_bound_and_isolation_is_not_inferred(self):
        doc = manifest()
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(result['rejected_rows'][0]['reasons'], ['EXACT_TRACE_SOURCE_RECEIPT_REQUIRED'])
        doc['measurement_source'] = source('trace.sqlite')
        doc['isolation_evidence'] = None
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(len(result['rejected_rows']), 2)
        self.assertEqual(result['rejected_rows'][0]['reasons'], ['ISOLATED_EXECUTION_EVIDENCE_REQUIRED'])

    def test_nsight_cannot_claim_unprofiled_or_bind_another_kernel_name(self):
        doc = manifest(observation(0, profiled=False))
        doc['measurement_source'] = source('trace.sqlite')
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(result['rejected_rows'][0]['reasons'], ['NSYS_REQUIRES_PROFILED_ISOLATED_BASIS'])
        doc = manifest()
        doc['measurement_source'] = source('trace.sqlite')
        doc['kernel_bindings'] = {'0': dict(config_id='config', launch_identity={
            k: kernel()[k] for k in ('nsys_start_ns', 'nsys_end_ns', 'pid', 'stream', 'correlation_id')})}
        doc['configurations']['config']['kernel_id'] = 'other_kernel'
        result = adapt_nsys_trace(trace(), doc)
        self.assertEqual(result['rejected_rows'][-2]['reasons'], ['BOUND_KERNEL_IDENTITY_MISMATCH'])


if __name__ == '__main__':
    unittest.main()
