"""CPU-only, evidence-bound isolated kernel baseline regression.

Profiler observations are legitimate training inputs, but stay profiler-domain
predictions. This module does not recover natural timing, infer memory traffic,
or qualify any runtime predictor. Input adapters never derive kernel features
from names, launch geometry, host module durations, or guessed model weights.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from ..linear_solver import linprog


SCHEMA = 'gsdeploy.kernel_baseline_observations.v2'
MANIFEST_SCHEMA = 'gsdeploy.kernel_baseline_capture.v2'
FEATURES = ('batch_size', 'weight_bytes', 'input_tensor_bytes',
            'output_tensor_bytes', 'flops')
GROUP_FIELDS = ('hardware_id', 'operator', 'kernel_id', 'precision',
                'execution_key', 'execution_context_sha256', 'feature_schema_id',
                'timing_basis', 'profiler_id')
TARGETS = ('duration_ns', 'intermediate_peak_bytes')


def _unknown(reason, **extra):
    return dict(status='UNKNOWN', known=False, reasons=[reason],
                natural_timing_validated=False, deployment_qualified=False, **extra)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


def _source(value):
    return (isinstance(value, dict) and _nonempty(value.get('path')) and
            _sha(value.get('sha256')))


def _finite(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0))


def _validate_row(row, *, split=None, allow_cpu_fixture=False):
    """Reject malformed or incomplete normalized observations explicitly."""
    required = ('id', 'run_id', 'split', 'evidence_kind', 'measurement_source',
                'isolation_evidence', 'feature_evidence', 'features',
                'profiler_perturbed', 'perturbation_flags', *GROUP_FIELDS)
    missing = [key for key in required if key not in row or
               (row[key] is None and key != 'profiler_id')]
    if missing:
        raise ValueError('Missing observation fields: ' + ', '.join(missing))
    for key in ('id', 'run_id', *GROUP_FIELDS):
        if key != 'profiler_id' and not _nonempty(row[key]):
            raise ValueError('Nonempty string required: ' + key)
    if not _sha(row['execution_context_sha256']):
        raise ValueError('Execution context must be a SHA-256')
    if row['split'] not in ('training', 'validation') or (split and row['split'] != split):
        raise ValueError('Observation split mismatch')
    if row['evidence_kind'] == 'cpu_fixture':
        if not allow_cpu_fixture:
            raise ValueError('CPU fixtures require explicit opt-in')
    elif row['evidence_kind'] != 'measured':
        raise ValueError('Unknown observation evidence kind')
    if not _source(row['measurement_source']):
        raise ValueError('Measurement source receipt required')
    if not _source(row['isolation_evidence']):
        raise ValueError('Isolated execution evidence source receipt required')
    if type(row['profiler_perturbed']) is not bool or not isinstance(row['perturbation_flags'], list):
        raise ValueError('Explicit profiler perturbation declaration required')
    if any(not _nonempty(flag) for flag in row['perturbation_flags']):
        raise ValueError('Perturbation flags must be nonempty strings')
    if row['timing_basis'] == 'profiled_isolated':
        if not row['profiler_perturbed'] or not _nonempty(row['profiler_id']) or not row['perturbation_flags']:
            raise ValueError('Profiled data requires profiler identity and perturbation flags')
    elif row['timing_basis'] == 'unprofiled_isolated':
        if row['profiler_perturbed'] or row['profiler_id'] is not None or row['perturbation_flags']:
            raise ValueError('Unprofiled data cannot contain profiler perturbation')
    else:
        raise ValueError('Unsupported isolated timing basis')
    if not isinstance(row['features'], dict) or not isinstance(row['feature_evidence'], dict):
        raise ValueError('Feature values and evidence must be mappings')
    for name in FEATURES:
        value = row['features'].get(name)
        if type(value) is not int or value < (1 if name == 'batch_size' else 0):
            raise ValueError('Missing/noninteger/nonnegative feature: ' + name)
        evidence = row['feature_evidence'].get(name)
        if not isinstance(evidence, list) or not evidence or not all(_source(s) for s in evidence):
            raise ValueError('Feature source receipts required: ' + name)
    # A missing peak target does not invalidate a genuine duration observation.
    if not _finite(row.get('duration_ns'), positive=True):
        raise ValueError('Positive measured kernel duration required')
    peak = row.get('intermediate_peak_bytes')
    if peak is not None:
        if type(peak) is not int or peak < 0 or not _source(row.get('peak_measurement_source')):
            raise ValueError('Measured nonnegative integer peak and its source required')
    return row


def _group(row):
    # Different instrumentation flags and feature definitions cannot be pooled.
    return tuple(row[name] for name in GROUP_FIELDS) + (tuple(sorted(row['perturbation_flags'])),)


def adapt_affine_feature_bindings(document, operator_bindings, observer_source):
    """Read real affine observer metadata as explicitly defined covariates.

    Accepts either an affine document or CombinedObservation's nested document.
    Each binding keyed by observer record ID supplies kernel_id, precision,
    execution_key, batch_size, batch_evidence receipt, weight_tensor_name, and
    operator_to_kernel_evidence receipt. Batch semantics and kernel attribution
    are never inferred from a tensor shape or a module host duration.

    ``flops`` here means enclosing operator's declared dense GEMM arithmetic;
    it is a workload covariate, not per-kernel executed FLOPs. ``weight_bytes``
    means the one selected tensor's logical bytes, not unique storage bytes.
    Multiple kernels may share these operator covariates only with explicit
    attribution sidecars; Nsight exact-launch binding remains separately required.
    """
    document = document.get('affine_operator_observations', document)
    if document.get('schema') != 'gsdeploy.affine_operator_observations.v2':
        raise ValueError('Unsupported affine observer schema')
    if not _source(observer_source) or not isinstance(operator_bindings, dict):
        raise ValueError('Observer source receipt and explicit operator bindings required')
    configurations, rejected, seen = {}, [], set()
    for record in document.get('records', []):
        identity = record.get('id')
        if not _nonempty(identity) or identity in seen:
            raise ValueError('Affine observer identities must be unique nonempty strings')
        seen.add(identity)
        binding = operator_bindings.get(identity)
        reason = None
        if not isinstance(binding, dict):
            reason = 'EXPLICIT_OPERATOR_CONFIGURATION_BINDING_REQUIRED'
        elif (any(not _nonempty(binding.get(name)) for name in
                  ('kernel_id', 'precision', 'execution_key', 'weight_tensor_name')) or
              type(binding.get('batch_size')) is not int or binding['batch_size'] < 1 or
              not _source(binding.get('batch_evidence')) or
              not _source(binding.get('operator_to_kernel_evidence'))):
            reason = 'INCOMPLETE_OPERATOR_CONFIGURATION_EVIDENCE'
        elif record.get('operator_kind') != 'linear_declared':
            reason = 'UNSUPPORTED_OPERATOR_DECLARATION'
        else:
            weights = [tensor for tensor in record.get('direct_tensors', [])
                       if tensor.get('name') == binding['weight_tensor_name']]
            if len(weights) != 1:
                reason = 'ONE_EXPLICIT_WEIGHT_TENSOR_REQUIRED'
            elif any(not isinstance(record.get(name), dict) for name in ('input', 'output')):
                reason = 'INCOMPLETE_INPUT_OUTPUT_METADATA'
            else:
                values = dict(batch_size=binding['batch_size'],
                              weight_bytes=weights[0].get('logical_bytes'),
                              input_tensor_bytes=record['input'].get('logical_bytes'),
                              output_tensor_bytes=record['output'].get('logical_bytes'),
                              flops=record.get('dense_gemm_flops'))
                if any(type(value) is not int or value < 0 for value in values.values()):
                    reason = 'INCOMPLETE_LOGICAL_BYTES_OR_DECLARED_GEMM_FLOPS'
        if reason:
            rejected.append(dict(id=identity, **_unknown(reason)))
            continue
        configurations[identity] = dict(
            operator='linear_declared', kernel_id=binding['kernel_id'],
            precision=binding['precision'], execution_key=binding['execution_key'],
            feature_schema_id='gsdeploy.affine_single_weight_logical_bytes_operator_dense_gemm.v1',
            features=values,
            feature_evidence={name: [copy.deepcopy(binding['batch_evidence'] if name == 'batch_size'
                                                 else observer_source)] for name in FEATURES},
            operator_to_kernel_evidence=copy.deepcopy(binding['operator_to_kernel_evidence']),
            feature_interpretation='Operator-level covariates; selected weight logical bytes, not unique storage; dense operator FLOPs, not measured kernel work.')
    return dict(configurations=configurations, rejected_rows=rejected,
                status='ADAPTED' if not rejected else 'PARTIAL_OR_UNKNOWN',
                training_targets_created=False)


def adapt_raw_kernel_profiles(raw_rows, manifest):
    """Join real raw measurements with explicit configuration feature sidecars.

    The manifest carries capture provenance and ``configurations[config_id]``.
    Each configuration supplies operator/kernel/precision/execution identity,
    feature_schema_id, five feature values, and receipts for every feature.
    No inference from kernel names, dimensions, tensor aliases or host clocks.
    Missing evidence yields rejected UNKNOWN rows, never default feature zeros.
    """
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Unsupported capture manifest schema')
    accepted, rejected = [], []
    bindings = manifest.get('configurations', {})
    if not isinstance(bindings, dict):
        raise ValueError('Configuration bindings must be a mapping')
    seen = set()
    capture_fields = ('run_id', 'split', 'evidence_kind', 'hardware_id',
                      'execution_context_sha256', 'timing_basis', 'profiler_id',
                      'profiler_perturbed', 'perturbation_flags',
                      'isolation_evidence', 'measurement_source')
    config_fields = ('operator', 'kernel_id', 'precision', 'execution_key',
                     'feature_schema_id', 'features', 'feature_evidence')
    for index, raw in enumerate(raw_rows):
        identity = raw.get('id')
        if not _nonempty(identity) or identity in seen:
            raise ValueError('Raw measurement identities must be unique nonempty strings')
        seen.add(identity)
        config = bindings.get(raw.get('config_id'))
        if not isinstance(config, dict):
            rejected.append(dict(id=identity, index=index,
                                 **_unknown('MISSING_CONFIGURATION_BINDING')))
            continue
        row = {name: copy.deepcopy(manifest.get(name)) for name in capture_fields}
        row.update({name: copy.deepcopy(config.get(name)) for name in config_fields})
        for name in ('operator_to_kernel_evidence', 'feature_interpretation'):
            if name in config:
                row[name] = copy.deepcopy(config[name])
        row.update(id=identity, duration_ns=raw.get('duration_ns'),
                   intermediate_peak_bytes=raw.get('intermediate_peak_bytes'),
                   peak_measurement_source=copy.deepcopy(raw.get('peak_measurement_source')))
        try:
            # Fixtures remain clearly marked; the estimator enforces opt-in.
            _validate_row(row, allow_cpu_fixture=True)
        except ValueError as exc:
            rejected.append(dict(id=identity, index=index,
                                 **_unknown('INCOMPLETE_MEASUREMENT_EVIDENCE', detail=str(exc))))
        else:
            accepted.append(row)
    return dict(schema=SCHEMA, rows=accepted, rejected_rows=rejected,
                status='ADAPTED' if not rejected else 'PARTIAL_OR_UNKNOWN',
                natural_timing_validated=False, deployment_qualified=False)


def trace_kernel_order(trace):
    """Canonical exact launch order, including unattributed CUPTI launches.

    ``nvtx_ranges`` is canonical in the exporter. ``phases`` repeats some of
    those kernels and must not be concatenated, which would double count them.
    """
    if trace.get('schema') != 'gsdeploy.nsys_observed_phases.v1':
        raise ValueError('Unsupported Nsight export schema')
    kernels = [kernel for region in trace.get('nvtx_ranges', [])
               for kernel in region.get('kernels', [])]
    kernels += list(trace.get('unattributed_kernels', []))
    keys = ('nsys_start_ns', 'nsys_end_ns', 'pid', 'stream', 'correlation_id')
    seen = set()
    for kernel in kernels:
        identity = tuple(kernel.get(key) for key in keys)
        if any(type(value) is not int for value in identity) or identity[1] <= identity[0]:
            raise ValueError('Incomplete/invalid CUPTI launch identity or interval')
        if identity in seen:
            raise ValueError('Duplicate exact CUPTI launch identity')
        seen.add(identity)
    return sorted(kernels, key=lambda row: tuple(row[key] for key in keys))


def adapt_nsys_trace(trace, manifest):
    """Convert CUPTI intervals only with source-bound isolated-profile sidecar.

    ``kernel_bindings`` keys are decimal canonical launch indices. Each binding
    repeats the exact launch identity and maps to a config_id. An exporter alone
    proves neither isolation nor operator features; absent sidecars stay UNKNOWN.
    """
    kernels = trace_kernel_order(trace)
    reason = None
    source = manifest.get('measurement_source')
    if not _source(source) or source not in trace.get('sources', []):
        reason = 'EXACT_TRACE_SOURCE_RECEIPT_REQUIRED'
    elif manifest.get('timing_basis') != 'profiled_isolated':
        reason = 'NSYS_REQUIRES_PROFILED_ISOLATED_BASIS'
    elif not _source(manifest.get('isolation_evidence')):
        reason = 'ISOLATED_EXECUTION_EVIDENCE_REQUIRED'
    if reason:
        return dict(schema=SCHEMA, rows=[], rejected_rows=[
            dict(id=f"{manifest.get('run_id', 'unknown')}:kernel_{i:07d}", index=i,
                 **_unknown(reason)) for i in range(len(kernels))],
            status='UNKNOWN', natural_timing_validated=False, deployment_qualified=False)
    raw, rejected = [], []
    bindings = manifest.get('kernel_bindings', {})
    identity_fields = ('nsys_start_ns', 'nsys_end_ns', 'pid', 'stream', 'correlation_id')
    for index, kernel in enumerate(kernels):
        identity = f"{manifest.get('run_id', 'unknown')}:kernel_{index:07d}"
        binding = bindings.get(str(index))
        launch = {name: kernel[name] for name in identity_fields}
        if not isinstance(binding, dict) or binding.get('launch_identity') != launch:
            rejected.append(dict(id=identity, index=index,
                                 **_unknown('EXACT_LAUNCH_CONFIGURATION_BINDING_REQUIRED')))
            continue
        config = manifest.get('configurations', {}).get(binding.get('config_id'), {})
        if config.get('kernel_id') != kernel.get('name'):
            rejected.append(dict(id=identity, index=index,
                                 **_unknown('BOUND_KERNEL_IDENTITY_MISMATCH')))
            continue
        raw.append(dict(id=identity, config_id=binding['config_id'],
                        duration_ns=kernel['nsys_end_ns']-kernel['nsys_start_ns'],
                        intermediate_peak_bytes=None))
    result = adapt_raw_kernel_profiles(raw, manifest)
    result['rejected_rows'] += rejected
    if result['rejected_rows']:
        result['status'] = 'PARTIAL_OR_UNKNOWN'
    result['kernel_count'] = len(kernels)
    return result


class KernelBaselineEstimator:
    """Affine feature estimator with exact domains and convex-hull support.

    Parameters and diagnostics describe training fit, not confidence bounds.
    Collinear coefficients are allowed: predictions are only returned inside
    the observed affine subspace AND convex hull, where they are identifiable.
    """
    def __init__(self, rows, *, min_observations=3, min_runs=2,
                 max_relative_training_error=0.25, allow_cpu_fixture=False):
        if type(min_observations) is not int or min_observations < 2:
            raise ValueError('At least two observations required')
        if type(min_runs) is not int or min_runs < 1:
            raise ValueError('Positive minimum run count required')
        if not _finite(max_relative_training_error):
            raise ValueError('Nonnegative residual limit required')
        self.rows = copy.deepcopy(list(rows))
        self.allow_cpu_fixture = allow_cpu_fixture
        self.min_observations, self.min_runs = min_observations, min_runs
        self.max_relative_training_error = max_relative_training_error
        self.groups = {}
        self.training_ids, self.training_runs, self.training_sources = set(), set(), set()
        for row in self.rows:
            _validate_row(row, split='training', allow_cpu_fixture=allow_cpu_fixture)
            if row['id'] in self.training_ids:
                raise ValueError('Duplicate training observation ID')
            self.training_ids.add(row['id'])
            self.training_runs.add(row['run_id'])
            self.training_sources.add(row['measurement_source']['sha256'])
            if row.get('intermediate_peak_bytes') is not None:
                self.training_sources.add(row['peak_measurement_source']['sha256'])
            self.groups.setdefault(_group(row), []).append(row)
        self.models = {key: {target: self._fit(group, target) for target in TARGETS}
                       for key, group in self.groups.items()}
        frozen = dict(schema=SCHEMA, rows=self.rows, min_observations=min_observations,
                      min_runs=min_runs, max_relative_training_error=max_relative_training_error)
        self.training_sha256 = hashlib.sha256(json.dumps(frozen, sort_keys=True,
                                                        allow_nan=False).encode()).hexdigest()

    def _fit(self, rows, target):
        rows = [row for row in rows if row.get(target) is not None]
        runs = sorted({row['run_id'] for row in rows})
        if len(rows) < self.min_observations or len(runs) < self.min_runs:
            return _unknown('INSUFFICIENT_TARGET_OBSERVATIONS_OR_RUNS',
                            observation_count=len(rows), run_count=len(runs))
        values = np.array([[row['features'][name] for name in FEATURES] for row in rows], dtype=float)
        center = values.mean(axis=0)
        scale = np.ptp(values, axis=0)
        scale[scale == 0] = 1
        normalized = (values-center)/scale
        design = np.column_stack((np.ones(len(rows)), normalized))
        targets = np.array([row[target] for row in rows], dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
        fitted = design @ coefficients
        relative = np.abs(fitted-targets)/np.maximum(np.abs(targets), 1)
        max_error = float(relative.max())
        diagnostics = dict(observation_count=len(rows), run_count=len(runs),
                           affine_design_rank=int(rank), coefficient_count=len(FEATURES)+1,
                           coefficients_individually_identifiable=rank == len(FEATURES)+1,
                           max_relative_training_error=max_error,
                           training_error_is_generalization_bound=False)
        if max_error > self.max_relative_training_error:
            return _unknown('TRAINING_RESIDUAL_EXCEEDS_LIMIT', diagnostics=diagnostics)
        return dict(known=True, center=center, scale=scale, normalized=normalized,
                    design=design, coefficients=coefficients, diagnostics=diagnostics)

    def predict(self, query):
        """Return independent duration/peak estimates or explicit UNKNOWNs."""
        # Query requires the same evidence and timing domain as observations,
        # but no measured target; placeholder duration is used only by validation.
        checked = dict(query, id='__query__', run_id='__query__', split='training',
                       duration_ns=1, intermediate_peak_bytes=None,
                       measurement_source=dict(path='query_not_a_measurement', sha256='0'*64))
        _validate_row(checked, split='training', allow_cpu_fixture=self.allow_cpu_fixture)
        key = _group(checked)
        if key not in self.models:
            return _unknown('EXACT_TRAINING_DOMAIN_NOT_FOUND', targets={
                name: _unknown('EXACT_TRAINING_DOMAIN_NOT_FOUND') for name in TARGETS})
        predictions = {}
        for target, model in self.models[key].items():
            if not model['known']:
                predictions[target] = copy.deepcopy(model)
                continue
            point = (np.array([query['features'][name] for name in FEATURES], dtype=float)-model['center'])/model['scale']
            # The affine hull test makes the collinearity limitation explicit.
            query_design = np.r_[1., point]
            weights = np.linalg.lstsq(model['design'].T, query_design, rcond=None)[0]
            if np.max(np.abs(model['design'].T @ weights-query_design)) > 1e-8:
                predictions[target] = _unknown('OUTSIDE_TRAINING_AFFINE_SUBSPACE')
                continue
            hull = linprog(np.zeros(len(model['normalized'])), A_eq=model['design'].T,
                           b_eq=query_design, bounds=(0, None), method='highs')
            if not hull.success or np.max(np.abs(model['design'].T @ hull.x-query_design)) > 1e-7:
                predictions[target] = _unknown('OUTSIDE_TRAINING_CONVEX_HULL')
                continue
            value = float(query_design @ model['coefficients'])
            if not math.isfinite(value) or value < 0 or (target == 'duration_ns' and value == 0):
                predictions[target] = _unknown('NONPHYSICAL_REGRESSION_PREDICTION')
                continue
            predictions[target] = dict(status='ISOLATED_BASELINE_ESTIMATE', known=True,
                                       value=value, unit='ns' if target == 'duration_ns' else 'bytes',
                                       diagnostics=copy.deepcopy(model['diagnostics']))
        return dict(status='BASELINE_ESTIMATE' if predictions['duration_ns']['known'] else 'UNKNOWN',
                    known=predictions['duration_ns']['known'], targets=predictions,
                    timing_basis=query['timing_basis'], profiler_id=query['profiler_id'],
                    profiler_perturbed=query['profiler_perturbed'],
                    perturbation_flags=list(query['perturbation_flags']),
                    training_sha256=self.training_sha256,
                    cpu_fixture=any(row['evidence_kind'] == 'cpu_fixture' for row in self.groups[key]),
                    natural_timing_validated=False, deployment_qualified=False)

    def evaluate(self, rows):
        """Evaluate untouched validation observations; never update the fit."""
        rows = copy.deepcopy(list(rows))
        ids = set()
        for row in rows:
            _validate_row(row, split='validation', allow_cpu_fixture=self.allow_cpu_fixture)
            if row['id'] in ids or row['id'] in self.training_ids or row['run_id'] in self.training_runs:
                raise ValueError('Validation identities/runs overlap training or repeat')
            sources = {row['measurement_source']['sha256']}
            if row.get('intermediate_peak_bytes') is not None:
                sources.add(row['peak_measurement_source']['sha256'])
            if sources & self.training_sources:
                raise ValueError('Validation measurement sources overlap training')
            ids.add(row['id'])
        records, errors = [], {name: [] for name in TARGETS}
        for row in rows:
            prediction = self.predict(row)
            result = dict(id=row['id'], prediction=prediction, errors={})
            for target in TARGETS:
                estimate = prediction['targets'][target]
                if row.get(target) is not None and estimate['known']:
                    error = abs(estimate['value']-row[target])/max(abs(row[target]), 1)
                    errors[target].append(error)
                    result['errors'][target] = dict(absolute_error=abs(estimate['value']-row[target]),
                                                  relative_error=error)
            records.append(result)
        return dict(training_sha256=self.training_sha256, observation_count=len(rows), records=records,
                    metrics={name: dict(evaluated_count=len(values),
                                        max_relative_error=max(values) if values else None,
                                        mean_relative_error=float(np.mean(values)) if values else None)
                             for name, values in errors.items()},
                    natural_timing_validated=False, deployment_qualified=False)
