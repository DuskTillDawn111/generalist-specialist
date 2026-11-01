"""Read-only exact-source query bridge for the frozen baseline predictor.

Uses metadata covariates and fitted coefficients. Measured target durations are
discarded from the query index; missing kernels never fall back to trace times.
This exact observed-training-source bridge is not an unseen-input feature map.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from gsdeploy.forecasting.kernel_baseline import FEATURES, GROUP_FIELDS, KernelBaselineEstimator
from .kernel_baseline_intake_v3 import checked, _verify


def hydrate_estimator(report):
    """Reconstruct frozen predictor state without refitting or changing gates."""
    if report.get('schema')!='gsdeploy.real_kernel_baseline_fit.v3':raise ValueError('Real baseline fit required')
    fit=KernelBaselineEstimator.__new__(KernelBaselineEstimator)
    fit.allow_cpu_fixture=False;fit.training_sha256=report['training_sha256'];fit.models={};fit.groups={}
    for group in report['groups']:
        key=tuple(group['group'][k] for k in GROUP_FIELDS)+(tuple(sorted(group['perturbation_flags'])),)
        models=copy.deepcopy(group['models'])
        for model in models.values():
            if model['known']:
                for field in ('center','scale','normalized','design','coefficients'):
                    model[field]=np.array(model[field],dtype=float)
        fit.models[key]=models
        # predict() only uses this member field to preserve fixture provenance.
        fit.groups[key]=[dict(evidence_kind='measured')]
    return fit


def unknown(reason):
    return dict(known=False,status='UNKNOWN',reasons=[reason],
                natural_timing_validated=False,deployment_qualified=False,
                measured_target_fallback_used=False)


class ExactSourceBaselineResolver:
    """Callable canonical kernel -> prediction, with exact receipt/static join.

    Canonical kernel must carry source_sqlite receipt, pid, correlation_id,
    nsys_start_ns, nsys_end_ns, context_id and all observed static geometry.
    Returned duration_ns is a regression prediction, never the measured input.
    """
    def __init__(self,fit_report,intakes):
        from .native_collect import receipt
        report=checked(receipt(fit_report))
        for source in report['implementation_sources']:_verify(source)
        self.fit=hydrate_estimator(report);self.index={};self._prediction_cache={}
        accepted_intakes={s['sha256'] for s in report['inputs']}
        for path in intakes:
            source=receipt(path)
            if source['sha256'] not in accepted_intakes:raise ValueError('Intake is not an input of this fitted model')
            intake=checked(source)
            for implementation in intake['sources']:_verify(implementation)
            review=checked(intake['light_review']);_verify(intake['rows'])
            launches={}
            for coverage in intake['coverage']:
                binding=checked(coverage['bindings'])
                sql=review['owners'][coverage['alias']]['sqlite'];_verify(sql)
                for launch in binding['launches']:
                    actual=launch['launch'];key=(sql['sha256'],actual['correlation_id'],actual['nsys_start_ns'])
                    if key in launches:raise ValueError('Ambiguous exact launch source')
                    launches[key]=launch
            with Path(intake['rows']['path']).open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['run_id']!=intake['capture_run_id']:raise ValueError('Changed capture identity')
                    tail=row['id'].rsplit(':',2);key=(row['measurement_source']['sha256'],int(tail[-2]),int(tail[-1]))
                    launch=launches.get(key)
                    if launch is None:raise ValueError('Accepted observation lacks exact launch binding')
                    # Targets are intentionally absent from this query index.
                    query={k:v for k,v in row.items() if k not in ('duration_ns','intermediate_peak_bytes','peak_measurement_source')}
                    if key in self.index:raise ValueError('Duplicate source kernel')
                    self.index[key]=dict(query=query,launch=launch,source=row['measurement_source'])

    def __call__(self,kernel):
        if 'timing_evidence' in kernel:
            segment=kernel;evidence=segment['timing_evidence'];identity=evidence.get('identity',[])
            if len(identity)!=7:return unknown('EXACT_ORDERED_SEGMENT_IDENTITY_REQUIRED')
            kernel=dict(segment['kernel'],source_sqlite=evidence['source'],
                nsys_start_ns=identity[0],nsys_end_ns=identity[1],pid=identity[2],stream=identity[3],
                correlation_id=identity[4],device=identity[5],context_id=identity[6])
        source=kernel.get('source_sqlite')
        if not isinstance(source,dict):return unknown('EXACT_SQLITE_RECEIPT_REQUIRED')
        key=(source.get('sha256'),kernel.get('correlation_id'),kernel.get('nsys_start_ns'))
        entry=self.index.get(key)
        if entry is None:return unknown('NO_METADATA_FEATURE_BINDING_FOR_EXACT_KERNEL')
        if source!=entry['source']:return unknown('SQLITE_SOURCE_RECEIPT_MISMATCH')
        launch=entry['launch']
        for field in ('pid','context_id','correlation_id','nsys_start_ns','nsys_end_ns'):
            if kernel.get(field)!=launch['launch'][field]:return unknown('EXACT_LAUNCH_IDENTITY_MISMATCH:'+field)
        for field in ('name','block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster'):
            if field=='name' and 'name' not in kernel:continue  # Bound by the exact immutable source launch.
            if kernel.get(field)!=launch['signature'][field]:return unknown('EXACT_STATIC_SIGNATURE_MISMATCH:'+field)
        query=entry['query']
        cache_key=tuple(query[k] for k in GROUP_FIELDS)+(tuple(sorted(query['perturbation_flags'])),tuple(query['features'][f] for f in FEATURES))
        if not hasattr(self,'_prediction_cache'):self._prediction_cache={}
        if cache_key not in self._prediction_cache:self._prediction_cache[cache_key]=self.fit.predict(query)
        prediction=copy.deepcopy(self._prediction_cache[cache_key])
        if not prediction['known']:
            return dict(**unknown('BASELINE_MODEL_OR_FEATURE_SUPPORT_UNKNOWN'),prediction=prediction)
        return dict(known=True,status='PROFILED_KERNEL_BASELINE_PREDICTION',
            duration_ns=prediction['targets']['duration_ns']['value'],prediction=prediction,
            feature_source=entry['query']['operator_to_kernel_evidence'],
            observation_id=entry['query']['id'],training_sha256=self.fit.training_sha256,
            kernel_name=launch['signature']['name'],
            feature_scope='Exact measured training-profile metadata; not a heldout input mapper',
            measured_target_fallback_used=False,natural_timing_validated=False,deployment_qualified=False)
