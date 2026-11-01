"""Exact-source numeric semantic regression hypotheses for explicit gate modes.

The query index contains features and exact launch identities, never measured
duration targets. Numeric availability is separate from strict known/support.
This resolver maps retained training-source DAGs, not unseen-input metadata.
"""
from __future__ import annotations

import copy
import math

from .kernel_baseline_intake_v3 import checked,_verify
from .kernel_domain_bridge_review_v3 import request_kernels
from .kernel_launch_pooling_v4 import geometry
from .kernel_semantic_pooling_v5 import predict_group
from .native_collect import receipt


def unknown(reason):
    return dict(known=False,strict_known=False,numeric_hypothesis_available=False,status='UNKNOWN',
        reasons=[reason],strict_failure_reasons=[reason],hypothesis_only=True,physical_accuracy=False,
        measured_target_fallback_used=False,natural_timing_validated=False,deployment_qualified=False)


class ExactSourceSemanticHypothesisResolver:
    """Callable canonical kernel/ordered segment -> regression hypothesis.

    `known` retains the strict training-residual and convex-hull result. An
    explicitly authorized hypothesis consumer may use `duration_ns` only when
    `numeric_hypothesis_available` is true; this never grants physical accuracy.
    """
    def __init__(self,fit_report):
        self.fit_source=receipt(fit_report);report=checked(self.fit_source)
        if report['schema']!='gsdeploy.semantic_shape_kernel_baseline_fit.v5':raise ValueError('Actual semantic v5 fit required')
        for source in report['implementation_sources']:_verify(source)
        models=checked(report['models']);self.models=models['models'];self.groups=models['groups']
        self.index={};self._prediction_cache={};requests=checked(report['request_sequences']);verified=set();parsed_cache={}
        allowed_runs=set(report['capture_run_ids'])
        for request in requests:
            if request['run_id'] not in allowed_runs:raise ValueError('Query source capture absent from actual fit')
            for field in ('source_raw','source_sqlite','source_parsed','source_mapping','source_metadata'):
                source=request[field];key=(source['path'],source['sha256'])
                if key not in verified:_verify(source);verified.add(key)
            parsed_source=request['source_parsed'];parsed_key=(parsed_source['path'],parsed_source['sha256'])
            if parsed_key not in parsed_cache:parsed_cache[parsed_key]=checked(parsed_source)
            # The bridge request row must be recovered from its immutable raw,
            # keeping actual semantic ranges and request ownership explicit.
            raw_key=(request['source_raw']['path'],request['source_raw']['sha256'])
            if raw_key not in parsed_cache:parsed_cache[raw_key]=checked(request['source_raw'])
            raw=parsed_cache[raw_key]
            row=next(r for r in raw['request_records'] if r['request_id']==request['request_id'] and r['model']==request['alias'] and r['pool_id']==request['pool_id'])
            sequence=request_kernels(parsed_cache[parsed_key],request['source_sqlite']['path'],row)
            launches={(item['kernel']['correlation_id'],item['kernel']['nsys_start_ns']):item for item in sequence}
            if len(launches)!=len(sequence) or len(sequence)!=len(request['ordered_rows']):raise ValueError('Exact source request sequence changed')
            for ordered,item in zip(request['ordered_rows'],sequence):
                kernel=item['kernel'];signature=item['signature'];source=request['source_sqlite']
                expected=f'{request["run_id"]}:{request["alias"]}:{request["request_id"]}:{kernel["correlation_id"]}:{kernel["nsys_start_ns"]}'
                if ordered['id']!=expected:raise ValueError('Feature row does not bind this exact ordered launch')
                if any(ordered['features'][f]!=value for f,value in geometry(signature).items()):raise ValueError('Feature geometry differs from actual source launch')
                key=(source['sha256'],kernel['correlation_id'],kernel['nsys_start_ns'])
                if key in self.index:raise ValueError('Duplicate exact query launch')
                # Retain only identities/static signature/features. Raw response
                # timing and actual CUPTI interval duration are never indexed.
                self.index[key]=dict(source=copy.deepcopy(source),signature=copy.deepcopy(signature),
                    identity={f:kernel[f] for f in ('pid','context_id','correlation_id','nsys_start_ns','nsys_end_ns','stream')},
                    group_id=ordered['group_id'],features=copy.deepcopy(ordered['features']),observation_id=ordered['id'],
                    feature_sources=[request['source_metadata'],request['source_mapping'],request['source_parsed'],source])

    def __call__(self,kernel):
        if 'timing_evidence' in kernel:
            segment=kernel;ev=segment['timing_evidence'];identity=ev.get('identity',[])
            if len(identity)!=7:return unknown('EXACT_ORDERED_SEGMENT_IDENTITY_REQUIRED')
            kernel=dict(segment['kernel'],source_sqlite=ev['source'],nsys_start_ns=identity[0],nsys_end_ns=identity[1],
                pid=identity[2],stream=identity[3],correlation_id=identity[4],device=identity[5],context_id=identity[6])
        source=kernel.get('source_sqlite')
        if not isinstance(source,dict):return unknown('EXACT_SQLITE_RECEIPT_REQUIRED')
        entry=self.index.get((source.get('sha256'),kernel.get('correlation_id'),kernel.get('nsys_start_ns')))
        if entry is None:return unknown('NO_EXACT_SEMANTIC_FEATURE_BINDING')
        if source!=entry['source']:return unknown('EXACT_SQLITE_RECEIPT_MISMATCH')
        for field in ('pid','context_id','correlation_id','nsys_start_ns','nsys_end_ns'):
            if kernel.get(field)!=entry['identity'][field]:return unknown('EXACT_LAUNCH_IDENTITY_MISMATCH:'+field)
        for field in ('name','block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster'):
            if field=='name' and field not in kernel:continue
            if kernel.get(field)!=entry['signature'][field]:return unknown('EXACT_STATIC_SIGNATURE_MISMATCH:'+field)
        group_id=entry['group_id'];model=self.models[group_id]
        cache_key=(group_id,tuple((f,entry['features'][f]) for f in sorted(entry['features'])))
        if cache_key not in self._prediction_cache:self._prediction_cache[cache_key]=predict_group(model,entry['features'])
        prediction=copy.deepcopy(self._prediction_cache[cache_key]);value=prediction['duration_hypothesis_ns']
        numeric=type(value) in (int,float) and math.isfinite(value) and value>0
        result=dict(known=prediction['known'],strict_known=prediction['known'],numeric_hypothesis_available=numeric,
            status='NUMERIC_SEMANTIC_DURATION_HYPOTHESIS' if numeric else 'UNKNOWN',hypothesis_only=True,
            strict_failure_reasons=list(prediction['reasons']),prediction=prediction,
            observation_id=entry['observation_id'],kernel_name=entry['signature']['name'],
            feature_sources=entry['feature_sources'],fit_source=self.fit_source,
            feature_scope='Exact retained training-source features; not unseen-input feature synthesis',
            physical_accuracy=False,measured_target_fallback_used=False,natural_timing_validated=False,deployment_qualified=False)
        if numeric:result['duration_ns']=value
        return result
