"""CPU regression for real request-attributed CUPTI copy/memset training rows.

Bytes, direction, memory kind, API and measurement domain remain explicit.
Copy engine independence/DRAM interference and natural timing are unvalidated.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter,defaultdict
import json
import sqlite3
from pathlib import Path

from .kernel_baseline_intake_v3 import checked,_verify,sqlite_identity
from .kernel_semantic_pooling_v5 import fit_group,predict_group
from .native_execution_projection_v5 import native_execution_projection
from .native_collect import receipt,save
from .profiling import digest
from .trace_export import linux_pid


def fit(semantic_fit_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    fit_source=receipt(semantic_fit_path);semantic_fit=checked(fit_source);requests=checked(semantic_fit['request_sequences'])
    contexts={s['context']['path']:checked(s['context']) for s in semantic_fit['contexts']}
    projections={s['projection']['original_full_context_sha256']:s['projection'] for s in semantic_fit['contexts']}
    by_source=defaultdict(list)
    for request in requests:by_source[(request['source_sqlite']['path'],request['source_sqlite']['sha256'])].append(request)
    groups={};rows=[];group_rows=defaultdict(list);unattributed=Counter();seen=set()
    for (path,sha),source_requests in by_source.items():
        source=dict(path=path,sha256=sha);_verify(source);hardware,versions=sqlite_identity(source)
        parsed=checked(source_requests[0]['source_parsed']);phases=parsed['phases']
        context=next(context for context in contexts.values() if native_execution_projection(context)['original_full_context_sha256']==source_requests[0]['full_context_sha256'])
        projection=projections[source_requests[0]['full_context_sha256']]
        ranges=[]
        for request in source_requests:
            actual=[p for p in phases if p['request_id']==request['request_id']]
            if not actual:raise ValueError('Actual request stage ranges required')
            pids={p['pid'] for p in actual}
            if len(pids)!=1:raise ValueError('Single process per request required')
            for p in actual:ranges.append((p['pid'],p['nsys_start_ns'],p['nsys_end_ns'],request))
        with sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True) as connection:
            connection.row_factory=sqlite3.Row
            runtime={}
            for row in connection.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id'):
                key=(linux_pid(row['globalTid']),row['correlationId'])
                if key in runtime:raise ValueError('Ambiguous CUDA runtime correlation')
                runtime[key]=dict(row)
            tables={r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
            for table,kind in (('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')):
                if table not in tables:continue
                for row in connection.execute('select rowid,* from '+table):
                    pid=linux_pid(row['globalPid']);launch=runtime.get((pid,row['correlationId']))
                    if launch is None:unattributed['missing_runtime']+=1;continue
                    matched={r['request_id']:r for rp,start,end,r in ranges if rp==pid and start<=launch['start'] and launch['end']<=end}
                    if len(matched)!=1:unattributed['outside_or_ambiguous_request_stage']+=1;continue
                    request=next(iter(matched.values()));memory={k:row[k] for k in
                        ('copyKind','srcKind','dstKind','srcDeviceId','dstDeviceId','memKind','value') if k in row.keys()}
                    if row['bytes']<1:unattributed['nonpositive_bytes']+=1;continue
                    group=dict(native_projection_sha256=projection['projection_sha256'],hardware_id=hardware,
                        profiler_id='NVIDIA Nsight Systems:'+versions['EXPORT_PRODUCT_VERSION']+':light-kernel',
                        measurement_domain='light-kernel',family=request['family'],activity_kind=kind,
                        runtime_api=launch['api_name'],memory_direction_and_kind=memory,
                        feature_schema_id='gsdeploy.actual_cuda_copy_memset_bytes.v1')
                    group_id=digest(group);groups[group_id]=group
                    identity=dict(rowid=row['rowid'],kind=kind,pid=pid,device=row['deviceId'],context_id=row['contextId'],
                        stream=row['streamId'],correlation_id=row['correlationId'],start_ns=row['start'],end_ns=row['end'])
                    key=(sha,kind,row['rowid'])
                    if key in seen:raise ValueError('Duplicate exact copy/memset observation')
                    seen.add(key);duration=row['end']-row['start']
                    if duration<=0:raise ValueError('Positive actual CUPTI interval required')
                    record=dict(id=f'{request["run_id"]}:{request["alias"]}:{request["request_id"]}:{kind}:{row["rowid"]}',
                        run_id=request['run_id'],group_id=group_id,features=dict(actual_bytes=row['bytes']),duration_ns=duration,
                        source_sqlite=source,source_parsed=request['source_parsed'],source_raw=request['source_raw'],
                        identity=identity,memory_metadata=dict(bytes=row['bytes'],**{k:row[k] for k in ('copyKind','srcKind','dstKind','memKind') if k in row.keys()}),
                        runtime_api=launch['api_name'],family=request['family'],request_id=request['request_id'],alias=request['alias'])
                    rows.append(record);group_rows[group_id].append(record)
    models={key:fit_group(records,['actual_bytes']) for key,records in group_rows.items()}
    for model in models.values():model['regression']='nonnegative_linear_duration_from_actual_copy_or_memset_bytes'
    counts=Counter();row_counts=Counter();family=Counter()
    for key,model in models.items():
        label='ACCEPTED_TRAINING_FIT' if model['fit_accepted'] else '+'.join(model['reasons']);counts[label]+=1;row_counts[label]+=model['observation_count']
    for row in rows:family[row['family']+':'+groups[row['group_id']]['activity_kind']]+=1
    save(output/'models.json',dict(groups=groups,models=models))
    with (output/'observations.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    report=dict(schema='gsdeploy.copy_memset_bytes_baseline_fit.v5',semantic_fit=fit_source,
        models=receipt(output/'models.json'),observations=receipt(output/'observations.jsonl'),
        observation_count=len(rows),group_count=len(groups),capture_run_count=len({r['run_id'] for r in rows}),
        fit_group_counts=dict(counts),fit_observation_counts=dict(row_counts),family_activity_counts=dict(family),unattributed_activity_counts=dict(unattributed),
        thresholds=dict(min_observations=3,min_runs=2,max_relative_training_error=0.25),
        request_attribution='Actual owning process and CUDA runtime issue interval fully contained by one request stage NVTX range; initialization/priming/unmatched rows excluded',
        direction_values_are_actual_cupti_enums=True,byte_values_are_actual_cupti=True,
        independent_copy_engine_service_is_hypothesis=True,DRAM_interference_modeled=False,
        native_projection_pooling_is_hypothesis=True,heldout_validation_performed=False,natural_timing_validated=False,
        deployment_qualified=False,GPU_workload_executed=False,model_weights_loaded=False,
        implementation_sources=[receipt(__file__),receipt(Path(__file__).with_name('kernel_semantic_pooling_v5.py')),
            receipt(Path(__file__).with_name('native_execution_projection_v5.py'))])
    save(output/'report.json',report);return report


class ExactSourceCopyMemsetHypothesisResolver:
    def __init__(self,fit_path):
        self.fit_source=receipt(fit_path);fit=checked(self.fit_source)
        for source in fit['implementation_sources']:_verify(source)
        self.models=checked(fit['models'])['models'];self.index={};self.cache={};_verify(fit['observations']);verified=set()
        with Path(fit['observations']['path']).open() as stream:
            for line in stream:
                row=json.loads(line);source=row['source_sqlite']
                if source['sha256'] not in verified:_verify(source);verified.add(source['sha256'])
                identity=row['identity'];key=(source['sha256'],identity['kind'],identity['rowid'])
                self.index[key]={k:v for k,v in row.items() if k!='duration_ns'}

    def __call__(self,node,source_sqlite=None):
        from .kernel_semantic_query_v5 import unknown
        source=source_sqlite or node.get('source_sqlite')
        if not isinstance(source,dict):return unknown('EXACT_COPY_MEMSET_SQLITE_RECEIPT_REQUIRED')
        try:rowid=int(node['id'].split(':')[-1])
        except (ValueError,KeyError):return unknown('EXACT_COPY_MEMSET_ROWID_REQUIRED')
        entry=self.index.get((source.get('sha256'),node.get('kind'),rowid))
        if entry is None:return unknown('NO_REQUEST_ATTRIBUTED_COPY_MEMSET_TRAINING_BINDING')
        if source!=entry['source_sqlite']:return unknown('EXACT_COPY_MEMSET_SOURCE_MISMATCH')
        actual=entry['identity'];checks=dict(observed_start_ns=actual['start_ns'],observed_end_ns=actual['end_ns'],correlation_id=actual['correlation_id'])
        if any(node.get(k)!=v for k,v in checks.items()):return unknown('EXACT_COPY_MEMSET_IDENTITY_MISMATCH')
        if node.get('stream_key')!=[actual['pid'],actual['device'],actual['context_id'],actual['stream']]:return unknown('EXACT_COPY_MEMSET_STREAM_IDENTITY_MISMATCH')
        if node.get('memory_metadata')!=entry['memory_metadata'] or node.get('api_name')!=entry['runtime_api']:
            return unknown('EXACT_COPY_MEMSET_BYTES_KIND_DIRECTION_API_MISMATCH')
        key=(entry['group_id'],entry['features']['actual_bytes'])
        if key not in self.cache:self.cache[key]=predict_group(self.models[entry['group_id']],entry['features'])
        p=copy.deepcopy(self.cache[key]);value=p['duration_hypothesis_ns']
        result=dict(known=p['known'],strict_known=p['known'],numeric_hypothesis_available=value is not None,
            hypothesis_only=True,strict_failure_reasons=list(p['reasons']),prediction=p,
            fit_source=self.fit_source,observation_id=entry['id'],activity_kind=node['kind'],actual_bytes=entry['features']['actual_bytes'],
            physical_accuracy=False,measured_target_fallback_used=False,natural_timing_validated=False,deployment_qualified=False)
        if value is not None:result['duration_ns']=value
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--semantic-fit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=fit(a.semantic_fit,a.output);print({k:r[k] for k in ('observation_count','group_count','fit_group_counts','fit_observation_counts','family_activity_counts')})


if __name__=='__main__':main()
