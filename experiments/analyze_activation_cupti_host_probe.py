"""Match observed NVTX calls to CUDA API intervals within one profiled worker.

CUDA API host durations include driver work and waiting; they are not kernel
execution durations. Profiler and perf clocks are not equated. Only request /
role / ordinal identities join raw host measurements to profiler observations.
"""
import argparse
import copy
from bisect import bisect_left,bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity,summarize
from experiments.diagnose_activation_semantic_repeats import keyed_rows
from experiments.analyze_activation_host_diagnostic_v2 import validate_host_bracket
from gsdeploy.forecasting.native_training import validate_capture


def detail_key(text):
    if not text.startswith('GS_DETAIL:'):return None
    parts=text[len('GS_DETAIL:'):].rsplit(':',2)
    if len(parts)!=3 or not parts[2].isdigit():raise ValueError('Invalid GS_DETAIL marker')
    return parts[0],parts[1],int(parts[2])


class IntervalIndex:
    def __init__(self,rows):
        self.rows=sorted(rows,key=lambda r:(r['start_ns'],r['end_ns']))
        self.starts=[r['start_ns'] for r in self.rows];self.prefix_ends=[];end=-1
        for row in self.rows:
            if row['start_ns']>row['end_ns']:raise ValueError('Negative API interval')
            end=max(end,row['end_ns']);self.prefix_ends.append(end)
        self.by_end=sorted(rows,key=lambda r:(r['end_ns'],r['start_ns']));self.ends=[r['end_ns'] for r in self.by_end]
    def overlaps(self,start,end):
        left=bisect_right(self.prefix_ends,start);right=bisect_left(self.starts,end)
        return [r for r in self.rows[left:right] if r['end_ns']>start and r['start_ns']<end]
    def preceding(self,start):
        index=bisect_right(self.ends,start)-1
        return self.by_end[index] if index>=0 else None


def union_ns(rows,start,end):
    last=start;total=0
    for row in sorted(rows,key=lambda r:(r['start_ns'],r['end_ns'])):
        a,b=max(start,row['start_ns']),min(end,row['end_ns'])
        if b<=a:continue
        total+=max(0,b-max(a,last));last=max(last,b)
    return total


def analyze(base):
    design=json.loads((base/'design.json').read_text());directory=Path(design['capture']);alias=design['profiled_alias']
    bundle=load_bundle(directory);reference=load_bundle(Path(design['unprofiled_capture']))
    validate_capture(bundle,allowed_splits=('development_cupti_host_diagnostic',))
    validate_capture(reference,allowed_splits=('development_host_diagnostic',))
    if reference['source']!=design['unprofiled_source'] or bundle['plan']['jobs']!=reference['plan']['jobs']:
        raise ValueError('Exact matched unprofiled input order/source required')
    context=copy.deepcopy(bundle['context']);ref_context=reference['context']
    if (context.pop('observer_diagnostic')!='five_model_one_worker_cupti_host'
            or context.pop('profiler')!=design['nsys']):raise ValueError('Profiler policy mismatch')
    expected_command=[design['nsys']['tool'],*design['nsys']['flags'],*ref_context['worker_bindings'][alias]['command']]
    if context['worker_bindings'][alias]['command']!=expected_command:raise ValueError('Profiler command mismatch')
    context['worker_bindings'][alias]['command']=copy.deepcopy(ref_context['worker_bindings'][alias]['command'])
    context['observer_diagnostic']=ref_context['observer_diagnostic'];context['source_content']=ref_context['source_content']
    if context!=ref_context:raise ValueError('Matched runtime differs beyond declared profiler')
    for owner,ready in bundle['raw']['ready'].items():
        old=reference['raw']['ready'][owner]
        for key in ('runtime_scope','passive_gc_policy','passive_host_policy','spec_sha256','torch','clock'):
            if ready.get(key)!=old.get(key):raise ValueError('Actual worker policy differs: '+owner+':'+key)
    if any(sample.get('foreign_pids') for sample in bundle['raw']['board_samples']):raise ValueError('Foreign GPU activity')
    export=json.loads((base/'export-receipt.json').read_text());sql=Path(export['sqlite']['path']);report=Path(design['expected_nsys_report'])
    if (hashlib.sha256(sql.read_bytes()).hexdigest()!=export['sqlite']['sha256']
            or hashlib.sha256(report.read_bytes()).hexdigest()!=export['report']['sha256']):
        raise ValueError('Profiler export provenance changed')
    connection=sqlite3.connect('file:'+str(sql)+'?mode=ro',uri=True)
    strings=dict(connection.execute('select id,value from StringIds'));ranges={};api_by_thread=defaultdict(list)
    tables={name for name, in connection.execute("select name from sqlite_master where type='table'")}
    for start,end,text,textid,tid in connection.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null'):
        key=detail_key(text or strings.get(textid,''))
        if key is None:continue
        if key in ranges or start>=end:raise ValueError('Nonunique or invalid semantic NVTX range')
        ranges[key]=dict(start_ns=start,end_ns=end,global_tid=tid)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:continue
        columns={r[1] for r in connection.execute('pragma table_info('+table+')')}
        required={'start','end','globalTid','nameId','correlationId'}
        if not required<=columns:raise ValueError('Unsupported CUDA API schema')
        return_column='returnValue' if 'returnValue' in columns else 'NULL'
        for rowid,start,end,tid,name,corr,retval in connection.execute('select rowid,start,end,globalTid,nameId,correlationId,'+return_column+' from '+table):
            if tid is None:continue
            api_by_thread[tid].append(dict(table=table,rowid=rowid,start_ns=start,end_ns=end,
                api=strings.get(name,str(name)),correlation_id=corr,return_value=retval))
    connection.close();indices={tid:IntervalIndex(rows) for tid,rows in api_by_thread.items()}
    raw=bundle['raw'];lookup=keyed_rows(reference);deviations=defaultdict(list);changed=[];observations=[];expected=set()
    for key,row in keyed_rows(bundle).items():
        old=lookup[key];family=raw['resident_aliases'][row['model']]
        if output_identity(old['worker_response'])!=output_identity(row['worker_response']):changed.append(key)
        a=old['completion_time_ns']-old['scheduled_time_ns'];b=row['completion_time_ns']-row['scheduled_time_ns']
        deviations[family+'/'+row['task']].append(abs(b/a-1))
        if row['model']!=alias:continue
        oldevents={(e['role'],e['ordinal']):e for e in old['worker_response']['semantic_phase_events']}
        for event in row['worker_response']['semantic_phase_events']:
            marker=(row['request_id'],event['role'],event['ordinal']);expected.add(marker)
            if marker not in ranges:raise ValueError('Missing profiled semantic NVTX range: '+repr(marker))
            nvtx=ranges[marker];start,end=nvtx['start_ns'],nvtx['end_ns'];index=indices.get(nvtx['global_tid'])
            if index is None:raise ValueError('NVTX thread has no CUDA API records')
            validate_host_bracket(event);bracket=event['passive_host_bracket']
            apis=index.overlaps(start,end);coverage=union_ns(apis,start,end)
            oldevent=oldevents.get((event['role'],event['ordinal']))
            observations.append(dict(request_id=row['request_id'],pool_id=row['pool_id'],component=event['component'],ordinal=event['ordinal'],
                nvtx=nvtx,nvtx_elapsed_ms=(end-start)/1e6,cuda_api_union_ms=coverage/1e6,
                nvtx_time_outside_observed_cuda_apis_ms=((end-start)-coverage)/1e6,
                cuda_apis=apis,preceding_cuda_api=index.preceding(start),
                semantic_host_ms=(event['host_end_ns']-event['host_start_ns'])/1e6,
                unprofiled_semantic_host_ms=(oldevent['host_end_ns']-oldevent['host_start_ns'])/1e6 if oldevent else None,
                observed_cpu_bracket=bracket,
                scope='NVTX starts after the semantic observer start Event.record and ends after end Event.record. The preceding API is reported, not assumed to be an event-record call. Profiler time is not mapped onto perf time.'))
    formal={key for key in ranges if key[0].startswith('training:')}
    if formal!=expected:raise ValueError('Profiler formal semantic roster differs from target raw requests')
    metrics={k:summarize(v) for k,v in sorted(deviations.items())}
    return dict(schema='gsdeploy.cupti_host_probe_diagnostic.v1',source=bundle['source'],reference=reference['source'],
        export=export,profiled_alias=alias,output_changes=changed,paired_e2e_deviations=metrics,observations=observations,
        profiler_timing_perturbed=True,physical_predictor_qualified=False,formal_utility_qualified=False,
        limitations='Single development profiler capture. CUDA API host durations include blocking/driver costs. NVTX and API intervals share profiler clock; raw host/CPU brackets are joined only by identities. Nested runtime/driver intervals use unions. No latency subtraction or independent accuracy qualification.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.base.resolve())
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(profiled_alias=result['profiled_alias'],calls=len(result['observations']),output_changes=len(result['output_changes']),physical_predictor_qualified=False)))
