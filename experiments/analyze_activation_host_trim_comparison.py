"""Matched runtime effects of post-prime heap trim; no predictor admission."""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity,summarize
from experiments.diagnose_activation_semantic_repeats import keyed_rows
from experiments.analyze_activation_host_diagnostic_v2 import validate_trace,validate_host_bracket,overlap_ms
from experiments.activation_completion_bands import completion_band_layout
from gsdeploy.forecasting.native_training import validate_capture,digest


def analyze(base):
    names=['control-round1','trim-round1','trim-round2','control-round2'];bundles={n:load_bundle(base/n) for n in names}
    contexts=[];actual=[];keyed={};receipts=[];band_groups=defaultdict(list);issues=[];episodes=[];cpu_groups=defaultdict(list)
    for name,bundle in bundles.items():
        validate_capture(bundle,allowed_splits=('development_host_trim_comparison',));raw=bundle['raw'];plan=bundle['plan']
        mode='on' if name.startswith('trim-') else 'off';ctx=copy.deepcopy(bundle['context'])
        if (plan['host_trim_mode']!=mode or plan.get('not_for_training') is not True
                or plan['gc_trace_mode']!='on' or plan['host_trace_mode']!='on'
                or any(s.get('foreign_pids') for s in raw['board_samples'])
                or digest(plan['jobs'])!=plan['matched_job_content_sha256']):raise ValueError('Invalid trim comparison capture')
        if ctx.pop('observer_diagnostic')!='five_model_postprime_trim_'+mode:raise ValueError('Wrong comparison arm')
        if ctx['postprime_host_trim']!=dict(mode=mode,api='glibc malloc_trim(0)',timing='after full declared priming before worker READY',gc_policy_changes=False,extra_cuda_sync=False):raise ValueError('Trim policy drift')
        ctx['postprime_host_trim']['mode']='MATCHED_ARM'
        for binding in ctx['worker_bindings'].values():
            if binding['env'].pop('GS_POSTPRIME_HOST_TRIM')!=mode:raise ValueError('Worker trim mode mismatch')
        contexts.append(ctx);ready_policy={}
        for alias,ready in raw['ready'].items():
            policy=copy.deepcopy(ready['postprime_trim_policy'])
            if policy.pop('mode')!=mode or policy['forced_gc'] or policy['gc_thresholds_changed'] or policy['extra_cuda_sync']:raise ValueError('Actual trim policy mismatch')
            ready_policy[alias]={k:ready[k] for k in ('runtime_scope','passive_gc_policy','passive_host_policy','spec_sha256','torch')}
            ready_policy[alias]['trim_policy']=policy
            primes=[r for r in raw['priming_request_records'] if r['model']==alias]
            found=[r for r in primes if 'postprime_trim_receipt' in r['previous_handler_next_request_ready']]
            if len(primes)!=policy['priming_requests'] or len(found)!=1 or found[0]['request_id']!=primes[-1]['request_id']:
                raise ValueError('Exactly one trim receipt after full priming required')
            row=found[0];boundary=row['previous_handler_next_request_ready'];t=boundary['postprime_trim_receipt']
            if (t['mode']!=mode or (mode=='on' and t['allocator_return'] not in (0,1)) or (mode=='off' and t['allocator_return'] is not None)
                    or t['gc_policy_unchanged'] is not True or t['extra_cuda_sync'] or t['live_allocations_released']
                    or not row['worker_response']['end_perf_ns']<=t['procedure_start_ns']<=t['allocator_call_start_ns']<=t['allocator_call_end_ns']<=t['procedure_end_ns']<=boundary['perf_ns']):
                raise ValueError('Invalid or unaccounted trim lifecycle')
            receipts.append(dict(capture=name,alias=alias,receipt=t,
                rss_reduction_mib=(t['host_memory_before']['VmRSS_kib']-t['host_memory_after']['VmRSS_kib'])/1024,
                procedure_ms=(t['procedure_end_ns']-t['procedure_start_ns'])/1e6))
        actual.append(ready_policy);keyed[name]=keyed_rows(bundle)
        for row in raw['priming_request_records']+raw['request_records']:
            response=row['worker_response'];ready=raw['ready'][row['model']];family=raw['resident_aliases'][row['model']]
            if response.get('success') is not True:issues.append(name+':'+row['request_id']+':OUTPUT_FAILURE')
            validate_trace(response['passive_gc_trace'],response['start_perf_ns'],response['end_perf_ns'],ready['passive_gc_policy'])
            boundary=row['previous_handler_next_request_ready'];validate_trace(boundary['passive_gc_lifecycle_trace'],response['start_perf_ns'],boundary['perf_ns'],ready['passive_gc_policy'])
            for event in response['semantic_phase_events']:validate_host_bracket(event)
            if family=='generalist_2':
                m=response.get('memory_optimizations',{}).get('exllama_reconstruct_runtime',{})
                if m.get('matmul_recons_thd')!=1 or m.get('configured_after_model_load') is not True or m.get('quantized_weights_changed') is not False:raise ValueError('G2 kernel configuration changed')
            if row['request_id'].startswith('priming:'):continue
            if 'postprime_trim_receipt' in boundary:raise ValueError('Undeclared formal heap trim')
            for event in response['semantic_phase_events']:
                b=event['passive_host_bracket'];cpu_groups[mode,family,row['task'],event['component']].append(dict(capture=name,request_id=row['request_id'],pool_id=row['pool_id'],ordinal=event['ordinal'],
                    host_ms=(event['host_end_ns']-event['host_start_ns'])/1e6,
                    thread_cpu_ms=(b['cpu_end_ns']-b['cpu_start_ns'])/1e6,
                    system_cpu_ms=1000*(b['usage_after']['system_cpu_seconds']-b['usage_before']['system_cpu_seconds'])))
            for span in completion_band_layout(row,family):
                if span['kind']!='gpu':continue
                band_groups[mode,family,row['task'],span['native_band_role'],row['pool_id']].append(dict(capture=name,request_id=row['request_id'],
                    duration_ms=(span['end_ns']-span['start_ns'])/1e6,
                    gc_overlap_ms=overlap_ms(response['passive_gc_trace']['events'],span['start_ns'],span['end_ns'])))
        episodes.append(dict(capture=name,observed_episode_seconds=(raw['observation_end_ns']-raw['observation_start_ns'])/1e9,
            startup_through_all_priming_ready_seconds=(max(r['previous_handler_next_request_ready']['perf_ns'] for r in raw['priming_request_records'])-raw['startup_begin_time_ns'])/1e9))
    if any(c!=contexts[0] for c in contexts[1:]) or any(p!=actual[0] for p in actual[1:]):raise ValueError('Matched contexts differ beyond heap policy')
    if any(set(rows)!=set(keyed[names[0]]) for rows in keyed.values()):raise ValueError('Complete common cohort required')
    changes=[];ratios=defaultdict(list)
    for name in names[1:]:
        for key,row in keyed[name].items():
            if output_identity(row['worker_response'])!=output_identity(keyed[names[0]][key]['worker_response']):changes.append(dict(capture=name,key=key))
    for i in (1,2):
        left,right=f'control-round{i}',f'trim-round{i}'
        if bundles[left]['plan']['jobs']!=bundles[right]['plan']['jobs']:raise ValueError('Within-pair order differs')
        for key,old in keyed[left].items():
            row=keyed[right][key];path=bundles[right]['raw']['resident_aliases'][row['model']]+'/'+row['task']
            a=old['completion_time_ns']-old['scheduled_time_ns'];b=row['completion_time_ns']-row['scheduled_time_ns']
            ratios[path].append(dict(round=i,key=key,control_ms=a/1e6,trim_ms=b/1e6,trim_over_control=b/a))
    bands=[]
    for key,rows in sorted(band_groups.items()):
        if len(rows)!=2:raise ValueError('Each input band must repeat twice per arm')
        values=[r['duration_ms'] for r in rows];lo,hi=min(values),max(values)
        bands.append(dict(mode=key[0],family=key[1],task=key[2],band=key[3],pool_id=key[4],observations=rows,raw_constant_minimax=(hi-lo)/(hi+lo)))
    cpu=[dict(mode=k[0],family=k[1],task=k[2],component=k[3],host_ms=summarize([r['host_ms'] for r in rows]),
        system_cpu_ms=summarize([r['system_cpu_ms'] for r in rows]),observations=rows) for k,rows in sorted(cpu_groups.items())]
    return dict(schema='gsdeploy.host_trim_runtime_diagnostic.v1',sources={n:b['source'] for n,b in bundles.items()},issues=issues,output_changes=changes,
        trim_receipts=receipts,episodes=episodes,paired_latency={p:dict(ratio=summarize([r['trim_over_control'] for r in rows]),observations=rows) for p,rows in sorted(ratios.items())},
        bands=bands,cpu_summary=cpu,physical_predictor_qualified=False,formal_utility_qualified=False,
        limits='Runtime intervention comparison, not an observer-overhead gate. All durations retained. Raw constant minimax lacks root-order adjustment and is not qualification. RSS release and paired changes do not identify a kernel mechanism. Fresh exact-context observer/baseline/training/independent qualification required.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.base.resolve())
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(issues=result['issues'],output_changes=len(result['output_changes']),
        ratios={k:v['ratio'] for k,v in result['paired_latency'].items()},raw_minimax_over_point2={mode:sum(b['mode']==mode and b['raw_constant_minimax']>.2 for b in result['bands']) for mode in ('off','on')},physical_predictor_qualified=False)))
