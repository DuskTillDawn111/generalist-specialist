"""Matched passive GC observations; correlation only, never predictor qualification."""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity,summarize
from experiments.diagnose_activation_semantic_repeats import keyed_rows
from experiments.activation_completion_bands import completion_band_layout
from gsdeploy.forecasting.native_training import validate_capture,digest


def overlap_ms(events,start,end):
    # Union avoids double counting even if a future collector records nested events.
    intervals=sorted((max(start,e['start_ns']),min(end,e['end_ns'])) for e in events
                     if e['start_ns']<end and e['end_ns']>start)
    total=0;last=start
    for a,b in intervals:
        total+=max(0,b-max(a,last));last=max(last,b)
    return total/1e6


def validate_trace(trace,start,end,policy):
    if (trace.get('schema')!='gsdeploy.passive_gc_trace.v1' or trace.get('enabled') is not True
            or trace.get('clock')!='host_perf_counter_ns' or trace.get('window_start_ns')!=start
            or trace.get('window_end_ns')!=end or trace.get('coverage_complete') is not True
            or trace.get('errors') or trace.get('gc_policy_unchanged') is not True
            or trace.get('gc_enabled')!=policy['gc_enabled'] or trace.get('gc_thresholds')!=policy['thresholds']):
        raise ValueError('Incomplete or inconsistent passive GC trace')
    previous=None
    for event in trace['events']:
        a,b=event['start_ns'],event['end_ns']
        if (type(a) is not int or type(b) is not int or a>=b or a>=end or b<=start
                or event['generation'] not in (0,1,2) or previous is not None and a<previous):
            raise ValueError('Invalid passive GC interval')
        previous=b


def analyze(base):
    design=json.loads((base/'design.json').read_text());criteria=design['criteria']
    if criteria!=dict(maximum_mean_absolute_e2e_deviation=.1,maximum_p95_absolute_e2e_deviation=.25,
                       all_outputs_unchanged=True,complete_gc_trace=True):raise ValueError('Original gates changed')
    names=['control-round1','trace-round1','trace-round2','control-round2']
    bundles={n:load_bundle(base/n) for n in names};keyed={};contexts=[];policies=[];settings=[]
    issues=[];observations=[]
    expected={f+'#0':f for f in ('generalist_2','text','asr','image','ocr')}
    for name,bundle in bundles.items():
        validate_capture(bundle,allowed_splits=('development_gc_diagnostic',))
        raw,plan,ctx=bundle['raw'],bundle['plan'],copy.deepcopy(bundle['context'])
        mode='on' if name.startswith('trace-') else 'off'
        if (raw['kind']!='measured' or raw['resident_aliases']!=expected
                or any(s.get('foreign_pids') for s in raw['board_samples'])
                or plan.get('not_for_training') is not True or plan.get('gc_trace_mode')!=mode
                or any(len(j['requests'])!=1 or j['requests'][0]['offset_ms']!=0 for j in plan['jobs'])
                or digest(plan['jobs'])!=plan['matched_job_content_sha256']):raise ValueError('Clean serial diagnostic capture required')
        if ctx.pop('observer_diagnostic')!='five_model_passive_gc_'+mode:raise ValueError('Wrong observer arm')
        if ctx.pop('passive_gc_policy')!=dict(mode=mode,forced_collection=False,threshold_changes=False,gc_disabled=False):
            raise ValueError('GC policy intervention not allowed')
        for binding in ctx['worker_bindings'].values():
            if binding['env'].pop('GS_PASSIVE_GC_TRACE')!=mode or binding['env']['GS_SCOPED_OBSERVER']!='semantic':
                raise ValueError('Worker instrumentation mismatch')
        contexts.append(ctx);actual_settings={};actual_policy={}
        for alias,family in expected.items():
            ready=raw['ready'][alias];e=ready['runtime_scope'];strict=family=='generalist_2'
            if (e['scope']!='g2_strict' or e['slot']!=family or e['strict_requested'] is not strict
                    or e['deterministic_algorithms'] is not strict
                    or e['cublas_workspace_config']!=(':4096:8' if strict else None)
                    or strict and (e['deterministic_warn_only'] or e['cudnn_benchmark'] or not e['cudnn_deterministic'])):
                raise ValueError('Runtime settings differ from scoped policy')
            policy=copy.deepcopy(ready['passive_gc_policy'])
            if (policy.pop('mode')!=mode or policy['forced_collections'] is not False
                    or policy['gc_disabled'] is not False or policy['thresholds_changed'] is not False):
                raise ValueError('Passive observation changed GC policy')
            actual_policy[alias]=policy;actual_settings[alias]=e
        policies.append(actual_policy);settings.append(actual_settings)
        keyed[name]=keyed_rows(bundle)
        for phase,rows in [('priming',raw['priming_request_records']),('diagnostic',raw['request_records'])]:
            for row in rows:
                response=row['worker_response'];family=expected[row['model']];label=name+':'+row['request_id']
                if response.get('success') is not True:issues.append(label+':OUTPUT_FAILURE')
                if family=='generalist_2':
                    m=response.get('memory_optimizations',{}).get('exllama_reconstruct_runtime',{})
                    if (m.get('matmul_recons_thd')!=1 or m.get('configured_after_model_load') is not True
                            or m.get('quantized_weights_changed') is not False or m.get('quantized_layers',0)<=0):
                        raise ValueError('G2 reconstruction path not verified')
                completion_band_layout(row,family)
                if mode=='off':
                    if 'passive_gc_trace' in response:raise ValueError('Control arm unexpectedly traced')
                    continue
                start,end=response['start_perf_ns'],response['end_perf_ns'];trace=response['passive_gc_trace']
                validate_trace(trace,start,end,actual_policy[row['model']])
                boundary=row['previous_handler_next_request_ready'];life=boundary['passive_gc_lifecycle_trace']
                validate_trace(life,start,boundary['perf_ns'],actual_policy[row['model']])
                sem=[]
                for event in response['semantic_phase_events']:
                    a,b=event['host_start_ns'],event['host_end_ns'];device=event['device_elapsed_ms']
                    if not start<=a<b<=end or not event['device_event_complete'] or device is None or device<0:
                        raise ValueError('Invalid semantic interval')
                    sem.append(dict(component=event['component'],role=event['role'],ordinal=event['ordinal'],
                        host_start_ns=a,host_end_ns=b,host_ms=(b-a)/1e6,device_interval_ms=device,
                        host_minus_device_interval_ms=(b-a)/1e6-device,gc_overlap_ms=overlap_ms(trace['events'],a,b)))
                before=trace['usage_at_first_phase_hook'];after=trace['usage_at_result_hook']
                observations.append(dict(capture=name,phase=phase,request_id=row['request_id'],
                    job=row.get('profile_job_id'),path=family+'/'+row['task'],pool_id=row.get('pool_id'),
                    handler_ms=(end-start)/1e6,gc_overlap_ms=overlap_ms(trace['events'],start,end),
                    lifecycle_gc_overlap_ms=overlap_ms(life['events'],start,boundary['perf_ns']),
                    gc_events=trace['events'],usage_delta={k:after[k]-before[k] for k in before},semantic_events=sem))
    if any(c!=contexts[0] for c in contexts[1:]) or any(p!=policies[0] for p in policies[1:]) or any(s!=settings[0] for s in settings[1:]):
        raise ValueError('Matched captures differ beyond passive instrumentation')
    if any(set(k)!=set(keyed[names[0]]) for k in keyed.values()):raise ValueError('Cross-round cohort mismatch')
    metrics=defaultdict(list);changes=[]
    for i in (1,2):
        left,right=f'control-round{i}',f'trace-round{i}'
        if bundles[left]['plan']['jobs']!=bundles[right]['plan']['jobs']:raise ValueError('Matched order differs')
        for key,old in keyed[left].items():
            new=keyed[right][key];path=expected[old['model']]+'/'+old['task']
            a=old['completion_time_ns']-old['scheduled_time_ns'];b=new['completion_time_ns']-new['scheduled_time_ns']
            metrics[path].append(abs(b-a)/a)
    for name in names[1:]:
        for key,row in keyed[name].items():
            if output_identity(row['worker_response'])!=output_identity(keyed[names[0]][key]['worker_response']):
                changes.append(dict(capture=name,key=key))
    # Repeated target inputs within each capture must also agree.
    for name,rows in keyed.items():
        seen={}
        for row in rows.values():
            key=(row['model'],row['task'],row['pool_id']);value=output_identity(row['worker_response'])
            if key in seen and seen[key]!=value:changes.append(dict(capture=name,within_capture_input=key))
            seen[key]=value
    if changes:issues.append('OUTPUT_CHANGED')
    metrics={p:summarize(v) for p,v in sorted(metrics.items())}
    for path,m in metrics.items():
        if m['mean']>.1 or m['p95']>.25:issues.append(path+':OBSERVER_OR_REPEAT_VARIATION_TOO_LARGE')
    groups=defaultdict(list)
    for obs in observations:
        if obs['phase']=='diagnostic':groups[obs['path']].append(obs)
    summaries={p:dict(n=len(rows),requests_with_gc=sum(r['gc_overlap_ms']>0 for r in rows),
        gc_ms=summarize([r['gc_overlap_ms'] for r in rows]),handler_ms=summarize([r['handler_ms'] for r in rows])) for p,rows in groups.items()}
    return dict(schema='gsdeploy.passive_gc_diagnostic.v1',criteria=criteria,
        sources={n:b['source'] for n,b in bundles.items()},issues=issues,output_changes=changes,
        path_metrics=metrics,trace_summary=summaries,observations=observations,
        observer_repeat_check_passed=not issues,physical_predictor_qualified=False,formal_utility_qualified=False,
        causal_gc_removal_benefit_established=False,
        limitations='ABBA natural variation plus instrumentation; GC overlap is association. CUDA event intervals include host supply/scheduling gaps, not pure kernels. Priming retained separately. No observations removed.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=analyze(a.base)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','observations')},ensure_ascii=False))


if __name__=='__main__':main()
