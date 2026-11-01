"""CPU-only descriptive phase/supply evidence; never a fitted scheduler.

Nsight kernel lifetime, runtime API wall intervals and natural Python phase wall
intervals are retained as separate observation domains. They are not summed into
SM execution demand or converted into a request slowdown coefficient.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time

from .native_collect import receipt,save
from .dag_host_release_v7 import checked,verify_model
from .small_training_domain_bridge_review_v4 import request_kernels
from .paired_training_work_review_v4 import actual_work
from .trace_export import linux_pid


def load_capture(ref,domain):
    doc=checked(ref)
    from .kernel_domain_bridge_review_v3 import load_report
    if doc.get('schema')=='gsdeploy.startup_training_collection_strict_review.v4':
        from .small_training_domain_bridge_review_v4 import load_report
    strict,plan,raw=load_report(ref['path'],domain)
    if (raw.get('kind')!='measured' or raw.get('status')!='completed'
            or any(r.get('split')!='training' or r.get('status')!='completed' for r in raw['request_records'])):
        raise ValueError('Complete actual training capture required')
    return strict,plan,raw


def percentile(values,q):
    if not values:return None
    values=sorted(values);p=(len(values)-1)*q;lo=math.floor(p);hi=math.ceil(p)
    return values[lo]+(values[hi]-values[lo])*(p-lo)


def stats(values):
    values=list(values)
    return dict(count=len(values),median=statistics.median(values) if values else None,
        p95=percentile(values,.95),p99=percentile(values,.99),max=max(values) if values else None)


def union_ns(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('Reversed observed interval')
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def peer_exposure(a,b,peers):
    """Disjoint joint peer Python wall-phase exposure, not GPU-busy exposure."""
    if b<a:raise ValueError('Reversed phase')
    cuts={a,b}
    for lo,hi,_,_ in peers:
        if lo<b and hi>a:cuts.update((max(a,lo),min(b,hi)))
    result=Counter();cuts=sorted(cuts)
    for lo,hi in zip(cuts,cuts[1:]):
        active=sorted((owner,phase) for start,end,owner,phase in peers if start<hi and end>lo)
        label='|'.join(owner+':'+phase for owner,phase in active) or 'NO_PEER_PYTHON_PHASE'
        result[label]+=hi-lo
    if sum(result.values())!=b-a:raise ValueError('Incomplete peer wall-phase exposure')
    return dict(result)


def natural_rows(label,loaded,solo):
    strict,plan,raw=loaded;byjob=defaultdict(list)
    for row in raw['request_records']:byjob[row['profile_job_id']].append(row)
    output=[]
    for rows in byjob.values():
        for row in rows:
            key=row['model'],row['task'],row['pool_id']
            if key not in solo:continue
            worker=row['worker_response'];baselines=solo[key]
            if any(actual_work(w)!=actual_work(worker) for w in baselines):raise ValueError('Different actual natural work')
            phases=worker.get('semantic_forward_observations',{}).get('records',[])
            peers=[]
            for other in rows:
                if other is row:continue
                for p in other['worker_response'].get('semantic_forward_observations',{}).get('records',[]):
                    peers.append((p['start_ns'],p['end_ns'],other['model'],p['semantic_phase']))
            phase_rows=[]
            for p in phases:
                base=[next(q['end_ns']-q['start_ns'] for q in w['semantic_forward_observations']['records'] if q['id']==p['id']) for w in baselines]
                duration=p['end_ns']-p['start_ns']
                phase_rows.append(dict(id=p['id'],phase=p['semantic_phase'],wall_ns=duration,
                    ratio_to_two_solo_median=duration/statistics.median(base),
                    joint_peer_python_wall_exposure_ns=peer_exposure(p['start_ns'],p['end_ns'],peers)))
            handler=worker['end_perf_ns']-worker['start_perf_ns']
            base_handler=statistics.median(w['end_perf_ns']-w['start_perf_ns'] for w in baselines)
            output.append(dict(capture=label,job_id=row['profile_job_id'],alias=key[0],task=key[1],pool_id=key[2],
                handler_ns=handler,handler_ratio_to_two_solo_median=handler/base_handler,phases=phase_rows,
                natural_stages_ns={s['id']:s['end_ns']-s['start_ns'] for s in worker.get('observed_stages',[])},
                python_semantic_wall_is_gpu_busy=False))
    return output


def light_groups(label,loaded):
    strict,plan,raw=loaded;output=[]
    for alias in ('text#0','text#1','generalist_2#0'):
        if alias not in strict['owners']:continue
        owner=strict['owners'][alias];parsed=checked(owner['parsed']);sql=owner['sqlite']['path']
        with sqlite3.connect(f'file:{Path(sql).resolve()}?mode=ro',uri=True) as conn:
            conn.row_factory=sqlite3.Row
            strings={r['id']:r['value'] for r in conn.execute('select id,value from StringIds')}
            runtime=[dict(r) for r in conn.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
        for row in raw['request_records']:
            if row['model']!=alias or row['task']!='text':continue
            items=request_kernels(parsed,sql,row);groups=defaultdict(list)
            for item in items:groups[item['group']].append(item)
            phases=[p for p in parsed['phases'] if p['request_id']==row['request_id']]
            if not phases:raise ValueError('Missing actual request ranges')
            pid=phases[0]['pid'];start=min(p['nsys_start_ns'] for p in phases);end=max(p['nsys_end_ns'] for p in phases)
            forwards=[p for p in phases if p['segment_id'] in {f['id'] for f in row['worker_response']['semantic_forward_observations']['records']}]
            apis=defaultdict(list);launches={}
            for api in runtime:
                if linux_pid(api['globalTid'])!=pid or api['start']<start or api['end']>end:continue
                matching=[p for p in forwards if p['nsys_start_ns']<=api['start'] and api['end']<=p['nsys_end_ns']]
                if len(matching)>1:raise ValueError('Ambiguous semantic API range')
                group=matching[0]['segment_id'] if matching else '__outside_language_forwards__'
                apis[group].append(dict(api,name=strings.get(api['nameId'],'UNKNOWN_API')))
                launches[api['correlationId']]=api
            for phase,kernels in groups.items():
                duration=[i['kernel']['nsys_end_ns']-i['kernel']['nsys_start_ns'] for i in kernels]
                phaseapis=apis[phase];sync=[a for a in phaseapis if 'Synchronize' in a['name']]
                issue=[i['launch_ns'] for i in kernels]
                output.append(dict(capture=label,job_id=row['profile_job_id'],alias=alias,task='text',pool_id=row['pool_id'],phase=phase,
                    signatures=[i['signature'] for i in kernels],lifetimes=duration,
                    kernel_lifetime_ns=stats(duration),kernel_lifetime_sum_ns=sum(duration),
                    kernel_wall_union_ns=union_ns((i['kernel']['nsys_start_ns'],i['kernel']['nsys_end_ns']) for i in kernels),
                    kernel_span_ns=max(i['kernel']['nsys_end_ns'] for i in kernels)-min(i['kernel']['nsys_start_ns'] for i in kernels),
                    launch_entry_spacing_ns=stats(b-a for a,b in zip(issue,issue[1:])),
                    launch_entry_to_kernel_start_ns=stats(i['kernel']['nsys_start_ns']-launches[i['kernel']['correlation_id']]['start'] for i in kernels),
                    launch_exit_to_kernel_start_signed_ns=stats(i['kernel']['nsys_start_ns']-launches[i['kernel']['correlation_id']]['end'] for i in kernels),
                    runtime_API_count=len(phaseapis),runtime_API_wall_union_ns=union_ns((a['start'],a['end']) for a in phaseapis),
                    synchronize_API_count=len(sync),synchronize_API_wall_union_ns=union_ns((a['start'],a['end']) for a in sync),
                    synchronize_API_names=dict(Counter(a['name'] for a in sync)),
                    kernel_lifetime_is_SM_service=False,API_wall_is_CPU_work=False))
        print(json.dumps(dict(stage='light_owner_complete',capture=label,alias=alias)),flush=True)
    return output


def analyze(parameters_path,output):
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True);start=time.monotonic()
    parameters=checked(receipt(parameters_path));host=checked(parameters['host_model']);verify_model(host)
    expected={'host_model','repeat_review','solo_light_reviews','solo_natural_reviews','shared_light_reviews','shared_natural_reviews'}
    if set(parameters)!=expected or any(len(parameters[k])!=2 for k in expected if k.endswith('_reviews')):
        raise ValueError('Explicit two-solo/two-shared original source parameter sets required')
    repeated=checked(parameters['repeat_review'])
    if repeated['status']!='EXACT_WORK_AND_KERNEL_ORDER_MATCH':raise ValueError('Exact repeated work/order evidence required')
    captures={};sources=[parameters['host_model'],parameters['repeat_review'],receipt(parameters_path),receipt(__file__)]
    for kind in ('solo','shared'):
        for domain,short in (('light-kernel','light'),('natural','natural')):
            refs=parameters[kind+'_'+short+'_reviews'];raw_ids=[]
            for i,ref in enumerate(refs):
                loaded=load_capture(ref,domain);captures[kind,domain,i]=loaded;raw_ids.append(loaded[0]['raw']['sha256'])
                sources.extend((ref,loaded[0]['raw'],loaded[0]['plan'],loaded[0]['context']))
                for owner in loaded[0]['owners'].values():
                    for field in ('parsed','sqlite'):
                        if field in owner:sources.append(owner[field])
            if len(set(raw_ids))!=2:raise ValueError('Two independent actual captures required')
    if {c['raw']['sha256'] for c in host['captures']}!={captures['solo','natural',i][0]['raw']['sha256'] for i in range(2)}:
        raise ValueError('Host v7 must retain exactly the original two isolated capture sources')
    solo_natural=defaultdict(list)
    for i in range(2):
        for row in captures['solo','natural',i][2]['request_records']:
            if row['task']=='text':solo_natural[row['model'],row['task'],row['pool_id']].append(row['worker_response'])
    natural=[]
    for i in range(2):natural.extend(natural_rows('shared'+str(i+1),captures['shared','natural',i],solo_natural))
    solo={}
    for i in range(2):
        for g in light_groups('solo'+str(i+1),captures['solo','light-kernel',i]):
            key=g['alias'],g['task'],g['pool_id'],g['phase'];solo.setdefault(key,[]).append(g)
    groups=[]
    for i in range(2):
        for g in light_groups('shared'+str(i+1),captures['shared','light-kernel',i]):
            key=g['alias'],g['task'],g['pool_id'],g['phase'];base=solo[key]
            if len(base)!=2 or any(b['signatures']!=g['signatures'] for b in base):raise ValueError('Solo/shared exact ordered static launch signature mismatch')
            baseline=[statistics.median(values) for values in zip(*(b['lifetimes'] for b in base))]
            excess=sorted((max(a-b,0) for a,b in zip(g['lifetimes'],baseline)),reverse=True);n=max(1,math.ceil(len(excess)*.01))
            g.update(exact_ordered_solo_signature_match=True,solo_capture_count=2,
                lifetime_sum_ratio_to_two_solo_ordinal_median=sum(g['lifetimes'])/sum(baseline),
                ordinal_lifetime_ratio=stats(a/b for a,b in zip(g['lifetimes'],baseline)),
                largest_one_percent_positive_lifetime_excess_fraction=sum(excess[:n])/sum(excess) if sum(excess) else 0.,
                descriptive_positive_lifetime_excess_ns=sum(excess),
                synchronize_API_wall_ratio_to_two_solo_median=(g['synchronize_API_wall_union_ns']/statistics.median(b['synchronize_API_wall_union_ns'] for b in base) if statistics.median(b['synchronize_API_wall_union_ns'] for b in base)>0 else None))
            del g['lifetimes'];del g['signatures'];groups.append(g)
    summaries=[]
    for capture in ('shared1','shared2'):
        for alias in ('text#0','text#1','generalist_2#0'):
            for phase in ('prefill','decode','outside'):
                selected=[g for g in groups if g['capture']==capture and g['alias']==alias and ('prefill' if g['phase'].startswith('prefill') else 'decode' if g['phase'].startswith('decode') else 'outside')==phase]
                summaries.append(dict(capture=capture,alias=alias,phase=phase,groups=len(selected),
                    lifetime_sum_ratio=stats(g['lifetime_sum_ratio_to_two_solo_ordinal_median'] for g in selected),
                    ordinal_ratio_median=stats(g['ordinal_lifetime_ratio']['median'] for g in selected),
                    ordinal_ratio_p99=stats(g['ordinal_lifetime_ratio']['p99'] for g in selected),
                    top_one_percent_excess_share=stats(g['largest_one_percent_positive_lifetime_excess_fraction'] for g in selected),
                    synchronize_API_wall_ms=stats(g['synchronize_API_wall_union_ns']/1e6 for g in selected)))
    for source in sources:
        if receipt(source['path'])!=source:raise ValueError('Evidence changed during descriptive review')
    report=dict(schema='gsdeploy.phase_scheduler_supply_identifiability_diagnostic.v8',status='DESCRIPTIVE_CPU_REVIEW_COMPLETE',
        sources=sources,host_model=parameters['host_model'],elapsed_seconds=time.monotonic()-start,
        natural_rows=natural,light_phase_groups=groups,summary=summaries,
        measured_shared_request_count=len(natural),measured_shared_light_phase_group_count=len(groups),
        fitted_scheduler=False,quantum_identified=False,context_ownership_identified=False,
        resource_service_identified=False,kernel_lifetime_used_as_SM_service=False,
        future_validation_labels_used=False,whole_SSG_multiplier_fitted=False,GPU_started=False,
        accuracy_or_deployment_claimed=False)
    save(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parameters',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    report=analyze(args.parameters,args.output)
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds','measured_shared_request_count','measured_shared_light_phase_group_count')}))

if __name__=='__main__':main()
