"""Terminal-only strong/fine tail analysis; actual durations are post-hoc labels.

No fitting, predictor change, raw discovery beyond all-six audited receipts, or
GPU work. Worker timestamps are Unix ns, separately normalized from host clocks.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(v,q):
    if not v:return None
    v=sorted(v);i=(len(v)-1)*q;a=math.floor(i);b=math.ceil(i);return v[a]+(v[b]-v[a])*(i-a)
def stat(v):return dict(n=len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,p95=pct(v,.95),maximum=max(v) if v else None,total=sum(v))

def fifo(rows,services,releases=None):
    clocks=defaultdict(float);out=[]
    for i,r in enumerate(rows):
        release=releases[i] if releases is not None else r['arrival_seconds'];start=max(release,clocks[r['alias']]);end=start+services[i]/1000;clocks[r['alias']]=end
        out.append(dict(index=i,task=r['task'],alias=r['alias'],sample_id=r['sample_id'],latency_ms=(end-r['arrival_seconds'])*1000,queue_ms=(start-release)*1000,finish_seconds=end))
    tt=[r for r in out if r['task']=='text']
    return dict(text_latency=stat([r['latency_ms'] for r in tt]),text_queue=stat([r['queue_ms'] for r in tt]),rows=out)

def overlap_seconds(start,end,intervals):
    clipped=sorted((max(start,a),min(end,b)) for a,b in intervals if min(end,b)>max(start,a));total=0.;last=None
    for a,b in clipped:
        if last is None:last=(a,b)
        elif a<=last[1]:last=(last[0],max(last[1],b))
        else:total+=last[1]-last[0];last=(a,b)
    return total+(last[1]-last[0] if last else 0.)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);inputs={}
    def read(path,role):
        path=Path(path).resolve();inputs[str(path)]=dict(sha256=sha(path),role=role);return json.loads(path.read_text())
    audit_path=STUDY/'staged-continuation-v1/execution/all-six-comparison/report.json';audit=read(audit_path,'explicit_authorized_all_six_terminal_audit')
    assert len(audit['observations'])==6 and all(o['audit_status']=='VERIFIED_TERMINAL_EVIDENCE' for o in audit['observations'])
    queue_path=STUDY/'staged-continuation-v1/execution/strong-controls-gpu/queue-progress.json';queue=read(queue_path,'authorized_strong_terminal_receipts')
    assert queue['status']=='ALL_DECLARED_DEVELOPMENT_ATTEMPTS_COLLECTED' and len(queue['jobs'])==2
    source=ROOT/'tools/spillover_capacity_20261022/exposure.py';spec=importlib.util.spec_from_file_location('_strong_tail_cpu',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);api=module.frozen_api()
    inputs[str(source)]=dict(sha256=sha(source),role='isolated_frozen_cpu_loader')
    for rel in ['gsdeploy/trace.py','gsdeploy/routing.py','gsdeploy/runtime/workload.py']:
        path=module.SOURCE/rel;inputs[str(path)]=dict(sha256=sha(path),role='frozen_cpu_trace_sample_route_source')
    pool_path=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl';inputs[str(pool_path)]=dict(sha256=sha(pool_path),role='frozen_common_input_pool')
    pairs=[];all_windows=[];per_input=[];detail_rows=[];background_rows=[]
    for seed in [107,108]:
        obs=[next(o for o in audit['observations'] if o['label']==label) for label in [f'text185_g2share_0of1_seed{seed}',f'text185_all_specialist_text2_seed{seed}']]
        raws=[];planned=[]
        for o in obs:
            receipt=o['raw_receipt'];assert sha(receipt['path'])==receipt['sha256']
            if 'all_specialist' in o['label']:
                q=next(j for j in queue['jobs'] if j['job_id']==o['job_id']);assert q['raw']==receipt and q['status']=='completed' and q['eligible']
            raw=read(receipt['path'],'authorized_terminal_fine_or_strong_raw');assert raw['status']=='completed' and raw['transport_config']['random_seed']==seed
            assert o['utility']['eligible'] and o['route_and_input_replay_verified'] and raw['experiment_job_id']==o['job_id']
            cfg=raw['config'];assert cfg['catalog']['quality']['identity']['input_pool_sha256']==sha(pool_path)
            trace=api['trace'].make_trace(cfg['workload'],raw['transport_config']['measurement_seconds'],seed,raw['transport_config']['arrival_process']);assert api['trace'].trace_digest(trace)==o['trace_sha256']
            aliases,routes=api['trace'].expand_routes(raw['plan']['deployment'],raw['plan']['routing']);pool=api['workload'].SamplePool(pool_path,list(cfg['workload']),seed);router=api['routing'].Router(routes);rows=[]
            assert len(trace)==len(raw['request_records'])
            for i,((arrival,task),r) in enumerate(zip(trace,raw['request_records'])):
                sample=pool.next(task);alias=router.choose(task);assert (task,sample['pool_id'],alias)==(r['task'],r['pool_id'],r['model']);assert abs(r['scheduled_offset_seconds']-arrival)<1e-8
                assert r['status']=='completed' and r['worker_response']['success'];w=r['worker_response']
                assert abs((w['start_ns']-w['submit_ns'])/1e6-w['queue_wait_ms'])<1e-5
                assert abs((w['end_ns']-w['start_ns'])/1e6-w['worker_latency_ms'])<1e-5
                start_unix=raw['measurement_window']['start_unix_ns'];start=(w['start_ns']-start_unix)/1e9;end=(w['end_ns']-start_unix)/1e9
                assert start>=arrival-.01 and end>start
                rows.append(dict(index=i,arrival_seconds=arrival,task=task,alias=alias,family=aliases[alias],sample_id=sample['pool_id'],
                    native_service_ms=w['service_ms'],handler_ms=w['worker_latency_ms'],worker_start_seconds=start,worker_end_seconds=end,
                    submit_seconds=(r['actual_submit_time_ns']-raw['measurement_window']['start_time_ns'])/1e9,
                    observed_e2e_ms=r['e2e_latency_ms'],observed_queue_ms=w['queue_wait_ms']))
            raws.append(raw);planned.append(rows)
        fine,strong=raws;fr,sr=planned
        assert [(r['arrival_seconds'],r['task'],r['sample_id']) for r in fr]==[(r['arrival_seconds'],r['task'],r['sample_id']) for r in sr]
        text_indices=[i for i,r in enumerate(fr) if r['task']=='text'];assert all(fr[i]['alias']==sr[i]['alias'] for i in text_indices)
        text_identity={k:sum(fine['request_records'][i]['worker_response'].get(k)==strong['request_records'][i]['worker_response'].get(k) for i in text_indices) for k in ['request_sha256','protocol_sha256','effective_generation','generated_tokens','raw_output']}
        tables=[]
        for rows in [fr,sr]:
            groups=defaultdict(list)
            for r in rows:
                if r['task']=='text':groups[r['alias'],r['sample_id']].append(r['native_service_ms'])
            tables.append({k:statistics.mean(v) for k,v in groups.items()})
        assert set(tables[0])==set(tables[1])
        for key in sorted(tables[0]):per_input.append(dict(seed=seed,alias=key[0],sample_id=key[1],fine_mean_ms=tables[0][key],strong_mean_ms=tables[1][key],strong_over_fine=tables[1][key]/tables[0][key]))
        jobs=[]
        for name,raw,rows,observation in zip(['fine_nosplit','strong_all_specialist'],raws,planned,obs):
            native=[r['native_service_ms'] for r in rows];handler=[r['handler_ms'] for r in rows];release=[r['submit_seconds'] for r in rows]
            modes=[('actual_native',native,None),('actual_handler_actual_submit',handler,release),
                ('fine_alias_input_mean_text_actual_background',[tables[0][r['alias'],r['sample_id']] if r['task']=='text' else r['native_service_ms'] for r in rows],None),
                ('strong_alias_input_mean_text_actual_background',[tables[1][r['alias'],r['sample_id']] if r['task']=='text' else r['native_service_ms'] for r in rows],None)]
            replays=[]
            for label,service,submit in modes:
                result=fifo(rows,service,submit);replays.append(dict(name=label,text_latency=result['text_latency'],text_queue=result['text_queue']))
                if label=='actual_native':native_replay=result
                if label=='actual_handler_actual_submit':handler_residual=stat([r['observed_e2e_ms']-p['latency_ms'] for r,p in zip(rows,result['rows'])])
            background=[r for r in rows if r['task']!='text'];intervals=[(r['worker_start_seconds'],r['worker_end_seconds']) for r in background]
            for r in background:background_rows.append(dict(seed=seed,condition=name,**r))
            text=[]
            for i in text_indices:
                r=rows[i];fine_reference=fr[i]['native_service_ms'];overlap=overlap_seconds(r['worker_start_seconds'],r['worker_end_seconds'],intervals)
                text.append(dict(seed=seed,condition=name,**r,exact_request_fine_service_ms=fine_reference,
                    strong_or_fine_over_exact_fine_service=r['native_service_ms']/fine_reference,
                    relative_to_own_alias_input_mean=r['native_service_ms']/tables[0 if name=='fine_nosplit' else 1][r['alias'],r['sample_id']],
                    background_handler_overlap_seconds=overlap,background_handler_overlap_fraction=overlap/(r['worker_end_seconds']-r['worker_start_seconds']),
                    posthoc_only_not_predictor_covariate=True))
            detail_rows.extend(text)
            overlap_summary=[]
            for overlaps in [False,True]:
                rr=[r for r in text if (r['background_handler_overlap_seconds']>0)==overlaps]
                overlap_summary.append(dict(overlaps_background=overlaps,requests=len(rr),
                    exact_fine_service_ratio=stat([r['strong_or_fine_over_exact_fine_service'] for r in rr]),
                    own_alias_input_mean_normalized_service=stat([r['relative_to_own_alias_input_mean'] for r in rr]),
                    caveat='Observed handler intervals overlap is post-hoc, not proof of GPU activity or causality; small selected samples and length bias.'))
            alias_stats=[]
            for alias in sorted({r['alias'] for r in text}):
                rr=[r for r in text if r['alias']==alias];alias_stats.append(dict(alias=alias,n=len(rr),service=stat([r['native_service_ms'] for r in rr]),
                    observed_e2e=stat([r['observed_e2e_ms'] for r in rr]),queue=stat([r['observed_queue_ms'] for r in rr])))
            # Scheduled 30s blocks are predetermined; labels/correlations remain post-hoc.
            windows=[]
            for begin in range(0,600,30):
                rr=[r for r in text if begin<=r['arrival_seconds']<begin+30]
                windows.append(dict(begin_seconds=begin,end_seconds=begin+30,count=len(rr),service=stat([r['native_service_ms'] for r in rr]),
                    exact_fine_total_ms=sum(r['exact_request_fine_service_ms'] for r in rr),
                    excess_over_exact_fine_total_ms=sum(r['native_service_ms']-r['exact_request_fine_service_ms'] for r in rr),
                    observed_e2e=stat([r['observed_e2e_ms'] for r in rr]),
                    overlapping_background_requests=sum(r['background_handler_overlap_seconds']>0 for r in rr)))
            episodes=[]
            for alias in sorted({r['alias'] for r in text}):
                block=[]
                for r in [r for r in text if r['alias']==alias]+[None]:
                    if r is not None and r['observed_queue_ms']>1000:block.append(r)
                    elif block:
                        episodes.append(dict(alias=alias,requests=len(block),first_scheduled_seconds=block[0]['arrival_seconds'],last_scheduled_seconds=block[-1]['arrival_seconds'],
                            last_completion_seconds=block[-1]['worker_end_seconds'],scheduled_span_seconds=block[-1]['arrival_seconds']-block[0]['arrival_seconds'],
                            queue=stat([v['observed_queue_ms'] for v in block])));block=[]
            jobs.append(dict(condition=name,seed=seed,label=observation['label'],utility=observation['utility'],
                measured_text_metrics=raw['per_task_metrics']['text'],alias_text=alias_stats,replays=replays,
                handler_submit_FIFO_e2e_residual_ms=handler_residual,background_handler_intervals=background,
                background_overlap_diagnostics=overlap_summary,scheduled30s_blocks=windows,
                longest_queue_wait_episodes=sorted(episodes,key=lambda e:e['scheduled_span_seconds'],reverse=True)[:8],
                longest_native_text_requests=sorted(text,key=lambda r:r['native_service_ms'],reverse=True)[:20]))
        # Replace one scheduled block at a time. The large-tail block is selected
        # after outcomes and cannot become an observed predictor input.
        replacements=[];actual_native=fifo(sr,[r['native_service_ms'] for r in sr])
        for width in [30,60]:
            for begin in range(0,601-width,30):
                selected=[i for i,r in enumerate(sr) if r['task']=='text' and begin<=r['arrival_seconds']<begin+width]
                values=[fr[i]['native_service_ms'] if i in selected else r['native_service_ms'] for i,r in enumerate(sr)]
                replay=fifo(sr,values);replacements.append(dict(begin_seconds=begin,width_seconds=width,replaced_text_requests=len(selected),
                    actual_strong_native_p95_ms=actual_native['text_latency']['p95'],replayed_p95_ms=replay['text_latency']['p95'],
                    p95_reduction_ms=actual_native['text_latency']['p95']-replay['text_latency']['p95'],
                    excess_native_service_removed_ms=sum(sr[i]['native_service_ms']-fr[i]['native_service_ms'] for i in selected),
                    interpretation='Post-hoc substitution freezes other real durations; neither causal attribution nor a feasible counterfactual execution.'))
        all_windows.append(dict(seed=seed,window_replacements=replacements))
        pairs.append(dict(seed=seed,exact_text_arrival_input_alias_equal=True,text_requests=len(text_indices),text_identity_equal_counts=text_identity,
            jobs=jobs,largest_block_substitutions=sorted(replacements,key=lambda r:r['p95_reduction_ms'],reverse=True)[:8],
            actual_native_strong_minus_fine_p95_ms=jobs[1]['replays'][0]['text_latency']['p95']-jobs[0]['replays'][0]['text_latency']['p95']))
    result=dict(schema='gsdeploy.strong_fine_tail_posthoc.v1',pairs=pairs,fitted=False,predictor_activated=False,GPU_started=False,
        timestamp_contract='Worker start/end/submit are Unix ns normalized with measurement start_unix_ns; host submit/release uses monotonic start_time_ns separately. Queue/handler equalities verified.',
        limitations=['Only four audited terminal fine-nosplit/strong raws were read; no single-specialist live raw.',
            'Background handler interval overlap is post-hoc association and includes CPU/launch/sync, not measured GPU kernel overlap.',
            'Native durations include synchronize/timing and may reflect concurrent work, scheduling or host/device state; no cause is identified.',
            'Block substitutions and conditional means use realized labels and are not fitted covariates or prospective predictions.',
            'Two nonrandom ordered deployment pairs do not establish a global deployment preference or optimality.'])
    save(out/'report.json',result);save(out/'per-input-conditional-service.json',per_input);save(out/'text-request-posthoc-details.json',detail_rows)
    save(out/'background-handler-intervals.json',background_rows);save(out/'all-window-substitutions.json',all_windows);save(out/'opened-inputs.json',inputs)
    for path,r in inputs.items():assert sha(path)==r['sha256']
    save(out/'manifest.json',dict(script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),inputs=inputs,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},all_inputs_unchanged=True))
    print(json.dumps([dict(seed=p['seed'],jobs=[dict(condition=j['condition'],replays=j['replays'],overlap=j['background_overlap_diagnostics']) for j in p['jobs']],largest_blocks=p['largest_block_substitutions'][:2]) for p in pairs]))

if __name__=='__main__':main()
