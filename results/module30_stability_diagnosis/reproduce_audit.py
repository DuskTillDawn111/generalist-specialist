"""Module30 CPU-only forensic audit. No workers/runtime/controller imports."""
import sys
sys.dont_write_bytecode=True
import ast,collections,copy,csv,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=Path(__file__).resolve().parent
M28=ROOT/'results/module28_dynamic_split';M29=ROOT/'results/module29_offline_opportunity'
D=M29/'stage2_20261009T153923Z'
csv.field_size_limit(128*1024*1024)

def save(name,x):
    assert Path(name).name==name
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvsave(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def stats(values):
    v=[float(x) for x in values if x is not None]
    return dict(n=len(v),mean_ms=float(np.mean(v)) if v else None,p10_ms=float(np.percentile(v,10)) if v else None,
                p50_ms=float(np.percentile(v,50)) if v else None,p95_ms=float(np.percentile(v,95)) if v else None,max_ms=max(v) if v else None)
def load_source(path,allowed):
    source=Path(path).read_text();tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(a.name in allowed for a in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module in allowed
    space={};exec(compile(tree,str(path),'exec'),space);return space
def state(router):
    return dict(mode=router.mode,tokens=router.tokens,last_time=router.last_time,next_g_time=router.next_g_time,
        fallback_until=router.fallback_until,service=[dict(worker=w,task=t,ewma_ms=v) for (w,t),v in sorted(router.service.items())],
        predictions=dict(router.predictions),prediction_errors=copy.deepcopy(router.prediction_errors))
def verify_inputs(final=False):
    p=OUT/'input_integrity_before_after.json';freeze=json.loads(p.read_text());changes=[]
    for r in freeze['inputs']:
        q=Path(r['path'])
        if not q.is_file() or q.stat().st_size!=r['bytes'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:changes.append(r['path'])
    sets={d:sorted(str(p.resolve()) for p in Path(d).rglob('*') if p.is_file()) for d in freeze['historical_directory_file_sets']}
    sets_changed=[d for d in sets if sets[d]!=freeze['historical_directory_file_sets'][d]]
    assert not changes and not sets_changed,(changes,sets_changed)
    if final:
        freeze['after']=dict(checked=len(freeze['inputs']),all_sets_sizes_SHA256_unchanged=True,changes=[],set_changes=[],GPU_inferences_started=0)
        save(p.name,freeze)
    return len(freeze['inputs'])
def native(r,start):
    v=r.get('worker_response',{})
    if any(v.get(k) is None for k in ['start_ns','end_ns']) or r.get('actual_submit_unix_ns') is None:return None
    delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns']
    return ((v['start_ns']+delta-start)/1e9,(v['end_ns']+delta-start)/1e9)
def busy_seconds(rows,left,right):
    # Per-worker handler intervals are verified disjoint; clipped work, not CUDA utilization.
    return sum(max(0.,min(r['_native'][1],right)-max(r['_native'][0],left)) for r in rows if r['_native'])
def integrate(rows,start,end,arrival,completion):
    events=collections.Counter()
    for r in rows:
        if r.get(arrival) is not None:events[r[arrival]]+=1
        if r['status']=='completed' and r.get(completion) is not None:events[r[completion]]-=1
    b=0;cursor=start;area=0.;busy=0.;peak=0;points=[(start,0)]
    for t,db in sorted(events.items()):
        if t<start:b+=db;continue
        if t>end:break
        area+=b*(t-cursor)/1e9;busy+=(t-cursor)/1e9 if b>0 else 0.
        b+=db;assert b>=0
        peak=max(peak,b);points.append((t,b));cursor=t
    area+=b*(end-cursor)/1e9;busy+=(end-cursor)/1e9 if b>0 else 0.
    return dict(mean=area/((end-start)/1e9),end=b,peak=peak,busy_seconds=busy,points=points)
def ols(bins):
    ts=np.array([x['offset_seconds'] for x in bins]);ys=np.array([x['mean_backlog_requests'] for x in bins])
    slope=float(np.dot(ts-ts.mean(),ys-ys.mean())/np.sum((ts-ts.mean())**2))
    predicted=ys.mean()+slope*(ts-ts.mean());residual=ys-predicted
    return dict(slope=slope,growth=float(ys[-1]-ys[0]),increasing_fraction=float(np.mean(np.diff(ys)>0)),
        values=ys.tolist(),fitted=predicted.tolist(),residuals=residual.tolist(),rmse=float(np.sqrt(np.mean(residual**2))))
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;h=z*z;n=float(n);p=k/n;center=(p+h/(2*n))/(1+h/n)
    radius=z*math.sqrt(p*(1-p)/n+h/(4*n*n))/(1+h/n)
    return dict(lower=center-radius,upper=center+radius,method='Nominal 95% Wilson; assumes independent Bernoulli requests, not independent runs')

def main():
    protected=verify_inputs()
    classifier=load_source(ROOT/'performance_model/local_calibration/stability_v2.py',{'collections','math','numpy'})
    Router=load_source(M28/'dynamic_router.py',{'collections'})['DynamicRouter']
    f=json.loads((M28/'frozen_candidate_manifest.json').read_text());p=f['parameters'];S=p['S_worker'];G=p['G_worker']
    raws={'S_only':json.loads((M28/'raw_runs/C_steady_s_only.json').read_text()),'Dynamic':json.loads((D/'gpu_validation_raw.json').read_text())}
    trajectory=json.loads(Path(f['exploration_slots'][0]['trajectories']['measurement']['path']).read_text())
    input_inventory=['# 输入盘点与字段证据\n\n','所有历史输出也作为输入冻结；完整309个文件的路径、大小和SHA-256见input_integrity_before_after.json。\n\n',
        '|来源|字段/用途|缺口|\n|---|---|---|\n',
        '|Module28 raw_runs/C_steady_s_only.json|request_id/task/scheduled/enqueued/actual_submit/completion、worker_response start/end、service/queue、LID routing_decision|无collector commit和完整在线router状态；四个旧快照的reader/collector滞后不能用未来事件回填|\n',
        '|Module29 stage2 gpu_validation_raw.json|同上，新增collector_commit_time_ns、decide及完整arrival开销|预测文本和reference已脱敏；G LID无valid_closed_set_output字段|\n',
        '|Module29 stage2 decision_event_journal.jsonl|实际locked observation时点、全部pending、历史completion harvest、完整路由器状态|无worker receipt/START ACK，真实worker队列仍UNKNOWN|\n',
        '|Module29 stage2 routing_decision_trace.csv|含warmup；本审计筛phase=measurement得到7005条|不得将warmup338条混入目标分母|\n',
        '|原冻结轨迹|同一正式请求的pool_id/sample_id、任务、计划偏移和LID语言标签|不能因此认定两个时段/插桩不同运行构成合法匹配效用对照|\n',
        '|原分类器stability_v2.py|20个30s bin、尾5个bin、OLS及全部硬/警告条件|原实现恢复/超时/全观察peak涉及排空，原样复算；排空不进入正式bin/OLS、不解除硬失败|\n\n',
        '主时钟为每次formal start的perf_counter_ns；native Unix用每请求paired submit映射，仅用于事后执行分析。所有正式bin为[0,600]内，计数半开区间；边界事件单列核验。host enqueued pending、host submitted B、native未开始handler代理与真实worker等待队列分别命名。\n']
    (OUT/'input_inventory.md').write_text(''.join(input_inventory))
    reproduction={};reconciliation={};bins_rows=[];tail_rows=[];timelines={};run_analysis={};native_rows=[];latency_rows=[];quality_rows=[];label_rows=[]
    for name,raw in raws.items():
        ph=raw['phases']['measurement'];rs=ph['request_records'];win=ph['window'];start=win['start_time_ns'];end=win['end_time_ns'];stop=win['drain_end_time_ns']
        assert end-start==600000000000 and len(rs)==7607 and all(r['status']=='completed' for r in rs)
        assert len({r['request_id'] for r in rs})==len(rs)
        missing={k:sum(r.get(k) is None for r in rs) for k in ['scheduled_time_ns','enqueued_time_ns','actual_submit_time_ns','completion_time_ns','actual_submit_unix_ns']}
        assert not any(missing.values())
        for r,e in zip(rs,trajectory['events']):
            assert r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['pool_id']==e['item']['pool_id'] and r['frozen_offset_seconds']==e['offset_seconds']
            assert start<=r['scheduled_time_ns']<=r['enqueued_time_ns']<=r['actual_submit_time_ns']<=r['completion_time_ns']<=stop
            r['_native']=native(r,start);assert r['_native']
            if name=='Dynamic':assert r['completion_time_ns']<=r['collector_commit_time_ns']
            r['_observable_completion_ns']=r.get('collector_commit_time_ns',r['completion_time_ns'])
        sent=sum(start<=r['actual_submit_time_ns']<end for r in rs);within=sum(start<=r['completion_time_ns']<end for r in rs)
        reconciliation[name]=dict(formal_seconds=600.,sent=sent,within_success=within,end_unfinished=sent-within,
            drain_success=len(rs)-within,total_success_after_drain=len(rs),drain_seconds=(stop-end)/1e9,
            offered_qps=len(rs)/600.,within_completed_qps=within/600.,completed_cohort_divided_by_600_qps=len(rs)/600.,
            missing_fields=missing,duplicate_IDs=0,task_counts=dict(collections.Counter(r['task'] for r in rs)),
            bin_boundary_events={k:sum((r[k]-start)%30000000000==0 for r in rs) for k in ['actual_submit_time_ns','completion_time_ns']},
            original_legal_U=None,scope='All counts from events; drain is separate, not added to within-window throughput')
        assert (sent,within,sent-within)==(7607,7575,32) if name=='S_only' else (sent,within,sent-within)==(7607,7588,19)
        aliases={w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for w,m in raw['replica_to_family'].items()}
        data=dict(request_records=rs,measurement_window=win,errors=raw['errors'],shutdown=raw['shutdown'],
            status='completed',drain_timeout_seconds=60.,worker_identities=raw['worker_identities'],system_metrics=ph['system_metrics'])
        ev=classifier['reconstruct'](data,aliases,classifier['DEFAULT_THRESHOLDS']);classification=classifier['classify'](ev,classifier['DEFAULT_THRESHOLDS'])
        assert classification==ph['stability']
        assert ev['backlog_timeline']==ph['stability_evidence']['backlog_timeline']
        tail=ols(ev['backlog_timeline'][-5:]);assert abs(tail['slope']-ev['tail_backlog_slope'])<1e-12
        conditions=dict(slope_ge_0_02=tail['slope']>=.02,growth_ge_5=tail['growth']>=5.,increasing_ge_0_6=tail['increasing_fraction']>=.6,end_pending_ge_5=ev['end_backlog_requests']>=5)
        reproduction[name]=dict(result=classification,identical_to_saved=True,formal_tail_only=tail,
            conjunctive_tail_failure_conditions=conditions,tail_end_pending=ev['end_backlog_requests'],
            formal_bin_edges_seconds=list(range(30,601,30)),tail_bin_intervals=[[450,480],[480,510],[510,540],[540,570],[570,600]],
            original_non_tail_checks={k:ev[k] for k in ['completion_ratio','incomplete_requests','worker_or_request_errors','run_errors','run_status','unclean_shutdown','drain_timeout','final_backlog_after_drain','peak_backlog_requests','peak_service_demand_backlog','peak_queue_wait_ms','drain_duration']},
            saved_state=ph['stability']['stability_state_v2'])
        timelinegroups={'ALL':rs,'S':[r for r in rs if r['model']==S],'G':[r for r in rs if r['model']==G]}
        timelines[name]={}
        for key,rows in timelinegroups.items():
            a=integrate(rows,start,stop,'actual_submit_time_ns','completion_time_ns')
            h=integrate(rows,start,stop,'enqueued_time_ns','_observable_completion_ns')
            timelines[name][key]=dict(B=[((t-start)/1e9,b) for t,b in a['points']],H=[((t-start)/1e9,b) for t,b in h['points']],drain_seconds=(stop-end)/1e9,
                H_scope='collector-visible enqueued unfinished' if name=='Dynamic' else 'reader-based enqueued unfinished proxy; collector commit UNKNOWN')
        # Scopes are alternative partitions. Sum task slopes OR worker slopes, never both.
        groups=[('ALL','ALL',rs)]
        groups += [('WORKER',w,[r for r in rs if r['model']==w]) for w in sorted(raw['replica_to_family'])]
        groups += [('TASK',t,[r for r in rs if r['task']==t]) for t in sorted(f['rates'])]
        groups += [('WORKER_TASK',w+'|'+t,[r for r in rs if r['model']==w and r['task']==t]) for w,t in sorted({(r['model'],r['task']) for r in rs})]
        for scope,key,rows in groups:
            tm=classifier['timeline'](rows,start,end,stop,aliases,20,'actual_submit_time_ns');fit=ols(tm['bins'][-5:])
            senttail=sum(start+450000000000<=r['actual_submit_time_ns']<end for r in rows)
            donetail=sum(start+450000000000<=r['completion_time_ns']<end for r in rows)
            b450=integrate(rows,start,start+450000000000,'actual_submit_time_ns','completion_time_ns')['end']
            tail_rows.append(dict(run=name,scope=scope,key=key,slope_contribution_rps=fit['slope'],mean_bin_growth=fit['growth'],
                pending_at_450=b450,pending_at_600=tm['end_backlog_requests'],tail_sent=senttail,tail_success=donetail,
                endpoint_growth=tm['end_backlog_requests']-b450,tail_executed_handler_seconds=busy_seconds(rows,450.,600.),
                tail_bin_means=fit['values'],tail_bin_end_pending=[x['end_backlog_requests'] for x in tm['bins'][-5:]],
                fitted=fit['fitted'],residuals=fit['residuals'],tail_increasing_fraction=fit['increasing_fraction']))
            for i,b in enumerate(tm['bins']):
                left=i*30;right=left+30;l=start+left*1000000000;rtime=start+right*1000000000
                completed=[r for r in rows if l<=r['completion_time_ns']<rtime]
                sentbin=sum(l<=r['actual_submit_time_ns']<rtime for r in rows)
                busy=busy_seconds(rows,left,right)
                bins_rows.append(dict(run=name,scope=scope,key=key,left_s=left,right_s=right,sent=sentbin,successful_responses=len(completed),
                    pending_at_right=b['end_backlog_requests'],time_mean_submitted_pending=b['mean_backlog_requests'],
                    service_completed_in_bin=stats([r['worker_latency_ms'] for r in completed]),
                    queue_completed_in_bin=stats([r['worker_queue_wait_ms'] for r in completed]),
                    G_route_completions=sum(r['model']==G for r in completed),S_route_completions=sum(r['model']==S for r in completed),
                    executed_handler_seconds_in_bin=busy,tail_fitted_mean=fit['fitted'][i-15] if i>=15 else None,
                    tail_OLS_residual_requests=fit['residuals'][i-15] if i>=15 else None,
                    metric_population='Successful responses in [left,right); handler work separately clipped to actual execution interval',
                    worker_pending='UNKNOWN_NO_RECEIPT_ACK',worker_waiting_queue='UNKNOWN_NO_RECEIPT_ACK'))
        for scope in ['TASK','WORKER']:
            assert abs(sum(r['slope_contribution_rps'] for r in tail_rows if r['run']==name and r['scope']==scope)-tail['slope'])<1e-10
        busy_analysis={}
        for worker in sorted(raw['replica_to_family']):
            wr=[r for r in rs if r['model']==worker];iv=sorted(r['_native'] for r in wr)
            overlaps=sum(a[1]>b[0]+.0001 for a,b in zip(iv,iv[1:]));assert overlaps==0
            busy_analysis[worker]=dict(native_busy_formal_seconds=busy_seconds(wr,0,600),native_busy_tail_seconds=busy_seconds(wr,450,600),
                busy_by_task={t:busy_seconds([r for r in wr if r['task']==t],0,600) for t in sorted({r['task'] for r in wr})},
                busy_tail_by_task={t:busy_seconds([r for r in wr if r['task']==t],450,600) for t in sorted({r['task'] for r in wr})},
                simultaneous_same_worker_intervals=overlaps)
        run_analysis[name]=dict(worker_busy=busy_analysis,formal_mean_e2e=stats([r['e2e_latency_ms'] for r in rs]),
            formal_S_host_pending=integrate(timelinegroups['S'],start,end,'enqueued_time_ns','_observable_completion_ns')|{'points':None},
            S_tail_native_busy_fraction=busy_seconds(timelinegroups['S'],450,600)/150.)
        for r in rs:
            a,b=r['_native'];mapped_end=start+round(b*1e9)
            delay=dict(preenqueue_ms=(r['enqueued_time_ns']-r['scheduled_time_ns'])/1e6,
                host_sender_wait_ms=(r['actual_submit_time_ns']-r['enqueued_time_ns'])/1e6,
                host_schedule_to_send_ms=r['schedule_delay_ms'],worker_wait_transport_ms=r['worker_queue_wait_ms'],
                handler_service_ms=r['worker_latency_ms'],response_propagation_ms=(r['completion_time_ns']-mapped_end)/1e6)
            assert abs(delay['host_schedule_to_send_ms']+delay['worker_wait_transport_ms']+delay['handler_service_ms']+delay['response_propagation_ms']-r['e2e_latency_ms'])<.00001
            native_rows.append(dict(run=name,request_id=r['request_id'],task=r['task'],worker=r['model'],
                scheduled_s=(r['scheduled_time_ns']-start)/1e9,enqueued_s=(r['enqueued_time_ns']-start)/1e9,
                submitted_s=(r['actual_submit_time_ns']-start)/1e9,completion_s=(r['completion_time_ns']-start)/1e9,
                collector_commit_s=(r['collector_commit_time_ns']-start)/1e9 if r.get('collector_commit_time_ns') else None,
                native_start_s=a,native_end_s=b,**delay,worker_queue='UNKNOWN',native_scope='RETROSPECTIVE_ONLY'))
        for w,t in sorted({(r['model'],r['task']) for r in rs}):
            rows=[r for r in rs if r['model']==w and r['task']==t];nr=[r for r in native_rows if r['run']==name and r['worker']==w and r['task']==t]
            latency_rows.append(dict(run=name,worker=w,task=t,n=len(rows),e2e=stats([r['e2e_latency_ms'] for r in rows]),
                preenqueue=stats([r['preenqueue_ms'] for r in nr]),sender_queue=stats([r['host_sender_wait_ms'] for r in nr]),
                waiting_transport=stats([r['worker_wait_transport_ms'] for r in nr]),service=stats([r['handler_service_ms'] for r in nr]),
                response_propagation=stats([r['response_propagation_ms'] for r in nr])))
            correct=[r['worker_response']['correct'] for r in rows if isinstance(r['worker_response'].get('correct'),bool)]
            wer=[r['worker_response']['wer'] for r in rows if r['worker_response'].get('wer') is not None]
            valid=[r['worker_response']['valid_closed_set_output'] for r in rows if 'valid_closed_set_output' in r['worker_response']]
            quality_rows.append(dict(run=name,worker=w,task=t,n=len(rows),correct_available=len(correct),correct=sum(correct),
                accuracy=sum(correct)/len(correct) if correct else None,Wilson95=wilson(sum(correct),len(correct)),
                WER_n=len(wer),WER_mean=float(np.mean(wer)) if wer else None,closed_set_flag_n=len(valid),explicit_invalid=sum(v is False for v in valid),
                closed_set_validity='UNKNOWN' if len(valid)!=len(rows) else 'OBSERVED_FLAGS',
                incorrect_type='UNKNOWN_PREDICTION_REDACTED' if name=='Dynamic' else 'Incorrect flag; predictions not published',
                unique_sample_ids=len({r['sample_id'] for r in rows})))
            if t=='lid':
                labels=collections.defaultdict(list)
                for r in rows:
                    sample=trajectory['events'][r['arrival_index']]['item']['sample'];label=sample.get('label')
                    if not isinstance(label,str) or len(label)>32:label='UNKNOWN'
                    labels[label].append(r)
                for label,rr in sorted(labels.items()):
                    cc=[r['worker_response'].get('correct') for r in rr if isinstance(r['worker_response'].get('correct'),bool)]
                    label_rows.append(dict(run=name,worker=w,task=t,language_label=label,n=len(rr),correct_available=len(cc),correct=sum(cc),
                        accuracy=sum(cc)/len(cc) if cc else None,error_count=len(cc)-sum(cc),predicted_error_class='UNKNOWN' if name=='Dynamic' else 'Not published',
                        scope='Observed routed subset, not same-condition causal quality comparison'))
    csvsave('stability_bins.csv',bins_rows);csvsave('tail_growth_decomposition.csv',tail_rows)
    csvsave('normalized_events_and_latencies.csv',native_rows);csvsave('latency_component_summary.csv',latency_rows)
    csvsave('quality_by_task_worker.csv',quality_rows);csvsave('quality_LID_by_language.csv',label_rows)
    save('window_reconciliation.json',dict(runs=reconciliation,cohort_identical=True,legally_matched_utility_contrast=False,
        time_basis='Each formal perf_counter_ns origin; formal [0,600), drain separate; no future data substituted for online state'))
    save('stability_reproduction.json',dict(source=str(ROOT/'performance_model/local_calibration/stability_v2.py'),
        source_sha256=hashlib.sha256((ROOT/'performance_model/local_calibration/stability_v2.py').read_bytes()).hexdigest(),
        original_thresholds=classifier['DEFAULT_THRESHOLDS'],runs=reproduction,
        drain_scope='Formal bin means, four hard tail-growth tests and OLS exclude drain. Original classifier recovery/timeout/entire-observation peaks also read drain, reproduced unchanged and separated; drain never cancels hard tail failure.'))
    save('pending_curves.json',timelines);save('run_analysis.json',run_analysis)
    # Full causal policy replay; handler times and answers are never replay inputs.
    replay=None;commits={};outstanding={};issues=[];target_events=[];last_seq=0
    for line in (D/'decision_event_journal.jsonl').open():
        e=json.loads(line);assert e['seq']==last_seq+1;last_seq=e['seq']
        if e['kind']=='ENQUEUE' and e['status'] in ['queued','submitted']:outstanding[e['request_id']]=(e['worker'],e['task'],e['time_ns'])
        if e['kind']=='COLLECTOR_COMMIT':commits[e['request_id']]=e;outstanding.pop(e['request_id'],None)
        if e['kind']=='PHASE_START':replay=Router(copy.deepcopy(p));assert state(replay)==e['router_state']
        if e['kind']!='ARRIVAL_DECISION':continue
        now=e['time_ns'];assert state(replay)==e['router_before_harvest']
        for r in e['harvested_completions']:
            assert r['completion_time_ns']<=now and r['collector_commit_time_ns']<=now
            assert commits[r['request_id']]['time_ns']==r['collector_commit_time_ns']
            replay.observe_completion(r,now/1e9)
        assert state(replay)==e['router_before_decide']
        seen={r['request_id']:(r['model'],r['task'],r['enqueued_time_ns']) for rows in e['pending'].values() for r in rows}
        assert seen==outstanding
        if e['target_router_invoked']:
            d=e['decision'];route,recovered=replay.decide(e['request_id'],now/1e9,e['pending'],d['admitted'],d['capacity_ok'])
            assert recovered==d and route==e['selected_worker'] and state(replay)==e['router_after_decide']
            if e['phase']=='measurement':target_events.append(e)
    assert len(target_events)==7005
    reasons=collections.Counter(e['decision']['reason'] for e in target_events);guards=collections.defaultdict(collections.Counter)
    eligible_reasons=collections.Counter();eligible_guard_fail=collections.Counter();empty_entry_pass=0
    dynraw=raws['Dynamic'];drs=dynraw['phases']['measurement']['request_records'];byid={r['request_id']:r for r in drs};start=dynraw['phases']['measurement']['window']['start_time_ns']
    diversions=[]
    for e in target_events:
        d=e['decision'];st=e['router_before_decide'];at=e['time_ns'];now=at/1e9
        tokens=min(1.,st['tokens']+max(0.,now-st['last_time'])*p['G_token_rate_rps']) if st['last_time'] is not None else st['tokens']
        mode=st['mode'];gap=d['margin_ms']
        if mode and gap<p['exit_margin_ms']:mode=False
        elif not mode and gap>p['entry_margin_ms']:mode=True
        gs=dict(admitted=d['admitted'],capacity_ok=d['capacity_ok'],prediction_fallback=now>=st['fallback_until'],
            G_target_inflight=d['G_target_inflight']<p['G_max_target_inflight'],G_empty=d['G_pending']==0,
            cooldown=now>=st['next_g_time'],token_backpressure=tokens>=1.,hysteresis_mode_after_update=mode,
            exit_cost_gap=gap>p['exit_margin_ms'],entry_cost_gap=gap>p['entry_margin_ms'])
        assert {k:'PASS' if v else 'FAIL' for k,v in gs.items()}==e['guards']
        for k,v in gs.items():guards[k]['PASS' if v else 'FAIL']+=1
        cost_eligible=mode and gap>p['exit_margin_ms']
        if cost_eligible:
            eligible_reasons[d['reason']]+=1
            for k,v in gs.items():
                if not v:eligible_guard_fail[k]+=1
        empty_entry_pass+=gs['G_empty'] and gs['entry_cost_gap']
        if e['selected_worker']!=G:continue
        r=byid[e['request_id']]
        diversions.append(dict(request_id=e['request_id'],task=e['task'],decision_s=(at-start)/1e9,
            scheduled_s=(r['scheduled_time_ns']-start)/1e9,S_pending=d['S_pending'],G_pending=d['G_pending'],
            S_composition=dict(collections.Counter(r['task'] for r in e['pending'].get(S,[]))),
            G_composition=dict(collections.Counter(r['task'] for r in e['pending'].get(G,[]))),
            C_S_ms=d['S_system_cost_proxy_ms'],C_G_ms=d['G_system_cost_proxy_ms'],gap_ms=gap,
            entry_margin_ms=p['entry_margin_ms'],exit_margin_ms=p['exit_margin_ms'],
            guards=e['guards'],mode_before=st['mode'],mode_after=d['hysteresis_mode'],tokens_before=st['tokens'],tokens_refilled=tokens,tokens_after=d['tokens'],
            cooldown_until_perf_seconds=st['next_g_time'],fallback_until_perf_seconds=st['fallback_until'],
            cooldown_remaining_seconds=max(0.,st['next_g_time']-now),fallback_remaining_seconds=max(0.,st['fallback_until']-now),
            G_completed_s=(r['completion_time_ns']-start)/1e9,G_service_ms=r['worker_latency_ms'],G_worker_wait_ms=r['worker_queue_wait_ms'],
            G_E2E_ms=r['e2e_latency_ms'],correct=r['worker_response'].get('correct'),outcome_scope='Completion/service/correct are retrospective, not decision inputs'))
    assert len(diversions)==67;csvsave('diversion_timeline.csv',diversions)
    guarddoc=dict(denominator=7005,actual_primary_reasons=dict(reasons),mutually_exclusive_primary_reasons_sum=sum(reasons.values()),
        guards={k:dict(PASS=v['PASS'],FAIL=v['FAIL'],UNKNOWN=0,denominator=7005) for k,v in guards.items()},
        cost_eligible_by_actual_hysteresis=sum(eligible_reasons.values()),primary_reasons_given_cost_eligible=dict(eligible_reasons),
        nonexclusive_guard_failures_given_cost_eligible=dict(eligible_guard_fail),
        G_empty_and_entry_cost_pass=empty_entry_pass,full_policy_replay_verified=True,
        warning='Guard failures overlap and must not be summed. Entry margin only required to enter mode; post-update mode AND gap>exit is actual final cost condition.',
        inputs='Actual arrival-time journal, only prior committed and harvested RESULTs; no native busy states or future completions in replay')
    save('dynamic_guard_breakdown.json',guarddoc)
    routing_bins=[]
    for width in [30,60]:
        for left in range(0,600,width):
            right=left+width;evs=[e for e in target_events if left<=(e['time_ns']-start)/1e9<right]
            dvs=[d for d in diversions if left<=d['decision_s']<right];completed=[r for r in drs if left<=(r['completion_time_ns']-start)/1e9<right]
            gg=[r for r in drs if r['model']==G];ss=[r for r in drs if r['model']==S]
            hi=integrate(gg,start+left*1000000000,start+right*1000000000,'enqueued_time_ns','_observable_completion_ns')
            routing_bins.append(dict(left_s=left,right_s=right,width_s=width,LID_arrival_decisions=len(evs),G_diversions=len(dvs),
                actual_diversion_rate_rps=len(dvs)/width,reasons=dict(collections.Counter(e['decision']['reason'] for e in evs)),
                G_host_observable_pending_empty_seconds=width-hi['busy_seconds'],G_native_idle_seconds=width-busy_seconds(gg,left,right),
                S_native_busy_seconds=busy_seconds(ss,left,right),S_ASR_busy_seconds=busy_seconds([r for r in ss if r['task']=='asr'],left,right),
                S_LID_service_completed=stats([r['worker_latency_ms'] for r in completed if r['model']==S and r['task']=='lid']),
                G_LID_service_completed=stats([r['worker_latency_ms'] for r in completed if r['model']==G and r['task']=='lid']),
                S_wait_completed=stats([r['worker_queue_wait_ms'] for r in completed if r['model']==S]),
                G_wait_completed=stats([r['worker_queue_wait_ms'] for r in completed if r['model']==G]),
                native_idle_scope='Retrospective handler idle, not GPU utilization or online ACK'))
    csvsave('routing_30_60s_windows.csv',routing_bins)
    # Actual overlap statistics and G-background waits: both descriptive, duration-selection biased.
    givs=[(r['_native'][0],r['_native'][1],r['request_id']) for r in drs if r['model']==G and r['task']=='lid']
    overlaps=[];strata=collections.defaultdict(list)
    for r in drs:
        a,b=r['_native'];pairs=[(max(a,c),min(b,d),rid) for c,d,rid in givs if rid!=r['request_id'] and max(a,c)<min(b,d)]
        if r['model']==S:
            for c,d,rid in pairs:overlaps.append(dict(S_request_id=r['request_id'],G_request_id=rid,S_task=r['task'],overlap_ms=(d-c)*1000))
            strata[(r['task'],bool(pairs))].append(r)
    assert len(overlaps)==403
    csvsave('execution_overlap_pairs.csv',overlaps)
    save('overlap_associations.json',{t+'|'+str(o):dict(n=len(rr),service=stats([r['worker_latency_ms'] for r in rr]),
        queue=stats([r['worker_queue_wait_ms'] for r in rr])) for (t,o),rr in strata.items()})
    metadata=dict(protected_input_files=protected,formal_target_decisions=7005,diversions=67,
        G_diversion_by_60s=[r['G_diversions'] for r in routing_bins if r['width_s']==60],first_diversion_s=diversions[0]['decision_s'],last_diversion_s=diversions[-1]['decision_s'],
        tail_diversions=sum(d['decision_s']>=450 for d in diversions),overlap_pairs=403,
        G_LID_correct=sum(r['worker_response'].get('correct') is True for r in drs if r['model']==G and r['task']=='lid'),
        G_LID_Wilson95=wilson(61,67),G_LID_unique_sample_ids=len({r['sample_id'] for r in drs if r['model']==G and r['task']=='lid'}),
        G_LID_closed_set_field_count=sum('valid_closed_set_output' in r['worker_response'] for r in drs if r['model']==G and r['task']=='lid'),
        prediction_error_events=dynraw['phases']['measurement']['prediction_errors'],
        evidence='READ_ONLY_OBSERVATIONAL_NOT_CAUSAL')
    save('audit_findings.json',metadata)
    plot_pending(timelines);plot_diversions(diversions)
    verify_inputs(True)
    print(json.dumps(dict(counts=reconciliation,stability={k:v['formal_tail_only'] for k,v in reproduction.items()},
        primary_reasons=dict(reasons),cost_eligible=guarddoc['cost_eligible_by_actual_hysteresis'],cost_eligible_reasons=dict(eligible_reasons),
        G60s=metadata['G_diversion_by_60s'],worker_busy={k:v['worker_busy'] for k,v in run_analysis.items()},G_Wilson95=metadata['G_LID_Wilson95']),ensure_ascii=False))

def plot_pending(timelines):
    from html import escape
    width=1550;height=980;s=['<svg xmlns="http://www.w3.org/2000/svg" width="1550" height="980" viewBox="0 0 1550 980">',
        '<rect width="1550" height="980" fill="white"/><style>text{font-family:sans-serif;font-size:14px}</style>',
        '<text x="40" y="25">Descriptive separate runs: submitted B (blue), enqueued unfinished H (green dashed); worker waiting queue UNKNOWN</text>']
    for col,(name,groups) in enumerate(timelines.items()):
        for row,key in enumerate(['ALL','S','G']):
            x=80+col*760;y=80+row*290;w=620;h=205;curve=groups[key];maxy=max(1,max(v for _,v in curve['B']),max(v for _,v in curve['H']))
            def X(t):return x+w*t/603.
            def Y(b):return y+h-h*b/maxy
            s.append(f'<text x="{x}" y="{y-17}">{escape(name)} / {key}; end drain {600+curve["drain_seconds"]:.6f}s</text>')
            s.append(f'<rect x="{X(600)}" y="{y}" width="{X(603)-X(600)}" height="{h}" fill="#ffe1bb"/>')
            for tick in [0,150,300,450,600]:s.append(f'<line x1="{X(tick)}" y1="{y}" x2="{X(tick)}" y2="{y+h}" stroke="#ddd"/><text x="{X(tick)-10}" y="{y+h+23}">{tick}</text>')
            for value in [0,maxy/2,maxy]:s.append(f'<text x="{x-45}" y="{Y(value)+5}">{value:.1f}</text>')
            for label,color,dash in [('B','#1f65b0',''),('H','#17905c',' stroke-dasharray="5,3"')]:
                pts=[];old=0
                for t,b in curve[label]:pts.extend([f'{X(t):.3f},{Y(old):.3f}',f'{X(t):.3f},{Y(b):.3f}']);old=b
                pts.append(f'{X(600+curve["drain_seconds"]):.3f},{Y(old):.3f}')
                s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1"{dash}/>')
            s.append(f'<line x1="{X(600)}" y1="{y}" x2="{X(600)}" y2="{y+h}" stroke="#a33"/>')
    s.append('<text x="40" y="963">Relative formal-window seconds; drain shaded. S-only H uses reader proxy; Dynamic H uses collector commit. Native times never policy inputs.</text></svg>')
    (OUT/'pending_trajectories.svg').write_text(''.join(s))

def plot_diversions(rows):
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="330" viewBox="0 0 1500 330"><rect width="1500" height="330" fill="white"/><style>text{font-family:sans-serif;font-size:14px}</style>',
       '<text x="40" y="25">67 actual LID diversions: S pending at decision; line ends at actual G response (retrospective)</text>']
    maxy=max(r['S_pending'] for r in rows)
    def X(t):return 70+t*1360/603
    def Y(v):return 270-v*215/maxy
    for tick in range(0,601,60):s.append(f'<line x1="{X(tick)}" y1="45" x2="{X(tick)}" y2="270" stroke="#ddd"/><text x="{X(tick)-10}" y="295">{tick}</text>')
    for r in rows:
        x=X(r['decision_s']);y=Y(r['S_pending']);s.append(f'<line x1="{x}" y1="{y}" x2="{X(r["G_completed_s"])}" y2="{y}" stroke="#777"/><circle cx="{x}" cy="{y}" r="3" fill="#bc402e"><title>{r["request_id"]}; S={r["S_pending"]}; gap={r["gap_ms"]:.2f}ms</title></circle>')
    s.append('<text x="40" y="320">Seconds since formal start. See diversion_timeline.csv for all states, guards, costs and outcomes.</text></svg>')
    (OUT/'diversion_timeline.svg').write_text(''.join(s))

if __name__=='__main__':main()
