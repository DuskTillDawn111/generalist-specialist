from common_replication import *
sys.path.append(str(OLD))
import report as original_report
from scipy.stats import t as student_t
from module26.common import cohort

def decide_gate(complete,valid,both_paths,deltas,cis):
    if not complete or not valid:return 'REPLICATION_INVALID'
    if not both_paths or len(deltas)!=6 or not all(x>.01 for x in deltas):return 'DYNAMIC_NOT_REPRODUCED'
    if not all(c['lower']>0 for c in cis):return 'DYNAMIC_POSITIVE_INCONCLUSIVE'
    return 'DYNAMIC_POSITIVE_REPLICATED'

def paired_stats(values):
    n=len(values)
    if n!=3:return dict(n=n,mean=None,std=None,lower=None,upper=None)
    mean=float(np.mean(values));std=float(np.std(values,ddof=1));half=float(student_t.ppf(.975,2)*std/np.sqrt(3))
    return dict(n=3,mean=mean,std=std,lower=mean-half,upper=mean+half,method='Paired Student t, df=2; independent seeds only')

def describe_run(raw,f):
    s=raw['slot'];rid=s['run_id'];v=original_report.evaluate(raw,f)
    ph=raw.get('phases',{}).get('measurement');rs=ph['request_records'] if ph else []
    done=[r for r in rs if r['status']=='completed'];win=ph['window'] if ph else {}
    base=dict(run_id=rid,group=s['group'],seed=s['seed'],arm=s['arm'],order=s['run_order'])
    target=[r for r in done if r['task']=='lid'];ng=sum(r['model']==f['parameters']['G_worker'] for r in target)
    lat=[r['routing_decision']['overhead_us'] for r in target if 'routing_decision' in r]
    accepted=sum(r.get('enqueued_time_ns') is not None for r in rs)
    within=sum(r['completion_time_ns']<=win['end_time_ns'] for r in done) if win else 0
    run=dict(**base,status=raw['status'],eligible=v['eligible'],U=v['U'],worker_states=v['worker_states'],
        system_state=ph['stability']['stability_state_v2'] if ph else None,actual_G=ng,actual_S=len(target)-ng,
        G_fraction=ng/len(target) if target else None,formal_seconds=win.get('planned_duration_seconds'),
        offered=len(rs),accepted=accepted,completed=len(done),completed_in_window=within,
        offered_qps=len(rs)/2400,accepted_qps=accepted/2400,completed_qps=len(done)/2400,within_qps=within/2400,
        pooled_mean_ms=v.get('pooled_mean_ms'),pooled_p95_ms=v.get('pooled_p95_ms'),
        overhead_mean_us=float(np.mean(lat)) if lat else None,overhead_p95_us=float(np.percentile(lat,95)) if lat else None,
        safe_memory_mib=v.get('safe_memory_mib'),sampled_peak_mib=v.get('sampled_peak_mib'),
        peak_scope='NVML 0.5s samples; true instantaneous peak unavailable; native allocator shutdown peaks separate',
        budget_mib=f['budget_mib'],drain_seconds=win.get('drain_seconds'),errors=raw['errors'],
        valid_session=raw.get('valid_session'),capability=v['capability'],measured_replicas=v['measured_counts'],
        semantic_valid=v['semantic_valid'],memory_ok=v['memory_ok'],recovery_confirmed=v['recovery_confirmed'],
        raw=ref(OUT/'raw_runs'/(rid+'.json')))
    tasks=[];workers=[];stability=[];decomposition=[];decisions=[];environment=[];tails=[]
    for t,q in s['lambda_rps'].items():
        rr=[r for r in done if r['task']==t];responses=[r['worker_response'] for r in rr]
        quality={}
        for k in ['correct','wer','cer','exact_match']:
            a=[r[k] for r in responses if k in r]
            quality[k+'_n']=len(a);quality[k+'_mean']=float(np.mean(a)) if a else None
        tasks.append(dict(**base,task=t,nominal_qps=q,offered=sum(r['task']==t for r in rs),
            accepted=sum(r['task']==t and r.get('enqueued_time_ns') is not None for r in rs),completed=len(rr),
            within=sum(r['completion_time_ns']<=win['end_time_ns'] for r in rr) if win else 0,
            e2e=stats(rr,'e2e_latency_ms'),service=stats(rr,'worker_latency_ms'),queue=stats(rr,'worker_queue_wait_ms'),
            eligible=v['eligible'],quality_admission='Original quality policy unchanged; subset diagnostic only',**quality))
        if t in v.get('per_task',{}):
            p=v['per_task'][t];weight=q/sum(s['lambda_rps'].values())
            decomposition.append(dict(**base,component=t,offered_weight=weight,
                mean_contribution=-f['weights']['L']*weight*p['mean_ms']/f['refs'][t],
                p95_contribution=-f['weights']['P']*weight*p['p95_ms']/f['refs'][t],eligible=v['eligible']))
        if rr:
            values=sorted(rr,key=lambda r:r['e2e_latency_ms']);a=np.array([r['e2e_latency_ms'] for r in values]);n=len(a)
            # These diagnostics do not replace frozen U, remove rows from U, or gate success.
            top=max(1,int(np.ceil(n*.01)));trim=a[:-top]
            tails.append(dict(**base,task=t,n=n,top1pct_count=top,top1pct_latency_sum_share=float(a[-top:].sum()/a.sum()),
                original_mean=float(a.mean()),original_p95=float(np.percentile(a,95)),
                diagnostic_trimmed_mean=float(trim.mean()) if len(trim) else None,
                diagnostic_trimmed_p95=float(np.percentile(trim,95)) if len(trim) else None,
                p95_neighbor_request_ids=[r['request_id'] for r in values[max(0,int(.95*n)-3):min(n,int(.95*n)+4)]],
                max_request_id=values[-1]['request_id'],max_ms=float(a[-1]),scope='SENSITIVITY_ONLY; no utility or sample-count changes'))
    for k in ['T','L','P','M']:
        decomposition.append(dict(**base,component=k,norm=v.get(k+'_norm'),contribution=v.get(k+'_contribution'),eligible=v['eligible']))
    if ph:
        scopes=[('system',None,ph['stability_evidence'],ph['stability'])]
        for w,m in raw['replica_to_family'].items():
            rr=[r for r in rs if r['model']==w]
            temp=dict(request_records=rr,measurement_window=win,status='completed',errors=raw['errors'],drain_timeout_seconds=60,worker_identities={w:raw['worker_identities'][w]})
            ev=reconstruct(temp,{w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms'])},DEFAULT_THRESHOLDS)
            scopes.append(('worker',w,ev,classify(ev,DEFAULT_THRESHOLDS)))
            ww=[r for r in done if r['model']==w];busy=0
            for r in ww:
                delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns'];res=r['worker_response']
                busy+=max(0,min(res['end_ns']+delta,win['end_time_ns'])-max(res['start_ns']+delta,win['start_time_ns']))/1e9
            for t in sorted({r['task'] for r in ww}):
                tr=[r for r in ww if r['task']==t]
                workers.append(dict(**base,worker=w,task=t,n=len(tr),service=stats(tr,'worker_latency_ms'),
                    queue=stats(tr,'worker_queue_wait_ms'),e2e=stats(tr,'e2e_latency_ms'),native_busy_fraction=busy/2400))
        for scope,w,ev,cl in scopes:
            stability.append(dict(**base,scope=scope,worker=w,state=cl['stability_state_v2'],hard_reasons=cl['hard_reasons'],
                peak_backlog=ev['peak_backlog_requests'],end_backlog=ev['end_backlog_requests'],
                tail_slope_rps=ev['tail_backlog_slope'],tail_growth=ev['tail_backlog_growth'],
                increasing_fraction=ev['tail_increasing_fraction'],final_backlog=ev['final_backlog_after_drain'],
                completion_ratio=ev['completion_ratio'],drain_seconds=win['drain_seconds'],eligible=v['eligible']))
    for r in rs:
        if 'routing_decision' in r:
            d=r['routing_decision'];assert d['latest_observed_completion_time_ns']<=d['decision_time_ns']
            decisions.append(dict(**base,request_id=r['request_id'],sample_id=r['sample_id'],request_status=r['status'],
                offset_seconds=r['frozen_offset_seconds'],actual_route=r['model'],service_ms=r.get('worker_latency_ms'),
                queue_ms=r.get('worker_queue_wait_ms'),e2e_ms=r.get('e2e_latency_ms'),**d))
    for row in raw.get('telemetry',{}).get('samples',[]):
        environment.append(dict(**base,**{k:row.get(k) for k in ['time_ns','unix_ns','phase','temperature_gpu_c','power_w',
            'sm_clock_mhz','memory_clock_mhz','performance_state','used_mib','owned_used_mib','gpu_util_percent','foreign_pids',
            'host_available_RAM_bytes','safety_host_outstanding','safety_oldest_pending_seconds']}))
    save('run_summaries/'+rid+'.json',dict(run=run,evaluation=v,tasks=tasks,workers=workers,stability=stability,
        decomposition=decomposition,tails=tails,postwarm_temperature=float(np.mean([r['temperature_gpu_c'] for r in raw['postwarm_idle']])) if raw.get('postwarm_idle') else None))
    csvsave('per_run/'+rid+'_routing.csv',decisions,['run_id','request_id','actual_route','reason'])
    csvsave('per_run/'+rid+'_environment.csv',environment,['run_id','phase','temperature_gpu_c'])
    if ph:csvsave('per_run/'+rid+'_backlog.csv',curve(raw),['run_id','offset_seconds'])
    return v

def combine_csv(name,paths,required_fields):
    # Stream full decision/environment/backlog CSVs instead of retaining nine raw runs in RAM.
    fields=list(required_fields)
    for p in paths:
        with p.open() as stream:
            fields.extend(k for k in next(csv.reader(stream),[]) if k not in fields)
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for p in paths:
            with p.open() as source:
                for row in csv.DictReader(source):writer.writerow(row)

def write_report(f,fatal=None,final=False):
    csv.field_size_limit(128*1024*1024)
    summaries=[json.loads(p.read_text()) for p in sorted((OUT/'run_summaries').glob('*.json'))]
    byid={r['run']['run_id']:r for r in summaries};runs=[]
    for slot in f['independent_slots']:
        runs.append(byid[slot['run_id']]['run'] if slot['run_id'] in byid else dict(run_id=slot['run_id'],group=slot['group'],seed=slot['seed'],arm=slot['arm'],order=slot['run_order'],status='NOT_EXECUTED' if final else 'PENDING',U=None,eligible=False))
    csvsave('independent_gpu_runs.csv',runs,['run_id','group','seed','arm','U','eligible'])
    for name,key in [('per_task_latency.csv','tasks'),('worker_latency_analysis.csv','workers'),('backlog_stability.csv','stability'),('utility_decomposition.csv','decomposition'),('tail_sensitivity.csv','tails')]:
        csvsave(name,[row for r in summaries for row in r[key]],['run_id'])
    for name,suffix,fields in [('routing_decision_analysis.csv','routing',['run_id','request_id','actual_route','reason']),('environment_control.csv','environment',['run_id','phase','temperature_gpu_c']),('backlog_timeseries.csv','backlog',['run_id','offset_seconds'])]:
        combine_csv(name,sorted((OUT/'per_run').glob('*_'+suffix+'.csv')),fields)
    pairs=[];interference=[];blocks=[]
    for group in [1,2,3]:
        ids={a:f'G{group}_{a}' for a in ['s_only','fixed10','dynamic']}
        if not all(rid in byid for rid in ids.values()):continue
        alllegal=all(byid[rid]['run']['eligible'] for rid in ids.values())
        paths={a:OUT/'raw_runs'/(rid+'.json') for a,rid in ids.items()}
        raws={a:json.loads(p.read_text()) for a,p in paths.items()}
        if not all(r.get('phases',{}).get('measurement') and all(q['status']=='completed' for q in r['phases']['measurement']['request_records']) for r in raws.values()):
            continue
        dy=raws['dynamic'];dv=byid[ids['dynamic']]['evaluation'];dr=byid[ids['dynamic']]['run']
        for arm in ['s_only','fixed10']:
            ct=raws[arm];cv=byid[ids[arm]]['evaluation'];same=cohort(dy)==cohort(ct)
            td=abs(byid[ids['dynamic']]['postwarm_temperature']-byid[ids[arm]]['postwarm_temperature'])
            delta=dv['U']-cv['U'] if alllegal and same else None
            pair=dict(group=group,seed=dy['slot']['seed'],dynamic_run=ids['dynamic'],control_run=ids[arm],control_arm=arm,
                all_three_legal_stable=alllegal,same_trace_inputs=same,thermal_difference_c=td,thermal_comparable=td<=5,
                delta_U=delta,G=dr['actual_G'],S=dr['actual_S'],G_fraction=dr['G_fraction'],
                pooled_mean_delta_ms=dv.get('pooled_mean_ms',0)-cv.get('pooled_mean_ms',0),
                pooled_p95_delta_ms=dv.get('pooled_p95_ms',0)-cv.get('pooled_p95_ms',0))
            if delta is not None:
                for k in ['T','L','P','M']:pair[k+'_delta']=dv[k+'_contribution']-cv[k+'_contribution']
                pair['task_contributions']={t:dict(mean_delta=-f['weights']['L']*q/11.5*(dv['per_task'][t]['mean_ms']-cv['per_task'][t]['mean_ms'])/f['refs'][t],p95_delta=-f['weights']['P']*q/11.5*(dv['per_task'][t]['p95_ms']-cv['per_task'][t]['p95_ms'])/f['refs'][t]) for t,q in f['rates'].items()}
                assert abs(delta-sum(pair[k+'_delta'] for k in ['T','L','P','M']))<1e-10
            pairs.append(pair)
            cr={r['request_id']:r for r in ct['phases']['measurement']['request_records']}
            rr=dy['phases']['measurement']['request_records']
            for w in dy['replica_to_family']:
                for t in f['rates']:
                    common=[(r,cr[r['request_id']]) for r in rr if r['task']==t and r['model']==w and cr[r['request_id']]['model']==w]
                    if not common:continue
                    interference.append(dict(group=group,control_arm=arm,worker=w,task=t,n=len(common),
                        service_delta_ms=float(np.mean([r['worker_latency_ms']-c['worker_latency_ms'] for r,c in common])),
                        queue_delta_ms=float(np.mean([r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in common])),
                        e2e_delta_ms=float(np.mean([r['e2e_latency_ms']-c['e2e_latency_ms'] for r,c in common])),
                        scope='Matched inputs on shared actual route; routing/thermal/coexistence bundled; not an unexecuted counterfactual'))
            # Eight 300s arrival-cohort blocks, matched inputs; descriptive only.
            for b in range(8):
                recs={a:[r for r in raws[a]['phases']['measurement']['request_records'] if b*300<=r['frozen_offset_seconds']<(b+1)*300] for a in ['dynamic',arm]}
                if not alllegal:continue
                def lp(recs):
                    return sum(-f['weights'][k]*q/11.5*(float(np.mean([r['e2e_latency_ms'] for r in recs if r['task']==t])) if k=='L' else float(np.percentile([r['e2e_latency_ms'] for r in recs if r['task']==t],95)))/f['refs'][t] for t,q in f['rates'].items() for k in ['L','P'])
                if all({r['task'] for r in rs}==set(f['rates']) for rs in recs.values()):blocks.append(dict(group=group,control_arm=arm,block=b,latency_component_delta=lp(recs['dynamic'])-lp(recs[arm]),scope='Matched arrival blocks; dependent within run, not extra independent samples'))
        del raws
    csvsave('paired_utility_comparison.csv',pairs,['group','control_arm','delta_U'])
    csvsave('worker_interference_analysis.csv',interference,['group','control_arm','worker','task'])
    csvsave('block_sensitivity.csv',blocks,['group','control_arm','block'])
    cis=[dict(control_arm=arm,**paired_stats([p['delta_U'] for p in pairs if p['control_arm']==arm and p['delta_U'] is not None])) for arm in ['s_only','fixed10']]
    complete=len(summaries)==9
    valid=bool(complete and not fatal and all(r['eligible'] for r in runs) and len(pairs)==6 and all(p['same_trace_inputs'] and p['thermal_comparable'] for p in pairs))
    both=bool(complete and all(r['actual_G']>0 and r['actual_S']>0 for r in runs if r['arm']=='dynamic'))
    deltas=[p['delta_U'] for p in pairs if p['delta_U'] is not None]
    gate=decide_gate(complete,valid,both,deltas,cis)
    evidence=dict(module='29',gate=gate,final=final,completed_and_stopped=final,executed=[r['run']['run_id'] for r in summaries],
        all_nine_legal_stable=valid,all_dynamic_both_paths=both,paired_statistics=cis,pairs=pairs,fatal=fatal,
        all_six_deltas_over_threshold=bool(len(deltas)==6 and all(x>.01 for x in deltas)),
        pooled_mean_improves_all_pairs=bool(len(pairs)==6 and all(p['pooled_mean_delta_ms']<0 for p in pairs)),
        pooled_p95_improves_all_pairs=bool(len(pairs)==6 and all(p['pooled_p95_delta_ms']<0 for p in pairs)),
        development_excluded=True,independent_n=3,small_sample='t intervals assume approximately normal paired seed effects; n=3 cannot establish broad robustness; minimum exact one-sided sign p=0.125',
        stage_e_triggered=bool(gate=='DYNAMIC_POSITIVE_REPLICATED'),causal_scope='Matched trajectory/input controlled routing comparison; thermal/run noise remains. Retrospective queue associations do not isolate per-event causality.',
        optimizer_modified=False,next_module_automatic=False,model_pool_search=False)
    save('module29_gate.json',evidence)
    lines=['# Module 29：Independent GPU Replication\n\n',f'状态：{"已停止" if final else "进行中"}。Gate：**{gate}**。完成 {len(summaries)}/9 个预注册 2400 秒窗口；三个独立种子，开发数据不计入统计。\n\n',
        '|组|方案|合法 U|G / S|系统状态|pooled mean / P95 ms|\n|---|---|---:|---:|---|---:|\n']
    for r in runs:lines.append(f'|{r["group"]}|{r["arm"]}|{r.get("U")}|{r.get("actual_G")} / {r.get("actual_S")}|{r.get("system_state",r["status"])}|{r.get("pooled_mean_ms")} / {r.get("pooled_p95_ms")}|\n')
    lines.append('\n|对照|独立配对 ΔU|均值|样本标准差|95% t 区间|\n|---|---|---:|---:|---|\n')
    for c in cis:lines.append(f'|{c["control_arm"]}|{[p["delta_U"] for p in pairs if p["control_arm"]==c["control_arm"]]}|{c["mean"]}|{c["std"]}|[{c["lower"]}, {c["upper"]}]|\n')
    lines += [f'\n全部九方案合法稳定：{valid}；三个 Dynamic 均实际走 G 和 S：{both}；六个 ΔU 均 >+0.01：{evidence["all_six_deltas_over_threshold"]}。\n\n',
        f'系统 pooled mean 全部改善：{evidence["pooled_mean_improves_all_pairs"]}；P95 全部改善：{evidence["pooled_p95_improves_all_pairs"]}。逐任务加权 P95 才参与 U，pooled P95 单独报告。\n\n',
        'n=3 配对 t 区间依赖效应近似正态；最小单侧符号检验 p=0.125。300 秒块和尾部删除敏感性仅诊断，不能增加独立样本量，也不能替代原效用。原稳定性标准完整保留，不因最终排空自动认定窗口稳定；非法 U 保持 NULL。\n\n',
        '服务、排队和背景任务分项见 worker_latency_analysis.csv、worker_interference_analysis.csv 与 utility_decomposition.csv。动态选择的实时状态、保护/背压/回退和计算开销见 routing_decision_analysis.csv。共置变化是匹配输入运行关联，不唯一归因于物理 GPU 干扰；若未达到可信复现门槛，Stage E 不启动。\n\n',
        f'技术故障与停止：{json.dumps(fatal,ensure_ascii=False)}。零 GPU 重试；负效用不换种子、不改窗口或策略。NVML 峰值仅 0.5 秒采样高水位，真实瞬时峰值不可用；原生 allocator 输出另存，不编造峰值。\n\n',
        '模型、权重哈希、软件环境和完整轨迹已冻结。历史完整性另报；本 Module 不修改优化器，不继续搜索模型池或启动下一 Module。是否值得集成取决于独立结果，探索期正结果不能作为本轮成功证据。\n']
    (OUT/'module29_summary.md').write_text(''.join(lines))
    return evidence
