"""Separate capacity and legal utility evidence; no unstable finite-U contrasts."""
from audit_followup import *
import time

def write_reports(raws,f,fatal=None,final=False):
    import report as original_report
    runs=[];tasks=[];workers=[];stability=[];utilities=[];decisions=[];thermals=[];interference=[];backlog=[];decomp=[]
    baseline=json.loads((OLD/'raw_runs/C_steady_s_only.json').read_text())
    backlog.extend(curve(baseline))
    groups=collections.defaultdict(dict)
    evaluations={}
    for raw in raws:
        slot=raw['slot'];rid=slot['run_id'];scene=slot['scenario'];arm=slot['arm'];groups[scene][arm]=raw
        v=original_report.evaluate(raw,f);evaluations[rid]=v
        ph=raw.get('phases',{}).get('measurement');rs=ph['request_records'] if ph else [];done=[r for r in rs if r['status']=='completed']
        wr=ph['window'] if ph else {};target=[r for r in done if r['task']=='lid'];ng=sum(r['model']==f['parameters']['G_worker'] for r in target)
        overhead=[r['routing_decision']['overhead_us'] for r in target if 'routing_decision' in r]
        ss=ph['stability']['stability_state_v2'] if ph else None
        mem=v.get('sampled_peak_mib');nvmltotal=max((s['used_mib'] for s in raw.get('telemetry',{}).get('samples',[])),default=None)
        base=dict(run_id=rid,scenario=scene,arm=arm,order=slot['run_order'],status=raw['status'])
        runs.append(dict(**base,eligible=v['eligible'],U=v['U'],system_stability=ss,worker_states=v['worker_states'],
            offered=v.get('offered'),completed=v.get('completed'),offered_qps=v.get('offered_qps'),completed_cohort_qps=v.get('completed_cohort_qps'),
            completed_in_window_qps=v.get('within_qps'),actual_G=ng,actual_S=len(target)-ng,actual_G_fraction=ng/len(target) if target else None,
            pooled_mean_ms=v.get('pooled_mean_ms'),pooled_p95_ms=v.get('pooled_p95_ms'),
            latency_comparison_legal=v['eligible'],latency_scope='FULL_MATCHED_COHORT_WITH_DRAIN' if v['eligible'] else 'INELIGIBLE_DIAGNOSTIC_NOT_UTILITY_OR_SUCCESS_COMPARISON',
            routing_overhead_mean_us=float(np.mean(overhead)) if overhead else None,routing_overhead_p95_us=float(np.percentile(overhead,95)) if overhead else None,
            overhead_scope='DYNAMIC_ROUTER_PLUS_HOST_OBSERVATION' if arm=='dynamic' else 'CONTROL_FRAMEWORK_DIAGNOSTIC',
            safe_idle_memory_mib=v.get('safe_memory_mib'),sampled_owned_peak_mib=mem,sampled_total_peak_mib=nvmltotal,
            peak_scope='NVML_0.5s_SAMPLED_HIGH_WATER; instantaneous hardware peak unavailable',budget_mib=f['budget_mib'],
            drain_seconds=wr.get('drain_seconds'),valid_session=raw['valid_session'],original_quality_capability=v['capability'],
            original_measured_replicas=v['measured_counts'],semantic_valid=v['semantic_valid'],memory_ok=v['memory_ok'],errors=raw['errors']))
        for k in ['T','L','P','M']:
            decomp.append(dict(**base,component=k,norm=v.get(k+'_norm'),contribution=v.get(k+'_contribution'),eligible=v['eligible'],scope='LEGAL' if v['eligible'] else 'INELIGIBLE_DIAGNOSTIC'))
        utilities.append(dict(**base,U=v['U'],eligible=v['eligible'],**{k:v.get(k) for k in ['T_norm','L_norm','P_norm','M_norm','T_contribution','L_contribution','P_contribution','M_contribution']},
            diagnostic_arithmetic_U=v.get('raw_arithmetic_U'),diagnostic_usable_for_ranking=False if not v['eligible'] else True,
            delta_dynamic_vs_S=None,delta_dynamic_vs_fixed=None,all_three_legal_stable=False))
        if ph:
            backlog.extend(curve(raw))
            e=ph['stability_evidence']
            stability.append(dict(**base,scope='system',worker=None,state=ss,original_hard_reasons=ph['stability']['hard_reasons'],
                peak_backlog=e['peak_backlog_requests'],end_backlog=e['end_backlog_requests'],tail_slope_rps=e['tail_backlog_slope'],
                tail_growth=e['tail_backlog_growth'],tail_increasing_fraction=e['tail_increasing_fraction'],final_backlog=e['final_backlog_after_drain'],
                completion_ratio=e['completion_ratio'],drain_seconds=wr['drain_seconds'],U=v['U']))
        for st in v['workers']:
            w=st['worker'];ww=[r for r in done if r['model']==w]
            stability.append(dict(**base,scope='worker',worker=w,state=st['state'],original_hard_reasons=st['hard_reasons'],peak_backlog=st['peak_backlog'],
                end_backlog=st['end_backlog'],tail_slope_rps=st['tail_slope_rps'],final_backlog=st['final_backlog'],U=v['U']))
            busy=0
            for r in ww:
                delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns'];res=r['worker_response']
                busy+=max(0,min(res['end_ns']+delta,wr['end_time_ns'])-max(res['start_ns']+delta,wr['start_time_ns']))/1e9
            for t in sorted(set(r['task'] for r in ww)):
                tr=[r for r in ww if r['task']==t]
                workers.append(dict(**base,worker=w,task=t,n=len(tr),service=stats(tr,'worker_latency_ms'),wait=stats(tr,'worker_queue_wait_ms'),
                    e2e=stats(tr,'e2e_latency_ms'),native_worker_busy_fraction=busy/600,stability=st['state']))
        for t,q in slot['lambda_rps'].items():
            rr=[r for r in done if r['task']==t];res=[r['worker_response'] for r in rr]
            quality={}
            for key in ['correct','wer','cer','exact_match']:
                a=[r[key] for r in res if key in r]
                quality[key+'_n']=len(a);quality[key+'_mean']=float(np.mean(a)) if a else None
            tasks.append(dict(**base,task=t,nominal_offered_qps=q,offered=sum(r['task']==t for r in rs),completed=len(rr),
                in_window_completed=sum(r['completion_time_ns']<=wr['end_time_ns'] for r in rr) if wr else 0,
                e2e=stats(rr,'e2e_latency_ms'),service=stats(rr,'worker_latency_ms'),wait=stats(rr,'worker_queue_wait_ms'),
                eligible=v['eligible'],quality_scope='ORIGINAL_ADMISSION_UNCHANGED; sampled quality diagnostic; missing not imputed',**quality))
            if t in v.get('per_task',{}):
                p=v['per_task'][t];weight=q/sum(slot['lambda_rps'].values())
                decomp.append(dict(**base,component=t,offered_weight=weight,mean_contribution=-f['weights']['L']*weight*p['mean_ms']/f['refs'][t],
                    p95_contribution=-f['weights']['P']*weight*p['p95_ms']/f['refs'][t],eligible=v['eligible']))
        for r in rs:
            if 'routing_decision' in r:
                d=r['routing_decision'];assert d['latest_observed_completion_time_ns']<=d['decision_time_ns']
                decisions.append(dict(**base,request_id=r['request_id'],task=r['task'],sample_id=r['sample_id'],actual_route=r['model'],
                    request_status=r['status'],service_ms=r.get('worker_latency_ms'),wait_ms=r.get('worker_queue_wait_ms'),e2e_ms=r.get('e2e_latency_ms'),**d))
        for s in raw.get('telemetry',{}).get('samples',[]):
            thermals.append(dict(**base,**{k:s.get(k) for k in ['time_ns','unix_ns','phase','temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','performance_state','power_w','used_mib','owned_used_mib','gpu_util_percent','foreign_pids','host_available_RAM_bytes','worker_CPU_core_percent','safety_host_outstanding','safety_oldest_pending_seconds']}))
    pairs=[]
    from module26.common import cohort
    for scene,arms in groups.items():
        alllegal=set(arms)=={'s_only','fixed10','dynamic'} and all(evaluations[r['slot']['run_id']]['eligible'] for r in arms.values())
        for a in ['s_only','fixed10']:
            if 'dynamic' in arms and a in arms and all(arms[k].get('phases',{}).get('measurement') for k in ['dynamic',a]):
                dy=arms['dynamic'];ct=arms[a];dv=evaluations[dy['slot']['run_id']];cv=evaluations[ct['slot']['run_id']]
                same=cohort(dy)==cohort(ct);assert same
                legal=dv['eligible'] and cv['eligible']
                # LOW: no pairwise utility acceptance unless all three are legal.
                compare=legal and (alllegal if scene=='LOW' else a=='fixed10' or alllegal)
                delta=dv['U']-cv['U'] if compare else None
                td=abs(np.mean([r['temperature_gpu_c'] for r in dy['postwarm_idle']])-np.mean([r['temperature_gpu_c'] for r in ct['postwarm_idle']]))
                pairs.append(dict(scenario=scene,dynamic=dy['slot']['run_id'],control=ct['slot']['run_id'],control_arm=a,same_trace_inputs=same,
                    both_legal_stable=legal,all_three_legal_stable=alllegal,delta_U=delta,thermal_difference_c=float(td),thermal_comparable=td<=5,
                    reason='LEGAL_STABLE_COMPARISON' if compare else 'U_NULL_OR_INCOMPLETE_THREE_ARM_GROUP'))
                for row in utilities:
                    if row['run_id']==dy['slot']['run_id']:
                        row['delta_dynamic_vs_S' if a=='s_only' else 'delta_dynamic_vs_fixed']=delta;row['all_three_legal_stable']=alllegal
        if 's_only' in arms:
            control=arms['s_only']; cr={r['request_id']:r for r in control.get('phases',{}).get('measurement',{}).get('request_records',[]) if r['status']=='completed'}
            for a in ['fixed10','dynamic']:
                if a not in arms:continue
                raw=arms[a]
                for w in raw['replica_to_family']:
                    for t in raw['slot']['lambda_rps']:
                        common=[(r,cr[r['request_id']]) for r in raw.get('phases',{}).get('measurement',{}).get('request_records',[]) if r['status']=='completed' and r['request_id'] in cr and r['task']==t and r['model']==w and cr[r['request_id']]['model']==w]
                        if not common:continue
                        assert all(r['sample_id']==c['sample_id'] and r['frozen_offset_seconds']==c['frozen_offset_seconds'] for r,c in common)
                        interference.append(dict(scenario=scene,arm=a,run_id=raw['slot']['run_id'],control_run=control['slot']['run_id'],worker=w,task=t,matched_inputs=len(common),
                            control_service_mean_ms=float(np.mean([c['worker_latency_ms'] for r,c in common])),arm_service_mean_ms=float(np.mean([r['worker_latency_ms'] for r,c in common])),
                            service_delta_ms=float(np.mean([r['worker_latency_ms']-c['worker_latency_ms'] for r,c in common])),
                            control_wait_mean_ms=float(np.mean([c['worker_queue_wait_ms'] for r,c in common])),arm_wait_mean_ms=float(np.mean([r['worker_queue_wait_ms'] for r,c in common])),
                            wait_delta_ms=float(np.mean([r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in common])),
                            causal_scope='MATCHED_INPUT_OBSERVATION; physical GPU/CPU/thermal causes not isolated'))
    attempted={r['run_id'] for r in runs}
    for s in f['exploration_slots']:
        if s['run_id'] not in attempted:
            runs.append(dict(run_id=s['run_id'],scenario=s['scenario'],arm=s['arm'],order=s['run_order'],status='NOT_EXECUTED' if final else 'PENDING',U=None,eligible=False,errors=fatal))
    csvsave('gpu_routing_comparison.csv',runs)
    csvsave('per_task_latency.csv',tasks)
    csvsave('worker_metrics.csv',workers)
    csvsave('worker_interference_analysis.csv',interference,['scenario','arm','run_id','control_run','worker','task','matched_inputs'])
    csvsave('stability_comparison.csv',stability)
    csvsave('system_utility_comparison.csv',utilities)
    csvsave('utility_pair_comparison.csv',pairs,['scenario','dynamic','control','delta_U'])
    csvsave('utility_decomposition.csv',decomp)
    csvsave('routing_decision_trace.csv',decisions)
    csvsave('GPU_thermal_state.csv',thermals)
    csvsave('backlog_timeseries.csv',backlog)
    capacity='DYNAMIC_EVIDENCE_GAP';utility='DYNAMIC_EVIDENCE_GAP'
    high=groups.get('HIGH',{});low=groups.get('LOW',{})
    def val(scene,a):return evaluations[groups[scene][a]['slot']['run_id']]
    if 's_only' in high and 'dynamic' in high:
        cv,dv=val('HIGH','s_only'),val('HIGH','dynamic')
        if cv['hard_unstable'] and dv['eligible']:capacity='DYNAMIC_CAPACITY_IMPROVEMENT'
        elif cv['eligible'] and dv['eligible']:capacity='HIGH_STABILITY_EVIDENCE_ONLY'
        elif dv['hard_unstable']:capacity='DYNAMIC_NO_CAPACITY_IMPROVEMENT'
    lowpairs=[p for p in pairs if p['scenario']=='LOW']
    lowall=set(low)=={'s_only','fixed10','dynamic'} and all(val('LOW',a)['eligible'] for a in low)
    positive=False
    if lowall and len(lowpairs)==2:
        drow=next(r for r in runs if r['scenario']=='LOW' and r['arm']=='dynamic')
        counts_ok=min(t['completed'] for t in tasks if t['scenario']=='LOW')>=72
        positive=bool(drow['actual_G']>0 and drow['actual_S']>0 and counts_ok and all(p['delta_U'] is not None and p['delta_U']>.01 and p['thermal_comparable'] for p in lowpairs))
        utility='DYNAMIC_EXPLORATORY_POSITIVE_UTILITY' if positive else 'DYNAMIC_NO_POSITIVE_UTILITY'
    gate=dict(module='28R',final=final,completed_and_stopped=final,at_unix_ns=time.time_ns(),
        original_HIGH_audit_gate='HIGH_STABILITY_EVIDENCE_ONLY',original_long_term_capacity_deficit='INSUFFICIENT_EVIDENCE',
        capacity_gate=capacity,utility_gate=utility,gate_vocabulary=['HIGH_STABILITY_EVIDENCE_ONLY','DYNAMIC_CAPACITY_IMPROVEMENT','DYNAMIC_NO_CAPACITY_IMPROVEMENT','DYNAMIC_EXPLORATORY_POSITIVE_UTILITY','DYNAMIC_NO_POSITIVE_UTILITY','DYNAMIC_EVIDENCE_GAP'],
        capacity_and_utility_separate=True,LOW_all_three_legal_stable=lowall,LOW_exploratory_positive=positive,pairs=pairs,
        high_dynamic_stability=val('HIGH','dynamic')['worker_states'] if 'dynamic' in high else None,
        executed=[r['slot']['run_id'] for r in raws],fatal=fatal,
        no_positive_delta_against_unstable=True,original_stability_thresholds_unchanged=True,router_parameters_unchanged=True,
        independent_replication='NOT_EXECUTED_THIS_ROUND',optimizer_modified=False,next_module_automatic=False,
        G_only_screen=f['G_only'],scope='One matched trace per load, six maximum 600s runs; exploratory, not independent replication or global optimality.')
    save('module28r_gate.json',gate)
    if positive and final:
        save('frozen_positive_candidate.json',dict(candidate=low['dynamic']['slot'],parameters=f['parameters'],router_source=ref(OUT/'dynamic_router.py'),
            workloads=ref(OUT/'frozen_workloads.json'),pairs=lowpairs,replication_recommended=True,replication_started=False))
    lines=['# Module 28R：Dynamic G/S Routing Follow-up\n\n',f'状态：{"已完成并停止" if final else "GPU比较进行中"}。容量 Gate：**{capacity}**；效用 Gate：**{utility}**。\n\n',
        '历史Module29及以后不作为证据；原Module28 Gate、20A定义、模型池、B4预算、准入、副本、Fixed10和动态阈值均保持不变。HIGH LID11.5、LOW10.5；其余五任务各0.2QPS。30s warmup、原冷却门槛、600s measurement、60s有界排空；两个场景交错运行，不补种子、不增加GPU运行。\n\n',
        '|场景|方案|系统状态|合法U|G / S目标数|窗口完成QPS|末端积压|\n|---|---|---|---:|---:|---:|---:|\n']
    for r in sorted(runs,key=lambda r:r['order']):
        st=next((s for s in stability if s['run_id']==r['run_id'] and s['scope']=='system'),{})
        lines.append(f'|{r["scenario"]}|{r["arm"]}|{r.get("system_stability",r["status"])}|{r.get("U")}|{r.get("actual_G")} / {r.get("actual_S")}|{r.get("completed_in_window_qps")}|{st.get("end_backlog")}|\n')
    lines+=['\n所有U均按完整cohort及原硬约束；不稳定延迟仅INELIGIBLE诊断，不用于效用排名或成功比较。有效任务数与质量子集见per_task_latency.csv；pooled mean/P95见gpu_routing_comparison.csv，不与加权归一化P混同。worker单独稳定性见stability_comparison.csv。NVML峰值仅0.5s采样高水位，未声称瞬时硬件峰值；worker allocator MEMORY_END输出在gpu_campaign.log，安全显存约束按原驻留系数及采样峰值检验。\n\n',
        '## 六个结论\n\n',
        '1. 原Module28 S-only有真实末150s队列增长与32请求积压，原分类器unstable、U=NULL；全窗口S需求约0.897、570s恢复至0、2.445s排空。**长期持续容量不足证据不足**，局部波动与边界效应无法排除。\n',
        f'2. 同轨迹新HIGH的容量结果：{capacity}；以新S-only和Dynamic的系统及所有worker原分类器为准。若新S-only也稳定，只能报告本窗口稳定，不能声称恢复了已测容量缺口，更不能追溯修改原Gate。\n',
        f'3. LOW三个方案均合法稳定：{lowall}；recoverable_congested仅在原最终恢复规则通过时合法。\n',
        f'4. LOW真实系统正效用候选：{positive}；两项预注册ΔU={[(p["control_arm"],p["delta_U"]) for p in lowpairs]}。必须两项都>+0.01、G/S共同承担，三个方案完整合法后才接受。\n',
        '5. S同输入服务和等待、G慢路径及背景任务变化逐项见worker_interference_analysis.csv、worker_metrics.csv和utility_decomposition.csv。服务膨胀是共置运行关联证据，不能唯一断言物理GPU、CPU或热态机制。\n',
        f'6. 值得独立复现的正收益区间：{"仅LOW单点探索候选，已冻结，建议独立长窗口；本轮不启动" if positive else "未形成满足门槛的可信正收益候选；不扩展负载、参数或随机种子"}。G-only未测，同副本历史服务需求筛选约HIGH4.41、LOW4.03 handler-s/s且还含VQA；历史筛选不是本轮实测稳定性，不宣称优于所有合法方案。\n\n',
        '## 限制与停止\n\n每负载一个匹配轨迹、每方案单窗口，不能提供独立复现置信区间。历史元数据与来源SHA校验在historical_integrity.json；正式优化器未改。所有异常、warning和原始完整请求保留，未执行项目留NULL。']
    if fatal:lines.append('\n\n安全/基础设施停止：'+json.dumps(fatal,ensure_ascii=False)+'\n')
    (OUT/'module28r_summary.md').write_text(''.join(lines))
    return gate

if __name__=='__main__':
    sys.path.insert(0,str(OLD))
    import report as original_report
    original_report.OUT=OUT
    f=json.loads((OUT/'frozen_candidate_manifest.json').read_text())
    raws=[json.loads(p.read_text()) for p in sorted((OUT/'raw_runs').glob('*.json'))]
    write_reports(raws,f,final=(OUT/'campaign_completed.json').exists())
