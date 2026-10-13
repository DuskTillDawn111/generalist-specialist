"""CPU-only final integrity, numerical attribution, and evidence verification."""
from audit_followup import *
import time,re

def main():
    assert (OUT/'campaign_completed.json').exists(), 'Wait for frozen GPU campaign'
    sys.path.insert(0,str(OLD))
    import report as original_report
    original_report.OUT=OUT
    f=json.loads((OUT/'frozen_candidate_manifest.json').read_text())
    manifest=json.loads((OUT/'gpu_run_manifest.json').read_text())
    gate=json.loads((OUT/'module28r_gate.json').read_text())
    ledger=json.loads((OUT/'attempt_ledger.json').read_text())
    audit=json.loads((OUT/'stage_a_audit.json').read_text())
    before=json.loads((OUT/'historical_metadata_before.json').read_text());after=metadata()
    changes=[p for p,v in before.items() if after.get(p)!=v]
    additions=[p for p in after if p not in before]
    source_changes=[r['path'] for r in audit['sources'] if ref(r['path'])['sha256']!=r['sha256']]
    frozen_changes=[r['path'] for r in f['implementation']+f['reused_implementation']+[manifest['workloads'],manifest['protocol'],f['catalog'],f['utility_definition'],f['utility_reference'],f['quality_policy']] if ref(r['path'])['sha256']!=r['sha256']]
    save('historical_integrity.json',dict(files_checked=len(before),historical_metadata_changes=changes,historical_additions=additions,
        original_source_sha256_changes=source_changes,frozen_source_sha256_changes=frozen_changes,passed=not(changes or additions or source_changes or frozen_changes),at_unix_ns=time.time_ns()))
    assert not(changes or additions or source_changes or frozen_changes)
    assert len(ledger)<=6 and len({r['run_id'] for r in ledger})==len(ledger)
    raws={}
    for slot in f['exploration_slots']:
        p=OUT/'raw_runs'/(slot['run_id']+'.json')
        if p.exists():raws[slot['run_id']]=json.loads(p.read_text())
    assert all(raw['slot']==next(s for s in f['exploration_slots'] if s['run_id']==rid) for rid,raw in raws.items())
    groups=collections.defaultdict(dict);checks=[];attribution=[];routing=[];summary=[];vmap={}
    for rid,raw in raws.items():
        slot=raw['slot'];scene=slot['scenario'];arm=slot['arm'];groups[scene][arm]=raw
        v=original_report.evaluate(raw,f);vmap[rid]=v
        ph=raw.get('phases',{}).get('measurement')
        if not ph:continue
        rs=ph['request_records'];tr=json.loads(Path(slot['trajectories']['measurement']['path']).read_text())
        assert len(rs)==len(tr['events'])
        assert all(r['task']==e['task'] and r['sample_id']==e['item']['sample']['id'] and r['frozen_offset_seconds']==e['offset_seconds'] for r,e in zip(rs,tr['events']))
        assert ph['window']['planned_duration_seconds']==600
        for r in rs:
            d=r.get('routing_decision')
            if d:assert d['latest_observed_completion_time_ns']<=d['decision_time_ns']
        assert v['U'] is None if not v['eligible'] else abs(v['U']-sum(v[k+'_contribution'] for k in ['T','L','P','M']))<1e-12
        checks.append(dict(run_id=rid,full_frozen_trace_matched=True,original_evaluation=v['worker_states'],U=v['U'],eligible=v['eligible'],
            raw=ref(OUT/'raw_runs'/(rid+'.json')),GPU_release_verified=raw['gpu_release']['verified']))
        if arm=='dynamic':
            target=[r for r in rs if r['task']=='lid'];gr=[r for r in target if r['model']==f['parameters']['G_worker']]
            reasons=dict(collections.Counter(r['routing_decision']['reason'] for r in target))
            for r in gr:
                d=r['routing_decision'];assert d['G_pending']==0 and d['G_target_inflight']==0 and d['admitted'] and d['capacity_ok']
                assert d['reason']=='G_LOWER_SYSTEM_MARGINAL_COST'
            row=dict(scenario=scene,run_id=rid,total_target=len(target),actual_G=len(gr),actual_S=len(target)-len(gr),G_fraction=len(gr)/len(target),
                reasons=reasons,G_route_S_pending=stats([dict(x=r['routing_decision']['S_pending']) for r in gr],'x'),
                G_route_service=stats(gr,'worker_latency_ms'),G_route_wait=stats(gr,'worker_queue_wait_ms'),
                prediction_errors=ph.get('prediction_errors'),live_observation_checks_passed=True)
            routing.append(row)
    for scene,arms in groups.items():
        if 's_only' not in arms:continue
        control=arms['s_only'];cp=control.get('phases',{}).get('measurement')
        if not cp:continue
        cr={r['request_id']:r for r in cp['request_records'] if r['status']=='completed'}
        for arm in ['fixed10','dynamic']:
            if arm not in arms:continue
            raw=arms[arm];ph=raw.get('phases',{}).get('measurement')
            if not ph:continue
            ar=[r for r in ph['request_records'] if r['status']=='completed']
            cv=vmap[control['slot']['run_id']];av=vmap[raw['slot']['run_id']]
            legal=cv['eligible'] and av['eligible']
            # LOW requires complete all-three group for any legitimate ΔU.
            legal=legal and (scene!='LOW' or set(arms)=={'s_only','fixed10','dynamic'} and all(vmap[r['slot']['run_id']]['eligible'] for r in arms.values()))
            if legal:
                du=av['U']-cv['U'];parts={k:av[k+'_contribution']-cv[k+'_contribution'] for k in ['T','L','P','M']}
                assert abs(sum(parts.values())-du)<1e-12
                for t,q in raw['slot']['lambda_rps'].items():
                    p,c=av['per_task'][t],cv['per_task'][t];weight=q/sum(raw['slot']['lambda_rps'].values())
                    attribution.append(dict(scenario=scene,arm=arm,kind='LEGAL_TASK_UTILITY',component=t,
                        mean_delta_ms=p['mean_ms']-c['mean_ms'],p95_delta_ms=p['p95_ms']-c['p95_ms'],
                        mean_U_delta=-f['weights']['L']*weight*(p['mean_ms']-c['mean_ms'])/f['refs'][t],
                        p95_U_delta=-f['weights']['P']*weight*(p['p95_ms']-c['p95_ms'])/f['refs'][t]))
                summary.append(f'{scene} {arm} vs S-only：ΔU={du:+.6f}，分项T/L/P/M={parts}。')
            else:summary.append(f'{scene} {arm} vs S-only：合法ΔU=NULL；以下服务/等待仅完整观测诊断。')
            # Observed target mean difference partition, no hypothetical replay.
            target=[r for r in ar if r['task']=='lid' and r['request_id'] in cr];n=len(target)
            for path,w in [('S',f['parameters']['S_worker']),('G',f['parameters']['G_worker'])]:
                shared=[(r,cr[r['request_id']]) for r in target if r['model']==w]
                if not shared:continue
                ds=float(np.mean([r['worker_latency_ms']-c['worker_latency_ms'] for r,c in shared]))
                dq=float(np.mean([r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in shared]))
                de=float(np.mean([r['e2e_latency_ms']-c['e2e_latency_ms'] for r,c in shared]))
                fraction=len(shared)/n
                attribution.append(dict(scenario=scene,arm=arm,kind='OBSERVED_TARGET_MEAN_PARTITION',component=path,
                    n=len(shared),target_fraction=fraction,service_delta_ms=ds,wait_delta_ms=dq,e2e_delta_ms=de,
                    residual_host_transport_delta_ms=de-ds-dq,weighted_e2e_delta_ms=fraction*de,
                    causal_scope='Matched observations across complete runs; not an unexecuted-request counterfactual'))
                summary.append(f'{scene} {arm}目标{path}路径：份额{fraction:.3%}，同输入服务Δ={ds:+.3f}ms、等待Δ={dq:+.3f}ms、E2EΔ={de:+.3f}ms，对目标整体mean差贡献={fraction*de:+.3f}ms。')
            for t in ['asr','visual_qa','text','image','ocr']:
                rr=[(r,cr[r['request_id']]) for r in ar if r['task']==t and r['request_id'] in cr]
                if not rr:continue
                ds=float(np.mean([r['worker_latency_ms']-c['worker_latency_ms'] for r,c in rr]));dq=float(np.mean([r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in rr]))
                summary.append(f'{scene} {arm}背景{t}：n={len(rr)}，同输入服务Δ={ds:+.3f}ms、等待Δ={dq:+.3f}ms。')
    capacity_details={}
    for arm,raw in groups.get('HIGH',{}).items():
        ph=raw.get('phases',{}).get('measurement')
        if not ph:continue
        ev=ph['stability_evidence'];value=vmap[raw['slot']['run_id']]
        capacity_details[arm]=dict(system_state=ph['stability']['stability_state_v2'],eligible=value['eligible'],
            U=value['U'],end_backlog=ev['end_backlog_requests'],tail_slope_rps=ev['tail_backlog_slope'],
            tail_growth=ev['tail_backlog_growth'],tail_increasing_fraction=ev['tail_increasing_fraction'],
            completed_in_window_qps=value['within_qps'],final_backlog=ev['final_backlog_after_drain'])
    gate['HIGH_capacity_details']=capacity_details
    gate['capacity_evidence_scope']='ORIGINAL_FINITE_WINDOW_CLASSIFIER_FEASIBILITY_ONLY; long-horizon stability unproven'
    gate['HIGH_dynamic_still_positive_tail_slope']=bool(capacity_details.get('dynamic',{}).get('tail_slope_rps',0)>0)
    if 'dynamic' in capacity_details:
        d=capacity_details['dynamic']
        summary.append(f'HIGH Dynamic原判据状态={d["system_state"]}、合法={d["eligible"]}，但末端积压{d["end_backlog"]}、末段斜率{d["tail_slope_rps"]:+.6f}请求/s、增长{d["tail_growth"]:.6f}、递增比例{d["tail_increasing_fraction"]:.2f}。原四项联合拒绝条件中，递增比例0.50未达到0.60，因此通过；不能事后改变规则，也不能将这个通过写成长时域持续容量恢复。')
    csvsave('latency_utility_attribution.csv',attribution)
    save('dynamic_routing_analysis.json',routing)
    # Native allocator peaks are per-worker scoped and not simultaneous device peaks.
    native=[];active=None
    for line in (OUT/'gpu_campaign.log').read_text().splitlines():
        if line.startswith('START '):active=line.split()[1]
        if 'MEMORY_END|' in line:
            prefix,payload=line.split('MEMORY_END|',1)
            try:native.append(dict(run_id=active,worker=prefix.split(']')[0].lstrip('[').strip(),**json.loads(payload.strip()),scope='PER_WORKER_ALLOCATOR_PEAK_NOT_SIMULTANEOUS_GPU_PEAK'))
            except json.JSONDecodeError:pass
    csvsave('native_allocator_peaks.csv',native,['run_id','worker','scope'])
    # Thermal matching and sample count remain explicit qualifications.
    if gate['LOW_exploratory_positive']:
        assert gate['LOW_all_three_legal_stable']
        assert all(p['delta_U']>.01 for p in gate['pairs'] if p['scenario']=='LOW')
    gate['historical_integrity_passed']=True;gate['historical_files_checked']=len(before)
    gate['formal_attempts']=len(ledger);gate['model_startup_cpu_failure_attempts']=0
    gate['prelaunch_CPU_failure']=dict(preserved='prelaunch_cpu_failure/',GPU_inferences=0,reason='Reporting bracket syntax error; fixed before model startup, original workload/routing unchanged')
    gate['CPU_report_repairs']=[ref(OUT/name) for name in ['report_repair_amendment.json','report_serialization_amendment.json'] if (OUT/name).exists()]
    gate['numerical_attribution']=ref(OUT/'latency_utility_attribution.csv')
    save('module28r_gate.json',gate)
    (OUT/'worker_interference_analysis.md').write_text('# 实测服务、排队与效用归因\n\n'+'\n\n'.join(summary)+'\n\n'+
        '目标路径的mean差按实际G/S份额相加等于观测目标mean差；P95不可按路径P95线性相加，效用P95只使用完整任务P95。所有对不稳定控制的合法ΔU保持NULL。上述观测没有单独隔离GPU/CPU/温度因果。\n')
    with (OUT/'module28r_summary.md').open('a') as stream:
        stream.write('\n\n## 数值归因补充\n\n'+'\n\n'.join(summary)+'\n\n')
        stream.write(f'完整校验：{len(before)}个历史文件元数据、原始来源SHA和冻结GPU实现均未变化；{len(ledger)}次正式GPU探索，未重试。启动前CPU语法失败（模型启动/推理0次）保存在prelaunch_cpu_failure/；另有两次GPU结束后的CPU报告错误（重复status字段、numpy布尔序列化），修复及继续剩余槽位记录在两个report amendment文件，修正时负载、比例、动态阈值与数值公式未变。\n')
    save('final_verification.json',dict(passed=True,at_unix_ns=time.time_ns(),checks=checks,
        historical_integrity=True,attempts=len(ledger),routing_checks=routing,capacity_gate=gate['capacity_gate'],utility_gate=gate['utility_gate'],
        final_outputs=[ref(OUT/name) for name in ['module28_backlog_audit.md','frozen_workloads.json','gpu_run_manifest.json','backlog_timeseries.csv','gpu_routing_comparison.csv','worker_interference_analysis.csv','stability_comparison.csv','system_utility_comparison.csv','module28r_gate.json','module28r_summary.md']]))
    print(json.dumps(dict(capacity_gate=gate['capacity_gate'],utility_gate=gate['utility_gate'],history_files=len(before),attempts=len(ledger)),ensure_ascii=False))

if __name__=='__main__': main()
