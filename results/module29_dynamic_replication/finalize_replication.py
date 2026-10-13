"""Post-campaign diagnostics only; never changes frozen routing, utility or gates."""
from common_replication import *
import bisect

def main():
    f=json.loads((OUT/'frozen_replication_manifest.json').read_text())
    gate=json.loads((OUT/'module29_gate.json').read_text())
    assert gate['final'] and (OUT/'campaign_completed.json').exists()
    csv.field_size_limit(128*1024*1024)
    groups=collections.defaultdict(dict);env=[];alloc=[];quality=[];actual_windows=[];queue_parts=[]
    for p in sorted((OUT/'raw_runs').glob('*.json')):
        raw=json.loads(p.read_text());s=raw['slot'];base=dict(run_id=s['run_id'],group=s['group'],arm=s['arm'])
        ph=raw.get('phases',{}).get('measurement')
        if ph:
            groups[s['group']][s['arm']]=p
            win=ph['window'];rs=ph['request_records'];done=[r for r in rs if r['status']=='completed']
            assert win['planned_duration_seconds']==2400. and win['end_time_ns']-win['start_time_ns']==2400*10**9
            assert raw['transport_config']['measurement_seconds']==2400.
            assert s['lambda_rps']==f['rates'] and s['deployment']==f['independent_slots'][0]['deployment']
            accepted_within=sum(r.get('enqueued_time_ns') is not None and r['enqueued_time_ns']<=win['end_time_ns'] for r in rs)
            actual_windows.append(dict(**base,formal_seconds=2400.,offered=len(rs),accepted_within=accepted_within,
                accepted_within_qps=accepted_within/2400.,warmup_seconds=raw.get('warmup',{}).get('window',{}).get('planned_duration_seconds'),
                trace=ph['trajectory'],config_measurement_seconds=raw['transport_config']['measurement_seconds']))
            for w in raw['replica_to_family']:
                for t in sorted({r['task'] for r in done if r['model']==w}):
                    rr=[r for r in done if r['model']==w and r['task']==t];parts=[]
                    for r in rr:
                        native=r['worker_queue_wait_ms'];sender=(r['actual_submit_time_ns']-r['enqueued_time_ns'])/1e6
                        dispatch=(r['enqueued_time_ns']-r['scheduled_time_ns'])/1e6
                        parts.append(dict(native_queue=native,sender_queue=sender,host_to_handler_wait=native+sender,dispatch_delay=dispatch))
                    queue_parts.append(dict(**base,worker=w,task=t,n=len(rr),native_queue=stats(parts,'native_queue'),
                        sender_queue=stats(parts,'sender_queue'),host_to_handler_wait=stats(parts,'host_to_handler_wait'),dispatch_delay=stats(parts,'dispatch_delay')))
        samples=raw.get('telemetry',{}).get('samples',[])
        for v in samples:
            env.append(dict(**base,**{k:v.get(k) for k in ['time_ns','unix_ns','phase','temperature_gpu_c','sm_clock_mhz',
                'memory_clock_mhz','performance_state','pstate','power_draw_w','power_limit_w','used_mib','owned_used_mib',
                'gpu_util_percent','foreign_pids','host_available_RAM_bytes','worker_CPU_core_percent',
                'safety_host_outstanding','safety_oldest_pending_seconds']}))
        for w,r in raw.get('shutdown',{}).items():
            alloc.append(dict(**base,worker=w,exit_code=r.get('exit_code'),forced=r.get('forced'),bye_received=r.get('bye_received'),
                native_allocator_peak='SEE MEMORY_END log where supported; not available for every worker'))
        quality.append(dict(**base,worker_ready_metadata=raw.get('worker_identities'),probes=raw.get('path_probe_records'),
            status=raw['status'],errors=raw['errors']))
        del raw
    csvsave('environment_control.csv',env,['run_id','group','arm','phase','temperature_gpu_c','power_draw_w'])
    csvsave('worker_exit_allocator_scope.csv',alloc,['run_id','worker'])
    save('worker_runtime_identity_and_probes.json',quality)
    csvsave('actual_window_accounting.csv',actual_windows,['run_id','formal_seconds','accepted_within_qps'])
    csvsave('worker_queue_decomposition.csv',queue_parts,['run_id','worker','task'])
    # Add the disabled quality term without inventing a missing cardinal score.
    decomp=list(csv.DictReader((OUT/'utility_decomposition.csv').open()))
    for p in (OUT/'run_summaries').glob('*.json'):
        r=json.loads(p.read_text())['run']
        decomp.append(dict(run_id=r['run_id'],group=r['group'],seed=r['seed'],arm=r['arm'],component='Q',
            norm=None,contribution=0.,enabled=False,weight=f['weights']['Q'],eligible=r['eligible'],
            scope='Disabled wQ=0; Q_norm remains NULL; original measured quality admission is a hard constraint'))
    csvsave('utility_decomposition.csv',decomp,['run_id','component'])
    log=(OUT/'gpu_campaign.log').read_text()
    (OUT/'native_allocator_peak_log.txt').write_text('\n'.join(line for line in log.splitlines() if 'MEMORY_END' in line)+'\n')
    summaries=[]
    for (rid,arm) in sorted({(r['run_id'],r['arm']) for r in env}):
        rr=[r for r in env if r['run_id']==rid and r['phase']=='measurement']
        summary=dict(run_id=rid,arm=arm,n=len(rr))
        for key in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','power_draw_w','power_limit_w','gpu_util_percent']:
            a=[r[key] for r in rr if r.get(key) is not None]
            summary[key]=dict(min=min(a),max=max(a),mean=float(np.mean(a))) if a else None
        summary['foreign_process_seen']=any(r['foreign_pids'] for r in rr)
        summary['pstate_sample_counts']=dict(collections.Counter(r['pstate'] for r in rr))
        summaries.append(summary)
    save('environment_comparison.json',summaries)
    contributions=[];sensitivity=[];selected=[]
    for group,paths in sorted(groups.items()):
        if set(paths)!={'s_only','fixed10','dynamic'}:continue
        raws={a:json.loads(p.read_text()) for a,p in paths.items()}
        records={a:r['phases']['measurement']['request_records'] for a,r in raws.items()}
        if not all(all(r['status']=='completed' for r in rs) for rs in records.values()):continue
        for control in ['s_only','fixed10']:
            cr={r['request_id']:r for r in records[control]}
            for task,q in f['rates'].items():
                rr=[r for r in records['dynamic'] if r['task']==task]
                for route in sorted({r['model'] for r in rr}):
                    matched=[(r,cr[r['request_id']]) for r in rr if r['model']==route]
                    assert all(r['sample_id']==c['sample_id'] and r['frozen_offset_seconds']==c['frozen_offset_seconds'] for r,c in matched)
                    contributions.append(dict(group=group,control_arm=control,task=task,dynamic_route=route,n=len(matched),
                        dynamic_service_mean=float(np.mean([r['worker_latency_ms'] for r,c in matched])),
                        control_service_mean=float(np.mean([c['worker_latency_ms'] for r,c in matched])),
                        dynamic_queue_mean=float(np.mean([r['worker_queue_wait_ms'] for r,c in matched])),
                        control_queue_mean=float(np.mean([c['worker_queue_wait_ms'] for r,c in matched])),
                        weighted_task_service_delta_ms=float(sum(r['worker_latency_ms']-c['worker_latency_ms'] for r,c in matched)/len(rr)),
                        weighted_task_queue_delta_ms=float(sum(r['worker_queue_wait_ms']-c['worker_queue_wait_ms'] for r,c in matched)/len(rr)),
                        weighted_task_e2e_delta_ms=float(sum(r['e2e_latency_ms']-c['e2e_latency_ms'] for r,c in matched)/len(rr)),
                        scope='Matched cross-run inputs; route subsets are selected, not randomized event counterfactuals'))
                # Remove the union of the two runs' largest single latency sample
                # per task from both cohorts. Diagnostic only: official U is intact.
                dr={r['request_id']:r for r in rr};ct=[cr[k] for k in dr]
                excluded={max(rr,key=lambda r:r['e2e_latency_ms'])['request_id'],max(ct,key=lambda r:r['e2e_latency_ms'])['request_id']}
                da=np.array([r['e2e_latency_ms'] for r in rr if r['request_id'] not in excluded]);ca=np.array([r['e2e_latency_ms'] for r in ct if r['request_id'] not in excluded])
                sensitivity.append(dict(group=group,control_arm=control,task=task,excluded_request_ids=sorted(excluded),remaining=len(da),
                    diagnostic_mean_delta=float(da.mean()-ca.mean()),diagnostic_p95_delta=float(np.percentile(da,95)-np.percentile(ca,95)),
                    diagnostic_latency_utility_delta=-q/11.5*(f['weights']['L']*(da.mean()-ca.mean())+f['weights']['P']*(np.percentile(da,95)-np.percentile(ca,95)))/f['refs'][task],
                    scope='Matched leave-two-extremes sensitivity, never changes U, gate, or independent n'))
        if gate['stage_e_triggered']:
            dy=raws['dynamic'];win=dy['phases']['measurement']['window'];g=f['parameters']['G_worker'];s=f['parameters']['S_worker']
            iv={}
            for w in [g,s]:
                intervals=[]
                for r in records['dynamic']:
                    if r['model']!=w:continue
                    delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns'];response=r['worker_response']
                    intervals.append((response['start_ns']+delta,response['end_ns']+delta,r['request_id']))
                iv[w]=sorted(intervals)
            for r in records['dynamic']:
                if r['task']!='lid' or r['model']!=g:continue
                d=r['routing_decision'];at=d['decision_time_ns'];scope=dict(group=group,request_id=r['request_id'],offset_seconds=r['frozen_offset_seconds'])
                def state(w):
                    rows=iv[w];i=bisect.bisect_right([q[0] for q in rows],at)-1
                    busy=i>=0 and rows[i][0]<=at<rows[i][1]
                    previous_end=rows[i][1] if i>=0 and not busy else rows[i-1][1] if i>=1 else None
                    return dict(native_busy=busy,idle_since_last_native_end_ms=(at-previous_end)/1e6 if previous_end is not None and not busy else None)
                selected.append(dict(**scope,**d,G_native_state=state(g),S_native_state=state(s),
                    actual_service_ms=r['worker_latency_ms'],actual_queue_ms=r['worker_queue_wait_ms'],actual_e2e_ms=r['e2e_latency_ms'],
                    interpretation='Real-time host queue decision; native states retrospective only, not policy inputs or proof of causal event-level benefit'))
        del raws
    csvsave('latency_path_attribution.csv',contributions,['group','control_arm','task','dynamic_route'])
    csvsave('matched_extreme_sensitivity.csv',sensitivity,['group','control_arm','task'])
    csvsave('positive_replication_mechanism.csv',selected,['group','request_id','offset_seconds'])
    current=metadata();before=json.loads((OUT/'historical_metadata_before.json').read_text())
    changed=[p for p,v in before.items() if current.get(p)!=v];added=[p for p in current if p not in before]
    sources=f['reused_implementation']+f['implementation']+[f[k] for k in ['catalog','utility_definition','utility_reference','quality_policy','model_files','environment','development_inventory']]
    bad_hashes=[r['path'] for r in sources if ref(r['path'])['sha256']!=r['sha256']]
    inventory=json.loads(Path(f['development_inventory']['path']).read_text())
    bad_development=[r['path'] for r in inventory['development_files'] if ref(r['path'])['sha256']!=r['sha256']]
    bad_models=[r['path'] for r in json.loads(Path(f['model_files']['path']).read_text()) if ref(r['path'])['sha256']!=r['sha256']]
    manifest_sha=ref(OUT/'frozen_replication_manifest.json')['sha256']
    assert all(json.loads(r['manifest'])['sha256']==manifest_sha for r in csv.DictReader((OUT/'gpu_run_manifest.csv').open()))
    save('historical_integrity.json',dict(protected_files=len(before),changed=changed,added=added,
        source_hash_mismatches=bad_hashes,development_hash_mismatches=bad_development,model_hash_mismatches=bad_models,
        manifest_sha256=manifest_sha,passed=not changed and not added and not bad_hashes and not bad_development and not bad_models))
    assert not changed and not added and not bad_hashes and not bad_development and not bad_models,'Historical/frozen source/model integrity failure'
    save('postprocessing_manifest.json',dict(source=ref(Path(__file__)),scope='CPU diagnostics only after final GPU campaign; frozen gate and U unchanged'))
    lines=['\n## 最终问题与诊断\n\n']
    cs=gate['paired_statistics']
    positive=gate['gate']=='DYNAMIC_POSITIVE_REPLICATED'
    lines.append(f'1. 三组独立配对是否均正收益：{gate["all_six_deltas_over_threshold"]}；完整合法比较：{gate["all_nine_legal_stable"]}。\n')
    lines.append(f'2. 是否同时优于 S-only 和 Fixed10：{positive}（通过全部 +0.01 门槛且两项 t95 区间下界 >0 才认定）。\n')
    lines.append(f'3. 独立均值、标准差与区间：{cs}；开发数据未使用。\n')
    dynamic=[json.loads(p.read_text())['run'] for p in (OUT/'run_summaries').glob('*.json') if '_dynamic.' in p.name]
    lines.append(f'4. G/S 实际分流：{[(r["group"],r["actual_G"],r["actual_S"],r["G_fraction"]) for r in dynamic]}。仅三个种子，不能证明比例长期恒定。\n')
    lines.append('5. 收益/退化分项完整保存在 utility_decomposition.csv、paired_utility_comparison.csv、latency_path_attribution.csv；同输入 S 服务/排队、G 慢路径及背景任务分别报告。tail_sensitivity.csv 和 matched_extreme_sensitivity.csv 检查极端样本影响，block_sensitivity.csv 检查时间集中，全部仅诊断。环境温度/频率/功率实际值见 environment_control.csv；加载/warmup/drain 均分开。\n')
    lines.append(f'6. {"该冻结 LOW 单点值得进一步研究动态分流规律；尚不构成正式优化器集成授权或跨负载鲁棒性认证" if positive else "未获得满足门槛的可信独立复现，不据此集成优化器；停止，不调参、不加种子或模型池搜索"}。\n\n')
    lines.append(f'Stage E：{"已按可信正复现条件分析 G 选择状态；逐请求关联不等于单次卸载因果证明" if positive else "未触发；仅保留 Stage C 诊断"}。历史 {len(before)} 个文件元数据及开发全目录/冻结源码 SHA256 均保持一致。GPU 运行结束后停止。\n')
    lines.append('\n协议字段说明：frozen_replication_manifest.json 中 independent_slots、replication_protocol 和 maximum_formal_gpu_seconds 是本轮执行协议；复制的祖先 independent_protocol / exploration_seconds 字段是开发来源元数据，不参与运行。actual_window_accounting.csv 逐次核对实际计划、运行配置均为 2400 秒，且 warmup 为 30 秒。没有改写冻结 manifest。\n')
    with (OUT/'module29_summary.md').open('a') as stream:stream.write(''.join(lines))
    print('FINAL_CPU_DIAGNOSTICS_DONE',gate['gate'])

if __name__=='__main__':main()
