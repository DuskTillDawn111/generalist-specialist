"""Offline verification after the sole GPU session. No inference/control imports."""
import sys
sys.dont_write_bytecode=True
import collections
import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/root/autodl-tmp/generalist-specialist');sys.path.insert(0,str(ROOT))
from instrumentation import OUT,ref,save,router_state,restore_router,guards
from dynamic_router import DynamicRouter
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def csvsave(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def stats(values):
    import numpy as np
    if not values:return dict(count=0,mean_ms=None,p10_ms=None,p95_ms=None)
    return dict(count=len(values),mean_ms=float(np.mean(values)),p10_ms=float(np.percentile(values,10)),p95_ms=float(np.percentile(values,95)))

def response_native(row,start):
    v=row.get('worker_response') or {}
    if not all(v.get(k) is not None for k in ['start_ns','end_ns']):return None
    if row.get('actual_submit_time_ns') is None or row.get('actual_submit_unix_ns') is None:return None
    delta=row['actual_submit_time_ns']-row['actual_submit_unix_ns']
    return ((v['start_ns']+delta-start)/1e9,(v['end_ns']+delta-start)/1e9)

def main():
    manifest=json.loads((OUT/'controlled_manifest.json').read_text());p=manifest['parameters']
    raw=json.loads((OUT/'gpu_validation_raw.json').read_text())
    f=json.loads(Path(manifest['original_frozen_configuration']['path']).read_text())
    issues=[];trace=[];counts=collections.Counter();guardcounts=collections.defaultdict(collections.Counter)
    decisions=[];arrivals=[];commit={};phases={};last_state={};last_snapshot={}
    # Recompute entire policy state from phase initialization + recorded causal completion harvests.
    replay=None;replay_phase=None;lastseq=0;outstanding={}
    with (OUT/'decision_event_journal.jsonl').open() as stream:
        for line in stream:
            e=json.loads(line)
            if e['seq']!=lastseq+1:issues.append('journal sequence gap')
            lastseq=e['seq'];kind=e['kind'];counts[kind]+=1
            if kind=='ENQUEUE' and e['status'] in ['queued','submitted']:
                outstanding[e['request_id']]=dict(task=e['task'],worker=e['worker'],enqueued_time_ns=e['time_ns'])
            if kind=='COLLECTOR_COMMIT':
                rid=e['request_id']
                if rid in commit:issues.append('duplicate collector commit:'+rid)
                commit[rid]=e
                outstanding.pop(rid,None)
                if e['reader_received_time_ns']>e['time_ns']:issues.append('commit before reader:'+rid)
            if kind=='PHASE_START':
                phases[e['phase']]=e;replay=DynamicRouter(copy.deepcopy(p));replay_phase=e['phase']
                if router_state(replay)!=e['router_state']:issues.append('phase init mismatch')
            if kind!='ARRIVAL_DECISION':continue
            arrivals.append((e['phase'],e['request_id'],e['task']))
            now=e['time_ns'];rid=e['request_id'];phase=e['phase']
            if phase!=replay_phase:issues.append('unexpected phase')
            if router_state(replay)!=e['router_before_harvest']:issues.append('preharvest replay mismatch:'+rid)
            if e['scheduled_time_ns']>now:issues.append('snapshot precedes scheduled arrival:'+rid)
            for r in e['harvested_completions']:
                c=commit.get(r['request_id'])
                if not c or c['time_ns']!=r['collector_commit_time_ns']:issues.append('harvest commit missing:'+rid)
                if r['completion_time_ns']>now or r['collector_commit_time_ns']>now:issues.append('future completion leakage:'+rid)
                replay.observe_completion(r,now/1e9)
            if router_state(replay)!=e['router_before_decide']:issues.append('predecide replay mismatch:'+rid)
            pending=e['pending']
            logged_pending={r['request_id']:dict(task=r['task'],worker=r['model'],enqueued_time_ns=r['enqueued_time_ns'])
                            for rows in pending.values() for r in rows}
            if logged_pending!=outstanding:issues.append('pending membership/ordering source mismatch:'+rid)
            for rows in pending.values():
                if rows!=sorted(rows,key=lambda x:x['enqueued_time_ns']):issues.append('pending FIFO mismatch:'+rid)
                for r in rows:
                    if r['enqueued_time_ns']>now:issues.append('future enqueue leakage:'+rid)
            if e['target_router_invoked']:
                d=e['decision'];state=e['router_before_decide']
                route,recovered=replay.decide(rid,now/1e9,pending,d['admitted'],d['capacity_ok'])
                if recovered!=d or route!=e['selected_worker']:issues.append('decision replay mismatch:'+rid)
                gs=guards(p,state,d,now/1e9,d['admitted'],d['capacity_ok'])
                if gs!=e['guards']:issues.append('guard replay mismatch:'+rid)
                if router_state(replay)!=e['router_after_decide']:issues.append('postdecide replay mismatch:'+rid)
                for k,v in gs.items():guardcounts[k][v]+=1
                origin=phases[phase]['time_ns'];gap=d['margin_ms']
                row=dict(phase=phase,request_id=rid,task=e['task'],scheduled_t_s=(e['scheduled_time_ns']-origin)/1e9,
                    decision_t_s=(now-origin)/1e9,decision_delay_ms=(now-e['scheduled_time_ns'])/1e6,
                    selected_worker=route,reason=d['reason'],S_pending=d['S_pending'],G_pending=d['G_pending'],
                    S_composition=dict(collections.Counter(r['task'] for r in pending.get(p['S_worker'],[]))),
                    G_composition=dict(collections.Counter(r['task'] for r in pending.get(p['G_worker'],[]))),
                    C_S_ms=d['S_system_cost_proxy_ms'],C_G_ms=d['G_system_cost_proxy_ms'],gap_ms=gap,
                    entry_margin_ms=p['entry_margin_ms'],exit_margin_ms=p['exit_margin_ms'],
                    S_wait_est_ms=d['S_wait_est_ms'],S_exec_est_ms=d['S_exec_est_ms'],
                    G_wait_est_ms=d['G_wait_est_ms'],G_exec_est_ms=d['G_exec_est_ms'],
                    S_interference_est_ms=d['S_interference_est_ms'],background_externality_est_ms=d['background_externality_est_ms'],
                    guards=gs,pre_state=state,post_state=e['router_after_decide'],
                    harvested_request_ids=[r['request_id'] for r in e['harvested_completions']],
                    scope='ACTUAL_DYNAMIC_DECISION_NOT_S_ONLY_COUNTERFACTUAL')
                trace.append(row);decisions.append(e)
            elif e['harvested_completions'] or router_state(replay)!=e['router_after_decide']:
                issues.append('non-target modified router:'+rid)
    csvsave('routing_decision_trace.csv',trace)
    ph=raw.get('phases',{}).get('measurement',{});rs=ph.get('request_records',[])
    warm=raw.get('warmup',{}).get('request_records',[])
    all_cohort=rs+warm+raw.get('path_probe_records',[])
    for r in all_cohort:
        if r['status']=='completed' and r['request_id'] not in commit:issues.append('missing successful commit:'+r['request_id'])
    expected_arrivals={(r['phase'],r['request_id'],r['task']) for r in rs+warm if r.get('enqueued_time_ns')}
    if set(arrivals)!=expected_arrivals or len(arrivals)!=len(expected_arrivals):issues.append('arrival log coverage mismatch')
    byid={r['request_id']:r for r in rs+warm}
    for e in decisions:
        r=byid.get(e['request_id'])
        if r is None or r['model']!=e['selected_worker']:issues.append('selected/actual worker mismatch:'+e['request_id'])
        elif r['scheduled_time_ns']!=e['scheduled_time_ns']:issues.append('schedule mismatch:'+e['request_id'])
    assert ref(OUT/'dynamic_router.py')['sha256']==manifest['original_router_source']['sha256']
    if ph:
        win=ph['window'];start=win['start_time_ns'];end=win['end_time_ns'];duration=win['duration_seconds']
        success=[r for r in rs if r['status']=='completed'];target=[r for r in rs if r['task']==p['target']]
        g=[r for r in target if r['model']==p['G_worker']];s=[r for r in target if r['model']==p['S_worker']]
        within=[r for r in success if r['completion_time_ns']<end]
        sent=[r for r in rs if r.get('actual_submit_time_ns') and r['actual_submit_time_ns']<end]
        aliases={w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for w,m in raw['replica_to_family'].items()}
        temp=dict(request_records=rs,measurement_window=win,errors=raw['errors'],shutdown=raw['shutdown'],
            status='completed' if raw['valid_session'] and len(success)==len(rs) else 'completed_with_failures' if raw['valid_session'] else 'failed',
            drain_timeout_seconds=60.,worker_identities=raw['worker_identities'])
        ev=reconstruct(temp,aliases,DEFAULT_THRESHOLDS);stability=classify(ev,DEFAULT_THRESHOLDS)
        save('stability_unchanged_classifier.json',dict(classifier_source=ref(ROOT/'performance_model/local_calibration/stability_v2.py'),
            thresholds=DEFAULT_THRESHOLDS,result=stability,evidence=ev,does_not_reclassify_Module28=True))
        reconciliation=dict(formal_seconds=duration,offered=len(rs),actually_sent_within_window=len(sent),
            successful_within_window=len(within),unfinished_at_window_end=len(sent)-len(within),
            successful_during_drain=len(success)-len(within),successful_total_after_drain=len(success),
            offered_qps=len(rs)/duration,completed_within_window_qps=len(within)/duration,
            drain_seconds=win['drain_seconds'],actual_G_target_sent=len(g),actual_S_target_sent=len(s),
            actual_G_target_successful=sum(r['status']=='completed' for r in g),
            actual_S_target_successful=sum(r['status']=='completed' for r in s),
            actual_G_target_fraction=len(g)/len(target) if target else None,
            cohort_matches_original_frozen_trajectory=len(rs)==7607,
            legal_U=None,delta_U=None,scope='Single functional validation; not formal utility evidence')
        save('window_reconciliation.json',reconciliation)
        metrics=[]
        for task in sorted(f['rates']):
            rr=[r for r in rs if r['task']==task];dd=[r for r in rr if r['status']=='completed']
            metrics.append(dict(task=task,offered=len(rr),success=len(dd),within=sum(r['completion_time_ns']<end for r in dd),
                mean_e2e_ms=stats([r['e2e_latency_ms'] for r in dd])['mean_ms'],
                p95_e2e_ms=stats([r['e2e_latency_ms'] for r in dd])['p95_ms'],
                worker_service=stats([r['worker_latency_ms'] for r in dd]),
                worker_queue=stats([r['worker_queue_wait_ms'] for r in dd]),
                mean_instrumented_arrival_us=sum(r['instrumented_arrival_total_us'] for r in rr)/len(rr) if rr else None))
        csvsave('six_task_metrics.csv',metrics)
        intervals={r['request_id']:response_native(r,start) for r in success}
        givs=[(iv[0],iv[1],r['request_id']) for r in g if r['status']=='completed' and (iv:=intervals.get(r['request_id']))]
        native_rows=[];strata=collections.defaultdict(list);overlap_pairs=[]
        for r in success:
            iv=intervals.get(r['request_id'])
            if iv is None:issues.append('native interval missing:'+r['request_id']);continue
            if iv[1]<iv[0]:issues.append('negative native interval:'+r['request_id'])
            overlap=[(max(a,iv[0]),min(b,iv[1]),rid) for a,b,rid in givs if rid!=r['request_id'] and max(a,iv[0])<min(b,iv[1])]
            ms=sum((b-a)*1000 for a,b,_ in overlap)
            quality={k:r.get('worker_response',{}).get(k) for k in ['correct','wer','cer','exact_match','valid_closed_set_output'] if k in r.get('worker_response',{})}
            row=dict(request_id=r['request_id'],task=r['task'],worker=r['model'],
                scheduled_t_s=(r['scheduled_time_ns']-start)/1e9,
                actual_submit_t_s=(r['actual_submit_time_ns']-start)/1e9,
                native_start_t_s=iv[0],native_end_t_s=iv[1],response_t_s=(r['completion_time_ns']-start)/1e9,
                collector_commit_t_s=(r['collector_commit_time_ns']-start)/1e9,
                service_ms=r['worker_latency_ms'],worker_queue_ms=r['worker_queue_wait_ms'],
                e2e_ms=r['e2e_latency_ms'],overlap_G_target_service_ms=ms,
                overlapping_G_target_ids=[rid for _,_,rid in overlap],quality_diagnostic=quality,
                native_time_scope='Retrospective per-request Unix/perf paired mapping; never router input',
                handler_time_scope='Worker elapsed handler time includes CPU and GPU; not CUDA kernel duration')
            native_rows.append(row);strata[(r['model'],r['task'],ms>0)].append(row)
            if r['model']==p['S_worker']:
                for a,b,rid in overlap:overlap_pairs.append(dict(G_target_request_id=rid,S_request_id=r['request_id'],
                    S_task=r['task'],overlap_start_t_s=a,overlap_end_t_s=b,overlap_ms=(b-a)*1000,
                    S_service_ms=r['worker_latency_ms'],S_queue_ms=r['worker_queue_wait_ms']))
        csvsave('request_service_timeline.csv',native_rows);csvsave('S_G_execution_overlaps.csv',overlap_pairs)
        interference=[]
        for (w,t,overlapping),rr in sorted(strata.items()):
            other=strata.get((w,t,not overlapping),[])
            service=stats([r['service_ms'] for r in rr]);other_service=stats([r['service_ms'] for r in other])
            interference.append(dict(worker=w,task=t,overlap_G_target_service=overlapping,n=len(rr),
                service=service,queue=stats([r['worker_queue_ms'] for r in rr]),e2e=stats([r['e2e_ms'] for r in rr]),
                other_overlap_stratum_n=len(other),service_mean_difference_to_other_ms=service['mean_ms']-other_service['mean_ms'] if other else None,
                scope='DESCRIPTIVE_ASSOCIATION_ONLY; overlap groups differ in time, queue, input and thermal state; no causal inflation estimate'))
        csvsave('worker_interference_analysis.csv',interference)
        # Background requests serialized after G-target work: distinguish queueing from GPU coexecution.
        gbackground=[]
        ordered=sorted([r for r in success if r['model']==p['G_worker']],key=lambda r:intervals[r['request_id']][0])
        for j,r in enumerate(ordered):
            if r['task']==p['target']:continue
            before=ordered[j-1] if j else None
            submit=(r['actual_submit_time_ns']-start)/1e9
            submit_overlap=[rid for a,b,rid in givs if a<=submit<b]
            gbackground.append(dict(request_id=r['request_id'],task=r['task'],worker_queue_ms=r['worker_queue_wait_ms'],
                service_ms=r['worker_latency_ms'],e2e_ms=r['e2e_latency_ms'],
                preceding_G_request_id=before['request_id'] if before else None,
                preceding_G_request_task=before['task'] if before else None,
                G_target_executing_at_submit_ids=submit_overlap,scope='OBSERVED_SERIAL_ORDER; causal background delay UNSUPPORTED'))
        csvsave('G_background_queue_context.csv',gbackground)
        first_g=next((e for e in decisions if e['phase']=='measurement' and e['selected_worker']==p['G_worker']),None)
        first_record=byid[first_g['request_id']] if first_g else None
        if first_g:
            first=dict(request_id=first_g['request_id'],decision_t_s=(first_g['time_ns']-start)/1e9,
                route_reason=first_g['decision']['reason'],decision=first_g['decision'],guards=first_g['guards'],
                pre_state=first_g['router_before_decide'],post_state=first_g['router_after_decide'],
                pending=first_g['pending'],successful=first_record['status']=='completed',
                service_ms=first_record['worker_latency_ms'],worker_queue_ms=first_record['worker_queue_wait_ms'],
                completion_t_s=(first_record['completion_time_ns']-start)/1e9 if first_record.get('completion_time_ns') else None,
                native_interval_t_s=intervals.get(first_g['request_id']))
        else:first=None
        save('first_actual_diversion.json',first)
        overhead=dict(router_decide_only_us=stats([r['routing_decision']['router_decide_only_us'] for r in target]),
            router_instrumented_observation_decision_us=stats([r['routing_decision']['router_observation_and_decision_us'] for r in target]),
            instrumented_arrival_total_us=stats([r['instrumented_arrival_total_us'] for r in rs]),
            note='All overhead units are microseconds. Decide-only times the unchanged function; observation/decision and arrival totals include state-copy instrumentation and lock acquisition.')
        # Rename overhead units explicitly to avoid misleading fields.
        overhead={k:{kk.replace('_ms','_us'):vv for kk,vv in v.items()} if isinstance(v,dict) else v for k,v in overhead.items()}
        save('router_overhead.json',overhead)
        quality=[]
        for w,t in sorted({(r['model'],r['task']) for r in success}):
            rr=[r for r in success if r['model']==w and r['task']==t]
            known=[r['worker_response']['correct'] for r in rr if 'correct' in r['worker_response']]
            invalid=sum(r['worker_response'].get('valid_closed_set_output') is False for r in rr)
            quality.append(dict(worker=w,task=t,n=len(rr),correct_available=len(known),correct_true=sum(x is True for x in known),
                accuracy=sum(x is True for x in known)/len(known) if known else None,closed_set_invalid=invalid,
                scope='Original catalog admission preserved; subset diagnostic does not certify deployment quality'))
        csvsave('quality_diagnostics.csv',quality)
        peak=max((v['owned_used_mib'] for v in raw['telemetry']['samples']),default=None)
        safe_idle=[f['safe_factor']*sum(v['owned_used_mib'] for v in raw.get(key,[]))/len(raw[key]) for key in ['postwarm_idle','postmeasurement_idle'] if raw.get(key)]
        mem=peak is not None and peak<=f['budget_mib'] and len(safe_idle)==2 and max(safe_idle)<=f['budget_mib']
        native_g_complete=all(intervals.get(r['request_id']) and r.get('worker_queue_wait_ms') is not None and r.get('worker_latency_ms') is not None for r in g if r['status']=='completed')
        passed=bool(raw['valid_session'] and not issues and len(g)>0 and len(s)>0
            and all(r['status']=='completed' for r in g+s) and native_g_complete and overlap_pairs and mem)
        gate=dict(status='CONTROLLED_SPLIT_LOGGING_VERIFIED' if passed else 'CONTROLLED_VALIDATION_FAILED_OR_INCOMPLETE',
            functional_logging_passed=passed,actual_G_target_successful=reconciliation['actual_G_target_successful'],
            actual_S_target_successful=reconciliation['actual_S_target_successful'],
            logging_replay_issues=issues,log_event_counts=dict(counts),
            native_S_G_overlap_pairs=len(overlap_pairs),memory_passed=mem,sampled_owned_peak_mib=peak,
            safe_idle_mib=safe_idle,budget_mib=f['budget_mib'],stability=stability,
            legal_U=None,delta_U=None,positive_utility_evidence=False,independent_replication=False,
            formal_GPU_comparison_started=False,original_Module28_gate='DYNAMIC_EVIDENCE_GAP',optimizer_modified=False,
            evidence_scope='Actual split execution and logging only; interference changes descriptive and confounded',
            recommendation='Functional chain ready for preregistered matched GPU comparison design; retain original hard stability and quality gates' if passed else 'Resolve observed execution/logging gap before designing formal comparison',
            next_experiment_automatic=False)
        if stability['stability_state_v2']=='unstable':
            gate['recommendation']='Execution/logging may pass, but original load remains hard-unstable; not ready for legal comparison at this load. No automatic retuning or experiment.'
        save('stage2_gate.json',gate)
        summary=(f'# Module29 第二阶段：冻结动态分流执行与日志验证\n\n'
            f'状态：**{gate["status"]}**。仅执行一次GPU会话，正式600秒；没有固定分流/S-only新对照、正式效用比较或优化器修改。\n\n'
            f'原动态路由器代码逐字节相同，参数、模型、单副本、显存预算、六任务QPS及原稳定性判据未改。复用旧冻结轨迹，故不属于独立复现。\n\n'
            f'正式窗口发送{len(sent)}，窗口内成功{len(within)}，窗口末未完成{reconciliation["unfinished_at_window_end"]}；排空{win["drain_seconds"]:.6f}秒后累计成功{len(success)}。提供{len(rs)/duration:.6f}QPS，窗口内完成{len(within)/duration:.6f}QPS。\n\n'
            f'LID真实路由到G共{len(g)}个、成功{reconciliation["actual_G_target_successful"]}个；到S共{len(s)}个、成功{reconciliation["actual_S_target_successful"]}个。G实际比例{reconciliation["actual_G_target_fraction"]:.4%}。路径probe/warmup不计入这些分流证据。\n\n')
        if first:
            summary+=(f'首个实际S→G分流：`{first["request_id"]}`，决策t={first["decision_t_s"]:.9f}秒，'
                f'S pending={first["decision"]["S_pending"]}，G pending={first["decision"]["G_pending"]}，'
                f'C_S={first["decision"]["S_system_cost_proxy_ms"]:.6f}ms，C_G={first["decision"]["G_system_cost_proxy_ms"]:.6f}ms。'
                f'服务{first["service_ms"]:.6f}ms，成功响应t={first["completion_t_s"]:.9f}秒。\n\n')
        summary+=(f'完整日志记录collector提交、已接收completion harvest、实际观测时点、完整pending组成、决策前后EWMA/预测/令牌/滞回/冷却/回退状态及逐项守卫。'
            f'从phase初始化连续重放实际动态决策，发现{len(issues)}项一致性问题；原生handler START/END只用于事后分析，没有作为路由输入。\n\n'
            f'记录到{len(overlap_pairs)}对S执行与G目标执行时间重叠，逐请求服务/等待、按任务及重叠分层的服务变化、G背景串行队列上下文均已输出。'
            f'这些是可分析的观测关联；没有匹配新对照，不能将服务差异归因于分流，也不能声称S排队改善或动态正收益。\n\n'
            f'采样显存峰值{peak:.1f}MiB，安全idle显存{safe_idle}，预算{f["budget_mib"]}MiB。原分类器结果：**{stability["stability_state_v2"]}**。'
            f'本验证不计算合法U或ΔU，均为NULL。质量子集诊断单列，保持原准入；不以成功响应代替质量合格。\n\n'
            f'建议：{gate["recommendation"]}。本次完成后停止，不自动开展正式对照。\n\n'
            f'原Module28 Gate、unstable与U=NULL保持不变；输入哈希核验见input_integrity_after.json。\n')
        (OUT/'summary.md').write_text(summary)
    else:
        save('stage2_gate.json',dict(status='CONTROLLED_VALIDATION_FAILED_OR_INCOMPLETE',raw_status=raw['status'],errors=raw['errors'],issues=issues,formal_GPU_comparison_started=False,legal_U=None,delta_U=None,optimizer_modified=False,next_experiment_automatic=False))
        (OUT/'summary.md').write_text('# Module29 第二阶段\n\n正式窗口未完成。唯一尝试完整保留，不重试。详见gpu_validation_raw.json与stage2_gate.json。\n')
    protected=json.loads((OUT/'input_freeze.json').read_text())
    changes=[r['path'] for r in protected['inputs'] if not Path(r['path']).is_file() or ref(r['path'])['sha256']!=r['sha256'] or Path(r['path']).stat().st_size!=r['bytes']]
    current=sorted(str(x) for x in (ROOT/'results/module28_dynamic_split').rglob('*') if x.is_file())
    assert not changes and current==protected['module28_files'],changes
    save('input_integrity_after.json',dict(unchanged=True,checked=len(protected['inputs']),changes=[],module28_file_set_unchanged=True))
    artifacts=sorted(str(x.resolve()) for x in OUT.iterdir() if x.is_file())
    save('artifact_paths.json',dict(output_directory=str(OUT),created_file_paths=sorted(set(artifacts+[str(OUT/'artifact_paths.json')]))))
    print(json.dumps(json.loads((OUT/'stage2_gate.json').read_text()),ensure_ascii=False))

if __name__=='__main__':main()
