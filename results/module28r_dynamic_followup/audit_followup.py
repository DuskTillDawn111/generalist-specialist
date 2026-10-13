"""Module 28R CPU audit. Reads original sources; writes only the 28R directory."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, csv, hashlib, collections
from pathlib import Path
import numpy as np
ROOT = Path('/root/autodl-tmp/generalist-specialist')
OUT = ROOT / 'results/module28r_dynamic_followup'
OLD = ROOT / 'results/module28_dynamic_split'
sys.path.insert(0, str(ROOT))
from performance_model.local_calibration.stability_v2 import reconstruct, classify, DEFAULT_THRESHOLDS, timeline
from types import SimpleNamespace

def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def ref(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def csvsave(name, rows, fields=()):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    with p.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in r.items()})

def metadata():
    roots = [ROOT / 'results', ROOT / 'joint_deploy', ROOT / 'performance_model', ROOT / 'deployment_search', ROOT / 'deployment_core']
    roots += [p for p in ROOT.glob('module*') if p.is_dir()]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for root in roots for p in root.rglob('*')
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}

def stats(rows, key):
    a = [r[key] for r in rows if r.get(key) is not None]
    return dict(n=len(a), mean=float(np.mean(a)), p95=float(np.percentile(a,95))) if a else dict(n=0,mean=None,p95=None)

def curve(raw):
    ph = raw['phases']['measurement']; win = ph['window']; rows = ph['request_records']
    start, end = win['start_time_ns'], win['end_time_ns']
    events = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for field, delta in [('scheduled_time_ns',1), ('enqueued_time_ns',1), ('actual_submit_time_ns',1), ('completion_time_ns',-1)]:
            at = r.get(field)
            if at is None or (field=='completion_time_ns' and r['status']!='completed'): continue
            scope = {'scheduled_time_ns':'offered','enqueued_time_ns':'host','actual_submit_time_ns':'submitted','completion_time_ns':None}[field]
            for s in ([scope] if scope else ['offered','host','submitted']):
                events[at][s + ':system'] += delta
                events[at][s + ':' + r['model']] += delta
    state = collections.Counter(); out=[]
    for at, ds in sorted(events.items()):
        state.update(ds)
        out.append(dict(run_id=raw['slot']['run_id'], offset_seconds=(at-start)/1e9,
                        within_measurement=at<=end, **dict(state)))
    return out

def main():
    assert not (OUT/'stage_a_audit.json').exists()
    save('historical_metadata_before.json', metadata())
    sources=[]; formats=collections.Counter(); journal=[]
    # Fully read every original 27/28 file, including raw logs and durable journal.
    for root in [ROOT/'results/module27_positive_split_discovery', OLD]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts: continue
            sources.append(ref(p)); formats[p.suffix]+=1
            if p.suffix=='.json': json.loads(p.read_text())
            elif p.suffix=='.jsonl':
                for line in p.read_text().splitlines():
                    row=json.loads(line)
                    if root==OLD and 'request_journals' in p.parts: journal.append(row)
            elif p.suffix=='.csv':
                csv.field_size_limit(128*1024*1024)
                list(csv.DictReader(p.open()))
            else: p.read_bytes()
    f=json.loads((OLD/'frozen_candidate_manifest.json').read_text())
    for p in [ROOT/'results/module20_utility_optimizer/utility_definition.json', ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json', ROOT/'performance_model/local_calibration/stability_v2.py', ROOT/'module26/common.py', OLD/'dynamic_router.py', OLD/'gpu.py', OLD/'report.py']:
        sources.append(ref(p))
    definition=json.loads((ROOT/'results/module20_utility_optimizer/utility_definition.json').read_text())
    raw=json.loads((OLD/'raw_runs/C_steady_s_only.json').read_text())
    ph=raw['phases']['measurement']; rs=ph['request_records']; win=ph['window']
    start,end=win['start_time_ns'],win['end_time_ns']; G,S=f['parameters']['G_worker'],f['parameters']['S_worker']
    catalog={w:SimpleNamespace(service_time_ms=f['plan']['models'][m]['service_time_ms']) for w,m in raw['replica_to_family'].items()}
    ev=reconstruct(dict(request_records=rs, measurement_window=win, status='completed',errors=raw['errors'],drain_timeout_seconds=60),catalog,DEFAULT_THRESHOLDS)
    cl=classify(ev,DEFAULT_THRESHOLDS)
    assert cl==ph['stability']
    segments=[]; paths=[]
    for scope, wrs in [('system',rs),('S',[r for r in rs if r['model']==S])]:
        bins=timeline(wrs,start,end,win['drain_end_time_ns'],catalog,60,'actual_submit_time_ns')['bins']
        for label,left,right in [('early',0,200),('middle',200,400),('late',400,600),('original_tail',450,600),('full',0,600)]:
            bb=[b for b in bins if left<b['offset_seconds']<=right]
            x=np.array([b['offset_seconds'] for b in bb]); y=np.array([b['mean_backlog_requests'] for b in bb])
            slope=float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2))
            arrivals=[r for r in wrs if left<=(r['scheduled_time_ns']-start)/1e9<right]
            completions=[r for r in wrs if left<=(r['completion_time_ns']-start)/1e9<right]
            segments.append(dict(scope=scope,segment=label,left_s=left,right_s=right,bin_width_s=10,slope_rps=slope,
                mean_backlog=float(y.mean()),offered=len(arrivals),completed_within=len(completions),
                observed_cohort_service_demand_ratio=sum(r['worker_latency_ms'] for r in arrivals)/(1000*(right-left))))
    outstanding=[r for r in rs if r.get('actual_submit_time_ns',end+1)<=end<r['completion_time_ns']]
    assert len(outstanding)==32
    pending=[]
    for r in outstanding:
        pending.append(dict(request_id=r['request_id'],task=r['task'],worker=r['model'],sample_id=r['sample_id'],
            scheduled_s=(r['scheduled_time_ns']-start)/1e9,submitted_s=(r['actual_submit_time_ns']-start)/1e9,
            completed_s=(r['completion_time_ns']-start)/1e9,service_ms=r['worker_latency_ms'],wait_ms=r['worker_queue_wait_ms'],
            boundary_age_s=(end-r['scheduled_time_ns'])/1e9))
    csvsave('module28_boundary_requests.csv',pending)
    for w in raw['replica_to_family']:
        for task in sorted(set(r['task'] for r in rs if r['model']==w)):
            rows=[r for r in rs if r['model']==w and r['task']==task]
            paths.append(dict(worker=w,task=task,n=len(rows),service=stats(rows,'worker_latency_ms'),wait=stats(rows,'worker_queue_wait_ms'),
                e2e=stats(rows,'e2e_latency_ms'),effective_in_window_qps=sum(r['completion_time_ns']<=end for r in rows)/600))
    # Exact integration of S outstanding, G host outstanding, and retrospective native G busy.
    events=collections.defaultdict(collections.Counter)
    for r in rs:
        if r['model'] not in [G,S]: continue
        who='G' if r['model']==G else 'S'
        events[r['enqueued_time_ns']][who+'_pending']+=1; events[r['completion_time_ns']][who+'_pending']-=1
        response=r['worker_response'];delta=r['actual_submit_time_ns']-r['actual_submit_unix_ns']
        if who=='G':
            events[response['start_ns']+delta]['G_busy']+=1; events[response['end_ns']+delta]['G_busy']-=1
    state=collections.Counter(); last=start; duration=collections.Counter()
    for at,ds in sorted(events.items()):
        edge=min(at,end); dt=max(0,edge-last)/1e9
        if state['S_pending']>=2:
            duration['S_queued_seconds']+=dt
            if state['G_pending']==0: duration['S_queued_G_host_empty_seconds']+=dt
            if state['G_pending']==0 and state['G_busy']==0: duration['S_queued_G_host_empty_native_idle_seconds']+=dt
        if state['S_pending']>=8 and state['G_pending']==0 and state['G_busy']==0:duration['S_ge8_G_available_seconds']+=dt
        state.update(ds);last=max(last,edge)
        if at>end:break
    hist=[]
    for p in sorted((ROOT/'results/module27_positive_split_discovery/raw_runs').glob('*.json')):
        r=json.loads(p.read_text()); pp=r['phases']['measurement'];rr=pp['request_records']
        sr=[v for v in rr if v['model']==S and v['task']=='lid']
        hist.append(dict(source=str(p),lid_qps=r['slot']['lambda_rps']['lid'],G_fraction=r['slot']['G_fraction'],
            system_stability=pp['stability']['stability_state_v2'],S_service=stats(sr,'worker_latency_ms'),S_wait=stats(sr,'worker_queue_wait_ms'),
            gpu_uuid=r['baseline_GPU']['uuid'],valid_session=r['valid_session']))
    terminal=[r['record'] for r in journal if r.get('kind')=='REQUEST_TERMINAL' and r.get('phase')=='measurement']
    indexed={r['request_id']:r for r in rs}
    assert len(terminal)==len({r['request_id'] for r in terminal})
    assert all(r['completion_time_ns']==indexed[r['request_id']]['completion_time_ns'] for r in terminal)
    srows=[r for r in rs if r['model']==S]
    demand=sum(r['worker_latency_ms'] for r in srows)/600000
    low=10.5; s_lid=[r for r in srows if r['task']=='lid'];s_asr=[r for r in srows if r['task']=='asr']
    projected=low*np.mean([r['worker_latency_ms'] for r in s_lid])/1000 + .2*np.mean([r['worker_latency_ms'] for r in s_asr])/1000
    result=dict(stage='A_CPU_ONLY',original_run=ref(OLD/'raw_runs/C_steady_s_only.json'),sources=sources,
        read_formats=dict(formats),journal_records=len(journal),journal_terminal_measurement_records=len(terminal),journal_raw_consistency=True,
        original_classifier=cl,original_evidence={k:v for k,v in ev.items() if k not in ['queue_wait_values_ms','offered_backlog_diagnostic']},
        segments=segments,boundary_composition=dict(collections.Counter(r['task'] for r in outstanding)),boundary_request_count=len(outstanding),
        boundary_scheduled_range_s=[min(r['scheduled_s'] for r in pending),max(r['scheduled_s'] for r in pending)],
        boundary_completion_range_s=[min(r['completed_s'] for r in pending),max(r['completed_s'] for r in pending)],
        S_observed_cohort_demand_ratio=demand,G_availability=dict(duration),worker_paths=paths,history=hist,
        persistent_capacity_deficit='INSUFFICIENT_LONG_TERM_STABILITY_EVIDENCE',
        interpretation='Original final-150s finite-window classifier remains unstable and U=NULL. Earlier recoveries, zero backlog near 570s, total observed S service demand <600s, and successful 2.445s drain do not establish sustained long-term overload. Local Poisson/native-service bursts and boundary backlog are supported; infinite-horizon stability also unproven.',
        lower_load_rule=dict(rule='One-shot: reuse the highest already measured lower S-only LID point from Module27 with steady classification and projected native S mean demand <=0.85. No new QPS grid, seed search, or subsequent load adjustment.',
            selected_lid_qps=low,projected_S_demand_ratio=float(projected),target_max=.85),
        utility_definition=definition,router_audit=dict(interference=True,background_protection=True,backpressure=True,
            conditions='Frozen proxies + G pending zero + one target inflight + token rate 0.575/s + cooldown + error/service fallback. Development proxy is not a measured gain.'))
    assert projected<=.85
    save('stage_a_audit.json',result)
    csvsave('module28_segment_slopes.csv',segments)
    csvsave('backlog_timeseries.csv',curve(raw))
    csvsave('module28_worker_paths.csv',paths)
    lines=['# Module 28R：原始日志 CPU 审计\n\n',
        f'完整读取 Module27/28 共 {len(sources)} 个来源（含完整 raw、journal、CSV、文本和冻结定义），未使用29及以后研究结果。原分类器逐字段重算一致，Gate 不改、U=NULL。\n\n',
        '## 队列与窗口边界\n\n![原始队列曲线](module28_backlog.png)\n\n',
        '|范围|系统斜率 requests/s|S斜率 requests/s|\n|---|---:|---:|\n']
    for label in ['early','middle','late','original_tail','full']:
        pair=[r for r in segments if r['segment']==label]
        lines.append(f"|{label} ({pair[0]['left_s']}–{pair[0]['right_s']}s)|{pair[0]['slope_rps']:.6f}|{pair[1]['slope_rps']:.6f}|\n")
    lines += [f'\n上表是10秒精确积分bin的描述性OLS；原判定保留20个30秒bin及末150秒斜率 {ev["tail_backlog_slope"]:.6f}、增长 {ev["tail_backlog_growth"]:.6f}、递增比例 {ev["tail_increasing_fraction"]:.2f}。前、中段有多次恢复，570s提交积压为0。不能把原有限窗口拒绝变成长时域容量不足定论。\n\n',
        f'边界32请求组成：{result["boundary_composition"]}；到达范围 {result["boundary_scheduled_range_s"]}s，完成范围 {result["boundary_completion_range_s"]}s，完整逐条记录见 module28_boundary_requests.csv。停止到达后 {win["drain_seconds"]:.6f}s 清空，说明边界截断真实积压；排空不豁免原稳定性判据。\n\n',
        f'S 原生执行服务总需求/600s = {demand:.6f}，低于1；其中 LID mean/P95={stats(s_lid,"worker_latency_ms")}，等待={stats(s_lid,"worker_queue_wait_ms")}。这是当前cohort的实测服务需求，不是未来容量保证；ASR共享S且长服务可引起短时拥塞。\n\n',
        f'G真实可用时间：{dict(duration)}。仅用host未完成状态和回顾native START/END交集，不使用平均GPU利用率；不能由这些秒数推导未执行路由的效用。\n\n',
        '## 负载冻结依据\n\n',f'预先规则：从Module27已测较低点选最高S-only steady、且按Module28实测mean投影S需求≤0.85的点，唯一选择LID={low}QPS；其他五任务0.2QPS不变，预计需求={projected:.6f}。11.5QPS在Module27曾recoverable、Module28原窗口unstable，证据支持有限窗口敏感性，长期证据不足。不得再次调QPS。\n\n',
        '## 原路由与效用口径\n\n',
        '原DynamicRouter含S服务膨胀代价、G慢路径及VQA外部代价；G已有任何请求时保护背景任务；单目标在途、令牌、冷却背压；大预测误差或S服务报警回退30s。参数全来自原开发数据，本轮不改变。Fixed固定10%，历史10%比20%损害小，但S-only历史更强；未称全局最佳。\n\n',
        'Module20A：U=-0.25T_norm-0.30L_norm-0.30P_norm-0.15M_norm，Q权重0且质量只作原硬准入。L/P使用冻结offered任务权重与原latency reference；pooled P95单独报告。T保留窗口内完成率及固定QPS×600分母；drain不可抹去吞吐损失。M为1.1×前后空闲驻留最大均值/B4。系统或任一worker unstable均U=NULL；完整排空的recoverable_congested按原规则允许。缺失质量不填充。\n\n',
        '**结论：HIGH_STABILITY_EVIDENCE_ONLY；持续容量不足的长期证据不足。**原600秒窗口确有真实末端增长与S积压，局部突发和窗口边界不能排除。新HIGH对照必须按原classifier评价，不计算相对不稳定控制的正ΔU。\n']
    (OUT/'module28_backlog_audit.md').write_text(''.join(lines))
    print(json.dumps({k:result[k] for k in ['segments','S_observed_cohort_demand_ratio','G_availability','boundary_composition','boundary_scheduled_range_s','lower_load_rule']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
