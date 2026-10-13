"""Keep aborted records diagnostic; do not reclassify or assign a finite U."""
from common_replication import *

def main():
    f=json.loads((OUT/'frozen_replication_manifest.json').read_text())
    gate=json.loads((OUT/'module29_gate.json').read_text());assert gate['final'] and gate['gate']=='REPLICATION_INVALID'
    csv.field_size_limit(128*1024*1024)
    runs=list(csv.DictReader((OUT/'independent_gpu_runs.csv').open()));tasks=[];failures=[];curves=[];stability=[];decisions=[];workers=[]
    for p in (OUT/'raw_runs').glob('*.json'):
        raw=json.loads(p.read_text());ph=raw.get('aborted_phase')
        if not ph:continue
        s=raw['slot'];rs=ph['request_records'];stop=ph['abort_time_ns'];start=ph['planned_start_time_ns'];end=ph['planned_end_time_ns']
        done=[r for r in rs if r['status']=='completed'];accepted=[r for r in rs if r.get('enqueued_time_ns') is not None]
        offered_before=[r for r in rs if r['scheduled_time_ns']<=stop]
        base=dict(run_id=s['run_id'],group=s['group'],seed=s['seed'],arm=s['arm'])
        evidence=dict(**base,planned_seconds=ph['planned_duration_seconds'],observed_seconds_until_abort=(stop-start)/1e9,
            frozen_planned_requests=len(rs),scheduled_before_abort=len(offered_before),never_scheduled_before_abort=len(rs)-len(offered_before),
            accepted=len(accepted),completed_after_clean_shutdown=len(done),completed_before_abort=sum(r['completion_time_ns']<=stop for r in done),
            unaccepted_before_abort=len(offered_before)-len(accepted),incomplete_in_registered_cohort=len(rs)-len(done),
            status_counts=dict(collections.Counter(r['status'] for r in rs)),errors=raw['errors'],telemetry_errors=raw['telemetry']['errors'],
            GPU_release_verified=raw['gpu_release']['verified'],all_worker_clean_exit=all(r['exit_code']==0 and not r['forced'] and r['bye_received'] for r in raw['shutdown'].values()),
            U=None,physical_instability='NOT_ESTABLISHED_BY_TECHNICAL_ABORT',
            eligible=False,late_or_missing_requests_not_dropped=True,
            prefix_latency_scope='CENSORED_TECHNICALLY_ABORTED_DIAGNOSTIC_ONLY; never a 2400s utility comparator')
        failures.append(evidence)
        for row in runs:
            if row['run_id']!=s['run_id']:continue
            row.update(formal_seconds=2400.,actual_seconds_until_abort=evidence['observed_seconds_until_abort'],
                frozen_planned_requests=len(rs),offered=len(offered_before),accepted=len(accepted),completed=len(done),
                offered_qps=len(offered_before)/2400.,accepted_qps=len(accepted)/2400.,completed_qps=len(done)/2400.,
                within_qps=sum(r['completion_time_ns']<=end for r in done)/2400.,
                counts_scope='Observed scheduled prefix and accepted requests; planned denominator stays 2400s; future frozen slots kept separately',
                U=None,eligible=False,system_state='UNASSESSABLE_TECHNICAL_ABORT',
                actual_G=sum(r['task']=='lid' and r['model']==f['parameters']['G_worker'] for r in done),
                actual_S=sum(r['task']=='lid' and r['model']==f['parameters']['S_worker'] for r in done),
                diagnostic_pooled_mean_ms=float(np.mean([r['e2e_latency_ms'] for r in done])),
                diagnostic_pooled_p95_ms=float(np.percentile([r['e2e_latency_ms'] for r in done],95)),
                diagnostic_latency_usable_for_comparison=False,errors=raw['errors'])
        for task in f['rates']:
            rr=[r for r in done if r['task']==task]
            tasks.append(dict(**base,task=task,planned_requests=sum(r['task']==task for r in rs),
                scheduled_before_abort=sum(r['task']==task for r in offered_before),accepted=sum(r['task']==task for r in accepted),
                completed=len(rr),e2e=stats(rr,'e2e_latency_ms'),service=stats(rr,'worker_latency_ms'),queue=stats(rr,'worker_queue_wait_ms'),
                eligible=False,U=None,scope='CENSORED_TECHNICALLY_ABORTED_DIAGNOSTIC_ONLY'))
        for w in raw['replica_to_family']:
            for task in sorted({r['task'] for r in done if r['model']==w}):
                rr=[r for r in done if r['model']==w and r['task']==task]
                workers.append(dict(**base,worker=w,task=task,n=len(rr),service=stats(rr,'worker_latency_ms'),
                    queue=stats(rr,'worker_queue_wait_ms'),e2e=stats(rr,'e2e_latency_ms'),eligible=False,
                    scope='CENSORED_TECHNICALLY_ABORTED_DIAGNOSTIC_ONLY'))
        for r in rs:
            if 'routing_decision' not in r:continue
            d=r['routing_decision'];assert d['latest_observed_completion_time_ns']<=d['decision_time_ns']
            decisions.append(dict(**base,request_id=r['request_id'],sample_id=r['sample_id'],request_status=r['status'],
                offset_seconds=r['frozen_offset_seconds'],actual_route=r['model'],service_ms=r.get('worker_latency_ms'),
                queue_ms=r.get('worker_queue_wait_ms'),e2e_ms=r.get('e2e_latency_ms'),
                scope='TECHNICALLY_ABORTED_PREFIX_DIAGNOSTIC_ONLY',**d))
        # Exclude never-arrived future slots from observed backlog, while retaining
        # them in the immutable raw/frozen cohort and registered denominator.
        prefix=copy.deepcopy(raw);prefix['phases']['measurement']=dict(request_records=offered_before,
            window=dict(start_time_ns=start,end_time_ns=end))
        for row in curve(prefix):
            row.update(observed_until_abort_seconds=evidence['observed_seconds_until_abort'],
                after_technical_abort=row['offset_seconds']>evidence['observed_seconds_until_abort'],
                scope='OBSERVED_PREFIX_PLUS_CLEAN_SHUTDOWN; future frozen arrivals excluded, no stability or U comparison')
            curves.append(row)
        for scope,w in [('system',None)]+[('worker',w) for w in raw['replica_to_family']]:
            stability.append(dict(**base,scope=scope,worker=w,state='UNASSESSABLE_TECHNICAL_ABORT',
                hard_reasons=['TELEMETRY_ERROR','REGISTERED_WINDOW_INCOMPLETE'],eligible=False,U=None,
                classification_scope='Full registered window unavailable; no inference of physical capacity instability'))
    csvsave('independent_gpu_runs.csv',runs,['run_id','group','seed','arm','U','eligible'])
    csvsave('aborted_task_diagnostics.csv',tasks,['run_id','task','scope'])
    csvsave('aborted_backlog_timeseries.csv',curves,['run_id','offset_seconds','scope'])
    failedids={r['run_id'] for r in failures}
    def merge(name,rows,fields):
        existing=[r for r in csv.DictReader((OUT/name).open()) if r['run_id'] not in failedids]
        csvsave(name,existing+rows,fields)
    existing=[r for r in csv.DictReader((OUT/'backlog_stability.csv').open()) if r['run_id'] not in failedids]
    csvsave('backlog_stability.csv',existing+stability,['run_id','scope','state'])
    merge('routing_decision_analysis.csv',decisions,['run_id','request_id','actual_route','reason'])
    merge('worker_latency_analysis.csv',workers,['run_id','worker','task'])
    merge('per_task_latency.csv',tasks,['run_id','task'])
    save('technical_stop_audit.json',failures)
    completed=sum(r['status']=='COMPLETED' and str(r['eligible']).lower()=='true' for r in runs)
    gate.update(attempted_runs=5,complete_legal_2400s_windows=completed,technical_abort_runs=len(failures),
        unexecuted_registered_runs=sum(r['status']=='NOT_EXECUTED' for r in runs),complete_independent_three_arm_groups=1,
        paired_independent_n=1,requested_three_pair_mean_std_CI_available=False,
        observed_group1_dynamic_vs_S_negative=True,technical_stop=failures,
        abort_not_evidence_of_S_capacity_deficit=True,zero_GPU_retries_observed=True)
    save('module29_gate.json',gate)
    failure=failures[0]
    text=['# Module 29 技术停止审计\n\n',
        f'第 5 次 G2 S-only 在 {failure["observed_seconds_until_abort"]:.6f}/2400 秒出现遥测错误 `dictionary changed size during iteration`。错误由 telemetry collector 捕获；没有保存该异常的具体 Python 栈帧，不能唯一定位触发语句。\n\n',
        '源码存在一致的竞态解释：gpu_replication.py 的 DurableMonitor.sample 在未持有 runtime.lock 时 deepcopy 正被 sender/collector 更新的请求字典；deployment_runtime.py 对这些字典的更新持有 runtime.lock。原始监测器来自 Module 28/28R。此为源码支持的候选原因，未用新 GPU 实验验证、未修补冻结代码。\n\n',
        f'冻结队列共 {failure["frozen_planned_requests"]} 个请求；中止前计划到达 {failure["scheduled_before_abort"]}，接受 {failure["accepted"]}；清洁关闭后已接受请求完成 {failure["completed_after_clean_shutdown"]}。未来 {failure["never_scheduled_before_abort"]} 个计划槽和异常时尚未接受的 {failure["unaccepted_before_abort"]} 个请求全部保留；共 {failure["incomplete_in_registered_cohort"]} 个注册请求未完成。\n\n',
        '五个 worker 均 BYE、exit=0、未强杀；GPU 释放已验证。未发现已接受请求错误、服务崩溃、GPU OOM 或温度停止证据。遥测异常使第 5 次会话及完整 2400 秒窗口无效；不得用它的完成前缀计算 U，也不能据此声称 S-only 物理不稳定。\n\n',
        '冻结 failure_policy 为 ZERO GPU retries/replacements，并在技术故障时停止整轮。因此剩余四次未运行；不修复后重跑本轮、不换种子、不补 GPU。Gate 保持 REPLICATION_INVALID。第一组合法负收益保留，开发窗口不计入统计。\n']
    (OUT/'technical_failure_analysis.md').write_text(''.join(text))
    print('TECHNICAL_STOP_AUDIT_DONE',failure['observed_seconds_until_abort'],failure['accepted'])

if __name__=='__main__':main()
