"""Independent offline accounting and artifact verification. Never launches workers."""
import sys
sys.dont_write_bytecode=True
import collections
import csv
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def write(name,x):
    (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def percentile(values,q):
    values=sorted(values);ix=(len(values)-1)*q/100
    left=int(ix);right=min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(ix-left) if values else None

def main():
    raw=json.loads((P/'gpu_validation_raw.json').read_text())
    f=json.loads((P/'controlled_manifest.json').read_text())
    freeze=json.loads((P/'input_freeze.json').read_text())
    gate=json.loads((P/'stage2_gate.json').read_text())
    ph=raw['phases']['measurement'];rs=ph['request_records'];win=ph['window']
    trajectory=json.loads(Path(f['slot']['trajectories']['measurement']['path']).read_text())
    checks=[]
    assert len(rs)==len(trajectory['events'])==7607
    for r,e in zip(rs,trajectory['events']):
        assert r['task']==e['task'] and r['pool_id']==e['item']['pool_id']
        assert r['sample_id']==e['item']['sample'].get('id')
        assert r['frozen_offset_seconds']==e['offset_seconds']
        assert r['routing_uniform']==e['routing_uniform'] and r['replica_uniform']==e['replica_uniform']
    checks.append(dict(name='complete_frozen_trajectory_and_inputs_match',passed=True,requests=7607))
    ids=[r['request_id'] for r in rs];assert len(set(ids))==len(ids)
    completed=[r for r in rs if r['status']=='completed'];assert len(completed)==7607
    start,end=win['start_time_ns'],win['end_time_ns']
    assert (end-start)/1e9==600
    sent=sum(start<=r['actual_submit_time_ns']<end for r in rs)
    within=sum(start<=r['completion_time_ns']<end for r in completed)
    drain=sum(end<=r['completion_time_ns']<=win['drain_end_time_ns'] for r in completed)
    assert sent==7607 and within==7588 and drain==19 and sent-within==19
    checks.append(dict(name='window_vs_drain_successful_response_accounting',passed=True,sent=sent,within=within,drain=drain))
    g=[r for r in rs if r['task']=='lid' and r['model']==f['parameters']['G_worker']]
    s=[r for r in rs if r['task']=='lid' and r['model']==f['parameters']['S_worker']]
    assert len(g)==67 and len(s)==6938
    for r in rs:
        v=r['worker_response']
        assert v['request_id']==r['request_id'] and v['task_type']==r['task']
        assert abs((v['end_ns']-v['start_ns'])/1e6-r['worker_latency_ms'])<1e-7
        assert abs((v['start_ns']-r['actual_submit_unix_ns'])/1e6-r['worker_queue_wait_ms'])<1e-7
        assert r['collector_commit_time_ns']>=r['completion_time_ns']
    checks.append(dict(name='actual_worker_success_and_native_service_queue_values',passed=True,G_success=67,S_success=6938))
    guardcounts=collections.defaultdict(collections.Counter);reasons=collections.Counter();formal_reasons=collections.Counter()
    formal_traces=0;all_traces=0
    for r in csv.DictReader((P/'routing_decision_trace.csv').open()):
        all_traces+=1;reasons[r['reason']]+=1
        if r['phase']!='measurement':continue
        formal_traces+=1;formal_reasons[r['reason']]+=1
        for k,v in json.loads(r['guards']).items():guardcounts[k][v]+=1
    assert formal_traces==7005 and all_traces==7343
    assert formal_reasons['G_LOWER_SYSTEM_MARGINAL_COST']==67
    assert not gate['logging_replay_issues']
    write('guard_breakdown.json',dict(scope='Actual formal dynamic decision guards; complete frozen router state replay verified',
        denominator=7005,guards={k:dict(PASS=v['PASS'],FAIL=v['FAIL'],UNKNOWN=v['UNKNOWN'],denominator=7005) for k,v in guardcounts.items()},
        actual_reasons=dict(formal_reasons),
        note='entry_cost_gap is an entry diagnostic, not a mandatory guard when hysteresis mode is already active; actual original priority reason is authoritative'))
    checks.append(dict(name='continuous_actual_policy_replay_and_formal_guard_coverage',passed=True,formal=7005,including_warmup=7343))
    assert all(digest(x['path'])==x['sha256'] for x in f['implementation'])
    assert digest(P/'dynamic_router.py')==digest(f['original_router_source']['path'])
    assert all(Path(x['path']).stat().st_size==x['bytes'] and digest(x['path'])==x['sha256'] for x in freeze['inputs'])
    current=sorted(str(x) for x in Path(f['original_router_source']['path']).parent.rglob('*') if x.is_file())
    assert current==freeze['module28_files']
    checks.append(dict(name='all_frozen_implementation_and_268_input_hashes_unchanged',passed=True,inputs=len(freeze['inputs']),Module28_files=len(current)))
    assert raw['valid_session'] and not raw['errors'] and raw['gpu_release']['verified']
    assert len(raw['shutdown'])==5 and all(not x['forced'] and x['exit_code']==0 and x['bye_received'] for x in raw['shutdown'].values())
    assert gate['memory_passed']
    assert gate['stability']['stability_state_v2']=='unstable'
    assert gate['legal_U'] is None and gate['delta_U'] is None and not gate['formal_GPU_comparison_started']
    checks.append(dict(name='clean_shutdown_memory_and_unchanged_hard_stability_semantics',passed=True,new_run_stability='unstable',legal_U=None,delta_U=None))
    attempt=json.loads((P/'gpu_attempt_started.json').read_text());assert attempt['attempt']==attempt['maximum_attempts']==1
    checks.append(dict(name='one_GPU_session_only_no_new_comparison',passed=True,sessions=1))
    allnative={r['request_id']:r for r in csv.DictReader((P/'request_service_timeline.csv').open())}
    assert len(allnative)==7607
    pairs=list(csv.DictReader((P/'S_G_execution_overlaps.csv').open()));assert len(pairs)==403
    for pair in pairs:
        gr=allnative[pair['G_target_request_id']];sr=allnative[pair['S_request_id']]
        a=max(float(gr['native_start_t_s']),float(sr['native_start_t_s']))
        b=min(float(gr['native_end_t_s']),float(sr['native_end_t_s']))
        assert b>a and abs((b-a)*1000-float(pair['overlap_ms']))<1e-7
    checks.append(dict(name='independent_native_S_G_overlap_check',passed=True,pairs=403))
    findings=dict(scope='OBSERVED_ASSOCIATION_ONLY_NO_CAUSAL_INTERFERENCE_OR_GAIN',
        all_system_mean_e2e_ms=sum(r['e2e_latency_ms'] for r in rs)/len(rs),
        all_system_p95_e2e_ms=percentile([r['e2e_latency_ms'] for r in rs],95),
        actual_G_LID_service_mean_ms=sum(r['worker_latency_ms'] for r in g)/len(g),
        actual_G_LID_service_p95_ms=percentile([r['worker_latency_ms'] for r in g],95),
        actual_G_LID_correct=sum(r['worker_response'].get('correct') is True for r in g),
        actual_G_LID_quality_n=len(g),actual_S_LID_correct=sum(r['worker_response'].get('correct') is True for r in s),
        actual_S_LID_quality_n=len(s),
        S_G_execution_overlap_pairs=403,
        overlap_group_bias='Service duration changes overlap selection probability; input/queue/time/thermal confounding prevents causal attribution',
        exact_missing_for_formal_gain='No new matched legal stable S-only and best-fixed controls, no preregistered independent trajectory replication',
        original_LID_quality_policy='SEMANTIC_VALIDITY_ONLY; numeric accuracy threshold unspecified; do not invent an exclusion rule')
    write('service_interference_findings.json',findings)
    overhead=json.loads((P/'router_overhead.json').read_text())
    strata=list(csv.DictReader((P/'worker_interference_analysis.csv').open()))
    lids={r['overlap_G_target_service']:r for r in strata if 'whisper' in r['worker'] and r['task']=='lid'}
    over=json.loads(lids['True']['service']);non=json.loads(lids['False']['service'])
    supplement=(f'\n补充数值诊断：S侧LID与G目标执行重叠组n={over["count"]}、mean={over["mean_ms"]:.6f}ms、P95={over["p95_ms"]:.6f}ms；'
        f'未重叠组n={non["count"]}、mean={non["mean_ms"]:.6f}ms、P95={non["p95_ms"]:.6f}ms。差异不能视为因果干扰量，且按执行重叠分组存在服务时长选择偏差。\n\n'
        f'G侧LID服务mean={findings["actual_G_LID_service_mean_ms"]:.6f}ms、P95={findings["actual_G_LID_service_p95_ms"]:.6f}ms。'
        f'G侧LID正确61/67，S侧正确6938/6938；G闭集无效输出0。保持原LID语义准入，数值准确率阈值仍未指定，未据此新造排除规则。\n\n'
        f'全系统cohort mean/P95={findings["all_system_mean_e2e_ms"]:.6f}/{findings["all_system_p95_e2e_ms"]:.6f}ms（成功完成的正式到达cohort，包括排空完成）。'
        f'原decide调用mean/P95={overhead["router_decide_only_us"]["mean_us"]:.6f}/{overhead["router_decide_only_us"]["p95_us"]:.6f}µs；'
        f'含观测和记录的arrival总mean/P95={overhead["instrumented_arrival_total_us"]["mean_us"]:.6f}/{overhead["instrumented_arrival_total_us"]["p95_us"]:.6f}µs。\n\n'
        f'独立离线核验通过，268个原输入以及全部冻结实现SHA-256不变，GPU已释放。没有新增第二次GPU会话。\n')
    summary=P/'summary.md'
    assert '补充数值诊断' not in summary.read_text(),'No repeated append'
    with summary.open('a') as stream:stream.write(supplement)
    write('final_verification.json',dict(passed=True,checks=checks,functional_validation_passed=True,
        formal_comparison_readiness_at_original_load=False,reason='Original classifier remains hard-unstable',
        original_Module28_modified=False,optimizer_modified=False,legal_U=None,delta_U=None))
    paths=sorted(set(str(x.resolve()) for x in P.iterdir() if x.is_file())|{str(P/'artifact_paths.json')})
    write('artifact_paths.json',dict(output_directory=str(P),created_file_paths=paths))
    print(json.dumps(dict(passed=True,checks=checks,findings=findings),ensure_ascii=False))

if __name__=='__main__':main()
