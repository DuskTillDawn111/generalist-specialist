"""Combine unchanged first-pass and separately frozen protocol-corrected runs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/retained-replica-profiles-20261021'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_retained_replica_outputs_20261021.py')],check=True)
    audit=read(OUT/'output-equivalence.json')
    rows=[];stops={};states={};artifacts={}
    for package in [OUT,OUT/'protocol-binding-correction-v2']:
        if not (package/'state.json').exists():continue
        state=read(package/'state.json');states[str(package)]=state['status'];stops.update(state.get('stopped',{}))
        release=read(package/'release.json');artifacts[str(package/'release.json')]=sha(package/'release.json')
        for entry in release['entries']:
            for n in entry['grid']:
                profile=package/'runs'/f'{entry["id"]}-n{n}'/'profile.json'
                if not profile.exists():continue
                p=read(profile);row=dict(id=entry['id'],slot=entry['slot'],task=entry['task'],replicas=n,
                    status=p['status'],profile=str(profile),package=str(package),
                    error=p.get('error'),failure_stage=p.get('failure_stage'),
                    board_peak_including_load_GiB=p.get('peak_board_used_mib',0)/1024)
                if p['status']!='COMPLETE':
                    partial=[]
                    for raw in profile.parent.glob(entry['task']+'-round*.jsonl'):
                        partial.extend(json.loads(v) for v in raw.read_text().splitlines())
                        artifacts[str(raw)]=sha(raw)
                    row.update(partial_formal_requests_returned=len(partial),
                        partial_formal_valid=sum(v.get('success',False) for v in partial),
                        partial_rounds_completed=len(p['rounds']),
                        partial_measurement_not_complete=True)
                    if (profile.parent/'warmup.json').exists():
                        warmups=read(profile.parent/'warmup.json')
                        failures=[v for v in warmups if v.get('status')!='RESULT' or not v.get('success',False)]
                        row['warmup_failure_details']=failures
                    row['partial_completed_round_energy_J']=sum(
                        v['board_energy']['joules'] for v in p['rounds'] if v['board_energy']['joules'] is not None)
                    row['partial_energy_scope']='Completed rounds only; failed/incomplete round raw board samples remain in telemetry.json. No completed-profile throughput inferred.'
                if p['status']=='COMPLETE':
                    rounds=p['rounds'];data=[];raws=list(profile.parent.glob(entry['task']+'-round*.jsonl'))
                    for raw in raws:
                        data.extend(json.loads(v) for v in raw.read_text().splitlines())
                        artifacts[str(raw)]=sha(raw)
                    valid=sum(v.get('success',False) for v in data)
                    errors=sum(not v.get('success',False) for v in data)
                    eq=next(a for a in audit['rows'] if a['id']==entry['id'] and a['replicas']==n and a['package']==str(package))
                    before=p['startup_energy']['end_ns'];telemetry=read(profile.parent/'telemetry.json')
                    warm=min(telemetry,key=lambda v:abs(v['time_ns']-before))
                    periods=[(r['board_energy']['start_ns'],r['board_energy']['end_ns']) for r in rounds]
                    active=[v for v in telemetry if any(start<=v['time_ns']<=end for start,end in periods)]
                    expected_pids={v['pid'] for v in p['ready']}
                    assert len(expected_pids)==n
                    assert all({q['pid'] for q in v['processes']}==expected_pids for v in active), str(profile)
                    counts={sid:sum(v['sample_id']==sid for v in data) for sid in {v['sample_id'] for v in data}}
                    row.update(valid=valid,requests=len(data),failed_requests=errors,
                        median_rps=statistics.median(r['successful_rps'] for r in rounds),
                        rps_by_round=[r['successful_rps'] for r in rounds],
                        aggregate_rps=valid/sum(r['wall_seconds'] for r in rounds),
                        mean_service_ms=statistics.mean(v['service_ms'] for v in data),
                        p50_service_ms=statistics.median(v['service_ms'] for v in data),
                        p95_service_ms=percentile([v['service_ms'] for v in data],.95),
                        mean_client_ms=statistics.mean(v['client_ms'] for v in data),
                        board_warm_GiB=p['resident_after_warmup_mib']/1024,
                        process_warm_sum_GiB=sum(v['used_mib'] for v in warm['processes'])/1024,
                        board_measurement_peak_GiB=max(v['used_mib'] for v in active)/1024,
                        simultaneous_process_peak_sum_GiB=max(sum(q['used_mib'] for q in v['processes']) for v in active)/1024,
                        board_mean_power_watts=sum(r['board_energy']['joules'] for r in rounds)/sum(
                            (r['board_energy']['end_ns']-r['board_energy']['start_ns'])/1e9 for r in rounds),
                        board_J_per_request=sum(r['board_energy']['joules'] for r in rounds)/len(data),
                        energy_coverage_valid=all(r['board_energy']['coverage_valid'] for r in rounds),
                        safe_peak_mib=p['peak_board_used_mib']*p['safe_factor'],budget_mib=p['budget_mib'],
                        unique_inputs=len(counts),min_input_repetitions=min(counts.values()),max_input_repetitions=max(counts.values()),
                        original_difference_requests=eq['changed_requests_vs_prior'],
                        n1_difference_requests=eq['changed_requests_vs_n1'],
                        gpu_process_isolation_verified=True,
                        all_replicas_used={v['replica'] for v in data}==set(range(n)))
                    assert len(rounds)==3 and len(data)==sum(r['completed_requests'] for r in rounds)
                    assert valid==len(data) and not errors and row['all_replicas_used']
                    assert row['safe_peak_mib']<=row['budget_mib']
                    assert row['energy_coverage_valid']
                artifacts[str(profile)]=sha(profile);rows.append(row)
    # Prefer corrected binding package for each of its three combinations, while
    # keeping every first-pass failure in a separate explicit list.
    corrected_ids={r['id'] for r in rows if r['package']!=str(OUT)}
    corrected_ids.update(e['id'] for e in read(OUT/'protocol-binding-correction-v2/release.json')['entries'])
    initial_failures=[r for r in rows if r['id'] in corrected_ids and r['package']==str(OUT)]
    current=[r for r in rows if not (r['id'] in corrected_ids and r['package']==str(OUT))]
    for r in current:
        singles=[v for v in current if v['id']==r['id'] and v['replicas']==1 and v['status']=='COMPLETE']
        if r['status']=='COMPLETE' and singles:r['median_rps_ratio_vs_n1']=r['median_rps']/singles[0]['median_rps']
    complete=all(s=='COMPLETE_RESOURCE_GRID' for s in states.values()) and len(states)==2
    completed_ids={r['id'] for r in current if r['status']=='COMPLETE'}
    if complete:assert len(completed_ids)==16,completed_ids
    # A ratio of these measured traces is descriptive. It is not a controlled
    # all-input-frequency-matched causal estimate when min-duration repeats differ.
    limits=['Closed-loop resource sample; no open-loop SLO, full-data worst-case memory, mixed tasks or mixed-model capacity.',
        'Identical unique input sets at all counts, but the 30s minimum can repeat a partial cycle: actual per-input frequencies and request counts differ. Ratios are descriptive.',
        'Three rounds share loaded model state. Cache, checkpoint filesystem cache, CPU contention and GPU scheduling are uncontrolled; round values are preserved.',
        'Warmup exercises the three largest inputs by the declared size proxy. Size proxies do not prove worst-case output length or memory.',
        'Sampled NVML peaks may miss shorter spikes. Whole-board power integral includes idle board power and excludes CPU energy.',
        'Protocol binding corrections for Baichuan image/ASR/OCR are frozen separately; original failed runs remain evidence.',
        'Resource-grid completion does not confer whole-pool admission or revise quality thresholds.']
    result=dict(status='COMPLETE_VERIFIED_RESOURCE_GRID' if complete else 'RUNNING_PARTIAL',
        completed_model_task_combinations=len(completed_ids),expected_combinations=16,
        pool_admitted=False,states=states,rows=current,initial_protocol_binding_failures=initial_failures,
        stop_receipts=stops,limitations=limits,artifact_sha256=artifacts)
    save(OUT/'combined-summary.json',result)
    columns=['id','replicas','status','requests','median_rps','median_rps_ratio_vs_n1',
        'mean_service_ms','p50_service_ms','p95_service_ms','process_warm_sum_GiB',
        'board_warm_GiB','board_measurement_peak_GiB','board_peak_including_load_GiB',
        'board_mean_power_watts','board_J_per_request','original_difference_requests','n1_difference_requests','error']
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(current)
    (OUT/'combined-summary.csv').write_text(stream.getvalue())
    names={'generalist_1':'百川去合成','generalist_2':'MiniCPM-o4.5 GPTQ4','generalist_3':'Qwen2.5-Omni BNB4',
        'text':'ChatQA INT8+LoRA','image':'SigLIP2 FP32','ocr':'PP-OCRv6 FP32','asr':'Granite4.1 FP32'}
    lines=['# 七模型多副本资源画像','',f'状态：{result["status"]}；已覆盖 {len(completed_ids)}/16 个模型与任务组合。','',
        '|模型|任务|副本|状态|吞吐 req/s|相对n1|平均/p95延迟 ms|进程驻留合计 GiB|整卡测量峰 GiB|平均整卡功率 W|整卡 J/req|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(current,key=lambda r:(r['slot'],r['task'],r['replicas'])):
        if r['status']=='COMPLETE':
            lines.append(f'|{names[r["slot"]]}|{r["task"]}|{r["replicas"]}|完成|{r["median_rps"]:.3f}|{r.get("median_rps_ratio_vs_n1",1):.2f}|{r["mean_service_ms"]:.0f}/{r["p95_service_ms"]:.0f}|{r["process_warm_sum_GiB"]:.2f}|{r["board_measurement_peak_GiB"]:.2f}|{r["board_mean_power_watts"]:.1f}|{r["board_J_per_request"]:.1f}|')
        else:lines.append(f'|{names[r["slot"]]}|{r["task"]}|{r["replicas"]}|{r["status"]}: {r["error"]}；部分返回{r["partial_formal_requests_returned"]}次|||||{r["board_peak_including_load_GiB"]:.2f}（含加载）|||')
    lines += ['', '吞吐为三个轮次的中位数；平均与p95延迟按全部实测请求计算。每档独立加载，每副本一个进程、batch=1。',
        '平均整卡功率为三轮正式测量窗口的NVML功率积分总能量除以窗口总时长，按时长加权，包含整卡基础功耗，不含加载、预热或轮次间隔。',
        '整卡J/req为上述总能量除以三轮请求总数；平均功率对应全程聚合吞吐，不用表中的轮次吞吐中位数反推。失败档位不报告完整测量平均功率。',
        '唯一输入集固定（图像64，其他32），按输入大小分层并含极值。每轮至少30秒会导致不同档位重复次数不同，',
        '表中倍率只描述该闭环轨迹，不能据此断言严格同频输入下的因果加速；各轮值、实际次数见 JSON/CSV。',
        '每档三轮使用同一加载状态。加载、缓存、系统调度状态并未完全控制。没有恢复QA训练或修改准入门槛。',
        '', '提前停止或排除：']
    for key,v in sorted(stops.items()):lines.append('- '+key+': '+json.dumps(v,ensure_ascii=False))
    lines+=['','原协议绑定失败保留：'+str(len(initial_failures))+' 档；修正包使用原已验证新数据集协议，原生解码和输入不变。',
        '完整逐请求输出差异见 output-equivalence.json；不宣称所有模型逐字稳定。','',*['- '+s for s in limits]]
    (OUT/'combined-report.zh.md').write_text('\n'.join(lines)+'\n')
    receipt_path=OUT/'completion-receipt.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        receipt['reports']={path:sha(path) for path in receipt['reports']}
        save(receipt_path,receipt)
    print(result['status'],len(completed_ids),'/16 combinations;',len(current),'profiles')


def percentile(values,q):
    values=sorted(values);i=(len(values)-1)*q;left=int(i)
    return values[left]+(values[min(left+1,len(values)-1)]-values[left])*(i-left)


if __name__=='__main__':main()
