"""Finite direct-JSON descriptive diagnosis; standard library only, no fitting."""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

MODES = ('shared_spill_low', 'admission1_spill_low', 'admission2_spill_low')


def require(condition, reason):
    if not condition: raise ValueError(reason)


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stats(values):
    values = sorted(values)
    def quantile(q):
        x = (len(values)-1)*q; i = int(x)
        return values[i]+(values[min(i+1, len(values)-1)]-values[i])*(x-i)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=values[0], maximum=values[-1], p95=quantile(.95))


def compare(predicted, actual):
    delta = [a-p for p, a in zip(predicted, actual)]
    positive = [(p, a) for p, a in zip(predicted, actual) if a > 0]
    return dict(predicted=stats(predicted), actual=stats(actual), delta_actual_minus_predicted=stats(delta),
                mean_absolute_error=statistics.mean(abs(x) for x in delta),
                maximum_absolute_error=max(abs(x) for x in delta),
                mean_absolute_relative_error=statistics.mean(abs(p-a)/a for p, a in positive) if positive else None,
                relative_error_denominator='actual > 0 only', relative_error_defined_n=len(positive),
                relative_error_undefined_zero_actual_n=len(actual)-len(positive))


def summarize(rows):
    result = dict(n=len(rows))
    for metric in ('service_ms', 'wait_ms', 'e2e_ms', 'remaining_ms'):
        result[metric] = compare([r['predicted'][metric] for r in rows], [r['actual'][metric] for r in rows])
    ps = sum(r['predicted']['service_ms'] for r in rows); ac = sum(r['actual']['service_ms'] for r in rows)
    result['cumulative_worker_service_wall_ms'] = dict(predicted=ps, actual=ac, delta_actual_minus_predicted=ac-ps,
        delta_fraction_of_actual=(ac-ps)/ac, scope='Sum of worker handler walls, not physical GPU demand or busy time')
    pf = max(r['predicted_finish_ns'] for r in rows); af = max(r['actual_finish_ns'] for r in rows)
    result['tail_completion'] = dict(predicted_last_finish_ns=pf, actual_last_parent_finish_ns=af,
        delta_last_finish_ms=(af-pf)/1e6,
        paired_finish_delta_ms=stats([(r['actual_finish_ns']-r['predicted_finish_ns'])/1e6 for r in rows]),
        delta_of_maximum_wait_ms=result['wait_ms']['actual']['maximum']-result['wait_ms']['predicted']['maximum'],
        delta_of_maximum_e2e_ms=result['e2e_ms']['actual']['maximum']-result['e2e_ms']['predicted']['maximum'])
    result['mean_error_decomposition_ms'] = {m: result[m]['delta_actual_minus_predicted']['mean'] for m in ('service_ms','wait_ms','remaining_ms','e2e_ms')}
    result['max_identity_residual_ms'] = max(abs(r['decomposition_identity_residual_ms']) for r in rows)
    return result


def diagnose(sealed, report, sealed_ref):
    require(report['sealed'] == sealed_ref and report['status'] == 'DIAGNOSTIC_COMPLETE'
            and report['requested'] == report['paired_known'] == 3600, 'Complete exact three-mode report required')
    predictions = {p['candidate_id']: p for p in sealed['predictions']}
    candidates = {p['candidate_id']: p for p in report['candidates']}
    require(len(predictions) == len(sealed['predictions']) == 3 and len(candidates) == len(report['candidates']) == 3
            and set(predictions) == set(candidates) == set(MODES), 'Unique exact mode domain required')
    all_rows = []; modes = {}
    for mode in MODES:
        p = predictions[mode]; c = candidates[mode]; a = c['actual_evidence']
        require(c['complete'] is True and c['evidence_error'] is None
                and c['requested'] == c['predicted_known'] == c['observed_known'] == c['paired_known'] == 1200,
                'Full paired mode required')
        pr = {r['request_id']: r for r in p['requests']}; ar = {r['request_id']: r for r in a['requests']}
        require(len(pr) == len(p['requests']) == len(ar) == len(a['requests']) == 1200 and set(pr) == set(ar), 'Strict one-to-one full request pairing required')
        rows = []
        for rid in pr:
            left, right = pr[rid], ar[rid]
            require(all(left[k] == right[k] for k in ('task','pool_id','alias')), 'Paired task/input/alias mismatch')
            for side in (left, right):
                require(all(type(side[k]) in (int,float) and math.isfinite(side[k]) and side[k] >= 0
                            for k in ('service_ms','wait_ms','e2e_ms','start_ns','finish_ns')), 'Finite known existing timing required')
            # Existing START - wait recovers the scheduled offer in the phase-relative clock.
            po = left['start_ns']-round(left['wait_ms']*1e6); ao = right['start_ns']-round(right['wait_ms']*1e6)
            require(abs(po-ao) <= 2, 'Predicted and actual recovered offer differ')
            for side in (left, right):
                require(abs(side['finish_ns']-po-side['e2e_ms']*1e6) <= 3, 'Existing finish/e2e/offer inconsistency')
            row = dict(mode=mode, request_id=rid, alias=left['alias'], task=left['task'], pool_id=left['pool_id'],
                       offer_offset_ns=po, predicted_finish_ns=left['finish_ns'], actual_finish_ns=right['finish_ns'])
            for name, side in [('predicted',left),('actual',right)]:
                row[name] = {k:side[k] for k in ('service_ms','wait_ms','e2e_ms')}
                row[name]['remaining_ms'] = side['e2e_ms']-side['service_ms']-side['wait_ms']
                require(row[name]['remaining_ms'] >= -1e-6, 'Negative unexplained existing-field remainder')
            row['delta_actual_minus_predicted_ms'] = {k:row['actual'][k]-row['predicted'][k] for k in row['actual']}
            d = row['delta_actual_minus_predicted_ms']
            row['decomposition_identity_residual_ms'] = d['e2e_ms']-d['service_ms']-d['wait_ms']-d['remaining_ms']
            rows.append(row)
        rows.sort(key=lambda r:(r['offer_offset_ns'],int(r['request_id'].split(':')[-1])))
        aliases = {}
        for alias in sorted({r['alias'] for r in rows}):
            group = [r for r in rows if r['alias'] == alias]; half = len(group)//2
            prefix = []; ps = ac = 0.
            for row in group:
                ps += row['predicted']['service_ms']; ac += row['actual']['service_ms']
                prefix.append(dict(request_id=row['request_id'], offer_offset_ns=row['offer_offset_ns'],
                                   predicted_service_cumulative_ms=ps, actual_service_cumulative_ms=ac, delta_ms=ac-ps))
            aliases[alias] = dict(all=summarize(group), first_half=summarize(group[:half]), second_half=summarize(group[half:]),
                                  cumulative_service_prefix=prefix)
        pq = {q['offset_ns']:q['depth'] for q in p['queue_samples']}; aq = {q['offset_ns']:q['depth'] for q in a['queue_samples']}
        require(len(pq) == len(p['queue_samples']) == len(aq) == len(a['queue_samples']) and set(pq) == set(aq), 'Strict identical queue sample grid required')
        require(all(type(v) is int and v >= 0 for v in [*pq.values(),*aq.values()]), 'Known queue depths required')
        queue_rows = [dict(offset_ns=t,predicted_depth=pq[t],actual_depth=aq[t],delta_actual_minus_predicted=aq[t]-pq[t]) for t in sorted(pq)]
        queue = compare([q['predicted_depth'] for q in queue_rows], [q['actual_depth'] for q in queue_rows])
        queue.update(samples=queue_rows, signed_depth_time_trapezoid_seconds=sum(
            (x['delta_actual_minus_predicted']+y['delta_actual_minus_predicted'])/2*(y['offset_ns']-x['offset_ns'])/1e9
            for x,y in zip(queue_rows,queue_rows[1:])),
            exact_depth_match_samples=sum(q['predicted_depth']==q['actual_depth'] for q in queue_rows),
            scope='Existing whole-mode outstanding scheduled-but-not-parent-completed depth; not waiting-only or alias-specific. Trapezoid is sample-grid description, not exact continuous queue area.')
        half = len(rows)//2
        modes[mode] = dict(existing_field_definitions={k:p[k] for k in ('service_definition','wait_definition','e2e_definition','admission_model_limits')},
            all=summarize(rows), first_half=summarize(rows[:half]), second_half=summarize(rows[half:]),
            final_ten_percent_by_offer=summarize(rows[-120:]), aliases=aliases, queue=queue)
        all_rows.extend(rows)
    return modes, all_rows


def write_markdown(summary, output):
    lines = ['# Pilot 已知到达下的队列误差：描述性分解', '', 'UTC: '+summary['created_at_utc'], '',
        '仅解析已有 predictions/report 两份 JSON，三模式各 1200 个请求完整配对；request_id 一对一且 task/pool_id/alias 相同。由既有 start−wait 恢复的到达时刻亦一致。未读取原生 raw、拟合、回填、运行验证器或选择 controller；本诊断不替代 V4 门槛。', '',
        '全过程到达已知只固定 offer，不能固定 worker 服务时间、实际准入 ticket/release 与父进程回收。已有模型把 permit 在 handler END 释放，实际 ticket/release 边界并未重建。服务墙钟偏差会改变后续可用时刻，积累为 wait 偏差；这些只是当前单个 cohort 的描述，不是因果识别。', '',
        '下文 Δ = actual − predicted，正值表示模型低估。MARE = mean(abs(pred−actual)/actual)，分母为正的请求；queue 实际为零的样本不计算相对误差并单列计数。service 是 worker START→handler END 墙钟，累计值是所有请求墙钟求和，不是 GPU demand/busy。', '',
        '| mode / alias | n | 累计service预测/实际 s | Δservice均值 ms | service MARE | Δwait均值 ms | Δe2e均值 ms | Δ剩余均值 ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,m in summary['modes'].items():
        for alias,g in m['aliases'].items():
            x=g['all'];cum=x['cumulative_worker_service_wall_ms'];d=x['mean_error_decomposition_ms']
            lines.append(f"| {mode} / {alias} | {x['n']} | {cum['predicted']/1000:.2f}/{cum['actual']/1000:.2f} | {d['service_ms']:.2f} | {x['service_ms']['mean_absolute_relative_error']:.3f} | {d['wait_ms']:.2f} | {d['e2e_ms']:.2f} | {d['remaining_ms']:.2f} |")
    lines += ['', '剩余项直接定义为 e2e−service−wait。预测 finish 指 handler END，实际 finish 指 parent receipt；因此实际剩余包括 handler END 之后的父进程传输/执行外时间，绝不重新命名为 service。逐请求恒等式 Δe2e=Δservice+Δwait+Δ剩余在浮点舍入内成立。', '',
        '| mode | Δwait 前半/后半均值 ms | Δe2e 前半/后半均值 ms | 最后完成时刻Δ ms | queue MAE / max绝对误差 | queue平均预测/实际 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,m in summary['modes'].items():
        f=m['first_half'];s=m['second_half'];q=m['queue']
        lines.append(f"| {mode} | {f['wait_ms']['delta_actual_minus_predicted']['mean']:.2f}/{s['wait_ms']['delta_actual_minus_predicted']['mean']:.2f} | {f['e2e_ms']['delta_actual_minus_predicted']['mean']:.2f}/{s['e2e_ms']['delta_actual_minus_predicted']['mean']:.2f} | {m['all']['tail_completion']['delta_last_finish_ms']:.2f} | {q['mean_absolute_error']:.2f}/{q['maximum_absolute_error']:.0f} | {q['predicted']['mean']:.2f}/{q['actual']['mean']:.2f} |")
    lines += ['', '前/后半按 offer 排序，在 mode 汇总各 600 个请求；alias 子表按该 alias offer 顺序各分一半。它们是时间位置描述，不能当作独立重复。所有 wait/e2e 的预测和实际 mean/max、signed/absolute误差、尾部请求与最后完成偏差、101 个队列采样及累计服务前缀均在 summary.json。queue 为全 mode 的已到达未完成数，包含正在执行者，不是纯等待队列；不虚构 alias queue。', '',
        '### 各模式观察', '']
    for mode,m in summary['modes'].items():
        d=m['all']['mean_error_decomposition_ms'];cum=m['all']['cumulative_worker_service_wall_ms']
        lines.append(f"- {mode}：累计 worker service Δ={cum['delta_actual_minus_predicted']/1000:.2f}s ({cum['delta_fraction_of_actual']*100:.2f}% of actual)；平均 Δservice={d['service_ms']:.2f}ms、Δwait={d['wait_ms']:.2f}ms、Δe2e={d['e2e_ms']:.2f}ms。")
    lines += ['', '汇总 signed bias 可在不同 alias 间抵消，应连同逐 alias、MARE 与前后半结果解读。单次 cohort 无法把服务律、准入顺序、计时边界与运行噪声的影响因果拆开；不支持新容量、GPU 互补或部署收益结论。', '',
        '### 直接来源 SHA256', '']
    for ref in summary['input_receipts_after']: lines.append('- '+ref['path']+': `'+ref['sha256']+'`')
    lines += ['', '两份 JSON 在解析前和发布前 SHA256 一致；只写入新诊断目录，未修改已有来源或报告。', '']
    with (output/'report.md').open('x') as stream: stream.write('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output.resolve();require(not output.exists(), 'New diagnostic output required')
    before=[receipt(args.sealed),receipt(args.report)]
    sealed,report=[json.loads(Path(r['path']).read_text()) for r in before]
    modes,rows=diagnose(sealed,report,before[0]);after=[receipt(r['path']) for r in before]
    require(before==after, 'Direct source changed during diagnosis')
    summary=dict(schema='gsdeploy.pilot_queue_error_decomposition.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_receipts_before=before,input_receipts_after=after,direct_SHA256_unchanged=True,script=receipt(__file__),
        paired_requests=len(rows),delta_definition='actual minus predicted; positive means underprediction',
        decomposition='delta_e2e = delta_worker_service + delta_wait + delta_existing_field_remainder',
        modes=modes,all_modes=summarize(rows),GPU_started=False,fitting_performed=False,
        native_raw_read=False,frozen_module_imports=False,expensive_verify_performed=False,
        V4_gate_replacement=False,controller_selection=False)
    require(len(rows)==3600 and summary['all_modes']['max_identity_residual_ms']<1e-7, 'Full cohort decomposition identity failed')
    output.mkdir(parents=True)
    for name,value in [('summary',summary),('requests',rows)]:
        with (output/(name+'.json')).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    write_markdown(summary,output)
    print(json.dumps(dict(output=str(output),paired_requests=len(rows),max_identity_residual_ms=summary['all_modes']['max_identity_residual_ms'],direct_SHA256_unchanged=True)))


if __name__=='__main__':main()
