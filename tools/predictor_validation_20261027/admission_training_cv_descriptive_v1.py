"""Describe completed T8 manifest/CV artifacts; no raw reads or recomputation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]/'artifacts/predictor-gpu-20261027'


def error(predicted, actual):
    known = predicted is not None and actual is not None
    delta = predicted-actual if known else None
    return dict(predicted=predicted, actual=actual, predicted_minus_actual=delta,
        absolute_error=abs(delta) if known else None,
        absolute_relative_error=abs(delta)/abs(actual) if known and actual != 0 else None)


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    receipts = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(BASE) or path.name in ('raw.json', 'scored-raw.json'):
            raise ValueError('Only small campaign report artifacts may be read')
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError('Direct report receipt SHA mismatch: '+str(path))
        receipts[str(path)] = dict(path=str(path), sha256=sha)
        return json.loads(data)

    def linked(ref):
        return read(ref['path'], ref['sha256'])

    completed = read(BASE/'admission-training-continuation-v1/completed.json')
    manifests = {kind: linked(completed['manifests'][kind+'_protocol'])
        for kind in ('capacity', 'resource', 'quality')}
    capacity = linked(manifests['capacity']['fit'])
    resources = linked(manifests['resource']['resource_cv'])
    quality = linked(manifests['quality']['quality_cv'])
    cap = {(r['seed'],r['candidate_id']): r for r in capacity['whole_seed_cv']}
    res = {(r['seed'],r['candidate_id']): r for r in resources['rows']}
    qual = {(r['seed'],r['candidate_id']): r for r in quality['rows']}
    if set(cap) != set(res) or set(cap) != set(qual):
        raise ValueError('The three actual CV domains differ')
    full = {}
    for kind, key in [('resource','route_resources'),('quality','route_quality')]:
        values = []
        for cid, ref in manifests[kind]['full'][key].items():
            model = linked(ref)
            evidence = model['admission_training_evidence'] if kind == 'resource' else model
            values.append(dict(candidate_id=cid, training_seeds=evidence['training_seeds'],
                source_run_count=len(evidence['runs']),
                raw_receipts=[r['raw'] for r in evidence['runs']],
                request_count=model.get('request_count'),
                quality_counts=[r['quality_audit']['count'] for r in evidence['runs']]
                    if kind == 'quality' else None))
        full[kind] = values
    rows = []
    for key in sorted(cap):
        c,r,q = cap[key],res[key],qual[key]
        p = c['prediction']
        if key[0] in c['training_seeds'] or key[0] in r['training_seeds'] or key[0] in q['training_seeds']:
            raise ValueError('Heldout seed is present in a retained training seed list')
        rows.append(dict(seed=key[0],candidate_id=key[1],training_seeds=c['training_seeds'],
            capacity=dict(service_applicability_diagnostic=c['service_applicability_diagnostic'],
                service_checks=c['service_checks'],max_family_service_are=max(
                    (check['absolute_relative_mean_error'] for check in c['service_checks']
                        if check['absolute_relative_mean_error'] is not None),default=None),
                rho_by_family=p['rho_by_family'],max_rho=max(p['rho_by_family'].values()),
                rho_threshold_satisfied=p['rho_threshold_satisfied'],threshold=p['threshold'],tolerance=p['tolerance'],
                original_qualification_passed=c['qualification_passed'],
                global_permit_capacity_qualified=p['global_permit_capacity_qualified']),
            resources=dict(actual_context_energy_replay=error(
                    r['actual_context_replay_predicted_measurement_and_drain_joules'],r['actual_measurement_and_drain_joules']),
                startup_joules=error(r['predicted_startup_joules'],r['actual_startup_joules']),
                owned_sampled_peak_mib=error(r['predicted_owned_sampled_peak_mib'],r['actual_owned_sampled_peak_mib']),
                unknown_contexts=r['unknown_contexts']),
            quality=dict(predicted_Q=q['predicted']['Q'],actual_Q=q['observed']['Q'],
                Q_absolute_error=q['Q_absolute_error'],
                heldout_request_count=sum(t['request_count'] for t in q['observed']['per_task'].values()),
                task_counts={task:value['request_count'] for task,value in q['observed']['per_task'].items()})))
    qcounts = [value['request_count'] for value in full['quality']]
    counts = dict(completed_training_jobs=completed['job_count'],cv_cells=len(rows),
        capacity_full_training_references=len(capacity['full_training_references']),
        capacity_cv_rows=len(capacity['whole_seed_cv']),resource_cv_rows=len(resources['rows']),quality_cv_rows=len(quality['rows']),
        heldout_quality_requests=sum(r['quality']['heldout_request_count'] for r in rows),
        full_quality_requests=sum(qcounts),
        full_resource_unique_native_raw_receipts=len({(ref['path'],ref['sha256'])
            for row in full['resource'] for ref in row['raw_receipts']}),
        full_quality_unique_native_raw_receipts=len({(ref['path'],ref['sha256'])
            for row in full['quality'] for ref in row['raw_receipts']}))
    summary = dict(schema='gsdeploy.admission_training_cv_descriptive.v1',
        snapshot_utc=datetime.now(timezone.utc).isoformat(),status='DESCRIPTIVE_TRAINING_CV_COMPLETE',
        counts=counts,rows=rows,full_route_source_counts=full,
        original_labels={key:capacity[key] for key in ('original_rho_admission_passed',
            'global_permit_capacity_qualified','physical_capacity_bound','final_gain_eligible')},
        manifest_labels={kind:{key:value.get(key) for key in
            ('independent_validation_sources_used','no_validation_data','final_gain_eligible')}
            for kind,value in manifests.items()},
        scope='Entire seed held out across all four routes; fixed known-input training diagnostic only',
        energy_scope=resources['scope'],
        error_denominators=dict(service='Preserved original diagnostic denominator: predicted worker mean',
            displayed_resource_ARE='abs(predicted-actual)/abs(actual); undefined at zero',quality='Original Q absolute error'),
        no_new_acceptance_threshold=True,independent_V4=False,prospective_queue_validation=False,
        service_refitted=completed['service_refitted'],raw_files_read=False,verify_fit_rollout_run=False,
        GPU_started=False,final_gain_eligible=False,source_receipts=list(receipts.values()),
        script=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    def pct(v):
        return 'UNKNOWN' if v is None else f'{100*v:.3f}%'
    def num(v):
        return 'UNKNOWN' if v is None else f'{v:.6f}'
    def label(v):
        return 'UNKNOWN' if v is None else str(v)
    lines = ['# Admission2 T8 整seed留出训练诊断', '',
        f"真实训练{counts['completed_training_jobs']}次；三份CV各{counts['cv_cells']}个单元，"
        f"heldout质量计数{counts['heldout_quality_requests']}，full质量计数{counts['full_quality_requests']}；"
        f"resource/quality各引用{counts['full_resource_unique_native_raw_receipts']}/{counts['full_quality_unique_native_raw_receipts']}份独立原生raw收据（本汇总不读取raw）。", '',
        '每个单元排除该seed全部四路线，仅使用另一seed的同route/controller画像。'
        '这是训练留出诊断；独立V4、前瞻全过程排队/P95和最终收益须另验。没有新增接受门槛、服务重拟合或V/F结果回流。', '',
        '|留出seed/路线|服务适用性原标签|最大family服务ARE|最大原rho|原rho通过|qualification原标签|global permit|',
        '|---|---|---:|---:|---|---|---|']
    for row in rows:
        c=row['capacity'];lines.append(f"|{row['seed']}/{row['candidate_id']}|{c['service_applicability_diagnostic']}|{pct(c['max_family_service_are'])}|{num(c['max_rho'])}|{c['rho_threshold_satisfied']}|{c['original_qualification_passed']}|{label(c['global_permit_capacity_qualified'])}|")
    lines += ['', 'family服务误差直接保留原诊断（分母为预测worker均值）；原rho门槛0.8、容差1e-8。'
        'CV中qualification_passed=False是原报告的“不授予资格”标签，不能解释为独立资格检验失败；'
        '总体original_rho_admission_passed与global_permit_capacity_qualified均保持UNKNOWN，physical bound=False。', '',
        '|留出seed/路线|真实context能量回放ARE|启动能量ARE|拥有显存峰误差MiB（预测−实际）|峰ARE|未知context数|',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        r=row['resources'];lines.append(f"|{row['seed']}/{row['candidate_id']}|{pct(r['actual_context_energy_replay']['absolute_relative_error'])}|{pct(r['startup_joules']['absolute_relative_error'])}|{num(r['owned_sampled_peak_mib']['predicted_minus_actual'])}|{pct(r['owned_sampled_peak_mib']['absolute_relative_error'])}|{len(r['unknown_contexts'])}|")
    lines += ['', '能量回放使用heldout真实上下文时长，覆盖measurement＋drain；这不是前瞻能量/队列预测，也不是最终效用E（最终E排除drain并按3600s摊销启动）。'
        '启动误差独立列示。显存是未加安全系数的startup-through-drain拥有进程离散采样峰，不能当连续真峰或容量上界。', '',
        '|留出seed/路线|预测Q|实际Q|Q绝对误差|heldout请求|', '|---|---:|---:|---:|---:|']
    for row in rows:
        q=row['quality'];lines.append(f"|{row['seed']}/{row['candidate_id']}|{num(q['predicted_Q'])}|{num(q['actual_Q'])}|{num(q['Q_absolute_error'])}|{q['heldout_request_count']}|")
    lines += ['', 'Q保留原任务error/reference汇总与offered-rate加权，未改成逐请求CER/WER平均；所有数值与直接来源SHA见summary.json。'
        '本摘要只核对直接小报告的SHA与CV域，不重做上游源闭包、质量打分或资格验证。', '']
    for ref in receipts.values():
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('Direct source changed while summarizing')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'report.zh.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'admission-training-cv-descriptive-v1')
    args=parser.parse_args();result=run(args.output)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],GPU_started=False)))
