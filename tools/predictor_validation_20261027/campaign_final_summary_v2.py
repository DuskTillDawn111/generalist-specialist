"""Read existing campaign reports into a new, non-certifying CPU-only snapshot.

No experiment/forecast/fit/verifier modules are imported. Upstream reports are
projected with their original statuses; their complete source closures are not
reaudited here. Each run requires a new output directory, preserving snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/predictor-gpu-20261027'
SCHEMA = 'gsdeploy.campaign_final_summary.v2'
SOURCES = {
    'inventory': 'request-evidence-inventory-v1.json',
    'shared_report': 'flow-validation-h600-selected-v16/report/report.json',
    'shared_gate': 'flow-validation-h600-selected-v16/gate.json',
    'shared_capacity': 'flow-validation-h600-selected-v16/capacity-report.json',
    'shared_completed': 'flow-validation-h600-selected-v16/completed.json',
    'pilot_report': 'h600-admission-pilot-v1/report/report.json',
    'training_completed': 'admission-training-continuation-v1/completed.json',
    'capacity_manifest': 'admission-training-continuation-v1/capacity-fit/manifest.json',
    'resource_manifest': 'admission-training-continuation-v1/resource-fit/manifest.json',
    'quality_manifest': 'admission-training-continuation-v1/quality-fit/manifest.json',
    'validation_report': 'admission-validation-v1/report/report.json',
    'validation_gate': 'admission-validation-v1/report/gate.json',
    'validation_audit': 'admission-validation-v1/report/validation-audit.json',
    'validation_capacity': 'admission-validation-v1/report/capacity-qualification.json',
    'validation_completed': 'admission-validation-continuation-v2/completed.json',
    'final_report': 'four-group-continuation-v2/execution/report/report.json',
    'final_completed': 'four-group-continuation-v2/execution/completed.json',
    'shared_search': 'cpu-search/h600-selected-v16/summary.json',
    'admission_search': 'admission-same-domain-search-v1/summary.json',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Snapshot:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.loaded = {}
        self.refs = {}

    def read(self, name):
        return self.read_path(name, self.base / SOURCES[name])

    def read_path(self, name, path, expected_sha256=None):
        path = Path(path).resolve()
        if not path.is_relative_to(self.base):
            raise ValueError(f'Projected artifact must remain inside campaign root: {path}')
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.refs[name] = dict(path=str(path), status='PENDING', sha256=None)
            if expected_sha256 is not None:
                self.refs[name]['expected_sha256'] = expected_sha256
            self.loaded[name] = None
            return None
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f'Object report required: {path}')
        if expected_sha256 is not None and digest(data) != expected_sha256:
            raise ValueError(f'Direct referenced report SHA mismatch: {path}')
        self.refs[name] = dict(path=str(path), status='PRESENT', sha256=digest(data))
        self.loaded[name] = value
        return value

    def read_ref(self, name, ref):
        if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not isinstance(ref.get('sha256'), str):
            raise ValueError(f'Exact directly referenced receipt required: {name}')
        return self.read_path(name, ref['path'], ref['sha256'])

    def recheck(self):
        # Check only the small directly projected files. This is not a source-
        # closure audit or a substitute for upstream native verification.
        for ref in self.refs.values():
            if ref['status'] == 'PRESENT':
                if digest(Path(ref['path']).read_bytes()) != ref['sha256']:
                    raise ValueError(f"Input changed while summarizing: {ref['path']}")


def select(value, keys):
    return {key: value.get(key) for key in keys}


def gate_summary(gate):
    if gate is None:
        return dict(status='PENDING', failed_checks=None, unknown_checks=None)
    return select(gate, ('status', 'failed_checks', 'unknown_checks'))


def prediction_summary(report, gate, audit=None):
    if report is None:
        return dict(status='PENDING', gate=gate_summary(gate))
    result = select(report, ('status', 'candidate_coverage', 'paired_delta_coverage',
                            'delta_sign_accuracy', 'rank_accuracy', 'rank_coverage'))
    result['gate'] = gate_summary(gate)
    result['candidates'] = []
    for row in report.get('candidates', []):
        actual = row.get('observed_metrics', {})
        timing = row.get('timing_errors', {})
        result['candidates'].append(dict(
            **select(row, ('candidate_id', 'requested', 'known', 'actual_incomplete_requests')),
            service_mare=timing.get('service', {}).get('mean_absolute_relative_error'),
            e2e_mare=timing.get('e2e', {}).get('mean_absolute_relative_error'),
            queue_mae=row.get('queue_error', {}).get('mae'),
            actual_terms=actual.get('terms'), actual_U=actual.get('full_U'),
            actual_six_terms_complete=actual.get('six_term_comparison_complete')))
    # Paired prediction coverage and native actual/quality counts are distinct.
    result['native_evidence'] = None if audit is None else [dict(
        **select(row, ('candidate_id', 'requested', 'known')),
        quality_count=row.get('quality_audit', {}).get('count'),
        quality_independently_recomputed=row.get('quality_audit', {}).get('independently_recomputed'),
        actual_six_terms_complete=row.get('actual_metrics', {}).get('six_term_comparison_complete'),
        original_stability_passed=row.get('original_stability', {}).get('original_stability_passed'))
        for row in audit.get('evidence', [])]
    return result


def pilot_summary(report):
    if report is None:
        return dict(status='PENDING')
    result = select(report, ('status', 'requested', 'paired_known',
        'complete_three_mode_mechanism_conclusion', 'no_accuracy_gate_predeclared', 'unsupported_predictions'))
    result['candidates'] = [dict(
        **select(row, ('candidate_id', 'requested', 'paired_known', 'complete', 'activation')),
        service_mare=row.get('timing_errors', {}).get('service_ms', {}).get('mean_absolute_relative_error'),
        e2e_mare=row.get('timing_errors', {}).get('e2e_ms', {}).get('mean_absolute_relative_error'),
        queue_mae=row.get('queue_error', {}).get('mae'),
        per_task_p95=row.get('per_task_p95'),
        original_stability_passed=row.get('original_stability', {}).get('original_stability_passed'))
        for row in report.get('candidates', [])]
    return result


def capacity_summary(report):
    if report is None:
        return dict(status='PENDING', rows=None)
    return dict(status='PRESENT_ORIGINAL_LABELS_RETAINED', rows=[dict(
        **select(row, ('candidate_id', 'statistical_profile_applicable', 'statistical_service_applicability',
            'original_rho_threshold_satisfied', 'statistical_original_rho_planning_admission_passed',
            'original_measured_profile_resolver_admission', 'global_permit_capacity_qualified')),
        original_stability_passed=row.get('original_stability', {}).get('original_stability_passed'))
        for row in report.get('rows', report.get('candidates', []))])


def search_summary(report):
    if report is None:
        return dict(status='PENDING')
    return select(report, ('status', 'candidate_count', 'request_count', 'same_search_output',
        'same_optimum_and_all_ties', 'same_model_optimum_and_all_ties', 'maximum_model_comparison_U',
        'all_tie_ids', 'model', 'original_qualified', 'pruning_count', 'forecast_calls_saved',
        'original_gate_constrained_optimality_certified', 'physical_optimality_certified',
        'global_permit_capacity_proof', 'search_acceleration_claim', 'domain', 'unsearched'))


def training_manifests(snapshot, data):
    """Project actual manifest/CV/model receipts, never reconstruct a fit."""
    result = {}
    completed_refs = (data.get('training_completed') or {}).get('manifests', {})
    for kind in ('capacity', 'resource', 'quality'):
        key = kind+'_manifest'
        manifest = data[key]
        expected = completed_refs.get(kind+'_protocol')
        if expected is not None and manifest is not None:
            actual_ref = snapshot.refs[key]
            if (str(Path(expected['path']).resolve()), expected['sha256']) != (actual_ref['path'], actual_ref['sha256']):
                raise ValueError(f'T8 completion manifest receipt differs: {kind}')
        if manifest is None:
            result[kind] = dict(status='PENDING', cv=None, full_source_counts=None)
            continue
        row = dict(status='MANIFEST_PRESENT', schema=manifest.get('schema'),
            declared_source_receipt_count=len(manifest.get('source_receipts', [])),
            independent_validation_sources_used=manifest.get('independent_validation_sources_used'),
            no_validation_data=manifest.get('no_validation_data'))
        cv_ref_key = 'fit' if kind == 'capacity' else kind+'_cv'
        cv = snapshot.read_ref(kind+'_training_cv', manifest[cv_ref_key])
        cv_rows = [] if cv is None else cv.get('whole_seed_cv' if kind == 'capacity' else 'rows', [])
        row['cv'] = dict(status='PENDING' if cv is None else 'CV_REPORT_PRESENT',
            rows=None if cv is None else [select(value, (
                'seed', 'candidate_id', 'training_seeds', 'service_applicability_diagnostic',
                'service_checks', 'qualification_passed', 'unknown_contexts',
                'predicted_startup_joules', 'actual_startup_joules',
                'predicted_owned_sampled_peak_mib', 'actual_owned_sampled_peak_mib',
                'actual_context_replay_predicted_measurement_and_drain_joules',
                'actual_measurement_and_drain_joules', 'Q_absolute_error', 'predicted', 'observed')) for value in cv_rows],
            row_count=None if cv is None else len(cv_rows),
            independent_validation_sources_used=None if cv is None else cv.get('independent_validation_sources_used'),
            scope=None if cv is None else cv.get('scope', cv.get('policy', {}).get('cv')))
        if kind == 'capacity':
            full = [] if cv is None else cv.get('full_training_references', [])
            row['full_source_counts'] = None if cv is None else dict(
                source_rows=len(full), unique_native_raw_receipts=len({
                    (value['heldout_raw']['path'], value['heldout_raw']['sha256']) for value in full}),
                seeds=sorted({value['seed'] for value in full}),
                routes=sorted({value['candidate_id'] for value in full}))
        else:
            full = manifest.get('full', {})
            models = []
            unique_raw = set()
            for cid, ref in full.get('route_resources' if kind == 'resource' else 'route_quality', {}).items():
                model = snapshot.read_ref(kind+'_full_'+cid, ref)
                if model is None:
                    models.append(dict(candidate_id=cid, status='PENDING'))
                    continue
                evidence = model.get('admission_training_evidence', {}) if kind == 'resource' else model
                runs = evidence.get('runs', [])
                for run in runs:
                    if isinstance(run.get('raw'), dict):
                        unique_raw.add((run['raw']['path'], run['raw']['sha256']))
                models.append(dict(candidate_id=cid, status='FULL_ROUTE_MODEL_PRESENT',
                    training_seeds=evidence.get('training_seeds'), source_run_count=len(runs),
                    request_count=model.get('request_count') if kind == 'quality' else None,
                    independently_recomputed_quality_counts=[run.get('quality_audit', {}).get('count') for run in runs]
                        if kind == 'quality' else None))
            row['full_source_counts'] = dict(route_models=models,
                unique_native_raw_receipts=len(unique_raw), training_seeds=full.get('training_seeds'),
                folds=[select(fold, ('omitted_seed', 'training_seeds')) for fold in manifest.get('folds', [])],
                total_quality_requests=sum(value['request_count'] for value in models)
                    if kind == 'quality' and models and all(value.get('request_count') is not None for value in models) else None)
        result[kind] = row
    return result


def final_cells(snapshot, report):
    """Fstats emits utility maps, not original rows; read actual-audit receipts."""
    audits = {}
    for index, ref in enumerate((report or {}).get('actual_audits', [])):
        value = snapshot.read_ref(f'final_actual_audit_{index:02d}', ref)
        if value is not None:
            cid = value['candidate_id']
            if cid in audits:
                raise ValueError(f'Duplicate final cell actual audit: {cid}')
            audits[cid] = value
    modes = (report or {}).get('descriptive_comparison', {}).get('modes', {})
    result = []
    for mode in ('shared', 'admission2'):
        comparison = modes.get(mode, {})
        for route in ('balanced', 'skewed', 'spill_low', 'spill_high'):
            cid = mode+'_'+route
            audit = audits.get(cid)
            score = {} if audit is None else audit.get('six_term_score', {})
            stability = {} if audit is None else audit.get('original_stability', {})
            result.append(dict(mode=mode, route=route, candidate_id=cid,
                status='PENDING' if audit is None else 'ACTUAL_AUDIT_PRESENT',
                requested=None if audit is None else audit.get('requests'),
                known=None if audit is None else audit.get('known'),
                U=comparison.get('utility_by_route', {}).get(route),
                diagnostic_U=score.get('diagnostic_U'), terms=score.get('terms'),
                six_terms_complete=score.get('utility_complete'),
                quality_count=None if audit is None else audit.get('quality', {}).get('count'),
                original_qualification=comparison.get('original_qualification', {}).get(route, 'UNKNOWN'),
                original_stability_passed=stability.get('original_stability_passed'),
                original_stability_state=stability.get('stability', {}).get('stability_state_v2')))
    return result


def build(snapshot):
    data = {name: snapshot.read(name) for name in SOURCES}
    inventory = data['inventory'] or {}
    training = data['training_completed']
    final = data['final_report']
    final_done = data['final_completed']
    manifests = training_manifests(snapshot, data)
    cells = final_cells(snapshot, final)
    stages = dict(
        microprofiles=dict(status='COMPLETE_WITH_LIMITATIONS' if inventory else 'PENDING',
            rows=[select(row, ('name', 'batch_count', 'request_count', 'completed', 'context_counts',
                'samples', 'owned_sampled_peak_mib', 'max_sample_gap_ms'))
                for row in inventory.get('microprofiles', [])],
            physical_complementarity='UNKNOWN', counter_status=inventory.get('hardware', {})),
        shared_validation=prediction_summary(data['shared_report'], data['shared_gate']),
        shared_capacity=capacity_summary(data['shared_capacity']),
        admission_pilot=pilot_summary(data['pilot_report']),
        admission_training=dict(status='PENDING' if training is None else 'COMPLETED_RECORD_PRESENT',
            **({} if training is None else select(training, ('job_count', 'manifests', 'service_refitted'))),
            fitted_manifests=manifests),
        admission_validation=prediction_summary(data['validation_report'], data['validation_gate'], data['validation_audit']),
        admission_capacity=capacity_summary(data['validation_capacity']),
        final_four_groups=dict(status='PENDING', cells=cells) if final is None else dict(
            **select(final, ('status', 'known', 'requested', 'primary_comparison', 'primary_value',
                'independent_blocks', 'policy_scope', 'strong_baseline_scope', 'invalid_attempts',
                'original_capacity_qualification', 'final_gain_eligible', 'statistical_significance_claim')),
            descriptive_comparison=final.get('descriptive_comparison'),
            cells=cells,
            execution_completion=None if final_done is None else select(final_done,
                ('status', 'known', 'requested', 'execution_error', 'independent_gate_status', 'final_seal_authenticated'))),
        shared_same_domain_DFS=search_summary(data['shared_search']),
        admission_same_domain_DFS=search_summary(data['admission_search']))
    requirements = [
        dict(id='R1', request='固定四驻留、活动组合/输入/错位及阶段功率拥有显存画像',
            status=stages['microprofiles']['status'], evidence=['inventory'],
            limitation='离散owned显存采样和板级功率；完整kernel/SM/DRAM/L2物理归因未建立'),
        dict(id='R2_shared', request='shared独立服务/全过程排队/P95/ΔU符号及候选排序验证',
            status=stages['shared_validation']['gate']['status'], evidence=['shared_report', 'shared_gate', 'shared_capacity'],
            limitation='排序正确不能覆盖误差门槛失败；单块已知输入域'),
        dict(id='R3_pilot', request='默认共享与整请求max1/max2真实准入机制比较',
            status=stages['admission_pilot']['status'], evidence=['pilot_report'],
            limitation='只检验迁移与激活；没有预声明准确度gate，六项净收益未知'),
        dict(id='R3_T8', request='准入mode/route画像、Q/资源/容量及整seed CV',
            status=stages['admission_training']['status'], evidence=['training_completed'],
            limitation='服务Gamma保持冻结；没有用独立V/F拟合'),
        dict(id='R2_V4', request='准入独立预测验证与原容量/稳定性分开报告',
            status=stages['admission_validation']['gate']['status'],
            evidence=['validation_report', 'validation_gate', 'validation_audit', 'validation_capacity'],
            limitation='实际完成、质量计数与预测配对覆盖分开；原resolver/global permit资格不由完成数推得'),
        dict(id='R4', request='同到达输入的两执行方式×禁/允许分流及强禁分流基线',
            status=stages['final_four_groups']['status'], evidence=['final_report', 'final_completed'],
            limitation='R1描述块；主值为admission2锁定high减本块balanced/skewed中较优者，非全部部署最优'),
        dict(id='DFS_shared', request='shared四固定候选同域fresh DFS',
            status=stages['shared_same_domain_DFS']['status'], evidence=['shared_search'],
            limitation='模型算术核验；不认证原约束或物理最优，无剪枝不声称提速'),
        dict(id='DFS_admission', request='admission2四固定候选同域fresh DFS',
            status=stages['admission_same_domain_DFS']['status'], evidence=['admission_search'],
            limitation='未知资格保留；无有效候选集合不声称存在最优解'),
    ]
    mechanisms = [
        dict(request='计算型/显存型互补放置的物理验证', status='UNKNOWN', reason='SM/DRAM/L2计数器权限拒绝，未建立互补放置干预对照'),
        dict(request='Roomie启发的阶段/上下文经验建模', status='IMPLEMENTED_EMPIRICAL', reason='真实prefill/decode与全过程事件模拟；不能等同已验证occupancy/资源公式'),
        dict(request='硬件资源份额/MPS/互斥SM', status='UNKNOWN', reason='native适配器无生效收据；配置百分比不证明SM隔离'),
        dict(request='跨请求共同batch/分batch执行', status='NOT_IMPLEMENTED', reason='执行仍batch1，无联合执行器及配批等待/额外KV/质量对照'),
        dict(request='同模型公共权重与异构最大公共子矩阵/cache复用', status='NOT_IMPLEMENTED', reason='独立owner，无精确公共块/算子合并/cache-hit证据；预测不添加复用折扣'),
    ]
    pending = [row['id'] for row in requirements if row['status'] == 'PENDING']
    snapshot.recheck()
    return dict(schema=SCHEMA, snapshot_utc=datetime.now(timezone.utc).isoformat(),
        status='PENDING_CAMPAIGN' if pending else 'REPORTS_AVAILABLE_NOT_DEPLOYMENT_CERTIFICATION',
        pending_requirements=pending, requirements=requirements, mechanisms=mechanisms, stages=stages,
        objective=dict(formula='U=-(Q+T+L+P+M+E)/6', terms='QTLPME',
            Q='原error/reference totals，offered-rate加权', T='测量截止未完成比例',
            L_P='计划到达至parent receipt；原归一化与线性经验P95',
            M='startup-through-drain拥有进程采样峰/49140 ×1.110854，仅一次',
            E='测量期整板能量＋启动按3600s摊销，不含drain',
            original_stability='原g(n)、rho .8+1e-8及DEFAULT_THRESHOLDS均保留'),
        limitations=['只投影已存在报告；本汇总没有重新执行上游native/source-closure验证',
            '新seed使用有限重叠输入池，不是未见输入总体泛化',
            '请求不是独立运行重复；R1没有运行级置信区间或显著性',
            'FAIL/UNKNOWN/未实现机制不因描述ΔU为正而改变；本轮不认证部署正收益'],
        source_receipts=snapshot.refs, summarizer=dict(path=str(Path(__file__).resolve()),
            sha256=digest(Path(__file__).read_bytes())), GPU_started=False, model_fitted=False,
        forecasts_run=False, upstream_closure_reverified=False, final_gain_eligible=False,
        physical_complementarity_certified=False, deployment_optimum_certified=False)


def fmt(value, precision=4):
    if value is None:
        return 'UNKNOWN'
    return f'{value:.{precision}f}' if isinstance(value, float) else str(value)


def markdown(summary):
    stages = summary['stages']
    shared = stages['shared_validation']
    gate = shared['gate']
    lines = ['# GPU画像、预测与分流：有限轮证据汇总', '',
        f"快照：{summary['snapshot_utc']}。状态：**{summary['status']}**。",
        '', f"shared原独立预测门槛 **{gate['status']}**；失败{len(gate['failed_checks'] or [])}项。"
        f"ΔU符号准确率{fmt(shared.get('delta_sign_accuracy'))}，候选排序准确率{fmt(shared.get('rank_accuracy'))}。"
        '排序与均值拟合不能替代全过程排队和任务P95验收。', '',
        '本报告是现有产物的只读投影；直接输入有SHA收据，上游全闭包与原始请求未在此重新核验。'
        '后续产物不会自动改变本快照。', '',
        '|要求|状态|范围与限制|', '|---|---|---|']
    lines += [f"|{row['request']}|{row['status']}|{row['limitation']}|" for row in summary['requirements']]
    lines += ['', '## Shared独立预测', '',
        '|路线|配对已知/请求|服务MARE|E2E MARE|queue MAE|实际U|', '|---|---:|---:|---:|---:|---:|']
    for row in shared.get('candidates', []):
        lines.append(f"|{row['candidate_id']}|{row['known']}/{row['requested']}|{fmt(row['service_mare'])}|{fmt(row['e2e_mare'])}|{fmt(row['queue_mae'])}|{fmt(row['actual_U'],6)}|")
    lines += ['', '失败项：' + ('、'.join(gate['failed_checks']) if gate['failed_checks'] else '无或PENDING，以原gate状态为准') + '。',
        '', '## 准入机制与独立验证', '',
        f"三模式pilot：{stages['admission_pilot']['status']}；只检验服务迁移与真实激活，"
        '没有事先定义独立准确度gate，六项资源/容量/ΔU仍UNKNOWN。已观测准入减执行干扰但增等待与文本尾部。', '',
        '|pilot路线|service MARE|E2E MARE|queue MAE|文本实际P95/ms|原稳定性通过|',
        '|---|---:|---:|---:|---:|---|']
    for row in stages['admission_pilot'].get('candidates', []):
        p95 = (row.get('per_task_p95') or {}).get('text', {}).get('observed_ms')
        lines.append(f"|{row['candidate_id']}|{fmt(row['service_mare'])}|{fmt(row['e2e_mare'])}|{fmt(row['queue_mae'])}|{fmt(p95)}|{row['original_stability_passed']}|")
    admission = stages['admission_validation']
    lines += ['', f"T8：{stages['admission_training']['status']}；V4原gate：**{admission['gate']['status']}**。",
        '实际请求/质量/六项完成与预测配对覆盖分开记录；原rho、统计适用性、稳定性及旧resolver/global permit资格分开保留。',
        '', '|T8画像|manifest状态|CV行数|完整画像独立raw来源数|完整Q请求数|', '|---|---|---:|---:|---:|']
    for kind, value in stages['admission_training']['fitted_manifests'].items():
        cv = value.get('cv') or {}
        full = value.get('full_source_counts') or {}
        lines.append(f"|{kind}|{value['status']}|{fmt(cv.get('row_count'))}|{fmt(full.get('unique_native_raw_receipts'))}|{fmt(full.get('total_quality_requests')) if kind=='quality' else '不适用'}|")
    lines += ['', 'CV按完整seed留出；resource CV是使用真实heldout上下文的回放，不是前瞻队列/能量验证。'
        '计数只来自实际manifest、CV和完整route模型；不存在产物时不填计划数。逐fold与误差见机器审计。', '',
        '|V4路线|预测配对/请求|原生/Qcount|六项complete|服务MARE|E2E MARE|queue MAE|实际U|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    predicted = {row['candidate_id']: row for row in admission.get('candidates', [])}
    native = {row['candidate_id']: row for row in admission.get('native_evidence') or []}
    for route in ('balanced', 'skewed', 'spill_low', 'spill_high'):
        cid = 'admission2_'+route
        row, actual = predicted.get(cid, {}), native.get(cid, {})
        lines.append(f"|{cid}|{fmt(row.get('known'))}/{fmt(row.get('requested'))}|{fmt(actual.get('known'))}/{fmt(actual.get('quality_count'))}|{fmt(actual.get('actual_six_terms_complete'))}|{fmt(row.get('service_mare'))}|{fmt(row.get('e2e_mare'))}|{fmt(row.get('queue_mae'))}|{fmt(row.get('actual_U'),6)}|")
    agate = admission['gate']
    lines += ['', f"V4 ΔU符号准确率：{fmt(admission.get('delta_sign_accuracy'))}；候选排序准确率：{fmt(admission.get('rank_accuracy'))}。",
        'V4失败项：'+('、'.join(agate['failed_checks']) if agate['failed_checks'] else '无或PENDING，以原gate为准')+'。',
        'V4未知项：'+('、'.join(agate['unknown_checks']) if agate['unknown_checks'] else '无或PENDING，以原gate为准')+'。',
        '', '## 最终两执行方式×禁/允许分流', '']
    final = stages['final_four_groups']
    lines += [f"F8：**{final['status']}**。主比较值：{fmt(final.get('primary_value'),6)}。",
        '主比较是admission2预锁定spill_high减本块balanced/skewed两条声明禁分流路线中实测较优者；'
        '该强基线仍只覆盖声明有限域。R1只给描述差值，不给运行级置信区间、显著性、部署正收益认证。']
    comparison = final.get('descriptive_comparison') or {}
    if comparison:
        lines += ['', '|模式|锁定allow减锁定禁分流|锁定allow减强禁分流|原资格|', '|---|---:|---:|---|']
        for mode, value in comparison.get('modes', {}).items():
            lines.append(f"|{mode}|{fmt(value.get('locked_allow_minus_locked_no_spill'),6)}|{fmt(value.get('locked_allow_minus_strong_no_spill'),6)}|{json.dumps(value.get('original_qualification'),ensure_ascii=False)}|")
    lines += ['', '|F8模式/路线|状态|已知/请求|U|Q|T|L|P|M|E|原资格|稳定性通过/状态|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for cell in final['cells']:
        terms = cell.get('terms') or {}
        costs = '|'.join(fmt(terms.get(key),6) for key in 'QTLPME')
        lines.append(f"|{cell['mode']}/{cell['route']}|{cell['status']}|{fmt(cell['known'])}/{fmt(cell['requested'])}|{fmt(cell['U'],6)}|{costs}|{cell['original_qualification']}|{fmt(cell['original_stability_passed'])}/{fmt(cell['original_stability_state'])}|")
    lines += ['', '八格效用投影自原descriptive_comparison；六项及稳定性来自逐格actual_audits直接收据。'
        '没有逐格真实审计时保留PENDING，不从效用map反推六项或稳定性。']
    lines += ['', '## 创新机制实现边界', '', '|机制|状态|证据边界|', '|---|---|---|']
    lines += [f"|{row['request']}|{row['status']}|{row['reason']}|" for row in summary['mechanisms']]
    lines += ['', '## 原指标与复现', '',
        '`U=-(Q+T+L+P+M+E)/6`。Q保留原error/reference及offered-rate加权；T按测量截止；'
        'L/P由计划到达至parent receipt，沿用归一化与线性经验P95；'
        'M为startup-through-drain拥有进程离散峰/49140×1.110854一次；'
        'E为测量整板能量加启动按3600s摊销、排除drain。原g(n)、rho和稳定性阈值保留。', '',
        '机器报告 `summary.json` 包含逐要求审计、原门槛/容量标签、预测与实际质量完成计数、F8完整描述对照、'
        'DFS状态和输入SHA。`source-schema.md`列出待产出字段。再次运行必须使用新输出目录；不覆盖既有快照。', '']
    return '\n'.join(lines)


def schema_markdown():
    return '''# 上游字段与完成后补充方式

生成器只读现有JSON。原冻结报告不补写字段、不编辑；完成后另选输出目录重新运行。

|来源|投影字段|缺失处理|
|---|---|---|
|T8 completed.json|job_count、manifests、service_refitted|PENDING|
|T8 capacity-fit/manifest.json及fit收据|whole_seed_cv、full_training_references的已有行/seed/route/raw来源计数|PENDING；不从计划补训练数|
|T8 resource/quality manifest及CV/full route收据|已有CV rows、full训练seed/独立raw/run/request_count|PENDING；直接SHA核对，不重拟合/重验闭包|
|V4 report/report.json|candidate_coverage、candidates[].known/requested、timing_errors.service/e2e、queue_error、observed_metrics.terms/full_U、ΔU符号/排序与覆盖|PENDING；缺少配对不等同实际未完成|
|V4 report/gate.json|status、failed_checks、unknown_checks|原标签原样保留|
|V4 report/validation-audit.json|evidence[].known/requested、quality_audit.count/independently_recomputed、actual_metrics.six_term_comparison_complete、original_stability|PENDING；不从配对覆盖替代质量|
|V4 capacity-qualification.json|rows[].statistical_service_applicability、original_rho_threshold_satisfied、original_stability、statistical_original_rho_planning_admission_passed、global_permit_capacity_qualified、original_measured_profile_resolver_admission|UNKNOWN不升级|
|F8 execution/report/report.json|status、known/requested、primary_value、descriptive_comparison、invalid_attempts、strong_baseline_scope、original_capacity_qualification|PENDING；不自行重算/挑选主策略|
|F8 report.actual_audits[]逐格直接收据|six_term_score.terms/diagnostic_U/utility_complete、quality.count、known/requests、original_stability|Fstats不输出原rows；无真实审计不能反推六项，保持PENDING|
|F8 execution/completed.json|status、execution_error、independent_gate_status、final_seal_authenticated|缺失时completion=null，不凭report推断封存认证|
|admission-same-domain-search-v1/summary.json|status、same_search_output、model、original_qualified、pruning_count、forecast_calls_saved及认证字段|PENDING；UNKNOWN/EMPTY_DOMAIN不变成最优证明|

shared既有报告和早期画像盘点保留原SHA快照。硬件计数器/互补分类UNKNOWN，以及batch/公共权重/cache执行器NOT_IMPLEMENTED，不能靠T/V/F请求完成自动升级。

此汇总器不执行昂贵verify_report、拟合、预测模拟或GPU采集。直接文件SHA仅证明本次所投影的字节；源闭包认证仍由既有上游严格流程负责。
'''


def run(base, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Summary snapshots are immutable; choose a new output directory')
    snapshot = Snapshot(base)
    summary = build(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (output / 'results.zh.md').write_text(markdown(summary))
    (output / 'source-schema.md').write_text(schema_markdown())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=BASE/'campaign-final-summary-v2-preview')
    args = parser.parse_args()
    summary = run(args.base, args.output)
    print(json.dumps(dict(status=summary['status'], pending=summary['pending_requirements'],
        output=str(args.output.resolve()), GPU_started=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
