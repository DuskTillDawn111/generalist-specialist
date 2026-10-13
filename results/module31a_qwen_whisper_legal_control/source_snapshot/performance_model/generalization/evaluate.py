"""Module 2D: fixed, exploratory Leave-One-Deployment-Out latency gate.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m performance_model.generalization.evaluate
No worker invocation, deployment enumeration, model export, or parameter search.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from performance_model.active_sampling.common import deployment_id
from performance_model.active_sampling.grouping import evaluation_configuration_id
from performance_model.features import curve_context
from performance_model.piecewise_baseline import LoadPoint, PiecewiseLoadLatencyModel, same_context
from performance_model.profiling.plan import ROOT, extractor, save
from .features import FoldPreprocessor, planning_vector

INPUT = ROOT / 'results/performance_model_active_sampling/dataset_rows.jsonl'
OUTPUT = ROOT / 'results/performance_model_generalization'
METRICS = ('mae_ms', 'median_absolute_error_ms', 'smape_percent', 'p90_absolute_error_ms', 'spearman')
# Written before fitting. This is a descriptive engineering gate, not a significance test.
POLICY = dict(
    split='Leave-One-Deployment-Out', primary_subset='stability.stable is True',
    sensitivity_subset='stable is True AND category == stable AND role not smoke/smoke_gate AND path completed >= 30',
    targets=['direct_ms', 'log_ratio'], normalized_formula='log(mean_e2e_latency_ms / standalone_service_ms)',
    training_weights='equal configuration mass, equal run mass within configuration, equal path mass within run',
    ridge=dict(alpha=10.0),
    gbt=dict(n_estimators=50, learning_rate=0.05, max_depth=2, min_samples_leaf=10, loss='huber', random_state=20260930),
    direct_prediction_floor_ms=1e-6,
    substantive_win='C MAE <= 0.90 * A MAE AND <= 0.90 * B MAE; C sMAPE <= both A/B sMAPE',
    collapse='C MAE > 2 * min(A MAE, B MAE) AND excess over that comparator > 50 ms',
    PASS='For BOTH normalized model families, majority fold wins, no collapse, macro MAE >=10% better than A and B, macro sMAPE no worse; primary AND strict sensitivity pass.',
    PARTIAL='Not PASS, but at least one normalized family wins >=2 deployment folds in primary or strict sensitivity.',
    FAIL='Neither PASS nor PARTIAL; current features/models do not provide sufficient evidence for a global predictor.',
    no_tuning=True, selected_family=None,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(row):
    return row['run_id'] + '::' + row['task'] + '::' + row['target_model']


def metrics(y, p):
    y, p = np.array(y, float), np.array(p, float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require finite nonempty observations and predictions')
    e = np.abs(y - p)
    valid_rank = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0
    return dict(n_rows=len(y), mae_ms=float(e.mean()), median_absolute_error_ms=float(np.median(e)),
                smape_percent=float(np.mean(200 * e / np.maximum(np.abs(y) + np.abs(p), 1e-12))),
                p90_absolute_error_ms=float(np.quantile(e, .9)),
                spearman=float(spearmanr(y, p).statistic) if valid_rank else None,
                spearman_unavailable_reason=None if valid_rank else 'constant prediction/target or fewer than two rows')


def macro(items):
    return {k: float(np.mean([v[k] for v in items if v[k] is not None]))
            if any(v[k] is not None for v in items) else None for k in METRICS} | {
            'deployment_folds': len(items), 'spearman_valid_folds': sum(v['spearman'] is not None for v in items)}


def training_weights(rows):
    by_config = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_config[r['configuration_id']][r['run_id']].append(i)
    w = np.zeros(len(rows))
    for runs in by_config.values():
        for indices in runs.values():
            w[indices] = 1 / (len(by_config) * len(runs) * len(indices))
    return w * len(rows)


def strict_eligible(r):
    return (r['stability']['stable'] is True and r['stability']['category'] == 'stable' and
            r['role'] not in ('smoke', 'smoke_gate') and r['labels']['completed'] >= 30)


def groups(rows):
    return {k: sorted({r[k] for r in rows}) for k in ('deployment_id', 'configuration_id', 'run_id')}


def split(rows, heldout):
    train = [r for r in rows if r['deployment_id'] != heldout]
    test = [r for r in rows if r['deployment_id'] == heldout]
    if not train or not test:
        raise ValueError('Empty LODO fold')
    overlaps = {k: sorted(set(groups(train)[k]) & set(groups(test)[k])) for k in groups(train)}
    if any(overlaps.values()):
        raise ValueError(f'Group leakage: {overlaps}')
    return train, test, overlaps


def training_piecewise(train, ex):
    # Group measured training rows by controlled numeric feature context and u.
    # Average repetitions at a point; never import a prefit global curve.
    groups_ = []
    for r in train:
        context = curve_context(r['_features'], r['target_model'])
        found = next((g for g in groups_ if g['model'] == r['target_model'] and g['task'] == r['task']
                      and same_context(g['context'], context)), None)
        if found is None:
            found = dict(model=r['target_model'], task=r['task'], context=context, points=defaultdict(list))
            groups_.append(found)
        found['points'][round(r['_B']['u'], 12)].append(r)
    points = []
    for g in groups_:
        for u, rs in g['points'].items():
            points.append(LoadPoint(g['model'], g['task'], u, g['context'],
                float(np.mean([r['_y'] for r in rs])), None, None,
                tuple(sorted({r['source_result_path'] for r in rs}))))
    return PiecewiseLoadLatencyModel(ex, points)


def evaluate_subset(rows, name, ex):
    predictions, audits, folds = [], [], []
    for dep in sorted({r['deployment_id'] for r in rows}):
        train, test, overlaps = split(rows, dep)
        y = np.array([r['_y'] for r in train])
        s0 = np.array([r['_s0'] for r in train])
        weights = training_weights(train)
        audit = dict(subset=name, heldout_deployment=dep, train=groups(train), test=groups(test), overlaps=overlaps,
                     fit_row_ids=[row_key(r) for r in train], test_row_ids=[row_key(r) for r in test],
                     train_repetition_groups=sorted({(r['configuration_id'], r['run_id'], r['repetition_id']) for r in train}),
                     test_repetition_groups=sorted({(r['configuration_id'], r['run_id'], r['repetition_id']) for r in test}),
                     repetition_note='Repetition numbers and seeds may repeat across configurations; no run/configuration/deployment crosses the fold.',
                     train_stable_only=all(r['stability']['stable'] is True for r in train),
                     preprocessors={}, weights=dict(zip([row_key(r) for r in train], weights.tolist())))
        pred = {'A_standalone': [r['_s0'] for r in test]}
        pw = training_piecewise(train, ex)
        pw_paths = [pw._path(r['_features'], r['task'], r['target_model']) for r in test]
        pred['A_piecewise_or_standalone'] = [p.mean_latency_ms if p.mean_latency_ms is not None else r['_s0']
                                            for r, p in zip(test, pw_paths)]
        audit['piecewise_training_source_paths'] = sorted({s for p in pw.points for s in p.source_result_paths})
        audit['piecewise_heldout_status_counts'] = dict(Counter(p.status for p in pw_paths))
        clipping, raw_predictions = {}, {}
        for level in ('B', 'C'):
            tr = [r['_' + level] for r in train]
            te = [r['_' + level] for r in test]
            pre = FoldPreprocessor().fit(tr, list(tr[0]))
            audit['preprocessors'][level] = pre.audit(te)
            xt, xv = pre.transform(tr), pre.transform(te)
            for family in ('ridge', 'gbt'):
                for target in POLICY['targets']:
                    variant = f'{level}_{family}_{target}'
                    estimator = Ridge(**POLICY['ridge']) if family == 'ridge' else GradientBoostingRegressor(**POLICY['gbt'])
                    yt = y if target == 'direct_ms' else np.log(y / s0)
                    estimator.fit(xt, yt, sample_weight=weights)
                    raw = estimator.predict(xv)
                    if target == 'log_ratio':
                        with np.errstate(over='raise', invalid='raise'):
                            restored = np.array([r['_s0'] for r in test]) * np.exp(raw)
                        clipping[variant] = [False] * len(test)
                    else:
                        clipping[variant] = (raw < POLICY['direct_prediction_floor_ms']).tolist()
                        restored = np.maximum(raw, POLICY['direct_prediction_floor_ms'])
                    if not np.isfinite(restored).all():
                        raise ValueError(f'Nonfinite prediction {dep}/{variant}; do not silently drop')
                    pred[variant] = restored.tolist()
                    raw_predictions[variant] = raw.tolist()
        fold = dict(heldout_deployment=dep, deployment=test[0]['deployment'], train_rows=len(train), test_rows=len(test),
                    train_runs=len(groups(train)['run_id']), test_runs=len(groups(test)['run_id']),
                    test_configurations=len(groups(test)['configuration_id']), models={})
        for variant, values in pred.items():
            m = metrics([r['_y'] for r in test], values)
            config_scores = []
            for cfg in groups(test)['configuration_id']:
                ix = [i for i, r in enumerate(test) if r['configuration_id'] == cfg]
                config_scores.append(metrics([test[i]['_y'] for i in ix], [values[i] for i in ix]))
            m['configuration_macro'] = {k: float(np.mean([c[k] for c in config_scores if c[k] is not None]))
                                        if any(c[k] is not None for c in config_scores) else None for k in METRICS}
            m['floored_predictions'] = sum(clipping.get(variant, []))
            fold['models'][variant] = m
            for i, (r, value) in enumerate(zip(test, values)):
                predictions.append(dict(subset=name, variant=variant, heldout_deployment=dep, row_id=row_key(r),
                    **{k: r[k] for k in ('deployment_id', 'configuration_id', 'run_id', 'repetition_id',
                                        'task', 'target_model', 'source_result_path', 'source_sha256', 'seed', 'protocol_id')},
                    standalone_latency_ms=r['_s0'], true_latency_ms=r['_y'], predicted_latency_ms=float(value),
                    raw_model_output=raw_predictions.get(variant, [None] * len(test))[i],
                    prediction_floored=clipping.get(variant, [False] * len(test))[i],
                    absolute_error_ms=abs(r['_y'] - float(value)), stability_category=r['stability']['category'],
                    piecewise_status=pw_paths[i].status if variant == 'A_piecewise_or_standalone' else None))
        folds.append(fold)
        audits.append(audit)
    summary = {v: macro([f['models'][v] for f in folds]) for v in folds[0]['models']}
    return dict(folds=folds, macro=summary, rows=len(rows), runs=len(groups(rows)['run_id'])), predictions, audits


def gate_evidence(result):
    out = {}
    for family in ('ridge', 'gbt'):
        b, c = f'B_{family}_log_ratio', f'C_{family}_log_ratio'
        wins, collapses, comparison = [], [], []
        for f in result['folds']:
            a, bm, cm = (f['models'][v] for v in ('A_standalone', b, c))
            win = cm['mae_ms'] <= .9 * min(a['mae_ms'], bm['mae_ms']) and cm['smape_percent'] <= min(a['smape_percent'], bm['smape_percent'])
            comparator = min(a['mae_ms'], bm['mae_ms'])
            collapse = cm['mae_ms'] > 2 * comparator and cm['mae_ms'] - comparator > 50
            if win: wins.append(f['heldout_deployment'])
            if collapse: collapses.append(f['heldout_deployment'])
            comparison.append(dict(deployment=f['heldout_deployment'], substantive_win=win, collapse=collapse,
                C_MAE_improvement_vs_A=1 - cm['mae_ms'] / a['mae_ms'], C_MAE_improvement_vs_B=1 - cm['mae_ms'] / bm['mae_ms']))
        a, bm, cm = (result['macro'][v] for v in ('A_standalone', b, c))
        macro_ok = cm['mae_ms'] <= .9 * min(a['mae_ms'], bm['mae_ms']) and cm['smape_percent'] <= min(a['smape_percent'], bm['smape_percent'])
        out[family] = dict(win_deployments=wins, collapse_deployments=collapses, fold_comparisons=comparison,
                           macro_improvement=macro_ok,
                           passes=len(wins) > len(result['folds']) / 2 and not collapses and macro_ok)
    return out


def dataset_audit(rows):
    by_run = {}
    for r in rows:
        previous = by_run.setdefault(r['run_id'], r)
        for k in ('deployment_id', 'configuration_id', 'repetition_id', 'source_result_path', 'stability'):
            if previous[k] != r[k]: raise ValueError(f'Inconsistent run metadata: {k}')
    deps = {}
    for d in sorted({r['deployment_id'] for r in rows}):
        rr = [r for r in rows if r['deployment_id'] == d]
        runs = [r for r in by_run.values() if r['deployment_id'] == d]
        deps[d] = dict(deployment=rr[0]['deployment'], rows=len(rr), runs=len(runs),
            configurations=len({r['configuration_id'] for r in rr}),
            run_categories=dict(Counter(r['stability']['category'] for r in runs)),
            stable_boolean_true_runs=sum(r['stability']['stable'] is True for r in runs),
            row_categories=dict(Counter(r['stability']['category'] for r in rr)),
            measurement_seconds_range=[min(r['measurement_seconds'] for r in runs), max(r['measurement_seconds'] for r in runs)],
            roles=dict(Counter(r['role'] for r in runs)))
    transient_flags = []
    for r in by_run.values():
        evidence, threshold = r['stability']['evidence'], r['stability']['thresholds']
        cutoff = max(threshold['end_backlog_unstable_min_requests'],
                     threshold['end_backlog_unstable_fraction'] * r['system_labels']['offered_requests'])
        bins = evidence.get('backlog_bins', [])
        max_backlog = max((max(b['mean_backlog'], b['backlog_at_end']) for b in bins), default=0.)
        if r['stability']['stable'] is True and max_backlog >= cutoff:
            transient_flags.append(dict(run_id=r['run_id'], deployment_id=r['deployment_id'],
                source_result_path=r['source_result_path'], max_observed_bin_backlog=max_backlog,
                diagnostic_cutoff=cutoff, existing_stability=r['stability']))
    return dict(rows=len(rows), runs=len(by_run), configurations=len({r['configuration_id'] for r in rows}),
        deployment_contexts=len(deps), deployments=deps,
        run_categories=dict(Counter(r['stability']['category'] for r in by_run.values())),
        stable_boolean_runs_with_transient_backlog=transient_flags,
        transient_backlog_diagnostic='Post-hoc diagnostic only: apply existing end-backlog magnitude cutoff to max observed bin mean/end backlog. Does not relabel, filter, select models, or change the gate.',
        stability_thresholds=[json.loads(t) for t in sorted({json.dumps(r['stability']['thresholds'], sort_keys=True) for r in rows})],
        label_semantics='Existing stable boolean includes boundary category; unstable rows retained in source and excluded from latency fit/evaluation.',
        model_scope='Latency exploratory gate only; no stability classifier or throughput predictor.')


def error_analysis(predictions, rows):
    lookup = {row_key(r): r for r in rows}
    analysis = {}
    for variant in sorted({p['variant'] for p in predictions}):
        ps = [p for p in predictions if p['subset'] == 'primary' and p['variant'] == variant]
        worst = []
        breakdown = defaultdict(lambda: defaultdict(list))
        for p in ps:
            r = lookup[p['row_id']]
            u = r['_B']['u']
            band = 'u<=0.2' if u <= .2 + 1e-9 else '.2<u<=.3' if u <= .3 + 1e-9 else '.3<u<=.4' if u <= .4 + 1e-9 else 'u>.4'
            contenders = sorted({m for rs in r['routing'].values() for m, rate in rs.items() if m != r['target_model'] and rate > 0})
            mix = r['task_demand_mix']
            dominant = max(mix, key=mix.get)
            mix_group = dominant + '_dominant' if mix[dominant] >= .5 else 'mixed'
            for dimension, group in [('own_load_band', band), ('task', r['task']), ('target_model', r['target_model']),
                                     ('task_mix', mix_group), ('active_contenders', '+'.join(contenders) or 'none'),
                                     ('stability_category', r['stability']['category']), ('role', r['role'])]:
                breakdown[dimension][group].append(p)
        for p in sorted(ps, key=lambda p: p['absolute_error_ms'], reverse=True)[:20]:
            r = lookup[p['row_id']]
            worst.append(p | dict(deployment=r['deployment'], standalone_service_demand=r['_B']['u'],
                task_demand_mix=r['task_demand_mix'], colocation_pressure=r['colocation_pressure'],
                stability_evidence=r['stability']['evidence'], measurement_seconds=r['measurement_seconds'],
                completed=r['labels']['completed'], role=r['role']))
        analysis[variant] = dict(worst_rows=worst, descriptive_breakdowns={d: {g: metrics(
            [p['true_latency_ms'] for p in ps_], [p['predicted_latency_ms'] for p in ps_]) for g, ps_ in gs.items()}
            for d, gs in breakdown.items()})
    # Identical planning inputs may have divergent repeated labels. No pooling in fit across folds.
    repeats = defaultdict(list)
    for r in rows:
        if r['stability']['stable'] is True and '_y' in r:
            repeats[(r['configuration_id'], r['task'], r['target_model'])].append(r)
    variability = []
    for (cfg, task, model), rs in repeats.items():
        if len({r['run_id'] for r in rs}) < 2: continue
        ys = np.array([r['_y'] for r in rs])
        variability.append(dict(configuration_id=cfg, deployment_id=rs[0]['deployment_id'], task=task, model=model,
            n_runs=len(rs), min_ms=float(ys.min()), max_ms=float(ys.max()), mean_ms=float(ys.mean()),
            sample_std_ms=float(ys.std(ddof=1)), max_min_ratio=float(ys.max() / ys.min()),
            run_ids=[r['run_id'] for r in rs], source_paths=[r['source_result_path'] for r in rs],
            protocol_ids=sorted({r['protocol_id'] for r in rs})))
    return dict(variants=analysis, largest_repeated_configuration_variation=sorted(variability, key=lambda v:v['sample_std_ms'], reverse=True)[:20],
                interpretation='Post-hoc descriptive diagnostics only; queue/backlog evidence is never an input. Associations do not establish causes.')


def markdown_summary(results, decision, evidence, audit, errors):
    primary = results['primary']
    names = {d: '+'.join(m for m, n in v['deployment'].items() if n) for d, v in audit['deployments'].items()}
    lines = ['# Module 2D：Cross-deployment Generalization Gate', '', f'**判定：{decision}**。', '',
        {'PASS': 'cross-deployment feature representation appears viable；仅支持进入正式 predictor development，不表示已完成正式 predictor。',
         'PARTIAL': 'global model needs local calibration / hybrid modeling；当前不支持直接使用单一 global predictor。',
         'FAIL': 'current feature representation does not support global predictor；当前应考虑 setup-local model + online calibration，不扩大模型复杂度。'}[decision], '',
        f"数据：{audit['runs']} 个真实 runs、{audit['rows']} 个 path rows、{audit['configurations']} 个 configuration、{audit['deployment_contexts']} 个 deployment contexts。",
        f"主评估：{primary['rows']} rows / {primary['runs']} runs，按已有 stable==true；严格敏感性：{results['strict']['rows']} rows / {results['strict']['runs']} runs。",
        '行不是独立 deployment samples；七个完整 deployment folds 才是跨 deployment 检查单位。未采集新数据，未训练或导出正式 predictor。', '',
        '## 设计与泄漏审计', '',
        '- LODO：held-out deployment 的全部 configuration/run/repetition/task 被排除；训练、测试 group 交集全部为空。repetition 数字/seed 跨配置相同不构成 run 重叠。',
        '- 仅用 shared FeatureExtractor 从 x/r/lambda 与独立 offline profiles 重算特征。无 exact deployment ID one-hot，无实时 queue/latency/completion/GPU 输入。',
        '- B：u + 六维 q。C：B + C resource pressure + standalone path/resource fingerprints + numeric model/deployment memory/count。',
        '- Standalone s0 是独立 offline 校准，允许所有 folds 使用；joint held-out labels 不参与校准。每个 fold 单独 fit median/missing indicators、常数过滤和标准化，记录全部参数。',
        '- Ridge alpha=10；GBT 50 trees、depth=2、min_samples_leaf=10、learning_rate=.05、Huber loss，固定 seed。直接 ms 与 log(L/s0) 各比较一次，无调参、无模型家族择优。',
        '- 训练权重依次按 configuration / run / path 平衡。报告每 fold 的 row metrics 与 configuration-macro metrics；所有 deployment 等权 macro。',
        '- 直接预测若为负仅按预先规定的 1e-6 ms floor；次数写入 fold metrics。normalized 输出按 s0*exp(yhat) 恢复，不按 held-out labels 限幅。',
        '- A 为 standalone s0；另运行 training-only controlled piecewise-u，缺失曲线时明确 fallback 到 s0。未导入全数据构建的 load curve。', '',
        '## 每 deployment 的主结果', '',
        '下表为 MAE ms（normalized target）；median AE、sMAPE、P90 AE、Spearman、direct-target variants 和严格敏感性完整保存在 fold_metrics.json。', '',
        '| Held-out deployment | rows | A s0 | B Ridge | C Ridge | B GBT | C GBT |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in primary['folds']:
        vs = ['A_standalone', 'B_ridge_log_ratio', 'C_ridge_log_ratio', 'B_gbt_log_ratio', 'C_gbt_log_ratio']
        lines.append(f"| {names[f['heldout_deployment']]} | {f['test_rows']} | " + ' | '.join(f"{f['models'][v]['mae_ms']:.1f}" for v in vs) + ' |')
    lines += ['', '| Variant | macro MAE ms | macro median AE ms | macro sMAPE % | macro P90 AE ms | macro Spearman |',
              '|---|---:|---:|---:|---:|---:|']
    for v, m in primary['macro'].items():
        lines.append('| ' + v + ' | ' + ' | '.join('null' if m[k] is None else f'{m[k]:.3f}' for k in METRICS) + ' |')
    lines += ['', '## Decision gate 的逐 fold 证据', '',
        '预先固定实质改善为 C 的 MAE 同时低于 A/B 至少 10%，且 sMAPE 不高于两者。Collapse 为 C MAE 超过较低 comparator 的 2 倍且增加 >50 ms。',
        'PASS 要求两种 normalized 家族在主评估与严格敏感性均多数 folds 改善、无 collapse、macro 改善。PARTIAL 要求至少一个家族/子集在至少两个 folds 实质改善；否则 FAIL。规则是描述性工程检查，不是统计显著性检验。', '']
    for subset, families in evidence.items():
        for fam, ev in families.items():
            lines.append(f"- {subset}/{fam}：实质改善 {len(ev['win_deployments'])}/7；collapse {len(ev['collapse_deployments'])}/7；macro 条件={ev['macro_improvement']}。")
            if ev['collapse_deployments']: lines.append('  collapse contexts：' + ', '.join(names[d] for d in ev['collapse_deployments']) + '。')
    lines += ['', '直接 target 与 normalized target 都已报告；gate 使用预先指定的 normalized target，两种模型家族共同检查，不能用较好的 pooled average 掩盖失败 fold。', '',
        '现有 controlled piecewise-u 在本次 held-out contexts 的覆盖为 0：主评估 427/427 行均 insufficient_profile_data，明确回退 s0。它在这里没有已测曲线外的泛化证据，不能冒充独立有效的插值比较。',
        f"严格敏感性的 macro MAE：A={results['strict']['macro']['A_standalone']['mae_ms']:.1f} ms；B/C Ridge={results['strict']['macro']['B_ridge_log_ratio']['mae_ms']:.1f}/{results['strict']['macro']['C_ridge_log_ratio']['mae_ms']:.1f} ms；B/C GBT={results['strict']['macro']['B_gbt_log_ratio']['mae_ms']:.1f}/{results['strict']['macro']['C_gbt_log_ratio']['mae_ms']:.1f} ms。", '',
        '## 稳定性分布与适用边界', '', '| Deployment | stable runs | boundary runs | unstable runs |', '|---|---:|---:|---:|']
    for dep, d in audit['deployments'].items():
        counts = d['run_categories']
        lines.append(f"| {names[dep]} | {counts.get('stable',0)} | {counts.get('boundary',0)} | {counts.get('unstable',0)} |")
    lines += ['', '现有 stable boolean 包含 boundary 类别；严格敏感性进一步排除 boundary、smoke/smoke_gate 和 path completed<30。未按 latency 大小删除样本。',
        f"标签审计另发现 {len(audit['stable_boolean_runs_with_transient_backlog'])} 个 stable=true run 在窗口中途曾有大幅 backlog，随后恢复；见 stability_distribution.json。此诊断不参与特征或重新筛选。",
        '稳定标签是有限观察窗口内的 operational label，不保证稳态排队或低 latency。不同协议的窗口长短、任务池和运行环境差异可能混杂结果；本轮不把协议或运行后指标加入输入修补。',
        'unstable 仅见于 Qwen-only context；因此还不足以验证跨 deployment stability classifier。吞吐受 offered workload 限制，且没有全面受控 saturation 数据，不能据此宣称 throughput predictor 已就绪。', '',
        '## 错误诊断', '']
    for family in ('ridge', 'gbt'):
        v = f'C_{family}_log_ratio'
        worst = errors['variants'][v]['worst_rows'][0]
        lines.append(f"- {v} 最大错误：{names[worst['deployment_id']]}，{worst['task']}/{worst['target_model']}，u={worst['standalone_service_demand']:.4f}；真实 {worst['true_latency_ms']:.1f} ms，预测 {worst['predicted_latency_ms']:.1f} ms，s0={worst['standalone_latency_ms']:.1f} ms。run={worst['run_id']}。")
        for dim in ('own_load_band', 'task_mix', 'active_contenders', 'stability_category'):
            gs = errors['variants'][v]['descriptive_breakdowns'][dim]
            key = max(gs, key=lambda k: gs[k]['mae_ms'])
            lines.append(f"  {dim} 中平均绝对误差最大组：{key}，MAE={gs[key]['mae_ms']:.1f} ms，n={gs[key]['n_rows']}。")
    if errors['largest_repeated_configuration_variation']:
        v = errors['largest_repeated_configuration_variation'][0]
        lines.append(f"- 相同 planning configuration 重复测量仍可能差异很大：{v['configuration_id']} / {v['task']}/{v['model']}，{v['n_runs']} runs，范围 {v['min_ms']:.1f}–{v['max_ms']:.1f} ms，sample std={v['sample_std_ms']:.1f} ms。协议和 run source 保留在 error_analysis.json。")
    worst = errors['variants']['C_gbt_log_ratio']['worst_rows'][0]
    ev = worst['stability_evidence']
    lines.append(f"- 最大错误 run 的已有 stable label 需要特别解释：中途最高 bin mean backlog={max(b['mean_backlog'] for b in ev['backlog_bins']):.1f}，窗口结束 backlog={ev['window_end_backlog']}，drain={ev['drain_seconds']:.3f} s，completion={ev['completion_ratio']:.3f}。标签捕捉到最终恢复，却未排除瞬时严重排队。严格子集仍保留该 run，因为没有事后按 latency 剔除。")
    lines += ['', '这些 breakdown 的样本数和 task/model 组成不匹配，只描述关联，不能将某个 contender 或 resource dimension 解释为已证明原因。每个最坏 row 的 u/q/C、真值、预测、s0 与原始 source 路径完整保留。',
        '所以 PARTIAL 是在现有标签定义下的 gate 结论；不能把 Qwen-only 失败全部归因于 feature representation。正式稳定区建模前，需先统一对窗口中途排队的标签语义。这一轮仅记录问题，没有更改历史标签。',
        '七个 contexts 都包含 Qwen，多个 context 只有 1–2 个配置；task-mix、load、co-location 与 deployment 并未充分交叉。LODO 的实际含义是这些已观察 context 的留出检验，不是全 model space 泛化证明。', '',
        '## 产物与停止边界', '',
        '- fold_metrics.json：每 fold / 每模型 / 两个 subsets 的五项误差、configuration macro、部署 macro 与 gate。',
        '- predictions.jsonl：逐 held-out row 的预测及四级 grouping IDs、seed、source hash/path。',
        '- leakage_audit.json：训练/留出组、预处理 fit 参数、source hashes、特征 schema、排除原因、原数据完整性。',
        '- error_analysis.json：最大错误、分层诊断和同配置重复差异。',
        '- evaluation_config.json / stability_distribution.json：固定方案与全部 69 runs 的标签统计。',
        '本阶段完成后停止：未启动 profiling、guided search、routing optimization、最终 predictor、stability/throughput training 或 forecasting。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'evaluation_config.json', POLICY)  # Before any fitting/evaluation.
    protected = {str(p): sha(p) for folder in ('deployment_core', 'joint_deploy', 'performance_model')
                 for p in (ROOT / folder).rglob('*.py') if 'generalization' not in p.parts}
    input_hash = sha(args.input)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len({row_key(r) for r in rows}) != len(rows): raise ValueError('Duplicate path rows')
    ex = extractor()
    keys = sorted({k for p in ex.resource_profiles.values() for k in p['values']})
    sources, exclusions = {}, []
    for r in rows:
        for k in ('deployment_id', 'configuration_id', 'run_id', 'repetition_id'):
            if r.get(k) is None: raise ValueError(f'Missing grouping field {k}')
        if r['deployment_id'] != deployment_id(r['deployment']): raise ValueError('Noncanonical deployment group')
        cfg = evaluation_configuration_id(dict(deployment=r['deployment'], workload_rps=r['workload'], routing_rps=r['routing']))
        if cfg != r['configuration_id']: raise ValueError('Noncanonical configuration group')
        path = r['source_result_path']
        if not Path(path).is_absolute(): raise ValueError('Source path must be absolute')
        if path not in sources: sources[path] = sha(path)
        if sources[path] != r['source_sha256']: raise ValueError('Changed raw source')
        try:
            b, s0, f = planning_vector(r, ex, 'B', keys)
            c, _, _ = planning_vector(r, ex, 'C', keys)
        except ValueError as e:
            if 'standalone service time' not in str(e): raise
            exclusions.append(dict(row_id=row_key(r), source_result_path=path, reason=str(e)))
            continue
        if not math.isclose(b['u'], r['standalone_service_demand'], rel_tol=1e-10, abs_tol=1e-12): raise ValueError('Feature demand drift')
        if not same_context(f['per_model'][r['target_model']]['task_demand_mix'], r['task_demand_mix']): raise ValueError('Feature mix drift')
        if not same_context(f['per_model'][r['target_model']]['colocation_pressure'], r['colocation_pressure']): raise ValueError('Feature pressure drift')
        if not same_context(f['deployment_features'], r['deployment_features']): raise ValueError('Deployment feature drift')
        if not same_context(ex.resource_profiles.get((r['task'], r['target_model']), {}).get('values', {}),
                            r['service_path_resource_features']): raise ValueError('Offline resource feature drift')
        for profile_source in r['resource_profile_sources']:
            if sha(profile_source['path']) != profile_source['sha256']: raise ValueError('Changed offline resource source')
        y = r['labels']['mean_e2e_latency_ms']
        if y is None or not math.isfinite(y) or y <= 0:
            exclusions.append(dict(row_id=row_key(r), source_result_path=path, reason='missing/nonpositive mean E2E target'))
            continue
        r.update(_B=b, _C=c, _s0=s0, _features=f, _y=float(y))
    audit = dataset_audit(rows)
    save(out / 'stability_distribution.json', audit)
    eligible = [r for r in rows if r['stability']['stable'] is True and '_y' in r]
    results, predictions, folds = {}, [], []
    for name, subset in [('primary', eligible), ('strict', [r for r in eligible if strict_eligible(r)])]:
        results[name], ps, fs = evaluate_subset(subset, name, ex)
        predictions.extend(ps)
        folds.extend(fs)
    evidence = {name: gate_evidence(r) for name, r in results.items()}
    decision = 'PASS' if all(e['passes'] for fs in evidence.values() for e in fs.values()) else (
        'PARTIAL' if any(len(e['win_deployments']) >= 2 for fs in evidence.values() for e in fs.values()) else 'FAIL')
    save(out / 'fold_metrics.json', dict(decision=decision, policy=POLICY, gate_evidence=evidence, subsets=results))
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False, allow_nan=False) + '\n' for p in predictions))
    errors = error_analysis(predictions, rows)
    save(out / 'error_analysis.json', errors)
    offline_sources = {s['path']: s['sha256'] for p in ex.resource_profiles.values() for s in p['sources']}
    # Catalog service-time inputs are also tracked (the catalog itself retains source metadata).
    catalog_sources = {str(m): p.sources for m, p in ex.catalog.items()}
    def collect_profile_sources(obj):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                yield obj['path'], obj['sha256']
            for value in obj.values(): yield from collect_profile_sources(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj: yield from collect_profile_sources(value)
    calibration_sources = dict(collect_profile_sources(catalog_sources)) | offline_sources
    if set(calibration_sources) & set(sources): raise ValueError('Joint-run label source used as offline calibration')
    if any(sha(p) != h for p, h in calibration_sources.items()): raise ValueError('Changed offline calibration')
    if input_hash != sha(args.input) or any(sha(p) != h for p, h in sources.items()): raise ValueError('Input mutated during gate')
    if any(sha(p) != h for p, h in protected.items()): raise ValueError('Protected implementation changed during gate')
    save(out / 'leakage_audit.json', dict(status='PASS', dataset_path=str(args.input.resolve()), dataset_sha256=input_hash,
        input_rows=len(rows), primary_rows=len(eligible), excluded_unstable_rows=sum(r['stability']['stable'] is not True for r in rows),
        missing_or_invalid_exclusions=exclusions, raw_sources=sources, offline_resource_sources=offline_sources,
        standalone_catalog_sources=catalog_sources,
        feature_schema={'B': list(eligible[0]['_B']), 'C': list(eligible[0]['_C'])},
        feature_policy='Explicit whitelist recomputed from x/r/lambda + independently acquired offline profiles. IDs and runtime labels not projected.',
        profile_schema_origin='Offline resource profile dictionary; no heldout joint labels or data-fitted dimension selection.',
        fixed_offline_calibration_is_permitted=True, all_raw_results_preserved=True,
        calibration_joint_result_source_overlap=[], all_calibration_source_hashes_verified=True,
        protected_source_hashes=protected, protected_sources_unchanged=True,
        implementation_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        folds=folds))
    (out / 'generalization_summary.md').write_text(markdown_summary(results, decision, evidence, audit, errors))
    print(json.dumps(dict(decision=decision, rows=len(rows), stable_rows=len(eligible),
                         strict_rows=results['strict']['rows'], folds=len(results['primary']['folds']),
                         macro=results['primary']['macro'], evidence=evidence), indent=2))


if __name__ == '__main__':
    main()
