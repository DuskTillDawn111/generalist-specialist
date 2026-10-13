"""Read-only simulation: v2 labels, outer LODO, inner configuration calibration."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from performance_model.profiling.plan import ROOT, extractor, save
from performance_model.generalization.evaluate import sha, row_key, metrics, METRICS, training_weights
from performance_model.generalization.features import planning_vector, FoldPreprocessor
from performance_model.piecewise_baseline import same_context
from performance_model.active_sampling.common import deployment_id
from performance_model.active_sampling.grouping import evaluation_configuration_id
from .stability_v2 import DEFAULT_THRESHOLDS, scaled_thresholds, reconstruct, classify
from .calibration import SELECTION_POLICY, planning_descriptor, selection_plan, bias_from_probes

INPUT = ROOT / 'results/performance_model_active_sampling/dataset_rows.jsonl'
OUT = ROOT / 'results/performance_model_local_calibration'
GBT = dict(n_estimators=50, learning_rate=.05, max_depth=2, min_samples_leaf=10, loss='huber', random_state=20260930)
SCENARIOS = {'warning_075': .75, 'nominal': 1., 'warning_125': 1.25}
POLICY = dict(
    global_model=GBT, target='log(mean_e2e_latency_ms / standalone_service_ms)',
    global_training='Only v2 steady rows from other deployments. The SAME global fit evaluates strict and extended test subsets.',
    calibration_model='Bias-only: median of all accepted calibration path-row log residuals; no affine model or model selection.',
    selection=SELECTION_POLICY, thresholds=DEFAULT_THRESHOLDS, sensitivity_warning_factors=SCENARIOS,
    sensitivity_scope='Relabel AND rerun fixed global/calibration procedure at each factor; hard instability thresholds unchanged. No retuning.',
    metric_comparison='Paired global and local predictions on identical held-out test configurations. Equal-deployment macro of trial means; report std and worst case.',
    budgets=[0, 1, 2, 3],
    win='Trial-mean MAE improves at least 10% AND trial-mean sMAPE does not increase.',
    harm='Trial-mean MAE worsens >10% AND >5 ms.',
    worst_collapse='Any trial local MAE >2x its matched global MAE AND >50 ms increase.',
    PASS='Same K on >=5 strict folds; repairs BOTH original GBT non-win contexts; >=80% no harm; >=10% macro MAE improvement, no worse macro sMAPE, no worst collapse; extended macro no >10% harm; all warning settings agree.',
    PARTIAL='Not PASS; at least two strict or extended folds have substantive mean improvement at some K in nominal thresholds.',
    FAIL='No stable evidence of benefit by the above rule; do not add calibration model complexity.',
    original_failure_definition='Complement of Module 2D primary C_gbt_log_ratio substantive-win contexts (not the separate Ridge failures).',
)


def jsonl(path, values):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in values))


def summary_stats(scores):
    result = {}
    for k in METRICS:
        values = [s[k] for s in scores if s[k] is not None]
        result[k] = dict(mean=float(np.mean(values)) if values else None,
            sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            worst_case=float(min(values) if k == 'spearman' else max(values)) if values else None,
            valid_trials=len(values))
    return result


def aggregation(trials):
    result = {}
    for subset in ('strict', 'extended'):
        result[subset] = {}
        for k in (0, 1, 2, 3):
            ts = [t for t in trials if t['subset'] == subset and t['K'] == k]
            by_dep = {}
            for dep in sorted({t['heldout_deployment'] for t in ts}):
                ds = [t for t in ts if t['heldout_deployment'] == dep]
                gs, ls = summary_stats([d['global'] for d in ds]), summary_stats([d['bias_only'] for d in ds])
                gm, lm = gs['mae_ms']['mean'], ls['mae_ms']['mean']
                win = lm <= .9 * gm and ls['smape_percent']['mean'] <= gs['smape_percent']['mean']
                harm = lm > 1.1 * gm and lm - gm > 5
                worst = any(d['bias_only']['mae_ms'] > 2 * d['global']['mae_ms'] and d['bias_only']['mae_ms'] - d['global']['mae_ms'] > 50 for d in ds)
                gains = [1 - d['bias_only']['mae_ms'] / d['global']['mae_ms'] for d in ds]
                by_dep[dep] = dict(trials=len(ds), global_metrics=gs, bias_only_metrics=ls,
                    unique_calibration_sets=len({tuple(sorted(d['calibration_configurations'])) for d in ds}),
                    substantive_win=win, material_harm=harm, worst_trial_collapse=worst,
                    paired_relative_MAE_improvement=dict(mean=float(np.mean(gains)), sample_std=float(np.std(gains, ddof=1)) if len(gains) > 1 else None, worst_case=min(gains)),
                    accepted_calibration_configuration_range=[min(d['accepted_configuration_count'] for d in ds), max(d['accepted_configuration_count'] for d in ds)],
                    global_fallback_trials=sum(d['accepted_configuration_count'] == 0 for d in ds))
            macro = {}
            for method in ('global_metrics', 'bias_only_metrics'):
                macro[method] = {m: float(np.mean([d[method][m]['mean'] for d in by_dep.values() if d[method][m]['mean'] is not None]))
                                if any(d[method][m]['mean'] is not None for d in by_dep.values()) else None for m in METRICS}
            result[subset][str(k)] = dict(deployments=by_dep, deployment_count=len(by_dep), macro=macro,
                note='Only supported folds; compare matched global at this K, never a different-coverage K=0 macro.')
    return result


def marginal_gains(trials):
    out = []
    for subset in ('strict', 'extended'):
        index = {(t['heldout_deployment'], t['trial_id'], t['K']): t for t in trials if t['subset'] == subset}
        for k in (1, 2, 3):
            changes = defaultdict(list)
            for (dep, tid, budget), curr in index.items():
                previous = index.get((dep, tid, k - 1))
                if budget != k or previous is None: continue
                if previous['test_row_ids'] != curr['test_row_ids']: raise ValueError('Unpaired marginal budget comparison')
                changes[dep].append(previous['bias_only']['mae_ms'] - curr['bias_only']['mae_ms'])
            out.append(dict(subset=subset, from_K=k-1, to_K=k, deployments=list(changes),
                mean_MAE_reduction_ms_by_deployment={dep: float(np.mean(v)) for dep, v in changes.items()},
                macro_MAE_reduction_ms=float(np.mean([np.mean(v) for v in changes.values()])) if changes else None,
                scope='Same trial, same test configurations, common deployment support only. Positive = improvement.'))
    return out


def gate(aggregates, failures):
    evidence = {}
    for k in (1, 2, 3):
        strict, extended = aggregates['strict'][str(k)], aggregates['extended'][str(k)]
        ds = strict['deployments']
        g, l = strict['macro']['global_metrics'], strict['macro']['bias_only_metrics']
        eg, el = extended['macro']['global_metrics'], extended['macro']['bias_only_metrics']
        repaired = [d for d in failures if d in ds and ds[d]['substantive_win']]
        wins = [d for d, v in ds.items() if v['substantive_win']]
        harms = [d for d, v in ds.items() if v['material_harm']]
        collapse = [d for d, v in ds.items() if v['worst_trial_collapse']]
        passed = (len(ds) >= 5 and len(repaired) == len(failures) and len(harms) <= .2 * len(ds)
            and g['mae_ms'] is not None and l['mae_ms'] <= .9 * g['mae_ms']
            and l['smape_percent'] <= g['smape_percent'] and not collapse
            and eg['mae_ms'] is not None and el['mae_ms'] <= 1.1 * eg['mae_ms'])
        evidence[str(k)] = dict(passes=passed, strict_wins=wins, strict_harms=harms, worst_collapses=collapse,
            original_failures_repaired=repaired, original_failures_unavailable=[d for d in failures if d not in ds],
            strict_supported_deployments=len(ds),
            extended_wins=[d for d, v in extended['deployments'].items() if v['substantive_win']])
    return evidence


def simulate(rows, labels, ex, resource_keys, scenario, original_failures):
    all_predictions, trials, audits, all_global = [], [], [], []
    for dep in sorted({r['deployment_id'] for r in rows}):
        training = [r for r in rows if r['deployment_id'] != dep and labels[r['run_id']]['stability_state_v2'] == 'steady' and '_y' in r]
        heldout = [r for r in rows if r['deployment_id'] == dep]
        if set(r['configuration_id'] for r in training) & set(r['configuration_id'] for r in heldout): raise ValueError('Outer configuration leakage')
        if set(r['run_id'] for r in training) & set(r['run_id'] for r in heldout): raise ValueError('Outer run leakage')
        pre = FoldPreprocessor().fit([r['_v'] for r in training], list(training[0]['_v']))
        model = GradientBoostingRegressor(**GBT)
        model.fit(pre.transform([r['_v'] for r in training]),
                  [math.log(r['_y'] / r['_s0']) for r in training], sample_weight=training_weights(training))
        def predict(vectors): return model.predict(pre.transform(vectors))
        yh = {r['_id']: float(v) for r, v in zip(heldout, predict([r['_v'] for r in heldout]))}
        cfg_reps = {r['configuration_id']: r for r in heldout}
        train_cfg_reps = {r['configuration_id']: r for r in training}
        desc = {c: planning_descriptor(r, ex, resource_keys, predict) for c, r in cfg_reps.items()}
        training_desc = {c: planning_descriptor(r, ex, resource_keys, predict) for c, r in train_cfg_reps.items()}
        plan = selection_plan(desc, training_desc)
        fold_audit = dict(scenario=scenario, heldout_deployment=dep,
            global_training_deployments=sorted({r['deployment_id'] for r in training}),
            global_training_configurations=sorted(train_cfg_reps), global_training_run_ids=sorted({r['run_id'] for r in training}),
            global_training_row_ids=[r['_id'] for r in training],
            heldout_run_ids=sorted({r['run_id'] for r in heldout}), heldout_configurations=sorted(cfg_reps),
            preprocessing=pre.audit([r['_v'] for r in heldout]), selection_plan=plan,
            outer_overlap=dict(deployment=[], configuration=[], run=[]), trials=[])
        def predict_record(r, tid, k, method, bias, role):
            ypred = yh[r['_id']] + bias
            predicted = r['_s0'] * math.exp(ypred)
            if not math.isfinite(predicted): raise ValueError('Invalid restored latency')
            return dict(scenario=scenario, evaluation_role=role, heldout_deployment=dep, trial_id=tid, K=k, method=method,
                **{key:r[key] for key in ('deployment_id','configuration_id','run_id','repetition_id','source_result_path','source_sha256','task','target_model','seed')},
                row_id=r['_id'], stability_state_v2=labels[r['run_id']]['stability_state_v2'],
                standalone_latency_ms=r['_s0'], true_latency_ms=r['_y'], predicted_latency_ms=predicted,
                global_predicted_log_ratio=yh[r['_id']], bias_log_ratio=bias,
                absolute_error_ms=abs(predicted-r['_y']))
        eligible = [r for r in heldout if labels[r['run_id']]['stability_state_v2'] != 'unstable' and '_y' in r]
        global_records = [predict_record(r, 'all_configurations', 0, 'global', 0., 'global_K0_all') for r in eligible]
        all_predictions.extend(global_records)
        for subset in ('strict', 'extended'):
            ps = [p for p in global_records if subset == 'extended' or p['stability_state_v2'] == 'steady']
            all_global.append(dict(heldout_deployment=dep, subset=subset, status='evaluated' if ps else 'no_eligible_rows',
                metrics=metrics([p['true_latency_ms'] for p in ps], [p['predicted_latency_ms'] for p in ps]) if ps else None))
        for trial in plan['trials']:
            test = [r for r in eligible if r['configuration_id'] in trial['test_configurations']]
            reserved = [r for r in heldout if r['configuration_id'] in trial['calibration_chain']]
            if set(r['run_id'] for r in test) & set(r['run_id'] for r in reserved): raise ValueError('Inner repetition/run leakage')
            for k in range(trial['max_supported_K'] + 1):
                chosen = trial['calibration_chain'][:k]
                correction = bias_from_probes(heldout, chosen, yh, labels)
                part = dict(trial_id=trial['trial_id'], K=k, calibration_configurations=chosen,
                    reserved_unused_configurations=trial['calibration_chain'][k:], test_configurations=trial['test_configurations'],
                    selected_run_ids=correction['measurement_run_ids'], test_run_ids=sorted({r['run_id'] for r in test}),
                    correction=correction, configuration_overlap=[], run_overlap=[],
                    selected_repetition_groups=sorted({(r['configuration_id'],r['run_id'],r['repetition_id']) for r in heldout if r['configuration_id'] in chosen}),
                    test_repetition_groups=sorted({(r['configuration_id'],r['run_id'],r['repetition_id']) for r in test}))
                fold_audit['trials'].append(part)
                paired = {}
                for method, bias in [('global', 0.), ('bias_only', correction['bias_log_ratio'])]:
                    ps = [predict_record(r, trial['trial_id'], k, method, bias, 'paired_fixed_test') for r in test]
                    paired[method] = ps
                    all_predictions.extend(ps)
                for subset in ('strict', 'extended'):
                    by_method = {m: [p for p in ps if subset == 'extended' or p['stability_state_v2'] == 'steady'] for m, ps in paired.items()}
                    if not by_method['global']:
                        part.setdefault('unavailable_evaluation_subsets', []).append(subset)
                        continue
                    score = {m: metrics([p['true_latency_ms'] for p in ps], [p['predicted_latency_ms'] for p in ps]) for m, ps in by_method.items()}
                    trials.append(dict(scenario=scenario, heldout_deployment=dep, trial_id=trial['trial_id'], K=k,
                        subset=subset, calibration_configurations=chosen, test_configurations=trial['test_configurations'],
                        test_row_ids=[p['row_id'] for p in by_method['global']], **score,
                        accepted_configuration_count=correction['effective_configuration_count'],
                        calibration_measurement_seconds=correction['calibration_measurement_seconds']))
        audits.append(fold_audit)
    aggregate = aggregation(trials)
    all_global_macro = {subset: {m: float(np.mean([x['metrics'][m] for x in all_global if x['subset'] == subset and x['metrics'] and x['metrics'][m] is not None]))
        if any(x['subset'] == subset and x['metrics'] and x['metrics'][m] is not None for x in all_global) else None for m in METRICS}
        for subset in ('strict', 'extended')}
    return dict(global_K0_all=all_global, global_K0_all_macro=all_global_macro,
        paired_trial_metrics=trials, paired_aggregates=aggregate, marginal_gains=marginal_gains(trials),
        decision_evidence=gate(aggregate, original_failures)), all_predictions, audits


def report(results, decision, sensitivity, rows, audits, failures):
    names = {r['deployment_id']: '+'.join(m for m, n in r['deployment'].items() if n) for r in rows}
    nominal = results['nominal']
    lines = ['# Module 2E：Stability Semantics + Few-shot Local Calibration', '', f'**Decision Gate：{decision}**。', '',
        {'PASS':'支持 Global Predictor + Few-shot Local Calibration 的当前表示与固定方案；本轮不导出正式 predictor。',
         'PARTIAL':'部分配置的局部校准有收益，但尚不能稳定修复跨 deployment 失败。下一步可考虑 task-aware residual correction 或 targeted active calibration，本轮不实现。',
         'FAIL':'当前 bias-only 校准没有稳定收益；不继续增加 calibration model 复杂度，可考虑 setup-local model + guided profiling，本轮不执行。'}[decision], '',
        '## A. 三态语义与时间线', '',
        '保持全部原始 result、stable 字段及 v1 stability 对象不变。新增 run-level stability_state_v2 / stability_reason_v2；dataset_rows_v2.jsonl 同时保留原始 row 与新增标签。',
        'B(t) 使用 actual_submit_time_ns（host send-start）减成功完成，不等价于 worker ACK。W(t) 对每个已提交未完成请求求完整 standalone s0 之和，单位 standalone service-seconds，不是剩余 GPU 工作量。',
        '同一单调时钟上的逐请求事件精确重建峰值及 bin 时间积分；输出窗口末值与 drain 后最终值。scheduled-arrival backlog 单列，不能与提交 backlog 混用。',
        '硬不稳定：incomplete、drain timeout、completion<0.98、worker/run error 或有非零末积压的持续 tail 增长。即便最后 drain 完成，持续增长的 tail 仍可 unstable。',
        'recoverable_congested：完整恢复但达到 backlog>=30、W>=5 s、queue wait>=1 s 的请求数达到 max(10,5% offered)、或 drain>=5 s 中任一警戒条件。steady 为其余完整完成且未触警戒的运行。',
        'tail 使用最后 25% 的 20-bin 时间线；同时要求 slope>=.02 requests/s、growth>=5、>=60% 相邻 bin 增长、end backlog>=5。没有单一 E2E latency 阈值。', '',
        '| Warning 倍率 | steady runs | recoverable runs | unstable runs | steady rows | extended rows |',
        '|---|---:|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        s = sensitivity[scenario]
        c = s['run_counts']
        lines.append(f"| {SCENARIOS[scenario]:.2f} | {c.get('steady',0)} | {c.get('recoverable_congested',0)} | {c.get('unstable',0)} | {s['row_counts'].get('steady',0)} | {sum(v for k,v in s['row_counts'].items() if k!='unstable')} |")
    lines += ['', '仅缩放 congestion warning 的四个幅度阈值；hard instability 条件固定。每个 warning 设置都重做相同 LODO/校准流程，未调参。',
        '先前异常 B_text_dominant_rep_2 现在为 recoverable_congested：实际提交 backlog 峰值 158；旧诊断 >600 是 scheduled backlog 的 bin mean。两者定义不同，原始数据未被修改。', '',
        '## B. 校准选择与防泄漏', '',
        '- 外层完整 deployment LODO，global 使用其他部署的 v2 steady rows；固定 Module 2D GBT 参数、41 个 planning features 和 log(L/s0) target。strict 与 extended 共享同一个 global fit，差异只在评估区域。',
        '- preprocessing、distance normalization 仅 fit 在 training deployment 数据。新 deployment 的任何标签都不进入 global training。',
        '- planning safety screen：单 resident 每模型 u<=.20，多 resident 每模型 u<=.40；只是保守筛查，不能证明 steady。排序用 planning load、global prediction 和 feature distance，不读 test labels。',
        '- 最多三个不同起始 anchor，每个做 farthest-first nested selection。选中 probe 才揭示标签；若该配置任一 repetition 非 steady，拒绝其 residual，但仍计入已花费 K，不用 test labels 找替补。',
        '- primary correction 只有 b=median(log(L/s0)-global prediction)。使用所选配置的全部现有 repetitions/path rows；K 是配置数，真实测量成本还包含重复次数和窗口秒数。无 affine fitting。',
        '- 每个 trial 预留整个最大 calibration chain；余下 configuration 固定作为 test。因此同 trial 的 K=0/1/2/3 使用完全相同 test rows，未用到的 reserve 也不能混入 test。',
        '- 一个配置的所有 repetitions/tasks 始终在同一侧。trial 之间可能复用配置，因此不是独立实验，也不报告统计显著性。', '',
        '| Deployment | total configs | planning-safe configs | max K with independent test | trials |',
        '|---|---:|---:|---:|---:|']
    for a in audits:
        if a['scenario'] != 'nominal': continue
        p = a['selection_plan']
        lines.append(f"| {names[a['heldout_deployment']]} | {len(p['candidates'])} | {len(p['safe_candidates'])} | {p['max_supported_K']} | {len(p['trials'])} |")
    lines += ['', 'Qwen+ResNet50 只有一个配置，所有 K>0 不可独立评估；绝不拆 repetitions 凑出 calibration/test。K=3 的支持范围很窄，不能把不同 fold 组成的 macro 差异当作预算收益。', '',
        '## C. 配对评估结果', '',
        '下表为 nominal thresholds 下、同一 test 配置上的 deployment-macro MAE。每个 deployment 先对所有 deterministic trials 取平均，再等权 macro。', '',
        '| Evaluation region | K | supported deployments | matched Global MAE ms | Bias-only MAE ms |',
        '|---|---:|---:|---:|---:|']
    for subset, budgets in nominal['paired_aggregates'].items():
        for k, a in budgets.items():
            g, l = a['macro']['global_metrics']['mae_ms'], a['macro']['bias_only_metrics']['mae_ms']
            lines.append(f"| {subset} | {k} | {a['deployment_count']} | {g:.2f} | {l:.2f} |" if g is not None else f'| {subset} | {k} | 0 | null | null |')
    lines += ['', '不预留 calibration 的全数据 K=0 macro 单独保存为 global_K0_all_macro；不能拿它与上表改变了 test 支持集的 K>0 直接比较。', '',
        '本轮也不能与 Module 2D 的 449.8 ms 直接比较：v2 改变了健康区标签、global training subset 和本地 calibration/test 支持集；数值变小并不是重新调优模型的证据。', '',
        '每 deployment 的 strict MAE（mean ± trial sample std；括号为 worst trial）：', '',
        '| Deployment | K | matched Global mean | local mean ± std (worst) | result |',
        '|---|---:|---:|---:|---|']
    for k in ('1','2','3'):
        for dep, d in nominal['paired_aggregates']['strict'][k]['deployments'].items():
            g, l = d['global_metrics']['mae_ms'], d['bias_only_metrics']['mae_ms']
            sd = 'null' if l['sample_std'] is None else f"{l['sample_std']:.2f}"
            result = 'improves' if d['substantive_win'] else 'degrades' if d['material_harm'] else 'mixed/small change'
            lines.append(f"| {names[dep]} | {k} | {g['mean']:.2f} | {l['mean']:.2f} ± {sd} ({l['worst_case']:.2f}) | {result} |")
    lines += ['', 'MAE、median AE、sMAPE、P90 AE、Spearman 的每 trial、每 deployment、macro、sample std 和 worst-case 全部保留在 calibration_fold_metrics.json。以下是相同 trial/test 支持下的边际 MAE 减少，正数才是收益：', '']
    for m in nominal['marginal_gains']:
        value = m['macro_MAE_reduction_ms']
        lines.append(f"- {m['subset']} K={m['from_K']}→{m['to_K']}：{value:.2f} ms，{len(m['deployments'])} deployments。" if value is not None else f"- {m['subset']} K={m['from_K']}→{m['to_K']}：不可评估。")
    lines += ['', '## D. 原失败 context、退化与 decision gate', '',
        '原来 GBT 的两个非改善 context 是：' + '、'.join(names[d] for d in failures) + '。这不同于 Module 2D 的 Ridge collapse contexts。',
        '其中单配置 context 的修复能力无法验证。PASS 必须修复两个原失败 context、覆盖至少五个 strict folds、macro MAE 至少下降10%、sMAPE 不升、至少80% folds无明显退化且无最坏 trial collapse；extended不能显著退化，三个阈值设置一致。',
        'PARTIAL 要求 nominal 某个 K 在 strict 或 extended 至少两个 folds 明显改善；否则 FAIL。阈值是预先固定的工程 gate，不是统计显著性检验。', '']
    for scenario, result in results.items():
        for k, ev in result['decision_evidence'].items():
            lines.append(f"- {scenario}, K={k}：strict 改善 {len(ev['strict_wins'])}/{ev['strict_supported_deployments']}；明显退化 {len(ev['strict_harms'])}；最坏 trial collapse {len(ev['worst_collapses'])}；原失败修复 {len(ev['original_failures_repaired'])}/2；extended 改善 {len(ev['extended_wins'])}。")
    # Cost judgement uses paired evidence, not the best individual trial.
    options = [(k,a) for k,a in nominal['paired_aggregates']['strict'].items() if k!='0' and a['deployment_count']>=3]
    supported_benefits = [(k,a) for k,a in options if a['macro']['bias_only_metrics']['mae_ms']<a['macro']['global_metrics']['mae_ms']]
    lines += ['', '## E. 结论与限制', '']
    if supported_benefits:
        best = min(supported_benefits, key=lambda ka:int(ka[0]))
        lines.append(f"在至少三个可评估 contexts 中，K={best[0]} 是观察到 macro MAE 收益的最小预算；这不是对全部新 deployment 的性价比保证。K 间的正式比较以 common-support marginal gain 为准。")
    else:
        lines.append('当前数据下 K=0 更合算：1–3 shot 未在至少三个 contexts 上提供正 macro MAE 收益，还增加配置测量成本。没有证据支持支付 bias-only local calibration 的成本。')
    lines += ['Strict 衡量健康稳定区；Extended 加入可恢复拥塞且仍排除 unstable。Global 仅训练 steady，所以 extended 也检验健康区模型向拥塞区外推，不能把失败自动归因于校准算法。',
        '宏观结论不能用最好的 anchor 掩盖最坏 trial。部分 probe 会在揭示三态标签后被拒绝，仍计入成本；该机制的收益/损失也计入结果。',
        '现有数据覆盖稀疏、每个 deployment 的配置数不等，并有不同 measurement protocols。此为回顾性 few-shot simulation，尚无新部署的前瞻性验证。',
        '告警阈值敏感性只覆盖0.75–1.25倍的局部范围；不表示阈值是客观稳定性真理。依据 request records 的 operational state 也不是严格排队稳定性的数学证明。', '',
        '## 文件与边界', '',
        '- stability_v2_labels.jsonl / dataset_rows_v2.jsonl：新增标签、原始 stable、source hashes 与 group IDs。',
        '- stability_threshold_sensitivity.json：完整阈值、三态分布与 deployment 分布。',
        '- calibration_fold_metrics.json：外层 folds、预算/试验、完整五项指标与敏感性 gate。',
        '- calibration_predictions.jsonl：逐测试 row、global/local 配对预测与四级分组。',
        '- calibration_selection_audit.json：选择特征/距离/安全筛查、global fit 组、预处理、probe reveal 与 test 隔离。',
        '仅 CPU 读取旧结果；没有运行 worker、新增 profiling、deployment search、routing optimizer、最终 predictor training 或 forecasting。完成后停止。', '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=INPUT)
    ap.add_argument('--output', type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save(out/'evaluation_config.json', POLICY)
    previous_gate = ROOT/'results/performance_model_generalization'
    protected = {str(p):sha(p) for folder in ('performance_model','deployment_core','joint_deploy') for p in (ROOT/folder).rglob('*.py')
                 if 'local_calibration' not in p.parts}
    protected.update({str(p):sha(p) for p in previous_gate.iterdir() if p.is_file()})
    source_hash = sha(args.input)
    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    ex = extractor()
    resource_keys = sorted({k for p in ex.resource_profiles.values() for k in p['values']})
    calibration_metadata = dict(standalone_catalog_sources={m:p.sources for m,p in ex.catalog.items()},
        resource_profile_sources=[s for p in ex.resource_profiles.values() for s in p['sources']])
    def collect_sources(obj):
        if isinstance(obj,dict):
            if 'path' in obj and 'sha256' in obj: yield obj['path'],obj['sha256']
            for value in obj.values(): yield from collect_sources(value)
        elif isinstance(obj,(list,tuple)):
            for value in obj: yield from collect_sources(value)
    offline_sources = dict(collect_sources(calibration_metadata))
    runs, sources, exclusions = {}, {}, []
    for r in rows:
        if r['deployment_id'] != deployment_id(r['deployment']): raise ValueError('Noncanonical deployment ID')
        if r['configuration_id'] != evaluation_configuration_id(dict(deployment=r['deployment'],workload_rps=r['workload'],routing_rps=r['routing'])):
            raise ValueError('Noncanonical configuration ID')
        r['_id'] = row_key(r)
        prev = runs.setdefault(r['run_id'], r)
        if any(prev[k] != r[k] for k in ('stability','configuration_id','deployment_id','repetition_id','source_result_path')):
            raise ValueError('Inconsistent run metadata')
        v, s0, features = planning_vector(r, ex, 'C', resource_keys)
        if not math.isclose(v['u'],r['standalone_service_demand'],rel_tol=1e-10,abs_tol=1e-12): raise ValueError('Standalone feature drift')
        if not same_context(features['per_model'][r['target_model']]['colocation_pressure'],r['colocation_pressure']): raise ValueError('Pressure feature drift')
        if not same_context(ex.resource_profiles[(r['task'],r['target_model'])]['values'],r['service_path_resource_features']): raise ValueError('Offline fingerprint drift')
        r.update(_v=v, _s0=s0)
        y = r['labels']['mean_e2e_latency_ms']
        if y is not None and math.isfinite(y) and y > 0: r['_y'] = y
        else: exclusions.append(dict(row_id=r['_id'],reason='invalid/missing measured mean E2E target'))
    labels, sensitivity = {s:{} for s in SCENARIOS}, {}
    label_rows = []
    for rid, r in runs.items():
        path = r['source_result_path']
        sources[path] = sha(path)
        if sources[path] != r['source_sha256']: raise ValueError('Raw source changed')
        data = json.loads(Path(path).read_text())
        evidence = reconstruct(data, ex.catalog, DEFAULT_THRESHOLDS)
        for scenario, factor in SCENARIOS.items(): labels[scenario][rid] = classify(evidence, scaled_thresholds(factor))
        public_evidence = {k:v for k,v in evidence.items() if k!='queue_wait_values_ms'}
        label_rows.append(dict(**{k:r[k] for k in ('deployment_id','configuration_id','run_id','repetition_id','source_result_path','source_sha256','seed')},
            stable=r['stability']['stable'], original_stability=r['stability'], **labels['nominal'][rid],
            evidence=public_evidence, sensitivity_labels={s:labels[s][rid] for s in SCENARIOS}))
    for scenario, ls in labels.items():
        sensitivity[scenario] = dict(thresholds=scaled_thresholds(SCENARIOS[scenario]),
            run_counts=dict(Counter(l['stability_state_v2'] for l in ls.values())),
            row_counts=dict(Counter(ls[r['run_id']]['stability_state_v2'] for r in rows)),
            per_deployment={d: dict(run_counts=dict(Counter(ls[rid]['stability_state_v2'] for rid,r in runs.items() if r['deployment_id']==d))) for d in sorted({r['deployment_id'] for r in rows})},
            transitions_from_original_category=dict(Counter(runs[rid]['stability']['category']+' -> '+l['stability_state_v2'] for rid,l in ls.items())))
    jsonl(out/'stability_v2_labels.jsonl', label_rows)
    jsonl(out/'dataset_rows_v2.jsonl', [{k:v for k,v in r.items() if not k.startswith('_')} |
        dict(stable=r['stability']['stable'],stability_state_v2=labels['nominal'][r['run_id']]['stability_state_v2'],
             stability_reason_v2=labels['nominal'][r['run_id']]['stability_reason_v2']) for r in rows])
    save(out/'stability_threshold_sensitivity.json', sensitivity)
    old = json.loads((previous_gate/'fold_metrics.json').read_text())
    original_failures = sorted({r['deployment_id'] for r in rows} - set(old['gate_evidence']['primary']['gbt']['win_deployments']))
    results, predictions, audit = {}, [], []
    for scenario in SCENARIOS:
        results[scenario], ps, fs = simulate(rows, labels[scenario], ex, resource_keys, scenario, original_failures)
        predictions.extend(ps)
        audit.extend(fs)
        print(scenario, sensitivity[scenario]['run_counts'], 'complete', flush=True)
    pass_budgets = [k for k in ('1','2','3') if all(r['decision_evidence'][k]['passes'] for r in results.values())]
    partial = any(len(e['strict_wins'])>=2 or len(e['extended_wins'])>=2 for e in results['nominal']['decision_evidence'].values())
    decision = 'PASS' if pass_budgets else 'PARTIAL' if partial else 'FAIL'
    save(out/'calibration_fold_metrics.json', dict(decision=decision, policy=POLICY, original_GBT_nonwin_deployments=original_failures, scenarios=results))
    jsonl(out/'calibration_predictions.jsonl', predictions)
    if sha(args.input) != source_hash or any(sha(p)!=h for p,h in sources.items()): raise ValueError('Historical data changed')
    if any(sha(p)!=h for p,h in protected.items()): raise ValueError('Protected code or Module 2D result changed')
    if any(sha(p)!=h for p,h in offline_sources.items()): raise ValueError('Offline calibration profile changed')
    if set(offline_sources) & set(sources): raise ValueError('Joint evaluation result used as offline calibration')
    save(out/'calibration_selection_audit.json', dict(status='PASS', input_path=str(args.input.resolve()), input_sha256=source_hash,
        original_rows=len(rows), original_runs=len(runs), source_hashes=sources, protected_source_hashes=protected,
        exclusions=exclusions, feature_schema=list(rows[0]['_v']), selection_policy=SELECTION_POLICY,
        heldout_labels_used_for_selection=False, selected_probe_labels_used_only_after_selection_for_acceptance=True,
        offline_profile_metadata=calibration_metadata, offline_profile_hashes=offline_sources,
        offline_joint_result_source_overlap=[],
        original_labels_preserved=True, old_global_results_unchanged=True,
        trials_are_deterministic_not_independent=True, affine_model_implemented=False,
        implementation_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}, folds=audit))
    (out/'local_calibration_summary.md').write_text(report(results, decision, sensitivity, rows, audit, original_failures))
    print('Decision:', decision, 'prediction rows:',len(predictions),flush=True)


if __name__ == '__main__': main()
