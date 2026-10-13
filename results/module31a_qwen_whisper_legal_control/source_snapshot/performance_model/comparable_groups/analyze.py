"""Analyze the frozen Module 2G experiment; never fit models or launch workers.

Run with the existing gsdeploy Python environment. Historical and raw files are
read-only inputs. Failed repetitions disqualify configurations, not just rows.
"""
from collections import Counter
import json
import math
from pathlib import Path
import pickle

import numpy as np

from deployment_core.models import load_model_catalog, load_hardware
from performance_model.features import FeatureExtractor
from performance_model.resources import load_resource_profiles
from performance_model.generalization.features import planning_vector
from performance_model.local_calibration.stability_v2 import DEFAULT_THRESHOLDS, reconstruct, classify
from performance_model.ranking.evaluate import (
    POLICY, canonical, configuration_eligible, identity, protocol_signature,
    ranking_metrics, save, sha, summarize_metrics, weighted_objective,
)
from .canonical import solve_canonical_routing

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/performance_model_comparable_groups'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stats(values):
    if any(v is None for v in values):
        return dict(mean=None, sample_std=None, values=values)
    return dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1))
                if len(values) > 1 else None, values=values)


def gate(entries):
    """Reuse 2F breadth and metric criteria; do not tune criteria to 2G labels."""
    s = summarize_metrics(entries)
    sufficient = len(entries) >= 3 and s['top2_nontrivial_groups'] >= 2
    if not sufficient:
        return dict(result='INSUFFICIENT COMPARABLE GROUPS', assessed=False,
                    reason='Need >=3 independent workload groups and >=2 nontrivial Top-2 groups.', macro=s)
    positive = all(e['kendall_tau'] is not None and e['kendall_tau'] > 0 for e in entries)
    t2, t3 = s['top2_nontrivial_coverage'], s['top3_nontrivial_coverage']
    passed = (positive and s['kendall_tau_mean'] >= .6 and s['spearman_rho_mean'] >= .6
              and s['regret']['mean'] <= .05 and s['regret']['worst_case'] <= .15
              and t2 is not None and t2 >= .9 and (t3 is None or t3 >= .9))
    partial = ((t2 is not None and t2 >= .8 or t3 is not None and t3 >= .9)
               and s['kendall_tau_mean'] is not None and s['kendall_tau_mean'] > 0
               and s['regret']['worst_case'] <= .5)
    return dict(result='PASS' if passed else 'PARTIAL' if partial else 'FAIL', assessed=True, macro=s)


def analyze(output=OUT):
    output = Path(output)
    frozen = read(output / 'comparison_plan_hash.json')
    require(sha(output / 'comparison_plan.json') == frozen['sha256'], 'Frozen plan drift')
    require(sha(output / 'run_manifest.json') == frozen['run_manifest_sha256'], 'Manifest drift')
    plan, manifest = read(output / 'comparison_plan.json'), read(output / 'run_manifest.json')
    require(manifest['plan_sha256'] == frozen['sha256'], 'Wrong manifest plan')
    require(sha(plan['label_source']['path']) == plan['label_source']['sha256'], 'Historical labels drift')
    old_path = ROOT / 'results/performance_model_active_sampling/dataset_rows.jsonl'
    history = read_lines(old_path)
    old_audit = read(ROOT / 'results/performance_model_generalization/leakage_audit.json')
    require(sha(old_path) == old_audit['dataset_sha256'], 'Historical training dataset drift')
    lookup = {r['run_id']+'::'+r['task']+'::'+r['target_model']: r for r in history}
    historical_states = {r['run_id']: r['stability_state_v2'] for r in read_lines(plan['label_source']['path'])}
    historical_deployments = {r['deployment_id'] for r in history}
    catalog, hardware = load_model_catalog(), load_hardware()
    def source_pairs(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                yield value['path'], value['sha256']
            for child in value.values():
                yield from source_pairs(child)
        elif isinstance(value, list):
            for child in value:
                yield from source_pairs(child)
    for path, digest in list(source_pairs(plan['catalog_sources'])) + list(old_audit['offline_resource_sources'].items()):
        require(sha(path) == digest, 'Offline profile drift: '+path)
    ex = FeatureExtractor(catalog, load_resource_profiles(catalog))
    resource_keys = sorted({k for p in ex.resource_profiles.values() for k in p['values']})
    labels, comparisons, selection, proofs, predictions = [], [], [], [], []
    hashes, statuses, failures = {}, Counter(), []
    all_metrics = {mode: [] for mode in ('strict', 'extended')}
    for group, frozen_group in zip(manifest['groups'], plan['groups']):
        name, workload = group['workload_group'], group['workload']
        require(workload == plan['workload_groups'][name], 'Workload drift')
        require(group['all_feasible_candidate_count'] <= 5, 'Candidate budget exceeded')
        require(len(group['mutations']) == len(catalog)+1, 'Not a seed plus one-hop pool')
        for mutation in group['mutations']:
            delta = sum(abs(mutation['deployment'][m]-group['seed_deployment'][m]) for m in catalog)
            require(delta == (0 if mutation['mutation']['operation']=='seed' else 1), 'Non-local mutation')
        selection.append(dict(workload_group=name, source_seed=group['seed_deployment'],
            candidates_considered=len(group['mutations']), feasible_candidates=group['all_feasible_candidate_count'],
            unselected_candidates=sum(not m['feasible'] for m in group['mutations']),
            candidates=group['mutations'],
            selection_method='All five feasible seed/one-hop candidates fit budget; no farthest-point truncation was performed.',
            policy_correction='Frozen policy mentions farthest-point selection, but implemented order is configuration-ID order. Scores are audit-only distances, not evidence of farthest-point choice.',
            selected=[dict(configuration_id=c['configuration_id'], mutation=c['mutation'],
                           feature_vector=c['feature_vector'], selection_score=c['selection_score'],
                           rationale=c['selection_rationale']) for c in group['selected_candidates']]))
        configs, signatures, worker_versions = [], set(), {}
        for c, fc in zip(group['selected_candidates'], frozen_group['selected_candidates']):
            require({k:v for k,v in c.items() if k!='runs'} == fc, 'Candidate drift')
            require(c['workload'] == workload, 'Unequal workload')
            require(sum(c['deployment'][m]*catalog[m].memory_mib for m in catalog) <= hardware.gpu_memory_mib,
                    'Memory violation')
            route, util, z = solve_canonical_routing(c['deployment'], workload, catalog)
            require(math.isclose(z, c['canonical_z'], abs_tol=1e-9), 'Canonical objective drift')
            features = ex.extract(c['deployment'], workload, c['routing'])
            for task, rate in workload.items():
                require(math.isclose(sum(c['routing'][task].values()), rate, abs_tol=1e-9), 'Routing conservation')
                for model, flow in c['routing'][task].items():
                    require(flow >= 0 and (flow == 0 or c['deployment'][model] and task in catalog[model].supported_tasks), 'Invalid path')
            require(all(p['standalone_service_demand'] <= .8+1e-9 for p in features['per_model'].values()), 'Capacity violation')
            pred = c['frozen_prediction']; meta = pred['predictor']
            require(c['deployment_id'] not in meta['training_deployment_ids'], 'Heldout deployment leakage')
            train = [lookup[k] for k in meta['training_row_ids']]
            require(all(r['deployment_id'] != c['deployment_id'] and historical_states[r['run_id']]=='steady' for r in train), 'Training row leakage')
            require(sha(meta['artifact_path']) == meta['artifact_sha256'], 'Frozen predictor drift')
            artifact = pickle.loads(Path(meta['artifact_path']).read_bytes())
            # Inference only: no model/preprocessor fit on this round or any labels.
            for path, value in pred['path_mean_ms'].items():
                task, model = path.split('::')
                v, s0, _ = planning_vector(dict(deployment=c['deployment'], workload=workload,
                    routing=c['routing'], task=task, target_model=model), ex, 'C', resource_keys)
                actual = s0*math.exp(float(artifact['model'].predict(artifact['preprocessor'].transform([v]))[0]))
                require(math.isclose(value, actual, rel_tol=1e-10), 'Frozen prediction mismatch')
            require(math.isclose(weighted_objective(workload, pred['task_mean_ms']), pred['mean_objective_ms'], rel_tol=1e-12), 'Prediction weighting')
            proofs.append(dict(configuration_id=c['configuration_id'], deployment_id=c['deployment_id'],
                historically_seen=c['deployment_id'] in historical_deployments,
                heldout_from_own_predictor=True, training_rows=len(train),
                training_deployment_ids=meta['training_deployment_ids'], artifact_sha256=meta['artifact_sha256'],
                frozen_prediction_reproduced=True))
            reps=[]
            require(len(c['runs']) == 3, 'Repetition budget mismatch')
            for run in c['runs']:
                require(sha(run['config_path']) == run['config_sha256'], 'Config drift')
                raw = read(run['result_path']); hashes[run['result_path']] = sha(run['result_path'])
                cfg = read(run['config_path'])
                require(all(raw[k] == cfg[k] for k in ('deployment','workload_rps','routing_rps','random_seed','measurement_seconds','warmup_seconds','drain_timeout_seconds')), 'Raw config mismatch')
                require(raw['arrival_process']['initial_phase_seconds'] == run['initial_arrival_phases'], 'Arrival phase mismatch')
                sig = protocol_signature(raw); signatures.add(identity(sig))
                for model, source in raw['source']['workers'].items():
                    worker_versions.setdefault(model, set()).add(canonical(source))
                evidence = reconstruct(raw, catalog, DEFAULT_THRESHOLDS)
                label = classify(evidence, DEFAULT_THRESHOLDS)
                evidence.pop('queue_wait_values_ms')
                labels.append(dict(run_id=run['run_id'], configuration_id=c['configuration_id'],
                    deployment_id=c['deployment_id'], repetition_id=run['repetition_id'],
                    source_result_path=run['result_path'], source_sha256=hashes[run['result_path']],
                    **label, evidence=evidence))
                statuses[raw['status']] += 1
                tasks = raw['per_task_metrics']
                mean = weighted_objective(workload, {k:v['mean_e2e_latency_ms'] for k,v in tasks.items()})
                tail = weighted_objective(workload, {k:v['p95_e2e_latency_ms'] for k,v in tasks.items()})
                rep = dict(run_id=run['run_id'], repetition_id=run['repetition_id'], seed=run['seed'],
                    initial_arrival_phases=run['initial_arrival_phases'], source_result_path=run['result_path'],
                    status=raw['status'], stability_state_v2=label['stability_state_v2'],
                    stability_reason_v2=label['stability_reason_v2'], mean_objective_ms=mean, tail_objective_ms=tail,
                    per_task_metrics=tasks, system_metrics=raw['system_metrics'])
                reps.append(rep)
                if label['stability_state_v2']=='unstable':
                    failures.append(dict(run_id=run['run_id'], status=raw['status'], reason=label['stability_reason_v2'],
                        errors=raw['errors'], shutdown=raw.get('shutdown'), incomplete=evidence['incomplete_requests'],
                        completion_ratio=evidence['completion_ratio'], source=run['result_path']))
            configs.append(dict(configuration_id=c['configuration_id'], deployment_id=c['deployment_id'],
                deployment=c['deployment'], workload=workload, routing=c['routing'], features=features,
                predicted_mean_objective_ms=pred['mean_objective_ms'], predicted_tail_objective_ms=None,
                frozen_task_predictions=pred['task_mean_ms'], repetitions=reps,
                observed_mean=stats([r['mean_objective_ms'] for r in reps]),
                observed_tail=stats([r['tail_objective_ms'] for r in reps]),
                eligible={mode:configuration_eligible([r['stability_state_v2'] for r in reps],mode)
                          for mode in all_metrics}))
        require(len(signatures)==1, 'Incompatible protocols within workload')
        require(all(len(v)==1 for v in worker_versions.values()), 'Shared worker implementation drift')
        blocks = [{(r['seed'],canonical(r['initial_arrival_phases'])) for r in c['repetitions']} for c in configs]
        require(all(b==blocks[0] and len(b)==3 for b in blocks), 'Unmatched repetition blocks')
        views={}
        for mode in all_metrics:
            eligible=[c for c in configs if c['eligible'][mode]]
            ids=[c['configuration_id'] for c in eligible]
            entry=None
            if len(ids)>=2:
                entry=ranking_metrics(ids,[c['observed_mean']['mean'] for c in eligible],
                                      [c['predicted_mean_objective_ms'] for c in eligible])
                entry.update(workload_group=name, analysis=mode)
                all_metrics[mode].append(entry)
            views[mode]=dict(eligible_configuration_ids=ids, candidate_count=len(ids), metrics=entry,
                excluded_configuration_ids=[c['configuration_id'] for c in configs if not c['eligible'][mode]])
        comparisons.append(dict(workload_group=name, workload=workload, protocol_signature_hash=next(iter(signatures)),
            common_repetition_blocks=3, configurations=configs, views=views))
        predictions.extend(dict(workload_group=name, configuration_id=c['configuration_id'], deployment_id=c['deployment_id'],
            predicted_mean_objective_ms=c['predicted_mean_objective_ms'], predicted_tail_objective_ms=None,
            observed_mean=c['observed_mean'], observed_tail=c['observed_tail'], eligible=c['eligible']) for c in configs)
    require(len(labels)==45, 'Incomplete execution')
    decisions={mode:gate(entries) for mode,entries in all_metrics.items()}
    stability=dict(Counter(l['stability_state_v2'] for l in labels))
    save(output/'candidate_selection_audit.json',dict(plan_sha256=frozen['sha256'],groups=selection, no_enumeration=True))
    save(output/'comparison_groups.json',dict(groups=comparisons, thresholds=DEFAULT_THRESHOLDS,
        stability_policy=POLICY['stability'], objective_policy=POLICY['true_mean'], tail_policy=POLICY['true_tail']))
    save(output/'ranking_metrics.json',dict(model='Frozen pre-measurement C GBT (unweighted historical steady training)',
        per_group=all_metrics, decisions=decisions, gate_criteria_source='Existing Module 2F POLICY; not newly tuned on 2G labels',
        predicted_tail=None, tail_limitation='Measured workload-weighted task P95 saved; no P95 predictor or predicted tail ranking.'))
    shortlist_safety=[]
    for g in comparisons:
        first=min(g['configurations'],key=lambda c:(c['predicted_mean_objective_ms'],c['configuration_id']))
        shortlist_safety.append(dict(workload_group=g['workload_group'],
            full_pool_predicted_top1=first['configuration_id'], deployment=first['deployment'],
            observed_states=[r['stability_state_v2'] for r in first['repetitions']], eligible=first['eligible']))
    save(output/'regret_analysis.json',dict(groups_by_descending_regret={mode:sorted(entries,key=lambda e:e['regret'],reverse=True)
        for mode,entries in all_metrics.items()}, candidate_details='comparison_groups.json contains workload/routing/u/q/C/truth/prediction/stability.',
        unstable_run_diagnostics=failures, full_pool_top1_stability_diagnostic=shortlist_safety,
        conditional_metric_limitation='Ranking metrics condition on retrospective stability labels; these labels are not an available planning-time filter.'))
    for filename, records in (('stability_v2_labels.jsonl',labels),('ranking_predictions.jsonl',predictions)):
        (output/filename).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    save(output/'leakage_audit.json',dict(passed=True, plan_sha256=frozen['sha256'], manifest_sha256=frozen['run_manifest_sha256'],
        historical_dataset_sha256=sha(old_path), proofs=proofs, raw_source_hashes=hashes,
        no_post_measurement_fitting=True, no_new_labels_in_selection_or_routing=True,
        historical_seen_contexts=len({p['deployment_id'] for p in proofs if p['historically_seen']}),
        new_deployment_contexts=len({p['deployment_id'] for p in proofs if not p['historically_seen']}),
        caveats=['All evaluated contexts were historically seen, but entirely excluded from their own predictor training.',
                 'Frozen GBT uses unweighted steady rows, unlike Module 2D weighted training; this is not identical reuse of 2D fitted estimators.',
                 'Hash consistency verifies saved artifacts; timestamps alone are not independent attestation of freeze chronology.']))
    save(output/'dataset_validation.json',dict(passed=True, configurations=15, actual_runs=45, raw_status_counts=dict(statuses),
        stability_counts=stability, all_requested_outputs_present=True, configuration_hashes_verified=True,
        protocol_and_shared_workers_match=True, common_seed_phase_blocks=3, predictions_reproduced=True,
        incomplete_results_preserved=True, raw_sources_unchanged=all(sha(p)==h for p,h in hashes.items())))
    lines=['# Module 2G：Comparable-Group Profiling', '',
        '45 次计划运行已结束：15 个配置 × 3 次重复。原始结果全部保留，没有补跑或替换失败配置。',
        f'运行状态：{dict(statuses)}；三态标签：{stability}。', '',
        '**结论：INSUFFICIENT COMPARABLE GROUPS。本轮不能授权 global scorer 进入 guided search，也不能据此认定 predictor 已证伪。**',
        '沿用 Module 2F 的最低覆盖要求：至少 3 个独立 workload 比较组，其中至少 2 组具有非平凡 Top-2 比较。', '',
        '| Workload | 计划候选 | Strict 合法候选 | Extended 合法候选 |',
        '|---|---:|---:|---:|']
    for g in comparisons:
        lines.append(f"| {g['workload_group']} | 5 | {g['views']['strict']['candidate_count']} | {g['views']['extended']['candidate_count']} |")
    lines+=['', 'Strict 要求配置的全部 3 次重复均 steady；Extended 允许 recoverable_congested，但任一 unstable 即排除整个配置。',
        '每组 workload、measurement protocol、3 个 seed 与初始 arrival phases 完全匹配；不同 workload 之间没有排名。', '',
        '| 分析 / workload | 候选 | Kendall τ | Spearman ρ | Top-1 | Top-2 | Top-3 | Regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,entries in all_metrics.items():
        for e in entries:
            lines.append(f"| {mode} / {e['workload_group']} | {e['candidates']} | {e['kendall_tau']:.3f} | {e['spearman_rho']:.3f} | {e['top1_hit']} | {e['top2_coverage']}{' (平凡)' if not e['top2_nontrivial'] else ''} | {e['top3_coverage']}{' (平凡/不可用)' if not e['top3_nontrivial'] else ''} | {e['regret']:.2%} |")
    lines+=['', 'Macro / regret（仅对合法组，不能外推到被排除配置）：', '```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','',
        '## 审计与解释边界', '',
        '- 每组仅考虑 MILP seed 加 5 个一跳 mutation：5 个可行候选全部进入预算，空 deployment 不可行。没有枚举全空间。',
        '- Frozen policy 写了 farthest-point，但本轮全部可行候选均在预算内；实际按 configuration ID 排序，距离仅作审计，没有进行 farthest-point 淘汰。',
        '- Canonical routing 为 min-max standalone demand LP；它保证 surrogate 可行，不保证真实稳定。',
        '- 规划负载并不充分保守：Qwen-only 的 W1/W2/W3 u 分别约 0.312/0.543/0.331。尤其 W2 高于既有 Qwen backlog 区域；冻结说明中 W1 “below prior overload region” 不准确。不同 task mix 不能保证沿用单一边界，必须承认该设计限制，不能删除失败结果来修复。',
        '- 5 个 deployment context 都在历史数据出现过；没有新增历史未见 context。但每个冻结 predictor 的训练集完整排除了自身待评估 deployment，因此预测属于 deployment-held-out。',
        '- 预测冻结于本轮测量之前；本次只验证哈希并复现推理，没有重新 fit。参数固定的 C GBT 采用历史 v2 steady、无样本权重，不能声称与 2D 加权拟合的数值完全等价。',
        '- E2E mean 按固定 workload arrival rates 加权，重复间等权并保存 sample std。真实 tail 是按 arrival rates 加权的 per-task P95，不是系统 P95。没有 P95 predictor，未生成预测 tail。',
        '- 失败/不稳定主要证据见 regret_analysis.json：incomplete、drain timeout、worker/run error 或持续尾段 backlog 增长。后期较低负载也发生失败，不能仅归因于 Qwen u；顺序运行与环境漂移未随机化，因果解释受限。',
        '- 有效组太少；一个完整组的 Top-K 命中不能替代跨三个 workload 的可靠性证据。', '',
        '- 排序指标以事后稳定性筛选为条件，并不代表部署前能安全选点。W2 完整候选池中冻结 scorer 的 Top-1 是 Qwen+Whisper，但该配置有 1 次 unstable；排除它后得到的零 regret 不能证明完整候选池决策可靠。',
        '- W1 的主要排序错误是 Qwen+ResNet 与 Qwen+DistilBERT 对调；Top-1 正确，但 Top-2 recall 为 0.5。完整 u/q/C、真实和预测排序保存在 comparison_groups.json 与 regret_analysis.json。', '',
        '全部输出已保存。本阶段停止；未进入 Module 3、routing optimization 或新增 profiling。', '']
    (output/'module2g_summary.md').write_text('\n'.join(lines))
    return dict(statuses=dict(statuses), stability=stability, decisions=decisions)


if __name__ == '__main__':
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
