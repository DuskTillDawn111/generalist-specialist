"""Summarize the frozen comparison after the bounded campaign finishes."""
from .common import *

def main():
 selection=load(OUT/'model_selection.json');metrics=load(OUT/'pair_validation_metrics.json');comparison=load(OUT/'paired_bootstrap_model_comparison.json');fit=load(OUT/'fitted_model.json');overhead=load(OUT/'runtime_overhead.json');robust=load(OUT/'uncertainty_sensitivity.json');validation=load(OUT/'validation_results.json')
 assert validation['status']=='PASS' and validation['9A_complete']
 selected=selection['selected_model'];features=load(OUT/'path_features.json')
 pressure=['| co-path | delta (ms / target kernel) | process-refit 95% CI |','|---|---:|---|']
 for q,d in fit['delta_path_ms_per_kernel'].items():pressure.append(f"| {q} | {d:.9g} | {robust['delta_path_process_refit_95CI'][q]} |")
 table=['| Model | MAE (slowdown) | RMSE | Spearman | Kendall | latency MAE (ms) |','|---|---:|---:|---:|---:|---:|']
 for variant,m in metrics.items():table.append(f"| {variant} | {m['MAE']:.6g} | {m['RMSE']:.6g} | {m['Spearman']} | {m['Kendall']} | {m['latency_MAE_ms']:.6g} |")
 lines=[f'# Module 9A 第一版完成：选择 {selected}', '',
  f'保留全部 {len(features["paths"])} 个 admitted task-path 的 offline profiling；Roomie spatial occupancy predictor 已退出当前路线。未启用 MPS，未修改 native worker/runtime、v2、RC-SG 或冻结结果。', '',
  '模型为 `T_hat[p <- q] = T_p + n_p * delta_q`。`delta_q` 属于 co-resident model-task path；对多个 calibration targets 的归一化额外延迟取中位数，没有 target fitted tax、pair-specific coefficient 或 target susceptibility。', '',
  '6 个 calibration identities、6 个 validation identities，各 3 次独立进程重复；8 个 matched standalone paths 各 3 次。每个 co-path 的 calibration/validation target paths 完全分离。所有 repetitions 留在同一个 split；旧的混合 co-task 测量不作为纯 co-path 数据复用。', '',
  '## 参数与验证', '',*pressure,'',*table,'',
  f'冻结规则选中 **{selected}**；eligible models = {selection["eligible_models"]}。所有更复杂模型必须相对每个更简单模型具有严格负的 paired error-difference 95% CI，才能参与最低 MAE 选择。没有百分比 improvement threshold，没有结果出来后的调参、追加 identities 或特征。', '']
 for name,r in comparison.items():lines.append(f'- {name}: mean paired MAE difference = {r["mean_paired_MAE_difference"]:.6g}, 95% CI = {r["95CI"]}.')
 lines += ['', '## 范围与限制', '',
  f'I0 同时具有最低整体 MAE 和 RMSE；I1/I2 相对 I0 的 paired CI 均跨零，I2 相对 I1 的 CI 也跨零。结论是未获得整体迁移改善支持，不是证明所有 co-pressure 恒为零。I2 的 Generalist-target MAE 为 {metrics["I2"]["generalist"]["MAE"]:.6g}（I0 为 {metrics["I0"]["generalist"]["MAE"]:.6g}），Specialist-target MAE 为 {metrics["I2"]["specialist"]["MAE"]:.6g}（I0 为 {metrics["I0"]["specialist"]["MAE"]:.6g}）。Whisper-turbo ASR/LID 在同一 Qwen VQA co-path 下的 ordering 被 I1/I2 反向预测。不得用 subgroup 的较低误差宣称整体模型成功。', '',
  '统计比较按 pair identity bootstrap，条件于已拟合的 calibration。多个 identities 共享 target/co-path，不能声称独立 co-model population 的显著性。`uncertainty_sensitivity.json` 另报进程重复/重拟合与 target、co-path cluster 检查；这些审计不重新选择模型。', '',
  '第一版只校准 3/21 co-paths（2 个 co-models），验证 4 个独立于 calibration 的 target paths。21/21 profiling coverage 不代表 21/21 co-pressure calibration coverage。未校准 coefficient 使用明确标记的 zero fallback，不 hard-prune candidate。I1 对未测 co-task 的 model-level extrapolation 也不等于该 co-task 已被验证。', '',
  'delta 是观察到的 handler-latency residual proxy，可能混合 CPU、时钟和 temporal scheduling 影响；不是因果分离的 context-switch duration，不假定固定时间片、FCFS、round-robin 或 kernel alternation。每个 n_p 只来自三个 offline samples。', '',
  'routing evaluator 使用 baseline-demand-weighted co-task mixture、co-model busy proxy 和不同 co-model 的 additive contributions。Own replicas 仅由 empirical g 表示。busy probability、3+ models 的 additive aggregation 和多 replica co-pressure 尚无 GPU validation；本轮不追加 campaign。', '',
  f'CPU pair median = {overhead["pair_median_ms"]:.6g} ms；21-path/11-model evaluate median = {overhead["full_21_path_11_model_median_ms"]:.6g} ms，p95 = {overhead["full_p95_ms"]:.6g} ms。驻留 estimator memory = {overhead["resident_estimator_tracemalloc_bytes"]} bytes。每次评价均不启动 GPU/profiler。', '',
  '借用 iGniter 的 kernel-count × extra-scheduling-delay decomposition（Sec. III-A Eq. 5–6），没有复现其 MPS spatial provisioning。co-path pressure 与 routing-conditioned aggregation 是项目 adaptation/extension。来源：[iGniter](https://arxiv.org/html/2211.01713v1)。', '',
  '9A 在模型选择后结束，不继续增加特征。按最新用户指示随后进入固定 deployment/workload 的 9B bounded routing 阶段；不进入 9C，不把 CPU surrogate 改善声称为 serving/E2E 改善。', '']
 (OUT/'module9a_summary.md').write_text('\n'.join(lines))
 raw=[load(p) for p in (OUT/'raw_runs').glob('*.json')]
 save('run_accounting.json',{'planned':60,'attempts':len(raw),'successful':sum(v['status']=='COMPLETE' for v in raw),'failed':sum(v['status']=='FAILED' for v in raw),'new_profiling_runs':0,'standalone_runs':24,'calibration_runs':18,'validation_runs':18,'multi_model_runs':0,'retries':0,'outcome_based_additions':0})
 save('interference_model_gate.json',{'scope':'FIRST_LIGHTWEIGHT_CO_PATH_PRESSURE_TRANSFER_VERSION','selected_model':selected,'primary_conditional_statistical_eligibility':selection['eligible_models'],'first_version_9A_complete':True,'original_Roomie_spatial_route_retired':True,'full_multi_model_routing_accuracy_SUPPORTED_claim':False,'multi_model_GPU_aggregation_validated':False,'status':'FINITE_DOMAIN_COMPARISON_COMPLETE_AGGREGATION_UNVALIDATED','9B_CPU_routing_authorized_by_latest_user_instruction':True,'old_frozen_gate_files_modified':False,'no_feature_expansion':True})
 save('artifact_hashes.json',{str(p.relative_to(OUT)):{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
 print(json.dumps({'9A':selected,'report':str(OUT/'module9a_summary.md'),'status':'COMPLETE'}))

if __name__=='__main__':main()
