import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32c.common import *
g=read(OUT/'module32c_gate.json');a=read(OUT/'final_data_acceptance.json');obs=read(OUT/'run_observations.json');pairs=list(csv.DictReader((OUT/'gpu_independent_pairs.csv').open()));tasks=list(csv.DictReader((OUT/'per_task_utility_decomposition.csv').open()));dec=list(csv.DictReader((OUT/'dynamic_decision_log.csv').open()));checks=dict(final_integrity_passed=a.get('integrity_passed',a['passed']),all_measured_pairs_matched=bool(pairs) and all(p['matched_inputs_arrivals']=='True' for p in pairs),six_task_contrasts_per_measured_pair=len(tasks)==6*len(pairs),original_router_unchanged=a['checks']['original_router_byte_identical'],original_parameters_unchanged=a['checks']['original_parameters_byte_identical'],history_preserved=a['history_preserved'],final_gate_not_pending=not g['integrity_audit_pending'])
if (OUT/'planned_run_disposition.json').exists():
 slots=read(OUT/'planned_run_disposition.json');checks['six_planned_slots_explicitly_accounted']=len(slots)==6
 checks['no_fabricated_unexecuted_raw']=all(not (OUT/'raw_runs'/(s['run_id']+'.json')).exists() for s in slots if s['status']=='NOT_EXECUTED')
 checks['invalid_dynamic_U_and_delta_NULL']=all(p['dynamic_U']=='' and p['delta_U']=='' for p in pairs if p['both_legal_stable']=='False')
checks['actual_G_matches_decisions']=all(o.get('G_completed',0)==sum(d['run_id']==o['run_id'] and d['select_G']=='True' for d in dec) for o in obs if o['arm']=='dynamic')
checks['U_component_sums']=all(o['U'] is None or abs(o['U']-sum(o['components'][k+'_contribution'] for k in ['Q','T','L','P','M']))<1e-12 for o in obs)
if g['gate']=='DYNAMIC_REPLICATION_CONFIRMED':checks['strict_replication_success_gate']=g['all_three_delta_above_001'] and g['mean_CI95_above_zero'] and g['real_G_and_S_each_dynamic'] and g['all_legal_stable']
save('final_artifact_review.json',dict(passed=all(checks.values()),checks=checks))
(OUT/'README.md').write_text('''# Module32C 产物索引

最终报告：module32c_summary.md；Gate：module32c_gate.json；原32B与所有历史Gate保持不变。

|类别|文件|内容|
|---|---|---|
|CPU归因|utility_attribution.md / text_tail_audit.md|原完整六任务U复算、Text分位支撑请求、四次触发、时钟与机制限制|
|CPU请求|text_tail_requests.csv / text_quantile_support.csv|原Text全部前10%尾样本和P95插值支撑|
|CPU效用|historical_per_task_utility_decomposition.csv / historical_utility_sensitivity.json|官方六任务U及明确区分的代数敏感性|
|CPU完整日志|historical_all_requests.csv / historical_all_GPU_telemetry.csv / historical_log_inventory.json|完整原raw、journal、worker、GPU与文本日志清单|
|CPU Gate|stage_A_gate.json / CPU_duration_adapter_acceptance.json|匹配合法性、原计量与2400s分母等价验证|
|冻结|experiment_manifest.json / frozen_routing_manifest.json|原源码阈值参数与六模型、QPS、预算、质量、顺序和新轨迹SHA|
|冻结原型|vqa_aware_dynamic_router.py / frozen_routing_parameters.json|与32B按字节相同；未修改正式优化器|
|独立轨迹|trajectories|三组预先冻结新seed，同组相同到达/输入；同样本池而非新数据集|
|GPU完整原始|raw_runs / request_journals / formal_acceptance / run_attempts|六次2400秒正式运行、30秒预热与完整有界排空|
|GPU配对|gpu_independent_runs.csv / gpu_independent_pairs.csv|六臂U、每组合法ΔU、实际G/S、稳定性、吞吐与环境差|
|GPU六任务|gpu_per_task_latency.csv / per_task_utility_decomposition.csv|全任务mean/P95及效用贡献|
|延迟分解|paired_request_latency_decomposition.csv / Text_VQA_service_queue_decomposition.csv|同输入等待、handler服务、E2E与host残差|
|尾部归因|gpu_text_VQA_tail_requests.csv / gpu_text_VQA_quantile_support.csv / Text_VQA_p95_component_attribution.csv|新轨迹尾请求与代数P95分解，非执行counterfactual|
|共置关联|Text_VQA_Phi_overlap_association.csv / routing_dose_utility_attribution.csv|实测Phi重叠关联，不宣称GPU因果识别|
|Whisper|gpu_worker_service_wait.csv / Whisper_paired_queue_service.csv|全部worker服务/等待、保留S同输入排队收益及G执行代价|
|动态接纳|dynamic_decision_log.csv / dynamic_Phi_ASR_trigger_execution.csv|实时状态、实际G路由、事后未来VQA暴露和开销|
|VQA|gpu_VQA_execution_overlap.csv|实际ASR覆盖VQA等待和Phi串行执行验证|
|稳定性|gpu_backlog_timeseries.csv / stability_environment_report.md|所有任务/worker/system积压与原判据|
|GPU环境|gpu_sensor_samples.csv / gpu_resource_comparison.csv / gpu_clock_audit.csv|显存温度时钟功率CPU、Unix/perf计时口径|
|统计|pair_level_statistics.json / utility_sensitivity_diagnostic.csv|预定n=3组级Student t设计；本轮无有效三组差值，未计算区间|
|停止诊断|long_window_failure_diagnosis.md / long_window_router_failure.json|原1ms开销验收失败，历史扫描增长，未更改原策略|
|完整计划|planned_run_disposition.json|六个slot实际测量/未执行状态；未伪造后两组|
|CPU诊断|router_overhead_by_300s_block.csv / completed_history_projection_CPU_profile.csv|实测wrapper开销增长与独立CPU画像，非GPU重跑|
|非法数值诊断|ineligible_full_cohort_arithmetic_diagnostic.json / ineligible_task_arithmetic_diagnostic.csv|全cohort原公式代数，非合法U/ΔU，不用于Gate|
|最终保护|final_data_acceptance.json / final_environment_identity.json / causal_router_replay_audit.json|全权重SHA、全部历史保护、环境、因果重放与GPU释放|
|图|independent_replication.png / .pdf|独立配对结果与六任务贡献的科学图|
|索引|final_artifact_review.json / artifact_manifest.json|交叉审查与递归SHA/大小清单|

不按结果换seed、不调策略、不删请求或修改效用。本轮在P1_D触发原冻结技术停止规则，实际2/6次；后两组NOT_EXECUTED，不能报告三组均值或置信区间。正式Dynamic U/ΔU=NULL；最终完整性审查不表示实验验收通过。不自动比较Fixed10、不改正式部署优化器、不启动下一阶段。
''')
files=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','artifact_index.log']]
save('artifact_manifest.json',dict(at_unix_ns=time.time_ns(),gate=g['gate'],artifacts=files,recursive=True,history_preserved=a['history_preserved'],independent_replication=True,optimizer_modified=False));print('INDEX',len(files),'REVIEW',checks)
