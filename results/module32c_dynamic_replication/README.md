# Module32C 产物索引

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
