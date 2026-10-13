import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32b.common import *
gate=read(OUT/'module32b_gate.json');a=read(OUT/'final_data_acceptance.json');obs=read(OUT/'run_observations.json');b={o['run_id']:o for o in obs};checks={'final_audit_passed':a['passed'],'Gate_integrity_finalized':not gate['preliminary_integrity_pending'],'all_six_task_latency_rows':len(list(csv.DictReader((OUT/'gpu_per_task_latency.csv').open())))==18,'dynamic_real_G_matches_logs':b.get('A2',{}).get('G_completed',0)==sum(r['select_G']=='True' for r in csv.DictReader((OUT/'dynamic_decision_log.csv').open()))}
if len(obs)==3 and all(x['eligible'] for x in obs):checks['U_component_sums']=all(abs(o['U']-sum(o['components'][k+'_contribution'] for k in ['Q','T','L','P','M']))<1e-12 for o in obs)
if gate['gate']=='DYNAMIC_EXPLORATORY_POSITIVE':checks['strict_positive_gate']=gate['dynamic_cooperation_really_occurred'] and gate['dynamic_delta_U_vs_S_only']>.01 and gate['dynamic_delta_U_vs_fixed']>.01
save('final_artifact_review.json',dict(passed=all(checks.values()),checks=checks,at_unix_ns=time.time_ns()))
text='''# Module32B 数据索引

最终报告：module32b_summary.md；最终Gate：module32b_gate.json。Gate中preliminary_integrity_pending=false表示最终保护/因果重放审计已完成。中途Gate记录只表示未完成状态。

|阶段|文件|内容|
|---|---|---|
|A|vqa_interference_attribution.md / attribution_summary.json|Phi ASR/VQA串行阻塞、服务/排队/E2E与八项归因问题|
|A|per_task_utility_decomposition.csv|历史Fixed5/Fixed10的完整六任务mean/P95效用贡献及P95代数Shapley|
|A|phi_execution_overlap_analysis.csv / VQA_tail_requests.csv|VQA等待被ASR handler覆盖和尾请求|
|A|Phi_ASR_arrival_state.csv / Phi_ASR_state_clock_basis.csv|原fixed接纳时可观察状态，以及arrival/enqueue/send口径区别|
|A|cross_worker_service_overlap.csv / cross_worker_overlap_group_summary.csv|Whisper及各Specialist共置服务关联，非kernel因果|
|A|clock_and_cuda_timing_audit.json|Unix/perf时钟与原CUDA同步审计|
|A|all_historical_request_timings.csv / historical_path_probe_timings.csv|全部历史原始请求（含归档失败/预热/探测，重复来源标明）|
|A|all_historical_GPU_CPU_telemetry.csv / module32_complete_log_inventory.json|全部历史raw、journal、worker、GPU与文本日志读取清单|
|B|dynamic_opportunity_gate.json / historical_observable_opportunities.csv|机制A及有限可控机会，只读状态回放，无反事实U|
|C|vqa_aware_dynamic_router.py / routing_design.md|因果路由原型与非抢占限制、冻结风险模型|
|C|frozen_routing_manifest.json / frozen_routing_parameters.json|GPU前冻结的参数、模型池、六任务负载、副本、预算及Gate|
|C|CPU_router_acceptance.json / CPU_model_definition_revision.json|CPU安全/开销检查及测前风险模型定义修正|
|D|raw_runs / request_journals / formal_acceptance|最多三次600秒GPU原始cohort及durable journal|
|D|gpu_exploration_comparison.csv / gpu_per_task_latency.csv|系统U、合法ΔU、六任务mean/P95、吞吐与稳定性|
|D|gpu_worker_service_wait.csv / gpu_Phi_VQA_overlap.csv|实际Phi ASR/VQA服务、等待与执行区间|
|D|gpu_native_ASR_decomposition.csv|同输入S实际等待变化、G新增执行/等待与S共置服务变化|
|D|gpu_task_utility_decomposition.csv|当前三臂相对两基线的全任务L/P效用贡献|
|D|dynamic_decision_log.csv / dynamic_prediction_diagnostics.csv|全部在线决策、past状态、开销及预测误差诊断|
|D|gpu_backlog_timeseries.csv / gpu_resource_comparison.csv / gpu_sensor_samples.csv|各任务/worker队列、显存、温度、频率、功率、CPU与异常|
|D|cross_run_same_input_service_drift.json / .csv|历史与重测Fixed10同输入/路由的服务需求漂移审计|
|审计|final_data_acceptance.json / final_environment_identity.json|全权重SHA、环境版本、历史保护、同输入、GPU释放|
|审计|causal_router_replay_audit.json / final_artifact_review.json|逐决策复算、past时间与派生表交叉检查|
|证据限制|positive_evidence_limitations.md / positive_gain_sensitivity.json|Text尾变化主导正值；代数敏感性不重定义U|
|证据限制|within_trace_block_bootstrap_diagnostics.json|单轨迹分块重采样，仅诊断，非独立复现|
|冻结候选|frozen_exploratory_positive_candidate.json|冻结探索性候选、审计与参数引用，未部署|
|图|vqa_aware_exploration.png / .pdf|完整合法三臂科学图|
|索引|artifact_manifest.json|递归SHA与大小清单，排除清单自身|

全部Module32历史raw/报告/Gate及Module14F–31A、模型权重与正式优化器均保护。校准/回放使用历史同轨迹，正式三臂重新测量；这是探索而非独立验证。没有事后调参重跑、人工推理延迟或隐含副本。未执行独立长窗口和下一Module。
'''
(OUT/'README.md').write_text(text)
files=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','artifact_index.log']];save('artifact_manifest.json',dict(at_unix_ns=time.time_ns(),recursive=True,gate=gate['gate'],artifacts=files,history_preserved=gate['history_and_weights_unchanged'],independent_replication=False));print('INDEX',len(files),'REVIEW',checks)
