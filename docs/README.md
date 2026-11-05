# 文档导航

当前效用入口：[六项归一化 v2 正式切换](utility_normalization_switch_20261103.md)；真实参考测量待冻结。

## 系统与维护

- [项目结构与 Git 边界](project_structure.md)
- [系统架构](architecture.md)、[运行时与原生后端](runtime.md)
- [效用规范](utility_specification.md)、[联合 MILP](joint_milp.md)
- [六项归一化文献依据与共同 1 秒参考](utility_normalization_literature_20261101.md)
- [有文献依据的六项归一化重设计](utility_normalization_redesign_20261101.md)
- [历史与版本边界](history.md)、[源码来源](source_provenance.json)

## 四任务与静态研究

- [四任务协议](compact_protocol.md)、[CPU 准备](cpu_preparation.md)、[GPU 实验](gpu_experiments.md)
- [静态工作流 v6](static_study_workflow_v6.md)
- [静态研究方法与结果](static_study_methods_and_results_20261022.md)
- [MILP/DFS 同域 CPU 基准](static_joint_dfs_benchmark_report_20261021.md)
- [J3O 风格静态部署建模](j3o_style_static_deployment_model_20261026.md)

## 连续路由与离散部署优化

- [当前目标算法：激活感知 BO–MADS 双层连续优化](activation_bilevel_target_20261104.md)
- [当前实现：BO/MADS、渐进 MILP、旧算法对照与复现](activation_bilevel_implementation_20261104.md)
- [C1-Lite CPU 实施、同路由核对与资格阻塞报告](c1_lite_hybrid_experiments.md)
- [C1-Lite 实现设计与证书范围](c1_lite_hybrid_design.md)、[预测资格](c1_lite_hybrid_prediction_qualification.md)、[定向场景包](c1_lite_hybrid_scenarios.md)
- [任务书 V2：预测与场景资格、C1-Lite 和 Hybrid C1](c1_lite_hybrid_task_v2.md)
- [最新：论文适配、共享 FIFO 界与结构化部署搜索的 CPU 阶段交付](paper_structured_completion_20261103.md)

- [下一阶段计划书 V3.0（原文）](continuous_routing_plan_v3.md)
- [Phase 0 源码与接口审计](continuous_routing_audit.md)
- [下一目标与 Phase 1 实现设计](continuous_routing_design.md)
- [Phase 1 接口与复现](continuous_routing_usage.md)
- [Phase 1 实际结果与阶段判断](continuous_routing_phase1.md)
- [内层安全界、候选热启动与消融结果](continuous_routing_inner_improvement.md)
- [Phase 1 性能分解、同域对照与合法扩展诊断](continuous_routing_phase1_diagnostics.md)
- [Phase 1 已验证回放界、不可变缓存与受控消融](continuous_routing_validated_improvement.md)
- [Joint MILP 扩展性、固定部署连续精修与独立部署邻域](joint_scalability_and_refinement.md)
- [G/S 连续路由收益、共驻解释边界与 FIFO 请求映射诊断](routing_benefit_diagnosis.md)
- [多候选部署与连续路由精修：排名反转、成本及阶段决定](multi_candidate_continuous_refinement.md)
- [候选连续域安全界与自适应路由预算分配](adaptive_candidate_refinement.md)
- [优化问题难度、完整排名与一次预测认证诊断](gsdeploy_optimization_difficulty_diagnosis.md)
- [路由与负载相关 GPU 共驻干扰审计及覆盖清单](route_load_interference_audit.md)

Phase 1 已新增连续域/激活、固定路由枚举、固定路由 MILP、DE 与独立对比入口。
原优化器保留，预测器按现有接口适配，真实支持域与合成验证分开记录。

## 资源与 spillover

- [资源分区与 spillover](gpu_resource_partition_and_spillover_20261024.md)
- [资源控制 CPU 实现](resource_control_cpu_implementation_20261024.md)
- [文献指导的 spillover 算法](literature_guided_spillover_algorithm_20261024.md)
- [USHER 方法与项目适配](usher_methods_and_project_adaptation_20261024.md)

## 全过程预测与 kernel / twoFIFO

- [论文训练链路、受限上下文泛化与 MILP 结构优化补齐](paper_milp_completion_20261102.md)
- [J3O / USHER 机制落实、主 MILP/DFS 接线与小范围验证](paper_mechanism_integration_20261102.md)
- [当前 Joint＋连续域重评与暂缓实测事项](joint_refinement_current_status.md)
- [全过程预测器](full_horizon_predictor_20261027.md)
- [可组合阶段响应设计](composable_stage_response_design_20261028.md)
- [画像恢复与执行记录](predictor_recovery_execution_20261029.md)
- [kernel DAG 映射审计](kernel_dag_training_mapping_audit_v1_20261029.md)
- [kernel DAG 重复采集生产器](kernel_dag_response_replication_producer_v1_20261029.md)
- [生产全过程预测接口契约](production_whole_forecast_provider_contract_v1_20261029.md)

研究记录保留当次时间、前提和来源。后续段落可能更新前文状态；
CPU 测试通过不等于真实精度、收益或完整最优性已取得资格。
`artifacts/` 链接指向未随 Git 分发的本地证据。

其他设计、版本和独审材料见[完整文档索引](index.md)，按日期分组、组内按文件名排序。
