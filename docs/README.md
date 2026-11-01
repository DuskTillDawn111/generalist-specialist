# 文档导航

## 系统与维护

- [项目结构与 Git 边界](project_structure.md)
- [系统架构](architecture.md)、[运行时与原生后端](runtime.md)
- [效用规范](utility_specification.md)、[联合 MILP](joint_milp.md)
- [历史与版本边界](history.md)、[源码来源](source_provenance.json)

## 四任务与静态研究

- [四任务协议](compact_protocol.md)、[CPU 准备](cpu_preparation.md)、[GPU 实验](gpu_experiments.md)
- [静态工作流 v6](static_study_workflow_v6.md)
- [静态研究方法与结果](static_study_methods_and_results_20261022.md)
- [MILP/DFS 同域 CPU 基准](static_joint_dfs_benchmark_report_20261021.md)
- [J3O 风格静态部署建模](j3o_style_static_deployment_model_20261026.md)

## 资源与 spillover

- [资源分区与 spillover](gpu_resource_partition_and_spillover_20261024.md)
- [资源控制 CPU 实现](resource_control_cpu_implementation_20261024.md)
- [文献指导的 spillover 算法](literature_guided_spillover_algorithm_20261024.md)
- [USHER 方法与项目适配](usher_methods_and_project_adaptation_20261024.md)

## 全过程预测与 kernel / twoFIFO

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
