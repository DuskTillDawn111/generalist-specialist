# 研究工具导航

这里保留按课题、日期冻结的采集、拟合、分析和独立复核实现。
现有模块路径参与导入和源码凭据，整理时保留路径与各历史版本。
基础安装只分发 `gsdeploy/`；从源码 checkout 的仓库根目录调用这些工具。

| 目录或文件组 | 研究范围 |
| --- | --- |
| `seven_pool_baseline_20261022/` | 七模型池基线 |
| `confirmation_statistics_audit_20261022/` | 独立确认的统计复核 |
| `spillover_background_20261022/`、`spillover_context_20261022/` | 后台负载与上下文 |
| `spillover_capacity_20261022/`、`spillover_discovery_20261022/` | 容量、搜索与发现 |
| `spillover_gpu_discovery_20261022/`、`spillover_confirmation_20261022/` | GPU 发现与独立确认 |
| `text_load_screen_20261022/` | 文本负载筛选 |
| `queue_aware_20261023/`、`queue_coupled_utility_20261023/` | 队列与耦合效用 |
| `queue_five_term_20261023/`、`queue_refined_20261023/`、`queue_refined_calibration_20261023/` | 历史五项队列研究及校准 |
| 顶层 `queue_*.py`、`coupled_*.py` | 对应队列研究的报告、复核与诊断 |
| 顶层 `gpu_contention_*.py`、`unified_weight_*.py` | GPU 干扰及共享权重分析 |
| `resource_control_20261024/` | 资源约束、路由策略与 CPU 实验包 |
| `predictor_validation_20261027/` | 全过程预测、kernel DAG、生命周期、twoFIFO、资源模型与前瞻验证 |

`predictor_validation_20261027/` 内继续保留原文件名：`kernel_*` 对应 kernel/DAG，
`twoFIFO_*` 对应请求队列/匹配画像/前瞻验证，其他模块以模型或协议命名。
同名版本链可能包含失败候选和历史反例，版本数字不能单独证明可执行资格。

查看接口可用 `python -m tools.<directory>.<module> --help`（模块需提供 CLI）；
具体调用及所需证据见[文档导航](../docs/README.md)。原生采集和拟合的参数、
冻结来源及输出独占规则由各协议约束。测试入口见[测试说明](../tests/README.md)。
