# C1-Lite CPU 实验与阶段结论

后续更新（2026-11-04）：用户指定完整目标算法，已在本轮组件上接入激活感知 BO/MADS 和可选 Joint 初值，见[新交付](activation_bilevel_implementation_20261104.md)。本文“Hybrid 未进入”及阶段决定记录的是 2026-11-03 当时状态；真实预测与收益资格尚未取得。

2026-11-03，依据[任务书 V2](c1_lite_hybrid_task_v2.md)。**交付了 C1-Lite 原型和资格准备；真实验证与正式优势尚未成立，Hybrid 未进入。** 本报告区分软件正确性、冻结 CPU 样例比较及真实资格。

## 软件验证

新增 22 项 C1-Lite 测试、5 项资格准备测试。与原连续路由、安全界、论文集成、v2 归一化/冻结、系统及 MILP 合计 **135 项，80.529 秒，全部通过**。日志见 [tests.log](../artifacts/c1-lite-v2-20261103/tests.log)。

覆盖副本键、总激活、池淘汰与 incumbent、独立有限域核对、非网格路由、新路由重新预测、追加求解、UNKNOWN 保留全域 UB、无可行值与无解区分、负成本下界/正 U、显存必要界、局部与全局预算、身份变化、非法界、扩池后旧种群刷新和确定性。资格测试确认旧字段不能造阶段/READY，小收益区间不能自动授予稳定优势。

## 同路由内层对照

固定同一 CPU fixture、同一组 6 条路由与预测身份，独立 session 运行三种内层；全部路由的认证 U 一致。数据见 [same-routes/comparison.json](../artifacts/c1-lite-v2-20261103/same-routes/comparison.json)。

| 内层 | 完整预测 | 叶子 | master | 外层评价/细化 | 总 wall 秒 |
|---|---:|---:|---:|---:|---:|
| 原固定路由 MILP | 8 | 24 | 14 | 6 | 1.0134 |
| 池 + exact-inner | 8 | 24 | 9 | 6 | 0.9786 |
| 池 + 限额后细化 | 8 | 24 | 9 | 12 | 1.7156 |

池减少了 5 次 master，没有减少完整预测。exact-inner 这次 wall 略低，渐进细化这次明显更慢；单次短运行和组件计时差异不足以证明稳定加速。本结果支持结构复用与证书一致性，不支持“master 少就一定端到端快”。

## 完整搜索工程对照

显式 CPU fixture：1 任务、2 模型、3 个非空部署、2 个请求，使用 v2 参考和分析构造的阶段规律。每次运行同样上限：4 秒、64 次完整预测、32 次外层评价；不同方法的提前退出/预算消耗方式不同。三个**优化种子**为 11/29/47，底层输入与到达轨迹相同；不是三个真实独立采集轮次，也不宣称为盲确认。

30 次运行的 [JSON](../artifacts/c1-lite-v2-20261103/reference-v2-cpu/comparison.json)、[CSV](../artifacts/c1-lite-v2-20261103/reference-v2-cpu/comparison.csv)、[图](../artifacts/c1-lite-v2-20261103/reference-v2-cpu/comparison.png)及[源码摘要](../artifacts/c1-lite-v2-20261103/reference-v2-cpu/source-manifest.json)已保存；摘要与交付代码一致。

| 方法 | U（种子 11 / 29 / 47） | 完整预测 | master | wall 中位秒 |
|---|---|---|---|---:|
| Joint K1 | -3.294667 / -3.294667 / -3.294667 | 4 / 4 / 4 | 5 / 5 / 5 | .0882 |
| Joint K2 | -3.294667 / -3.294667 / -3.294667 | 5 / 5 / 5 | 6 / 6 / 6 | .1905 |
| 原 C1 | -3.282986 / -3.288137 / -3.291692 | 34 / 34 / 34 | 66 / 66 / 66 | 1.4057 |
| C1-Lite exact | -3.294667 / -3.280012 / -3.280611 | 16 / 15 / 14 | 17 / 16 / 15 | .7933 |
| C1-Lite always | 同 exact | 16 / 15 / 14 | 17 / 16 / 15 | .8317 |
| C1-Lite gap | 同 exact | 16 / 15 / 14 | 17 / 16 / 15 | .8931 |
| C1-Lite budget | 同 exact | 16 / 14 / 13 | 8 / 7 / 8 | .7756 |
| C1-Lite pool=1 | 同 exact | 16 / 15 / 14 | 30 / 26 / 16 | .8017 |
| C1-Lite 无逐候选界 | 同 exact | 16 / 14 / 13 | 8 / 7 / 8 | .8045 |
| C3 | -3.294667 / -3.288137 / -3.294667 | 17 / 17 / 17 | 13 / 13 / 13 | .6815 |

这里 budget 结果随种子变化：11 的 U 低于原 C1，29/47 则更高，不能宣称统一保持解质量。原 C1 与新控制器初始化不同，这张表是完整方法工程对照，不能作池机制的单独因果证据。同路由对照承担内层可比核查。

always/gap/exact 在小域几乎重合；取消逐候选界也没有改变这里的预算次数，不能据此证明安全界无用。容量 1 增加 master，说明池覆盖有影响，但尚未做完整容量/多样性扫描。样例太小，不能支持 Hybrid 的正式进入条件。

收敛图只画种子 11，原始日志包含全部种子；C3 当前没有统一 convergence 转接，未画其曲线。`first_target_seconds=None` 在 C3 表示达标时间不可用，不能推断为未达到。其余达标时间只能解释为已记录事件中的首次达标；共同阈值是本次 Joint K1/K2 最好 U。

样例中的非网格 U 改善还受到两个请求的离散映射与 aggregate quality proxy 影响，不是稳定真实分流证据；这里不能复现或验证五模型 G2+image+OCR 的实际排序。

## Gate 与尚未交付部分

| Gate | 当前决定 |
|---|---|
| G0 数据与目标 | 待定：阶段/READY/参考及采集接线缺失 |
| G1 预测资格 | 未执行：没有新匹配真实训练与确认 |
| G2 场景资格 | 已给定向设计，实际容量/输入分层/内部收益待测 |
| G3 软件正确性 | 本报告列出的 CPU 反例与有限域核对通过；不外推真实资格 |
| G4 C1-Lite 价值 | 有工程减少 master 证据，端到端优势与解质量保障未成立 |
| G5 Hybrid 价值 | 未进入；不因 CPU 小例更快就增加 Hybrid |

阶段 A 已完成当前接口/旧记录审计和实验设计初稿，执行身份绑定与采集生产器仍未完成。B/C 不能在当前环境继续实测；D 原型已交付；E 只做了 CPU 工程预实验，不是正式比较。尚缺真实服务/等待/能耗图、六项误差及独立确认、完整指标拆分、同初始种群外层消融和有资格的规模扩展。

## 复现入口

以下输出目录必须不存在；复现不会覆写本次证据。前两条只接受显式 CPU fixture，第三条只做准备，不启动 GPU。

```bash
python experiments/benchmark_c1_lite.py --output artifacts/c1-lite-reproduction --seconds 4 --routes 32 --predictions 64 --seeds 11 29 47
python experiments/benchmark_c1_lite_routes.py --output artifacts/c1-lite-routes-reproduction
python experiments/prepare_c1_qualification.py --output artifacts/c1-qualification-reproduction
python -m unittest tests.test_c1_lite tests.test_c1_qualification tests.test_continuous_routing tests.test_continuous_bounds tests.test_continuous_validated tests.test_paper_integration tests.test_reference_normalization tests.test_reference_freeze tests.test_system tests.test_milp -v
```

实测包的后续接线及身份缺项见[预测资格](c1_lite_hybrid_prediction_qualification.md)，负载/输入/副本与预算设计见[场景包](c1_lite_hybrid_scenarios.md)。当前保持原目标、硬约束与默认入口，下一步需要补齐这些具体缺项再判断专用/连续路由的真实权衡。
