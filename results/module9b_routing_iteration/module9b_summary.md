# Module 9B 固定 deployment routing 原型

9A 选择 I0。9B 不重新拟合、增加或选择 predictor features；production 使用该固定模型。

实现 bounded routing：冻结当前 service coefficients，LP 最小化 peak normalized effective utilization，再以 weighted mean service time 打破 tie；以 damping 0.5 更新候选，重新调用纯 evaluator，只有 surrogate 改善才保留。最多 3 iterations、6 秒预算；无改善、LP timeout/infeasible surrogate 或 numerical cycle 即停止，保留最佳 witness。没有全局最优或固定点收敛保证。

验证了 24 个 8F-2 development states（12 个冻结 cells × v2/RC-SG 各自的固定 deployment）。只更新 routing；workload、memory budget、quality admission、replica bounds、empirical g 和 deployment 都保持原值。4 个 routing 发生变化，4 个 peak surrogate utilization 降低。所有 conservation/support/memory/immutability 与 best-witness 检查通过。

另测已存在 I2 ablation coefficient 下的 nonlinear evaluator control flow、zero workload、memory/unsupported/conservation guards，以及 surrogate infeasible 的 witness 保留；未改变 production selection。

I0 是 no-interference baseline，因此这些 CPU routing 结果不构成 task-conditioned interference modeling 成功或实际 serving/E2E 改善证据。其余模型的迁移预测未获得 9A 的统计支持。9A 的 3+ models additive/busy-probability extrapolation 也未被 GPU 验证。

新增 GPU runs = 0；未启用 MPS，未修改 worker/runtime、v2、RC-SG 或冻结结果。没有进入 9C candidate ranking。
