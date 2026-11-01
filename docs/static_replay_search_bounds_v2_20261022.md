# per-input replay v2 的搜索下界

这是独立实验分支的数学支持，不是真实 v2 校准已通过的声明。只有显式 `calibrated_replay_v2` 性能/显存模式、`gsdeploy.calibrated_replay.v2` schema，以及公共 `validated_replay_spec(data, rates)` 校验通过时才启用；未拟合、未验证、验证失败或身份发生变化均不提供这些正式界。v1 保持原来的服务、容量与资源计算；冻结的 execution-source-v3 未修改。

## 三种服务统计分别使用

同一路径 `(model, task, n)` 有三种不同数值，不能相互替代：

| 用途 | 数值 | 校准 CPU 例子，G/text/n1 |
| --- | --- | ---: |
| 延迟/FIFO 数学下界 | 全部冻结输入的 median service 取最小值，再乘全局 ratio | 66 ms |
| 冻结预测器的 rho | 全输入池等权 median service 均值，再乘 ratio | 165 ms |
| 整卡忙碌功耗特征 | 原任务副本画像 standalone mean，不乘 ratio | 150 ms |

某个短请求实际只需要 66 ms，输入池均值 165 ms 会高估其无排队延迟，因此均值不构成延迟下界。测试明确构造了这个反例。

`SurrogateBounds.family_service()` 在 v2 中只返回最小值。MILP 的单任务 FIFO/长轨迹服务多重集界和 DFS 的任务服务界均使用它。逐请求真正服务时间不低于这个最小值，且删除其他任务不会延后 FIFO 完成时间，因此任务平均延迟、95 分位延迟和窗外完成数仍给出有效下界。

`SurrogateBounds.pressure_service()` 返回输入池均值。MILP 任务列的 pressure 和 DFS 的 capacity/demand 更新均使用它，与完整预测器的显式 pool-mean rho 合同一致。这里没有把池均值解释成真实 GPU 的稳定性保证。

## 资源界与缺失证据

E 保持原整卡非负功耗分解：常数项、原 standalone mean 构成的忙碌项、常驻 worker 项，以及启动能耗摊销项。它不会把 per-input minimum 或 scaled pool mean 代入忙碌特征。M 保持 `memory_multiplier*additive_safe_residency` 下界；完整预测器还会和该部署已见显存峰值取最大值。

某个路径缺任何冻结输入时，该路径的最小值和 pool-mean pressure 在主问题中都退化为零松弛，其他已知路径、常数/worker 能耗及显存界继续有效。完整评价仍由共享 v2 预测器执行，缺失输入保持 UNKNOWN，不能因此给出正式效用或最优证书。

## 核验

`tests/test_replay_search_bounds_v2.py` 直接使用 stage2 的 `fitted_replay_fixture()`：通过真实 v2 拟合逻辑处理显式合成 CPU 运行记录；没有真实 GPU 数据拟合。它检查：

- 输入最小值、池均值、功耗基线三者分离，以及均值不是下界的具体反例。
- 所有部分路由和完整任务列的成本界，在周期/泊松到达及 nominal/robust 下均不超过实际预测成本。
- MILP、带界 DFS、无效用剪枝穷举 DFS 与独立笛卡尔枚举得到相同六指标最优效用。
- 任务列和 DFS 的 rho 使用 pool mean；E 分解等于完整预测器，正忙碌系数时使用原 standalone mean。
- 采用实际项目的 `per_sample_replay` 质量模式时，逐输入质量与逐输入服务同时回放，MILP、两类 DFS 和独立枚举仍得到同一最优效用，所有部分分支界仍有效。
- 单任务样本超过 1024 时，长轨迹 minimum-service 多重集界仍安全。
- 局部输入缺失保留其他界；完整计划保持未知；失败/未验证/身份变化的 v2 不启用正式搜索界。

```bash
python -m unittest tests.test_replay_search_bounds_v2 tests.test_calibrated_replay_v2 tests.test_calibrated_search_bounds_v6 tests.test_calibrated_surrogate tests.test_static_dfs_v6 tests.test_profiled_memory_search_v6 tests.test_residual_milp_cut_v6 tests.test_milp tests.test_search_contract -q
```

这些检查只验证冻结数学预测器上的下界和有限域搜索，不能作为真实 v2 的留出验证结果，也不能证明真实硬件上的所有部署组合最优。

上述命令的相关 68 项 CPU 回归通过（25.5 秒），其中新增 v2 搜索界专项 9 项；Python 编译和差异空白检查通过。没有运行 GPU，没有真实拟合 v2，也未修改冻结 execution-source-v3。
