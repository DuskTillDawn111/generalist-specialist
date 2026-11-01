# 已验证冻结代理 v1 的搜索结构界

适用范围是 `gsdeploy.calibrated_surrogate.v1` 的数学预测器。搜索证书只覆盖声明的有限副本/路由网格，不覆盖真实 GPU 上所有未测组合，也不覆盖连续路由。

## 共用门禁与局部缺测

`gsdeploy/surrogate_bounds.py` 直接调用校准模块的 `validated_spec(data, rates)`。只有 schema、版本哈希、留出验证状态、完整质量表哈希、模型基线身份、输入/质量协议身份、GPU/运行上下文和工作负载均匹配时，才传递下面的结构界。`VALIDATION_FAILED`、`FITTED_UNVALIDATED`、其他版本或被修改的上下文均不采用。

通过全局身份门禁后，各 `(model, task, n)` 分别读取正的任务副本服务画像。个别路径缺失时，仅该路径的服务和忙碌功耗项在松弛问题中取零；常数功耗、常驻副本功耗、已知路径和显存界仍生效。完整候选仍调用共享预测器；缺测不会被零值补成完整评价，更不会因此产生最优证书。

## 服务、排队、容量和能耗

记基线服务时间为 `b(m,t,n)` 毫秒，全局拟合倍率为 `a`。v1 中 `capacity_factor(m,n)=n`，因此 worker 的服务时间为 `a*b`，模型利用率为：

```
rho(m) = stress * sum_t(q(t,m) * a * b(m,t,n) / 1000) / n
```

已知路径的这一利用率直接用于任务列和 DFS 的必要容量条件。某一任务的路由/副本列确定后，删除其他任务，按相同 task router 在各 worker 上重放 FIFO，得到该任务平均延迟、95 分位延迟和超测量窗请求数的下界。长轨迹沿用无排队服务多重集界。质量损失由相同任务的冻结质量协议计算，已赋值任务的质量项精确。

整卡代理功耗不是逐模型功耗相加。它的非负参数可在搜索主问题中精确分解：

```
E = [c0 + c_busy * sum_(t,m)(q(t,m)*b(m,t,n)/1000)
        + (c_worker + startup_per_worker/A) * sum_m(n_m)] / P_ref
```

`A` 是显式启动摊销周期，未指定时为测量窗口。常数项按任务到达率占比分配给每个任务列；忙碌项属于任务列；副本及启动项属于模型状态。这样所有分量非负，且可直接进入既有 residual identity cut，无需重新引入整成本 big-M。

冻结显存代理为 `max(memory_multiplier*additive_safe_residency, known_deployment_peak)`，且倍率不小于 1。主问题采用第一项作为下界，并以它执行必要的显存预算剪枝。完整评价仍使用两项的最大值。

## 核验与复现

新增 `tests/test_calibrated_search_bounds_v6.py`，使用明确标注的合成 CPU 数据，检查：

- 任一完整候选及其所有部分路由分支的下界不超过完整预测成本；显存和能耗不重复计数。
- 常驻空闲副本、忙碌功耗、启动摊销均进入能耗，分解值等于完整预测器。
- 不同副本数的服务曲线和利用率与完整预测器一致。
- 六指标下，MILP、DFS、独立笛卡尔枚举在 nominal/robust、周期/泊松到达四种设置中得到相同最优效用。
- 专用拥塞时，搜索主动选择通用分流。
- 坏版本、坏校准状态、质量/上下文变更不提供正式结构界；局部缺测保留其他已知界，完整计划仍为未知。

相关 53 项测试通过：

```bash
python -m unittest tests.test_calibrated_search_bounds_v6 tests.test_calibrated_surrogate tests.test_static_dfs_v6 tests.test_profiled_memory_search_v6 tests.test_residual_milp_cut_v6 tests.test_milp tests.test_search_contract -q
```

`scripts/benchmark_calibrated_bounds_v6.py` 对照同一双任务、双模型、0/1/2 副本、K=3、完整六指标合成问题。关闭界的版本仅恢复旧的零服务/能耗界及未放大的 additive memory 下界，保留完全相同的完整预测器、路由域、目标与完整能耗 incumbent 处理。

主计时包含搜索器构造及求解，排除公共依赖导入和合成拟合；另存新进程总耗时。每个版本各运行五次、每次独立进程、交替顺序、固定 CPU 200–203，保存源码哈希和实际证书状态。最终报告在 `artifacts/static-study-gpu-20261022/structural-bound-comparison-final/report.json`；`benchmark_valid=true` 才能使用其结果。

```bash
taskset -c 200-203 python scripts/benchmark_calibrated_bounds_v6.py --output artifacts/static-study-gpu-20261022/structural-bound-comparison-final --repeats 5
```

这项对照用于确认结构界的内部改进。它不预先承诺 MILP 总耗时优于 DFS，也不把合成结果扩展为真实模型池的效率结论。

最终五次中位数如下。源码全程一致，`benchmark_valid=true`，20/20 次均取得相同有限域最优证书和 `U=-0.21169642857142867`。

| 算法与下界 | 搜索总耗时 | 完整评价次数 | HiGHS 调用次数 |
| --- | ---: | ---: | ---: |
| MILP，旧零服务/能耗界 | 4.599791 s | 104 | 105 |
| MILP，结构界 v1 | 0.012882 s | 1 | 2 |
| DFS，旧零服务/能耗界 | 0.260437 s | 104 | — |
| DFS，结构界 v1 | 0.008139 s | 1 | — |

该问题中结构界显著减少了两种算法的完整评价和搜索时间，MILP 的总耗时仍高于 DFS。因此这里只证明结构界带来的改进，不能据此声称 MILP 已获得相对 DFS 的效率优势。

## 用户要求的穷举 DFS 基线

`UtilityOptimizer(..., dfs_bound_pruning=False)` 现在提供真正不做效用剪枝的 DFS：不计算部分分支效用界，不按效用界排列部署或路由，仅保留合法显存和容量约束筛选；每个满足这些必要约束的完整身份均调用同一个完整预测器。新增独立笛卡尔枚举测试确认其评价次数覆盖全部这样的身份，且 `utility_bound_evaluations=0`、`bound_pruned=0`。相关测试累计 54 项通过。

使用同一合成配置、同一 K=3 网格、同一服务/容量/质量/六指标评分器，进行每种五次独立进程对照。最终报告为 `artifacts/static-study-gpu-20261022/plain-dfs-comparison/report.json`，源码全程一致，`benchmark_valid=true`。15/15 次均得到 `U=-0.21169642857142867` 和最优证书。

| 算法 | 搜索总耗时中位数，含构造 | 完整进程耗时中位数 | 完整评价次数 | 效用界计算次数 |
| --- | ---: | ---: | ---: | ---: |
| MILP，结构界 v1 | 0.013157 s | 0.415950 s | 1 | — |
| DFS，结构界 B&B | 0.008098 s | 0.393329 s | 1 | 58 |
| DFS，无效用剪枝穷举 | 0.206929 s | 0.573765 s | 86 | 0 |

搜索总耗时排除公共依赖导入和构造合成校准数据；完整进程耗时包含它们，两者均报告。穷举 DFS 保留了合法的容量筛选，因此只需评价 86 个身份；前一节关掉服务界的消融版本缺少对应的容量筛选，评价 104 次，二者不混为同一基线。

在这一合成问题中，MILP 比无效用剪枝穷举 DFS 的搜索总耗时快约 15.7 倍，完整 Python 进程耗时快约 1.38 倍；带界 DFS 仍比 MILP 快。结论仅限这个问题与本机实际测量，真实配置池仍需独立同域验证。

```bash
taskset -c 200-203 python scripts/benchmark_calibrated_bounds_v6.py --plain-comparison --output artifacts/static-study-gpu-20261022/plain-dfs-comparison --repeats 5
```

直接调用同域搜索的接口示例（模型表及其完整基线身份保持冻结）：

```python
dfs = UtilityOptimizer(
    c.catalog, c.workload, c.gpu_memory_mib, c.latency_reference_ms, c.weights,
    search_max_states=c.search_max_states, search_policy=c.search_policy,
    milp_capacity_witness=False, dfs_time_limit_seconds=c.milp_time_limit_seconds,
    dfs_bound_pruning=False, **DeploymentSystem(c).context(),
)
result = dfs.solve(replica_options=options, K=c.route_grid)
```
