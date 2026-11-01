# 同域 DFS 与 CPU 效率阶段报告

本阶段只使用 CPU。所有基准中的服务时间、质量和整卡能耗均为合成 fixture，不构成新模型池的真实 GPU 性能证据。

## 已实现

`gsdeploy/optimizer.py` 的独立 DFS 现在支持六项评分、完整共驻画像和任务相关副本代理。
设置 `milp_capacity_witness=False` 时，DFS 独立枚举测得副本状态以及分母 `1..K` 的有理分流网格，
与正式 MILP 搜索使用相同声明域；历史条件容量见证仍可显式启用，但不宣称与 MILP 的附加见证相同。

非独立性能模式不使用单模型独立服务时间做容量必要条件或延迟下界。
缺失性能、质量或能耗的候选保留为未决候选及有效效用上界，不能当成不可行或缺失值为零。
完整候选使用相同 `predict_scenarios` 评分接口，避免两种算法每次完整评价成本不同。

DFS 提供状态预算和与 MILP 相同的墙钟预算。结果包含完整评价次数、状态数、首次完整可行解时间、
最佳解发现时间、停止原因、未决候选、上界和 gap。预算停止及未决证据不能产生全域最优证书。
支持排除完整部署向量、排除完整部署/路由方案和禁止任务分流，供强竞争方案重新优化。

加强后的 DFS 对已分配任务加入精确冻结质量成本；短轨迹（每任务最多 1024 个请求）计算任务单独运行时的 FIFO 排队下界。
删除其他任务不能增加等待，故该任务的平均延迟、P95 和窗口外请求数都能作为完整方案对应成本的下界。
吞吐损失按实际该任务窗口外请求数除以全轨迹请求数计算，质量和延迟继续按固定负载权重计算。
长轨迹保留缓存的服务多重集下界。未分配任务与缺失质量使用有效的零质量损失下界。

## 验证

运行以下 39 项同域回归，全部通过：

```bash
python -m unittest tests.test_static_dfs_v6 tests.test_search_contract tests.test_milp tests.test_static_milp_v6 -q
```

同时通过针对改动文件的 `git diff --check` 和 Python 编译检查。
新增覆盖包括纯网格无额外见证、六项精度改变最优模型、与独立 Cartesian oracle/MILP 一致、
非独立模式不会错误剪掉独立画像慢而完整画像快的方案、未知证据保留有效上界、限时和状态预算、
强竞争方案身份排除，以及 Poisson 实际请求数下单任务 FIFO 界不低估效用上界。

## 可复现效率基准

入口：

```bash
python -m gsdeploy.static_study_benchmark --output artifacts/static_joint_cpu_v6_20261021/benchmark_optimized --repeats 5 --time-limit 10
```

三个固定问题共 5、246、11308 个容量剪枝前完整方案，依次增加模型数、任务数、副本数和分流精度。
每种算法每个问题重复 5 次，每次新进程、新优化器、独立缓存，调用顺序交替。
墙钟包含轨迹构建、MILP 建模及评分，排除 fixture 生成及共同预加载的库导入；另存包含进程启动和导入的墙钟。
Linux 峰值 RSS 包括解释器、库和冻结证据，另存相对搜索前高水位的增长。
两种算法完整候选评价口径一致，状态/迭代预算足够大，主要共享 10 秒墙钟停止条件。

优化前和优化后两套结果分别保存在：

- `artifacts/static_joint_cpu_v6_20261021/benchmark/report.json`、`report.md`。
- `artifacts/static_joint_cpu_v6_20261021/benchmark_optimized/report.json`、`report.md`。

两套计时均首尾核验相关源码 SHA256，`code_unchanged_during_benchmark=True`。
此后集成审查继续修复了能耗身份与受限证书范围；计时结果属于各报告记录的源码版本，
不冒充最终集成源码的重新计时结果。重跑上述入口会生成当前版本的新记录。
原始结果包含全部运行、未证明状态、数值未解决状态、分数、上界、gap、配置文件和问题/代码哈希。
优化包括单线程 HiGHS，以及两算法独立实现的单任务 FIFO 下界；未更改三个基准问题以迎合结果。

优化后中位数：

| 问题 | 算法 | 总墙钟/秒 | 完整评价次数 | 5 次中完成最优证明 | 峰值 RSS/MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny | MILP | 0.025695 | 5 | 5 | 69.21 |
| tiny | DFS | 0.005535 | 5 | 5 | 64.25 |
| small | MILP | 10.005033 | 44 | 0 | 71.71 |
| small | DFS | 1.472925 | 241 | 5 | 65.46 |
| medium | MILP | 10.170594 | 35 | 0 | 93.60 |
| medium | DFS | 10.193062 | 1174 | 1 | 81.15 |

所有问题的 10 次求解观察到相同最佳效用；tiny 两算法都完成证明，small 的最优值由 DFS 完整证明支持。
medium 只有一次 DFS 完整证明，相同分数的其他限时运行仍保留自己的未决 gap，不能升级为已完成证明。
medium MILP 有 1/5 次 `MILP_NUMERICAL_UNRESOLVED`，其余为预算停止，均保留在报告中。

本实验支持 MILP 在 small/medium 中减少完整评价调用，但不支持其 CPU 总时间或证明速度优于 DFS。
small 完整评价减少约 5.5 倍，仍比 DFS 慢；medium 约减少 33.5 倍，但多数求解均未完成证明，不能作同证明工作量的加速比。
独立性能评分很便宜时，重复 MILP 求解的开销占主导，因此当前实现的效率优势尚有明确限制。

## 实际瓶颈

额外一次 medium MILP 的 cProfile（不计入五次基准）保存在：

- `artifacts/static_joint_cpu_v6_20261021/benchmark_optimized/medium_milp_profile.pstats`。
- 同目录 `medium_milp_profile.txt`。

该次求解的 36 次 HiGHS 调用累计 9.434 秒，占求解器 10.004 秒约 94%；
35 次完整 `predict_scenarios` 只花 0.202 秒；196 列生成花 0.273 秒；问题复制另花 0.389 秒。
优先改进方向是减少重复 master 求解开销及加强有效整卡能耗下界。
当前数据不支持通过减少完整评分次数推导总墙钟更快，亦不支持向 GPU 实测转移这些合成场景的性能结论。
