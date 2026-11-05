# J3O / USHER 与阶段 FIFO 的代码落实

本轮完成了**独立论文适配算法，以及阶段共驻/FIFO 预测进入主 MILP/DFS 的可运行接线**。验证限于小型 CPU 数学案例与受影响模块回归，没有启动 GPU、重新画像或大规模扫描。当前不宣称真实预测精度、真实六项效用或物理执行效率已经提高。

## 1. 这次真正进入代码的机制

| 来源 | 采用的机制 | 具体实现与主链路 |
| --- | --- | --- |
| J3O，PDF 第 6 页 Algorithm 1–2 | 固定部署求连续路由、部署与路由交替、贪心比较候选 | `gsdeploy/paper_algorithms.py` 的 `fixed_deployment_lp` / `J3OInspiredOptimizer`：Joint MILP 初始解 → 固定部署 LP 提案 → 连续质量转移与精确端点 → 单编辑部署邻域和一次路由前瞻 → 完整 FIFO 效用接受 |
| USHER，PDF 第 8–9 页 §3.3、Algorithm 1–2 | 配置需求、计算/显存互补分组、C/M-heavy 交替、组内优先与最小剩余空间放置 | `UsherResourcePlanner.groups/placement/enumerate_configurations`；只枚举用户提供的合法配置，有明确预算和未遍历组合数。`order_profiled_deployments` 将已有阶段 SM/DRAM 需求用于 J3O 候选排序，未知需求保留在域中 |
| InferLine，PDF 第 4 页 §4.2 | 在完整样本到达轨迹上按事件推进、显式保留排队 | 主 `predict` → 新 `paper_profiled_fifo_v1` → 既有 `FullHorizonModel`，复用同一冻结请求轨迹、输入序列和 Router，不再执行独立标量服务的旧队列回放 |
| 既有阶段共驻与生命周期实现 | 活动阶段变化时重算全部活动请求速度；区分 handler 完成、结果接收、许可释放和下一请求 READY | `ProfileProvider` + `LifecycleProvider` + `FullHorizonModel` 的组合进入主评分；显式整请求准入配置也进入同一 FIFO 引擎 |

J3O 的原问题是分层节点部署/卸载，并不直接提供任意 GPU 共驻减速公式。这里明确称为 **J3O-inspired 项目适配**。GSDeploy 的完整经验 P95、T、动态阶段干扰及 M/E 不是线性目标，因此 LP 只是生成路由；每个候选的最终 U 都重新调用同一个全过程预测器。

部署步骤采用有限合法单编辑邻域和路由前瞻，未复现 J3O 的分层贪心拉格朗日算法，也不继承其子模性、近似比或复杂度保证。已有最好有效方案始终保留。

USHER 的计算/显存装箱与阶段 SM/DRAM 排序是两个接口：前者的 M 是容量 MiB，后者的 DRAM 是画像需求。代码没有把带宽、显存容量、occupancy、MPS 百分比互相换算，也没有把装箱成功称为真实干扰消失。

## 2. 共同预测接口与搜索安全

新增 `gsdeploy/paper_forecast.py`。配置显式选择：

```json
{
  "performance_mode": "paper_profiled_fifo_v1",
  "memory_mode": "paper_profiled_fifo_v1"
}
```

同名配置块绑定输入池、独立情景名和每个支持部署的完整执行画像、生命周期、原 rho/g(n) 服务证据、启动能耗及可选准入配置。每个部署只能对应一个明确执行配置；另一个 BS、后端或资源控制配置应建立另一个实验身份。

当前入口支持 fresh、已排空初态的 batch1 整请求执行。batch2 及以上拒绝进入这个版本，避免把整批时间当成每条独立请求时间；任意 busy native 初态也没有被冒充为已接入。原高级 twoFIFO/kernel 分支的全部机制并未因此自动进入这个模式。

主 MILP/DFS 的搜索结构保留。新模式只增加以下必要边界：

1. 构造时持有同一不可变画像快照；候选不反复加载、解析和哈希完整画像。
2. 非可分的服务、容量和显存剪枝界归零，禁用依赖独立服务的 capacity witness；不沿用旧服务画像排除新模型候选。
3. 全部完整候选走新全过程评价，反馈仍按完整 identity 生效；原质量列与原六项效用算术复用。
4. 零流量模型仍在原 resident roster 中，其显存和资源上下文不被删除。改变部署必须选取另一个完整画像并重新预测。
5. 原 `rho<=0.8` 及 `g(n)` 定义保留，容量服务由对应部署显式提供。没有把短轨迹上观测到的平均排队时间替代稳定性容量定义。
6. 情景必须有明确的阶段/生命周期画像，缺少 1.25/2 情景返回 UNKNOWN；不会将 nominal law 任意乘系数冒充独立证据。robust 搜索将缺情景记为未知，不发出“无可行解”的错误证书。

缓存键包含执行/画像、部署、连续 QPS、轨迹、输入池/seed、效用权重、参考值、原容量表、时间窗口、drain、能耗参考及摊销。默认每会话最多 128 条、16 MiB **序列化内容预算**，全局最多两个内容键会话；缓存结果防御性复制。事件/请求预算超限保留完整 offered 分母并返回 UNKNOWN，不截短队列。

`eligible` 和有限域最优证书只描述当前数学预测模型。输出另带 `native_eligible=False`、`independent_accuracy_qualified=False`、`final_gain_eligible=False`。公开原生执行入口拒绝将该实验模式直接当成已经完成实测验证的模型。

历史 `calibrated_replay_v2`、`route_load_replay_v1` 占位状态和既有实验配置均未改成新模式。旧结果不能继承为新预测器的结果。

## 3. 最小运行入口

`configs/paper_fifo_cpu_fixture.json` 是显式标记的**手算软件样例**，不是新增 GPU 测量。包含一个任务、两个模型、三个部署、每轨迹两个请求，K=2 总共五个完整候选。

```bash
# 原 Joint MILP，现在真正调用阶段共驻/FIFO 评价器
python -m gsdeploy plan --config configs/paper_fifo_cpu_fixture.json

# 独立 J3O/USHER 适配算法，默认最多32个未缓存情景评价、10秒
python experiments/paper_methods.py --config configs/paper_fifo_cpu_fixture.json \
  --max-evaluations 24 --seconds 3 --seed-seconds 1 --rounds 1

# 三算法接线与手算请求映射演示，输出目录必须为新路径
python experiments/validate_paper_methods.py --output /tmp/paper-fifo-smoke-new
```

时间预算在求解和评价边界检查；单次事件模拟受事件/请求数限制，不宣称操作系统级强制总超时。Joint MILP 初始解的全部时间和未缓存情景调用都计入适配算法预算。少于三个情景查询的极小预算会跳过 seed，避免单次 seed 直接超出预算。

## 4. 小范围验证与结果

新增 `tests/test_paper_integration.py` 覆盖以下有意义的反例：

- 两请求同到达、不同模型共跑：G 在 160ms 完成 handler，165ms 被接收；S 在 G 完成后加速，180ms 完成 handler，190ms 被接收。不是独立服务时间的重放。
- S 两请求 FIFO：第一条 110ms 被接收，但下一条必须等到 120ms READY，第二条 230ms 被接收。
- 12 条同时未来到达的 burst 保留数字请求顺序，避免字符串 ID 将第 10 条排到第 2 条前面。
- 固定双模型部署将 G 比例降零后显存仍为 40 MiB；删除 G 并改用单 S 部署才是 20 MiB。这两个数字仅为手算样例。
- 显式整请求准入：G 的许可 87ms 释放后 S 才开始；没有把 85ms 的结果接收当成许可释放。
- 缺共跑状态、缺压力情景、缺容量证据、预算不足均保留未知；CPU 画像要显式 opt-in；batch2 不能偷走 batch1 路径。
- 将旧 standalone 服务/显存设为不适用的大值，并让旧标量解析器直接抛错，主 MILP/DFS 仍与独立五候选枚举一致，证明新引擎确实进入了主链路。
- 精确端点、非旧网格 LP 比例、初始解保留、原 rho 拒绝、缓存隔离与字节预算。

最终新增 **15 项**集成测试；连同受影响模块共 **103 项通过，5.714 秒**。最终三算法及 FIFO 演示共 **0.215 秒**，结果如下：

| 算法 | 样例 U | 实际事件引擎调用 | MILP master 求解 |
| --- | ---: | ---: | ---: |
| 主 Joint MILP + 新预测模式 | -3.015777778 | 5 | 6 |
| 主 DFS + 新预测模式 | -3.015777778 | 5 | 0 |
| J3O-inspired 适配 | -3.015777778 | 6 | 6 |

最终测试日志、演示完整结果和源文件摘要在 `artifacts/paper-mechanism-integration-20261102-final/`。未支持的压力情景查询单独计数，不冒充真正执行过的完整事件模拟。演示三算法效用一致；这个极小样例用于验证实现，不支持效率优势或真实效用提升结论。

## 5. 完成边界与下一步

已经交付的是可运行算法、共同全过程评分接口、主搜索接线、受限缓存和小范围正确性验证。现有 `KernelBaselineEstimator`、`fit_progress` 等训练实现继续保留；未发起一轮新的大拟合，也未把旧的失败拟合或 UNKNOWN 输出改成成功画像。

USHER 完整 OG-Merger、真实 MPS/SM 隔离、跨模型 cache/权重合并、批执行器及新域物理校准仍不属于本次已完成的交付。这里没有用一个互补系数替代这些真实执行机制。

真实数据层面仍须提供匹配部署/输入/阶段活动状态的有效画像，并用少量保留请求核对延迟与候选排序。现有不足或验证失败的数据不能通过改 schema 自动成为可用 law。后续首先选一个已支持部署、两个路由点和短轨迹检查实际误差；只有这个小检查有明确结果，才决定是否补采。此处没有安排全池扫描、重复长验证、GP-BO 或 MADS。

## 6. 后续实测数据小范围复核

已执行 `experiments/validate_paper_measured.py`：一个 G2+image+OCR 部署、两条既有 600 秒原生留出轨迹，共 346 条请求。4 次旧预测调用合计 2.599 秒；包含两次同轨迹交叉路由复算。原基线的 U 绝对误差为 0.009697 / 0.004604，逐请求输入、路由及 FIFO 数值均匹配。另 4 项数据核验测试通过。

**新预测器尚未完成独立实测精度验证，也未进入三算法对比。** 这 346 条记录的有序阶段和显式 READY 字段均为零；8 条训练条件共 1480 条 measurement 请求没有相同三模型部署。额外检查的两份较新细粒度训练审查绑定另一执行配置和非空前史，不能直接拼接。当前 GPU 查询仍为 Permission denied。

这还暴露了实现的适用性限制：精确输入/活动上下文查表需要匹配证据，尚未形成可转移到未见共跑组合的实测规律。主接线与 CPU 正确性成立，并不表示真实动态干扰模型的训练工作已经完成。

完整报告、原始对照、图和源文件摘要位于 [小范围实测复核报告](/root/autodl-tmp/generalist-specialist-project/artifacts/paper-measured-pilot-20261102/REPORT.md)。报告将 A/B 不同 seed 的条件排序与相同轨迹下的预测路由收益分别列出，未将二者混同。
