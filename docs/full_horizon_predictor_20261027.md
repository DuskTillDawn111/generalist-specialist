# 全过程到达与执行段共驻预测器

> 后续 v2 实施见 [执行计划与状态](predictor_v2_execution_plan_20261027.md)、[kernel 资源预测](kernel_resource_predictor_v2_20261027.md)、[真实 batch runtime](batch_owner_runtime_v2_20261027.md) 和 [事件契约](runtime_event_contract_v2_20261027.md)。下面保留第一版历史说明。v2 已新增显式 batch 执行域、串行准备事件、条件 handoff 和队列诊断；这些 CPU 实现不等于物理预测精度已通过。

2026-10-27。此次为 CPU 软件实现与验证，没有运行 GPU 画像、模型推理、MPS 或硬件探测。

资源研究的 `Dispatcher` 默认改用 `FullHorizonPolicy`。历史实验入口继续保留原预测器，旧实测结果和最优证书不转移到新预测域。显式 `legacy_observed_prefix=True` 仅用于历史兼容，且不能用于整请求准入配置。

## 预测链

`FutureWorkload → ProfileProvider → FullHorizonModel → score_forecast → FullHorizonPolicy`

- `gsdeploy/forecasting/workload.py`：使用所有任务的已声明到达分布，生成从当前时刻到测量结束的完整未来请求。支持 Poisson、保留相位的周期流、分段 Poisson 负载和显式已知规划时间表。输入仅含公开 pool_id 与权重，不接收答案、标签、输出长度或实际在线未来轨迹。在线样本使用独立 forecast namespace，不接实际实验 seed／RNG 状态。
- `gsdeploy/forecasting/profiles.py`：以执行上下文、模型、任务、输入身份和执行配置绑定画像；每个输入对应有序的 host／GPU／同步／回收段。GPU 段可细到 kernel，也可使用经过画像的阶段聚合。分别记录 SM、DRAM 和 L2 需求；计算／访存分类只用于候选分析，不将 occupancy 或 MPS 上限线性换算成吞吐。
- `gsdeploy/forecasting/simulator.py`：推进未来到达、FIFO、许可、内部执行段边界与完成事件。每次状态变化都重新定价全部活动请求。段切换是预测事件，不意味着执行端获得了暂停／抢占能力。只模拟已存在的完整请求准入动作。
- `gsdeploy/forecasting/metrics.py`：对相同已完成＋当前积压＋未来请求集合重算六项指标，P95 使用各任务完整端到端延迟集合；不对局部 P95 求和。板级能耗按时间积分一次，测量结束后的 drain 不增加 E。M 对齐当前原生 runner 的采样峰值口径，使用 `safe_factor * max(已观测历史峰值, 预测未来峰值)/显存预算`，不能将历史峰值当作可消去的加法常数。峰值指本次拥有的全部 worker 进程显存，包括空闲驻留者；不混入外部进程，也不把已乘安全系数的值重复乘一次。
- `tools/resource_control_20261024/forecast_policy.py`：每个当前路由分支共用相同未来到达和输入情景，按同一六项目标比较。缺失任务观测、质量、服务、功率或显存证据时，不给出完整评分，不把缺失 M 当作零差值。

后台路由保持原 SWRR；预测从其当前状态复制一个 continuation router。当前动作之后采用声明的继续路由策略，真实在线到达后重新求解。这是完整未来负载下的一步 rollout，不是递归执行未来实际在线策略的精确预言。默认有 3 个到达情景，与配置中的服务情景交叉；mean／p90 不被称为置信界。

Q 使用冻结的模型／任务总体损失代理；OCR CER、ASR WER 可大于 1，沿用既有质量定义，不截断或增加绝对精度门槛。同一模型内部并发的 kernel 须先整理成不重复计时的执行段，不能直接把重叠 kernel 的耗时相加。

到达数、总请求数、事件数与原在线积压预算均有确定性上限。超预算返回 UNKNOWN／专用回退，不截断负载或改成更短预测窗口。全过程仿真的实际在线开销仍需后续测量；当前不作实时性或物理净收益承诺。

在线 audit 只保留当前请求活动期间的详细竞争轨迹，并另报完整仿真事件数，避免把每个候选的完整未来轨迹都写入日志。这只减少诊断日志，全部未来请求仍参与仿真和评分；独立 forecast 命令默认保留完整轨迹。

## USHER、Roomie 的借鉴范围

USHER 对应配置相关资源需求和计算／访存互补分析；Roomie 对应有序 kernel／阶段的重叠关系。相同活动模型数量下，不同执行段、输入和开始位置可以产生不同速度。没有直接照抄存在口径争议的 Roomie 公式。

共驻速度由完整活动执行段上下文的证据支持，不从 pairwise 结果直接外推三路共驻。`fit_progress` 用完整段的变化重叠区间拟合进度：

```
sum_context(duration_in_context_ms * progress_context / solo_segment_ms) = 1
```

拟合要求训练 split、匹配的执行上下文、唯一完整观测、足够样本和满列秩；进度非正或工作量重构误差过大时不接受。不可辨识的共驻状态保留 UNKNOWN，不靠正则项填成已知。拟合诊断只表示训练重构误差，不表示独立预测精度。

功率、显存和非 mean 场景需要各自证据，不能由进度拟合自动产生。`assess_prediction_pairs` 独立报告留出集上 ΔU 的误差、偏差和正收益判别正确率；不使用其标签在线选路，也不将经验误差分位数冒充置信界。

分 batch、公共权重子矩阵合并和 cache 复用不能只在画像中写一个折扣。画像保留 batch／backend／weight_reuse 配置身份，但当前独立请求执行器遇到联合 batch 或合并权重配置会明确返回 `JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED`。同一请求内部已有的 chunk／kernel 序列可以按真实画像建段。实际联合配批、OG-Merger、KV 管理和阶段调度仍需新的执行后端及其匹配预测模型。

## 调用与运行状态

CPU 独立入口：

```bash
python -m gsdeploy.forecasting forecast --input REQUEST.json --output NEW_RESULT.json
python -m gsdeploy.forecasting fit --input TRAINING.json --output NEW_PROFILES.json
python -m gsdeploy.forecasting assess --input VALIDATION_PAIRS.json --output NEW_ASSESSMENT.json
```

输出拒绝覆盖。CPU 构造画像必须标记 `provenance.kind=cpu_fixture`，并在 forecast 命令显式传 `--allow-cpu-fixture`；默认不能冒充物理画像。示例请求与结果见 `artifacts/predictor-cpu-20261027/`，全部为分析用 CPU 构造数值。

资源策略接收：

```python
snapshot = {
    "text#0": {"outstanding": [...]},
    # 其余所有驻留 alias，包括空闲后台 worker
    "_forecast_state": {
        "admission": {"active": {"request-id": "text#0"}, "waiting": [...]},
        "observed_peak_memory_mib": ...,  # 包括应计入评分的先前阶段
        "observed_board_energy_joules": ...,
    },
}
```

`admission` 仅在整请求准入模式使用；有既存积压时必须提供实际许可／sender FIFO 快照。运行时和仿真共用 `admission_available` 容量规则；初始 FIFO 保留实测顺序，预测中的同时 sender 就绪采用明确的 alias 排序。线程实际唤醒／提交抖动没有被宣称为精确预测。

已有运行头在新段模型中需要当前 `forecast_cursor`：`profile_id, segment_index, remaining_fraction, observed_at_ns, evidence`。不能用提交多久来猜已经完成了多少 GPU 工作，也不能在每个决策点重置成长为完整请求。缺少进度或准入观测时返回 UNKNOWN。显式接入旧服务提供者时，只能得到其原整 handler 条件残余代理，不生成虚构 kernel 信息。

历史能耗未提供时，共同缺失的加法项可在路由比较中消去，但绝对 U 保留未知。显存历史峰值未提供时不能如此处理。单个场景的 U 输出在 `predicted_U_by_scenario`，不把第一情景冒充全部未来的唯一结果。

独立 forecast 请求必须显式填写 `memory_safe_factor`；资源 Dispatcher 从有效 catalog 读取。传入的历史能耗须已经符合既有 startup 摊销口径，缺少这些共同历史项时不输出绝对 U。

有界 drain 内未完成的请求仍计入 offered cohort 和 T；完整 L/P 未知，不以截断延迟或无穷 drain 的完成时刻补齐。原 rho≤0.8 等稳定性准入规则仍由既有外层执行；本预测器未给出新的稳定性证明，也未生成新的 DFS／物理最优证书。

## 本次验证与后续数据需求

`tests/test_full_horizon_forecast.py` 使用可手算的 CPU 场景验证未来请求改变现有完成时间、阶段错位、全队列外部性反转分流选择、sender FIFO 队首阻塞、条件进度、测量／drain 边界、历史峰值、完整 P95、拟合可辨识性及缺失证据回退。既有资源与队列测试另行回归。

最终回归 172 项：171 通过，1 项因原可选音频数据依赖缺失跳过；其中新增预测器测试 41 项全部通过。编译、空白检查和独立 CLI 的分析场景精确复现通过。记录见 `artifacts/predictor-cpu-20261027/cpu-validation.json` 与 `cpu-regression.log`。

目前没有新的物理 kernel／阶段数据，因此没有声称新预测器已经在真实 GPU 上更准确。后续需要匹配执行配置的独立及共驻段画像、输入分布、全板功率／峰显存，以及与训练分开的 ΔU／排序验证。未经这些数据，不解锁原生运行或正收益结论。

相关源码已加入资源实验包的 source identity。旧冻结包因源码变化应校验失败；保留旧产物，后续需要时在新路径准备新包。
