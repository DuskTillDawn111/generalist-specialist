# 可识别的共驻响应模型与新训练画像设计

状态：这是被后续设计替代的探索草稿。整组 exact-cohort 经验轨迹只用于诊断/残差校验，不能作为主分流预测器。主线与实现合同见 `composable_stage_response_design_20261028.md`；请勿据本文推进完整 cohort 记忆模型。

本文是 CPU 科学设计，不是已拟合模型、采集授权包或预测资格报告。读取已有训练诊断和新观察器源码；不读取独立验证请求的结果，不改变冻结画像依赖，不启动 GPU。

## 1. 已有证据决定了什么

v57 的 N/T 与 Gamma-Poisson 算术审计通过，但整次 capture 留出时，标称 95% 区间只覆盖 45/71 个可评分状态单元。原固定 age 分箱后的组合诊断只覆盖 41/71。原分箱降低了部分计数偏差，但不能把 age 单独升级成合格模型。失败可能包含状态不足、异步发射、时钟投影、route 差异和依赖过程；这些诊断不能唯一归因于某一个因素。

native clock 诊断中，spill_low 三个进程的全局等速 additive-offset 交集为空。不能把 median offset 当成物理时间真值。CUDA_EVENT 的 3218 条时间戳在每次旧 capture 中都是 0；事件 record/wait 的身份可识别，不等于完成时刻可识别。

新的 DeviceProgressObserver 只覆盖三路语言模型的 caller current stream。一次成功 query 确认的是该 stream 中被记录 event 之前的工作已经完成，不是全请求、全部 stream、剩余工作比例或 SM 服务。OCR 没有这个探针。事件记录、查询和 stdout 都有开销，natural 应解释为“没有 nsys 的同仪表执行”，不能称为原始无仪表速度。

RTX4090 的硬件 SM/DRAM/L2 指标当前为 ERR_NVGPUCTRPERM。无需等待权限就可以估计有明确定义域的自然响应分布；不能因此声称识别了算力占用、带宽或 L2 命中率。

## 2. 最先能真正实现的模型：整组条件响应分布

第一版选 exact-domain empirical joint response provider，先不拟合潜在 SM 服务率。它的一个训练样本是一次完整共跑 job 或一次完整路由 capture 的联合轨迹，不是一百万个相互独立 kernel。训练单位明确保留各 owner 的相关减速、host 发射、同步、响应运输和 READY 尾部。

定义静态键 K 为：硬件/驱动/执行源与部署身份、四个实际 resident、输入内容与生成策略身份、task、batch、已声明准入方式、每个请求公开 arrival/dispatch 计划与 offset。四 resident 驻留与其中哪些活动是两个字段。为解释 length/shape，记录 BEGIN 才知道的真实形状；不能把最终生成长度放进 K。

同一 K 的自然训练 capture 给出联合轨迹 Z_r：实际 dispatch、handler START、可见 forward BEGIN/END、query receipt、RESULT reader/dequeue/process、WORKER_READY，以及采样的板级功率与各 owner 显存。训练响应标签允许使用训练轨迹的未来结果；运行时筛选条件只能用已经可用的前缀。初版输出联合样本、支持数和区间，不仅输出一个平均 slowdown。

运行时 t 的条件剩余模型为经验分布：从 K 相同、与当前已处理的 milestone 前缀一致的训练轨迹中，对剩余的联合 handler/response/READY 时刻取样。未完成事件须满足训练轨迹中相应终点晚于当前 elapsed；已确认完成事件须满足其逻辑顺序和声明的时间区间。若到达时间容差或时钟不确定性参与匹配，必须在看结果前定义为测量误差区间，不能为了保住粒子扩大容差。样本没有兼容前缀、支持只剩单次、或者请求/计划不在 exact K 内，返回 UNKNOWN 与原因，不做邻近回退。

该模型可以直接实现为一个 CPU manifest+trajectory reader，接口为 `(K, processed_prefix, cutoff) -> joint_suffix_samples, support, evidence_receipts`。它首先是 `CoScheduleResponseProvider`，不是可以任意拼接到 kernel simulator 的独立 request service 表。零个或一个兼容 whole capture 无法估计变异，只有明确的描述性标签，不能发出预测资格。

### 不重复计算排队

整组 provider 若以公开 arrival 为起点，已经包含该执行策略的 host 排队与准入等待，不能再让另一个队列引擎重复添加。若接入通用队列引擎，合同必须改为“给定实际 dispatch/当前并行状态到 handler/READY 的联合响应”，并让 host 队列只负责 arrival→dispatch。新 dispatch 改变并行计划时，只有 exact 的新联合计划有训练支持才可继续；不能把旧计划的独立 slowdown 拼过去。

第一阶段可预测当前 26 个封闭 job 的重复执行。预测任意全过程到达，必须采集完整路由 capture，K 包含完整公开到达与候选 dispatch 策略；至少覆盖已有 balanced/skewed/spill_low/spill_high 类完整候选。未来到达率已知可以用于队列规划，但未来设备完成、生成长度和未来 CUDA 发射不是已知输入。26 个短 job 本身不能证明全程 P95、路由排序或 ΔU。

### 当前如何使用 phase/issued 特征

可立即使用真实 parent collector processing cutoff 之前的 forward BEGIN shape/index、可见 END、query outcome 与等待时间。每个特征同时保存 worker 观测时刻、reader enqueue、collector dequeue、collector processing；未处理通知不能提前进入在线状态。

native kernel/issued API 关联可用于 kernel-mode 的训练解释和缺失性报告。当前并没有已审计的逐 launch 在线 notification producer，所以不能把 retrospective CUPTI start/end 或 API return 当作 natural 在线特征。以后若实现 prefix producer，再按 process/context/stream/launch identity 保留已发射计数与已确认完成计数；有未识别依赖时，`issued-minus-confirmed` 只表示未确认队列，不是 GPU-ready 工作。静态 phase binding 可以给实际已发射节点附标签，不能把未来 template 节点提前发射。

## 3. 第二个可拟合对象：区间删失的 milestone 响应

有足够重复后，可以将整组轨迹模型细化为可观测 milestone 的 multi-state / semi-Markov 响应模型。状态包括 owner/forward phase/index/BEGIN shape、已经发射并确认的同步状态和已处理 query、当前请求并行集合及各字段 known mask；时间变量是该响应状态的 attained age。模型预测下一次可观测 milestone 的等待分布与类别，继而预测 response/READY，仍不声称是 SM 服务量。

对于已 record 的 stream-end event，第一次 successful query 的 host end 是完成上界。若此前 query 返回 not-ready，保守下界只能取此前 query 的 begin（查询中途可能完成）；若无此前 false，使用该 event record call 的 host begin 作为保守下界。不能用 first-success 时间作精确完成点，也不能在 false 时设置剩余比例。缺少后续 true 的样本是右删失；多 stream 只覆盖被记录 stream，其他分支仍 UNKNOWN。事件可早于通知到达完成；模型的 runtime observable milestone 与潜在 device completion 必须使用不同字段。

训练可在预先固定的状态域内用区间删失生存 likelihood / Turnbull 分布估计 response milestone，或对存在完整 parent receipt 标签的可见 milestone 直接估计经验条件生存。不能用 kernel elapsed 加总拟造 SM 工作。状态太稀疏时不自动加经验保底；增加设计内重复，或提交一个新的、只由训练数据开发并重新冻结的分层模型。

跨状态 hazard 必须积分整个时间变动路径；特征改变不重置原事件的 attained age。训练阶段可以读取将来的标签来构造 likelihood，预测阶段必须联合模拟将来的 phase/host 发射路径，不能条件在验证轨迹实际发生的未来 phase 上。第一版整轨迹抽样正是保留这类依赖的最小实现。

## 4. 物理 context 单独建模，只作诊断

保留物理 SAVE/RESTORE 的原四 resident mark 域，包括 inactive owner 和 same-holder。另设诊断用 marked semi-Markov 模型：持有者 h、距真实 restore 的 age、已可观测的 phase/issued/sync 状态，给出退出等待和 restore mark 的联合条件分布。SAVE 与下一次 RESTORE 间的 gap 单独保留；边界、延迟进入、删失与无法分辨的同 holder reacquisition 显式记录。

无训练 exposure 的状态为 UNKNOWN；有 exposure 没事件是删失证据，不是 rate=0。phase 变化不能重置 restore age。same-holder 不能删除、renormalize 或假定已经成功复原。任一 inactive owner 得到 physical context，不能给另一个请求分配 reference work。这个诊断模型不参与“holder==SM 服务开关”，也不提供默认 SM 时间换算。若它仍不能通过独立 count/mark/survival 检验，报告失败，不阻塞有支持的自然整组 response 估计。

## 5. 最少采多少：固定下界，不能承诺统计充分

当前 pilot 的 26 个 job 是 18 个 context×length cell 加 8 个 SG offset/order cell，每 mode 每 cell 仅一次。这是可运行性和可解释性诊断，不足以估计 capture 之间的变异。

建议的初始设计下界如下。它是待 CPU manifest 实现的计划，不修改或扩大当前冻结的 26-job 计划。

| 阶段 | 完整 capture 数 | 每 capture job 数 | natural+kernel job 总数 | 用途 |
| --- | ---: | ---: | ---: | --- |
| 训练重复 | 3/mode | 26 | 156 | 同一 exact K 的初始变异与完整轨迹支持 |
| 独立 known-input | 2/mode | 26 | 104 | 冻结后的同域重复预测检验 |
| 独立 unseen-input | 2/mode | 26 | 104 | 记录泛化能力与 UNKNOWN；不能回流拟合 |

若当前训练 pilot 可通过来源/实际执行/仪表审计，可以计入训练三次中的一次，只新增两次。自然/kernel 必须执行相同公开 jobs；它们是不同执行，不能把 natural 的 query 与 kernel 的物理 switch 拼成同一次完成。模式顺序应在采集前平衡声明，记录热状态/顺序，以免把时间漂移当 profiler 效应。整次训练 capture 的顺序种子、进程生命周期与输入均有 receipt。没有 declared execution identity 的热启动复用不能算重复。

三次训练是估计变化的起步数量，不是三次足够识别分布的保证。两个 holdout capture 只能暴露大错误，不能可靠验证 P95 或达到置信覆盖。以独立 whole capture 为抽样单位报告误差与区间；kernel 数量不能提高独立样本数。样本精度不足应报告 INCONCLUSIVE，按提前定义的 precision/power 计划增加新 capture，不能看过 holdout 的方向后调模型。

现有 exact K 模型对 unseen input 应 UNKNOWN，这是正确的 coverage 结果，而非预测成功。如果要在 unseen input 上发出预测，另立只用训练 input 开发的 shape-conditioned 模型和独立 gate。全路由模型同样先采至少三个完整训练 capture/候选/执行策略；背景 OCR、G2 图像/ASR、mixed length、非 SG 启动差、持续流量等没有进入当前 26-cell 支持的情况均不能被继承为已覆盖。

三 owner shared-text batch 是新的部署，不继承四 owner 分布。共同 batch 的同输入成对请求、wait deadline、batch size、padding、启动 M/E 和内存/host 成本均进入新 K。每种 batch 配置独立重复、独立验证。

## 6. 无计数器时仍如何落实 USHER / Roomie 思路

USHER 的资源互补先成为可证伪的候选生成假设：用单路及配对自然响应、实际 shape/batch 和可静态估计的算术/访存结构提出 compute-/memory-sensitive 组合。当前没有 hardware counter，因此标签必须写成 proxy/UNKNOWN，不能宣称已经测到“显存带宽型”或“计算型”。比较配对执行与相同输入单路基线，报告总完成时间、各 owner 减速、板级能量与 owner 显存采样峰值。是否互补由独立自然测量判断，不能预先给互补候选打固定奖励。

Roomie 的 phase/sequence/启动偏移思路先通过已声明 offset/order、实际 forward phase 和整请求准入成为可执行条件。默认共享与整请求准入各自有 response 分布；准入等待与 host 控制成本计入。资源份额只有硬件与实际执行验证后才成为新条件。共同 batch、公共权重子矩阵和 cache reuse 必须有真实后端及严格共享身份；不同量化/adapter/base 的两个模型不能凭概念相近就认定公共矩阵。当前未实现的跨模型 reuse 不进入收益预测。

## 7. 独立资格门槛与交付

训练内部允许按 whole capture 交叉验证做模型开发。冻结前写入完整输入/源/配置 receipts、支持域、interval/clock/availability 合同、所有误差容限与 precision 规则。独立验证完成后不得把失败 cell 调参修成 PASS；新版本要重新冻结并使用新 holdout。

资格门槛按以下顺序执行：

1. 来源、四 resident/实际活动、计划 job、实际 offset/输入、READY/handler/response、telemetry owner 及 single-stream scope 全部可追溯；原始/解释/可用时钟分离，缺失进入分母。
2. 自然 response 与 profiler-perturbed response 分别检验。以 whole capture 比较联合完成时间、per-owner 减速、milestone 生存/区间删失校准；报告支持数、删失、unknown、区间覆盖和宽度。若需要迁移 profiler 特征，必须在训练中建立并独立检验迁移；没有迁移就各自限定解释范围。
3. 物理 switch 单独检验退出生存、age-rescaling、count coverage 与 raw mark log score/calibration，保留 idle/self 类；计算实现 PASS 不等于科学充分。
4. 扩展到全过程完整 capture 后，用完整已知到达比较 arrival→dispatch、dispatch→handler、handler→response/READY、完整 cohort P95、功率积分与采样显存峰值。整组 response 如果包含队列不得再加一次。无法估计 prefix energy 时六项效用仍 UNKNOWN。
5. 使用项目原六项效用与稳定性规则，在冻结候选集上检验 ΔU 正负、成对排序、选择 regret、UNKNOWN 覆盖率。容限、最小可识别 ΔU 与选择规则必须预先固定；真值区间跨 0 应作为不确定，不能硬判正收益。
6. 只有上述模型在支持域可靠后，进行原/优化执行×禁/允许分流四组同到达同输入对照。核心比较是优化执行内部的允许分流与最优禁分流。候选搜索、禁分流最优性与同域 DFS 放 CPU；不能用未覆盖部署的经验 response 宣称全域最优。

第一批 deliverable 是严格区分 scope 的 natural/kernel 训练画像、source/clock/availability/support 表、支持域内独立误差报告，以及明确未通过或仍 UNKNOWN 的部分。本文没有宣告预测器合格、没有最终收益结论，也没有提出固定经验 slowdown/SM multiplier。
