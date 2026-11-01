# native busy、HOL、恢复风险与 rolling 的最小主算法接入契约

2026-10-29。本文只读审查现有实现，给出下一版本的 CPU 实施路径；没有修改任何已有源码、冻结文件、manifest 或 report，没有启动 GPU、拟合或等待器。下文的新模块名称均为建议，尚未创建。目标是把已有队列模型接到真正的候选评估与搜索叶节点，同时保留无法从 native 观测识别的状态。

## 1. 现状：哪些已经有，哪些还没有接通

| 能力 | 现有代码与准确入口 | 当前边界与缺口 |
| --- | --- | --- |
| 全过程未来到达 | [workload.py:24](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/workload.py:24) 的 `FutureWorkload`，`:92` 的 declared schedule，`:108` 的 `suffix`；[forecasting/__main__.py:60](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/__main__.py:60) 实际把 suffix 传给 `FullHorizonModel` | 已能使用全部预声明日历，Poisson、gamma renewal 和时变率场景。不能把事后获知的真实未来到达冒充预声明输入。当前181请求已经有完整公共日历，不需要先出现积压才预测。 |
| 剩余工作与资源工作量 | [queue_diagnostics.py:18](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/queue_diagnostics.py:18) `remaining_work`；`:117` 的 active work 消耗与 HOL 统计 | 已实现已知 segment cursor 的剩余 solo-profile 毫秒以及 SM/DRAM/L2 需求加权工作量。它们不是物理 SM 时间，也不是共驻后的剩余墙钟时间。该 cursor 尚不能从当前 native 快照直接得到。 |
| 因果观察者 | [observed_state.py:80](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/observed_state.py:80) 的 runtime adapter，`:283` 的不可回退状态推进 | 保存事实而非预测完成时间；旧 adapter 的可见性以 reader receipt 为准。真正 twoFIFO 主续接必须采用更严格的 collector processing cutoff，不能直接把旧 feed 当成 collector 已见前缀。 |
| native busy 集合状态 | [twoFIFO_native_busy_state_v1.py:39](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_native_busy_state_v1.py:39) `_processed`；`:51` `from_snapshot`；`:244` `forecast` | 已按 collector processing 过滤，保留完整公共日历和可能状态，防止历史前缀被改写。`:237` 明确 `continuation_state=None`、busy numeric 不获准；只有可核验 settled 边界可调用 strict v5_2。 |
| strict 双 FIFO 与完整续接 | [actual_worker_twoFIFO_reference_v5_2.py:21](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/actual_worker_twoFIFO_reference_v5_2.py:21) `STATE_FIELDS`；`:66` checkpoint 检查；`:166` `simulate` | 已有精确有理数 CPU continuation，包括 active age、sender/worker FIFO、reader/collector stations 和通知在途。它是完整显式 latent world 的 checkpoint，尚不是从 native busy 快照反演出的状态。`:632` 的 global admission cap 是 `None`。 |
| busy 多世界传播 | [twoFIFO_busy_candidate_continuation_v1.py:29](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_busy_candidate_continuation_v1.py:29) `_screen`；`:67` `evaluate` | 已能从各自完整初态生成到 cutoff，再续跑各自 checkpoint，保留不兼容 slot 的 UNKNOWN 质量。`:138` 明确 START notice transport/full prefix likelihood 未识别，`:140` 不保证有限支持完备；不能直接称 native posterior。 |
| generic rolling | [rolling_visible_queue_v1.py:113](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/rolling_visible_queue_v1.py:113) `remaining_state`；`:179` `forecast_suffix` | 可以实际调用 generic segment engine，但要求当前直接进度或 waiting 证据；`:187` 拒绝不兼容主 lifecycle，`:189` 限制 shared、unbatched；`:221` 因缺 prefix energy 主动使完整六项 U 为 UNKNOWN。不能宣称已完成主算法 rolling。 |
| native stream 查询 | [device_progress_observer_v1.py:64](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/device_progress_observer_v1.py:64) `poll`；`:103` forward hooks；[visible_device_event_ledger_v1.py:42](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/visible_device_event_ledger_v1.py:42) `visible_feeds` | 已有无新增同步的 CUDA event/query 观测，不应说“完全没有 device 观测”。但只覆盖记录的 current stream；失败 query 无分数，成功 query 不证明其他 stream 或完整请求完成。旧 ledger 仍是 reader cutoff，不能未经升级进入 collector cutoff 主路径。 |
| 恢复风险 | [recovery_risk.py:34](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/recovery_risk.py:34) 特征、`:122` 训练、`:182` frozen provider、`:230` holdout；[native_queue_risk_inference_v9.py:17](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/native_queue_risk_inference_v9.py:17) sidecar | 已有 CPU 树模型和严格特征/上下文/训练域验证，且能作为诊断附件调用；`:225` 明确未校准概率，holdout 不会校准。尚没有把风险校准成主算法可解释概率的完整三分数据路径。 |
| 原始主搜索 | [optimizer.py:404](/root/autodl-tmp/generalist-specialist-project/gsdeploy/optimizer.py:404) 搜索叶调用 `predict_scenarios`；[prediction.py:101](/root/autodl-tmp/generalist-specialist-project/gsdeploy/prediction.py:101) 静态逐 alias FIFO | 核心仍是 service resolver 加静态 FIFO，不是上述 busy/rolling/twoFIFO provider。`:257` 的 scenarios 是1/1.25/2服务倍数，不能冒充已校准共驻尾分布。`coupled_mode.py:15` 仍要求 `UNIMPLEMENTED`，不能仅换一个 mode 名就算已接通。 |

旧 DAG 路径也有可复用组件，例如 [native_queue_phase_busy_v10.py:65](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/native_queue_phase_busy_v10.py:65) `NativeQueuePhaseFactory` 会保留原图节点和边，并检查完整 profile/context。它依赖另一套 availability/domain/support；当前最短路径应选 strict twoFIFO 为原执行语义主干，复用其观测与 support 检查思想，不再并排新建第三套模拟器，也不默认把旧 DAG 训练画像迁入当前 native4。

## 2. 第一优先：一个共享的 collector cutoff 输入投影

新版本先实现纯 CPU 的 `CollectorPrefix` 适配器，由所有 busy、rolling、HOL、risk 入口共同使用。复用 `twoFIFO_native_busy_state_v1.from_snapshot` 的四时钟与历史一致性检查，不改它的既有接口与保守资格位。

最小输入包含：execution/context/profile 标识、worker lifetime PID、每 owner 固定公共请求日历、原始 arrival 顺序、sender 已观测时间、collector 已处理事件及其原始四时钟、prior history、cutoff。完整公共日历是计划输入；native registry 里未来才出现的行及 body 不是当前事实。

投影顺序必须是：先读 `collector_processing_time_ns`，大于 cutoff 的 body 完全不访问，再检查 occurrence ≤ reader ≤ dequeue ≤ processing ≤ cutoff。START、PHASE、QUERY、RESULT、READY 统一规则。reader 已收到但 collector 尚未处理的 RESULT 仍是不可见输出，不能因 `completion_time_ns` 已写在事后 JSON 中提前移除请求。

现有真实数据位置是 [deployment.py:184](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/deployment.py:184) 的 collector，`:233` 的 READY、`:246` 的 START/PHASE、`:270` 的 RESULT 四时钟；[twoFIFO_owned_collect_v3.py:222](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_owned_collect_v3.py:222) 保存 prior snapshot。这些可立即用于 CPU 历史回放，不需要新 GPU 试验。

需要补的 runtime 数据必须在新的 runtime 版本中实现，不得从旧数据补造：原子 sender 队列及正在 write 的请求身份、worker 已接收/已开始确认、持续单调的 reader/collector issuance/processing 水位、collector 已 dequeue 但尚未 process 的 envelope 标识。当前 `history_snapshot` 锁了 runtime records 和 event queue，但 collector 可先 dequeue 再等 runtime lock；空 event queue 不能证明没有通知在途。`pending_request_ids` 是持久 registry，也不能当未完成 GPU 请求集合。

投影应保留 raw→prefix 的 request 映射及 unknown 原因；不追加审查层级、自动等待器或新的执行门禁。新增的是一次可复用数据转换与算法接口。

## 3. native busy 剩余工作：从集合/区间到可验证的条件预测

### 3.1 现在就能用 CPU 做的工作

实现新 `NativeBusyConditioner`，输出 `observed facts + feasible latent-state support + unknown mass`，调用现有 strict 引擎续跑。结构状态至少保留：

- 每 owner 的 parent enqueue/sender FIFO 与 worker stdin FIFO，包含已写入但 START notice 未处理的请求。
- active request、当前 macro/phase 的可能集合及 residual reference-work 区间；允许“工作实际结束但 RESULT/READY notice 仍在途”。
- 已发行而未处理的 PHASE/QUERY/RESULT/READY、worker START notice 的可能状态，reader 站与 collector 站的供给/排队状态。
- prior priming/前一个 case 的完整状态及 producer ordinal，不把每个 case 重置成空队列。

优先复用 `twoFIFO_busy_candidate_continuation_v1.evaluate` 的逐 world generate→cut→resume。新接口把它暴露为**有明确不完备性的集合预测**：有界可证明的指标输出区间，无可识别支持则 UNKNOWN。当前 `_screen` 只检查必要边界，遇到未映射 PHASE/QUERY 直接 UNKNOWN，这个真实缺口需要填写“观测到哪一条 macro mark、哪个 stream”，不能删掉观测让世界通过。

当前训练 law 在目标 owner/task/pool/phase/context 无支持，或 full/fold 拟合条件不通过时，主入口返回 UNKNOWN；不引入另一组经验系数填补。旧 law 的资格失败也不能靠 rolling 暗中修复。

### 3.2 不能从目前观测直接推断的量

1. `cutoff - START` 不是已完成 solo work。共享进度速率、CPU wait、同步/cleanup 与异步通知都可能不同。
2. 没看到 START 不证明仍在 sender 队列；没看到 RESULT 不证明 GPU 尚在执行；RESULT 已处理不证明 worker 已释放；READY processing 也不是实际 worker release 的发生时钟。
3. host forward END 不是 CUDA 完成；event.query(false) 不提供0到1分数；event.query(true) 只约束已绑定 stream 的 event 已完成，不能令整个 request 的 residual=0。
4. dynamic decode 的未来 EOS、输出 token 数不可见。public max_new_tokens 只是上界，不是观测到的剩余 token 数。
5. 各 owner 的 context/PID 常驻不代表一直占用 GPU 计算资源；NVML 利用率/功率不能恢复每请求 active macro 或分数。

如果需要点预测而非集合边界，必须先有完整的联合观测模型：工作/干扰速率、START/PHASE/QUERY/RESULT/READY 通知生成与传输、reader/collector station、动态 EOS 和 cross-stream 依赖。才能讨论 `P(latent state at cutoff | processed prefix, declared calendar)`。现有有限 donors 的“通过筛选”不是这个 likelihood；不能给兼容 donors 重新均分后称已识别 native posterior。

### 3.3 UNKNOWN 质量守恒

复用 [twoFIFO_forecast_ensemble_v1.py:46](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_forecast_ensemble_v1.py:46) `distribution` 与 `:56` `evaluate`：当前12个预定义 world 的原权重总和为1，不支持/不兼容 slot 保留 UNKNOWN。只要相关未知质量大于0，完整均值、完整 U、完整候选胜者不能从已知子集均值产生。

集合可推出的事件风险区间为 `[已知失败质量, 已知失败质量 + 未知质量]`。这是对声明 world 机制的条件边界，不是有限支持已覆盖 native 真相的概率保证。以后若新增一个确实可识别的 Bayesian/filter law，必须新版本声明条件化语义及离散化误差，不能把原 slot 重归一化悄悄改成后验。

## 4. resource-feasible HOL：说明阻塞，不改真实 FIFO

[RequestAdmission.acquire:67](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/admission.py:67) 只允许队首且 capacity rule 通过的 ticket 获准；[simulator.py:203](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/simulator.py:203) 的 `grant` 同样遇队首不可行即 break；[queue_diagnostics.py:151](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/queue_diagnostics.py:151) 已统计“队首不可行、后方可行”的 HOL 暴露时间。

最小接入增加的是统一诊断字段：真实 queue head、阻塞原因、实际约束、后方是否存在可行请求、阻塞持续区间；资源未知则 feasibility=UNKNOWN，不能推定为可行或不可行。这里的资源可行性有两种语义，必须分开：

- 对当前 shared/no-global-cap twoFIFO，真实约束是每 owner sender/worker 的先后、release、可观测依赖。一个 owner 队首忙时，同 owner 后来的互补请求不能越过；别的 owner 如真实执行器允许，可以独立推进。
- 对明确启用 request_admission 的独立执行配置，才调用其配置对应的 `admission_available`。这只是完整请求的 host permit 规则，不是 SM/DRAM/L2 的物理容量认证；不能把 demand 向量或“互补”标签直接改写为新 permit。

“后方有资源可行请求”首先作为 HOL 解释和**未来路由**候选的输入。要真的 bypass/reorder，必须另建显式执行策略并同时实现 runtime 与模拟器，属于新策略对照，不在当前最小路径内。原 uncapped_shared 不能出现全局 cap1/cap2；干扰严重也不自动引入该限制。模型互补、batch/cache 机制只有真实后端执行且计入等待/操作成本后才可影响候选评分。

## 5. 恢复风险：保留已有学习器，增加真正的校准与独立测试

现有 `RecoveryRiskProvider` 接受当前可见 work、oldest、HOL、waiting 与预声明未来 multiscale work envelopes，未知特征会 raise UNKNOWN；[recovery_risk.py:34](/root/autodl-tmp/generalist-specialist-project/gsdeploy/forecasting/recovery_risk.py:34) 明确禁止把模拟器的未来 backlog 最大值当当前观测特征。继续使用这个字段隔离，新增 collector-prefix adapter；不能把剩余 work 未知设为0使模型可运行。

最小新版本在原 fitted score 外加一个 calibration 对象，而不重造基学习器。实施顺序：

1. 预声明风险事件及 horizon。建议先处理 `P(current backlog clearance time > H)`，其意义可直接解释；未来 arrival burst 另为计划特征。当前 backlog 为空是 NOT_AT_RISK，不是数值0标签。复用 [horizon_recovery_risk_v31.py:23](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/horizon_recovery_risk_v31.py:23) 对“截至H仍未恢复可判1、H前行政删失为UNKNOWN”的正确处理。
2. 按**整次 native capture 和输入内容组**分为 training、calibration、final test，任何同 capture 的相邻 cut 不得跨分区。当前既定独立验证数据只作 final test，不能拿来事后拟合 calibration。训练缺少两类/独立 runs/groups，或 calibration 无足够支持，返回未校准/UNKNOWN。
3. 仅在 calibration 分区拟合预声明的单调映射，例如 regularized logistic calibration；样本和两类覆盖充分时可选 isotonic。映射与 base model、context、cutoff 可观测字段、H、训练/calibration IDs 一起冻结。新 request 只计算分数和映射，不在线拟合。
4. final test 报 Brier score、固定可靠性分箱、已知/未知覆盖、两类计数、按 run 聚类的不确定性；AUC 只表征排序，不能证明概率校准。小样本不宣称普适置信覆盖。完全超训练/校准域仍 UNKNOWN，不能因为映射值在[0,1]就视为支持。
5. 多 horizon 保持 clearance survival 的非增关系。已有 [joint_horizon_risk_v35.py:24](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/joint_horizon_risk_v35.py:24) 的 PAVA 可以复用作形状约束，但源码`:53` 明确不是校准；若在校准后投影，必须独立评估整个投影后映射，不能沿用投影前校准资格。

所需真实数据是同执行器/驻留/输入协议下多次独立 native 完整时序，包含正/负恢复事件、足够H观察长度与明确删失；目前的“2训练blocks+每候选1验证capture”不能自动提供独立 calibration/test/跨内容泛化。已有旧57请求/旧窗口/旧 instrumentation 的风险对象不能直接授予当前181/90+120概率资格。

风险的算法用途先限于解释、触发下一次重预测或提示补采观测。禁止 `U' = U - α risk`、人为加第七项、改变六权重，或用风险阈值暗改稳定性/ρ。风险缺失也不等于稳态通过。若以后要风险约束策略，应作为单独显式策略研究。

## 6. rolling：改当前与未来的预测，历史预测保持不可变

新 `RollingTwoFIFOModel` 维护 append-only 的 `(execution_id, cutoff, model version, observed prefix digest, prediction)`；复用现 observer 的 hash、time monotonic、prefix extension 检查，但原生时钟采用 collector cutoff。每个 cutoff 产生新记录，不覆写旧预测，也不修改旧预测对应的输入或准入资格。

在 cutoff t：固定已提交的 request route/FIFO，吸收只到t的事实，条件化 busy 集合，恢复完整 pending 通知，续跑**全部原公共日历的剩余部分**到原固定 drain。已完成前缀与未完成/未来请求合并，原181分母不变。先做固定路由 shadow rolling，检验 correction 本身；不要一开始同时引入路由切换，混淆收益来源。

随后才允许为尚未提交、没有被实际 sender/worker commit 的未来请求比较合法路由。所有同 cutoff 候选共享同一个实际已提交前缀；不能让每个候选重写自己的历史，也不能把训练得到的另一条完整运行历史当当前实际前缀。驻留改变、卸载/加载、batch/cache policy 改变需要其真实执行成本和新支持域；最小 rolling 版本只改固定驻留上的预测与未来合法路由。

原六项必须按完整窗口重算，而不是“前缀U+后缀U”：

- Q 使用原 offered-rate/task 口径；只吸收已经 collector-visible 的 quality 输出，未观测部分来自支持域内的训练条件预测，否则UNKNOWN。
- T 仍以原 measurement deadline 内的完成比例与完整181分母计算；drain 完成不算 measurement 内完成。
- L/P 合并原每请求 arrival、visible/predicted reader RESULT，按原 task 权重和 P95 定义计算；P95 不可把两个分位数相加，也不可把 world latency 池化后冒充平均 world P95。
- M 计算同一 world 的 `safe_factor × max(原premeasurement峰值, 已观测measurement峰值, 预测后续到drain峰值)`；先逐 world max 再求期望，不能逐段相加。
- E 计原启动一次，按原3600秒摊销；measurement 前缀能量只加 measurement 后缀能量，原口径不计 drain 能量。每个 rolling tick 不得再加 startup。对应原协议见 [twoFIFO_public181_expected_score_v2.py:65](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/twoFIFO_public181_expected_score_v2.py:65)。

当前 expected-score-v2 的 resource law 绑定的是 settled 初态/原90+120协议，不能直接当作任意 busy-cut 的条件 Q/M/E law。新版本若只有 queue continuation 支持，应给出 T/L/P 与完整 Q/M/E UNKNOWN 的诚实结果，不能提前宣布 full_U 完整。prefix power/memory 只采用截止t已实际可见采样；不存在的样本不能由未来 peak 回填。

## 7. 接入真正搜索叶节点的最小工程路径

按依赖实施四步，全部可先在 CPU 上完成，不触碰当前冻结 native4 验证链。

**A. 先统一 observation→state，而非新增执行协调器。** 新 `gsdeploy/forecasting/collector_prefix_v3.py` 适配当前 native raw/snapshot；new `native_twofifo_v1.py` 包装既有 busy state、strict引擎、ensemble 与原score。入口明确区分 settled/explicit latent/partially observed busy，返回 supported/UNKNOWN 及每slot/每请求原因。可以先复用现有真实训练 raw 做因果回放；缺失native字段维持集合状态。

**B. 做实际 rolling CPU 模型调用。** 新 `rolling_twofifo_v1.py` 负责 cutoff journal、不可变旧预测、完整日历、prefix+suffix记账；重用 strict v5_2 的 continuation API，不能把 generic `FullHorizonModel` 的 segment cursor 当同一种 checkpoint。busy 推断未识别时仍可验证 UNKNOWN、区间传播和 state consistency，而不是制造一个可数值续跑的 native 状态。baseline 保留当前固定 origin 预测，rolling影子版本另输出新cut预测，用预声明cut和误差口径比较。

**C. 新的主 prediction/search 接口。** 在新版本主入口中给 `UtilityOptimizer` 等价叶评估器注入 `whole_plan_forecaster(context, candidate, full_public_calendar, cutoff_state)`，默认旧后端；显式选择新 twoFIFO 后端时，实际调用上述模型，返回原 `U/terms/constraints/UNKNOWN` 字段与附属 busy/HOL/risk。不要只在实验脚本或CLI report挂一个sidecar就称主算法已升级。后续正式接线会需要修改一个新版本主叶调用与配置选择；本文只规划，不改现有源码。

新后端的共驻服务不是当前 `resolve_service` 的常数 service 表，不宜勉强塞进它。旧 [coupled_mode.py:34](/root/autodl-tmp/generalist-specialist-project/gsdeploy/coupled_mode.py:34) 的 nonnegative-cost zero bounds 可作为搜索安全起点：没有证明的 service、capacity、memory 剪枝不开启，先在同有限域用 exhaustive CPU enumeration 对照 DFS。不能沿用独立服务假设的容量剪枝，把互补候选事先剪掉。UNKNOWN候选保留在未解决域，不能作为负无穷丢弃后宣布最优。

原 planned-rate ρ 与 g(n) 仍单独使用原公式和适用性证据：[flow_eligibility.py:95](/root/autodl-tmp/generalist-specialist-project/tools/predictor_validation_20261027/flow_eligibility.py:95)。有限窗口 service/window 只是诊断，不能替代ρ。动态 future路由若已不符合其`:38` 静态 `routing_weights→Router` 合同，原gate应返回不适用/UNKNOWN；先以固定原路由验证rolling，不能把实际请求比例或后缀平均率伪装成原 planned routing rps。`DEFAULT_THRESHOLDS` 保持 [stability.py:7](/root/autodl-tmp/generalist-specialist-project/gsdeploy/stability.py:7) 原值，实际稳定性仍由原重建器判定。

**D. 最后增加校准 risk 映射与可视化诊断。** 模型训练/校准函数可以先用 CPU fixture 验证接口与隔离；真实校准只能在真实独立数据满足条件后进行。risk/HOL用于解释和重预测调度，不进入 U，不替代原稳定性与ρ。

建议CPU实施优先级为 `A → B的固定路由shadow → C的固定候选叶接线+同域DFS核验 → D的校准数据接口 → 满足真实证据后的busy概率/未来路由策略`。不能因最后一步缺数据，就不把前面的真实模型调用接好；也不能因CPU接线通过，就把最后一步的native资格设为true。

## 8. 必需的CPU反例与真实数据清单

| 必需测试 | 验证内容；优先复用的现有测试 |
| --- | --- |
| collector cutoff毒化body | worker occurrence/reader≤t、processing>t的START/PHASE/QUERY/RESULT/READY body设为一访问即报错；新adapter不得访问。对照 `test_twoFIFO_native_busy_state_v1`，避免旧reader adapter提前可见。 |
| 晚来观测不能改历史 | t2吸收晚到RESULT，但t1预测字节、prefix digest、旧source/世界权重不变；冻结cutoff不能推进到t2；篡改早期已处理clock/route要拒绝。 |
| registry与空queue歧义 | 未来新增registry行、scheduled早于t但无实际sender动作，都不能泄露active状态；队列空但有dequeued未处理message仍UNKNOWN。 |
| active work与通知在途 | 同一collector前缀构造“GPU仍忙”与“工作已完、RESULT/READY在途”两种合法世界，预测可不同且保留两者；不能双算完整service或令residual=0。 |
| strict continuation一致性 | 显式世界一次跑到底与t1/t2分段→JSON→恢复一致，包括通知/reader/collector、sender write record晚于worker START、READY processing晚于后续START；复用 `test_actual_worker_twoFIFO_reference_v5_2` 及其 independent review。 |
| event/query边界 | false无fraction，true只约束当前stream；缺cross-stream依赖、未来EOS、wrong event/profile/PID/ordinal均UNKNOWN/拒绝，不伪造kernel cursor。 |
| HOL不越真实FIFO | 队首资源不可行而后方可行时，诊断增加但同真实队列后方不得先执行；不同owner按真实无cap协议推进；无 admission 配置不得意外引入cap2。 |
| UNKNOWN质量与质量Q | 1/12未知保留原质量、全期望与完整胜者UNKNOWN；未知Q不补0、不重归一化其余任务/世界；已知全失败不能被其他未知项掩成已通过。 |
| 完整窗口评分 | prefix+suffix181身份不重不漏；P95从完整数组计算；startup只一次、E不含drain、M为world内max；缺prefix资源证据full_U仍UNKNOWN。 |
| 实际叶接线 | 在新优化入口注入可计数provider，确认每个完整候选真调用twoFIFO；不同未来路由导致不同队列/六项；附加risk前后U、排序与原稳定性逐字段相同。 |
| 剪枝与原gate | 小同域exhaustive与DFS结果一致；未知状态不删除；独立service剪枝不能偷开；plannedρ/g(n)/DEFAULT_THRESHOLDS完全一致，不以有限窗口负载或risk代替。 |
| calibration隔离 | 同run/同内容跨training/calibration/test拒绝；final test标签无法进入fit；H前删失UNKNOWN；域外/缺类不输出已校准概率；PAVA不是校准证据。 |

立即可复用的真实数据是已收集且通过既有原始协议核验的 training captures：完整公共计划、prior snapshots、各请求sender时间、四时钟PHASE/QUERY/RESULT/READY、原生输出、NVML拥有进程采样和板级功率。只取source核验后的训练角色；独立验证raw不能反向更新当前模型。

仍需后续真实采集、无法由CPU重放补出的，是原子sender/worker/collector在途水位、当前活跃macro的可识别剩余支持、固定stream/DAG映射及cross-stream边、动态EOS条件模型、同执行环境多run双类恢复事件与独立calibration数据、任意busy-cut条件下Q/M/E的支持。当前单stream event hooks能提供部分完成约束；普通应用CUDA event/query通常不需要特权kernel profiler权限，但它们不等同于具有完整kernel/SM观测权限，更不等于已识别全部剩余工作。

该路径的CPU完成标准是：新主叶确实调用已有strict引擎，合法显式/settled状态可数值跑通，busy不完整状态保留集合与UNKNOWN，全部时序/分母/六项/原稳定性反例通过。真实预测准确度、概率校准和最终分流增益仍由独立真实数据决定。
