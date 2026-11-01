# 主生命周期引擎续算的最小契约（设计与 CPU 反例）

本交付定义当前 `LifecycleDagPhaseModel` 的续算要求；没有实现主引擎 checkpoint/resume，也没有实现从真实设备观测恢复其隐藏状态。GPU 采集代码和采集计划未修改。

当前主引擎 `owner_lifecycle_engine_v21.py` 明确拒绝 `checkpoint`、`pause_after_event_count`、`pause_at_or_after_ns`，输出 `scheduler_checkpoint_supported=False`。`LifecycleSchedulerBridge.freeze_observation_state()` 只是可见生命周期快照。旧 v7/v9 checkpoint 与这些对象都不能冒充主生命周期续算。

## 两个必须分开的入口

1. **同一模拟路径的确定性续算**：保存当前模拟器、scheduler、oracle 的全部因果状态；重载后对相同输入得到与不暂停运行相同的后续节点完成、START、parent receipt、READY、资源积分和六项评分。这里的 remaining reference work、held context、SAVE budget 都是模型假设状态，不是新测得的物理真值。
2. **真实观测同化后再预测**：只接收截止决策时已收到的可验证事实，用它们约束模型状态分布或受支持的条件状态。不能把模拟剩余工作标为设备观测，不能用“没收到完成事件”证明未完成。改变观测前提后不再承诺原模拟路径逐位相同；应保留原预测和观测修正的独立 provenance。

先完成入口 1 的等价性，再接入口 2。入口 2 不能靠把 cutoff 当新的 START、RESTORE 或队列起点完成。

## 固定身份与时间边界

新 schema 必须独立于 v7/v9，绑定主引擎和整个执行闭包的源文件 SHA、冻结 graph/provider 内容与实现身份、execution/resource config、原 routes/owner 列表、完整输入 cohort 与原 FIFO tie order、原 phase start / deadline / drain、scenario / 假设开关、max event budget、模型和 controller identity、SAVE budget policy/seed、初始化 quantile/readiness 选择。不能重选 initial prior，不能重抽同一 restore 的 budget。

所有 host 时间为已核实的统一 `perf_counter_ns`。CUDA elapsed 时间不能直接当 host 时间。保留原测量与 drain 窗口，cutoff 只标记分界；不改变到达、路由、六项等权、rho 或原稳定性规则。四个 resident owner 保留，不引入全局 cap1/cap2。

## 最小可续算状态

“最小”指足以继续现有算法且不丢失已有输出/诊断；静态 graph 可以按冻结 identity 重建，不能把未知设备状态压成零。

| 层 | 必须保存的动态状态 | 缺失会造成什么 |
|---|---|---|
| 全请求与供给 | 完整 `cohort`；`arrivals` heap 的原 arrival/index；每 owner `pending` FIFO；原 now/deadline/drain | 后续已知负载被丢掉，队列与 T 分母变化 |
| handler 与 DAG | `running` 的 row/original START、done、incoming、successors 或等价冻结 graph 引用；`active[(owner,node)]` 的 kind 与精确 remaining reference work；`ready` 的绝对 release、sequence、owner/node | 重跑已完成节点、漏依赖或延迟重新计时 |
| host 生命周期 | `starts`、`queue_wait`、`finishes`；`receipt_pending`；`finish_events`；`wake_events`；`unready`；**`owner_last_ready`** | receipt 与 READY 合并；FIFO HEAD 提前 START |
| 引擎推进 | clock、sequence、event_count；固定预算；下一原子执行阶段；已验证 scheduling decision/rates（见下） | 同一边界算两次或超预算绕过 |
| 累计输出 | maximum_parallel、maximum_owner_parallel、all_completed_nodes、max_waiting、release_hypotheses；energy/peak 及 known flags、resource_transitions | 暂停导致峰值、计数、资源或假设报告被重置 |
| held scheduler | owners 绑定；held_owner、held_knowledge、restore_start_ns、service_available；review_deadline_ns；**pending_switch**；**绝对 context_events**；pending_since（owner/node/request）；last_clock_ns、last_frontier；transitions、queries | 宿主事件刷新 context age、延后交接、改变 pending age |
| scheduler 扩展 | initial_decision、initial_transition_decision、initial_service_at_ns；v49 expired_review_republication_count；适用版本扩展必须有显式 schema | cutoff 重选初始 holder 或丢掉等待 RESTORE 的状态 |
| oracle | initial_prediction、selected_initial；restore_key、previous_held_owner、last_restored_owner；hazard 封包；last_frontier；lifecycle、lifecycle_clock_ns、last_lifecycle；uncertainty、decisions | 丢失 previous-holder/生命周期历史或重复 SAVE |
| SAVE 累计 controller | 完整 sealed `hazard.state`：schema/identity、held/restore/original observation/last update、restore_key、frontier/lifecycle、hazard_budget、consumed_hazard、summation_compensation、terminal_save_candidate、next_review、uncertainty、visible_update_count、state hash | 相同当前状态却预测不同 SAVE；reset budget 造成系统性乐观 |
| mark bridge | 当前 observed states/clock 与冻结 candidate 身份；v56 原 resident 概率域 | 误用旧 lifecycle；把不具活动工作的物理 owner 概率删掉再归一化 |
| queue diagnostics | `initial_ns/now_ns`、cache/costs、registered/waiting/oldest、remaining 与各资源、所有 max、FIFO blocking 积分、episode_start/episodes、unknown；v19 public_cohort/public_unknown/resource_unknown/zero-duration counters | queue age/HOL 积分从 cutoff 重算，未来 offered envelope 缩小 |

`remaining_ns` 的浮点值必须按 Python float 往返精确保留，不得人为 round；正相对完成间隔继续使用现有 ceil 到整数 ns 规则。tuple/set/deque/Counter/heap 的有序语义需显式编码。模型本身若是有状态 provider，也必须有显式导出/导入契约，否则拒绝 checkpoint；函数对象 identity 不足以证明闭包状态相同。

旧 `dag_checkpoint_v7.STATE_FIELDS` 尤其缺少 v21 的 `owner_last_ready`；旧 v9 的 RESULT/READY 与冻结固定 delay 相等校验，仅适用于其模拟路径。观测修正后的实际时间不能为了通过旧校验被改回模型均值。

## 唯一原子暂停位置

建议只支持 `POST_DECISION_PRE_ADVANCE`：已消费当前钟所有到达，完成 parent receipt / READY / dispatch / 零工作节点的同钟 closure，形成 settled lifecycle view，并完成一次 scheduler decision 与 validate，尚未积分正时间区间。

checkpoint 同时保存 validated `gpu_rates`、独立 copy/memset rates、`next_transition` 和阶段标签。恢复时直接进入 boundary 计算，不重复增加本轮 event_count，不重复 `scheduler.decide()`、`oracle.observe_query()` 或 diagnostics observe。scheduler 的 `snapshot()` 本身会推进 context_events、更新 pending_since；导出不得调用它来“读取状态”，应纯读已保存字段。

未来 release 必须保存第一次计算出的绝对时间。现有 `max(original_START + not_before_offset, predecessor_completion + after_predecessors)` 不能在 cutoff 改用 cutoff + delay。`pending_switch` 中 SAVE/RESTORE/SERVICE_READY 的绝对钟同样不能重建。drain 精确边界先完成原 closure，再终止，不能为暂停额外越过边界。需要任意 ns cutoff 时，必须把它作为显式无副作用的边界纳入推进；“第一条 event 在 cutoff 之后”不是 cutoff 的状态。

导入先核实 schema/hash/全闭包，然后检查 cohort 无重复/无丢失、FIFO 前缀、node done/ready/active/dependency 分区、heap 与 owner 状态一致、事件单次且不越过原 drain、scheduler/oracle 同钟和 restore_key 一致、sealed budget identity/hash/累计范围。拒绝外部擅改重新封 hash 的矛盾状态。通过前不能推进任何对象。

## 观测证据能填什么

| 证据 | 可支持的事实 | 不能据此填的字段 |
|---|---|---|
| 公开完整 arrivals/inputs/routes | 所有未来请求、原 offer envelope 与原 FIFO 规则 | 未到达请求不能加入当前 backlog 或 handler-active |
| v2 gate snapshot/sequence | host admission 的真实 waiting/active/release 与顺序（启用 gate 才适用） | GPU holder、SM 服务、DAG remaining；共享模式不能虚构 gate |
| 收到的 START | 原 handler START，已发生的 owner dispatch | 每个 CUDA 节点执行/完成、当前 holder |
| 收到的 host PHASE start/end | host forward/阶段边界；阶段身份 | host END 不等于设备 END，也不等于已知 token 余量 |
| parent RESULT receipt | 原 parent completion 与其携带的 handler END | worker READY 或 admission release（独立事实） |
| 显式 WORKER_READY | 公布的 after-response-flush/before-next-read 边界 | 下一请求已 START；未收到它不能证明 not-ready |
| 新 current-stream end-event query=true | 在 query bracket 内至少一次观察到该记录 stream 前缀完成；有记录 identity/device/stream/host brackets | 精确完成 ns、其余 stream、整个 DAG/请求完成、future EOS、物理 holder、SM work、reference-work fraction |
| query=false | 该次非阻塞 query 没有确认对应 end event 完成 | 剩余比例、剩余时间、未发出未来节点、query 后直到 cutoff 一直未完成 |
| event elapsed ms | 两个已完成 stream events 间的设备时长（包括调度间隔） | 单独实际计算量或可直接替换孤立 reference work |
| owned-memory / board-power samples | 已验证采样域、窗口与证据支持的前缀资源积分/采样峰值 | 采样间真正连续峰值、缺失区间能耗、隐含 startup 资源 |

当前 `device_progress_observer_v1` 没有 forward-event → 冻结 graph 节点/多 stream join 的完整映射。即便 end-event=true，也不能直接删除该 phase 的所有 DAG 节点。需要单独可审查的映射与覆盖契约，或把事件作为部分顺序约束用于受支持的剩余工作分布；不能再制造经验 cursor。

现有 `observed_state.feedback_from_runtime_records` 将任何带 `event` 的 execution event 归类为 PHASE；新 `event=observation, kind=device_event_query` 随后被 `_apply_feedback` 的 `Unknown phase edge` 拒绝。新入口需要独立 typed DEVICE_EVENT 账本，保存原始 event id、request/stream/device、worker query bracket、reader receipt、消费 sequence、来源身份。禁止通过把 query 改成 PHASE end 绕过。

晚到事件按 reader receipt 决定可见性；真实 query 早于 cutoff 但 receipt 晚于 cutoff 时不能提前使用。增量 feed 不得通过回填更早 receipt 重新编号已消费事件。新设备 evidence 宜独立 cursor 或统一原生 ingestion sequence，不混入旧排序算法后改变旧 sequence。该账本仍不构成完整 main-engine checkpoint。

旧 v2 把 handler END 与 parent receipt 一起从 RESULT 导入；它还不能表达“独立看到 END 而 parent receipt 未到”的完整状态。真实同化还须独立支持这些状态、可靠 waiting proof 与条件 handoff；缺证据保留 UNKNOWN。没有可核实当前 device progress 时，不能把原 START 后的 elapsed 从独立总时长中扣掉当 residual，更不能把未观测 pending GPU work/holder/SAVE budget/资源前缀补零。

## 全过程队列与六项评分

维护完整原 cohort；在每个 cut 按请求状态分成 future、arrived-not-started、handler-active、handler-ended-receipt-pending、parent-received，不能重复/丢失。每 owner READY/unready 是独立维度，不能强求“parent-received 且 READY”才计入 completion；收到 parent 时已对 latency/T 完成计数，该 owner 仍可能不能 START 下一请求。

未来 arrivals 继续驱动动态事件仿真、完整 offered workload 与每 owner FIFO；它们不是 cutoff 时的当前队列。already-queued 的顺序需真实 coherent queue/gate evidence 或原执行规则，不能按 alias 重新排序；没有 START 不等于可证 waiting。queue waiting integral、max age、HOL episode、原始 remaining reference work 及 future offer diagnostics 都跨 cut 保存。

按真实 `gsdeploy/forecasting/metrics.py` 与 `prospective_flow_contract_v11.py`，六项是 cost，效用 `U = -(Q+T+L+P+M+E)/6`：

| 项 | 跨 cutoff 的计算要求 |
|---|---|
| Q | 各 task 按原 offered rates 权重，对完整请求的训练预期 numerator/denominator 先求和再相除；保留原 route/input 条件，不读取未来真实输出 |
| T | `1 - 原 deadline 前 parent-received 数 / 完整原 cohort 数`，不换成 suffix 到达数，不把 handler END 当 completion |
| L | 合并 prefix 实际 parent receipt 与 suffix 预测 parent receipt，逐请求减原 arrival 后按 task 求 mean/reference，再按原 rates 加权 |
| P | 同一完整 latency 数组求原插值 P95；不能加/平均 prefix P95 与 suffix P95 |
| M | startup 到 drain 的同一 owned-memory 域峰值取 max，再仅施加一次原 safety factor；不能只取 suffix |
| E | `[original_start,cutoff)` 前缀积分 + `[cutoff,original_deadline]` 后缀积分，边界不重叠；drain 不计能耗；startup/priming 按原 3600 秒摊销一次 |

原代码 P 是 P95 cost，不是 startup cost；startup 摊销属于 E，startup memory 纳入 M。若 prefix energy 未知，`full_U` 必须未知；原 scorer 允许共用同一未知 prefix 的差值比较，但这不等于绝对 U 完整，也不授权把两分支不相同的缺失资源约掉。prefix owned memory 未知同样不能声称完整六项。保留原评分入口，不另造第二套 U。

## 实际 CPU 反例与下一验收顺序

命令：

```bash
python -m tools.predictor_validation_20261027.lifecycle_resume_contract_audit_v1 --output artifacts/predictor-lifecycle-resume-contract-v1-20261028-cpu01/report.json
```

报告六个检查全部通过，使用实际 numeric v42 candidate + v24 controller + v49 scheduler、主 v21 引擎、实际 observer 的显式 FakeCuda fixture 和原 scorer，均为 CPU：

1. 同样 cutoff、restore origin、当前 lifecycle 和空 frontier，但早期 lifecycle 不同，得到不同 consumed hazard 和 SAVE 时间；不能只由当前状态重启 budget。
2. SAVE 后的 immutable pending handoff 在下一宿主边界原样保留，再提交同一 transition 被拒绝；不能 cutoff 再计一遍完整 handoff。
3. 主引擎产出不同 handler END / parent receipt / READY；下一请求 START 受 READY 约束；保留全 future cohort。
4. 主 v21 三个 checkpoint/pause 入口全部明确拒绝，证实本工作没有实现 resume。
5. 实际 observer 的 true/false query 都没有 residual fraction/physical SM/exact completion/all-request-complete，旧 PHASE adapter 拒绝新 query，晚 receipt 不可提前使用。
6. 原 scorer 对 prefix + suffix 四个请求给出 whole P95=285ms；平均两个局部 P95=195.025ms 是错误值。prefix 资源已知时得完整 U，prefix energy 缺失时 `full_U=None`。

报告 SHA-256：`2f8e21f53b124b68e20743058ec5a64942f61eeb26688808f4ed4c8060eccb40`。运行前后导入的项目 Python 源闭包逐字节未变。CPU 反例证明必要性与代码语义，不验证 GPU 预测准确率。

后续独立实现顺序：显式 state schema + 纯导出/事务导入 → POST_DECISION_PRE_ADVANCE 续算及所有边界多次暂停/序列化/整轨迹等价测试 → typed DEVICE_EVENT ledger 与延迟/缺失/乱序反例 → 有证据支持的 graph 映射与条件 residual 同化 → 原完整 cohort 六项与 ΔU 排序验证。每一步保留 UNKNOWN，不能以新增经验 cursor 或 profile campaign 代替主引擎状态契约。
