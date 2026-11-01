# Phase-aware context scheduler：最小可执行候选

## 2026-10-28 当前执行续接

两套独立 paired training packets 已按原 C 的 12 个 solo/SS/SG job、原 D 的 `SSG-mixed-long-short-long` 同起 job 和原 B 的 OCR short job 准备。每域 14 jobs / 22 measured requests，保留原 12 priming。原 D 并无全长同起 SSG，实际选项是 text#0 长、text#1 短、G2 长；不得把它描述为全长 SSG。参数与审阅在 `artifacts/predictor-phase-scheduler-training-v8_1-20261028-parameters01/`，固定 collector 每包绑定 302 源，新 scheduler/diagnostic 源不在其中。GPU 由主对话串行采集；本子任务只做 CPU 分析。

合格 common capture 的 `context_envelope_diagnostic_v8.py` 对四个受控创建窗口及 12 个 priming 请求重新核验唯一时间对应。`predictor-dag-context-envelope-v8-20261028-diagnostic02/report.json` 的 55,891 measured kernels 中，short SSG prefill 的 kernel lifetime 总和落于其他 context-only envelope 的比例分别为 G2 62.263%、text#0 58.658%、text#1 61.063%；约 1.9–2.2% 留在 unresolved。decode 的 other-only lifetime 为 0，text#1 仍有 0.985% unresolved。launch-exit→start 队列的 other-only 覆盖约 53–66%，同时前序 same-stream kernel lifetime 覆盖约 96–99.99%；二者重叠，不能相加，也不能把整个队列叫 context wait。这是 envelope 可用性代理的证据，并不是物理 SM service 分解。

packet01 的两域 strict 与 paired actual work 已由主对话检查 PASS；独立 CPU 分解 `predictor-dag-context-envelope-v8-20261028-training01/report.json` 覆盖 392,487 kernels / 22 requests。该更长采集中 30 个 unpaired SAVE 和 1 个 repeated RESTORE 原样留为 unresolved；活跃共跑 lifetime 的 unresolved 比例约 0.99–4.58%，solo/OCR 为 0。strict PASS 不能自动变成 context interval 完整配对，拟合必须保留这项缺口。

已新增 `conditional_availability_scheduler_v8.py` 的显式状态机：每个 exact frontier/current-owner/reason cell 决定 next-owner、grant budget 和 handoff，没有默认经验值或 equal-context fallback。grant/handoff 使用绝对截止时刻，host event、重复同 clock decide 与 phase 变化不重置；remaining reference work 由 DAG engine 唯一推进。尚未填入真实训练参数，不能称为已通过预测验证的模型。

独立审查发现并修复新 v8 engine 的两个时序反例：同 clock 上前请求完成、receipt、worker-ready、下一请求 dispatch 先求闭包再决定 early yield，避免短暂空 frontier 造成错误切换；drain 截止恰好完成的请求会完成同 clock 的 receipt 后再停止。旧 v6/v7 引擎原样保留，新 v8 checkpoint 仍明确拒绝。

`static_dag_phase_binding_v8.py` 已在原 8 templates / 129,545 GPU nodes 上核验原 SQLite/request/kernel denominator 与完整 kernel 内容，用原 launch API 的 NVTX membership 固定 prefill/decode/outside/OCR 标签。预测时仅查精确 node fingerprint，不读未来 Python wall。`solo_reference_work_v8.py` 对两次独立 solo capture 的完整 ordinal 静态签名、phase membership 和实际生成 work 作相等检查后，以两次 solo lifetime 中位数作为 reference ns；不引入 SSG lifetime 或整请求倍率。实际 source-bound 检查见 `predictor-solo-ordinal-reference-v8-20261028-diagnostic01/report.json`。同目录 `request-work-features.json` 导出 8 个 exact text/OCR input 的全请求 GPU reference-work 总和与输入/源闭包；它只是风险模型 work feature，未包含 host/copy/lifecycle，image/ASR 仍 UNKNOWN，也不具新 context 的性能资格。

现有 equal-context 模型把“已进入 DAG active 集合的 GPU 节点”直接视为随时按 owner 数均分进度的对象。同一 owner 的多个 ready stream 仍各自得到相同进度率。模型没有持有当前 context、切换等待、context 被切出后残余 kernel work、peer phase 或 device queue 状态。更准确的 host release 只能修补提交供给，不能让这些缺失状态自动出现。

这里不能把 G2 误差归结为已证实的某种硬件机制。独立诊断 `predictor-dag-phase-scheduler-v8-20261028-diagnostic01` 从两次 solo 和 original/repeat SSG 的 light/natural 原始证据重新核查：216 个 phase group 的逐 ordinal 静态 kernel signature 完全一致。两次 SSG 的 G2 prefill lifetime 总和 / solo 基线的组中位数分别为 2.629 / 2.812，但逐 kernel 比的中位数都约 0.991，组 p99 比的中位数约 22.7。G2 decode 的 lifetime 总和比为 1.758 / 1.656，最高 1% kernel 集中了约 98.9% / 98.7% 的正 lifetime excess。不同 phase、长尾 kernel 和请求相对启动状态必须保留；整 SSG 系数会丢掉这些区别。

长输入 G2 prefill 的 launch-entry→GPU-start 中位数约 485–599ms，natural prefill 与两个专用的 prefill Python phase 同时重叠约 78.7%–100%。这些是提交排队和 phase 依赖的重要观察，但 Python phase 不是 GPU busy，Nsight start→end lifetime 不是 SM service，约 6.5ms 的某些尾部/间距也不能直接叫 quantum。API wall、同步 wall、lifetime 不能直接相加作执行时间分解。

## 已实现的独立执行入口

新 `dag_phase_scheduler_v8.py` 保留旧 v6 的整数 ns 事件、DAG 依赖、host release、lifecycle、queue diagnostics 和 resource accounting，新增 scheduler 注入。旧 v6/v7 不修改。

每次决策只收到当前整数 clock、当前 active GPU 节点的 immutable view，包括 source-bound phase 标签、公开 request/task/pool 身份及剩余 reference work；没有未来 raw endpoint。返回每个 active GPU key 恰好一个有限 [0,1] rate，另可返回严格大于 clock 的整数 `next_transition_ns`。GPU 全部 rate=0 时必须提供未来切换点。到达、host release、GPU/copy completion、deadline 和 scheduler transition 一起决定下一事件，未执行 reference work 留待后续 context 再获供给。copy/memset 独立进度仍是原显式 hypothesis。

reference work 的单位是 duration provider 的基线 ns，不称为 SM 执行 ns。多个同 owner 节点各自累积进度沿用 intra-owner overlap hypothesis，不把 rate 之和当 GPU capacity。真实 policy 参数缺失时明确 UNKNOWN；显式 legacy adapter 可复现旧 equal-context 数值，仅用于 CPU 对照。当前版本明确拒绝 checkpoint/pause/resume，因为自定义 scheduler 状态尚未纳入完整续存契约。7 个 scheduler 测试覆盖剩余 work、arrival/release/completion 与切换边界、rate 契约、legacy 兼容及 checkpoint 拒绝；加诊断及旧 v6 回归共 26 项通过。

`dag_phase_diagnostic_v8.py` 是 CPU 证据入口，核对 v7 host training02 原两 solo 来源，分开输出 light lifetime/launch/runtime/sync 与 natural peer Python phase 联合暴露，不拟合 slowdown 或读取 validation labels。

## Common-session 的可识别对象

原 unified v8 小片的 strict FAIL 保持；它只能用于能力诊断，不能作为合格拟合 capture。`predictor-dag-phase-common-session-identifiability-v8-20261028-final02` 已用其受控逐一起动过程建立唯一 observed temporal correspondence：4 个 scheduler context 首现各自只落入一个新 owner 起动窗口，nonce/start_ticks 一致；最小窗口边界余量 335ms。再用同一 Nsight 原生时钟中的 12 个 priming `GS_STAGE` NVTX 区间，314321 kernels / 628642 endpoints 全部落入对应 context 包络，枚举的 23 个错误排列均被排除。这个归属是受控时序证据，不是 OS NSpid 恒等证明。后续 qualified capture 必须独立重做，不能复用旧 context id。

SQLite enum 明确 tag8=`RESTORE_START`，tag7=`SAVE_END`。两者围成的区间可以作为 **availability surrogate**；其中可能包括 restore/save 与 idle，不是精确 SM busy 或物理 residency。原 trace 有 6 条 SAVE_END 没有对应 RESTORE_START，不能补造区间；受影响节点/状态要单独计为 UNKNOWN/censored。

## 可以实际拟合的最小候选

候选保留 v7 host supply，以 exact-known solo 的逐 ordinal reference-work 基线驱动 DAG。scheduler 增加 current context、轮转顺序、剩余 grant budget 与切换等待状态，通过 availability surrogate 控制 GPU 节点 rate=0/1。先不加共驻乘数或物理资源效率系数。需要特别避免：当前某些 semantic numeric fit 使用过 SSG lifetime，把它直接放进 availability gating 会重复计入被切出时间。新候选必须使用重新明确为 solo-only 的 reference-work provider；不能沿用含共享 lifetime 的 duration 再加一次 context 等待。

训练分成三个互不重复的对象：

1. **提交与可运行状态。** API entry/exit、stream/event 依赖和静态 phase 标注重建队列 frontier。launch→start 包括同 stream 前序工作、device queue 和切换等待，不能全部叫 context 等待。可运行 phase 用“已提交而未完成的 DAG frontier”，不能用可能已经提前返回的 Python phase 替代。
2. **context 切换/预算。** 在至少两个 context 持续有待执行 DAG 工作的区间，统计 observed SAVE_END→下一 RESTORE_START 的 handoff gap。仅对整段始终 backlog、至少一个 peer 始终 backlog 的 grant 估计可重复 grant budget；owner 队列提前排空的 grant 是 early-yield/censored，不能当 quantum 样本。按 exact phase/frontier/输入域和运行顺序保留状态；没有支持的状态 UNKNOWN。只有跨两个合格独立 capture 可重复的 conditional budget/transition 才能进模型。
3. **执行进度。** 先固定 solo reference work，在 surrogate grant 内 rate=1、外 rate=0，恢复时继续剩余 work。kernel start→end 与其他 context availability 包络重叠的部分另报告；不将它再次加入 base duration。这个模型是假设的有效供给过程，是否解释真实 kernel 完成/phase 完成和排队必须留给独立 capture 检验。

如果完全 backlog 的 grant 长度不形成稳定的条件分布，固定 quantum 模型应当被拒绝。届时拟合“当前 frontier/上个 context/peer 状态→下一 context 与 budget”的有限支持半马尔可夫转移，而不是用整请求倍率掩盖差异。切换 gap 若包含无 runnable work 的 idle，也必须剔除或作 censored，不能都作为切换开销。

## 下一次最小新增训练与冻结检验

当前小片只有短输入 SSG 和 OCR。先在修后 qualified v8_1 上重新验证归属、队列 frontier、包络配对与 phase 对齐。需要第二次同协议 qualified common-session 独立 capture，才有两 run 支持。随后最小扩展加入已知长输入 SSG，并保留原 relative-start offset 的同步起动及 G2 延后起动两种状态；每个训练 capture 内都包含相同 solo/priming 锚点。它们是识别持有预算、early-yield、长 prefill/preemption 的训练数据，不读取旧失败 validation labels。

拟合完成后预先冻结 solo reference work、v7 host scenario、phase 映射、转移/预算/切换参数、支持域和六项评分；单独保留第三个 common-session/natural pair 只做预测检验。对照 legacy equal-context 与新 scheduler 的逐 kernel/phase 完成、launch queue、请求服务/全程排队/P95，再看 ΔU 符号和候选排序。不得用这次检验重新挑 scenario、修 grant 参数或回填共享倍率。超出当前短/长、offset、context/phase 支持域的部署候选继续 UNKNOWN。
