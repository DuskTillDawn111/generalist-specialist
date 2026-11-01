# 下一轮共享 GPU 耦合效用研究：独立设计审查

审查日期：2026-10-23（Asia/Shanghai）。这是对下一轮新包 `tools/queue_coupled_utility_20261023/` 的只读设计审查；没有修改旧冻结包、原等权目标或稳定性，没有启动 GPU。本文区分可执行建议与已有证据，不把新机制假设写成实测结论。

## 1. 必须先修复的三项问题

1. **新增通用任务会改变其他工作进程的实际服务，而不只是多一个通用队列。** 旧 `refined_policy.py::_forecast` 按各 alias 独立累加服务，`_context` 只能根据候选加入时的可见 outstanding 改变上下文，不能在后续重叠区间持续重算不同进程的剩余工作。旧 CPU `MeanEnvironment` 甚至把整项服务在 enqueue 时固定，之后新增并发不会修改已运行项。下一轮必须由一个共享资源事件模拟器推进全部队列的剩余工作，否则 CPU 中的收益不能检验 seed208 暴露的外部代价。
2. **在线局部代理不等于完整效用。** 旧 conservative 主要 Q/L，global_incremental 是 Q/L/P 和可选 E；旧正式合同 E 率为空，CPU 为 QTLP。旧 P 是固定期望窗口 top-K 待处理贡献，不是真实完整 cohort 的 P95。下一轮应诚实命名 `conditional_six_term_proxy`：依据历史信息模拟未来与未完成工作，评价的是有条件预测分布。只有终局 native Q/T/L/P/M/E 均有合格证据时，才能称为实测完整 U。
3. **同驻留不自动得到 ΔM=0。** 旧实测 M 是最大 owned-process 采样显存乘 safe_factor，路由会改变激活或暂存峰值；旧四对平均 M 已非相等。若保持原评价，在线 M 只能给经校准的候选条件估计/保守区间，缺证据则 UNKNOWN。只有另行证明相关候选共享同一固定预分配显存或采用一个路由无关且已验证的上界常数，在线差值 0 才能成立；这仍须与原实测峰值 M 区分。

## 2. 冻结与复用边界

保持 `tools/queue_refined_20261023/`、`tools/queue_refined_calibration_20261023/`、`artifacts/static-study-gpu-20261022/execution-source-v5/` 和原停止实验记录只读。新包显式导入冻结 worker/transport/trace/quality/metrics/stability 原语，不覆盖旧名字、不 monkeypatch 全局模块、不重封旧实验。

可复用原语：

- `tools/spillover_capacity_20261022/exposure.py::frozen_api` 使用隔离 module namespace，适合读冻结 trace、输入池、质量计量原语。
- 冻结 `DeploymentRuntime`、worker registry、模型预加载、largest-three priming、30 秒 warmup、600 秒 measurement、300 秒 drain、MemoryMonitor、PowerMonitor 可复用。
- `tools/queue_refined_20261023/runtime_window.py::choose_and_enqueue` 的同一把 runtime.lock 下快照、选择、enqueue，以及 explicit newly_observed_completions，是因果审计的基础。
- `tools/queue_refined_20261023/evaluation.py::audit_phase` 的 trace/InputPool/决策摘要、到达输入顺序与 replay 验证思路可复用；新 contract 和 dispatcher 必须绑定自己的实现与模型。

不能直接视为可插拔的新算法入口：旧 native runner 顶层导入的 `validate_contract`、`make_dispatcher`、`window` 绑定旧包。`run(..., evaluator=...)` 仅开放 evaluator，不会自动更换 policy contract 或路由。新 native adapter 应显式构造新 dispatcher/window/evaluator，或采用清晰的依赖注入后绑定 receipt；不能只传 evaluator 就称整个新链路完成。

若漂移守卫使用已经完成请求的实际 `worker_response.service_ms`，需新增显式因果 feedback hook。旧快照仅有 request_id/task/pool_id/enqueue/submit 和 last_completed；旧 newly_observed_completions 只带 ID、model、host 完成时间。新记录须把可见时刻、服务观测值、反馈序号、预测 receipt 一起保存，replay 按可见顺序重建。禁止策略从完整 run.json 或未可见 completion 读取真实未来服务。

## 3. 最小可实现的耦合模型

固定 exact deployment identity，不能从 text×2/G2/OCR 自动转移到 text×3 或多一个 SigLIP2。定义历史条件模型：

`base_work_ms(family, task, pool_id)` 与 `service_rate(alias_head, visible_joint_context, deployment)`，共享事件模拟器同时推进每个 alias 的头部请求；有任务开始/完成/新路由时更新全部活跃项速率和后续完成事件。

context 至少保留：其他活跃家族/alias 数，G2 当前任务类型，G2 可见 outstanding，专用副本状态、自己/其他头部任务的可见年龄、预测输入难度类别、当前总待处理工作量。在线仍只能看到 host 已提交和未返回状态，不能把隐藏 GPU kernel 活动当成真实已知；模型对此误差应有区间和支持域。

校准目标既要包括被分流请求的耗时，也要包括**分流干预前已存在的其他队列**的额外完成代价。随机 probe 相比纯观察能够缓解“排队大才分流”的选择混淆，但单个 selected-path service 拟合并不会自动成为因果 externality 证明。最小实现可使用观察服务模型加保守外部代价区间，将随机 probe 作为样本外验证；无法辨认的区间标 UNKNOWN/禁止可疑分流。

共享 GPU power 模型必须按**同一全卡时间轴**估计板卡功率并积分，不能把每 worker 的 `service_ms × watts` 相加；后者会重复计算相同时段的整卡能耗。保留 idle board baseline、context power 区间和启动摊销项。

模型支持不足（输入路径缺覆盖、未见并发组合、残余年龄超训练支持、漂移后速率明显偏离）不能用“最近历史均值”装成充分支持。应返回明确状态 `UNKNOWN_CONTEXT`/`UNSUPPORTED_AGE` 并触发不分流安全回退。

## 4. 六项代理与实测评价必须对齐的范围

固定原权重六项各 1/6；绝对质量门禁仍关闭；原 rho≤0.8、显存预算与稳定性规则不变。业务偏好的新权重是之后另行冻结的研究问题，本轮不依据验证结果继续调权。

在线可采用共享随机数的有限情景 rollouts：同一已完成历史、同一已到达未完成集合、由冻结静态到达率生成的未来任务/输入。比较各可执行 alias 只改变当前动作，后续路由用同一个冻结策略。**不读取预先生成但尚未到达的真实 trace 或未来输入**。已知静态率与固定窗口终点可以用；实际未来到达总数/顺序不可以用。

- Q：策略只读冻结 family/task aggregate loss；最终用实际输出与 reference 计质量。不能把当前输入精确真实质量作为在线 feature。
- T：每个 rollout 对同一个完整模拟 cohort 计算窗口内完成比例；不能只比较当前任务是否完成，或只拿已完成请求作分母。
- L/P：同一个 cohort 的 e2e 从 scheduled arrival 开始；包括 host enqueue/submit 的延迟估计。每个 task 独立计算平均和真实 P95，再按原 task rate/reference 聚合；不能用 top-K 条件代理替换后还叫 exact P95。
- M：保持前述条件；原峰值 M 未知时不记 0，在线 proxy 范围与实测 M 分开显示。
- E：主终点继续原 600 秒 Ewindow + 启动/预热摊销口径；并行输出到全部该 cohort 完成的 E_cohort 和 drain joules 诊断。E_cohort 不覆盖主 E，不从其较好结果反向选择主口径。

决策记录应同时导出六项预测差值、known/unknown mask、区间/情景风险、baseline/action 的共同 cohort 摘要、所选动作/被拒原因、模型/源码 SHA。即使某项最终在同驻留动作间证明抵消，亦应写明证据和抵消条件。

## 5. 安全回退、预算及漂移

分流是可选动作，算法输出零分流是有效结果。先在两个专用 alias 内采用同一排队感知均衡策略；若所有分流条件不满足，明确回退到专用中预测完成最早且合格的 alias。禁止强制最小分流比例。

只有候选分流在保守总代理差上超过预冻结 margin，并且通过以下检查才允许：

1. 支持域齐全，externality 区间下界仍预测有益；所有同驻留专用/通用未完成任务纳入。
2. token bucket 的允许率、burst、G2 outstanding 上限、可用显存门禁满足，所有 potential paths 已执行 admission。
3. host 决策计算预算满足；超预算按确定性路径回退，保存计时与中止原因。不能在 runtime.lock 内跑无界 rollout 导致排队。
4. 只依赖已可见 completion 的漂移守卫未触发：已完成服务/预测倍率、连续尾年龄超支持、反馈覆盖缺失。冷启动与触发后的最小冷却/恢复观察次数需在开发前冻结，不由验证效果调参。

guard 对照必须公平：同驻留 no-spill 使用相同专用队列均衡、同一预测器/反馈/风险，只屏蔽 G2 的 text 候选；不能拿新分流与弱 SWRR 专用对照因而把专用副本均衡收益误认成分流收益。

## 6. 搜索与 DFS 的精确声明

先冻结有限部署×策略×守卫参数域和 objective。候选应包括原主部署、纯专结构控制、另一有完整校准的 mixed 部署；若没有纯通的 exact deployment evidence，仍可列出七模型池的未支持候选，但 UNKNOWN 保留，不能宣称已证明全池最优。

同一冻结 CPU 环境、相同 cohort/scenario seeds、相同可执行性和 objective，分别运行 plain DFS 和优化搜索。检查 winner、exact maximum、全部 epsilon ties、UNKNOWN 集合。共享随机数使 oracle 确定，不能把某个候选额外采样的结果当成相同 oracle。

耦合服务模型不再允许旧 additive service/rho/memory 下界无条件剪枝。零成本上界 U≤0 在非负成本下成立但可能剪不动；需要正式可证明的界才启用。结构等价消除必须在新 context/guard/反馈语义下重新证明，不能照搬旧纯专证明。

效率报告分开统计 objective evaluations、真正执行的 simulation scenarios、结构等价减少、bound prune、省下的 scenario 数、wall time（含建界/分组）。若只省结构同义候选，应如实说明。DFS 一致仅证明这个有限代理域，native 完整 U 另有测量表验证；两者均不证明七模型池物理全局最优。

## 7. 建议的最小可重复 GPU 设计

这是建议，不是已冻结 packet。快速检查未发现精确 301–310/313/314 数值的新运行；正式生成 packet 时还须解析所有阶段 registry 的 seed 值，记录结果和完整禁用旧 seed 集，不能用字符串 grep 的未命中作为独立性证明。

|阶段|建议 seeds|条件|最小运行数|作用|
|---|---|---|---:|---|
|校准训练|301/302|同驻留禁止 text 分流、输入无关随机小概率 probe（两 seed 反向次序）|4|估计服务/功率与可见外部代价；不能算效益确认|
|CPU screening|313/314|冻结有限域全部候选，DFS 与优化算法分别从空状态运行|0 GPU|有限代理域精确一致与效率|
|开发|303/304|新 coupled、同模式 no-spill、旧 Q/L spill、一个独立校准的其他部署|8|机制与整体候选比较，各 seed 对四条件反向次序|
|冻结独立确认|305–310|固定一个同驻留 coupled vs no-spill 配对，运行次序 AB/BA 交错|12|六个新独立到达/输入种子验证方向及收益|

上述合计24次，是仅一个预定负载、一个新 exact deployment 的下限。其他部署若缺适用服务/功率证据需额外校准；不能把旧部署数据跨域套用来维持这个数量。负载建议首先固定 text=1.85、其他任务各0.01rps，因其已有研究理由；若在开发中同时探索1.89，则开发需明确增加8次或相应预定预算，选择负载后再冻结独立确认。不可从独立305–310挑正的负载/策略。

四个独立配对仅能描述一致性；即使全正，单侧符号检验也是1/16。六个全正对应1/64（以零差对称且独立假设为前提），但六对仍不代表广泛负载稳健性。效用极重尾时主报告原始逐对差、均值、范围和全体已完成配对，不因极端负值删除；统计区间/检验方法在 reserved 前固定。若需要“稳定正”的正式表述，至少同时要求完整证据、逐对差均正和预设均值下界通过；没通过就诚实报告未证实。

校准的4次训练最低可能不足以覆盖全部输入×并发×残余年龄；支持覆盖是门槛，不能为了时间预算越过。可预定额外校准预算和停止规则，但任何新增 probe 或负载都在 reserved 数据首次读取前锁定。若缺覆盖，合法结果是安全回退/不产生可检验分流，而不是填补有利模型系数。

## 8. 物理机制与记录完整性

每次 run 保留原 request IDs、scheduled/decision/enqueue/submit/write/worker service/receive timestamps、实际输出、alias route、可见反馈与候选六项差。报告：两专用同时存在 backlog 时的 G2 text 分流次数、G2 idle/忙状态、专用总工作和最忙副本工作、服务倍率随时间和 G2 overlap 的关联、host 调度延迟、窗口完成率和 drain、Q 与 E_cohort 的变化。

因果机制需看随机 probe 或冻结配对的整体差，overlap 与服务变慢的相关性不单独证明 GPU contention。若希望区分频率/热/功耗节流，应新包增加可获得的 clocks、temperature、utilization/throttle telemetry；采不到则保留未知，不把低功率直接解释成降频。

seed209 的原缺口来自 power integration 的 max_gap=2秒覆盖规则。新采样基础设施可采用单独低开销 NVML 持久采样或稳健常驻 sampler，但仍遵循原积分间隙门槛；显式在起止边界取样、预检覆盖、保存采样耗时与 gap 分布。不能扩大2秒规则、插值补过长 gap、重测到合格后隐藏失败。新协议应事先定义：任何一次 required evidence 失效即将配对标 UNKNOWN；是否终止整个 batch、是否有预注册补充新 seed，须在运行前固定，不得事后修复旧209。

结论接受三种结果：新机制获得正且稳定的实测收益；收益不稳定或接近零；模型识别共享GPU代价后主动选择不分流。第三种同样体现算法优化能力，但不能改称“证明了分流收益”。

## 9. 交付检查

- 新包与外部报告 receipt，新 packet 源码/模型/配置 hash，旧冻结 hash 保持一致。
- CPU 因果/跨 alias 外部代价/UNKNOWN/质量无泄漏/显存非零抵消/能耗积分/guard fallback 测试。
- 校准来源白名单与 fresh seed 审计；开发选择冻结在任何 reserved 执行前。
- 原规则完整六项 native U、同驻留配对、不同部署有限候选表、CPU DFS 一致及效率分别报告。
- 任一阶段失败均保留终端 raw 和错误；不得用部分效用填完整 U 或删除极端配对。

审查时读取的主要源文件：`tools/queue_refined_20261023/{refined_policy,refined_search,protocol,native_runner,runtime_window,evaluation,calibration,iteration}.py`，冻结 v5 的 `gsdeploy/{energy,metrics,prediction}.py`、`gsdeploy/runtime/{runner,telemetry}.py`，以及 `docs/queue_refined_full_weight_exploration_20261023.md` 与旧耦合零界说明。
