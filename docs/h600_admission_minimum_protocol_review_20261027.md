# H600 预测器之后的整请求准入：最小前瞻方案与缺口

本审阅只读现有代码、训练输入和旧设计，不启动 GPU、不 seal 新预测、不改冻结源码。当前 H600 训练与 CPU 审计 watcher 继续运行。下一阶段必须先等固定 H600 模型完成交叉验证和独立预测验证；本文件不授权越过该顺序，也不根据尚未完成的 CV 更换模型。

现有执行器已支持全请求 host 准入，`native_flow.prepare(include_admission=True)` 可产生 shared、admission1、admission2 各四条路由，共 12 个候选。`mechanism_audit.py` 能审计观测到的合同。**这说明可以执行一个机制试点，但尚不等于 12 个候选都有可信的服务、排队、资源、容量或效用预测。** 旧 `mechanism-audit-cpu/admission-transfer-protocol.json` 引用的是 v8 模型，状态明确是设计而非已 seal/已验证，不能直接沿用为最终 H600 模型的证据。

## 必须补齐的接口

|缺口|当前具体行为|新前瞻协议必须明确的处理|
|---|---|---|
|Worker 与 controller 身份|12 个候选复用同一 `execution_context_sha256`，只改变 `resource_configuration`。当前共享 context 没有绑定非共享资源配置。|保留不可变 base worker context，另绑定 controller 源码、标准化资源配置及候选 controller identity；显式表示 shared→admission 迁移假设。不能覆盖旧模型 SHA 或伪造新模式已训练。|
|严格实际画像映射|`flow_profiles.map_flow_collection` 要求非 shared 配置与 context 的资源配置精确一致；现有共享 context 的准入候选会被它拒绝。`mechanism_audit` 允许审计独立 controller 覆盖，但不替代这一语义映射条件。|为新模式设计独立、版本化的 controller-aware 映射/验证适配器，验证 base worker 完全相同且 controller/config 单独绑定。保留旧 mapper/模型不改；不能 monkeypatch 审计或删掉资源身份检查。|
|服务模型迁移|最终 full16 OCR 模型只在 shared 画像上拟合；准入会改变活跃组合、空闲/缓存/频率历史、阶段错开和供给。|把复用相同条件 phase law 写成待独立检验的假设；逐任务/输入/phase context 覆盖检查，未见上下文 UNKNOWN。若失败，另采明确训练数据，再新模型/新独立 trace；不能用本轮验证残差修正后再声称同轮通过。|
|启动能量|`include_admission=True` 生成的候选没有模式专属 `predicted_startup_joules`；共同 startup 省略政策仅覆盖 shared 候选。|先有各模式训练/校准证据，或在机制服务试点中显式保留 UNKNOWN。不能复制 shared 数值、扩大省略组后声称跨模式绝对六项效用完整。|
|拥有进程显存峰值与板级功率|prepare 的 prephase peak 是旧画像经验代理；H600 resource manifest 仅允许八条 shared primary，startup/peak 按 shared route 条件化。相同驻留不能保证模式峰值或条件功率可直接迁移。|单独绑定 mode/controller/route 的 startup→prime→measurement→drain owned sampled peak、空闲与活跃功率证据；只应用一次原 safe factor。资源迁移可另作显式假设，但不能把缺失 E/M 填成 shared 值。|
|容量适用性|现有 H600 capacity 条件包含 resource configuration，且仅接受两 seed×四 shared 路由；准入不是原 exact condition。|准入模式的统计容量/服务适用性另行声明并验证，保持原 rho、g(n)、0.8+1e-8 和稳定性阈值。n 仍按原定义，不能偷换成同时获准数。共享 catalog 不授予准入合格结论。|
|全局 permit 瓶颈|global max1/max2 约束跨全部四 alias，逐 family 的原 rho 不能单独证明 global permit 足够。max1 还串行化 host/同步/清理等完整 handler 时间。|CPU 先预测完整 cohort 和 bounded drain，公开模式的潜在超载/超 drain 风险；若做额外总 permit demand 诊断，说明其只是补充约束，不替代原 rho 或修改稳定性规则。不改变到达或丢弃排队请求。|
|调度顺序与释放时间|运行时 ticket 顺序是不同 sender 线程实际进入 condition 队列的顺序，permit 在 RESULT/ERROR 等 parent 处理时释放。模拟器同时 offer 按 alias 排序，并在预测 handler 结束时立即释放。|承认当前两处近似，将其计入独立队列/E2E误差。若要精确对齐，必须用新版本记录真实 ticket append 序号/时刻、acquire/release 时刻及 release reason，再以训练数据建模传输/释放延迟；不能用验证真实未来顺序或 tail 回填预先预测。|
|实际机制激活|max2 配置存在不等于实际用到两个 permit；过低负载也可“通过 cap”。|在 protocol 中预先冻结激活条件：max1 观测 permit peak=1、max2=2、至少一次正准入等待，并报告 competing alias 覆盖。激活不足为 INSUFFICIENT_ACTIVATION，不能当性能成功。|
|完整候选域与比较|旧四候选 H600 seal/suite/gate 绑定精确四 shared 路由及 seed，不能直接塞入 12 候选。|新建版本化前瞻 plan/seal/suite/report/audit，检查精确声明域及每请求 route/cohort，保留原 temporal/decision 阈值；不得改旧 reserved plan 或复用其测量输出。|

## 最小顺序

先在 CPU 上冻结最终 H600 模型、源码与资源证据，完成当前固定预测验证。机制阶段不必先跑完整 12 候选网格：可以前瞻声明一条确实让两专用与 G2 工作的固定路由，比较 shared、global max1、global max2 三模式；该选择必须在新 trace/结果之前固定。它只回答在同一路由下改变启动准入是否有效，不能选出每模式最优分流部署。

随后区分两种合法的证据目标。若先做服务/队列迁移试点，可明确 startup/E/M 或容量适用性 UNKNOWN，评价完整到达 cohort 的 service、scheduled→worker wait、P95、排队轨迹和机制激活；此时不宣称绝对六项 ΔU、候选最优性或最终分流收益。若本轮要判断六项 ΔU 的正负与排序，就必须在独立运行前补足模式专属训练资源与统计容量证据，并完成全六项前瞻预测；现有要求完整 six-term 的 suite 不能因数据缺失而放宽。

所有模式使用同一新输入/到达/逐请求 route、相同驻留与 priming、同一 H/drain/warmup、固定质量分母和原权重/稳定性规则。预先固定运行顺序、独占输出、无重试/无替换、失败动作和独立请求划分。原 H600 reserved shared 验证计划保持不动，机制阶段用新的 seed 与新 artifact 树。所有预测和源收据必须早于实际 startup，而不只是早于 measurement。

若观察到准入降低 G2 service，仍同时核算专用/OCR/背景任务的等待与完成；从原 public scheduled arrival 起计算 E2E。P95 差值不能通过逐段分位数相减分摊。整请求准入也会覆盖 priming/warmup；后台 OCR 在 global group 中占 permit，不能把它当免费背景。

## 当前 permit 审计能够证明什么

现有记录有 `admission_wait_start_time_ns`、`admission_granted_time_ns`、submit、真实 worker START/END 与 parent receipt。`mechanism_audit` 检查这些时序、每 alias 与 global 观测 cap、最终 active/waiting 为空、peak snapshot、READY/START PID 所属关系、输入/路由/cohort、干净退出等，适合验证已有观测合同。

记录的 `[grant,parent_receipt)` 只是实际 `[acquire,release)` 的内部区间，观测 peak 是 permit concurrency 的下界；wait_start 也不是 ticket append。当前审计明确不能重建精确 FIFO 或每一时刻的完整 permit lifetime。若 protocol 只要求“source/snapshot 加内部区间一致”，应沿用这个有限结论；若要声称精确 FIFO/全时刻 cap 或用毫秒级释放误差解释预测，必须先补新版本 instrumentation，不应抹掉该局限。

模拟器和执行器都采用一 sender/alias、严格 FIFO 与全请求非抢占；队首 alias 达到 max_inflight 时，即使别的 alias 可执行，也可能发生 head-of-line blocking。该行为是准入合同的一部分。知道全过程到达率不意味着知道线程竞争后的 ticket 顺序、每个真实服务随机量或 collector 释放延迟；这些才是完整队列预测仍需检验的部分。

## 后续四组收益确认

机制迁移、资源和容量通过后，才扩大到各执行模式内部的禁/允许分流搜索。保留同域 DFS 的 CPU 核验，并分别寻找优化执行方式内部最佳禁分流与最佳允许分流方案。最终核心比较仍是“优化方式允许分流相对其最佳禁分流”的六项收益；三模式固定路由试点本身不提供该结论。

全请求准入没有分配 SM/DRAM/L2，也未实现共同 batch、最大公共权重子矩阵或缓存共享。`prepare_registry` 对 MPS/green/static-SM 仍设置未验证的 launch blocker；模拟器也会对 batch>1/weight_reuse 请求返回明确缺后端错误。这些执行机制需要各自后端、身份、开销和资源证据，不能从本次准入成功外推。

## 只读依据

- `tools/predictor_validation_20261027/native_flow.py`：12 候选生成、启动/显存代理、共同 startup 比较组与实际执行。
- `gsdeploy/runtime/admission.py` 与 `gsdeploy/runtime/deployment.py`：ticket、acquire/release、sender/collector 生命周期。
- `gsdeploy/forecasting/simulator.py`：预测 offer 顺序、FIFO/head-of-line、预测 handler 结束释放 permit。
- `tools/predictor_validation_20261027/flow_profiles.py`：非 shared 资源/context 精确匹配与严格实际语义映射。
- `tools/predictor_validation_20261027/mechanism_audit.py`：实际观测合同及精确 FIFO/permit lifetime 局限。
- `tools/predictor_validation_20261027/h600_resources_v1.py` 与 `h600_capacity_v1.py`：当前 shared-only 数值来源与 exact-condition 范围。
- `artifacts/predictor-gpu-20261027/mechanism-audit-cpu/admission-transfer-protocol.json`：旧 v8 设计，尚未 seal/execute。
