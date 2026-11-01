# 论文机制与全过程队列预测第二阶段

本轮用户明确授权继续 goal、开子对话并按顺序实施。保留旧实验和工作区现有修改；旧四组结果不是新方法的验证。固定全局 cap1/cap2 只保留历史对照，不作为新算法要求。

## 执行顺序与验收

1. **运行时契约与能力审计。** 在实际锁内记录 enqueue/grant/release，区分 handler END、reader receipt、collector processing、permit release；核实 kernel 时间线、静态属性与计数器。验收为可重放事件及缺失字段清单，不能凭 host overlap 断言 GPU overlap。
2. **USHER/Roomie 服务预测。** 核实配置相关 kernel baseline、register/shared/block/warp 约束及序列对齐；静态 occupancy 与实际吞吐分开。缺资源、架构分配粒度或训练数据时明确 UNKNOWN，不用 context 经验表冒充物理模型。CPU 验证工作量守恒与边界事件。
3. **互补候选与真实执行。** 按配置区分计算/带宽/混合，显存容量单独约束；先实现同权重 owner、配批，再做兼容算子匹配、公共子矩阵与残差 GEMM。资源份额只有核实生效才进入执行候选；所有重排、配批和额外显存成本进入模型。
4. **全过程队列预测。** 在同一执行身份下集成下表全部内容；不只预测当前积压。模型不能执行的动作不进入评分候选。
5. **六项评分与 CPU 检验。** 保持原 Q/T/L/P/M/E、完整 offered cohort、arrival→receipt、原安全因子与稳定性规则。对齐真实许可释放、批处理和队列顺序；验证 ΔU 符号、候选排序与同域 DFS。
6. **GPU 证据。** 先核实正确性与机制生效，再收训练画像；冻结新预测器后独立验证共驻减速、队列、P95、ΔU/排序。验证未过不扩大最终收益搜索。过关后做机制消融及原/优化执行方式×禁/允分流公平四组比较。

## 队列部分必须包含的借鉴

| 来源 | 迁移内容 | 本项目约束 |
|---|---|---|
| InferLine, SoCC 2020 | 全时间轴到达、事件仿真、多尺度需求 envelope | 同GPU共驻服务由新服务模型提供，不能继承隔离/线性复制假设 |
| Clockwork, OSDI 2020 | 显式执行状态、动作时序、配置画像、滚动纠偏 | 不继承单GPU单EXEC；不能通过拒绝请求改变原T |
| AlpaServe, OSDI 2023 | 到达率+间隔CV、trace/分布情景、连续时间仿真 | Gamma指到达间隔，和旧服务Gamma回归不同；已知公开全过程schedule优先 |
| Sinan, ASPLOS 2021 | 延迟积压及恢复、短期详细预测与长期风险分开 | 先事件模型，再风险校准；风险诊断不替代原rho规则 |
| 项目适配扩展 | 剩余资源工作量替代单纯队列长度；配批等待/padding/microbatch；FIFO HOL/convoy；多尺度到达工作量；只用可见状态纠偏 | 任何绕行/配批策略必须执行端也实现；预测情景不冒充置信界；prospective冻结验证不得用未来实测校正 |

当前硬件证据：小探针 nsys CUDA/NVTX 成功，包含部分静态资源；ncu 计数器拒绝。不是完整模型画像，也不是 kernel 执行权限被拒。

## 实现状态

本文件是任务清单而非完成声明。各阶段以源码、测试、审计回执和新路径证据更新；未通过独立物理验证不得写“预测更准确”或“分流收益已证实”。


## 本轮实施进展（CPU 阶段，持续更新）

已落地：临界区内 admission v2 事件、实际 WORKER_READY、训练绑定三边界 lifecycle 与联合条件 handoff 残余；可见状态增量重放及 frozen 防泄漏；完整到达率/CV Gamma renewal（按可见年龄和全过程 rate 积分条件采样）；kernel 静态约束/序列 adapter/零资源等待；算术强度 roofline 互补候选（仅结构代理）与有计数器时的配置互补候选；真实单 owner HF batched generate、实际 arrival→prepare→batch→交付 runtime；公共连续权重块合并 GEMM 与残差原型；完整 batch formation/padding/串行 prepare/owner busy 事件与逐成员原六项评分。

队列新增多尺度 offered-work envelope、剩余 solo/resource work、最老等待、FIFO HOL、积压清空/恢复片段。诊断原 O(queue×events) 重算已由独立反例发现并改为 profile 缓存和增量工作量；等待最老时间使用 heap，不要求 sender FIFO 和 scheduled arrival 排序相同。诊断不更改稳定性规则。

实际 batch runtime 与仿真复核发现 prepare 不能无条件按 arrival+duration 预先结束，已改为与 generate/cleanup 共用 owner 的串行事件；初始已准备请求须提供实际 ready 时间，不能在预测时重置配批等待。准备阶段功耗/显存没有匹配证据时 UNKNOWN，不用空 GPU context 自动补值。

当前未证明：物理共驻预测精度、有效 SM 资源隔离、目标异构模型非平凡共同权重、L2 命中率改善、最终六项分流净收益。新的共享文本 owner 是单列物理部署，不能继承四进程原部署的 g(n)、容量资格或显存数字。旧 T8/V4/F8 保留原结论。

来源与细节见 runtime_event_contract_v2、kernel_resource_predictor_v2、shared_linear_cpu_v2、shared_text_batch_backend_v2、batch_owner_runtime_v2 和 predictor_v2_independent_review 文档。实际 GPU 采集需等新源码/计划封存和本阶段 CPU 检验完成；先原四 owner 匹配画像，再新执行机制消融。


能力审计产物：`artifacts/predictor-capability-audit-v2-20261027/`，明确 `RmProfilingAdminOnly=1` 且容器有效能力无 `CAP_SYS_ADMIN`，旧 NCUPERM 与此相符但没有重跑计数器。新 CUDA driver 只读属性查询见 `artifacts/predictor-device-attributes-v2-20261027/`，未创建 context/发 kernel；已获得当前 RTX4090 的硬件上限。**2026-10-28 更正：**旧tiny trace的 `TARGET_INFO_CUDA_DEVICE.uuid` 虽为NULL，但 `TARGET_INFO_GPU.uuid` 实际存在且与查询完整唯一匹配，早期“旧trace没UUID”结论不再成立。新只读更正产物 `artifacts/predictor-occupancy-declaration-old-probe-correction-v2-20261028/audit.json` 保留原source SHA；旧产物不改。新增arch-aware resolver已对旧6个launch和原prepared01 smoke全部55,891个launch给出观测条件等价的静态上界，见 `docs/occupancy_declaration_resolution_v2_20261028.md` 及 `artifacts/predictor-occupancy-declaration-audit-v2-20261028-final/audit.json`。真实函数最大值/请求carveout未测得。新两capture训练已有77,788真实算子关联行，4,164配置组中3,496组通过训练残差门槛；450组残差超限、218组样本不足仍UNKNOWN，全部kernel peak target缺证。smoke不得进入fit。

Sinan 启发的长期风险模块已完成 CPU 实现：冻结 JSON 的梯度提升树桩，特征仅当前可见队列和公开未来工作量，训练/验证 run 与 group 分离。CLI 可选输出 risk_score，不改变六项评分或 rho；没有物理风险准确性声明。

## 2026-10-28 新物理检查与接口

原四 owner `kernel-profile-campaign-v2-20261028-prepared01/smoke` 已真实执行，
4 次预热加 4 路共跑测量成功，独立 `strict-review-v3/report.json` PASS。
旧 v2 verifier 的空请求误通过由独立审查发现，v3 已核预声明全请求、实际 PID、
原执行源、阶段、内核、receipt/WORKER_READY 和 owned 遥测；不将旧检查器 PASS
单独用作训练门槛。此次 smoke 排除训练/验证，质量仍 UNKNOWN。

单路结构检查 `operator-profile-campaign-v2-20261028-prepared00/structure-smoke`
已完成，strict v3 和 operator-review-v2 均 PASS。新 opt-in worker 观察真实
leaf affine 输入/输出及 packed 参数元数据，未改原模型生成。text 两 owner 各
3365 calls，G2 1265 calls；关联内核耗时总和覆盖约 76.6%/78.6%，不是完整图。
OCR 明确在此 affine 覆盖域外。配对 natural-smoke 按同输入/预热/顺序采集；
插桩对耗时的影响必须单列，不能将 profiling 结果冒称自然 P95/分流收益。

USHER 特征 baseline 已新增独立 CPU 回归/输入适配模块，允许声明明确的
profiled_isolated 域训练；不把 profiler 时长重命名 natural，内存 peak target
独立缺证。官方 CUDA host header 的架构取整计算、严格静态 ceiling 已可调用；
实际 launch 配置缺口继续逐项解决，不能用任意默认常量填。

真实预测生产入口 `forecast_gate_v2.py` 已连接 FullHorizonModel 和六项评分；
配套 `prediction_observed_v2.py` 在采集前检查封存、采集后适配真实父收件端点。
两者尚须接入新全过程实际 collector，且须有新训练模型后才运行独立验证。
OCR/ASR 的预测 Q 已修为 expected numerator/denominator 按任务累计后相除，
缺 reference units 的标量不再自动当作原累计质量指标。详见
`forecast_gate_producer_v2_20261028.md` 和 `prediction_observed_adapter_v2_20261028.md`。

以上没有产生独立预测精度或新四组净收益结论，goal 保持进行中。

配对 smoke 的插桩开销已实测且输出／token工作一致：文本 handler 结构画像
607–613ms 对比自然 287–288ms，G2 为 269ms 对比 156ms，均 R1 描述。
因此新增三域 training bridge，位于
`artifacts/predictor-gpu-20261027/kernel-training-bridge-v3-20261028-prepared00/`：
metadata（逐算子钩子）、light-kernel（nsys 无逐算子钩子）、natural（无nsys）
各 8 个独占训练 job，包含短长输入；smoke 没有回流拟合。
三个域均已完成 8 请求且严格训练审计 PASS；三域桥接报告也已 PASS，
8 请求实际输出、输入、token 和 forward 工作一致，38 个 kernel 阶段签名与
启动顺序全部一致。38,894 个 light-domain kernel 可绑定 metadata 算子特征，
仍不是全 kernel 的 affine 覆盖（实际总数 129,545，OCR 为覆盖域外）。
自然端点保持独立，不使用阶段倍率恢复自然 kernel 耗时。
单 capture 不冒称独立 run 重复，尚不满足 baseline 默认 min_runs=2。
后续 62 jobs/域设计仅 prepared，先过三域对应核验再扩展。

本次 CPU 检查点 `artifacts/predictor-v2-cpu-checkpoint-20261028-prebridge00/`
记录 402 项测试通过、compileall 与 diffcheck 通过、校验期间源码未变化。
这些不是物理准确率。NVTX 导出器区间索引与旧算法的 55,891 个真实 kernel
逐项归属和四份完整 JSON 均相同，后续采集已绑定新索引来源。

原 native_flow collector 已接上可选的独立验证协议、封存预测及 run ID。
三项必须同时提供，实际 collector 源必须在协议内；训练计划拒绝重贴验证标签。
采集前核验全部绑定，并在获取 GPU 锁后再次核验，真实 Unix 开始时间在
板级采样和 worker 启动之前写入原始 raw；检查回执保存在独占 sibling 文件。
CPU 控制路径验证包括正常采集、封存模型被改、等待锁期间发生 drift、部分
验证参数被拒，相关 native_flow/observed/gate 共 28 项通过（无 GPU 准确率含义）。
真实训练模型及独立预测 seal 仍未完成，不启动独立验证或最终收益对照。

## 2026-10-28 小片训练与新生产入口

新增 startup-aware 小片的图像/ASR 三域、shared-SSG 的 light/natural
均已采集完成。图像/ASR 三域实际工作桥接 PASS：4 请求、87 阶段组、
184,776 kernel 的启动签名及顺序一致，22,151 affine kernel 可建立映射；
ASR 两输入实际生成 21/55 token。原 metadata median 审核失败和旧 shutdown
zero 误拒均保留，新的独立因果/host/combined 审核 lineage 明示，不能改写原 FAIL。
SSG light/natural 各 18 请求 strict v4_1 PASS，actual-work 配对也 PASS；
不据此声称自然 kernel 时序已恢复或 same-SM 并发已证实。

两个自然小片执行身份和 12 次实际预热一致，startup fit 已由两个独立 raw
得到启动能量均值 4065.137600198385 J、prephase owned sampled peak
29766 MiB，尚未乘安全因子，3600 s 摊销。新增自然 v4 adapter 逐 input
要求两个独立 capture；目前两个小片实测 job 输入组互不重复，handoff/quality
仍 UNKNOWN，不把单 capture 的多次 trial 或两个副本当独立重复。
详见 `startup_natural_training_fit_v4_20261028.md`。

tools-only `forecast_gate_v3.py` 已连接真实 trained baseline 查询、ordered
kernel/composite lifecycle、host supply 假设及 OCR CUDA dependency DAG。
缺 baseline 不使用实测目标时长回退；未支持的请求/能量/显存保留 UNKNOWN。
新的 disjoint-input 验证域、公开到达及阈值先以 prospective design draft 准备；
模型及源码仍在变化，尚无可执行 GPU 独立验证的完整冻结协议。

接下来先补 G2 模态第二 capture 和能区分 launch 规则的真实奇数文本长度
963，再判断新长度模板与调度资源模型的覆盖；不因训练拟合或 CPU 接线完成
就宣告物理预测或六项收益已经改善。goal 持续进行。

## 2026-10-28 后续执行：已完成画像，转向可检验的完整请求假设

上述第二模态三域、963 三域和第二 SSG natural 已完成，均保留严格审核与
实际工作桥接。新增两个独立 natural 资源采集，每个 30 jobs、84 请求，
原四 owner 与 12 priming 不变；24 种完整 handler 活动状态均在两个新 raw
中取得真实 NVML 内点样本。九 capture 资源模型 24/24 有直接重复支持，
不通过单路或双路功率相加构造未测状态。v6 provider 额外封存嵌套审核器、
完整训练索引和历史原字节闭包；条件均值和采样峰仍不是物理安全界。

完整 kernel 数值机制已有 semantic shape/state covariates、真实 copy/memset
回归和明确 opt-in 的双独立 capture 小样本均值。它们保留原 min_obs=3、
min_runs=2、25% 残差/feature hull 的严格拒绝，数值可运行不冒称严格资格。
单 capture 的 963 新 kernel 缺口继续 UNKNOWN；没有用 query 的实测耗时回退。
新的独立 co-run 请求级检验将检验这些明确声明的假设，而非要求噪声 kernel
全部先通过确定性拟合后无限加采。

主机供给改为移除未来实测 API issue release/handler 终点锚的 DAG 假设。
自然观测只用于训练条件可行集；front-loaded 和 issue-depth 是两种预声明
供给情景。OCR 原适配器明确同步，故其 native-call return 仅作为完成上界，
不冒充观测到的 final_sync。新的完整 CPU 诊断已跑通原 8 个 text/OCR 模板，
包括 129545 GPU nodes 与 13856 copy/memset nodes；不是独立准确率。

生成工作量 v4 对未见内容给冻结经验分布，v5 对已知 public content 单独
条件化。8/9 内容有两个独立 raw 支持；963 单 capture 保持 UNKNOWN。
已知 ChatQA 输入的 5-token 工作不会被无关输入的 6-token 全局混合污染。
预测接口拒绝未来实际输出，当前没有真实 token-prefix 生产器的在线结论。

六项核对发现旧 startup v4 的能量只到加载结束，原评分则积分到测量开始，
包含 priming。新增独立 v5 targets 重算七个 startup-aware raw 的原口径
预测量能量均值 5822.193265376832 J、owned sampled peak 29766 MiB；
旧结果不改。其目标 transfer 还须由 v6 校验实际 flow plan、warmup=0、
shared 候选、真实 priming 内容和所有源码；不能仅传未绑定的输入 ID。

未见内容验证设计 prepared04 已覆盖全部 20 captures、297 measured requests、
216 priming 和 9 个训练内容，保留 16 个内容与 ID 均不相交的公共输入。
独审报告 `artifacts/predictor-prospective-training-independent-review-v5-20261028-prepared04-final01/report.json`
核对四 candidate roots、1642 个闭包来源和 6 个嵌套索引；原三个历史 FAIL
都有相同 raw 的严格 PASS 修正。它仍是未冻结 draft，旧 provider defaults
不能直接运行。新的 text/OCR 已知输入小试验使用独立 schema/封存，明确只
验证该局部子域的新请求；不替代四任务未见内容验证、机制消融和最终四组收益。

保持原六项等权、原稳定性/g(n)/rho 与无默认全局 cap；质量继续按 numerator/
denominator 汇总，L/P 用完整 cohort 的计划到达至父 reader receipt，M 包含
startup-through-drain 并仅乘一次 safe factor，E 排除 drain、预测量能量按
3600 秒摊销。详见新增资源/启动、semantic、exact-content 和 DAG-host 文档。
