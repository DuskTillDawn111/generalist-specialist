# 当前 twoFIFO 主链机制与独立 gate 缺口

范围是当前 `phase12_fit_v2 → joint_request_law_v4 → generated_wholeblock_evaluation_v1 → forecast_v4/v5_2` 实际调用链。下表按现有论文机制清单核对代码接线，不声称复现论文完整系统。本次只读源码与 CPU／训练诊断收据，没有读取 validation 结果，也没有拟合、调参或扩建预测器。

| 机制来源 | 当前实际实现与接线 | 当前可核验产物 | 阻止新独立 gate 的关键缺口 |
| --- | --- | --- | --- |
| **USHER：共驻资源与干扰预测** | phase12 固定 12 列：11 个自身 family/phase 的 peer-count 系数及 1 个 same-phase 系数；由已 processed peer path 拟合相对响应，再生成 `(owner,phase,N,same)` rate。新 sidecar v2 已用真实 v3 review/macros 接入 source/header 绑定的七投影 FLOPs／逻辑 operand bytes。 | sidecar 首 shard：46 请求、266 宏、1775 pieces、365 BEGIN、994 priming events；13 项独审测试＋真实重导一致。报告见下方收据。 | sidecar **未进入 phase12 的矩阵或 rate_table**。当前 rate 仍是条件响应统计模型；未识别 runtime quantized 算子、attention/KV、实测 DRAM/L2、缓存收益或物理 compute/memory 分类。静态 FLOPs/bytes 不构成已落实的物理共驻模型，不能据此宣布互补倍率。 |
| **Roomie：有序工作与阶段对齐** | jointlaw 将一整请求的 entry/forward/OCR/tail、mark、query outcome、READY 和协议 slacks 联合搬运；v5_2 按实际 owner FIFO、虚拟工作守恒及 processed phase 更新速率。 | paired v5_2 独审、jointlaw v4 编译测试、runner 完整 8 世界 CPU 测试；现有完整请求向量和 query fingerprint 可审计。 | 当前是宏响应序列和显式协议模型，主链没有调用 kernel sequence alignment/occupancy provider。缺 kernel 配置、真实执行序列／资源竞争识别及新情景 transfer 资格；宏顺序不等于物理 kernel 共驻模型。paired transport 非点识别还可能在任何队列推进前中止整个世界。 |
| **InferLine：全到达时间轴与事件排队仿真** | public calendar 给出完整 offered cohort；每 case 固定 10s＋60s drain；sender write FIFO、worker actual READY、RESULT reader 与 collector processing 分开；生成输入不含 heldout future 标签。三整 block 留出，每 fold 8 donor attempts，固定 8 世界，所有权重／UNKNOWN 保留。 | runner 独审 17 测试；预声明 design；150 cases／546 请求／1200 世界的 CPU 编排 fixture；单次训练完成协调器仅在 exact12 最终报告后执行。 | 数学／OCR／精确输入支持门禁通过不等于 prospective 准确性；缺完整原生训练后的生成覆盖率与整 block 预测误差。case epoch 和 settled past 是已知条件；startup、priming、intercase orchestration 全时间轴并未被预测。任何 non-singleton transport 令整世界 UNKNOWN，阻断完整 cohort 点 E2E/P95。 |
| **Clockwork：显式状态、时序与滚动纠偏** | 已实现 sender/worker/protocol stations、生成世界精确 checkpoint/resume；settled-history seed 保留真实过去寿命、READY、sender 和 reader/collector 时钟及 producer offsets。 | settled seed 和 v5_2 resume 的 CPU／独审产物；runner 使用 immutable pre-case snapshot。 | **busy native rolling 未落地**。`continuation_state` 是生成世界，不是原生中途状态推断；settled seed 对 active／pending／未处理消息拒绝。尚缺实际 active macro/remaining work、pending sender/通知、visible watermark、协议队列的完整因果恢复与不回溯重算。不能把内部逐事件仿真叫作在线纠偏。 |
| **AlpaServe：rate/CV 与到达情景** | 当前优先已知完整 public calendar；固定 donor block × 4 strata 的有限联合请求向量情景已经预声明。库 `FutureWorkload` 另有 Gamma renewal/CV，但本 runner 不调用它。 | 固定 scheme 和 score/quality receipts、请求 ID 重命名不改变选 donor 的测试、全部 UNKNOWN 权重保留。 | donor 情景不是 arrival-CV 情景，也不是已识别 transport latent 概率。当前没有未知未来到达／burst 下的公开情景域与鲁棒排序资格；不应为了套用 CV 自动改当前已知 calendar。有限 draw 平均、世界 P95 的期望与混合 P95 必须继续区分。 |
| **Sinan：短期时序与长期恢复风险分开** | 生成世界诊断提供 queue/RESULT backlog、recovery episodes、多尺度 arrival/work envelope 和 E2E/P95；risk feature 与 resource-feasible HOL 显式为 None，原 rho／六项评分未改。 | `actual_twoFIFO_queue_diagnostics_v3` 与 finite ensemble 的 fixed-drain failure probability bounds；UNKNOWN 世界给概率区间而不归一化删除。 | 本 v3 新训练域没有 calibrated 长期 risk provider 接入。生成世界 recovery 不是原生当前可见风险特征，也不是独立误差校准。忙态 rolling 和可用全队列支持都未齐；旧风险模型／旧域结果不能自动继承。 |

有三项跨行门禁需要先分清：

1. **Transport 可以令所有 multiowner 世界 UNKNOWN，但不是由 multiowner 这个条件必然推出。** 每个 request 生成的 START 若对其所有 feasible u 一致，v5_2 会继续；例如 source 和 target 都 READY≥write，或两者都 READY≤submit 的已识别子域。若一个世界中任意 request 得到非单点 START，异常会上交 ensemble，该世界全部 offered 请求保持 UNKNOWN。若每个预声明世界都至少有一处这种失败，则已知世界权重为 0，不能靠挑 seed／uniform／zero latent 修复。本次没有读取训练标签计算此比例，因此“实际是否全部 UNKNOWN”仍待当前已授权完整训练诊断给出。新单 owner 区间模块只在显式 no-peer 子域证明完整 FIFO/RESULT/P95 包络，**未接入冻结 runner，也不解决 multiowner 覆盖**。保存的两 owner CPU 反例证明 endpoint-only P95 包络不可靠；一般多 owner 需 event-order/visible-rate/协议 max 的可行分支或经证明的保守外包络。
2. **M/E 有模型和隔离 fold adapter，whole-block runner 没有传它。** `resource_provider_v3.run` 可在 exact12 后生成 full＋三 fold；`FoldResourceTrainingProvider` 只装对应 8 donor/2 block。协调器顺序运行 resource 与 generated 两阶段，但没有将前者传入后者。`forecast_case` 当前只传 settled seed，故生成 score 的 M/E/full U 持续 UNKNOWN。一般 forecast 的资源接口存在，但仍需匹配 physical context、fold source、每 case 的真实过去 memory history 和全 observation timeline。现有 M 依赖 ordered state path 支持，未见状态／未知初 history 仍 UNKNOWN；采样 peak 不是容量证明。即便以后接线，transport 区间下资源积分也需要有界支持，不能用一个任选 timeline 的 M/E 填齐 full ΔU。
3. **先取得完整训练资格与生成支持，再决定独立 gate。** 当前 source 门禁要求 12 attempts、546 measured／216 priming-history、full＋三 fold root 数值资格、完整 nuisance support、每目标 OCR input 各 block 的 n0/npositive。失败必须保留 all-UNKNOWN，而不是删列、挑 fold。通过后优先报告所有 150 cases／546 请求的 point/UNKNOWN/incomplete 质量及原因；完整队列／资源／busy rolling 缺口仍需单独解决。Q 当前是冻结训练期望，原生质量尚未验证；训练留出误差也不等于新独立验证或最终收益证据。

可核验收据均来自本轮 CPU／训练域；这里不引用 validation 性能：

| 产物 | 路径（仓库根目录下） | SHA256 |
| --- | --- | --- |
| v3 结构真实提取独审 | `artifacts/predictor-twoFIFO-static-resource-sidecar-v2-independent-review-20261028-readonly01/report.json` | `2b07c74d4b03005556e3574161bbbbc8294eb4a88a274f194de5748810c1c9e7` |
| 生成 runner 独审 | `artifacts/predictor-twoFIFO-generated-wholeblock-runner-v1-independent-review-20261028-cpu01/report.json` | `fff8a565366f87ce15c6eb322dc27ec220bd125b4d4b00f0730497bb43515fea` |
| 资源 fold 独审 | `artifacts/predictor-twoFIFO-resource-fold-provider-v1-independent-review-20261028-cpu01/report.json` | `c1bde7096dc6dbb4cb98f9f4ed7dcdaaf124e7bdb5c55f0f153ccb825693952c` |
| 训练结束协调器 prepared02 | `artifacts/predictor-twoFIFO-training-completion-coordinator-v1-20261028-prepared02/manifest.json` | `de0b33e14e3b5fc3e82d1a4c6c2bc967e4398d33ec13d077c0b45e63b4921c86` |
| 单 owner 区间作者 CPU 产物（独审另行交付） | `artifacts/predictor-twoFIFO-single-owner-transport-intervals-v1-20261028-cpu01/report.json` | `93f84772bebb80581a4e628eeab93d7900dcb6505f720ef72ca0c4f7227fece8` |
| 两 owner endpoint 反例 | `artifacts/predictor-twoFIFO-single-owner-transport-intervals-v1-20261028-cpu01/two-owner-endpoint-counterexample.json` | `8365b6d45757a740c6e4b67a92354cf1a048fe97c290032115fa7ff1522da6f8` |
