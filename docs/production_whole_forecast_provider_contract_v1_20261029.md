# 主生产 whole-forecast provider 接入契约 v1

2026-11-02 补充：另一个显式实验模式 `paper_profiled_fifo_v1` 已将既有阶段共驻/FIFO 引擎接入主 MILP/DFS，见[实际代码落实与小范围验证](paper_mechanism_integration_20261102.md)。它不代表本文设计的 sealed 生产 provider、真实精度或执行资格已经实现；以下保留 10 月 29 日契约及其当时状态。

状态：**待审设计；未实现、未接入、未取得生产或实验资格**。本次只新增本文，未修改任何已有实现、冻结源、claim 或运行进程，也不新增等待器。本文补充已有 gap map 的 API 和搜索边界，不重新审查物理预测模型。

## 1. 范围与当前真实边界

复用现有 DFS/MILP 搜索、有限网格、预算和 UNKNOWN 账本，不复制优化器。新增能力应是一次**完整候选的证据评估**，不能把非可分的队列 law 压成标量 service 后交回旧 FIFO 算法。

当前调用为：

| 当前代码位置 | 已有行为 | 未来接入边界 |
| --- | --- | --- |
| `gsdeploy/optimizer.py:404` | DFS 叶调用 `predict_scenarios` | 共同的完整候选 evaluator；同时审查其叶前剪枝 |
| `gsdeploy/milp.py:683` | MILP 选完整 identity 后调用同一函数 | 同一 evaluator；同时审查列生成、master 和反馈 |
| `gsdeploy/prediction.py:78`、`:101` | resolve service，然后按 alias 独立 FIFO clocks 推进 | 新 provider 模式返回整个预测；不进入此标量 service 路径 |
| `gsdeploy/prediction.py:257` | 固定 service multiplier 1/1.25/2 | 新模式必须声明这些情景是否有依据；不能自动乘 fresh law |
| `gsdeploy/config.py:103`、`:207`、`:271` | 白名单 mode；service/rho/memory 验证 | 新版本 mode 和 provider spec 验证；复用同一完整证据判定 |
| `gsdeploy/system.py:70`、`:96` | 搜索结果复核、公开 predict | 传递并核对相同 spec/request/完整日历 |
| `gsdeploy/system.py:148`、`:154` | 执行前 validate 与重预测 | 复核 spec、候选、输入和执行问题是否一致 |
| `gsdeploy/evaluation.py:138`、`:146` | 实测再次 validate，原 stability/完整性决定 eligible | 用真实绑定复核原约束；不能绕过原稳定性和实测资格 |

在上述主文件中未发现现成的 `forecast_provider`、`provider_factory`、`prediction_provider` 或 sealed-provider 注入接口。`performance.resolve_service` 是 service-matrix 接口，不是 whole-forecast 接口。`gsdeploy/forecasting/__main__.py:24` 的独立 provider 入口不能据文件存在被认定为已接主搜索。

`gsdeploy/coupled_mode.py:15` 目前要求明确的 `UNIMPLEMENTED` placeholder；它的零界安全思路可以复用，但不得把已有 placeholder 直接改名为可执行 provider，也不得修改当前冻结 mode 的意义。

## 2. 最小 API：一个 spec、一个 request、一个 evidence

建议未来新版本提供以下逻辑接口；名称为设计名，不表示当前已有这些类或函数：

```python
open_provider(spec_ref) -> guarded_session
guarded_session.support(request) -> SupportDecision
guarded_session.bounds(partial_request) -> BoundEvidence
guarded_session.evaluate(request) -> WholeForecastEvidence
```

`evaluate` 是纯 CPU、只读、确定于已声明世界/seed 的评估；不得启动 worker/GPU、拟合、重新选 donor、读取未来 validation label、补采样或修改 claim。source 失效应停止该评估，不能重新加载新 law 自动续算。`bounds` 可以只返回有证明的零成本松弛，不要求先实现强界。

### 2.1 不可变 ProviderSpec

spec 是规范化 JSON 的 receipt `{path, sha256}`，至少绑定：

| 字段 | 必须绑定的真实内容 |
| --- | --- |
| `schema/provider_version/entrypoint` | 新版本 schema、明确注册的 factory/auditor；不能仅凭可变模块名动态接受实现 |
| `implementation_sources` | factory、适配器、数值引擎、provider/scorer、support/bounds、传递依赖的真实 source receipts；闭包算法和根入口也有 receipt |
| `training_law/resource_law` | 已发布 law、训练 manifest、denominator、full/fold 身份、资源 donor、审阅报告和支持资格的 receipts；不引用可变 latest 路径 |
| `support_contract` | 硬件/后端/模型与 adapter/物理 owner/副本数/机制、输入支持、精确候选与日历域；支持由完整 identity 决定，不能只看 candidate 名称 |
| `initial_state_contract` | 允许的 observed prefix 或公开初态、clock/cutoff、settled/busy 支持；不允许将 busy 支持集合冒充单点 checkpoint |
| `world_contract` | 原世界 IDs、权重、donor/fold、抽样种子、条件假设、UNKNOWN 规则；标记 joint 或 separate marginal |
| `scoring_contract` | 原 Q/T/L/P/M/E 权重、归一化、P95 定义、M/E 观察范围、startup 摊销、显存安全系数及全部源 receipts |
| `constraint_contract` | 原 executable routes、流量守恒、原 rho 阈值、g(n) 表及其版本/适用域、固定 window/drain、原稳定性阈值和实现 receipts |
| `scenario_contract` | nominal/stress 的定义、支持域及资格；不能把 1/1.25/2 service stress 当到达压力或统计置信界 |
| `bounds_contract` | 每个可用于剪枝/列生成的界的方向、部分候选量词、适用域、证明/审核 receipts；未证明则显式 UNKNOWN/零松弛 |
| `qualification` | 训练数值、transfer、独立预测、原稳定性和最终 gain 分开记录证据与范围；一个 boolean 不得代替证据 |

构建流程：先在未来获准的新版本中完成上述代码及审阅，再从真实入口构造传递闭包，核 source 与 frozen snapshot 一致，冻结实际训练/计划/评分/域，最后规范化序列化 spec 并保存 receipt。构建 spec 不创建实验 claim。任何代码、law、机制、计划域或评分改变都产生新的 spec；原 spec 不能原地更新。

### 2.2 CandidateRequest

request 至少包含以下规范化、可 hash 的字段；它们共同构成 `request_sha256`：

1. `provider_spec` receipt，搜索问题和候选域 receipt，以及 deployment/family→alias/副本数/实际 resident roster。
2. rate/fraction 声明与**精确逐请求 owner assignment**；route ID 不可替代实际 assignment。保存路由器版本、稳定排序、取整/平局规则。
3. 完整公开 cohort：每个 `request_id/task/pool_id/input_receipt/arrival_offset_ns/arrival_index`。保存 cohort/hash，不从同 rates/seed 重新生成替代日历。future workload scenario 必须与 actual future label 区分。
4. 因果初态/prefix receipt、observation cutoff、forecast origin、时钟域、sender/worker/reader/collector 状态；说明是否 settled。搜索时若没有当前 native prefix，只能使用有支持的规划初态，不能伪造 native-v4 prefix 或为每个搜索叶启动一次 native。
5. 机制、后端、硬件、输入处理、batch/公共权重/cache/resource 配置及 source/plan receipts；同路由下机制变化也是新 identity。
6. 原观察合同：window/drain、startup/priming/READY 语义、M/E 时间范围和原评分/约束合同；全部世界及边际分布的固定身份与权重。
7. 请求情景 ID/seed；不得由运行失败后改 seed、世界、请求或时间窗口补全结果。

当前 `System.context()` (`gsdeploy/system.py:85`) 只传 horizon/seed/arrival/drain/energy/startup 参数；两个优化器自己 `make_trace`，没有上述完整输入/状态/spec 通道。因此接叶函数之外还须传递一个共同的已认证 request builder，搜索、predict、执行复核不能各自生成不同日历。

### 2.3 SupportDecision 与 WholeForecastEvidence

support 返回三类，且附完整 source/domain 证据：

- `SUPPORTED`：精确候选和请求落在声明支持域；仍不等于数值完整、准确或稳定。
- `UNSUPPORTED`：域外或缺输入/机制/状态证据；在目标搜索域中记 UNKNOWN，不能当物理不可行。若搜索本来就声明仅枚举固定四 identity，则证书仅限那四个。
- `INVALID`：identity、receipt、来源角色、因果 cutoff 或合同矛盾；明确 HOLD/拒绝该证据，不能退回旧 service predictor。

完整 evidence 必须返回：spec/request/domain hashes，source receipts，support，所有世界/边际的 IDs/原权重/known 与 reason，完整 offered denominator，每请求完成/失败/未完成/删失身份，六项及诊断、UNKNOWN 质量，约束和资格的分项证据。分清 `supported`、`numerically_complete`、`conditional_hypothesis`、`independently_validated`、`constraints_verified` 和最终 `eligible`。

为兼容现搜索，统一适配器可产生 `U`、`diagnostic_partial_U`、`constraints_eligible`、`performance_unknown`、`memory_unknown`、`reason`、六项 contribution 与 scenario 字段；还必须新增/明确广义 `evidence_unknown`。DFS `optimizer.py:415` 与 MILP `milp.py:699` 当前只先识别 performance/memory unknown，否则可能在 `constraints_eligible=False` 时错误记为 infeasible。原稳定性或 rho 未知应进入 UNKNOWN 路径，不能靠假设 `rho=0` 通过，也不能因为缺证据而记物理不可行。

下面是实际消费者字段清单（源码 AST 核对，不含循环变量同名但不是 prediction 的 `p`）：

| 消费者 | 当前读取字段 | 新模式适配要求 |
| --- | --- | --- |
| DFS `optimizer.py:415–449` | `performance_unknown`, `memory_unknown`, `constraints_eligible`, `stress_constraints_valid`, `U`, `scenario_U_low`, `diagnostic_partial_U`, `scenario_diagnostic_partial_U` | 新增广义/策略相关 UNKNOWN 判定必须先于 infeasible；缺失 stress 不阻断已有 nominal 诊断，也不生成 robust incumbent |
| MILP `milp.py:696–806` | 上述字段，加 `reason`, `scenario_delay_costs`, `M_norm`, `Q_contribution`, `Q_norm` | 新 whole-provider 不可在部分项未知时进入 `max(None)`/`min(None)` 或把 nominal M 与另一 stress 的最大 delay 拼成不成立的 cost bound |
| 当前 scenario producer `prediction.py:257` | 返回 `scenario_U_values`, `scenario_diagnostic_partial_U`, `scenario_delay_costs`, `scenario_constraint_valid`, `stress_all_eligible`, `stress_constraints_valid`, `scenario_U_low/high`, `scenario_definition`, `scenario_bounds_are_statistical` | 固定三 legacy stress 槽保留清晰身份；nominal 的 12 世界是一个分布，不是这三个 stress |

当前源码没有 `scenario_M_costs` 消费字段；M 在 MILP `:720` 由 nominal `M_norm` 乘权重。未来若输出逐 stress 的 M/cost，必须明确新增字段和 consumer，不能声称旧 MILP 已支持。最小新 coupled 路径对完整已认证 score 使用同 identity 的完整 `-U`/robust cost；未知时只记有效通用 floor，不复用未证明的 `nominal M + max stress delay` 分解或独立 route feedback。

未支持的 1.25/2 stress 槽为 UNKNOWN，不能复制 nominal、把 posterior draws 填入三个槽或伪造 `stress_constraints_valid=True`。robust 请求只要原所需 stress 任一未知，就走策略相关 UNKNOWN 账本，不记 infeasible。三情景完整时 `scenario_U_low` 才有原策略含义。nominal 的完整条件均值与其独立资格仍单独保留。

## 3. 世界质量、六项评分与稳定性

完整保留原世界权重；正权重 UNKNOWN/失败/未完成/删失世界不能删除或对剩余世界重归一化。权重必须非负、总质量与原声明一致；未知质量和失败原因分别报告。完整 cohort 的任务分母和输入权重不随成功请求子集变化。

逐世界先按原 scorer 计算 T/L/P/Q/M/E，再做声明的期望；`E[P95(world)]` 与混合所有请求后的 P95 不同，必须标名。world 完整但个别请求失败如何评分，沿用原 scorer；scorer 无定义则 UNKNOWN，不能填 0。

线性期望允许在**相同候选/总体/观察合同**下组合完整六项边际期望，不要求伪造联合耦合。当前 `twoFIFO_public181_expected_score_v2.py:286`、`:330` 使用 12 队列世界和两 block 的 Q/M/E 分离边际，`:369` 明确不配对。这只能给既定 Hcpu 下的条件均值；不能 zip donor 与 queue draws，不能据此构造联合 `P(ΔU>0)`、联合资源违例概率或风险资格。若任一需要的正质量项未知，完整 U 仍未知；部分 U 只能诊断。

原 rho/g(n) 与实测稳定性是不同门禁：

- 原规划压力形式与阈值保持不变：`rho_m = sum_t(r_tm * s_mt)/[1000*g_m(n_m)] <= 0.8`，来源见 `config.py:225`、`:262` 和 `optimizer.py:132`。spec 必须绑定该式中 service 的原含义和 g(n) 的真实适用域。新的模型不能悄悄换 service 的定义、把 g(n)=n、以吞吐比例代替它，或用 recovery score 代替 rho。
- 如果候选机制/上下文下没有原 service/g(n) 意义的支持证据，应返回该约束 UNKNOWN；搜索松弛中的 0 压力不代表最终 rho=0，排空也不能补出 rho。
- `gsdeploy/stability.py:7` 的阈值、`:135` 的重构和 `:254` 的分类保持原样。`evaluation.py:151` 接受 `steady/recoverable_congested`，并另要求完整原生 cohort、资源/质量及原约束。原生实测稳定性必须由实际记录计算，不能由预测均值或 CPU 世界证明。
- 若评估 predicted stability，只能按原规则处理各完整世界并保留其证据类型；当前合同没有规定如何将随机世界集合变成一个稳定性许可，缺此规定时返回 UNKNOWN/未评估，不新增概率阈值。

当前 fresh4 aggregate (`twoFIFO_fresh181_validation_evaluation_v1.py:103`) 即使所有 pilot checks 通过，`:109–111` 仍保留 general accuracy、原 stability 和 final gain 未资格。接入时不得自动翻转这些字段。

## 4. 叶前搜索、列生成和证书：必须一起接入

只换完整叶预测不足以修复提前误删。新 provider 的 bounds 必须对部分候选的**全部允许 completion/机制世界**成立；经验 service 均值、静态 occupancy 和训练留出表现本身不是这样的证明。

| 约束/操作及当前行 | 保留或改变的最小规则 |
| --- | --- |
| 已验证 task/backend capability、有限副本选项、正副本、非负路由、流量守恒、alias linking、明确 no-split/exclusions、声明网格 | 结构合同可保留，但必须绑定同一实际后端及机制支持；源域变化不得沿用旧资格。g(n) 数表和 rho 阈值不修改 |
| DFS `optimizer.py:126` 容量、`:237` rows、`:324` 显存、`:343` capacity、`:468` 路由 caps | 仅在有支持的必要条件时提前删枝；非可分 service/memory 无证明则用无容量剪枝/零内存下界松弛，保留完整叶验证。不能借旧 solo service 删掉新 whole-plan 能支持的候选 |
| DFS `optimizer.py:196` utility upper、`:212` Q floor、`:486` pressure 累计 | service、能耗、显存和质量的 floor 均需对机制/context 有效；新 batch/权重机制可改变质量与开销，旧 frozen Q 不自动是下界。无更强证明时非负成本下界 0、U 上界 0 可用，但不是评分填 0 |
| MILP `milp.py:245` capacity witness | 只可在其旧独立域有依据时生成；不得当可行性证明或绕过声明候选域。新 provider 无相应支持时关闭该见证，不改原 g(n)/rho |
| MILP `milp.py:387` columns；`:415` memory、`:420–429` pressure | 当前是显式枚举 task fractions×replica counts，未见独立 reduced-cost pricing oracle。必须在生成前采用有效的保守 memory/service/pressure 界；不得先用旧 service 过滤，再期待叶 hook 补回缺列 |
| MILP `milp.py:537` state one-hot、`:543` task one-hot、`:563` links | 可复用结构和声明域；固定四 whole-plan keys 的 task-column 混合可能造出第五个不支持的候选，必须由 whole-identity support 识别，不能声称被四键覆盖 |
| MILP `milp.py:538` memory、`:574` pressure、`:588` objective floor | 只使用对完整 domain 成立的保守系数。未证明可分性时，保持原预算/rho为最终合同，master 只作安全松弛；不得让 surrogate=0 的最终约束冒充通过 |
| MILP `milp.py:721` route feedback 和后续 identity residual cut (`:789`) | 跨 deployment 的 route 成本转移需要可分性证明；新 whole-plan provider 禁止默认继承。只对已认证完整 identity 和有效全局 floor 的反馈可复用；UNKNOWN identity 可以为去重临时排除，但必须保留其未知界/域账本 |
| DFS 状态预算；MILP `:430` 列限制、`:407` 时间限制及求解预算 | 限额或缺列未覆盖必须留下 incomplete-domain/bound gap；没有完整 pricing/枚举证明时，restricted-master 最优不能标全域最优或无可行解 |

现有 `coupled_mode.py:34` 记录了非可分零界思路；`optimizer.py:46` 与 `milp.py:71` 已有相应开关，但目前仅对 UNIMPLEMENTED placeholder 生效。未来新增真实 provider mode 时须显式审阅这些开关、所有旧 surrogate 快路径和质量界，不能把 `separable_service=False` 当作已覆盖所有剪枝的证明。

无法评估的原合同不是可以删掉的约束。目标搜索域含不支持候选时，保留 UNKNOWN domain 及有效上界；证书只有在已评估 incumbent 不劣于全部未知上界且遍历/界证明充分时才可限域说明，不能把 UNKNOWN 当 infeasible。任何无界定遗漏都应明确证书不完整。

## 5. 固定 fresh4 支持域与公平四组

当前 target `_build` (`twoFIFO_fresh181_execute_v1.py:102`) 硬核对四候选/181 请求、90+120 秒、原六项；`:106` 要求相同 resident roster。资源 transfer (`twoFIFO_public181_expected_score_v2.py:265`) 要求 exact candidate+ordered calendar；`:296` 每个 block 必须有精确 candidate donor。native-v4 session (`twoFIFO_matching_prospective_bridge_v4.py:51`) 还要求 source-bound target manifest/plan。

因此第一版可执行 adapter 至多忠实接受这四个完整 identity 及已资格的精确计划域。generic phase12 law 的某些输入/phase 可支持其他 replay，不等于其他 candidate 的完整 Q/M/E 或原约束有支持。新增副本、路由、输入、机制必须经过明确 domain/transfer 检查与所需证据，不能仅修改 key/name。独立 native-v4 session 需要真实 prefix；不能直接移植到无 native prefix 的搜索叶并冒充已支持。

当前四 routing 候选不是以下四实验因子组：原执行禁分流、原执行允许分流、优化执行禁分流、优化执行允许分流。后一矩阵要求真实机制后端及画像、各组相同公共 calendar/input、同搜索域/预算、原 window/drain/startup/六项/稳定性和独立重复。batch、公共权重/cache、物理 USHER/Roomie 缺口不会由 provider 接口本身补齐。

## 6. 搜索、predict、执行与实测复核流程

1. **加载问题**：核 spec receipt 及完整实现/训练/合同闭包，建立 SourceChangeGuard。所有完整审核在 guard 下重做；路径+hash 也要防 change-and-restore。当前进程闭包不加新入口，必须未来新版本重新准备与审阅。
2. **构建共同请求**：一次冻结公共日历/输入身份和规划初态，由共同 builder 为不同候选派生路由，不改变 cohort。将 spec/request/domain/机制 receipts 写入搜索 problem 和完整候选 evidence；运行中的 model 对象不能代替持久 evidence。
3. **搜索**：先核结构合法性和 bound 适用域，再生成列/剪枝，最后 evaluate 完整候选。所有 UNKNOWN 保留，unsupported/stability-unknown 不落入 infeasible 计数。DFS、MILP、公开 predict 使用同一个 evaluator 和 scorer。
4. **保存 plan**：绑定真实 spec、候选 request、full evidence/seal、已声明搜索域及预算、证书范围、实现 closure。只有四-key 支持时证书明确为那四个，不声称自由七模型/路由网格最优。
5. **执行复核**：`system.run` 在 worker 启动前核原 plan/source/spec 和实际 samples/arrivals/backend/机制。固定四 sealed 计划若有一个路由、输入或时间参数改变，明确 HOLD 或建立新的已获准请求；不能继续使用旧 seal。一般可重预测模式也须留下新 request/evidence 和证书失效原因，禁止默默继承原搜索最优性。
6. **现场 prefix**：需要真实 prefix 的 provider 只在其已审预算/因果边界执行原 producer+independent replay/seal，源 guard 持续到 seal 接受和执行末端；deadline 失败保留原 HOLD。接口设计不增加当前等待器或重复采集。
7. **实测评分**：适配真实 raw schema 到原六项和 `stability.reconstruct/classify`，核逐请求身份与完整观察范围；`evaluation.py:138` 的配置复核同样走 spec 证据，不能回退旧 service 验证。实测 U/稳定性/最终比较和预测 U 分开保存，validation labels 不回流训练。

Session 性能约束也是契约：在 guard 已布置后一次认证大 raw/law/source，保存不可变已解析 payload，整个搜索共享 session。每个候选不得重新遍历/哈希/解析原生大 raw，也不得重复拟合；config/DFS/MILP/predict/执行复核共享同一 spec 身份。热路径使用持续 inode/祖先 inotify+metadata 失效检查，任何写入/替换/权限变化/queue overflow 都失效，即使 bytes 恢复也不重新接受。不能仅用 mtime/ctime 缓存，因为同长写入可不改变可见 stat。

现有 `twoFIFO_source_rebound_execute_v1.SourceChangeGuard.check` 会逐次 `_verify` 全部 refs，不能直接据此承诺上述 I/O 性能；未来需一个单独审阅、保留事件完整性的 session guard，或等价的明确支持接口。首次认证过程前后均检查事件队列，防止 hash 与 watch 建立间隙；需要结束/执行复核时再重认证应单独计量。测试须统计大 evidence 的读取次数，证明候选数增加不导致 raw 重读，同时验证 change-and-restore 仍拒绝。

## 7. 未来有意义的 CPU 测试和验收

这些测试验证接入正确性，不证明物理能力或预测准确。当前不执行未来实现。

| 测试 | 要捕获的失败 |
| --- | --- |
| 相同 spec/request 的 DFS、MILP、公开 predict、执行前复核返回同一 evidence hash | 各入口不同 calendar/seed/router/默认值；漏传 provider |
| 相同 rates 但交换输入/arrival/逐请求 routes；机制/backend/alias/g(n) receipt 改变 | 旧四-key或旧 seal 被错误命中；只有 candidate 名称相同即接受 |
| fixed-four 支持表外的第五个 identity，以及 task-column 混合产生的新 identity | 错称任意搜索支持；UNSUPPORTED 误作物理 infeasible |
| 正权重 UNKNOWN、失败、删失世界；缺一个 offered request；非法权重 | 删除/重归一化、成功子集 P95、完整 U 或错误分母 |
| 两个边际有相同期望但不同联合耦合；世界 P95 与混合 P95 不同 | 将分离 QME/queue draws zip 成联合风险；混淆统计量 |
| 六项完整但 rho/g(n) 证据缺失、busy native 初态或 stability 未评估 | 填 rho=0、将 drain/pilot PASS 当 stable、UNKNOWN 误进 infeasible 分支 |
| coupled fixture 中可行候选的真实成本小于旧 solo service floor，或 whole memory 小于旧加和 | DFS 在叶前误剪、MILP 漏列/误加 pressure 约束；只替换叶函数的假接入 |
| 极小有限域暴力遍历对照，逐一覆盖每条 bounds/pruning/feedback 开关 | 错的 bound/跨 identity cut 导致漏最优；比较必须包含未知候选域而非仅成功集 |
| 列/状态/时间预算截断、缺列但 restricted master 已最优、UNKNOWN identity 去重 | 错发全域证书、未知上界丢失或错误 NO_FEASIBLE |
| nominal 有支持而原 service stress 没有支持 | 任意缩放 law 补 robust score、统计解释或错误 stress 许可 |
| 计划冻结后 source 同长修改并还原、law/ref/父目录替换；执行输入参数改变 | bypass guard、复用旧 scope、重拟合/自动重试或替换 claim |
| 真实 raw 字段适配的小反例：未提交请求、迟到完成、失败、unclean shutdown、recoverable congestion | 改变原稳定性/评分语义或忽略完整 offered cohort |

实施前由 root 审阅本文，并单独确定最初要支持的实际有限域、已有资格证据和新版本 source/plan freeze。本文不授权修改当前活跃闭包，也不把尚缺的原生画像或独立验证视为已完成。

## 2026-10-29 临时副本原型与真实 fresh4 接线边界

原型只存在 `/tmp/whole-forecast-provider-v1-20261029-inpx39hz/work`，可交付统一补丁但不应用。新增显式 `sealed_whole_forecast_v1`，原 `route_load_replay_v1` 的 `UNIMPLEMENTED` 不变。当前唯一注册 auditor 是 `gsdeploy.whole_forecast.cpu_fixture_v1`；packet只声明原始 CPU fixture 世界、日历、服务定义及资源原语，既不能声明 eligibility/qualification，也不能用它替代真实模型。公共 API 为 `ProviderSession(ref)`、`session(data)`、`context_for(data)`、`evaluate(data,x,routing,rates,budget,refs,weights,**context)`。

config、predict、DFS、MILP、system使用同一守卫会话、确切请求摘要和评分契约。服务/内存/能量/**旧 Q** 的继承下界均为零；真实叶值重新执行原 `components` 与原 stability，原 rho/g(n)/.8 gate不变。缺独立 stress 只影响 robust支持，12个nominal posterior世界不会填充1.25/2.0 slot。未支持完整世界/约束保留UNKNOWN与原世界权重。fixture可以为合成有限域求解器发出数学最优证书，但 `native_eligible`/`final_gain_eligible` 始终 false；system.run和evaluate_run明确拒绝其原生执行与评价，不能由backend或JSON字段提升资格。

大payload仅初始化时读取/鉴别一次；完整package Python闭包、packet、spec及其祖先在鉴别前布置inotify。原文件修改再还原、替换、watch丢失、溢出、package新增/删除Python或子目录都会永久使会话UNKNOWN；不会重新读取或自动恢复。失败鉴别也留永久tombstone。返回的spec、ref、世界和request结构与内部/求解器可变对象隔离。

### 后续真实 fresh4 ProviderSession adapter 的确切方案

本原型没有注册真实adapter。下列方案只读审计了既有函数源码，未打开真实law/resource session、未读实时raw/prefix、未启动collector/fit/GPU。

1. 新独立注册schema/auditor，例如 `gsdeploy.fresh4_provider.v1`，spec包含确切原四候选、public contract、匹配fit、资源manifest及reviews、native-v4 manifest/plans、执行context、原input receipts及实现闭包。四身份只能为 `shared_balanced/shared_skewed/shared_spill_low/shared_spill_high`，不可称为执行机制×拆分的2×2实验。原181个请求、90+120秒、equal-six权重、四个resident owner、batch=false/weight_reuse=false和每请求固定owner都是硬边界。main的aggregate rate router如果无法表达固定每请求owner，必须新增显式固定路由request variant或返回UNKNOWN；禁止近似成相同比例。
2. 预鉴别共享构造必须真正调用 `twoFIFO_matching_prospective_bridge_v1.open_law_session(fit_report_path)` 的现有实现（typed-v4中的 `core.open_law_session`；勿重复open每候选），并调用 `twoFIFO_public181_expected_score_v2.open_resource_session(resource_manifest_path,review_paths)`，在同一长寿命 `ExitStack` 中保留其所有guard。若复用 `expected.open_target_session(...)`，注意它内部已经打开typed target/law/resource；不得再开第二份。四个目标plan审计与现有session缓存是否能共享同一law/resource需要一个明确重构，不能把单plan `ExpectedSession` 当四plan session。
3. 只读 `checked_payload()` 的已鉴别缓存；以候选完整request/calendar/context摘要索引**恰好四项**。超域候选返回UNKNOWN，不得扩展成整个DFS/MILP网格的支持域。保留有限域registry并让solver明确搜索该域，或其余域保持UNKNOWN使证书不能误称全域最优。共享law可通过现有 `core.replay_law(payload['plans'],payload['law'],payload['unknown_reason'],plan,prefix,gate)` 跑原12世界；必须保留 `all12_draws_and181_requests_retained`、原权重及UNKNOWN质量，不调用`fit`。
4. 每次实际prefix出现后，沿 typed-v4 `_prefix_and_gate` 重新导出canonical gate及prefix，校验实际READY ancestry、execution_id、input/calendar、epoch=snapshot+300s、先前READY settle预算等完整绑定。不能新造CPUprefix或预先宣称任意native prefix。prefix缺失/当前资格缺失/未知busy剩余状态均UNKNOWN。鉴别prefix/候选gate是小新证据的必要工作；15份训练raw和8份资源raw不得逐候选重新加载。
5. 数值链复用 `expected.compose(resource_payload,plan,prefix,gate,queue,configuration_ref=...,target_manifest_ref=...)`，其中queue保持typed-v4结构及core原replay。compose明确不鉴别任何东西，故只能由上述活动typed session调用。它把12个queue世界的TLP与8资源plans的QME当**分离边际**组合；不得zip成12个完整联合世界，也不得推导Pr(deltaU>0)、完整联合约束概率或将世界P95改成混合样本P95。若转main兼容六项数值，用相同rates/refs/weights和原components复核其期望项；联合约束资格仍需独立证明。
6. 当前真实链自身返回 `independent_accuracy_qualified=False`、`resource_horizon_qualified=False`、`original_stability_evaluated=False`、`full_U_qualified=False`。HCPU资源transfer仍是未校准条件均值，10+60的训练支持不能证明90+120原目标。adapter须把条件均值作为diagnostic，原rho/g(n)服务定义、原稳定性、完整资源horizon支持或当前native资格任一缺失则最终U/约束/资格UNKNOWN。严禁把已认证source、12世界、有条件六项均值、一个sealed文件当作这些资格的替代。
7. 真实执行前再次校验spec/request/config/owner/prefix/source会话及exact seal；沿现有独立producer/auditor分开session与 `expected.seal` 的第二次重放。原生评价仍走真实raw的原evaluation/stability/quality/energy边界，不能沿fixture的拒绝分支伪装已有真实adapter。滚动prefix另有因果collector adapter；其explicit latent continuation不能升级native conditional支持。

原型验收使用真实packet原语通过三个实际求解器与独立五身份枚举比较；包括会被旧service/memory/Q下界误剪的合成最优项、nominal/stress分离、UNKNOWN质量、原rho/稳定性失败、鉴别次数、回传可变对象隔离及change-and-restore/package closure失效。此验收只证明新连接和数学边界，不证明真实fresh4模型已接入或预测准确。
