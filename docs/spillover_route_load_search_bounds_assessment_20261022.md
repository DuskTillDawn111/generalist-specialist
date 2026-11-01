# 路由／负载相关服务模型接入搜索：只读实施评估

可以保留现有联合 MILP、DFS 和完整方案反馈架构，但不能把新模型直接塞入旧 `SurrogateBounds` 并继承旧下界。最小安全方案是：**完整方案用新模型逐输入回放；搜索只使用新模型在声明域内可证明的服务下包络，没有这种包络时服务与容量界取零。** 正确性可以保留，剪枝效率需要重新实验。

本文只审阅代码及[局部模型设计](/root/autodl-tmp/generalist-specialist-project/docs/spillover_local_model_design_20261022.md)。没有修改生产代码、拟合参数、运行 GPU、读取新增运行结果或作新验证声明。旧 `replay_transfer_v1`、61/62 的原协议及失败历史保持独立。

## 1. 当前架构中真正依赖“可分”的位置

现有逐输入服务为 `S(m,t,n,input)`；固定路径／副本／输入后，不随其他任务的路由改变。拟议新模型为 `S(m,t,n,input; x,R,λ)`，例如背景 G2/text 和 OCR/ocr 的服务也依赖 ASR 给 G2 的份额。相同背景列会对应多个不同完整服务值。

| 位置 | 当前假设／用途 | 新模型直接复用的风险及安全处理 |
|---|---|---|
| [SurrogateBounds](/root/autodl-tmp/generalist-specialist-project/gsdeploy/surrogate_bounds.py:13) | 构造器直接调用旧 `calibration_v2.input_service_ms`；`family_service` 为逐路径跨输入最小值，`pressure_service` 为输入池均值，`request_service` 为确切逐输入值 | 需要新适配器。不能以旧表代表 route/load 相关服务；完整预测值和下包络表必须分开。 |
| [MILP 初始化](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:75)、[DFS 初始化](/root/autodl-tmp/generalist-specialist-project/gsdeploy/optimizer.py:50) | `surrogate != None` 即令 `separable_service=True` | 新模型实际可耦合，但其**下包络**仍可分。须显式声明“有可分且有效的下界”，不能把非空适配器当成证明。 |
| [MILP.floor](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:138) | task 列只含该任务活跃模型的 counts／row；≤1024 请求跑单任务 FIFO，否则用逐输入无等待成本 | 列看不到 ASR 其他列及空闲驻留模型，不能调用依赖完整上下文的服务值。应使用所有兼容完整上下文上的逐输入最小服务。 |
| [MILP.columns](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:412)、[主问题容量约束](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:564) | 列的 `q * pressure_service / capacity_factor` 可相加，并提前删除 rho>0.8 的列 | 旧均值可能高于新模型某一上下文的均值，误删最优解。压力必须是可证明的下界；未知取0，保留列。 |
| [DFS.capacity／service_floor](/root/autodl-tmp/generalist-specialist-project/gsdeploy/optimizer.py:125) | 用固定路径服务算剩余容量；floor 缓存键为 deployment、task、row | 路由新增任务后既有任务服务会改变，旧 demand 不能作为精确压力。只累计固定的下包络压力；上下文条件界须扩展缓存键。 |
| [DFS 路由递归](/root/autodl-tmp/generalist-specialist-project/gsdeploy/optimizer.py:466) | `demand[m]` 逐任务相加；排序、容量覆盖预筛和全局上界也使用服务表 | 必须统一替换所有用途。即使关闭 `separable_service`，递归末尾仍会调用 `pressure_service` 更新 demand；不能只改一个开关就认为全部禁用。 |
| [完整 FIFO 预测](/root/autodl-tmp/generalist-specialist-project/gsdeploy/prediction.py:97) | `input_service_ms` 不接收完整路由；均值 service 用于 rho，逐输入 service 用于各 alias FIFO | 新模式应先绑定完整 context，再产生相同语义的逐输入服务及池均值。不能只改 `resolve_service` 的均值却让回放仍用旧逐输入表。 |
| [SystemConfig.validate_plan](/root/autodl-tmp/generalist-specialist-project/gsdeploy/config.py:196) | 单独求 service、rho 与 memory，与预测入口共享契约 | 新模式需同时接入这里，保证搜索、重放、物化检查对同一方案的可行性完全一致。 |

## 2. 可保留证明的服务下包络

令 `D` 为明确声明的完整部署、路由网格、负载及上下文域；固定本次负载后，λ 不再是搜索变量。对于任一路径／副本／输入，建立：

```text
ell(m,t,n,i) = inf S(m,t,n,i; x,R,lambda), over compatible complete contexts in D
ell >= 0; unknown/unproved lower envelope -> ell = 0 for bounds only
```

这里的 inf 是**冻结新模型函数的数学下界**，不是“观测过的最小 service”，也不是物理硬件下界。局部线性倍率在声明多面体上可用顶点最小值；不完整采样角点的最小值不构成整个长方体上的证明。完整模型必须在其声明域内给出有限正服务；倍率可能非正时应拒绝该模型域，不能临时裁剪来补救，除非裁剪本身事先属于新模型定义。旧单机基线或倍率1也不能自动充当下界，因为观测或模型可能允许加速。

下界无需各路径的最小值同时可达；分别取更低值形成松弛是安全的。随后可以安全复用以下证明：

- `family_service`：取 `min_i ell(m,t,n,i)`，供未指派任务的服务下界及全局上界使用。
- `request_service` 的**界接口**：返回对应输入的 `ell`，保留原始样本到 alias 的映射。长队列的均值、P95 使用同一批逐请求下界；不能以总体均值替代 P95 的逐请求界。
- 单任务 FIFO：固定真实完整方案的任务到达、顺序和 alias 分配，删去其他任务，且服务始终使用 `ell`。FIFO 完成时间对每个服务时间单调，因此每请求延迟、经验均值／P95、窗口外完成计数均不大于完整回放，可继续形成 L/P/T 下界。

最后一条不允许“删除任务后重新计算倍率／忙度并把新模型输出当下界”。删除任务可能改变交互服务；安全证明来自逐请求 `ell <= 完整服务`，不是来自未经证明的干扰单调性。动态忙度模型若要使用此界，`ell` 还必须覆盖所有允许的调度状态；否则先禁用单任务 FIFO 界，L/P/T 取0。

`pressure_service` 应为与完整 rho **同一输入池定义**下的下界。例如保留当前“输入池等权 service 均值”契约时，取 `mean_i ell`，即可保证：

```text
sum_t q(t,m) * pressure_lower(m,t,n) / (1000 * capacity_factor(m,n))
    <= complete_model_rho(m)
```

于是下界压力>0.8 才可安全剪枝；下界为0相当于放宽容量，不表示模型有无限真实容量。不能把“为安全而放大的 service 上界”放进必要容量约束，那会误删对新预测器可行的方案。完整方案仍用实际 context 的池均值检查 rho≤0.8。若改为实际路由样本条件均值或每 alias 忙度，就需要同时重写完整 rho 及对应下界，不得混用旧池均值。

## 3. 六项成本、未知域与证书

**MILP 的完整身份反馈仍可用。** [完整方案评价与 residual cut](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:663)只要求基础成本界 `B(z) <= C_new(z)`，然后对已经评价的完整部署＋全部任务路由添加身份修正。它不要求 `C_new` 可分、凸或单调。现有模式仅对 `independent` 允许跨部署的 route-feedback cut；新模式应继续禁止这种跨上下文转移。

完整评价缺证据、超出新模型适用域、动态回放多解／不收敛时应返回 unknown，保留该身份的效用上界／gap，不能添加“已不可行”的裁剪。只有模型明确给出的显存／rho／排空约束失败才是该模型下的不可行。现有 MILP／DFS 对 unknown 的记录可延用，但不能把局部未覆盖组合悄悄移出七模型全集后给出全域证书。

| 项 | 可保留什么 | 何时必须放宽 |
|---|---|---|
| Q | 当前历史逐输入质量、任务路由与输入序列未改时，task 列的质量成本仍可精确分解 | 若新模型引入负载相关质量、错误或提前丢弃，需新质量下界；无证明取0。质量未知仍不能生成完整 U。 |
| T/L/P | 同输入／alias 的 `ell` 回放下界；完整成本来自新模型整方案 FIFO | 无逐输入下包络时取0；不能传递训练实测时延作为其他上下文的界。 |
| M | 若新模型明确定义 `M=max(additive*固定系数, seen_peak)`，其 additive 部分仍是该**预测函数**的下界 | 局部实测最大值或新非加和峰值模型不保证旧 additive*learned 系数是下界；state／column／DFS 显存预筛应取0，完整方案再判断。 |
| E | 如果仍冻结旧非负 intercept/state/route 公式且其完整 E 也严格使用该公式，原分解可保留为新模型的数学组成 | 若功率依赖总路由、忙度或并发，旧 `route_energy` 不再自动精确；仅保留能证明的非负常量／逐状态／逐列下界，其余取0。不能让局部少量运行重新“识别”每 worker 系数。 |

“把新模式的 `surrogate_bounds()` 返回 None”**并不足够**。[solver 的 profiled_memory 与 diagnostic_mode 判定](/root/autodl-tmp/generalist-specialist-project/gsdeploy/milp.py:77)及 DFS 对应逻辑列举了旧模式；未经适配的新模式可能回退到原始安全显存剪枝，还会被当成缺少能耗的诊断搜索。必须显式区分完整预测是否有六项证据、是否有服务／压力／显存／能耗下界，不能由“存在 surrogate 对象”或“energy_profiles 字典为空”推断。

当前局部设计只覆盖 `ASR2+G2n1+OCR1` 与所声明负载／路由邻域。它可以支持**该限定域**内的最优证书，不能支持其他副本、其他通用模型或整个七模型池的证书。对未覆盖域若只有非负损失的零界，其 U 上界是0，通常高于负数 incumbent，不能排除。现有代码还对存在 unknown 的结果保守标为 incomplete；不建议本轮同时修改这套证书策略。

## 4. 最小安全集成顺序及代价

1. 新建模式、模型版本与明确的适用域；完整预测入口绑定全部署、全任务路由、λ、逐输入表、参数、质量／资源定义及验证来源。旧零再拟合模型不变。第一版使用事先由 `(x,R,λ)` 唯一确定的倍率，暂不使用实际 raw overlap 或需要自洽求解的活动特征。
2. 实现新模式的完整服务解析与逐输入 FIFO，并同步接入 `performance.resolve_service`、`prediction`、`config.validate_plan`、能耗／显存辅助入口和 `SearchSession` 正式／诊断门禁。n 状态画像已含副本影响；保持当前 `n/capacity_factor` 语义，避免重复乘副本数。
3. 新建 bounds 适配器，显式提供 `request_lower`、`family_lower`、`pressure_lower` 及 M/E 下界能力。第一版无证部分一律0，关闭 capacity witness，保持所有候选列；原 Q 界按现有合同保留。完整有效评分与“可提供强下界”是两件事。
4. 待证明逐输入下包络后，再恢复单任务 FIFO／长 trace 无等待界以及下界压力剪枝。MILP 身份 cut、DFS 的效用界排序／剪枝共享同一界；DFS 结果的全局上界、枚举前容量覆盖以及 demand 累计也须使用它。缓存键至少包含新版本、固定负载及下包络域；如采用条件界，还要包含相应全局 context，不能复用旧 task-row 缓存。
5. 先在小完整域穷举核对每个 `B(z)<=C_new(z)`、每个部分分支上界覆盖所有后继、所有被容量／显存剪枝的身份确实不可行，再比较 MILP、带界 DFS、plain DFS 同优。覆盖倍率随份额下降、跨任务影响、缺路径、域外 unknown、预算耗尽和原模型不变的回归；随后另做独立模型验证与效率实验。这里提出的是后续验收条件，不是已经通过的结论。

完整回放每方案仍约 O(N)，长 trace 的经验分位数计算依实现另计；若倍率是静态 context 函数，增加的路径系数计算很小。全局最小包络的预计算按路径×副本×输入×context 数增长，但可缓存；界变弱后完整预测次数可能接近穷举，不能继承旧效率倍数。

另一种更强但非第一版方案：把有限的全局 ASR 份额 context 作为显式 one-hot，并让各任务列与同一 context 联动，或逐 context 分开求解后取最大值。这样可获得条件下更紧的服务／压力界，代价是列数或搜索轮数乘以 context 数；负载若只是固定扫描点，无需再当成 MILP 变量。若所有任务的路由组合都成为 context，这会退化为完整组合枚举。固定部署下按份额 context 做精确有限回放通常比立即改七模型主问题更适合作为第一步。

活动／并发状态自洽模型会把一次 O(N) 回放变成多次回放或更复杂的事件模拟，需预声明唯一解／收敛准则和失败语义；实际 overlap 不能用于预测新方案。没有验证这一结构前，不引入额外拟合参数或假定更复杂模型必能恢复分流收益。

**建议本轮先验证局部完整预测是否能解释真实排序，再做局部有限域枚举；只有新模式对七模型声明域有明确覆盖与下界契约后，才恢复全池证书搜索。** 搜索器能求新函数的最优，不代表新函数已能预测真实 GPU。
