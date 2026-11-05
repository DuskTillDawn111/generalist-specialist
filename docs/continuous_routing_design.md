# 连续路由—离散部署：下一目标与 Phase 1 实现设计

日期：2026-11-01。依据：[计划书 V3.0](continuous_routing_plan_v3.md)、[实际源码审计](continuous_routing_audit.md)。
状态：Phase 0 审计与设计完成；Phase 1 已实现并通过 CPU 验证，见[阶段报告](continuous_routing_phase1.md)与[复现说明](continuous_routing_usage.md)。旧 Joint MILP/DFS 保持默认入口和证书语义。

## 1. 下一阶段唯一主目标

建立一个独立、可复现的 **连续路由 → 固定路由部署优化 → 完整候选评价 → DE 更新** 闭环。
先验证固定路由枚举与 MILP 在同一预测器下结果一致，再比较 DE 与旧有限网格搜索在相同总时间/完整预测预算下的最佳可行效用。

本阶段的结论是“结构是否正确、内层是否值得用 MILP、连续搜索是否值得继续”。不预设效用提升，不承诺连续全局最优；GP-BO、MADS、混合方法留待 Gate 1 后按实验证据决定。
预测器按既有组件接入，不设计、训练、校准新的预测器；当前完整真实接口缺口以 adapter/支持域状态记录。

## 2. 数学与评价语义

正负载任务 t 的合法模型集 E(t) 来自声明能力；每行 r_t 属于闭 simplex，维度 |E(t)|−1。
统一转换 `routing_qps[t][m] = workload[t] * routing_shares[t][m]`，保存原比例，不再套分母 K 网格。

激活集合 `A(r) = {m: sum_t λ_t r_t,m > 0}`。
对 m∈A，副本状态必须为合法正整数；对 m∉A，允许 0 **以及合法正副本**。一般共驻模型中，不能证明 idle resident 无益时不得直接删除。
部署 roster 和流量激活集合分别记录；小模型池同时覆盖多 G、共享 S、G-only、合法 S-only 和混合覆盖。

直接优化既有 U：外层最小化 `−best_feasible_utility`，不增加质量惩罚。问题没有声明 Q_min 时不加入绝对精度门槛。

### 多轨迹合同

冻结优化、验证、最终测试三个不相交的 trace 身份集合及 seed；每条绑定完整 cohort、输入池/顺序、窗口、drain 与来源。
arrival seed 和 input shuffle seed 分开记录，即使旧 adapter 暂时使用同一 seed。

Phase 1 采用显式权重 a_j（默认等权）聚合**完整轨迹效用**：

`V_train(r) = max_D sum_j a_j S(D,r; trace_j)`。

Nominal 中 S=原 `U`；Robust 中 S=原 `scenario_U_low`（每条 trace 的 1/1.25/2 service stress 最差 U），且全部 trace/stress 的硬约束必须通过。
明确使用 `mean_j(min_s U_js)`，不偷换为 `min_s(mean_j U_js)`；这一新增跨轨迹规则写入 objective ID。任一必要轨迹缺证据时该候选聚合效用 UNKNOWN，不丢弃轨迹或重分权重。

同一个 `(D,r)` 必须应用到所有轨迹后再聚合，不能计算 `mean_j(max_D U_j)`。最终验证/测试只评估冻结的训练赢家，不据测试结果回调优化器。

### 基线公平范围

旧算法没有跨轨迹聚合目标接口。第一组优化器直接对比采用单冻结轨迹、相同 predictor/目标/预算，并分别报告 K；多条配对轨迹分别重复该实验。
第二组以多轨迹训练新算法、冻结方案后在共同 heldout 集评估旧/新方案，明确这是方案泛化对比，不能把旧算法说成优化了相同的聚合训练目标。
若要报告多轨迹聚合目标下旧 Joint MILP 的优化效率，必须另实现并审核共享聚合 evaluator 的显式 baseline adapter，含一致 master 界；在这之前不作该项公平优化效率结论。小域可由独立有限网格×部署穷举提供聚合目标参考，名称不得冒充原 Joint MILP。

## 3. 模块与调用边界

```text
experiments/compare_continuous_optimizers.py
  └─ DE optimizer / fixed-route inner benchmarks
       └─ ContinuousRoutingEvaluator
            ├─ ContinuousRouteDomain + activation
            ├─ FixedRoutingEnumerationOptimizer
            └─ FixedRoutingMilpOptimizer
                 └─ shared CandidateOracle adapter
                      ├─ legacy predict_scenarios (explicit supported scope)
                      ├─ supplied whole-forecast provider
                      └─ synthetic fixture oracle (algorithm tests only)
```

建议新增 `gsdeploy/continuous/`：`route_domain.py`、`activation.py`、`contracts.py`、`evaluator.py`、`predictor_adapter.py`、`fixed_routing_enumeration.py`、`fixed_routing_milp.py`、`de_optimizer.py`。
按依赖逐项落地，不提前创建 BO/MADS/hybrid 空实现。

复用 Router、trace builder、原评分/质量实现、合法状态表和 `solve_master()`；不复制 FIFO、效用函数或共驻引擎。
旧 `SystemConfig` 用作输入问题的一部分；新独立 experiment config 包装 trace sets、oracle、连续算法和预算，先不扩大旧 solver 白名单。

## 4. 完整候选接口与结果

下面是 Phase 1 的接口关系（实际方法名与参数见 `gsdeploy/continuous/`）：

```python
oracle.evaluate(problem, deployment, routing_qps, trace_spec, budget)
oracle.bounds(problem, fixed_routing, partial_deployment)
inner.solve(problem, fixed_routing, oracle, budget)
evaluator.evaluate(routing_vector, evaluation_budget=None)
```

`oracle.evaluate` 返回完整预测/原效用、三态 constraints、证据类型/支持域、来源版本、trace identity 和实际调用成本。
legacy adapter 只在原模式范围调用 `predict_scenarios(trace=...)`，不注入虚假 service；whole-forecast adapter 直接接现成引擎结果，不能再经过旧 scalar FIFO。
synthetic oracle 使用显式 fixture 原语与共享评分逻辑；结果固定标记 `evidence_kind=synthetic_fixture`，与真实研究表分开。

模型/domain 支持、约束、完整效用与资格是不同字段。UNKNOWN 需要保留原因与未决身份；可行约束 + U 缺失不进入最佳有效解。未知约束不等于物理不可行。

统一外层结果至少包含：

| 类别 | 字段 |
| --- | --- |
| 身份 | `problem_sha256`, `oracle_version`, `objective_id`, `trace_set_sha256`, `routing_shares`, `routing_qps`, `activation`, `deployment` |
| 值与界 | `best_feasible_utility`, `valid_utility_upper_bound`, `optimality_gap`, `prediction` |
| 状态 | `feasible`（是否已有完整有效解）, `constraint_state`, `status`, `certified`, `certificate_scope`, `unresolved_candidates` |
| 成本 | `outer_evaluations`, `milp_solve_calls`, `full_prediction_calls`, `predictor_leaf_calls`, `cache_hits`, `evaluation_seconds` 及阶段时间 |
| 搜索边界 | `remaining_deployments`, `activation_coverage`, `inner_search_completed`, `outer_continuous_optimality=false` |

状态至少区分 `OPTIMAL_FIXED_ROUTE_FINITE_DOMAIN`、`PROVEN_INFEASIBLE`、`BUDGET_LIMIT_WITH_INCUMBENT`、`BUDGET_LIMIT_NO_INCUMBENT`、`INCOMPLETE_EVIDENCE`、`INVALID_INPUT`、`SOLVER_ERROR`。
`feasible=false` 只表示尚未找到有效解；只有 `PROVEN_INFEASIBLE` 才表示域内已证明无解。
预算终止时原样返回 incumbent；无证明上界时 upper/gap=null。
内层完成不表示连续外层全局最优。

## 5. 连续域与精确边界

第一版采用每任务 k−1 维 stick-breaking 参数，k 为合法模型数：

`r_1=u_1; r_i=u_i * remaining; r_k=remaining`，`u_i∈[0,1]`。

它覆盖闭 simplex、保留精确端点，无 softmax epsilon、round 或固定比例网格。固定模型顺序写入 domain hash。
stick-breaking 的采样分布依赖顺序，不能称为均匀 simplex 采样；记录该选择，后续只在实验必要时比较替代参数化。
校验 finite、非负、守恒与合法边；浮点误差只在最后残量处理，不删除正数路径。若正比例×正 QPS 下溢为 0，显式报数值范围错误，不默认为未激活。

Phase 1 必须在 DE 前加入显式合法端点和边界。小域枚举每任务非空 support faces，并去重、记录实际覆盖；face 内仍用连续参数搜索。
全局激活集合 A 不能替代任务级 support：同一 A 可能对应不同任务边界。边界点重复只通过精确 evaluator cache 消除。
面内样本触及端点时接受其降维 support 并重新计算激活，不强行改成小正数。
端点/face 预评价与初始化均计入总预算；覆盖不全时报告未覆盖 faces，不能称全激活域已搜索。

## 6. 内层枚举：第一份可靠参照

固定 r 后，按模型确定性枚举合法副本状态的笛卡尔积，强制正流量模型有副本。
先检查能力/流量等结构约束；显存与 rho 剪枝只在 oracle 提供适用的 necessary bound 合同时执行。
对所有未安全剪掉的完整 D，调用同一聚合 oracle。

保存 evaluated/pruned/unknown/unvisited 四类互斥 ledger，包含原因与完整 identity。小域参考默认关闭效用剪枝，确保与 MILP 的实现路径独立。
只有未决候选已排除（有有效界证明不可能改进，或该域已知完整）时才可认证；第一版保守地让任何未解析 UNKNOWN 阻止全域最优/无解认证。

## 7. 内层 MILP：仅选择部署状态

Master 变量只含模型 one-hot `x_m,k` 与成本 epigraph `theta`，删除旧 route/count 列变量；固定 r 是参数。

约束：每模型恰选一状态；正流量模型禁止 k=0；只有经过适用证明的 additive memory/聚合容量界可以进入 master。
例如在 independent 冻结模型下固定 QPS 的容量约束可用 `sum_t q_t,m s_m,t / 1000 <= 0.8 * sum_k g_m(k)x_m,k`；真实共驻中不默认采用该式。

第一版默认完整 identity feedback/no-good，不使用跨路由割，也不启用旧 independent route feedback。
若 U 是原非正损失效用、各成本项非负，可用成本全局下界 0；自定义 oracle 无此合同时不得假定 U≤0。无可用界时将 MILP 降为可行部署生成器，靠穷尽已评价身份完成认证，不伪造有效上下界。

在有全局成本 floor B(x) 时，对已评价 identity I 的 residual `c_I−B(I)≥0` 构造只在 I 生效的反馈，复用 residual identity cut 的形式并用小域全枚举验证。
无效/已评价身份用 no-good 割排除；UNKNOWN 身份即使从待搜 master 排除，也必须保留其乐观上界影响最终证书。
求解器 dual bound 只约束尚在 master 内的域；最终 upper 还要与 incumbent、UNKNOWN 及所有已移出但未解决身份的界合并。
数值容差内重复身份、非整数 witness、解算错误分别记录，不通过四舍五入制造完成状态。

只在同一完整问题、固定路由和 source identity 下热启动同一内层；邻近路由最多借用部署候选顺序，必须重新评价，并计成本。

## 8. DE 与预算

使用 SciPy `differential_evolution`，初始 `best1bin`、Sobol/Latin Hypercube、固定 seed、`polish=False`、`workers=1`。
单模型任务不占连续维度；全域零维时直接评价，不调用 DE。显式 faces/端点的搜索策略与预算分配写入配置。

全局 budget 对象覆盖准备、域枚举、边界种子、缓存、MILP、全部预测、DE 与输出；共享绝对 deadline、完整预测调用上限和外层候选上限，禁止每次内层重新领取全局时间。
`full_prediction_calls` 定义为实际一个 `(D,r,trace)` 场景 bundle 调用，`predictor_leaf_calls` 记录实际 stress/world 子调用，避免旧代码每 bundle 算 1 而新算法按 leaf 算 3。
比较旧基线时保留其原计数，并在 experiment ledger 归一化，初始化/热启动评价不得免费。

检查预算后才启动下一调用。正在执行的非抢占 oracle 会导致超时尾部，必须计入并报告，不承诺硬墙钟上限。CPU-only smoke 使用小域和预声明硬调用数，禁止自动扩成历史 GPU campaign。

Phase 1 的正式 DE 排名优先限定在内层能完成的小域。未认证 incumbent 可以改善全局已知解，但不能把它当准确 V(r)；不可行、UNKNOWN 和未完成分别存日志。
首次小域 smoke 若出现未完成 inner evaluations，标记该轮比较不满足精确评价前提，保留部分结果后收缩后续实验设计；不隐式向 DE 填“真实低分”。

## 9. 缓存与来源

分两层缓存：完整 oracle evaluation 与 fixed-route 内层结果。key 包含原始 float 精确编码（如 float.hex）、canonical QPS、完整 deployment、任务/轨迹/输入、画像/adapter/路由器/评分版本、资源与质量约束、机制及 nominal/robust 语义。
允许规范化 +0/−0 和键排序；不 round 份额，不按相近比例命中，不凭 assignment 相同合并。
缓存 scope 绑定当前不可变问题和 adapter；来源失效则整个 session 拒绝复用。
未完成 inner result 不能被当成已完成 V(r)；继续求解须显式携带剩余 ledger/累计成本，命中仍记录本次开销和继承成本。

## 10. 任务顺序、交付与验收

| 顺序 | 工作与交付 | 验收标准 |
| --- | --- | --- |
| M0（本次） | 源码审计、接口风险、本文、原计划归档 | 变量/QPS/激活、oracle 范围、认证/预算/多轨迹合同明确 |
| M1（下一开发任务） | continuous domain、activation、contracts、oracle adapter、预算与精确 key | 零/一/多模型边界正确；非网格浮点不量化；无隐式激活阈值；UNKNOWN 与不可行分离 |
| M2 | fixed-routing enumeration + synthetic/legacy 接口测试 | 独立穷举完整有限域；固定 r 多轨迹同 D；资源缺项不产生可行 U |
| M3 | fixed-routing MILP、feedback 与 ledger | 和 M2 的可行性、最佳 U、认证域一致；预算/UNKNOWN/数值错误正确；报告实际调用和成本差异 |
| M4 | DE + 独立 CLI + 配对小实验 | 端点/face 预算计入；固定 seed 可重现；任意连续比例保留；输出配置、JSONL、summary JSON/CSV |
| M5 | Phase 1 报告与图表 | 回答 RQ1–5，给出继续/调整/停止依据；真实证据与 synthetic 分开；旧 CPU 回归全通过 |

M1 的明确目标：**将一个任意合法连续路由，确定性地转换成 QPS、激活集合与完整候选评价请求，并且准确返回“有效/不可行/未知/预算不足”，不依赖旧路由网格。**

核心测试至少覆盖计划书的 15 项，补充：idle resident 保留、负载下溢、三个以上合法模型的闭 simplex、alias 数变化、缓存碰撞/版本失效、多轨迹 max/mean 顺序、UNKNOWN ledger、stress 不支持及 bundle/leaf 计数。
MILP feedback 在小域每个整数身份上校验有效性，并以非可分合成共驻场景验证独占剪枝不会误删赢家。
首轮 CPU smoke 固定小模型池、有限合法状态、少量冻结轨迹和 seeds，预声明预算；先获得 correctness/cost 结果再扩大，不写未经测量的耗时预估。

Gate 1 报告回答：MILP 是否省完整预测；DE 相对网格在相同预算下是否有收益；激活边界影响；有限轨迹平台的影响；外层/内层/预测各自成本。
MILP 若没有收益，允许后续选择枚举或重新设计内层，不为维护架构预设而隐藏结果。
未通过 Gate 1 不进入 Phase 2/3，也不变更旧默认优化器。
